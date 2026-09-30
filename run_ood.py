#!/usr/bin/env python
"""Temporal out-of-distribution experiment of the paper (train on LUCAS 2018, test on LUCAS 2022).

Computes, for each seed, every row of Tables 9 to 12: frozen embeddings (AlphaEarth, OlmoEarth,
Presto, Tessera, CLIP) and their fusions with a linear probe and a random forest, and the temporal
Transformers on the Sentinel-1/2 series with and without ancillary data (TMS) and CLIP. The
aggregation step writes mean and standard deviation over seeds as CSV and as the LaTeX tables.

Environment:
    GEOCROP_DATA       input directory with 2018/ and 2022/ (layout and checksums: README.md)
    GEOCROP_SEED       run this seed only (one GPU job per seed)
    GEOCROP_AGGREGATE  1: read the per-seed CSVs and write the tables, no computation
    GEOCROP_OUT        results directory (default: results/ood next to this file)
Without GEOCROP_SEED and GEOCROP_AGGREGATE, all SEEDS run in sequence and are then aggregated.
"""
import os
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

from crop_classes import CLMS_MAP

DATA = os.environ.get('GEOCROP_DATA')
OUT_DIR = os.environ.get('GEOCROP_OUT', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'ood'))
TRAIN_YEAR, TEST_YEAR = 2018, 2022
SEEDS = [17, 18, 19, 20, 21]
SEED = SEEDS[0]  # reset per seed inside the loop
_ONE_SEED_ENV = os.environ.get('GEOCROP_SEED')
ONE_SEED = int(_ONE_SEED_ENV) if _ONE_SEED_ENV not in (None, '') else None
AGGREGATE_ONLY = os.environ.get('GEOCROP_AGGREGATE', '0') == '1'

# Linear probe, projection and Transformer training; random forest size; fused dimension.
EPOCHS, LR, BATCH_SIZE, RF_TREES, COMMON_DIM, USE_CLASS_WEIGHTS = 200, 1e-3, 256, 100, 32, True
# Temporal Transformer encoder: width, attention heads, layers, dropout.
D_MODEL, N_HEADS, N_LAYERS, DROPOUT = 64, 4, 2, 0.1
EO_MODELS = ['OlmoEarth', 'AlphaEarth', 'Presto', 'Tessera']
REQUIRE = ['OlmoEarth', 'AlphaEarth', 'Presto', 'Tessera', 'CLIP']
RAW_MODS = ['S1', 'S2', 'ERA5', 'DEM', 'SOIL']
STATIC_RAW = {'DEM', 'SOIL'}
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

clms_map = CLMS_MAP
code_to_name = {v[0]: v[1] for v in clms_map.values()}
CLASS_CODES = sorted(set(v[0] for v in clms_map.values()))
label_map = {c: i for i, c in enumerate(CLASS_CODES)}
n_classes = len(CLASS_CODES)
target_names = [code_to_name[c] for c in CLASS_CODES]

# Per-class row order of Tables 10 and 12: (class name in the results | label in the table | \crop colour macro)
PC_ROWS = [
    ('Wheat',                 'Wheat',         'wheatcolor'),
    ('Barley',                'Barley',        'barleycolor'),
    ('Maize',                 'Maize',         'maizecolor'),
    ('Rice',                  'Rice',          'ricecolor'),
    ('Other cereals',         'Other Cereals', 'other_Cereals_color'),
    ('Fresh vegetables',      'Fresh Veg',     'fresh_veg'),
    ('Dry pulses',            'Pulses',        'dry_pulses'),
    ('Potatoes',              'Potatoes',      'potatoes'),
    ('Sugar Beet',            'Sugar beet',    'sugarbeet'),
    ('Sunflower',             'Sunflower',     'sunflowercolor'),
    ('Soybeans',              'Soybeans',      'soyacolor'),
    ('Rapeseed',              'Rapeseed',      'rapecolor'),
    ('Flax, cotton and hemp', 'Flax/Cotton',   'flax'),
    ('Grapes',                'Grapes',        'grapes'),
    ('Olives',                'Olives',        'olives'),
    ('Fruits',                'Fruit',         'fruits'),
    ('Nuts',                  'Nuts',          'nuts'),
]

# Transformer rows as recorded and as displayed in Tables 9 and 11.
T9_TRANSFORMER_ROWS = [('S1 only', 'Single'), ('S2 only', 'Single')]
T11_TRANSFORMER_ROWS = [('S1+S2', 'Mid-Level'), ('S1+S2+TMS', 'Mid-Level'),
                        ('S1+S2+CLIP', 'Mid-Level'), ('S1+S2+TMS+CLIP', 'Mid-Level')]

# Classifier of the embedding columns of the per-class Table 10.
PERCLASS_CLF = 'LinearProbe'


def load_labels(lulc_csv):
    df = pd.read_csv(lulc_csv, low_memory=False)
    df = df[df['LC1'].astype(str).str.startswith('B')].copy()
    df['clms_code'] = df['LC1'].map(lambda x: clms_map[x][0] if x in clms_map else None)
    df = df[df['clms_code'].notna()].copy()
    df['POINT_ID'] = df['POINT_ID'].astype(str).str.strip().astype(np.int64)
    return df

# ----- embedding files: one array of features and one of LUCAS point ids per file -----
def _emb_cfg(yr):
    b = f'{DATA}/{yr}'
    return {
        'Tessera':    {'path': f'{b}/TE_{yr}_data.npz',    'xk': ['X'],          'idk': ['point_ids'], 'deq': False},
        'Presto':     {'path': f'{b}/PE_{yr}_X.npz',       'xk': ['embeddings'], 'idk': ['point_ids'], 'deq': False},
        'OlmoEarth':  {'path': f'{b}/OE_{yr}_AI2_v2.npz',  'xk': ['data'],       'idk': ['point_ids'], 'deq': True},
        'AlphaEarth': {'path': f'{b}/AE_{yr}_Xy_v2.npz',   'xk': ['X'],          'idk': ['point_ids'], 'deq': False},
        'CLIP':       {'path': f'{b}/CLIP_{yr}_point.npz', 'xk': ['data'],       'idk': ['point_id'],  'deq': False},
    }

def _dequant_olmo(raw):
    return np.sign(raw) * (np.abs(raw) / 127.5) ** 2

def _to_2d(X):
    X = np.asarray(X)
    if X.ndim == 2:
        return X.astype(np.float32)
    if X.ndim == 3:  # raw time series (N, T, C) -> flattened feature vector for the frozen-feature grid
        return X.reshape(X.shape[0], -1).astype(np.float32)
    if X.ndim == 5 and X.shape[1] == 1:
        X = X[:, 0]
    if X.ndim == 4:
        return X.mean(axis=(2, 3)).astype(np.float32)
    if X.ndim == 5:
        return X.mean(axis=(1, 3, 4)).astype(np.float32)
    raise ValueError(f'unexpected embedding shape {X.shape}')

def _clean(X, fill_value=None):
    X = X.astype(np.float32, copy=False)
    X = np.where(np.isfinite(X), X, np.nan)
    if fill_value is not None:
        try:
            fv = float(np.ravel(fill_value)[0])
            X = np.where(X == fv, np.nan, X)
        except Exception:
            pass
    return np.where(X <= -9990, np.nan, X)

def _ids_int64(ids):
    return np.asarray([int(str(v).strip()) for v in np.asarray(ids).ravel()], np.int64)

def load_modality(name, cfg):
    p = cfg['path']
    if not os.path.exists(p):
        raise FileNotFoundError(f'{name}: {p}')
    z = np.load(p, allow_pickle=True)
    def first(keys):
        for k in keys:
            if k in z.files:
                return z[k]
        raise KeyError(f'none of {keys} in {p} (has {list(z.files)})')
    X = first(cfg['xk'])
    ids = first(cfg['idk'])
    if cfg['deq']:
        X = _dequant_olmo(X)
    X = _clean(_to_2d(X))
    ids = _ids_int64(ids)
    print(f'  [loaded] {name:11s} X={X.shape} ids={len(ids):,}')
    return X, ids

def load_year_mods(yr):
    cfgs = _emb_cfg(yr)
    return {nm: load_modality(nm, cfgs[nm]) for nm in REQUIRE}

# ----- raw series and ancillary files (S1/S2/ERA5 monthly; DEM/SOIL static) -----
def _raw_cfg(yr):
    return {m: [f'{DATA}/{yr}/modalities/{m}_{yr}_Xy_v2.npz'] for m in RAW_MODS}

def load_raw_modality(name, candidates):
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        print(f'  [missing] raw {name:5s}: no candidate file found')
        return None
    z = np.load(path, allow_pickle=True)
    def first(keys):
        for k in keys:
            if k in z.files:
                return z[k]
        raise KeyError(f'none of {keys} in {path} (has {list(z.files)})')
    X = first(['X', 'data', 'features'])
    ids = first(['point_ids', 'POINT_ID', 'ids'])
    fill = z['fill_value'] if 'fill_value' in z.files else None
    X = _clean(np.asarray(X), fill_value=fill)
    if X.ndim == 5:
        X = np.nanmean(X, axis=(-2, -1))              # (N,T,C,H,W) -> (N,T,C)
    elif X.ndim == 4:
        X = np.nanmean(X, axis=(-2, -1))[:, None, :]  # (N,C,H,W)   -> (N,1,C)
    elif X.ndim == 2:
        X = X[:, None, :]                             # (N,C)       -> (N,1,C)
    elif X.ndim != 3:
        raise ValueError(f'unexpected raw tensor shape for {name}: {X.shape}')
    ids = _ids_int64(ids)
    print(f'  [loaded] raw {name:5s} X={X.shape} ids={len(ids):,} | {path}')
    return X.astype(np.float32), ids

def load_year_raw(yr):
    cfg = _raw_cfg(yr)
    out = {}
    print(f'[loading raw tensors for {yr}]')
    for m in RAW_MODS:
        r = load_raw_modality(m, cfg[m])
        if r is not None:
            out[m] = r
    return out

def common_label_df(labels_df, mods):
    common = set(labels_df['POINT_ID'].values)
    for m in REQUIRE:
        common &= set(mods[m][1])
    common = np.array(sorted(common), np.int64)
    return labels_df[labels_df['POINT_ID'].isin(set(common))].set_index('POINT_ID').loc[common].reset_index()

def extract_by_ids(X, ids, target_ids):
    row = {int(p): r for r, p in enumerate(ids)}
    return X[[row[int(p)] for p in target_ids]]

def within_year_split(df, seed):
    df = df.copy()
    df['strat'] = df['clms_code'].astype(str) + '_' + df['Country'].astype(str)
    keep = df['strat'].value_counts()
    df = df[df['strat'].isin(keep[keep >= 5].index)].copy()
    tr, tmp = train_test_split(df, test_size=0.30, random_state=seed, stratify=df['strat'])
    tc = tmp['strat'].value_counts()
    s2 = tmp['clms_code'] if not (tc >= 2).all() else tmp['strat']
    va, te = train_test_split(tmp, test_size=0.50, random_state=seed, stratify=s2)
    return tr, va, te

# ----- probe, projector, random forest, scaling -----
def make_class_weights(y_train, k, device=DEVICE, normalize=True):
    y_train = np.asarray(y_train, dtype=np.int64)
    counts = np.bincount(y_train, minlength=k).astype(np.float32)
    present = counts > 0
    w = np.zeros(k, dtype=np.float32)
    if present.any():
        w[present] = len(y_train) / (present.sum() * counts[present])
        if normalize:
            w[present] = w[present] / w[present].mean()
    return torch.tensor(w, dtype=torch.float32, device=device)

def scale_2d(X_tr, X_va, X_te):
    med = np.nanmedian(X_tr, axis=0)
    def fill(X):
        X = np.array(X, dtype=np.float64)
        idx = np.where(~np.isfinite(X))
        X[idx] = med[idx[1]]
        return X
    X_tr, X_va, X_te = fill(X_tr), fill(X_va), fill(X_te)
    sc = StandardScaler()
    return sc.fit_transform(X_tr), sc.transform(X_va), sc.transform(X_te)

def scale_3d_temporal(X_tr, X_va, X_te):
    """Per-channel train-median imputation and StandardScaler over (N*T, C)."""
    Ntr, T, C = X_tr.shape
    tr2, va2, te2 = X_tr.reshape(-1, C), X_va.reshape(-1, C), X_te.reshape(-1, C)
    med = np.nanmedian(tr2, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    def fill(A):
        A = np.array(A, dtype=np.float64)
        idx = np.where(~np.isfinite(A))
        A[idx] = med[idx[1]]
        return A
    tr2, va2, te2 = fill(tr2), fill(va2), fill(te2)
    sc = StandardScaler()
    return (sc.fit_transform(tr2).reshape(Ntr, T, C).astype(np.float32),
            sc.transform(va2).reshape(X_va.shape[0], T, C).astype(np.float32),
            sc.transform(te2).reshape(X_te.shape[0], T, C).astype(np.float32))

class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)
    def forward(self, x):
        return self.fc(x.flatten(start_dim=1))

class LinearProjector(nn.Module):
    def __init__(self, input_dim, proj_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, proj_dim)
    def forward(self, x):
        return self.proj(x)

def _train_head(module_factory, X_tr, y_tr, X_va, y_va):
    """Shared trainer for probe (module) and projector+head. Returns the trained module."""
    device = DEVICE if (DEVICE != 'cuda' or torch.cuda.is_available()) else 'cpu'
    X_va_t = torch.tensor(X_va, dtype=torch.float32, device=device)
    y_va_t = torch.tensor(y_va, dtype=torch.long, device=device)
    loader = DataLoader(TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                                      torch.tensor(y_tr, dtype=torch.long)),
                        batch_size=BATCH_SIZE, shuffle=True)
    module, head, params = module_factory(X_tr.shape[1], device)
    optimizer = torch.optim.Adam(params, lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)
    weights = make_class_weights(y_tr, n_classes, device=device) if USE_CLASS_WEIGHTS else None
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    best_loss, best_state, bad = float('inf'), None, 0
    for _ in range(EPOCHS):
        module.train()
        if head is not None:
            head.train()
        for bX, by in loader:
            bX, by = bX.to(device), by.to(device)
            optimizer.zero_grad()
            logits = head(module(bX)) if head is not None else module(bX)
            loss = loss_fn(logits, by)
            loss.backward()
            optimizer.step()
        module.eval()
        if head is not None:
            head.eval()
        with torch.no_grad():
            logits = head(module(X_va_t)) if head is not None else module(X_va_t)
            val_loss = loss_fn(logits, y_va_t).item()
        scheduler.step(val_loss)
        if val_loss < best_loss - 1e-4:
            best_loss, best_state, bad = val_loss, copy.deepcopy(module.state_dict()), 0
        else:
            bad += 1
        if bad >= 20:
            break
    if best_state is not None:
        module.load_state_dict(best_state)
    return module

def train_probe(X_tr, y_tr, X_va, y_va):
    def factory(dim, device):
        m = LinearProbe(dim, n_classes).to(device)
        return m, None, list(m.parameters())
    return _train_head(factory, X_tr, y_tr, X_va, y_va)

def train_projector(X_tr, y_tr, X_va, y_va, proj_dim=COMMON_DIM):
    def factory(dim, device):
        m = LinearProjector(dim, proj_dim).to(device)
        h = nn.Linear(proj_dim, n_classes).to(device)
        return m, h, list(m.parameters()) + list(h.parameters())
    return _train_head(factory, X_tr, y_tr, X_va, y_va)

def get_probe_preds(model, X_te):
    device = DEVICE if (DEVICE != 'cuda' or torch.cuda.is_available()) else 'cpu'
    model = model.to(device).eval()
    with torch.no_grad():
        return model(torch.tensor(X_te, dtype=torch.float32, device=device)).argmax(1).cpu().numpy()

def project(projector, X):
    device = DEVICE if (DEVICE != 'cuda' or torch.cuda.is_available()) else 'cpu'
    projector = projector.to(device).eval()
    with torch.no_grad():
        return projector(torch.tensor(X, dtype=torch.float32, device=device)).cpu().numpy()

def get_rf_preds(X_tr, y_tr, X_te):
    rf = RandomForestClassifier(n_estimators=RF_TREES, random_state=SEED, n_jobs=-1,
                                class_weight='balanced_subsample', max_features='sqrt')
    rf.fit(X_tr, y_tr)
    return rf.predict(X_te)

# ----- temporal Transformer stack -----
def _as_3d(x):
    if x.ndim == 2:
        x = x.unsqueeze(1)
    return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]

class SequenceEncoder(nn.Module):
    def __init__(self, input_dim, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
                                           dropout=dropout, batch_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x):
        z = self.encoder(self.pos(self.proj(_as_3d(x))))
        return self.norm(z.mean(dim=1))

class SingleTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=D_MODEL):
        super().__init__()
        self.encoder = SequenceEncoder(input_dim=input_dim, d_model=d_model)
        self.cls_head = nn.Linear(d_model, num_classes)
    def forward(self, *xs):
        return self.cls_head(self.encoder(xs[0]))

class MidFusionTransformer(nn.Module):
    def __init__(self, input_dims, num_classes, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.encoders = nn.ModuleList([SequenceEncoder(d, d_model, n_heads, n_layers, dropout) for d in input_dims])
        fused = d_model * len(input_dims)
        self.cls_head = nn.Sequential(nn.LayerNorm(fused), nn.Linear(fused, d_model), nn.GELU(),
                                      nn.Dropout(dropout), nn.Linear(d_model, num_classes))
    def forward(self, *xs):
        return self.cls_head(torch.cat([e(x) for e, x in zip(self.encoders, xs)], dim=-1))

class DynamicStaticFusionTransformer(nn.Module):
    """Dynamic modalities -> temporal encoders; static modalities (incl. CLIP-LP) -> MLP encoders."""
    def __init__(self, dynamic_input_dims, static_input_dims, num_classes,
                 d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.dynamic_encoders = nn.ModuleList(
            [SequenceEncoder(d, d_model, n_heads, n_layers, dropout) for d in dynamic_input_dims])
        self.static_encoders = nn.ModuleList(
            [nn.Sequential(nn.Linear(d, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout),
                           nn.Linear(d_model, d_model), nn.LayerNorm(d_model)) for d in static_input_dims])
        fused = d_model * (len(dynamic_input_dims) + len(static_input_dims))
        self.cls_head = nn.Sequential(nn.LayerNorm(fused), nn.Linear(fused, d_model), nn.GELU(),
                                      nn.Dropout(dropout), nn.Linear(d_model, num_classes))
    def forward(self, *xs):
        nd = len(self.dynamic_encoders)
        reps = [e(x) for e, x in zip(self.dynamic_encoders, xs[:nd])]
        for e, x in zip(self.static_encoders, xs[nd:]):
            x = _as_3d(x)
            if x.shape[1] != 1:
                x = x.mean(dim=1, keepdim=True)
            reps.append(e(x.squeeze(1)))
        return self.cls_head(torch.cat(reps, dim=-1))

class MultiTensorDataset(Dataset):
    def __init__(self, X_list, y):
        self.X = [torch.tensor(x, dtype=torch.float32) for x in X_list]
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        return tuple([x[i] for x in self.X] + [self.y[i]])

def make_multimodal_loaders(Xs_tr, Xs_va, Xs_te, y_tr, y_va, y_te, batch_size=BATCH_SIZE):
    return (DataLoader(MultiTensorDataset(Xs_tr, y_tr), batch_size=batch_size, shuffle=True),
            DataLoader(MultiTensorDataset(Xs_va, y_va), batch_size=batch_size, shuffle=False),
            DataLoader(MultiTensorDataset(Xs_te, y_te), batch_size=batch_size, shuffle=False))

def train_sequence_model(model, train_loader, val_loader, y_tr, y_va, epochs=EPOCHS,
                         sched_patience=5, sched_factor=0.5, min_lr=1e-6,
                         early_stop_patience=15, min_delta=1e-4):
    """Weighted CE, plateau LR, best-validation checkpoint, early stopping."""
    device = DEVICE if (DEVICE != 'cuda' or torch.cuda.is_available()) else 'cpu'
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=sched_factor,
                                                           patience=sched_patience, min_lr=min_lr)
    weights = make_class_weights(y_tr, n_classes, device=device) if USE_CLASS_WEIGHTS else None
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    best_loss, best_state, bad = float('inf'), None, 0
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            *xs, y = batch
            xs = [x.to(device) for x in xs]
            y = y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(*xs), y)
            loss.backward()
            optimizer.step()
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                *xs, y = batch
                xs = [x.to(device) for x in xs]
                val_losses.append(loss_fn(model(*xs), y.to(device)).item())
        val_loss = float(np.mean(val_losses))
        scheduler.step(val_loss)
        if val_loss < best_loss - min_delta:
            best_loss, best_state, bad = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
        if epoch == 0 or (epoch + 1) % 25 == 0:
            print(f'      ep {epoch + 1:3d} | val_loss={val_loss:.4f} | bad={bad}')
        if bad >= early_stop_patience:
            print(f'      early stop at ep {epoch + 1}; best_val_loss={best_loss:.4f}')
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def predict_sequence_model(model, test_loader):
    device = DEVICE if (DEVICE != 'cuda' or torch.cuda.is_available()) else 'cpu'
    model = model.to(device).eval()
    preds = []
    with torch.no_grad():
        for batch in test_loader:
            *xs, _ = batch
            xs = [x.to(device) for x in xs]
            preds.append(model(*xs).argmax(1).cpu().numpy())
    return np.concatenate(preds)

# ----- fusion: each modality standardized, then concatenated (Naive), reduced by PCA, or projected (Proj) -----
def concat_local(splits, mods):
    parts = [[], [], []]
    for m in mods:
        s = scale_2d(*splits[m])
        for i in range(3):
            parts[i].append(s[i])
    return [np.concatenate(p, axis=1) for p in parts]

def pca_local(splits, mods, dim):
    tr, va, te = concat_local(splits, mods)
    d = min(dim, tr.shape[1], tr.shape[0])
    pca = PCA(n_components=d, random_state=SEED)
    return [pca.fit_transform(tr), pca.transform(va), pca.transform(te)]

def lp_local(splits, mods, y_tr, y_va, dim):
    tr, va, te = concat_local(splits, mods)
    proj = train_projector(tr, y_tr, va, y_va, proj_dim=dim)
    return [project(proj, tr), project(proj, va), project(proj, te)]

# ----- evaluate: second scaler, then LP + RF, then record all metrics -----
rows_overall, rows_perclass = [], []

def _record(data, fusion, clf, y_true, y_pred):
    rows_overall.append({
        'Data': data, 'Fusion': fusion, 'Classifier': clf, 'Seed': SEED,
        'Kappa': cohen_kappa_score(y_true, y_pred),
        'OA': 100 * accuracy_score(y_true, y_pred),
        'MacroF1': 100 * f1_score(y_true, y_pred, average='macro', zero_division=0),
        'WeightedF1': 100 * f1_score(y_true, y_pred, average='weighted', zero_division=0),
    })
    f = f1_score(y_true, y_pred, average=None, labels=list(range(n_classes)), zero_division=0)
    for i, nm in enumerate(target_names):
        rows_perclass.append({'Data': data, 'Fusion': fusion, 'Classifier': clf, 'Seed': SEED, 'Class': nm, 'F1': 100 * f[i]})
    print(f'    {data:42s} | {fusion:12s} | {clf:12s} | OA={rows_overall[-1]["OA"]:.2f} MF1={rows_overall[-1]["MacroF1"]:.2f}')

def evaluate(data, fusion, Xtr, Xva, Xte, y_tr, y_va, y_te):
    Xtr_s, Xva_s, Xte_s = (a.astype(np.float32) for a in scale_2d(Xtr, Xva, Xte))
    _record(data, fusion, 'LinearProbe', y_te, get_probe_preds(train_probe(Xtr_s, y_tr, Xva_s, y_va), Xte_s))
    _record(data, fusion, 'RandomForest', y_te, get_rf_preds(Xtr_s, y_tr, Xte_s))

# ----- temporal Transformer rows -----
def run_transformers_for_seed(tr_ids, va_ids, te_ids, y_tr, y_va, y_te,
                              raw_tr, raw_te, mods_tr, mods_te):
    lut_tr = dict(zip(tr_ids.tolist(), y_tr.tolist()))
    lut_va = dict(zip(va_ids.tolist(), y_va.tolist()))
    lut_te = dict(zip(te_ids.tolist(), y_te.tolist()))

    def prep(mods):
        """Filter split IDs to the coverage of the raw files of this experiment, extract, scale.
        Returns None if a file is missing or coverage is empty."""
        missing = [m for m in mods if m not in raw_tr or m not in raw_te]
        if missing:
            print(f'    [skip] raw file missing for {missing}')
            return None
        ok_tr = set(tr_ids.tolist()) | set(va_ids.tolist())
        ok_te = set(te_ids.tolist())
        for m in mods:
            ok_tr &= set(raw_tr[m][1].tolist())
            ok_te &= set(raw_te[m][1].tolist())
        tr_i = np.array([i for i in tr_ids if int(i) in ok_tr], np.int64)
        va_i = np.array([i for i in va_ids if int(i) in ok_tr], np.int64)
        te_i = np.array([i for i in te_ids if int(i) in ok_te], np.int64)
        print(f'    coverage {"+".join(mods)}: train {len(tr_i)}/{len(tr_ids)} val {len(va_i)}/{len(va_ids)} test {len(te_i)}/{len(te_ids)}')
        if not (len(tr_i) and len(va_i) and len(te_i)):
            print('    [skip] empty coverage on the split')
            return None
        splits = {}
        for m in mods:
            Xtr = extract_by_ids(raw_tr[m][0], raw_tr[m][1], tr_i)
            Xva = extract_by_ids(raw_tr[m][0], raw_tr[m][1], va_i)
            Xte = extract_by_ids(raw_te[m][0], raw_te[m][1], te_i)
            if Xtr.shape[2] != Xte.shape[2]:
                print(f'    [skip] {m}: train/test channel mismatch {Xtr.shape[2]} vs {Xte.shape[2]}')
                return None
            splits[m] = scale_3d_temporal(Xtr, Xva, Xte)
        ytr = np.array([lut_tr[int(i)] for i in tr_i], np.int64)
        yva = np.array([lut_va[int(i)] for i in va_i], np.int64)
        yte = np.array([lut_te[int(i)] for i in te_i], np.int64)
        return tr_i, va_i, te_i, splits, ytr, yva, yte

    def clip_lp_static(tr_i, va_i, te_i, ytr, yva):
        """CLIP -> scale -> learned 32-d projection -> scale."""
        ctr = extract_by_ids(mods_tr['CLIP'][0], mods_tr['CLIP'][1], tr_i)
        cva = extract_by_ids(mods_tr['CLIP'][0], mods_tr['CLIP'][1], va_i)
        cte = extract_by_ids(mods_te['CLIP'][0], mods_te['CLIP'][1], te_i)
        ctr, cva, cte = scale_2d(ctr, cva, cte)
        proj = train_projector(ctr, ytr, cva, yva, proj_dim=COMMON_DIM)
        return scale_2d(project(proj, ctr), project(proj, cva), project(proj, cte))

    def fit_and_record(data, fusion, model, Xs, ytr, yva, yte):
        tl, vl, sl = make_multimodal_loaders(Xs[0], Xs[1], Xs[2], ytr, yva, yte)
        model = train_sequence_model(model, tl, vl, ytr, yva, epochs=EPOCHS)
        _record(data, fusion, 'Transformer', yte, predict_sequence_model(model, sl))
        del model, tl, vl, sl
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 1+2. S1 only / S2 only (SingleTransformer)
    for m in ['S1', 'S2']:
        r = prep([m])
        if r is None:
            continue
        _, _, _, sp, ytr, yva, yte = r
        model = SingleTransformer(sp[m][0].shape[-1], n_classes)
        fit_and_record(f'{m} only', 'Single', model, ([sp[m][0]], [sp[m][1]], [sp[m][2]]), ytr, yva, yte)

    # 3. S1+S2 (MidFusionTransformer)
    r = prep(['S1', 'S2'])
    if r is not None:
        _, _, _, sp, ytr, yva, yte = r
        dims = [sp['S1'][0].shape[-1], sp['S2'][0].shape[-1]]
        model = MidFusionTransformer(dims, n_classes)
        Xs = ([sp['S1'][0], sp['S2'][0]], [sp['S1'][1], sp['S2'][1]], [sp['S1'][2], sp['S2'][2]])
        fit_and_record('S1+S2', 'Mid-Level', model, Xs, ytr, yva, yte)

    # 4. S1+S2+TMS (DynamicStaticFusionTransformer: dynamic S1,S2,ERA5; static DEM,SOIL)
    dyn, sta = ['S1', 'S2', 'ERA5'], ['DEM', 'SOIL']
    r = prep(dyn + sta)
    if r is not None:
        _, _, _, sp, ytr, yva, yte = r
        model = DynamicStaticFusionTransformer([sp[m][0].shape[-1] for m in dyn],
                                               [sp[m][0].shape[-1] for m in sta], n_classes)
        Xs = tuple([sp[m][k] for m in dyn + sta] for k in range(3))
        fit_and_record('S1+S2+TMS', 'Mid-Level', model, Xs, ytr, yva, yte)

    # 5. S1+S2+CLIP (dynamic S1,S2; static CLIP projection)
    r = prep(['S1', 'S2'])
    if r is not None:
        tr_i, va_i, te_i, sp, ytr, yva, yte = r
        ctr, cva, cte = clip_lp_static(tr_i, va_i, te_i, ytr, yva)
        model = DynamicStaticFusionTransformer([sp['S1'][0].shape[-1], sp['S2'][0].shape[-1]],
                                               [COMMON_DIM], n_classes)
        Xs = ([sp['S1'][0], sp['S2'][0], ctr], [sp['S1'][1], sp['S2'][1], cva], [sp['S1'][2], sp['S2'][2], cte])
        fit_and_record('S1+S2+CLIP', 'Mid-Level', model, Xs, ytr, yva, yte)

    # 6. S1+S2+TMS+CLIP (dynamic S1,S2,ERA5; static DEM,SOIL,CLIP projection)
    r = prep(dyn + sta)
    if r is not None:
        tr_i, va_i, te_i, sp, ytr, yva, yte = r
        ctr, cva, cte = clip_lp_static(tr_i, va_i, te_i, ytr, yva)
        model = DynamicStaticFusionTransformer([sp[m][0].shape[-1] for m in dyn],
                                               [sp[m][0].shape[-1] for m in sta] + [COMMON_DIM], n_classes)
        Xs = tuple([sp[m][k] for m in dyn + sta] + [(ctr, cva, cte)[k]] for k in range(3))
        fit_and_record('S1+S2+TMS+CLIP', 'Mid-Level', model, Xs, ytr, yva, yte)


def run_one_seed(seed, mods_tr, mods_te, df_tr_common, df_te_common, raw_tr, raw_te):
    global SEED
    SEED = seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rows_overall.clear(); rows_perclass.clear()
    print(f'\n===== seed {seed} =====')
    df_tr, df_va, _ = within_year_split(df_tr_common, SEED)

    def ids_y(df):
        return df['POINT_ID'].values.astype(np.int64), df['clms_code'].map(label_map).values.astype(np.int64)
    tr_ids, y_tr = ids_y(df_tr)
    va_ids, y_va = ids_y(df_va)
    te_ids, y_te = ids_y(df_te_common)
    print(f'[seed {SEED}] train={len(tr_ids):,} val={len(va_ids):,} test={len(te_ids):,}')

    splits = {m: (extract_by_ids(mods_tr[m][0], mods_tr[m][1], tr_ids),
                  extract_by_ids(mods_tr[m][0], mods_tr[m][1], va_ids),
                  extract_by_ids(mods_te[m][0], mods_te[m][1], te_ids)) for m in REQUIRE}

    # A. single modalities (Tables 9 and 10)
    for m in REQUIRE:
        evaluate(m, 'Single', *splits[m], y_tr, y_va, y_te)
    # B. each GeoFM + CLIP (Tables 11 and 12)
    for eo in EO_MODELS:
        pair = [eo, 'CLIP']
        lab = f'{eo}+CLIP'
        evaluate(lab, 'NaiveConcat', *concat_local(splits, pair), y_tr, y_va, y_te)
        evaluate(lab, f'PCAFusion{COMMON_DIM}', *pca_local(splits, pair, COMMON_DIM), y_tr, y_va, y_te)
        evaluate(lab, f'LPFusion{COMMON_DIM}', *lp_local(splits, pair, y_tr, y_va, COMMON_DIM), y_tr, y_va, y_te)
    # C. all GeoFMs (Table 11)
    all4 = '+'.join(EO_MODELS)
    evaluate(all4, 'NaiveConcat', *concat_local(splits, EO_MODELS), y_tr, y_va, y_te)
    evaluate(all4, f'PCAFusion{COMMON_DIM}', *pca_local(splits, EO_MODELS, COMMON_DIM), y_tr, y_va, y_te)
    evaluate(all4, f'LPFusion{COMMON_DIM}', *lp_local(splits, EO_MODELS, y_tr, y_va, COMMON_DIM), y_tr, y_va, y_te)
    # D. all GeoFMs + CLIP (Table 11)
    allm = EO_MODELS + ['CLIP']
    all5 = '+'.join(allm)
    evaluate(all5, 'NaiveConcat', *concat_local(splits, allm), y_tr, y_va, y_te)
    evaluate(all5, f'PCAFusion{COMMON_DIM}', *pca_local(splits, allm, COMMON_DIM), y_tr, y_va, y_te)
    evaluate(all5, f'LPFusion{COMMON_DIM}', *lp_local(splits, allm, y_tr, y_va, COMMON_DIM), y_tr, y_va, y_te)

    # E. temporal Transformer rows (Tables 9 to 12)
    run_transformers_for_seed(tr_ids, va_ids, te_ids, y_tr, y_va, y_te,
                              raw_tr, raw_te, mods_tr, mods_te)
    return list(rows_overall), list(rows_perclass)


def compute_seeds(seed_list, mods_tr, mods_te, df_tr_common, df_te_common, raw_tr, raw_te):
    """Run the given seeds, write one CSV pair per seed, return the concatenated frames."""
    ov_all, pc_all = [], []
    for seed in seed_list:
        ov_rows, pc_rows = run_one_seed(seed, mods_tr, mods_te, df_tr_common, df_te_common, raw_tr, raw_te)
        ov_seed, pc_seed = pd.DataFrame(ov_rows), pd.DataFrame(pc_rows)
        ov_seed.to_csv(f'{OUT_DIR}/results_seed{seed}_overall.csv', index=False)
        pc_seed.to_csv(f'{OUT_DIR}/results_seed{seed}_perclass.csv', index=False)
        print(f'[seed {seed}] wrote results_seed{seed}_overall.csv / _perclass.csv')
        ov_all.append(ov_seed); pc_all.append(pc_seed)
    return pd.concat(ov_all, ignore_index=True), pd.concat(pc_all, ignore_index=True)


def _read_all_seed_csvs():
    """Read every per-seed CSV in OUT_DIR (whatever seeds finished) and return the two frames."""
    import glob
    ov_files = sorted(glob.glob(f'{OUT_DIR}/results_seed*_overall.csv'))
    pc_files = sorted(glob.glob(f'{OUT_DIR}/results_seed*_perclass.csv'))
    if not ov_files or not pc_files:
        raise FileNotFoundError(f'no per-seed CSVs in {OUT_DIR}; run the seeds first')
    ov = pd.concat([pd.read_csv(f) for f in ov_files], ignore_index=True)
    pc = pd.concat([pd.read_csv(f) for f in pc_files], ignore_index=True)
    return ov, pc


def aggregate_and_write(ov, pc):
    """Aggregate the per-seed frames and write Tables 9 to 12 (LaTeX) and mean/std CSVs to OUT_DIR/tables."""
    tab_dir = f'{OUT_DIR}/tables'
    os.makedirs(tab_dir, exist_ok=True)
    n_seeds = ov['Seed'].nunique()
    seeds_present = sorted(ov['Seed'].unique().tolist())
    print(f'\naggregating {n_seeds} seeds: {seeds_present}')
    if n_seeds != 5:
        print(f'WARNING: captions say "five independent runs" but n_seeds={n_seeds}; edit captions or run the missing seeds.')

    ov_g = ov.groupby(['Data', 'Fusion', 'Classifier'])[['Kappa', 'OA', 'MacroF1', 'WeightedF1']].agg(['mean', 'std'])
    pc_g = pc.groupby(['Data', 'Fusion', 'Classifier', 'Class'])['F1'].agg(['mean', 'std'])

    def f(mean, std, dec=2):
        if pd.isna(mean):
            return ''
        s = 0.0 if pd.isna(std) else std
        return f'${mean:.{dec}f} \\pm {s:.{dec}f}$'

    def f01(mean, std, dec=2):
        # overall/per-class metrics are stored as percentages; per-class tables report the 0-1 scale
        if pd.isna(mean):
            return ''
        s = 0.0 if pd.isna(std) else std
        return f'${mean / 100:.{dec}f} \\pm {s / 100:.{dec}f}$'

    def ov_cells(key):
        """(kappa, W.F1, OA, M.F1) formatted cells for a (Data, Fusion, Classifier) key, or blanks."""
        if key not in ov_g.index:
            return ('', '', '', '')
        r = ov_g.loc[key]
        return (f(r[('Kappa', 'mean')], r[('Kappa', 'std')]), f(r[('WeightedF1', 'mean')], r[('WeightedF1', 'std')]),
                f(r[('OA', 'mean')], r[('OA', 'std')]), f(r[('MacroF1', 'mean')], r[('MacroF1', 'std')]))

    # ================= Table 9: single-modality overall (LinearProbe, RF, Transformer) =================
    SINGLE_MODS = ['CLIP', 'AlphaEarth', 'OlmoEarth', 'Presto', 'Tessera']

    def t8_row(data, fusion, clf, clf_label):
        cells = ' & '.join(ov_cells((data, fusion, clf)))
        return f'{data} & {clf_label} & {cells} \\\\'

    t8 = [r'\begin{table}[t]', r'\centering',
          r"\caption{RF/linear probe results on the 2022 dataset reported for representations extracted from: (1) geo-tagged field photos encoded using CLIP, and (2) the considered set of \ac{GeoFMs}. The achieved Cohen's Kappa ($\kappa$), Overall Accuracy (OA), macro F1-score (M. F1), and weighted F1-score (W. F1) are tested and compared with those of the transformer-based encoders trained independently on Sentinel-1 and Sentinel-2 data.}",
          r'\label{tab:results_single_2022}',
          r'\begin{tabular}{llcccc}', r'\toprule',
          r'\textbf{Data} & \textbf{Classifier} & \textbf{$\kappa$} & \textbf{W. F1 (\%)} & \textbf{OA (\%)} & \textbf{M. F1 (\%)} \\',
          r'\midrule']
    for mod in SINGLE_MODS:
        t8.append(t8_row(mod, 'Single', 'LinearProbe', 'LinearProbe'))
    t8.append(r'\hline')
    for mod in SINGLE_MODS:
        t8.append(t8_row(mod, 'Single', 'RandomForest', r'\ac{RF}'))
    t8.append(r'\hline')
    for data, fusion in T9_TRANSFORMER_ROWS:
        t8.append(t8_row(data, fusion, 'Transformer', 'Transformer'))
    t8 += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    open(f'{tab_dir}/table9.tex', 'w').write('\n'.join(t8) + '\n')

    # ================= Table 10: single-modality per-class F1 (0-1) =================
    def t9_cell(mod, cls_internal):
        key = (mod, 'Single', PERCLASS_CLF, cls_internal)
        return f01(pc_g.loc[key, 'mean'], pc_g.loc[key, 'std']) if key in pc_g.index else ''

    def t9_oa(mod):
        key = (mod, 'Single', PERCLASS_CLF)
        return f01(ov_g.loc[key, ('OA', 'mean')], ov_g.loc[key, ('OA', 'std')]) if key in ov_g.index else ''

    def t9_tf_cell(data, cls_internal):
        key = (data, 'Single', 'Transformer', cls_internal)
        return f01(pc_g.loc[key, 'mean'], pc_g.loc[key, 'std']) if key in pc_g.index else ''

    def t9_tf_oa(data):
        key = (data, 'Single', 'Transformer')
        return f01(ov_g.loc[key, ('OA', 'mean')], ov_g.loc[key, ('OA', 'std')]) if key in ov_g.index else ''

    T9_EMB = ['CLIP', 'AlphaEarth', 'OlmoEarth', 'Presto', 'Tessera']  # ClipPhoto == CLIP
    t9 = [r'\begin{table*}[t]', r'\centering',
          r'\caption{F1 scores across crop classes achieved on the 2022 test set for single-modality baselines. Results are reported as mean $\pm$ standard deviation across five independent runs.}',
          r'\label{tab:odd_individual_crop}',
          r'\renewcommand{\arraystretch}{1.2}', r'\resizebox{\textwidth}{!}{',
          r'\begin{tabular}{c l|ccccccc}', r'\toprule',
          r'\textbf{} & \textbf{Class} & \textbf{ClipPhoto} & \textbf{AlphaEarth} & \textbf{OlmoEarth} & \textbf{Presto} & \textbf{Tessera} & \textbf{S1} & \textbf{S2} \\',
          r'\midrule']
    for internal, disp, color in PC_ROWS:
        emb = ' & '.join(t9_cell(mod, internal) for mod in T9_EMB)
        s1 = t9_tf_cell('S1 only', internal)
        s2 = t9_tf_cell('S2 only', internal)
        t9.append(fr'\crop{{{color}}}{{}} & {disp} & {emb} & {s1} & {s2} \\')
    oa_emb = ' & '.join(t9_oa(mod) for mod in T9_EMB)
    t9 += [fr'\midrule & \textbf{{OA}} & {oa_emb} & {t9_tf_oa("S1 only")} & {t9_tf_oa("S2 only")}  \\ \bottomrule',
           r'\end{tabular}', r'}', r'\end{table*}']
    open(f'{tab_dir}/table10.tex', 'w').write('\n'.join(t9) + '\n')

    # ================= Table 11: fusion overall (LinearProbe then RF, + Transformer block) =================
    DATA = [('AlphaEarth+CLIP', 'AlphaEarth+CLIP'), ('OlmoEarth+CLIP', 'OlmoEarth+CLIP'),
            ('Presto+CLIP', 'Presto+CLIP'), ('Tessera+CLIP', 'Tessera+CLIP'),
            ('OlmoEarth+AlphaEarth+Presto+Tessera', 'All GeoFMs'),
            ('OlmoEarth+AlphaEarth+Presto+Tessera+CLIP', 'All GeoFMs+CLIP')]
    FUS = [('LPFusion32', 'Proj'), ('NaiveConcat', 'Naïve'), ('PCAFusion32', 'PCA')]

    t10 = [r'\begin{table*}[htp]', r'\centering',
           r"\caption{Cohen's Kappa ($\kappa$), Overall Accuracy (OA), macro F1-score (M. F1), and weighted F1-score (W. F1) obtained on the 2022 test set using linear probing and \ac{RF} under different fusion strategies (Naïve, Proj, and PCA). Results are reported for combinations of CLIP representations with individual and multiple \ac{GeoFMs}, as well as for models integrating Sentinel-1, Sentinel-2, and ancillary meteorological (M), topographic (T), and soil (S) data, with and without CLIP-derived representations. }",
           r'\label{tab:fusion_results_ood}',
           r'\renewcommand{\arraystretch}{1.1}', r'\resizebox{\textwidth}{!}{',
           r'\begin{tabular}{l l l|cccc}', r'\toprule',
           r'Classifier & Data & Fusion & $\kappa$ & W. F1 (\%) & OA (\%) & M. F1 (\%) \\', r'\midrule']
    for clf, clab in [('LinearProbe', 'Linear Probe'), ('RandomForest', 'Random Forest')]:
        t10.append(r'\multirow{18}{*}{' + clab + '}')
        for di, (dkey, dlab) in enumerate(DATA):
            for fi, (fkey, flab) in enumerate(FUS):
                k, w, o, m = ov_cells((dkey, fkey, clf))
                lead = f'&\t{dlab}\t' if fi == 0 else '&\t\t'
                t10.append(f'{lead}&\t{flab}\t&\t{k}\t&\t{w}\t&\t{o}\t&\t{m}\t\\\\')
            if di < len(DATA) - 1:
                t10.append(r'\cline{2-7}' if di == 3 else r'\cline{3-7}')
        t10.append(r'\midrule' if clf == 'LinearProbe' else r'\bottomrule')
    t10.append('')
    t10.append(r'\multirow{4}{*}{Transformer}')
    for i, (data, fus) in enumerate(T11_TRANSFORMER_ROWS):
        k, w, o, m = ov_cells((data, fus, 'Transformer'))
        tail = r'\\\cline{2-7}' if i == 1 else (r'\\\bottomrule' if i == len(T11_TRANSFORMER_ROWS) - 1 else r'\\')
        t10.append(f'&{data} & {fus} & {k} & {w} & {o} & {m} {tail}')
    t10 += [r'\end{tabular}', r'}', r'\end{table*}']
    open(f'{tab_dir}/table11.tex', 'w').write('\n'.join(t10) + '\n')

    # ================= Table 12: fusion per-class F1 (0-1). Tessera+CLIP = best-OA combo =================
    def best_oa_combo(dk):
        best, best_oa = None, -1.0
        for clf in ['LinearProbe', 'RandomForest']:
            for fk in ['LPFusion32', 'NaiveConcat', 'PCAFusion32']:
                key = (dk, fk, clf)
                if key in ov_g.index:
                    oa = ov_g.loc[key, ('OA', 'mean')]
                    if pd.notna(oa) and oa > best_oa:
                        best_oa, best = oa, (clf, fk)
        return best

    tc = best_oa_combo('Tessera+CLIP')
    if tc is None:
        print('WARNING: no Tessera+CLIP fusion results; Table 12 Tessera+CLIP column will be blank.')
    else:
        print(f'Table 12 Tessera+CLIP column uses best-OA combo: {tc[0]} + {tc[1]}')

    def t11_tc_cell(cls_internal):
        if tc is None:
            return ''
        key = ('Tessera+CLIP', tc[1], tc[0], cls_internal)
        return f01(pc_g.loc[key, 'mean'], pc_g.loc[key, 'std']) if key in pc_g.index else ''

    def t11_tc_oa():
        if tc is None:
            return ''
        key = ('Tessera+CLIP', tc[1], tc[0])
        return f01(ov_g.loc[key, ('OA', 'mean')], ov_g.loc[key, ('OA', 'std')]) if key in ov_g.index else ''

    def t11_tf_cell(cls_internal):
        key = ('S1+S2+TMS+CLIP', 'Mid-Level', 'Transformer', cls_internal)
        return f01(pc_g.loc[key, 'mean'], pc_g.loc[key, 'std']) if key in pc_g.index else ''

    def t11_tf_oa():
        key = ('S1+S2+TMS+CLIP', 'Mid-Level', 'Transformer')
        return f01(ov_g.loc[key, ('OA', 'mean')], ov_g.loc[key, ('OA', 'std')]) if key in ov_g.index else ''

    t11 = [r'\begin{table*}[htp]', r'\centering',
           r'\caption{F1 scores across crop classes achieved on the 2022 test set for multimodal fusion strategies combining \ac{GeoFMs}, Sentinel-1/2, ancillary variables (TMS), and photo embeddings. Results are reported as mean $\pm$ standard deviation across five independent runs.}',
           r'\label{tab:odd_fusion}',
           r'\renewcommand{\arraystretch}{1.2}',
           r'\begin{tabular}{c l|cc}', r'\toprule',
           r'\textbf{} & \textbf{Class} & \textbf{S1+S2+TMS+CLIP} & \textbf{Tessera+CLIP} \\', r'\midrule']
    for internal, disp, color in PC_ROWS:
        t11.append(fr'\crop{{{color}}}{{}} & {disp} & {t11_tf_cell(internal)} & {t11_tc_cell(internal)} \\')
    t11 += [fr'\midrule & \textbf{{OA}} & {t11_tf_oa()} & {t11_tc_oa()} \\ \bottomrule',
            r'\end{tabular}', r'\end{table*}']
    open(f'{tab_dir}/table12.tex', 'w').write('\n'.join(t11) + '\n')

    # tidy CSVs
    ov_flat = ov_g.copy()
    ov_flat.columns = ['_'.join(c) for c in ov_flat.columns]
    ov_flat.reset_index().to_csv(f'{tab_dir}/overall_meanstd.csv', index=False)
    pc_g.reset_index().to_csv(f'{tab_dir}/perclass_meanstd.csv', index=False)
    print(f'wrote table9-12.tex and mean/std CSVs to {tab_dir}')

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if AGGREGATE_ONLY:
        print(f'[aggregate] reading per-seed CSVs from {OUT_DIR}')
        ov, pc = _read_all_seed_csvs()
        aggregate_and_write(ov, pc)
        return
    if not DATA:
        raise SystemExit('set GEOCROP_DATA to the input directory (see README.md)')

    seeds = [ONE_SEED] if ONE_SEED is not None else SEEDS
    print(f'device={DEVICE} | seeds={seeds}')
    print(f'loading {TRAIN_YEAR} and {TEST_YEAR} embeddings once')
    mods_tr = load_year_mods(TRAIN_YEAR)
    mods_te = load_year_mods(TEST_YEAR)
    raw_tr, raw_te = load_year_raw(TRAIN_YEAR), load_year_raw(TEST_YEAR)
    df_tr_common = common_label_df(load_labels(f'{DATA}/{TRAIN_YEAR}/LUCAS_{TRAIN_YEAR}_LULC.csv'), mods_tr)
    df_te_common = common_label_df(load_labels(f'{DATA}/{TEST_YEAR}/LUCAS_{TEST_YEAR}_LULC.csv'), mods_te)

    ov, pc = compute_seeds(seeds, mods_tr, mods_te, df_tr_common, df_te_common, raw_tr, raw_te)

    if ONE_SEED is not None:
        print(f'[seed {ONE_SEED}] done; run GEOCROP_AGGREGATE=1 after all seeds finish.')
        return
    aggregate_and_write(ov, pc)


if __name__ == '__main__':
    main()
