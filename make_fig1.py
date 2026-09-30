#!/usr/bin/env python
"""Figure 1 of the paper: LUCAS crop points of 2018 and 2022 on a country basemap, one figure per year.

Environment:
    GEOCROP_DATA     input directory with 2018/, 2022/ and ne_50m_admin_0_countries.zip (README.md)
    GEOCROP_FIG_OUT  output directory (default: results/figures next to this file)
Writes lucas_sample_map_<year>.png and .pdf. Both years share the same extent and class colours.
"""
import os
import struct
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.path import Path as MplPath
from matplotlib.collections import PathCollection

from crop_classes import CLMS_MAP

DATA = os.environ['GEOCROP_DATA']
NE_ZIP = f'{DATA}/ne_50m_admin_0_countries.zip'
OUT = Path(os.environ.get('GEOCROP_FIG_OUT', Path(__file__).resolve().parent / 'results' / 'figures'))
YEARS = [2018, 2022]

CLS_CODES = sorted(set(v[0] for v in CLMS_MAP.values()))
CLS_NAME = {v[0]: v[1] for v in CLMS_MAP.values()}
# CLMS class colours, as in Table 1 of the paper
CLS_COLOR = {
    "1110": "#EB6E31", "1120": "#FAA24A", "1130": "#F9DC15", "1140": "#E94101", "1150": "#EAA993",
    "1210": "#ADC5E8", "1220": "#4895BC", "1310": "#C78B43", "1320": "#9B5C0D", "1410": "#FC7978",
    "1420": "#A76A95", "1430": "#E179C1", "1440": "#F7B7D2", "2100": "#DADB8D", "2200": "#BFCE10",
    "2310": "#79A038", "2320": "#5A7A31",
}
TITLE_FS, LABEL_FS, TICK_FS, LEGEND_FS = 15, 13, 11, 8


def read_shp_polygons(path, bbox):
    """Polygon records of the .shp inside a zip, as lists of rings (Nx2 arrays); records outside bbox are skipped."""
    with zipfile.ZipFile(path) as z:
        buf = z.read(next(n for n in z.namelist() if n.lower().endswith('.shp')))
    if struct.unpack('>i', buf[:4])[0] != 9994:
        raise ValueError(f'{path}: not an ESRI shapefile')
    flen = struct.unpack('>i', buf[24:28])[0] * 2
    pos, shapes = 100, []
    while pos < flen:
        clen = struct.unpack('>i', buf[pos + 4:pos + 8])[0] * 2
        rec = buf[pos + 8:pos + 8 + clen]
        pos += 8 + clen
        if struct.unpack('<i', rec[:4])[0] % 10 != 5:          # 5/15/25 = Polygon*
            continue
        x0, y0, x1, y1 = struct.unpack('<4d', rec[4:36])
        if x1 < bbox[0] or x0 > bbox[1] or y1 < bbox[2] or y0 > bbox[3]:
            continue
        nparts, npts = struct.unpack('<2i', rec[36:44])
        parts = struct.unpack(f'<{nparts}i', rec[44:44 + 4 * nparts]) + (npts,)
        pts = np.frombuffer(rec, '<f8', 2 * npts, 44 + 4 * nparts).reshape(-1, 2)
        shapes.append([pts[parts[i]:parts[i + 1]] for i in range(nparts)])
    return shapes


def draw_basemap(ax, shapes):
    """Country polygons behind points and grid; rings of one shape form one path, so holes render."""
    paths = []
    for rings in shapes:
        codes = np.concatenate([[MplPath.MOVETO] + [MplPath.LINETO] * (len(r) - 1) for r in rings])
        paths.append(MplPath(np.concatenate(rings), codes))
    ax.add_collection(PathCollection(paths, facecolor='0.95', edgecolor='0.6', linewidth=0.4, zorder=0))


def load_year(year):
    """Crop points of the 17 classes with LAT_USED/LON_USED coordinates."""
    df = pd.read_csv(f'{DATA}/{year}/LUCAS_{year}_LULC.csv', low_memory=False)
    df = df[df['LC1'].astype(str).isin(CLMS_MAP)].copy()
    df['clms_code'] = df['LC1'].map(lambda x: CLMS_MAP[x][0])
    df['lat'] = pd.to_numeric(df['LAT_USED'], errors='coerce')
    df['lon'] = pd.to_numeric(df['LON_USED'], errors='coerce')
    df = df.dropna(subset=['lat', 'lon'])
    df = df[df['lon'].between(-30, 50) & df['lat'].between(30, 75)]   # drop stray / zero coordinates
    print(f'{year}: {len(df):,} crop points')
    return df[['clms_code', 'lat', 'lon']]


panels = [(y, load_year(y)) for y in YEARS]
world = read_shp_polygons(NE_ZIP, bbox=(-30, 50, 25, 75))
allon = np.concatenate([d['lon'].values for _, d in panels])
allat = np.concatenate([d['lat'].values for _, d in panels])
xlim = (allon.min() - 1, allon.max() + 1)
ylim = (allat.min() - 1, allat.max() + 1)
aspect = 1.0 / np.cos(np.deg2rad(0.5 * (ylim[0] + ylim[1])))   # latitude-corrected aspect
handles = [Patch(facecolor=CLS_COLOR[c], edgecolor='0.3', linewidth=0.5, label=CLS_NAME[c]) for c in CLS_CODES]

OUT.mkdir(parents=True, exist_ok=True)
for year, d in panels:
    fig, ax = plt.subplots(figsize=(7.0, 7.4))
    draw_basemap(ax, world)
    for code in d['clms_code'].value_counts().index:              # common classes first, rare ones on top
        m = d['clms_code'].values == code
        ax.scatter(d['lon'].values[m], d['lat'].values[m], s=4, linewidths=0,
                   color=CLS_COLOR[code], alpha=0.55, rasterized=True)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect(aspect)
    ax.set_title(f'{year}   (n = {len(d):,} crop samples)', fontsize=TITLE_FS)
    ax.set_xlabel('longitude', fontsize=LABEL_FS); ax.set_ylabel('latitude', fontsize=LABEL_FS)
    ax.tick_params(labelsize=TICK_FS)
    ax.grid(True, lw=0.3, alpha=0.4)
    ax.legend(handles=handles, loc='upper left', fontsize=LEGEND_FS, framealpha=0.9, borderpad=0.4,
              labelspacing=0.25, handlelength=1.2, handleheight=0.7)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'lucas_sample_map_{year}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('saved', OUT / f'lucas_sample_map_{year}.png')
