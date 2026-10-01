# SPDX-FileCopyrightText: 2025-2026 AgroVision <s.maurogiovanni@fz-juelich.de>
#
# SPDX-License-Identifier: MIT

import itertools
from collections import defaultdict
from io import BytesIO

import safetensors.torch
from PIL import Image
from webdataset import WebDataset


def load_data(src, load_fn, batch_size=5, transform_fn=None):
    batches = itertools.batched(src, n=batch_size)
    for batch in batches:
        batch_dict = defaultdict(list)
        for path in batch:
            item = load_fn(path)
            item = item if transform_fn is None else transform_fn(item)
            batch_dict["path"].append(path)
            batch_dict["value"].append(item)

        yield batch_dict


def webdataset_pil_decoder(key, data):
    """Decode WebDataset shards storing RGB field-level photos"""
    if key.endswith(".image"):
        assert isinstance(data, bytes), type(data)
        return Image.open(BytesIO(data))
    return data


def webdataset_safetensors_decoder(key, data):
    """Decode WebDataset shards storing PyTorch tensors in the safetensors format"""
    if key.endswith(".mask"):
        assert isinstance(data, bytes), type(data)
        return safetensors.torch.load(data)
    return data


def get_samples_from_wds(folder_path, decoder, shuffle=False, seed=None, num_samples=5):
    """Randomly sample elements from a WebDataset dataset"""
    shards = sorted([str(path) for path in folder_path.glob("*.tar")])

    dataset = WebDataset(shards, shardshuffle=shuffle, seed=seed).decode(decoder)

    return list(itertools.islice(iter(dataset), num_samples))
