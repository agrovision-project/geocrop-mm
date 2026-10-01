# SPDX-FileCopyrightText: 2026 Stefano Maurogiovanni <s.maurogiovanni@fz-juelich.de>
#
# SPDX-License-Identifier: MIT

import os
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from time import perf_counter

import dvc.api
import numpy as np
import torch
from human_readable import time_delta
from loguru import logger
from PIL import Image
from tqdm.auto import tqdm
from transformers import (
    CLIPImageProcessor,
    MetaClip2VisionModelWithProjection,
    PreTrainedModel,
)
from webdataset import WebDataset

from agrofusion.processing_utils import webdataset_pil_decoder


def compute_img_embeddings(
    image: Image.Image, model: PreTrainedModel, processor: CLIPImageProcessor
) -> torch.Tensor:
    with torch.inference_mode():
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs.to(model.device))
        return outputs.image_embeds


project_dir = Path(os.getenv("HOME", ".")) / "shared" / "agrofusion"
data_dir = project_dir / "data"


def main(args) -> None:
    with open(
        project_dir / "artifacts" / "vlm_embeddings" / "found_2018_point.log", mode="r"
    ) as src:
        num_items = len(src.readlines())

    # ========================================================================
    # Dataset
    # ========================================================================

    folder_path = project_dir / args["metadata"]["photos_dir"]
    batch_size = args["model"].get("batch_size", 1)

    shards = sorted([str(path) for path in folder_path.glob("*.tar")])
    dataset = iter(
        WebDataset(shards, shardshuffle=100, seed=42)
        .decode(webdataset_pil_decoder)
        .batched(batch_size)
    )

    # ========================================================================
    # Model
    # ========================================================================

    model_id = args["model"]["id"]
    local_files_only = args["model"]["local_files_only"]
    device = args["model"]["device"]
    processor = CLIPImageProcessor.from_pretrained(
        model_id, local_files_only=local_files_only
    )
    model = MetaClip2VisionModelWithProjection.from_pretrained(
        model_id,
        device_map=device,
        local_files_only=local_files_only,
    )

    logger.debug(processor)
    logger.debug(model)

    # ========================================================================
    # Processing
    # ========================================================================

    data = defaultdict(list)

    outfile = project_dir / args["metadata"]["embeddings_file"]
    outfile.parent.mkdir(parents=True, exist_ok=True)

    num_steps = (num_items // batch_size) + (num_items % batch_size != 0)
    start = perf_counter()
    for item in tqdm(
        dataset, total=num_steps, desc="Computing and storing image embeddings"
    ):
        embeddings = compute_img_embeddings(item["image"], model, processor)

        data["point_id"].extend(item["__key__"])
        data["image_embeds"].append(embeddings)
        data["url"].extend(item["gisco_url"])
    data["data"] = torch.cat(data.pop("image_embeds")).cpu()

    end = perf_counter()
    logger.info(f"Took: {time_delta(timedelta(end - start))}")

    np.savez(outfile, **data, allow_pickle=False)


if __name__ == "__main__":
    stage = os.getenv("DVC_STAGE", "")
    args = dvc.api.params_show(stages=stage)

    logger.debug("\n" + (50 * "="))
    logger.debug(args)
    logger.debug(50 * "=" + "\n")

    main(args)
