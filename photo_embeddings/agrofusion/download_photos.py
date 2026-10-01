# SPDX-FileCopyrightText: 2025-2026 AgroVision <s.maurogiovanni@fz-juelich.de>
#
# SPDX-License-Identifier: MIT

import asyncio
import contextlib
import math
import os
import sys
from asyncio import Queue, Semaphore
from collections.abc import Iterable
from contextlib import AsyncExitStack
from io import BytesIO
from pathlib import Path

import dvc.api
import polars as pl
from aiohttp import ClientSession
from loguru import logger
from PIL import Image
from tqdm.asyncio import tqdm
from webdataset import ShardWriter

from agrofusion.download_utils import (
    LUCAS_METADATA_SCHEMA,
    LucasSuffix,
    extend_metadata_frame,
)

# TODO: automate project root discovery
project_dir = Path(os.getenv("HOME", ".")) / "shared" / "agrofusion"


# Avoid successful downloaded point IDs be printed to stdout
logger.remove()
logger.add(sink=sys.stdout, filter=lambda record: record["level"].name != "SUCCESS")

# Remove known corrupted photos to prevent download loop from stalling
# TODO: catch related exception and handle accordingly
NUTS_BLACKLIST = "402029"


async def fetch_items(
    batch: Iterable,
    queue: Queue,
    session: ClientSession,
    semaphore: Semaphore,
):
    async with semaphore:
        for item in batch:
            async with session.get(item["url"]) as response:
                await queue.put(item | {"response": response})


async def download_item(item_url, session, semaphore, sink, chunk_size: int = 1024):
    async with semaphore:
        async with session.get(item_url) as response:
            while chunk := await response.content.read(chunk_size):
                sink.write(chunk)
    return sink


async def stream_items(src_queue, writer, session, semaphore, progress_bar, logger):
    while (item := await src_queue.get()) is not None:
        if item["response"].ok:
            img_file = BytesIO()
            payload_buffer = await download_item(
                item["url"], session, semaphore, img_file
            )
            payload_buffer.seek(0)  # important: rewind to start

            writer.write(
                {
                    "__key__": item["POINT_ID"],
                    "gisco_url": item["url"],
                    "image": Image.open(payload_buffer).convert("RGB"),
                }
            )
            logger.success(item["POINT_ID"])
        else:
            logger.warning(item["POINT_ID"])
        src_queue.task_done()
        progress_bar.update(1)
    else:
        src_queue.task_done()


async def main(args) -> None:
    # ========================================================================
    # Datasets merging
    # ========================================================================

    num_samples = args["download"].get("num_samples")
    year = args["metadata"].get("year")
    direction_suffix = LucasSuffix(args["metadata"].get("direction"))

    metadata_file = project_dir / args["metadata"].get("metadata_file", ".")

    photos_dir = project_dir / args["metadata"].get("photos_dir", ".")
    photos_dir.mkdir(exist_ok=True, parents=True)

    logger.info(f"Reading {year} metadata from: {metadata_file}")
    metadata_frame = (
        pl.read_csv(metadata_file, schema_overrides=LUCAS_METADATA_SCHEMA)
        .filter(pl.col("clms_mapped"))
        .filter(~pl.col("POINT_ID").str.starts_with(NUTS_BLACKLIST))
    )
    if num_samples:
        metadata_frame = metadata_frame.sample(
            num_samples, seed=args["download"].get("seed")
        )
    """
    if args["download"]["ids_file"]:
        import json

        ids_file = Path(args["download"]["ids_file"])
        with open(ids_file, mode="r") as fin:
            ids_frame = pl.from_dict(json.load(fin))
        metadata_frame = metadata_frame.join(ids_frame, on="POINT_ID")

    logger.debug(ids_frame)
    """

    extended_frame = extend_metadata_frame(
        metadata_frame, year=year, suffix=direction_suffix
    )

    logger.info(f"Attempting download of {len(extended_frame)} items...")
    # ========================================================================
    # Photos download
    # ========================================================================

    total_num_items = len(extended_frame)
    items_batch_size = math.ceil(total_num_items / args["download"]["num_fetch_tasks"])

    batched_frame = extended_frame.select("POINT_ID", "url", "filename").iter_slices(
        n_rows=items_batch_size
    )

    # Collect item responses
    items_queue = Queue(maxsize=args["download"]["queue_size"])
    fetch_semaphore = Semaphore(args["download"]["max_fetch_requests"])
    stream_semaphore = Semaphore(args["download"]["max_stream_requests"])

    async with AsyncExitStack() as stack:
        progress_bar = stack.enter_context(
            tqdm(total=total_num_items, desc="Streaming images", unit="image")
        )
        with contextlib.redirect_stdout(None):
            writer = stack.enter_context(
                ShardWriter(
                    f"{photos_dir}/shard-%06d.tar",
                    maxsize=args["download"].get("shard_size", 2e9),
                )
            )
        session = await stack.enter_async_context(ClientSession())

        fetch_futures = [
            asyncio.create_task(
                fetch_items(
                    items_batch.iter_rows(named=True),
                    items_queue,
                    session,
                    fetch_semaphore,
                )
            )
            for items_batch in batched_frame
        ]

        stream_futures = [
            asyncio.create_task(
                stream_items(
                    src_queue=items_queue,
                    writer=writer,
                    session=session,
                    semaphore=stream_semaphore,
                    progress_bar=progress_bar,
                    logger=logger,
                )
            )
            for _ in range(args["download"]["num_stream_tasks"])
        ]

        await asyncio.gather(*fetch_futures)

        for _ in range(args["download"]["num_stream_tasks"]):
            await items_queue.put(None)

        await items_queue.join()
        await asyncio.gather(*stream_futures)

        for future in stream_futures:
            future.cancel()


if __name__ == "__main__":
    stage = os.getenv("DVC_STAGE", "")
    args = dvc.api.params_show(stages=stage)

    # ========================================================================
    # Artifacts generation
    # ========================================================================

    year = args["metadata"].get("year")
    direction_suffix = LucasSuffix(args["metadata"].get("direction"))

    if isinstance(artifact_args := args.get("artifacts"), dict):
        for artifact in artifact_args.values():
            artifact_path = project_dir / artifact["path"]
            artifact_dir = (
                artifact_path if artifact_path.is_dir() else artifact_path.parent
            )
            artifact_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(ids_artifact := artifact_args.get("point_ids"), dict):
            ids_file_found = (
                project_dir
                / ids_artifact["path"]
                / f"found_{year}_{direction_suffix.value}.log"
            )
            logger.add(
                sink=str(ids_file_found),
                level="SUCCESS",
                format="{message}",
                enqueue=True,
                filter=lambda record: record["level"].name == "SUCCESS",
            )

            ids_file_notfound = (
                project_dir
                / ids_artifact["path"]
                / f"notfound_{year}_{direction_suffix.value}.log"
            )
            logger.add(
                sink=str(ids_file_notfound),
                level="WARNING",
                format="{message}",
                enqueue=True,
                filter=lambda record: record["level"].name == "WARNING",
            )

    """
    stage_prefix = stage.split(":")[0]
    logger.debug(stage_prefix)
    logger.debug(dvc.api.artifacts_show(name=f"{stage_prefix}:point_ids", stage=stage))
   

    # FIXME: Remove this section after refactoring params file organization
    
    dataset_label = "-".join(stage.split("-")[1:])  # {suffix}-{year}
    
    params = dvc.api.params_show(stages=stage)
    download_args = params["download"]
    dataset_args = params["dataset"][dataset_label]
    metadata_args = params["metadata"][dataset_args["year"]]
    
    args = {
        "dataset": dataset_args,
        "metadata": metadata_args,
        "download": download_args,
    }
    logger.debug(stage)
    logger.debug(json.dumps(args))
    """
    logger.info(f"Logging information about existent URLs to: {ids_file_found}")
    logger.info(f"Logging information about non-existent URLs to: {ids_file_notfound}")

    asyncio.run(main(args))
