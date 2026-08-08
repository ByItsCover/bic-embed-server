import asyncio
from asyncio import Task
from aws_lambda_powertools.utilities.typing import LambdaContext
from aws_lambda_powertools import Logger
from onnxruntime import InferenceSession
from lancedb import AsyncTable
import numpy as np
from numpy.typing import NDArray
from pydantic import TypeAdapter
from utils.array_ops import normalize
from config.schemas import EmbedRecord, Cover
from config.constants import (CLIP_INPUT_NAME, TOWER_ITEM_INPUT,
                              TOWER_ID_INPUT, COVER_ID_FIELD)

logger = Logger()


async def process_content(records: list[EmbedRecord], lambda_context: LambdaContext):
    logger.info(lambda_context)
    logger.info({"items_count": len(records)})

    item_tower_task: Task[InferenceSession] = getattr(lambda_context, "item_tower_task")
    clip_vis_task: Task[InferenceSession] = getattr(lambda_context, "clip_vis_task")
    cover_table_task: Task[AsyncTable] = getattr(lambda_context, "cover_table_task")

    images_list = []
    id_list = []
    for record in records:
        images_list.append(record.image_array)
        id_list.append(record.cover_id)

    clip_vis = await clip_vis_task
    logger.info({"clip_vis": clip_vis})
    clip_embeddings = await asyncio.to_thread(
        clip_vis.run,
        None,
        {CLIP_INPUT_NAME: np.stack(images_list, dtype=np.float32)}
    )
    clip_embeds_processed: list[NDArray] = [
        normalize(embed) for embed in clip_embeddings[0].tolist()
    ]

    item_tower = await item_tower_task
    logger.info({"item_tower": item_tower})
    tower_embeddings = await asyncio.to_thread(
        item_tower.run,
        None,
        {
            TOWER_ITEM_INPUT: np.stack(clip_embeds_processed, dtype=np.float32),
            TOWER_ID_INPUT: np.array(id_list, dtype=np.int32)
        }
    )
    tower_embeds_processed: list[NDArray] = [
        normalize(embed) for embed in tower_embeddings[0].tolist()
    ]

    cover_list = [
        Cover(
            **records[i].model_dump(),
            cover_embedding=clip_embeds_processed[i],
            tower_embedding=tower_embeds_processed[i]
        )
        for i in range(len(records))
    ]
    logger.info({"cover_list": cover_list})

    covers_adapter = TypeAdapter(list[Cover])
    cover_table = await cover_table_task
    logger.info({"cover_table": cover_table})

    table_res = await (
        cover_table.merge_insert(COVER_ID_FIELD)
        .when_matched_update_all()
        .when_not_matched_insert_all()
        .execute(covers_adapter.dump_python(cover_list))
    )
    logger.info({"table_res": table_res})
