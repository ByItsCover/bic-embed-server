import asyncio
from asyncio import Task
from aws_lambda_powertools.utilities.typing import LambdaContext
from aws_lambda_powertools import Logger
from onnxruntime import InferenceSession
from lancedb import AsyncTable
from numpy.typing import NDArray
from utils.loop import ensure_loop
from config.schemas import EmbedRecord
from config.constants import CLIP_INPUT_NAME

logger = Logger()


async def process_content(items: list[EmbedRecord], lambda_context: LambdaContext):
    loop = ensure_loop()

    logger.info(lambda_context)
    logger.info({"items": items})

    #item_tower_task: Task[InferenceSession] = getattr(lambda_context, "item_tower_task")
    #clip_vis_task: Task[InferenceSession] = getattr(lambda_context, "clip_vis_task")
    cover_table_task: Task[AsyncTable] = getattr(lambda_context, "cover_table_task")

    images_list = [record.image_array for record in items]

    """
    clip_vis = loop.run_until_complete(clip_vis_task)
    embeddings = await asyncio.to_thread(
        clip_vis.run,
        None,
        {CLIP_INPUT_NAME: images_list}
    )
    processed_embeddings: list[NDArray] = embeddings[0].tolist()
    """
    
    cover_table = loop.run_until_complete(cover_table_task)
    logger.info({"cover_table": cover_table})
    #loop.run_until_complete(item_tower_task)
    #loop.run_until_complete(cover_table_task)
