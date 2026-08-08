from asyncio import Task
from aws_lambda_powertools.utilities.typing import LambdaContext
from aws_lambda_powertools import Logger
from onnxruntime import InferenceSession
from lancedb import AsyncTable
from numpy.typing import NDArray
from utils.loop import ensure_loop

logger = Logger()


async def process_content(items: list[NDArray], lambda_context: LambdaContext):
    loop = ensure_loop()

    logger.info(lambda_context)
    logger.info({"items": items})

    item_tower_task: Task[InferenceSession] = getattr(lambda_context, "item_tower_task")
    clip_vis_task: Task[InferenceSession] = getattr(lambda_context, "clip_vis_task")
    cover_table_task: Task[AsyncTable] = getattr(lambda_context, "cover_table_task")

    clip_vis = loop.run_until_complete(clip_vis_task)
    logger.info({"clip_vis": clip_vis})
    loop.run_until_complete(item_tower_task)
    loop.run_until_complete(cover_table_task)
