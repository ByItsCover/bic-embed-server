from asyncio import Task
from aws_lambda_powertools.utilities.typing import LambdaContext
from aws_lambda_powertools import Logger
from onnxruntime import InferenceSession
from transformers import CLIPImageProcessorPil

logger = Logger()


async def embed_content(items: list[int], lambda_context: LambdaContext):
    logger.info(lambda_context)
    logger.info({"items": items})

    item_tower_task: Task[InferenceSession] = getattr(lambda_context, "item_tower_task")
    clip_vis_task: Task[InferenceSession] = getattr(lambda_context, "clip_vis_task")
    processor_task: Task[CLIPImageProcessorPil] = getattr(lambda_context, "clip_processor_task")

    processor = await processor_task
    logger.info({"processor": processor})
    await item_tower_task
    await clip_vis_task
