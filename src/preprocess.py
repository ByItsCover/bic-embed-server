import asyncio
from asyncio import Task
from aws_lambda_powertools.utilities.data_classes.sqs_event import SQSRecord
from aws_lambda_powertools.utilities.typing import LambdaContext
from aws_lambda_powertools import Logger
from transformers import CLIPImageProcessorPil
from config.schemas import EmbedRecord, ProcessError

from aiohttp import ClientSession
from PIL import Image
from PIL.ImageFile import ImageFile
import io
from numpy.typing import NDArray

logger = Logger()


async def fetch_raw_image(record: EmbedRecord, http_session: ClientSession) -> ImageFile:
    try:
        async with http_session.get(url=record.image_url) as response:
            res = await response.read()
            image = Image.open(io.BytesIO(res))
    except Exception as ex:
        raise ProcessError(record.message_id, message=str(ex))
    else:
        return image

async def record_handler(record: SQSRecord, lambda_context: LambdaContext):
    logger.info(lambda_context)
    logger.info(f"Record: {record}")
    logger.info(record)
    logger.info(record.body)

    http_session_task: Task[ClientSession] = getattr(lambda_context, "http_session_task")
    processor_task: Task[CLIPImageProcessorPil] = getattr(lambda_context, "clip_processor_task")

    image_record = EmbedRecord.model_validate(record)
    logger.info({"mapped_image_record": image_record})

    http_session = await http_session_task
    raw_image_task = asyncio.create_task(fetch_raw_image(image_record, http_session))
    raw_image = await raw_image_task
    logger.info({"raw_image": raw_image})

    processor = await processor_task
    image_arr: NDArray = processor(raw_image)["pixel_values"]
    logger.info({"image_arr_shape": image_arr.shape})
    image_record.image_array = image_arr
    return image_record
