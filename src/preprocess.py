from asyncio import Task
from aws_lambda_powertools.utilities.data_classes.sqs_event import SQSRecord
from aws_lambda_powertools.utilities.typing import LambdaContext
from aws_lambda_powertools import Logger
import base64
from PIL import Image
from io import BytesIO
from numpy.typing import NDArray
from transformers import CLIPImageProcessorPil
from config.schemas import EmbedRecord

logger = Logger()


async def record_handler(record: SQSRecord, lambda_context: LambdaContext):
    logger.info(lambda_context)
    logger.info(f"Record: {record}")
    logger.info(record)
    logger.info(record.body)

    processor_task: Task[CLIPImageProcessorPil] = getattr(lambda_context, "clip_processor_task")

    image_record = EmbedRecord.model_validate(record)
    logger.info({"mapped_image_record": image_record})

    raw_image = Image.open(BytesIO(base64.b64decode(image_record.image_b64.encode('utf-8'))))
    logger.info({"raw_image": raw_image})

    processor = await processor_task
    image_arr: NDArray = processor(raw_image)["pixel_values"]
    logger.info({"image_arr_shape": image_arr.shape})
    image_record.image_array = image_arr
    return image_record
