import asyncio
from asyncio import Task
from aws_lambda_powertools.utilities.data_classes.sqs_event import SQSRecord
from aws_lambda_powertools.utilities.typing import LambdaContext
from aws_lambda_powertools import Logger
from transformers import CLIPImageProcessorPil
from pydantic import TypeAdapter
from types_aiobotocore_s3.service_resource import Bucket
from PIL import Image
from io import BytesIO
from numpy.typing import NDArray
from collections.abc import Awaitable
import json
from config.schemas import S3Record, EmbedRecord, ProcessError

logger = Logger()


async def build_record(message_id: str, metadata_task: Awaitable[dict[str, str]]):
    metadata = await metadata_task
    logger.info({"metadata": metadata})
    record = EmbedRecord(**metadata, messageId=message_id)
    logger.info({"parsed_record": record})
    return record

async def fetch_cover(message_id: str, s3_record: S3Record, bucket: Bucket, processor_task: Task[CLIPImageProcessorPil]):
    if s3_record.bucket_name != bucket.name:
        raise ProcessError(message_id, f"S3 record from incorrect bucket. "
                    f"Expected {bucket.name}, but received {s3_record.bucket_name}")

    obj = await bucket.Object(s3_record.key)
    metadata_task = obj.metadata
    record_task = build_record(message_id, metadata_task)

    response = await obj.get()
    image_file = await response.get("Body").read()
    raw_image = Image.open(BytesIO(image_file))
    logger.info({"raw_image": raw_image})

    processor = await processor_task
    image_arr: NDArray = processor(raw_image)["pixel_values"][0]
    logger.info({"image_arr": image_arr})

    record = await record_task
    record.image_array = image_arr
    return record

async def record_handler(record: SQSRecord, lambda_context: LambdaContext):
    logger.info(lambda_context)
    logger.info(f"Record: {record}")

    processor_task: Task[CLIPImageProcessorPil] = getattr(lambda_context, "clip_processor_task")
    cover_dump_task: Task[Bucket] = getattr(lambda_context, "cover_dump_task")

    record_body: dict = json.loads(record.body)
    s3_record_adapter = TypeAdapter(list[S3Record])
    s3_records = s3_record_adapter.validate_python(record_body.get("Records", []))

    cover_dump = await cover_dump_task
    cover_tasks = [
        fetch_cover(record.message_id, s3_rec, cover_dump, processor_task)
        for s3_rec in s3_records
    ]

    records: list[EmbedRecord] = await asyncio.gather(*cover_tasks)
    return records
