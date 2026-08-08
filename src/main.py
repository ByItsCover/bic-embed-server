from aws_lambda_powertools import Logger
from aws_lambda_powertools.utilities.batch import (
    AsyncBatchProcessor,
    EventType,
)
from aws_lambda_powertools.utilities.typing import LambdaContext
from middleware import model_middleware, lance_middleware, s3_middleware
from preprocess import record_handler
from postprocess import process_content
from utils.loop import ensure_loop
from config.schemas import EmbedRecord

processor = AsyncBatchProcessor(event_type=EventType.SQS)
logger = Logger()


@model_middleware
@lance_middleware
@s3_middleware
def lambda_handler(event, context: LambdaContext):
    loop = ensure_loop()
    records = event["Records"]
    with processor(records, record_handler, context):
        processed_messages = processor.async_process()

    items: list[EmbedRecord] = []
    for status, result, record in processed_messages:
        logger.info(result)
        items += result

    loop.run_until_complete(process_content(items, context))
    return processor.response()
