from aws_lambda_powertools import Logger
from aws_lambda_powertools.utilities.batch import (
    AsyncBatchProcessor,
    EventType,
)
from aws_lambda_powertools.utilities.typing import LambdaContext
from middleware import model_middleware, lance_middleware, http_middleware
from preprocess import record_handler

processor = AsyncBatchProcessor(event_type=EventType.SQS)
logger = Logger()


@model_middleware
@lance_middleware
@http_middleware
def lambda_handler(event, context: LambdaContext):
    records = event["Records"]
    with processor(records, record_handler, context):
        processed_messages = processor.async_process()

    for status, result, record in processed_messages:
        logger.info(result)

    return processor.response()
