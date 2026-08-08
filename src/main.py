import asyncio
from aws_lambda_powertools import Logger
from aws_lambda_powertools.utilities.batch import (
    AsyncBatchProcessor,
    EventType,
)
from aws_lambda_powertools.utilities.typing import LambdaContext
from middleware import model_middleware, lance_middleware, http_middleware
from preprocess import record_handler
from postprocess import embed_content

processor = AsyncBatchProcessor(event_type=EventType.SQS)
logger = Logger()
try:
    loop = asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)


@model_middleware
@lance_middleware
@http_middleware
def lambda_handler(event, context: LambdaContext):
    records = event["Records"]
    with processor(records, record_handler, context):
        processed_messages = processor.async_process()

    items = []
    for status, result, record in processed_messages:
        items.append(result)
        logger.info(result)

    loop.run_until_complete(embed_content(items, context))
    return processor.response()
