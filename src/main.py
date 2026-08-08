from aws_lambda_powertools import Logger
from aws_lambda_powertools.utilities.batch import (
    AsyncBatchProcessor,
    EventType,
)
from aws_lambda_powertools.utilities.typing import LambdaContext
from numpy.typing import NDArray
from middleware import model_middleware, lance_middleware, http_middleware
from preprocess import record_handler
from postprocess import embed_content
from utils.loop import ensure_loop

processor = AsyncBatchProcessor(event_type=EventType.SQS)
logger = Logger()


@model_middleware
@lance_middleware
@http_middleware
def lambda_handler(event, context: LambdaContext):
    loop = ensure_loop()
    records = event["Records"]
    with processor(records, record_handler, context):
        processed_messages = processor.async_process()

    items: list[NDArray] = []
    for status, result, record in processed_messages:
        logger.info(result)
        items.append(result)

    loop.run_until_complete(embed_content(items, context))

    # noinspection PyTypeChecker
    loop.call_soon_threadsafe(loop.stop)
    return processor.response()
