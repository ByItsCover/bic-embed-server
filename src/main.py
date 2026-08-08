from aws_lambda_powertools import Logger
from aws_lambda_powertools.middleware_factory import lambda_handler_decorator
from aws_lambda_powertools.utilities.batch import (
    AsyncBatchProcessor,
    EventType,
    async_process_partial_response,
)
from aws_lambda_powertools.utilities.data_classes.sqs_event import SQSRecord
from aws_lambda_powertools.utilities.typing import LambdaContext
import os
import time
from typing import Callable

class LeMickey:
    def __init__(self):
        self.thing = 2

    def call_stuff(self):
        return self.thing


processor = AsyncBatchProcessor(event_type=EventType.SQS)
logger = Logger()


@lambda_handler_decorator
def middleware_before(
        handler: Callable[[dict, LambdaContext], dict],
        event: dict,
        context: LambdaContext,
) -> dict:
    logger.info({"event:": event})
    logger.info({"context:": context})
    setattr(context, "cool_thing", LeMickey())

    return handler(event, context)


@lambda_handler_decorator
def middleware_after(
        handler: Callable[[dict, LambdaContext], dict],
        event: dict,
        context: LambdaContext,
) -> dict:
    start_time = time.time()
    response = handler(event, context)
    execution_time = time.time() - start_time

    # adding custom headers in response object after lambda executing
    logger.info({"full_response": response})
    logger.info({"execution_time": execution_time})


    return response

async def async_record_handler(record: SQSRecord, lambda_context: LambdaContext):
    logger.info(lambda_context)
    logger.info(f"Record: {record}")
    logger.info(record)
    logger.info(record.body)

    efs_dir = os.environ.get('MODEL_ROOT_DIR', '.')
    print("EFS dir:", efs_dir)
    files = os.listdir(efs_dir)
    logger.info(files)

    cool_thing: LeMickey = getattr(lambda_context, "cool_thing")
    logger.info({"cool_thing": cool_thing.call_stuff()})
    return 8

@middleware_before
@middleware_after
def lambda_handler(event, context: LambdaContext):
    records = event["Records"]
    with processor(records, async_record_handler, context):
        processed_messages = processor.async_process()

    for message in processed_messages:
        logger.info(message)

    return processor.response()
