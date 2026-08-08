from aws_lambda_powertools import Logger
from aws_lambda_powertools.middleware_factory import lambda_handler_decorator
from aws_lambda_powertools.utilities.batch import (
    AsyncBatchProcessor,
    EventType,
    async_process_partial_response,
)
from aws_lambda_powertools.utilities.data_classes.sqs_event import SQSRecord
from aws_lambda_powertools.utilities.typing import LambdaContext
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

async def async_record_handler(record: SQSRecord, lambda_context: LambdaContext):
    logger.info(record.json_body)
    logger.info(record.body)
    logger.info(lambda_context)
    logger.info(f"Record: {record}")

    cool_thing: LeMickey = getattr(lambda_context, "cool_thing")
    logger.info({"cool_thing": cool_thing.call_stuff()})

def lambda_handler(event, context: LambdaContext):
    return async_process_partial_response(
        event=event,
        record_handler=async_record_handler,
        processor=processor,
        context=context,
    )
