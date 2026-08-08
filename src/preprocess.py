from aws_lambda_powertools.utilities.data_classes.sqs_event import SQSRecord
from aws_lambda_powertools.utilities.typing import LambdaContext
from aws_lambda_powertools import Logger
from pydantic import BaseModel, Field, AliasPath, ConfigDict

logger = Logger()


class ImageRecord(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    message_id: str
    cover_id: int = Field(validation_alias=AliasPath('message_attributes', 'cover_id', 'string_value'))
    book_id: int = Field(validation_alias=AliasPath('message_attributes', 'book_id', 'string_value'))
    isbn_13: str = Field(validation_alias=AliasPath('message_attributes', 'isbn_13', 'string_value'))
    image_url: str = Field(alias='body')


async def record_handler(record: SQSRecord, lambda_context: LambdaContext):
    logger.info(lambda_context)
    logger.info(f"Record: {record}")
    logger.info(record)
    logger.info(record.body)

    image_record = ImageRecord.model_validate(record)
    logger.info({"mapped_image_record": image_record})

    # efs_dir = os.environ.get('MODEL_ROOT_DIR', '.')
    # print("EFS dir:", efs_dir)
    # files = os.listdir(efs_dir)
    # logger.info(files)
    #
    # cool_thing: LeMickey = getattr(lambda_context, "cool_thing")
    # logger.info({"cool_thing": cool_thing.call_stuff()})
    return 13
