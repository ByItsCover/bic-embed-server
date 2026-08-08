import asyncio
from aws_lambda_powertools.utilities.data_classes.sqs_event import SQSRecord
from aws_lambda_powertools.utilities.typing import LambdaContext
from aws_lambda_powertools import Logger
from config.schemas import ImageRecord, ProcessError

from aiohttp import ClientSession
from PIL import Image
from PIL.ImageFile import ImageFile
import io

logger = Logger()


async def fetch_raw_image(record: ImageRecord, http_session: ClientSession) -> ImageFile:
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

    image_record = ImageRecord.model_validate(record)
    logger.info({"mapped_image_record": image_record})

    http_session: ClientSession = getattr(lambda_context, "http_session")
    raw_image_task = asyncio.create_task(fetch_raw_image(image_record, http_session))
    logger.info({"raw_image": await raw_image_task})

    # efs_dir = os.environ.get('MODEL_ROOT_DIR', '.')
    # print("EFS dir:", efs_dir)
    # files = os.listdir(efs_dir)
    # logger.info(files)
    #
    # cool_thing: LeMickey = getattr(lambda_context, "cool_thing")
    # logger.info({"cool_thing": cool_thing.call_stuff()})
    return 13
