from aws_lambda_powertools import Logger
import aioboto3
from types_aiobotocore_s3.service_resource import Bucket

logger = Logger()


async def get_bucket(bucket_name: str) -> Bucket:
    session = aioboto3.Session()
    async with session.resource("s3") as s3:
        bucket = await s3.Bucket(bucket_name)

    logger.info(f"{bucket_name} bucket load complete")
    return bucket
