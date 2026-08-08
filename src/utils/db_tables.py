from aws_lambda_powertools import Logger
import lancedb
from lancedb import AsyncTable
from lancedb.index import BTree
from config.schemas import Cover
from config.constants import COVER_TABLE_NAME

logger = Logger()


async def get_cover_table(db_uri: str) -> AsyncTable:
    db = await lancedb.connect_async(db_uri)
    cover_table = await db.create_table(
        COVER_TABLE_NAME,
        schema=Cover.to_arrow_schema(),
        exist_ok=True,
    )

    id_stats = await cover_table.index_stats("cover_id_idx")
    if not id_stats:
        await cover_table.create_index("cover_id", config=BTree(), name="cover_id_idx")

    logger.info("Cover table load complete")
    return cover_table
