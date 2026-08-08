from aws_lambda_powertools import Logger
import lancedb
from lancedb import AsyncTable
from lancedb.index import BTree
from lancedb.pydantic import LanceModel, Vector
from config.constants import COVER_TABLE_NAME, TOWER_DIM, CLIP_DIM
from typing import Optional

logger = Logger()


class Cover(LanceModel):
    cover_id: int
    book_id: int
    isbn_13: str
    cover_url: str
    cover_embedding: Vector(CLIP_DIM)  # type: ignore[PyTypeChecker] # pyright: ignore[reportInvalidTypeForm]
    tower_embedding: Optional[Vector(TOWER_DIM)] = None  # type: ignore[PyTypeChecker] # pyright: ignore[reportInvalidTypeForm, reportInvalidTypeArguments]


async def get_cover_table(db_uri: str) -> AsyncTable:
    logger.info("Beginning to load lance db")
    db = await lancedb.connect_async(db_uri)
    logger.info("DB has loaded. Beginning to Cover Table")
    cover_table = await db.create_table(
        COVER_TABLE_NAME,
        schema=Cover.to_arrow_schema(),
        exist_ok=True,
    )

    logger.info("Cover Table has loaded. Beginning Index stuff")
    id_stats = await cover_table.index_stats("cover_id_idx")
    if not id_stats:
        await cover_table.create_index("cover_id", config=BTree(), name="cover_id_idx")

    logger.info("Cover table loaded completely")
    return cover_table
