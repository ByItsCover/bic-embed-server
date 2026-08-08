from lancedb.pydantic import LanceModel, Vector
from pydantic import BaseModel, Field, AliasPath, ConfigDict
from numpy.typing import NDArray
from typing import Optional
from config.constants import TOWER_DIM, CLIP_DIM


class S3Record(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    bucket_name: str = Field(validation_alias=AliasPath('s3', 'bucket', 'name'))
    key: str = Field(validation_alias=AliasPath('s3', 'object', 'key'))

class EmbedRecord(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    message_id: str = Field(alias='messageId')
    cover_id: int = Field(alias='x-amz-meta-cover_id')
    book_id: int = Field(alias='x-amz-meta-book_id')
    isbn_13: str = Field(alias='x-amz-meta-isbn_13')
    cover_url: str = Field(alias='x-amz-meta-image_url')
    image_array: Optional[NDArray] = None

class Cover(LanceModel):
    cover_id: int
    book_id: int
    isbn_13: str
    cover_url: str
    cover_embedding: Vector(CLIP_DIM)  # type: ignore[PyTypeChecker]
    tower_embedding: Optional[Vector(TOWER_DIM)] = None  # type: ignore[PyTypeChecker]

class ProcessError(Exception):
    def __init__(self, message_id: str, message: str):
        super().__init__(message)
        self.message_id = message_id
        self.message = message
