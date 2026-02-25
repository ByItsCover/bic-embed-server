from pydantic import BaseModel, Field, AliasPath, ConfigDict
from typing import Optional
from numpy.typing import NDArray
from PIL import Image
from lancedb.pydantic import LanceModel, Vector

class EmbedRecord(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    message_id: str = Field(alias='messageId')
    cover_id: str
    isbn_13: str
    image_url: str = Field(validation_alias=AliasPath('body', 'image_url'))
    raw_image: Optional[Image.Image] = None
    image_array: Optional[NDArray] = None

class BatchFailures(BaseModel):
    item_failures: list[str] = Field(alias='batchItemFailures')

class Cover(LanceModel):
    cover_id: int
    isbn_13: str
    cover_url: str = Field(alias='image_url')
    embedding: Vector(512) #pyright: ignore[reportInvalidTypeForm]
