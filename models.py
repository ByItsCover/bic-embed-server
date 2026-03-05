from pydantic import BaseModel, Field, AliasPath, ConfigDict
from typing import Optional
from numpy.typing import NDArray
from PIL import Image


class EmbedRecord(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    message_id: str = Field(alias='messageId')
    cover_id: int = Field(validation_alias=AliasPath('messageAttributes', 'cover_id', 'stringValue'))
    isbn_13: str = Field(validation_alias=AliasPath('messageAttributes', 'isbn_13', 'stringValue'))
    image_url: str = Field(alias='body')
    raw_image: Optional[Image.Image] = None
    image_array: Optional[NDArray] = None

class BatchFailures(BaseModel):
    item_failures: list[str] = Field(alias='batchItemFailures')
