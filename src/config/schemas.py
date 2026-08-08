from pydantic import BaseModel, Field, AliasPath, ConfigDict
from typing import Optional
from numpy.typing import NDArray


class EmbedRecord(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    message_id: str = Field(alias='messageId')
    cover_id: int = Field(validation_alias=AliasPath('messageAttributes', 'cover_id', 'stringValue'))
    book_id: int = Field(validation_alias=AliasPath('messageAttributes', 'book_id', 'stringValue'))
    isbn_13: str = Field(validation_alias=AliasPath('messageAttributes', 'isbn_13', 'stringValue'))
    image_url: str = Field(alias='body')
    image_array: Optional[NDArray] = None

class ProcessError(Exception):
    def __init__(self, message_id: str, message: str):
        super().__init__(message)
        self.message_id = message_id
        self.message = message
