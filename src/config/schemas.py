from pydantic import BaseModel, Field, AliasPath, ConfigDict


class ImageRecord(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    message_id: str
    cover_id: int = Field(validation_alias=AliasPath('message_attributes', 'cover_id', 'string_value'))
    book_id: int = Field(validation_alias=AliasPath('message_attributes', 'book_id', 'string_value'))
    isbn_13: str = Field(validation_alias=AliasPath('message_attributes', 'isbn_13', 'string_value'))
    image_url: str = Field(alias='body')

class ProcessError(Exception):
    def __init__(self, message_id: str, message: str):
        super().__init__(message)
        self.message_id = message_id
        self.message = message
