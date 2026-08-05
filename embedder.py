import onnxruntime as ort
from onnxruntime import InferenceSession

import lancedb
from lancedb.index import BTree
from lancedb.pydantic import LanceModel, Vector

from pydantic import TypeAdapter, Field
import asyncio
import numpy as np
from numpy.typing import NDArray
from typing import Coroutine, Optional
import os

from models import EmbedRecord


class Cover(LanceModel):
    cover_id: int
    book_id: int
    isbn_13: str
    cover_url: str = Field(alias='image_url')
    cover_embedding: Vector(512) #pyright: ignore[reportInvalidTypeForm]
    tower_embedding: Optional[Vector(256)] = None #pyright: ignore[reportInvalidTypeForm]

class Embedder:
    def __init__(self, model_path: str, db_uri: str):
        self.clip_path = os.path.join(model_path, "clip_quantized.onnx")
        self.clip_session = None
        self.session_opts = ort.SessionOptions()
        self.providers = ["CPUExecutionProvider"]
        self.input_name = "images"

        self.db_uri = db_uri
        self.db = None
        self.table = None
        self.table_name = "covers"
        self.id_field = "cover_id"

        self.covers_adapter = TypeAdapter(list[Cover])
    
    async def embed_records(
            self, 
            records: list[EmbedRecord], 
            clip_task: Coroutine[any, any, None], 
            load_db_task: Coroutine[any, any, None]
        ):

        images_list = [record.image_array for record in records]

        await clip_task
        embeddings = await asyncio.to_thread(
                self.clip_session.run, 
                None, 
                {self.input_name: images_list}
            )

        processed_embeddings: list[NDArray] = embeddings[0].tolist()
        cover_list: list[Cover] = []
        for i in range(len(processed_embeddings)):
            cover = Cover(
                    **records[i].model_dump(), 
                    cover_embedding = processed_embeddings[i] / np.linalg.norm(processed_embeddings[i], axis=-1, keepdims=True)
                )
            cover_list.append(cover)
        
        await load_db_task
        await (
            self.table.merge_insert(self.id_field)
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(self.covers_adapter.dump_python(cover_list))
        )

        print("Table schema:")
        head_res = await self.table.head()
        print(head_res)
    
    async def load_clip(self):
        if self.clip_session is not None:
            return;

        await asyncio.to_thread(self._load_clip_sync)
    
    def _load_clip_sync(self):
        self.clip_session = InferenceSession(
                self.clip_path, 
                self.session_opts,
                providers=self.providers
            )
    
    async def load_db(self):
        if self.db is not None and self.table is not None:
            return;

        self.db = await lancedb.connect_async(self.db_uri)
        self.table = await self.db.create_table(
                self.table_name,
                schema=Cover.to_arrow_schema(),
                exist_ok=True,
            )
        id_stats = await self.table.index_stats(f"{self.id_field}_idx")
        if not id_stats:
            await self.table.create_index(self.id_field, config=BTree(), name=f"{self.id_field}_idx")
