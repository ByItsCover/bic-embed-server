import asyncio
import onnxruntime as ort
from onnxruntime import InferenceSession

import lancedb
from pydantic import TypeAdapter

import numpy as np
from numpy.typing import NDArray
from typing import Coroutine

from models import EmbedRecord, Cover

class Embedder:
    def __init__(self, model_path: str, db_uri: str):
        self.model_path = model_path
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

        images_list = [np.expand_dims(record.image_array, axis=0) for record in records]
        images_array = np.concatenate(images_list, axis=0, dtype=np.float32)

        await clip_task
        embeddings = await asyncio.to_thread(
                self.clip_session.run, 
                None, 
                {self.input_name: images_array}
            )

        processed_embeddings: list[NDArray] = embeddings[0].tolist()
        cover_list: list[Cover] = []
        for i in range(len(processed_embeddings)):
            cover = Cover(
                    **records[i].model_dump(), 
                    embedding=processed_embeddings[i]
                )
            cover_list.append(cover)
        
        await load_db_task
        await (
            self.table.merge_insert(self.id_field)
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(self.covers_adapter.dump_python(cover_list))
        )
    
    async def load_clip(self):
        if self.clip_session is not None:
            return;

        await asyncio.to_thread(self._load_clip_sync)
    
    def _load_clip_sync(self):
        self.clip_session = InferenceSession(
                self.model_path, 
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
                mode="overwrite"
            )
        id_stats = await self.table.index_stats(self.id_field)
        if not id_stats:
            await self.table.create_index(self.id_field)
