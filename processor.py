import asyncio
from asyncio import Task
from aiohttp import ClientSession

import logging

from transformers import CLIPImageProcessor

from PIL import Image
import io
import numpy as np
from pydantic import TypeAdapter
from typing import Iterator, Coroutine, AsyncGenerator
import os

from models import EmbedRecord, BatchFailures

from embedder import Embedder


class ProcessError(Exception):
    def __init__(self, message_id: str, message: str):
        super().__init__(message)
        self.message_id = message_id
        self.message = message

class EmbedProcessor:
    def __init__(self, model_path: str, db_uri: str):
        self.records_adapter = TypeAdapter(list[EmbedRecord])
        self.http_session = None
        self.model_path = model_path
        self.embedder = Embedder(self.model_path, db_uri)
        self.processor = None
        self.log = logging.getLogger('api')
    
    async def process_images(
            self,
            record_json: list[dict]
        ) -> BatchFailures:

        batch_failures = BatchFailures(batchItemFailures=[])
        self.log.debug("Record json:")
        self.log.debug(record_json)
        records = self.records_adapter.validate_python(record_json)

        try:
            http_task = asyncio.create_task(self.load_http_session())
            processor_task = asyncio.create_task(self.load_processor())
            clip_task = asyncio.create_task(self.embedder.load_clip())
            db_task = asyncio.create_task(self.embedder.load_db())
            
            await http_task
            image_tasks = (asyncio.create_task(self._fetch_raw_image(record)) for record in records)

            await processor_task
            process_tasks: list[Task[EmbedRecord]] = []
            async for image_record in self._process_failures(image_tasks, batch_failures.item_failures):
                process_tasks.append(asyncio.create_task(asyncio.to_thread(self._preprocess, image_record)))
            
            processed_records: list[EmbedRecord] = []
            async for processed_record in self._process_failures(process_tasks, batch_failures.item_failures):
                processed_records.append(processed_record)

            if processed_records:
                await self.embedder.embed_records(processed_records, clip_task, db_task)
                self.log.info(f"Finished embedding {len(processed_records)} records")
            else:
                await clip_task
                await db_task
        except Exception as ex:
            self.log.warning(f"Unable to process images due to {ex.__class__}.")
            self.log.warning(ex)
            batch_failures.item_failures = [record.message_id for record in records]
        finally:
            return batch_failures
    
    async def _process_failures(
            self,
            tasks: Iterator[Coroutine[any, any, EmbedRecord]],
            failure_list: list[str]
        ) -> AsyncGenerator[EmbedRecord]:

        for coroutine in asyncio.as_completed(tasks):
            try:
                record = await coroutine
            except ProcessError as ex:
                self.log.warning(f"Got an exception: {ex.__class__}.")
                self.log.warning(ex)
                failure_list.append(ex.message_id)
            else:
                yield record
    
    def _preprocess(
            self,
            image_record: EmbedRecord
        ) -> EmbedRecord:

        if image_record.raw_image is None:
            raise ValueError(f"Image with url {image_record.image_url} was not retrieved")

        image_record.image_array = self.processor(image_record.raw_image)['pixel_values'][0]
        return image_record
    
    async def _fetch_raw_image(
            self,
            record: EmbedRecord
        ) -> EmbedRecord:

        try:
            async with self.http_session.get(url=record.image_url) as response:
                res = await response.read()
                record.raw_image = Image.open(io.BytesIO(res))
        except Exception as ex:
            raise ProcessError(record.message_id, message=str(ex))
        else:
            return record
    
    async def load_http_session(self):
        if self.http_session is not None:
            return;
    
        self.http_session = ClientSession()
    
    async def load_processor(self):
        if self.processor is not None:
            return;

        await asyncio.to_thread(self._load_processor_sync)
    
    def _load_processor_sync(self):
        self.processor = CLIPImageProcessor.from_pretrained(self.model_path)

model_path = os.path.join(
        os.environ.get('ROOT_DIR', '.'),
        "clip_model"
    )
processor = EmbedProcessor(model_path, os.environ["DB_URI"])
