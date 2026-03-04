import logging
import time
log = logging.getLogger('api')
log.debug("Within processor, Loading async stuff...")
start = time.time()

import asyncio
from asyncio import Task
from aiohttp import ClientSession

end = time.time()
log.debug(f"Within processor, Loading async stuff took {end - start} seconds")

log.debug("Within processor, Loading other module stuff...")
start = time.time()

from PIL import Image, ImageOps
import io
import numpy as np
from pydantic import TypeAdapter
from typing import Iterator, Coroutine, AsyncGenerator
import os

end = time.time()
log.debug(f"Within processor, Loading other module stuff took {end - start} seconds")

log.debug("Within processor, Loading models models...")
start = time.time()

from models import EmbedRecord, BatchFailures

end = time.time()
log.debug(f"Within processor, Loading models models took {end - start} seconds")

log.debug("Within processor, Loading embedder...")
start = time.time()

from embedder import Embedder

end = time.time()
log.debug(f"Within processor, Loading embedder took {end - start} seconds")


class ProcessError(Exception):
    def __init__(self, message_id: str, message: str):
        super().__init__(message)
        self.message_id = message_id
        self.message = message

class EmbedProcessor:
    def __init__(self, model_path: str, db_uri: str):
        self.records_adapter = TypeAdapter(list[EmbedRecord])
        self.http_session = None
        self.embedder = Embedder(model_path, db_uri)
        self.image_width = 224
        self.image_height = 224
        self.transform_mean = np.array([0.48145466, 0.4578275, 0.40821073])
        self.transform_std = np.array([0.26862954, 0.26130258, 0.27577711])
    
    async def process_images(
            self,
            record_json: list[dict]
        ) -> BatchFailures:

        batch_failures = BatchFailures(batchItemFailures=[])
        print("Record json:")
        print(record_json)
        records = self.records_adapter.validate_python(record_json)

        try:
            http_task = asyncio.create_task(self.load_http_session())
            clip_task = asyncio.create_task(self.embedder.load_clip())
            db_task = asyncio.create_task(self.embedder.load_db())
            
            await http_task
            image_tasks = (asyncio.create_task(self._fetch_raw_image(record)) for record in records)

            process_tasks: list[Task[EmbedRecord]] = []
            async for image_record in self._process_failures(image_tasks, batch_failures.item_failures):
                process_tasks.append(asyncio.create_task(asyncio.to_thread(self._preprocess, image_record)))
            
            processed_records: list[EmbedRecord] = []
            async for processed_record in self._process_failures(process_tasks, batch_failures.item_failures):
                processed_records.append(processed_record)

            if processed_records:
                await self.embedder.embed_records(processed_records, clip_task, db_task)
                print("Finished embedding", len(processed_records), "records")
            else:
                await clip_task
                await db_task
        except Exception as ex:
            print(f"Unable to process images due to {ex.__class__}.")
            print(ex)
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
                print(f"Got an exception: {ex.__class__}.")
                print(ex)
                failure_list.append(ex.message_id)
            else:
                yield record
    
    def _preprocess(
            self,
            image_record: EmbedRecord
        ) -> EmbedRecord:

        if image_record.raw_image is None:
            raise ValueError(f"Image with url {image_record.image_url} was not retrieved")

        processed_image = ImageOps.fit(
                image_record.raw_image, 
                (self.image_width, self.image_height), 
                method=Image.Resampling.BICUBIC, 
                centering=(0.5, 0.5)
            )
        processed_image = processed_image.convert('RGB')
        processed_array = (np.array(processed_image) - self.transform_mean) / self.transform_std

        image_record.image_array = processed_array.transpose(2, 0, 1)
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

model_path = os.path.join(
        os.environ.get('ROOT_DIR', '.'),
        "clip_model/clip_quantized.onnx"
    )
processor = EmbedProcessor(model_path, os.environ["DB_URI"])
