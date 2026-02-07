from mangum import Mangum

from fastapi import FastAPI, Depends
from contextlib import asynccontextmanager

from pydantic import BaseModel
from typing import Optional, Annotated

import numpy as np
from aiohttp import ClientSession

import os

import onnxruntime as ort
from onnxruntime import InferenceSession
import asyncio
from fastapi_injectable.util import get_injected_obj, async_get_injected_obj, injectable

from helpers import retrieve_images, process_images, get_embeddings

import time


app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    source_path = os.environ.get('LAMBDA_TASK_ROOT', '.')
    app_state["model_name"] = "ViT-B-32"
    app_state["pretrained_name"] = os.path.join(
            source_path,
            "clip_model/clip_quantized.onnx"
        )
    app_state["image_width"] = 224
    app_state["image_height"] = 224
    app_state["transform_mean"] = np.array([0.48145466, 0.4578275, 0.40821073])
    app_state["transform_std"] = np.array([0.26862954, 0.26130258, 0.27577711])
    app_state["session"] = ClientSession()
    print("Loaded state")
    yield
    await app_state["session"].close()
    app_state.clear()

app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    print(app_state)
    return {"message": "Hello World"}


def load_clip():
    print("just started clip load actually")
    # for i in range(10, 0, -1):
    #     print("time sleeping for", i)
    #     time.sleep(1)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 2
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    clip_session = ort.InferenceSession(app_state["pretrained_name"], opts, providers=["CPUExecutionProvider"])
    print("just loaded all of CLIP!")

    return clip_session

@injectable
async def get_clip_task():
    return asyncio.create_task(asyncio.to_thread(load_clip))

class EmbedRequest(BaseModel):
    image_urls: list[Optional[str]] = []

@app.post("/predict")
async def predict(embed_request: EmbedRequest, clip_task: Annotated[asyncio.Task, Depends(get_clip_task)]):
    raw_images = await retrieve_images(embed_request.image_urls, app_state["session"])
    processed_images, was_processed = await asyncio.to_thread(process_images,
            raw_images, 
            app_state["image_width"], 
            app_state["image_height"], 
            app_state["transform_mean"], 
            app_state["transform_std"]
        )
    images_array = np.concatenate(processed_images, axis=0) if processed_images else None

    clip_session = await clip_task
    image_embeddings = await get_embeddings(images_array, was_processed, clip_session)

    return {
            "image_embeddings": image_embeddings
        }

stage = os.environ.get("ENVIRONMENT", "")
handler = Mangum(app, api_gateway_base_path=f"/{stage}")
