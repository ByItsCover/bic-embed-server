from mangum import Mangum

from fastapi import FastAPI#, Depends
from contextlib import asynccontextmanager

from pydantic import BaseModel
from typing import Optional

import numpy as np


app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    from aiohttp import ClientSession
    import os

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
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    clip_session = ort.InferenceSession(app_state["pretrained_name"], opts, providers=["CPUExecutionProvider"])
    print("Loaded all of CLIP")

    return clip_session

class EmbedRequest(BaseModel):
    image_urls: list[Optional[str]] = []

@app.post("/predict")
async def predict(embed_request: EmbedRequest):
    import asyncio
    from fastapi_injectable.util import get_injected_obj
    from helpers import retrieve_images, process_images, get_embeddings

    clip_task = asyncio.to_thread(get_injected_obj, load_clip)
    clip_session, raw_images = await asyncio.gather(
            clip_task, 
            retrieve_images(embed_request.image_urls, app_state["session"])
        )

    processed_images, was_processed = process_images(
            raw_images, 
            app_state["image_width"], 
            app_state["image_height"], 
            app_state["transform_mean"], 
            app_state["transform_std"]
        )
    images_array = np.concatenate(processed_images, axis=0) if processed_images else None
    image_embeddings = get_embeddings(images_array, was_processed, clip_session)

    return {
            "image_embeddings": image_embeddings
        }

handler = Mangum(app)
