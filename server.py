from mangum import Mangum

from fastapi import FastAPI#, Depends
from contextlib import asynccontextmanager
from fastapi_injectable.util import get_injected_obj

import asyncio
from aiohttp import ClientSession
from pydantic import BaseModel
from typing import Optional
import os
#import time

import numpy as np

#import torch
#import open_clip

from helpers import retrieve_images, process_images, get_embeddings


app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading state...")
    source_path = os.environ.get('LAMBDA_TASK_ROOT', '.')
    app_state["model_name"] = "ViT-B-32"
    app_state["pretrained_name"] = os.path.join(
            source_path,
            "clip_model/clip_quantized.onnx"
        )
    app_state["preprocess_path"] = os.path.join(
            source_path,
            "clip_model/preprocess.pt"
        )
    app_state["image_width"] = 224
    app_state["image_height"] = 224
    app_state["transform_mean"] = np.array([0.48145466, 0.4578275, 0.40821073])
    app_state["transform_std"] = np.array([0.26862954, 0.26130258, 0.27577711])
    #app_state["pretrained_name"] = "laion2b_s34b_b79k"
    app_state["device"] = "cpu"
    app_state["session"] = ClientSession()
    yield
    await app_state["session"].close()
    app_state.clear()

app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    print(app_state)
    return {"message": "Hello World"}


def load_clip():
    print("loading clip (and torch)...")
    #await asyncio.sleep(10)
    #time.sleep(10)
    #import open_clip
    #import torch
    from onnxruntime import InferenceSession

    print("Pretrained path:", app_state["pretrained_name"])
    # clip_model, _, preprocess = open_clip.create_model_and_transforms(
    #         app_state["model_name"], 
    #         pretrained=app_state["pretrained_name"], 
    #         device=app_state["device"]
    #     )
    clip_session = InferenceSession(app_state["pretrained_name"])
    print("Clip session inputs:")
    print(clip_session.get_inputs())
    # preprocess = torch.load(
    #         app_state["preprocess_path"],
    #         weights_only=False
    #     )

    print("Loaded all of CLIP")

    return clip_session

# def load_torch():
#     #time.sleep(10)
#     print("loading torch...")
#     import torch
#     return torch

class EmbedRequest(BaseModel):
    image_urls: list[Optional[str]] = []

@app.post("/predict")
async def predict(embed_request: EmbedRequest):
    clip_task = asyncio.to_thread(get_injected_obj, load_clip)
    #torch_task = asyncio.to_thread(get_injected_obj, load_torch)

    #clip, torch, raw_images = await asyncio.gather(clip_task, torch_task, retrieve_images(embed_request.image_urls, app_state["session"]))
    clip_session, raw_images = await asyncio.gather(clip_task, retrieve_images(embed_request.image_urls, app_state["session"]))
    #clip_session, preprocess = clip

    processed_images, was_processed = process_images(raw_images, app_state)

    #torch = await torch_task
    images_array = np.concatenate(processed_images, axis=0) if processed_images else None

    image_embeddings = get_embeddings(images_array, was_processed, clip_session)

    return {
            "image_embeddings": image_embeddings
        }

handler = Mangum(app)
