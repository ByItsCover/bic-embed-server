from mangum import Mangum

from fastapi import FastAPI, Depends
from contextlib import asynccontextmanager
from fastapi_injectable.util import async_get_injected_obj, get_injected_obj

import asyncio
from aiohttp import ClientSession
from pydantic import BaseModel
from typing import Optional
import os
#import time

import torch
import open_clip
from onnxruntime import InferenceSession

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
    print("loading clip...")
    #await asyncio.sleep(10)
    #time.sleep(10)
    #import open_clip

    print("Pretrained path:", app_state["pretrained_name"])
    # clip_model, _, preprocess = open_clip.create_model_and_transforms(
    #         app_state["model_name"], 
    #         pretrained=app_state["pretrained_name"], 
    #         device=app_state["device"]
    #     )
    clip_session = InferenceSession(app_state["pretrained_name"])
    print("Clip session inputs:")
    print(clip_session.get_inputs())
    preprocess = torch.load(
            app_state["preprocess_path"],
            weights_only=False
        )

    return clip_session, preprocess

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
    clip, raw_images = await asyncio.gather(clip_task, retrieve_images(embed_request.image_urls, app_state["session"]))
    clip_session, preprocess = clip

    processed_images, was_processed = process_images(preprocess, raw_images, app_state["device"])

    #torch = await torch_task
    images_tensor = torch.cat(processed_images, dim=0) if processed_images else None

    image_embeddings = get_embeddings(images_tensor, was_processed, clip_session, torch)

    return {
            "image_embeddings": image_embeddings
        }

handler = Mangum(app)
