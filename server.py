from mangum import Mangum

from fastapi import FastAPI, Depends
from contextlib import asynccontextmanager
from fastapi_injectable.util import async_get_injected_obj

import asyncio
from aiohttp import ClientSession
from pydantic import BaseModel
from typing import Optional
import os

from helpers import retrieve_images, process_images, get_embeddings


app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading state...")
    app_state["model_name"] = "ViT-B-32"
    app_state["pretrained_name"] = os.path.join(
            os.environ.get('LAMBDA_TASK_ROOT', '.'),
            "clip_model/open_clip_model.safetensors"
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


async def load_clip():
    print("loading clip...")
    import open_clip

    print("Pretrained path:", app_state["pretrained_name"])
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
            app_state["model_name"], 
            pretrained=app_state["pretrained_name"], 
            device=app_state["device"]
        )

    return clip_model, preprocess

async def load_torch():
    print("loading torch...")
    import torch
    return torch

class EmbedRequest(BaseModel):
    image_urls: list[Optional[str]] = []

@app.post("/predict")
async def predict(embed_request: EmbedRequest):
    clip_task = asyncio.create_task(async_get_injected_obj(load_clip))
    torch_task = asyncio.create_task(async_get_injected_obj(load_torch))

    clip, raw_images = await asyncio.gather(clip_task, retrieve_images(embed_request.image_urls, app_state["session"]))
    clip_model, preprocess = clip

    processed_images, was_processed = process_images(preprocess, raw_images, app_state["device"])

    torch = await torch_task
    images_tensor = torch.cat(processed_images, dim=0) if processed_images else None

    image_embeddings = get_embeddings(images_tensor, was_processed, clip_model, torch)

    return {
            "image_embeddings": image_embeddings
        }

handler = Mangum(app)
