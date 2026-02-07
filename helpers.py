from aiohttp import ClientSession
import asyncio
from concurrent.futures import Future

from PIL import Image, ImageOps
import io

from onnxruntime import InferenceSession
import numpy as np

from typing import Optional


async def get_embeddings(
        images_array: Optional[np.array], 
        was_processed: list[bool], 
        clip_session: InferenceSession
    ) -> list[Optional[list[Optional[float]]]]:

    print("Getting embeddings...")

    if images_array is None:
        return [None for _ in was_processed]

    def callback(results: np.ndarray, user_data: asyncio.Queue, err: str):
        if err:
            user_data.set_exception(Exception(err))
            return;

        processed_embeddings = results[0]
    
        processed_embeddings_list = processed_embeddings.tolist()
        image_embeddings = []
        ind = 0
        for processed in was_processed:
            if processed:
                image_embeddings.append(processed_embeddings_list[ind])
                ind += 1
            else:
                image_embeddings.append(None)

        user_data.set_result(image_embeddings)

    user_data = Future()
    print("Shape of embedding input:", images_array.shape)

    clip_session.run_async(None, {"images": images_array}, callback, user_data)
    
    image_embeddings = await asyncio.wrap_future(user_data)

    print("Len embeddings after:", len(image_embeddings))

    return image_embeddings

def process_images(
        raw_images: list[Optional[Image.Image]], 
        image_width: int,
        image_height: int,
        transform_mean: np.array,
        transform_std: np.array
    ) -> tuple[list[np.array], list[bool]]:

    print("Processing images...")
    processed_images = []
    was_processed = []
    for image in raw_images:
        if image is None:
            was_processed.append(False)
        else:
            processed_image = preprocess(image, 
                    image_width, 
                    image_height, 
                    transform_mean, 
                    transform_std
                )
            processed_images.append(np.expand_dims(processed_image, axis=0))
            was_processed.append(True)

    print("done processing images")
    return processed_images, was_processed

def preprocess(
        image: Image, 
        image_width: int,
        image_height: int,
        transform_mean: np.array,
        transform_std: np.array
    ) -> np.array:

    processed_image = ImageOps.fit(
            image, 
            (image_width, image_height), 
            method=Image.Resampling.BICUBIC, 
            centering=(0.5, 0.5)
        )
    processed_image = processed_image.convert('RGB')
    processed_array = (processed_image - transform_mean) / transform_std

    processed_array = processed_array.transpose(2, 0, 1)
    return processed_array

async def retrieve_images(
        urls: list[Optional[str]], 
        session: ClientSession
    ) -> list[Optional[Image.Image]]:
    
    print("Retrieving images...")
    
    raw_images = await asyncio.gather(*(get_raw_image(url, session) for url in urls))
    print(f"Got {len(raw_images)} images")

    print("just retrieved images!")
    return raw_images
    
async def get_raw_image(
        url: Optional[str], 
        session: ClientSession
    ) -> Optional[Image.Image]:

    try:
        if url is None:
            return None

        async with session.get(url=url) as response:
            res = await response.read()
            image = Image.open(io.BytesIO(res))
            
            return image
    except Exception as e:
        print(f"Unable to get image url {url} due to {e.__class__}.")
        print(e)
        return None
