from PIL import Image, ImageOps
import io

import asyncio
from aiohttp import ClientSession

import numpy as np

from types import ModuleType
from typing import Optional
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    print("Importing onnxruntime (should not happen in lambda)")
    from onnxruntime import InferenceSession

def get_embeddings(
        images_array: Optional[np.array], 
        was_processed: list[bool], 
        clip_session: "InferenceSession"
    ) -> list[Optional[list[Optional[float]]]]:

    print("Getting embeddings...")

    if images_array is not None:
        print("Shape of embedding input:", images_array.shape)

        input_name = clip_session.get_inputs()[0].name
        outputs = clip_session.run(None, {input_name: images_array.astype(np.float32)})
        processed_embeddings = outputs[0]
        
        processed_embeddings_list = processed_embeddings.tolist()
        image_embeddings = []
        ind = 0
        for processed in was_processed:
            if processed:
                image_embeddings.append(processed_embeddings_list[ind])
                ind += 1
            else:
                image_embeddings.append(None)
    else:
        image_embeddings = [None for _ in was_processed]

    return image_embeddings

def process_images(
        raw_images: list[Optional[Image]], 
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

    return processed_images, was_processed

def preprocess(
        image: Image, 
        image_width: int,
        image_height: int,
        transform_mean: np.array,
        transform_std: np.array
    ) -> np.array:

    processed_image = ImageOps.fit(image, (image_width, image_height), method=Image.Resampling.BICUBIC, centering=(0.5, 0.5))
    processed_image = processed_image.convert('RGB')
    processed_array = (processed_image - transform_mean) / transform_std

    # 4. Transpose to (Channels, Height, Width) for the model
    processed_array = processed_array.transpose(2, 0, 1)
    return processed_array

async def retrieve_images(
        urls: list[Optional[str]], 
        session: ClientSession
    ) -> list[Optional[Image]]:
    
    print("Retrieving images...")
    raw_images = await asyncio.gather(*(get_raw_image(url, session) for url in urls))
    print(f"Got {len(raw_images)} images")

    return raw_images
    
async def get_raw_image(
        url: Optional[str], 
        session: ClientSession
    ) -> Optional[Image]:

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
