from PIL import Image
import io

import asyncio
from aiohttp import ClientSession

import numpy as np

from types import ModuleType
from typing import Optional
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from torch import Tensor, nn
    from torchvision.transforms import Compose
    from onnxruntime import InferenceSession

def get_embeddings(
        images_tensor: Optional["Tensor"], 
        was_processed: list[bool], 
        #clip_model: "nn.Module", 
        clip_session: "InferenceSession",
        torch: ModuleType
    ) -> list[Optional[list[Optional[float]]]]:

    print("Getting embeddings...")
    if images_tensor is not None:
        # with torch.no_grad():
        #     print("Input shape:", images_tensor.shape)
        #     processed_embeddings = clip_model.encode_image(images_tensor).cpu().numpy()
        if hasattr(images_tensor, "detach"):
            input_data = images_tensor.detach().cpu().numpy().astype(np.float32)
        else:
            input_data = np.array(images_tensor, dtype=np.float32)

        input_name = clip_session.get_inputs()[0].name
        outputs = clip_session.run(None, {input_name: input_data})
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
    
    print("Got em, returning embeddings")
    return image_embeddings

def process_images(
        preprocess: "Compose", 
        raw_images: list[Optional[Image]], 
        device: str
    ) -> tuple[list["Tensor"], list[bool]]:

    print("Processing images...")
    processed_images = []
    was_processed = []
    for image in raw_images:
        if image is None:
            was_processed.append(False)
        else:
            processed_images.append(preprocess(image).unsqueeze(0).to(device))
            was_processed.append(True)

    return processed_images, was_processed

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
