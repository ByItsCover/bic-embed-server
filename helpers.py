from PIL import Image, ImageOps
import io

import asyncio
from aiohttp import ClientSession

import numpy as np

from types import ModuleType
from typing import Optional
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    # from torch import Tensor, nn
    # from torchvision.transforms import Compose
    from onnxruntime import InferenceSession

def get_embeddings(
        images_array: Optional[np.array], 
        was_processed: list[bool], 
        #clip_model: "nn.Module", 
        clip_session: "InferenceSession",
        #torch: ModuleType
    ) -> list[Optional[list[Optional[float]]]]:

    print("Getting embeddings...")

    if images_array is not None:
        # with torch.no_grad():
        #     print("Input shape:", images_tensor.shape)
        #     processed_embeddings = clip_model.encode_image(images_tensor).cpu().numpy()

        # if hasattr(images_tensor, "detach"):
        #     input_data = images_tensor.detach().cpu().numpy().astype(np.float32)
        # else:
        #     input_data = np.array(images_tensor, dtype=np.float32)
        print("Shape of stuff:")
        print(images_array.shape)

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
    
    print("Got em, returning embeddings")
    return image_embeddings

def process_images(
        raw_images: list[Optional[Image]], 
        app_state: dict
    ) -> tuple[list["Tensor"], list[bool]]:

    print("Processing images...")
    processed_images = []
    was_processed = []
    for image in raw_images:
        if image is None:
            was_processed.append(False)
        else:
            #processed_images.append(preprocess(image, app_state).unsqueeze(0).to(app_state["device"]))
            processed_images.append(np.expand_dims(preprocess(image, app_state), axis=0))
            was_processed.append(True)

    return processed_images, was_processed

def preprocess(
    image: Image, 
    app_state: dict
) -> np.array:
    #processed_image = image.resize((app_state["image_width"], app_state["image_height"]), resample=Image.Resampling.BICUBIC)
    processed_image = ImageOps.fit(image, (app_state["image_width"], app_state["image_height"]), method=Image.Resampling.BICUBIC, centering=(0.5, 0.5))
    processed_image = processed_image.convert('RGB')
    processed_array = (processed_image - app_state["transform_mean"]) / app_state["transform_std"]

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
