from PIL import Image
import io

import asyncio
import aiohttp
from aiohttp import ClientSession

from litserve import Request
import litserve as ls

from typing import Optional
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from torch import Tensor


class EmbedServer(ls.LitAPI):
    def setup(self, device):
        print("Setting up")
        self.model_name = "ViT-B-32"
        self.pretrained_path = "clip_model/open_clip_model.safetensors"
        self.device = device
        print("Device used:", self.device)
        self.clip_model = None
        self._torch = None
    
    async def decode_request(self, request: Request):
        load_task = asyncio.to_thread(self._load_model)

        image_urls: list[str | None] = request["image_urls"]
        _, raw_images = await asyncio.gather(load_task, self._retrieve_images(image_urls))

        processed_images = []
        was_processed = []
        for image in raw_images:
            if image is None:
                was_processed.append(False)
            else:
                processed_images.append(self.preprocess(image).unsqueeze(0).to(self.device))
                was_processed.append(True)

        images_tensor = self._torch.cat(processed_images, dim=0) if processed_images else None
        
        return (images_tensor, was_processed)

    def _load_model(self):
        if self.clip_model is None:
            print("Loading model the first time")

            import torch
            import open_clip

            self._torch = torch

            self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.model_name, 
                pretrained=self.pretrained_path, 
                device=self.device
            )
    
    async def _retrieve_images(self, urls: list[str | None]):
        async with aiohttp.ClientSession() as session:
            raw_images = await asyncio.gather(*(self._get_raw_image(url, session) for url in urls))
        
        print(f"Got {len(raw_images)} images")

        return raw_images
    
    async def _get_raw_image(self, url: str | None, session: ClientSession):
        try:
            if url is None:
                return None
            async with session.get(url=url) as response:
                res = await response.read()
                image = Image.open(io.BytesIO(res))
                
                return image
        except Exception as e:
            print(f"Unable to get image url {url} due to {e.__class__}.")
            return None
    
    async def predict(self, processed_images: tuple[Optional["Tensor"], list[bool]]):
        images_tensor, was_processed = processed_images
        if images_tensor is not None:
            with self._torch.no_grad():
                processed_embeddings = self.clip_model.encode_image(images_tensor).cpu().numpy()
            
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

    async def encode_response(self, image_embeddings: list):
        return {
                "image_embeddings": image_embeddings
            }

server = ls.LitServer(EmbedServer(enable_async=True), devices=1, workers_per_device=1)

if __name__ == "__main__":
    print("About to run")
    server.run(port=8000, generate_client_file=False)
