import asyncio
from onnxruntime import InferenceSession, SessionOptions
from transformers import CLIPImageProcessorPil


async def get_model(
        model_path: str, session_options: SessionOptions, providers: list[str]
) -> InferenceSession:
    return await asyncio.to_thread(get_model_sync, model_path, session_options, providers)

def get_model_sync(
        model_path: str, session_options: SessionOptions, providers: list[str]
) -> InferenceSession:
    return InferenceSession(
        model_path,
        session_options,
        providers
    )

async def get_processor(processor_dir: str) -> CLIPImageProcessorPil:
    return await asyncio.to_thread(get_processor_sync, processor_dir)

def get_processor_sync(processor_dir: str) -> CLIPImageProcessorPil:
    return CLIPImageProcessorPil.from_pretrained(processor_dir)
