import asyncio
from aws_lambda_powertools import Logger
from onnxruntime import InferenceSession, SessionOptions
from transformers import CLIPImageProcessorPil

logger = Logger()


async def get_model(
        model_path: str, session_options: SessionOptions, providers: list[str]
) -> InferenceSession:
    model = await asyncio.to_thread(get_model_sync, model_path, session_options, providers)
    logger.info("Model loaded. Just checking if async worked")
    return model

def get_model_sync(
        model_path: str, session_options: SessionOptions, providers: list[str]
) -> InferenceSession:
    return InferenceSession(
        model_path,
        session_options,
        providers
    )

async def get_processor(processor_dir: str) -> CLIPImageProcessorPil:
    processor = await asyncio.to_thread(get_processor_sync, processor_dir)
    logger.info("Processor loaded. Just checking if async worked")
    return processor

def get_processor_sync(processor_dir: str) -> CLIPImageProcessorPil:
    return CLIPImageProcessorPil.from_pretrained(processor_dir)
