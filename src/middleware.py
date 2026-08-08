from aws_lambda_powertools.middleware_factory import lambda_handler_decorator
from aws_lambda_powertools.utilities.typing import LambdaContext
from aws_lambda_powertools import Logger
import lancedb
import onnxruntime as ort
from transformers import CLIPImageProcessorPil
from aiohttp import ClientSession
import os
from typing import Callable
from utils.db_tables import get_cover_table
from config.constants import CLIP_FOLDER

logger = Logger()


@lambda_handler_decorator
def model_middleware(
        handler: Callable[[dict, LambdaContext], dict],
        event: dict,
        context: LambdaContext,
) -> dict:
    efs_dir = os.environ.get('TOWER_ROOT_DIR', '.')
    item_tower_path = os.path.join(efs_dir, "item_tower.onnx")
    clip_dir = os.path.join(
        os.environ.get('CLIP_ROOT_DIR', '.'),
        CLIP_FOLDER
    )
    clip_path = os.path.join(clip_dir, "clip_vis.onnx")

    session_opts = ort.SessionOptions()
    providers = ["CPUExecutionProvider"]

    item_tower = ort.InferenceSession(
        item_tower_path,
        session_opts,
        providers
    )
    clip_vis = ort.InferenceSession(
        clip_path,
        session_opts,
        providers
    )
    processor = CLIPImageProcessorPil.from_pretrained(clip_path)

    setattr(context, "item_tower_session", item_tower)
    setattr(context, "clip_vis_session", clip_vis)
    setattr(context, "clip_processor", processor)

    return handler(event, context)

@lambda_handler_decorator
def lance_middleware(
        handler: Callable[[dict, LambdaContext], dict],
        event: dict,
        context: LambdaContext,
) -> dict:
    db = lancedb.connect(os.environ["DB_URI"])
    cover_table = get_cover_table(db)

    setattr(context, "cover_table", cover_table)

    return handler(event, context)

@lambda_handler_decorator
def http_middleware(
        handler: Callable[[dict, LambdaContext], dict],
        event: dict,
        context: LambdaContext,
) -> dict:
    http_session = ClientSession()

    setattr(context, "http_session", http_session)

    return handler(event, context)
