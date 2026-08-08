from aws_lambda_powertools.middleware_factory import lambda_handler_decorator
from aws_lambda_powertools.utilities.typing import LambdaContext
from onnxruntime import SessionOptions
import os
from typing import Callable
from utils.loop import ensure_loop
from utils.models import get_model, get_processor
from utils.lance import get_cover_table
from utils.s3 import get_bucket
from config.constants import CLIP_FOLDER


@lambda_handler_decorator
def model_middleware(
        handler: Callable[[dict, LambdaContext], dict],
        event: dict,
        context: LambdaContext,
) -> dict:
    loop = ensure_loop()

    efs_dir = os.environ.get('TOWER_ROOT_DIR', '.')
    item_tower_path = os.path.join(efs_dir, "item_tower.onnx")
    clip_dir = os.path.join(
        os.environ.get('CLIP_ROOT_DIR', '.'),
        CLIP_FOLDER
    )
    clip_path = os.path.join(clip_dir, "clip_vis.onnx")

    session_opts = SessionOptions()
    providers = ["CPUExecutionProvider"]

    item_tower_task = loop.create_task(get_model(item_tower_path, session_opts, providers))
    clip_vis_task = loop.create_task(get_model(clip_path, session_opts, providers))
    processor_task = loop.create_task(get_processor(clip_dir))

    setattr(context, "item_tower_task", item_tower_task)
    setattr(context, "clip_vis_task", clip_vis_task)
    setattr(context, "clip_processor_task", processor_task)

    return handler(event, context)

@lambda_handler_decorator
def lance_middleware(
        handler: Callable[[dict, LambdaContext], dict],
        event: dict,
        context: LambdaContext,
) -> dict:
    loop = ensure_loop()

    cover_table_task = loop.create_task(get_cover_table(os.environ["DB_URI"]))

    setattr(context, "cover_table_task", cover_table_task)

    return handler(event, context)

@lambda_handler_decorator
def s3_middleware(
        handler: Callable[[dict, LambdaContext], dict],
        event: dict,
        context: LambdaContext,
) -> dict:
    loop = ensure_loop()

    cover_dump_task = loop.create_task(get_bucket(os.environ["BUCKET_NAME"]))

    setattr(context, "cover_dump_task", cover_dump_task)

    return handler(event, context)
