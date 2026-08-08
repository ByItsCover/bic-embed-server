import os
import sys

from huggingface_hub import hf_hub_download

import torch
import open_clip


def hf_download(destination: str):

    repo_id = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
    filenames = ["open_clip_model.safetensors", "preprocessor_config.json"]

    os.makedirs(destination, exist_ok=True)

    for name in filenames:

        hf_hub_download(
            repo_id=repo_id,
            filename=name,
            local_dir=destination
        )

    print(f"Model {repo_id} downloaded to {destination}/")

def quantized_download(
        destination: str, 
        clean_cache: bool = True
    ):

    filename = "open_clip_model.safetensors"

    script_state = {
        "model_name": "ViT-B-32",
        "pretrained_name": os.path.join(
            destination,
            filename
        ),
        "onnx_model_path": os.path.join(
            destination,
            "clip_vis.onnx"
        ),
        "onnx_model_shapes_path": os.path.join(
            destination,
            "clip_vis_shapes.onnx"
        ),
        "quant_pre_model_path": os.path.join(
            destination,
            "clip_vis_pre_quantized.onnx"
        ),
        "quant_model_path": os.path.join(
            destination,
            "clip_vis_quantized.onnx"
        ),
        "preprocess_path": os.path.join(
            destination,
            "preprocess.onnx"
        ),
        "device": "cpu"
    }

    os.makedirs(destination, exist_ok=True)


    print("Loading Clip...")

    clip_model, _, _ = open_clip.create_model_and_transforms(
        script_state["model_name"],
        pretrained=script_state["pretrained_name"],
        device=script_state["device"]
    )
    clip_model.visual.eval()

    print(clip_model)


    print("Exporting model to onnx format...")

    input_tensor = torch.ones((2, 3, 224, 224), dtype=torch.float32)

    torch.onnx.export(clip_model.visual,
                  (input_tensor),
                  script_state["onnx_model_path"],
                  input_names = ['images'],
                  output_names = ['embeddings'],
                  dynamic_shapes=({0: torch.export.Dim.DYNAMIC},),
                  external_data=False
                  )


    if clean_cache:
        print("Cleaning up...")
        os.remove(script_state["pretrained_name"])

    if os.path.isfile(script_state["onnx_model_path"] + '.data'):
        os.remove(script_state["onnx_model_path"] + '.data')


    print(f"Model {script_state["pretrained_name"]} onnx exported to {destination}/")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Destination path is missing")

    destination = os.path.join(sys.argv[1], "clip_model")
    hf_download(destination)
    quantized_download(destination)
