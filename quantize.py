import torch
import open_clip

from onnxruntime.quantization import quantize_dynamic, QuantType

from huggingface_hub import hf_hub_download
import os
import sys

def main(source: str):
    repo_id = "laion/CLIP-ViT-B-32-256x256-DataComp-s34B-b86K"
    filename = "open_clip_model.safetensors"

    app_state = {}
    app_state["model_name"] = "ViT-B-32"
    app_state["pretrained_name"] = os.path.join(
            source,
            filename
        )
    app_state["onnx_model"] = os.path.join(
            source,
            "clip.onnx"
        )
    app_state["quant_model"] = os.path.join(
            source,
            "clip_quantized.onnx"
        )
    app_state["preprocess_path"] = os.path.join(
            source,
            "preprocess.pt"
        )
    #app_state["pretrained_name"] = "laion2b_s34b_b79k"
    app_state["device"] = "cpu"

    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        app_state["model_name"],
        pretrained=app_state["pretrained_name"],
        device=app_state["device"]
    )

    print("Model before:")
    print(clip_model)

    torch.onnx.export(clip_model.visual,                        
                  (torch.ones(1, 3, 224, 224),),       
                  app_state["onnx_model"],                     
                  #export_params=True,           
                  #opset_version=11,            
                  #do_constant_folding=True,    
                  input_names = ['input'],   
                  output_names = ['output'])

    print("Exported onnx model")

    torch.save(preprocess, app_state["preprocess_path"])

    print("Exported preprocess")

    
    quantized_model = quantize_dynamic(app_state["onnx_model"],
                                   app_state["quant_model"],
                                   weight_type=QuantType.QUInt8)

    # print("Preprocess:")
    # print(preprocess)

    print("Model quant:")
    print(quantized_model)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Model path is missing")

    source = os.path.join(sys.argv[1], "clip_model")
    main(source)
