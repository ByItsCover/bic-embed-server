import os
import sys

def hf_download(destination: str):
    from huggingface_hub import hf_hub_download

    repo_id = "laion/CLIP-ViT-B-32-256x256-DataComp-s34B-b86K"
    filename = "open_clip_model.safetensors"

    os.makedirs(destination, exist_ok=True)

    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=destination
    )

    print(f"Model {repo_id} downloaded to {destination}/")

def quantized_download(destination: str, clean_cache: bool = True):
    import torch
    import open_clip

    from onnxruntime.quantization import quantize_dynamic, QuantType
    from onnxruntime.quantization.shape_inference import quant_pre_process

    filename = "open_clip_model.safetensors"

    script_state = {
        "model_name": "ViT-B-32",
        "pretrained_name": os.path.join(
            destination,
            filename
        ),
        "onnx_model_path": os.path.join(
            destination,
            "clip.onnx"
        ),
        "quant_model_path": os.path.join(
            destination,
            "clip_quantized.onnx"
        ),
        "preprocess_path": os.path.join(
            destination,
            "preprocess.onnx"
        ),
        "device": "cpu",
        # "interpolation_mode": T.InterpolationMode.BICUBIC,
        # "transform_mean": OPENAI_DATASET_MEAN,
        # "transform_std": OPENAI_DATASET_STD,
    }

    os.makedirs(destination, exist_ok=True)


    print("Loading Clip...")

    clip_model, _, _ = open_clip.create_model_and_transforms(
        script_state["model_name"],
        pretrained=script_state["pretrained_name"],
        device=script_state["device"]
    )

    print(clip_model)


    # print("Exporting Preprocess...")

    # torch.save(preprocess, script_state["preprocess_path"])

    #print(preprocess)


    print("Exporting model to onnx format...")

    torch.onnx.export(clip_model.visual,
                  (torch.ones(1, 3, 224, 224),),
                  script_state["onnx_model_path"],
                  input_names = ['input'],
                  output_names = ['output'],
                  #dynamic_shapes = {'x': {0: torch.export.Dim("batch", min=1, max=1024)}}
                  dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                },
                dynamo=False
                  )

    # from onnxsim import simplify

    # model_simp, check = simplify(onnx_model.model)
    # if check:
    #     onnx.save(model_simp, script_state["onnx_model_path"])

    # Iterate through all nodes to find Concat nodes
    #print("doin something")
    #print(onnx_model.model.graph.node)

    # print(onnx_model.model.graph.node.len())
    # concat_nodes = []
    # for node in onnx_model.model.graph.node():
    #     if node.op_type == 'Concat':
    #         concat_nodes.append(node)

    # print(concat_nodes)

    # print("Exporting Preprocess to ONNX...")

    # class PreprocessModule(torch.nn.Module):
    #     def __init__(self):
    #         super().__init__()
    #         self.transform = T.Compose([
    #             T.Resize((224, 224), interpolation=script_state["interpolation_mode"]),
    #             T.CenterCrop((224, 224)),
    #             #MaybeToTensor(),
    #             #T.Normalize(mean=script_state["transform_mean"], std=script_state["transform_std"])
    #         ])

    #     def forward(self, x: torch.Tensor):
    #         x = self.transform(x)
    #         return (x - script_state["transform_mean"]) / script_state["transform_std"]


    # torch.onnx.export(PreprocessModule(),
    #               (torch.ones(1, 3, 224, 224),),
    #               script_state["preprocess_path"],
    #               input_names = ['input'],
    #               output_names = ['output'],
    #               dynamic_shapes = {
    #                     'x': {
    #                         0: 'batch_size',
    #                         2: 'height',
    #                         3: 'width'
    #                     },
    #                     # 'output': {
    #                     #     0: 'batch_size'
    #                     # }
    #                 })


    print("Quantizing model...")

    quant_pre_process(script_state["onnx_model_path"],
        script_state["quant_model_path"],
        skip_optimization=False,
        skip_symbolic_shape=True,
        verbose=3)

    quantize_dynamic(script_state["onnx_model_path"],
                                   script_state["quant_model_path"],
                                   weight_type=QuantType.QUInt8)


    if clean_cache:
        print("Cleaning up...")
        os.remove(script_state["pretrained_name"])
    
    os.remove(script_state["onnx_model_path"])
    if os.path.isfile(script_state["onnx_model_path"] + '.data'):
        os.remove(script_state["onnx_model_path"] + '.data')


    print(f"Model {script_state["pretrained_name"]} quantized to {destination}/")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Destination path is missing")

    destination = os.path.join(sys.argv[1], "clip_model")
    hf_download(destination)
    quantized_download(destination)
