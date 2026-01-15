import os
import sys

def force_ipv4():
    # Monkey patch to force IPv4, since FB seems to hang on IPv6
    import socket

    old_getaddrinfo = socket.getaddrinfo
    def new_getaddrinfo(*args, **kwargs):
        responses = old_getaddrinfo(*args, **kwargs)
        return [response
                for response in responses
                if response[0] == socket.AF_INET]
    socket.getaddrinfo = new_getaddrinfo

force_ipv4()
from huggingface_hub import hf_hub_download


def handler(event, context):
    repo_id = "laion/CLIP-ViT-B-32-256x256-DataComp-s34B-b86K"
    filename = "open_clip_model.safetensors"
    destination = os.path.join(event.get("destination"), "clip_model")

    os.makedirs(destination, exist_ok=True)

    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=destination
    )

    print(f"Model {repo_id} downloaded to {destination}/")
    return {"status": "success", "path": destination}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Destination path is missing")

    event = {"destination": sys.argv[1]}
    response = handler(event, None)
    print("Response:")
    print(response)
