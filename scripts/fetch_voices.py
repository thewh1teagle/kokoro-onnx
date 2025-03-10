# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy==2.0.2",
#     "requests",
#     "torch==2.5.1",
#     "tqdm==4.67.1",
# ]
# ///
"""
Run this file via:
uv run scripts/fetch_voices.py

See voices in
https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md
"""

import io
import numpy as np
import requests
import torch
import os
from tqdm import tqdm

config = {
    "Kokoro-82M-v1.1-zh": {
        "voice_url": "https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh/resolve/main/voices/{name}.pt",
        "api_url": "https://huggingface.co/api/models/hexgrad/Kokoro-82M-v1.1-zh/tree/main/voices",
        "npz_path": "voices-v1.1-zh.bin",
    },
    # "Kokoro-82M": {
    #     "voice_url": "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/voices/{name}.pt",
    #     "api_url": "https://huggingface.co/api/models/hexgrad/Kokoro-82M/tree/main/voices",
    #     "npz_path": "voices-v1.0.bin",
    # },
}
# Extract voice names


def get_voice_names(api_url):
    resp = requests.get(api_url)
    data = resp.json()
    names = [voice["path"][7:-3] for voice in data]
    return names


def download_voices(voice_url: str, names: list[str], npz_path: str):
    count = len(names)

    # Extract voice files
    print(f"Found {count} voices")
    voices = {}
    for name in tqdm(names):
        url = voice_url.format(name=name)
        print(f"Downloading {name}")
        r = requests.get(url)
        r.raise_for_status()  # Ensure the request was successful
        content = io.BytesIO(r.content)
        data: np.ndarray = torch.load(content, weights_only=True).numpy()
        voices[name] = data

    # Save all voices to a single .npz file
    with open(npz_path, "wb") as f:
        np.savez(f, **voices)

        mb_size = os.path.getsize(npz_path) // 1000 // 1000
        print(f"Created {npz_path} ({mb_size}MB)")


def main():
    for model_name, model_config in config.items():
        print(f"Downloading {model_name}")
        voice_url, api_url, npz_path = (
            model_config["voice_url"],
            model_config["api_url"],
            model_config["npz_path"],
        )
        voice_names = get_voice_names(api_url)
        download_voices(voice_url, voice_names, npz_path)


main()
