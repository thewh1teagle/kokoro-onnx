# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy==2.0.2",
#     "requests",
#     "torch==2.5.1",
# ]
# ///
# declaring requests is necessary for running
"""
Run this file via:
uv run scripts/fetch_voices.py
"""

import io
import json

import numpy as np
import requests
import torch

voices = [
    "af",
    "af_bella",
    "af_nicole",
    "af_sarah",
    "af_sky",
    "am_adam",
    "am_michael",
    "bf_emma",
    "bf_isabella",
    "bm_george",
    "bm_lewis",
]
voices_json = {}
pattern = "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/voices/{voice}.pt"
for voice in voices:
    url = pattern.format(voice=voice)
    print(f"Downloading {url}")
    r = requests.get(url)
    content = io.BytesIO(r.content)
    voice_data: np.ndarray = torch.load(content).numpy()
    voices_json[voice] = voice_data.tolist()

path = "voices.json"
with open(path, "w") as f:
    json.dump(voices_json, f, indent=4)
print(f"Created {path}")
