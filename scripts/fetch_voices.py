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
import numpy as np
import requests
import torch

names = [
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

pattern = "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/voices/{name}.pt"
voices = {}

for name in names:
    url = pattern.format(name=name)
    print(f"Downloading {url}")
    r = requests.get(url)
    r.raise_for_status()  # Ensure the request was successful
    content = io.BytesIO(r.content)
    data: np.ndarray = torch.load(content).numpy()
    voices[name] = data

# Save all voices to a single .npz file
npz_path = "voices.npz"
np.savez(npz_path, **voices)
print(f"Created {npz_path}")
