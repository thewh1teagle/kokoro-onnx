# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy==2.0.2",
#     "requests",
#     "torch==2.5.1",
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

# Extract voice names
voice_url = "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/voices/{name}.pt"
api_url = "https://huggingface.co/api/models/hexgrad/Kokoro-82M/tree/main/voices"
names = [voice["path"][7:-3] for voice in requests.get(api_url).json()]
count = len(names)

# Extract voice files
print(f"Found {count} voices")
voices = {}
for i, name in enumerate(names, 1):
    url = voice_url.format(name=name)
    print(f"Downloading {name} from {url} ({i}/{count})")
    r = requests.get(url)
    r.raise_for_status()  # Ensure the request was successful
    content = io.BytesIO(r.content)
    data: np.ndarray = torch.load(content, weights_only=True).numpy()
    voices[name] = data

# Save all voices to a single .npz file
npz_path = "voices-v1.0.bin"
with open(npz_path, "wb") as f:
    np.savez(f, **voices)

mb_size = os.path.getsize(npz_path) // 1000 // 1000
print(f"Created {npz_path} ({mb_size}MB)")
