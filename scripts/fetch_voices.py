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
import re

pattern = "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/voices/{name}.pt"
url = "https://huggingface.co/hexgrad/Kokoro-82M/tree/main/voices"
voices = {}

names = re.findall(
    'href="/hexgrad/Kokoro-82M/blob/main/voices/(.+).pt', requests.get(url).text
)
print(", ".join(names))

count = len(names)
for i, name in enumerate(names, 1):
    url = pattern.format(name=name)
    print(f"Downloading {url} ({i}/{count})")
    r = requests.get(url)
    r.raise_for_status()  # Ensure the request was successful
    content = io.BytesIO(r.content)
    data: np.ndarray = torch.load(content, weights_only=True).numpy()
    voices[name] = data

# Save all voices to a single .npz file
npz_path = "voices.bin"
with open(npz_path, "wb") as f:
    np.savez(f, **voices)
print(f"Created {npz_path}")
