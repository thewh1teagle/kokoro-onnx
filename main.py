"""
Install uv https://docs.astral.sh/uv/getting-started/

wget https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/voices/af.pt
wget https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/kokoro-v0_19.onnx
uv run main.py
"""

from onnxruntime import InferenceSession
import torch
import soundfile as sf
import numpy as np
from tokenizer import tokenize

tokens = tokenize('Hello world! How are you?', lang='en-us')
# tokens = [50, 157, 43, 135, 16, 53, 135, 46, 16, 43, 102, 16, 56, 156, 57, 135, 6, 16, 102, 62, 61, 16, 70, 56, 16, 138, 56, 156, 72, 56, 61, 85, 123, 83, 44, 83, 54, 16, 53, 65, 156, 86, 61, 62, 131, 83, 56, 4, 16, 54, 156, 43, 102, 53, 16, 156, 72, 61, 53, 102, 112, 16, 70, 56, 16, 138, 56, 44, 156, 76, 158, 123, 56, 16, 62, 131, 156, 43, 102, 54, 46, 16, 102, 48, 16, 81, 47, 102, 54, 16, 54, 156, 51, 158, 46, 16, 70, 16, 92, 156, 135, 46, 16, 54, 156, 43, 102, 48, 4, 16, 81, 47, 102, 16, 50, 156, 72, 64, 83, 56, 62, 16, 156, 51, 158, 64, 83, 56, 16, 44, 157, 102, 56, 16, 44, 156, 76, 158, 123, 56, 4]

# Context length is 512, but leave room for the pad token 0 at the start & end
assert len(tokens) <= 510, len(tokens)

# Style vector based on len(tokens), ref_s has shape (1, 256)
ref_s = torch.load('af.pt')[len(tokens)].numpy()
# Add the pad ids, and reshape tokens, should now have shape (1, <=512)
tokens = [[0, *tokens, 0]]

sess = InferenceSession('kokoro-v0_19.onnx')

audio = sess.run(None, dict(
    tokens=tokens, 
    style=ref_s,
    speed=np.ones(1, dtype=np.float32)
))[0]


sf.write('output.wav', audio, 24000)