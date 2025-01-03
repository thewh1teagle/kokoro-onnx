"""
Install uv https://docs.astral.sh/uv/getting-started/

wget https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/voices/af.pt
wget https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/kokoro-v0_19.onnx
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.json
uv run main.py
"""

from onnxruntime import InferenceSession
import soundfile as sf
import numpy as np
from tokenizer import tokenize
from voices import get_voice_style, get_voice_names

tokens = tokenize('Hello world! How are you?', lang='en-us')
# Context length is 512, but leave room for the pad token 0 at the start & end
assert len(tokens) <= 510, len(tokens)

# Style vector based on len(tokens), ref_s has shape (1, 256)
names = get_voice_names()
print("Available voices:", names)
ref_s = get_voice_style('bf_isabella')[len(tokens)]
# Add the pad ids, and reshape tokens, should now have shape (1, <=512)
tokens = [[0, *tokens, 0]]

sess = InferenceSession('kokoro-v0_19.onnx')

audio = sess.run(None, dict(
    tokens=tokens, 
    style=ref_s,
    speed=np.ones(1, dtype=np.float32) # 0.5 to 2
))[0]


sf.write('output.wav', audio, 24000)