"""
pip install kokoro-onnx soundfile

wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.json
python examples/with_blending.py
"""

import soundfile as sf
from kokoro_onnx import Kokoro
import numpy as np

kokoro = Kokoro("kokoro-v0_19.onnx", "voices.json")
nicole: np.ndarray = kokoro.get_voice_style("af_nicole")
michael: np.ndarray = kokoro.get_voice_style("am_michael")
blend = np.add(nicole * (50 / 100), michael * (50 / 100))
samples, sample_rate = kokoro.create(
    "Hello. This audio is generated by Kokoro!",
    voice=blend,
    speed=1.0,
    lang="en-us",
)
sf.write("audio.wav", samples, sample_rate)
print("Created audio.wav")