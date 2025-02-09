"""
pip install kokoro-onnx sounddevice

wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
python examples/with_phonemes.py
"""

import sounddevice as sd
from kokoro_onnx import Kokoro
from kokoro_onnx.tokenizer import Tokenizer


tokenizer = Tokenizer()
kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")

phonemes = tokenizer.phonemize("Hello world!", lang="en-US")
samples, sample_rate = kokoro.create(
    text="", phonemes=phonemes, voice="af_sarah", speed=1.0, lang="en-us"
)
print("Playing audio...")
sd.play(samples, sample_rate)
sd.wait()
