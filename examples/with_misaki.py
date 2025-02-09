"""
Note: on Linux you need to run this as well: apt-get install portaudio19-dev

uv venv --seed -p 3.11
source .venv/bin/activate
pip install kokoro-onnx sounddevice 'misaki[en]'

wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
python examples/with_misaki.py
"""

import sounddevice as sd
from kokoro_onnx import Kokoro
from misaki import en

g2p = en.G2P(trf=False, british=False, fallback=None)
kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
text = "[Misaki](/misˈɑki/) is a G2P engine designed for [Kokoro](/kˈOkəɹO/) models."
phonemes, _ = g2p(text)
samples, sample_rate = kokoro.create(
    text="", phonemes=phonemes, voice="af_sarah", speed=1.0, lang="en-us"
)
print("Playing audio...")
sd.play(samples, sample_rate)
sd.wait()
