"""
Note: on Linux you need to run this as well: apt-get install portaudio19-dev

1. Prepare virtual environment
    uv venv --seed -p 3.11
    source .venv/bin/activate

2. Install packages
    pip install kokoro-onnx sounddevice 'misaki[en]'

3. Download models
    wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
    wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin

4. Run
    python examples/with_misaki.py
    
Please read carefully https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md
To use other languages install misaki with the specific language. Example: pip install misaki[ko] (Korean). And change the import. Example: from misaki.ko import KOG2P
"""

import sounddevice as sd
from kokoro_onnx import Kokoro
from misaki import en, espeak

# Misaki g2p with espeak-ng fallback
fallback = espeak.EspeakFallback(british=False)
g2p = en.G2P(trf=False, british=False, fallback=fallback)

# Kokoro
kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")

# Phonemize
text = '[Misaki](/misˈɑki/) is a G2P engine designed for [Kokoro](/kˈOkəɹO/) models.'
phonemes, _ = g2p(text)

# Create
samples, sample_rate = kokoro.create(text="", phonemes=phonemes, voice="af_sarah")

# Play
print("Playing audio...")
sd.play(samples, sample_rate)
sd.wait()
