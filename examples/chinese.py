"""
Usage:
1.
    Install uv from https://docs.astral.sh/uv/getting-started/installation
2.
    Copy this file to new folder
3.
    Run
    uv venv -p 3.12
    uv pip install kokoro-onnx==0.4.4 soundfile==0.13.1 'misaki[zh]==0.8.4'
3.
    Download these files
    https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/kokoro-v1.1-zh.onnx
    https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/voices-v1.1-zh.bin
4. Run
    uv run main.py
"""

import soundfile as sf
from kokoro_onnx import Kokoro
from misaki import zh

# Misaki G2P with espeak-ng fallback
g2p = zh.ZHG2P()

text = "千里之行，始于足下。"
voice = "af_maple"
kokoro = Kokoro("kokoro-v1.1-zh.onnx", "voices-v1.1-zh.bin")
phonemes, _ = g2p(text)
samples, sample_rate = kokoro.create(phonemes, voice=voice, speed=1.0, is_phonemes=True)
sf.write("audio.wav", samples, sample_rate)
print("Created audio.wav")
