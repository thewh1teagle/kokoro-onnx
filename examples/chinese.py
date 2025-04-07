"""
Usage:
1.
    Install uv from https://docs.astral.sh/uv/getting-started/installation
2.
    Copy this file to new folder
3.
    Run
    uv venv -p 3.12
    uv pip install -U kokoro-onnx soundfile 'misaki[zh]'
3.
    Download these files
    https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/kokoro-v1.1-zh.onnx
    https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/voices-v1.1-zh.bin
    https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh/raw/main/config.json
4. Run
    uv run main.py
"""

import soundfile as sf
from misaki import zh

from kokoro_onnx import Kokoro

# Misaki G2P with espeak-ng fallback
g2p = zh.ZHG2P(version="1.1")

text = "千里之行，始于足下。"
voice = "zf_001"
kokoro = Kokoro("kokoro-v1.1-zh.onnx", "voices-v1.1-zh.bin", vocab_config="config.json")
phonemes, _ = g2p(text)
samples, sample_rate = kokoro.create(phonemes, voice=voice, speed=1.0, is_phonemes=True)
sf.write("audio.wav", samples, sample_rate)
print("Created audio.wav")
