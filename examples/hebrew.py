"""
Usage:
1.
    Install uv from https://docs.astral.sh/uv/getting-started/installation
2.
    Copy this file to new folder
3.
    Download these files
    wget https://huggingface.co/thewh1teagle/kokoro-hebrew-nc/resolve/main/kokoro.onnx
    wget https://huggingface.co/thewh1teagle/kokoro-hebrew-nc/resolve/main/voices-hebrew.bin
    wget https://huggingface.co/thewh1teagle/renikud/resolve/neobert-130m/model.onnx -O renikud.onnx
4. Run
    uv venv --seed -p 3.12
    source .venv/bin/activate
    uv pip install -U kokoro-onnx soundfile renikud-onnx
    uv run main.py

For other languages read https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md

Note: this Hebrew model is licensed for non-commercial use under the terms of
https://huggingface.co/avris/kokoro-hebrew-saspeech
"""

import soundfile as sf
from renikud_onnx import G2P

from kokoro_onnx import Kokoro

# Hebrew G2P
g2p = G2P("renikud.onnx")

# Kokoro. The model carries its own vocabulary, so no vocab_config is needed
kokoro = Kokoro("kokoro.onnx", "voices-hebrew.bin")

# Phonemize
text = "שימו לב נוסעים יקרים, הרכבת תכנס לתחנת תל אביב מרכז בעוד מספר דקות. אנא התרחקו מקצה הרציף."
phonemes = g2p.phonemize(text)

# Create. This model reports durations, so continuous keeps the prosody
# running across the joins of a long text
samples, sample_rate = kokoro.create(
    phonemes, "he_shaul", is_phonemes=True, continuous=True
)

# Save
sf.write("audio.wav", samples, sample_rate)
print("Created audio.wav")
