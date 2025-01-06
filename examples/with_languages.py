"""
pip install kokoro-onnx soundfile

wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.json
python examples/try_languages.py
"""

import soundfile as sf
from kokoro_onnx import Kokoro

sentences = {
    "en-us": "Hello, World!",
    "en-gb": "Hello, World!",
    "fr-fr": "Bonjour, Monde!",
    "ja": "こんにちは、世界！",
    "ko": "안녕하세요, 세계!",
    "cmn": "你好，世界！",  # Mandarin Chinese
}

kokoro = Kokoro("kokoro-v0_19.onnx", "voices.json")
for lang, sentence in sentences.items():
    samples, sample_rate = kokoro.create(sentence, voice="af", speed=1.0, lang=lang)
    sf.write(f"{lang}.wav", samples, sample_rate)
    print(f"Created {lang}.wav")
