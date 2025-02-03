"""
pip install kokoro-onnx soundfile

wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
python examples/with_language.py
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

kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
for lang, sentence in sentences.items():
    samples, sample_rate = kokoro.create(sentence, voice="af", speed=1.0, lang=lang)
    sf.write(f"{lang}.wav", samples, sample_rate)
    print(f"Created {lang}.wav")
