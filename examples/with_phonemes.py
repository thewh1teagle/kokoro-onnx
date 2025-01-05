"""
pip install kokoro-onnx soundfile

wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.json
python examples/with_phonemes.py
"""

import soundfile as sf
from kokoro_onnx import Kokoro

kokoro = Kokoro("kokoro-v0_19.onnx", "voices.json")
phonemes = "mˈʌsk sˈɛd ɪnðɪ ɑːktˈoʊbɚ twˈɛnti twˈɛnti θɹˈiː kˈɔːl."  # Musk said in the October 2023 call
samples, sample_rate = kokoro.create(
    "", phonemes=phonemes, voice="af_sarah", speed=1.0, lang="en-us"
)
sf.write("audio.wav", samples, sample_rate)
print("Created audio.wav")
