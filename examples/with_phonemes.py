"""
pip install kokoro-onnx sounddevice

wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.npz
python examples/with_phonemes.py
"""

import sounddevice as sd
from kokoro_onnx import Kokoro

kokoro = Kokoro("kokoro-v0_19.onnx", "voices.npz")
phonemes = "mˈʌsk sˈɛd ɪnðɪ ɑːktˈoʊbɚ twˈɛnti twˈɛnti θɹˈiː kˈɔːl."  # Musk said in the October 2023 call
samples, sample_rate = kokoro.create(
    "", phonemes=phonemes, voice="af_sarah", speed=1.0, lang="en-us"
)
print("Playing audio...")
sd.play(samples, sample_rate)
sd.wait()
