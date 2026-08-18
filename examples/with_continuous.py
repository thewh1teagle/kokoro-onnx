"""
Long text is normally synthesized in batches, and every batch starts and ends
as its own utterance, which is audible at the joins. With continuous=True the
text is synthesized as overlapping windows and only the middle of each window
is kept, so prosody carries across. It costs around 1.4x the work and needs a
model exported with a duration output, see scripts/export.py

pip install -U kokoro-onnx soundfile

wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/kokoro-v1.1-zh.onnx
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/voices-v1.1-zh.bin
wget https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh/raw/main/config.json
python examples/with_continuous.py
"""

import soundfile as sf

from kokoro_onnx import Kokoro

text = """
The lighthouse keeper wrote the same three words in his logbook every evening,
and never once explained them to anyone who asked. On the night the storm
finally broke, he climbed the stairs with a lamp in one hand and the book in
the other, and he did not come back down until morning.
"""

kokoro = Kokoro("kokoro-v1.1-zh.onnx", "voices-v1.1-zh.bin", vocab_config="config.json")
samples, sample_rate = kokoro.create(
    text, voice="af_maple", lang="en-us", continuous=True
)
sf.write("audio.wav", samples, sample_rate)
print("Created audio.wav")
