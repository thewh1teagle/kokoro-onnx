"""
pip install kokoro-onnx soundfile

wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.json
python examples/save.py
"""

import soundfile as sf
from kokoro_onnx import Kokoro

kokoro = Kokoro('kokoro-v0_19.onnx', 'voices.json')
text = """
A judge has ordered that Donald Trump will be sentenced on 10 January in his hush-money case in New York - less than two weeks before he is set to be sworn in as president.

New York Justice Juan Merchan signalled he would not sentence Trump to jail time, probation or a fine, and instead give him a "conditional discharge", and wrote in his order that the president-elect could appear in person or virtually for the hearing.

Trump had attempted to use his presidential election victory to have the case against him dismissed.

His team criticised the judge's decision to go forward with sentencing and said the "lawless" case should be dismissed "immediately".
"""
samples, sample_rate = kokoro.create(text, voice='af_sarah', speed=1.0, lang='en-us')
sf.write('audio.wav', samples, sample_rate)
print('Created audio.wav')