"""
Note: on Linux you need to run this as well: apt-get install portaudio19-dev

pip install kokoro-onnx sounddevice

wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
python examples/with_stream.py
"""

import sounddevice as sd
from kokoro_onnx import Kokoro
import asyncio


text = """
We've just been hearing from Matthew Cappucci, a senior meteorologist at the weather app MyRadar, who says Kansas City is seeing its heaviest snow in 32 years - with more than a foot (30 to 40cm) having come down so far.

Despite it looking as though the storm is slowly moving eastwards, Cappucci says the situation in Kansas and Missouri remains serious.

He says some areas near the Ohio River are like "skating rinks", telling our colleagues on Newsday that in Missouri in particular there is concern about how many people have lost power, and will lose power, creating enough ice to pull power lines down.

Temperatures are set to drop in the next several days, in may cases dipping maybe below minus 10 to minus 15 degrees Celsius for an extended period of time.

There is a special alert for Kansas, urging people not to leave their homes: "The ploughs are getting stuck, the police are getting stuck, everybody’s getting stuck - stay home."
"""


async def main():
    kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")

    stream = kokoro.create_stream(
        text,
        voice="af_nicole",
        speed=1.0,
        lang="en-us",
    )

    count = 0
    async for samples, sample_rate in stream:
        count += 1
        print(f"Playing audio stream ({count})...")
        sd.play(samples, sample_rate)
        sd.wait()


asyncio.run(main())
