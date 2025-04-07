"""
pip install -U kokoro-onnx soundfile

wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
python examples/podcast.py
"""

import random

import numpy as np
import soundfile as sf

from kokoro_onnx import Kokoro

# fmt: off
sentences = [
    { "voice": "af_sarah", "text": "Hello and welcome to the podcast! We’ve got some exciting things lined up today." }, # Sarah
    { "voice": "am_michael", "text": "It’s going to be an exciting episode. Stick with us!" }, # Michael
    { "voice": "af_sarah", "text": "But first, we’ve got a special guest with us. Please welcome Nicole!" },   # Sarah
    { "voice": "af_sarah", "text": "Now, we’ve been told Nicole has a very unique way of speaking today... a bit of a mysterious vibe, if you will." }, # Sarah
    { "voice": "af_nicole", "text": "Hey there... I’m so excited to be a guest today... But I thought I’d keep it quiet... for now..." },  # Nicole whispers
    { "voice": "am_michael", "text": "Well, it certainly adds some intrigue! Let’s dive in and see what that’s all about." }, # Sarah
    { "voice": "af_sarah", "text": "Today, we’re covering something that’s close to our hearts" }, # Sarah
    { "voice": "am_michael", "text": "It’s going to be a good one!" }   # Michael
]

def random_pause(min_duration=0.5, max_duration=2.0):
    silence_duration = random.uniform(min_duration, max_duration)
    silence = np.zeros(int(silence_duration * sample_rate))
    return silence


kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")

audio = []

for sentence in sentences:
    voice = sentence["voice"]
    text = sentence["text"]
    print(f"Creating audio with {voice}: {text}")
    
    samples, sample_rate = kokoro.create(
        text,
        voice=voice,
        speed=1.0,
        lang="en-us",
    )
    audio.append(samples)
    # Add random silence after each sentence
    audio.append(random_pause())

# Concatenate all audio parts
audio = np.concatenate(audio)

# Save the generated audio to file
sf.write("podcast.wav", audio, sample_rate)
print("Created podcast.wav")
