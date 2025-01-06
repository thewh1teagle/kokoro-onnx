"""
pip install kokoro-onnx soundfile

wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.json
python examples/podcast.py
"""

import soundfile as sf
from kokoro_onnx import Kokoro
import numpy as np
import random

# fmt: off
sentences_json = [
    { "voice": "af_sarah", "text": "Hello and welcome to the podcast! We’ve got some exciting things lined up today." },  
    { "voice": "am_michael", "text": "It’s going to be an exciting episode. Stick with us!" },  
    { "voice": "af_sarah", "text": "But first, we’ve got a special guest with us. Please welcome Nicole!" },  
    { "voice": "af_sarah", "text": "Now, we’ve been told Nicole has a very unique way of speaking today... a bit of a mysterious vibe, if you will." },  
    { "voice": "af_nicole", "text": "Hey there... I’m so excited to be a guest today... But I thought I’d keep it quiet... for now..." },  # Nicole whispers
    { "voice": "am_michael", "text": "Well, it certainly adds some intrigue! Let’s dive in and see what that’s all about." },  
    { "voice": "af_sarah", "text": "Today, we’re covering something that’s close to our hearts. Ready for it?" },  
    { "voice": "am_michael", "text": "It’s going to be a good one!" }  
]

def random_pause(min_duration=0.5, max_duration=2.0):
    silence_duration = random.uniform(min_duration, max_duration)
    silence = np.zeros(int(silence_duration * sample_rate))
    return silence


kokoro = Kokoro("kokoro-v0_19.onnx", "voices.json")

audio = []

# Loop through sentences_json and process each entry
for sentence in sentences_json:
    voice = sentence["voice"]
    text = sentence["text"]
    print(f"Creating audio for: {text} with voice: {voice}")
    
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
