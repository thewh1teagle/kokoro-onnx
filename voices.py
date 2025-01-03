import numpy as np
import json

def get_voice_style(name: str):
    with open('voices.json') as f:
        voices = json.load(f)
    return np.array(voices[name], dtype=np.float32)

def get_voice_names():
    with open('voices.json') as f:
        voices = json.load(f)
    return voices.keys()