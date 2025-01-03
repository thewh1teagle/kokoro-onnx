import numpy as np
import json
from .config import KoKoroConfig
from functools import lru_cache

@lru_cache
def get_voice_style(config: KoKoroConfig, name: str):
    with open(config.voices_path) as f:
        voices = json.load(f)
    return np.array(voices[name], dtype=np.float32)
