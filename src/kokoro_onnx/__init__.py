from .config import KoKoroConfig
from onnxruntime import InferenceSession
import numpy as np
from .tokenizer import tokenize, phonemize
from .voices import get_voice_style

class Kokoro:
    def __init__(self, model_path: str, voices_path: str):
        self.sess = InferenceSession(model_path)
        self.config = KoKoroConfig(model_path, voices_path)
        self.config.validate()
        self.voices: list[str] = self.config.get_voice_names()
    
    def create(self, text: str, voice: str, speed: float=1.0, lang = 'en-us'):
        """
        Create audio from text using the specified voice and speed.
        """
        
        assert lang in ['en-us', 'en-gb'], "Language must be either 'en-us' or 'en-gb'"
        assert speed >= 0.5 and speed <= 2.0, "Speed should be between 0.5 and 2.0"
        assert voice in self.voices, f"Voice {voice} not found in available voices"
        
        # Max is 510 phonemes
        phonemes = phonemize(text, lang)
        if len(phonemes) > 510:
            # TODO: Implement splitting of text into multiple parts
            print("Warning: Text is too long, must be less than 510 phonemes")
            phonemes = phonemes[:510]
        tokens = tokenize(phonemes)
        assert len(tokens) <= 510, "Context length is 512, but leave room for the pad token 0 at the start & end"

        
        style = get_voice_style(self.config, voice)[len(tokens)]
        tokens = [[0, *tokens, 0]]
        
        audio = self.sess.run(None, dict(
            tokens=tokens, 
            style=style,
            speed=np.ones(1, dtype=np.float32) * speed
        ))[0]
        
        return audio, 24000
    
    def get_voices(self):
        return self.voices
