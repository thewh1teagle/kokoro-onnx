from .config import KoKoroConfig, SUPPORTED_LANGUAGES, MAX_PHONEME_LENGTH, SAMPLE_RATE
from onnxruntime import InferenceSession
import numpy as np
from .tokenizer import tokenize, phonemize
from .voices import get_voice_style
from .log import log
import time

class Kokoro:
    def __init__(self, model_path: str, voices_path: str):
        self.sess = InferenceSession(model_path)
        self.config = KoKoroConfig(model_path, voices_path)
        self.config.validate()
        self.voices: list[str] = self.config.get_voice_names()
    
    def _create_audio(self, phonemes: str, voice: str, speed: float):
        start_t = time.time()
        tokens = tokenize(phonemes)
        assert len(tokens) <= MAX_PHONEME_LENGTH, f"Context length is {MAX_PHONEME_LENGTH}, but leave room for the pad token 0 at the start & end"

        style = get_voice_style(self.config, voice)[len(tokens)]
        tokens = [[0, *tokens, 0]]
        
        audio = self.sess.run(None, dict(
            tokens=tokens, 
            style=style,
            speed=np.ones(1, dtype=np.float32) * speed
        ))[0]
        audio_duration = len(audio) / SAMPLE_RATE
        create_duration = time.time() - start_t
        speedup_factor = audio_duration / create_duration
        log.debug(f"Created audio in length of {audio_duration:.2f}s for {len(phonemes)} phonemes in {create_duration:.2f}s (More than {speedup_factor:.2f}x real-time)")
        return audio, SAMPLE_RATE
    
    def create(self, text: str, voice: str, speed: float=1.0, lang = 'en-us'):
        """
        Create audio from text using the specified voice and speed.
        """
        
        assert lang in SUPPORTED_LANGUAGES, f"Language must be either {', '.join(SUPPORTED_LANGUAGES)}. Got {lang}"
        assert speed >= 0.5 and speed <= 2.0, "Speed should be between 0.5 and 2.0"
        assert voice in self.voices, f"Voice {voice} not found in available voices"
        
        
        start_t = time.time()
        phonemes = phonemize(text, lang)
        # Create batches of phonemes by splitting spaces to MAX_PHONEME_LENGTH
        batched_phoenemes = []
        for i in range(0, len(phonemes), MAX_PHONEME_LENGTH):
            batched_phoenemes.append(phonemes[i:i+MAX_PHONEME_LENGTH])
        
        audio = []
        silence = np.zeros(int(SAMPLE_RATE * 0.1), dtype=np.float32) # 0.1s silence
        log.debug(f'Phonemes: {phonemes}')
        log.debug(f"Creating audio for {len(batched_phoenemes)} batches for {len(phonemes)} phonemes")
        for phonemes in batched_phoenemes:
            audio_part, _ = self._create_audio(phonemes, voice, speed)
            audio.append(audio_part)
            audio.append(silence)
        audio = np.concatenate(audio)
        log.debug(f"Created audio in {time.time() - start_t:.2f}s")
        return audio, SAMPLE_RATE
        
    
    
    def get_voices(self):
        return self.voices

    def get_languages(self):
        return SUPPORTED_LANGUAGES