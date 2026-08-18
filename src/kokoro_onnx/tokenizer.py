import ctypes
import ctypes.util
import os
import platform
import sys
import threading

import espeakng_loader
import phonemizer
from phonemizer.backend.espeak.wrapper import EspeakWrapper

from .config import DEFAULT_VOCAB, MAX_PHONEME_LENGTH, EspeakConfig
from .log import log

# espeak-ng holds process-global state, so concurrent phonemization returns
# corrupted phonemes. Inference stays outside this lock and runs concurrently.
_espeak_lock = threading.Lock()


class Tokenizer:
    def __init__(self, espeak_config: EspeakConfig | None = None, vocab: dict = None):
        self.vocab = vocab or DEFAULT_VOCAB

        if not espeak_config:
            espeak_config = EspeakConfig()
        if not espeak_config.data_path:
            espeak_config.data_path = espeakng_loader.get_data_path()
        if not espeak_config.lib_path:
            espeak_config.lib_path = espeakng_loader.get_library_path()

        # Check if PHONEMIZER_ESPEAK_LIBRARY was set
        if os.getenv("PHONEMIZER_ESPEAK_LIBRARY"):
            espeak_config.lib_path = os.getenv("PHONEMIZER_ESPEAK_LIBRARY")

        # Check that the espeak-ng library can be loaded
        try:
            ctypes.cdll.LoadLibrary(espeak_config.lib_path)
        except Exception as e:
            log.error(f"Failed to load espeak shared library: {e}")
            log.warning("Falling back to system wide espeak-ng library")

            # Fallback system wide load
            error_info = (
                "Failed to load espeak-ng from fallback. Please install espeak-ng system wide.\n"
                "\tSee https://github.com/espeak-ng/espeak-ng/blob/master/docs/guide.md\n"
                "\tNote: you can specify shared library path using PHONEMIZER_ESPEAK_LIBRARY environment variable.\n"
                f"Environment:\n\t{platform.platform()} ({platform.release()}) | {sys.version}"
            )
            espeak_config.lib_path = ctypes.util.find_library(
                "espeak-ng"
            ) or ctypes.util.find_library("espeak")
            if not espeak_config.lib_path:
                raise RuntimeError(error_info)
            try:
                ctypes.cdll.LoadLibrary(espeak_config.lib_path)
            except Exception as e:
                raise RuntimeError(f"{e}: {error_info}")

        with _espeak_lock:
            EspeakWrapper.set_data_path(espeak_config.data_path)
            EspeakWrapper.set_library(espeak_config.lib_path)

    @staticmethod
    def normalize_text(text) -> str:
        return text.strip()

    def tokenize(self, phonemes, limit: int | None = MAX_PHONEME_LENGTH):
        """Map phonemes to token ids, dropping anything outside the vocabulary."""
        if limit is not None and len(phonemes) > limit:
            raise ValueError(f"text is too long, must be less than {limit} phonemes")
        return [i for i in map(self.vocab.get, phonemes) if i is not None]

    def known(self, phonemes: str) -> str:
        """The phonemes that survive tokenization, aligned one to one with it."""
        return "".join(p for p in phonemes if p in self.vocab)

    def phonemize(self, text, lang="en-us", norm=True) -> str:
        """
        lang can be 'en-us' or 'en-gb'
        """
        if norm:
            text = Tokenizer.normalize_text(text)

        with _espeak_lock:
            phonemes = phonemizer.phonemize(
                text, lang, preserve_punctuation=True, with_stress=True
            )
        phonemes = "".join(filter(lambda p: p in self.vocab, phonemes))
        return phonemes.strip()
