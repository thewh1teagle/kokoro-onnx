import json
from functools import lru_cache
from pathlib import Path
from dataclasses import dataclass

# https://github.com/espeak-ng/espeak-ng/blob/master/docs/languages.md
SUPPORTED_LANGUAGES = [
    "en-us",  # English
    "en-gb",  # English (British)
    "fr-fr",  # French
    "ja",  # Japanese
    "ko",  # Korean
    "cmn",  # Mandarin Chinese
]
MAX_PHONEME_LENGTH = 510
SAMPLE_RATE = 24000


@dataclass
class EspeakConfig:
    lib_path: str = None
    data_path: str = None


class KoKoroConfig:
    def __init__(
        self, model_path: str, voices_path: str, espeak_config: EspeakConfig = None
    ):
        self.model_path = model_path
        self.voices_path = voices_path
        self.espeak_config = espeak_config

    def validate(self):
        if not Path(self.voices_path).exists():
            error_msg = f"Voices file not found at {self.voices_path}"
            error_msg += (
                "\nYou can download the voices file using the following command:"
            )
            error_msg += "\nwget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.json"
            raise FileNotFoundError(error_msg)

        if not Path(self.model_path).exists():
            error_msg = f"Model file not found at {self.model_path}"
            error_msg += (
                "\nYou can download the model file using the following command:"
            )
            error_msg += "\nwget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx"
            raise FileNotFoundError(error_msg)

    @lru_cache
    def get_voice_names(self):
        with open(self.voices_path) as f:
            voices = json.load(f)
        return voices.keys()


def get_vocab():
    _pad = "$"
    _punctuation = ';:,.!?¡¿—…"«»“” '
    _letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
    symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)
    dicts = {}
    for i in range(len((symbols))):
        dicts[symbols[i]] = i
    return dicts


VOCAB = get_vocab()
