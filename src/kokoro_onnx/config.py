from pathlib import Path
from dataclasses import dataclass

MAX_PHONEME_LENGTH = 510
SAMPLE_RATE = 24000


@dataclass
class EspeakConfig:
    lib_path: str | None = None
    data_path: str | None = None


class KoKoroConfig:
    def __init__(
        self,
        model_path: str,
        voices_path: str,
        espeak_config: EspeakConfig | None = None,
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
            error_msg += "\nwget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
            raise FileNotFoundError(error_msg)

        if not Path(self.model_path).exists():
            error_msg = f"Model file not found at {self.model_path}"
            error_msg += (
                "\nYou can download the model file from https://github.com/thewh1teagle/kokoro-onnx/releases"
            )
            raise FileNotFoundError(error_msg)


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

def get_vocab_zh():
    dicts = {
        ";": 1, ":": 2, ",": 3, ".": 4, "!": 5, "?": 6, "/": 7, "—": 9, "…": 10, "\"": 11, "(": 12, ")": 13, """: 14, """: 15, " ": 16, "\u0303": 17, "ʣ": 18, "ʥ": 19, "ʦ": 20,
        "ʨ": 21, "ᵝ": 22, "ㄓ": 23, "A": 24, "I": 25, "ㄅ": 30, "O": 31, "ㄆ": 32, "Q": 33, "R": 34, "S": 35, "T": 36, "ㄇ": 37, "ㄈ": 38, "W": 39, "ㄉ": 40, "Y": 41, "ᵊ": 42, "a": 43, "b": 44,
        "c": 45, "d": 46, "e": 47, "f": 48, "ㄊ": 49, "h": 50, "i": 51, "j": 52, "k": 53, "l": 54, "m": 55, "n": 56, "o": 57, "p": 58, "q": 59, "r": 60, "s": 61, "t": 62, "u": 63, "v": 64,
        "w": 65, "x": 66, "y": 67, "z": 68, "ɑ": 69, "ɐ": 70, "ɒ": 71, "æ": 72, "ㄋ": 73, "ㄌ": 74, "β": 75, "ɔ": 76, "ɕ": 77, "ç": 78, "ㄍ": 79, "ɖ": 80, "ð": 81, "ʤ": 82, "ə": 83, "ㄎ": 84,
        "ㄦ": 85, "ɛ": 86, "ɜ": 87, "ㄏ": 88, "ㄐ": 89, "ɟ": 90, "ㄑ": 91, "ɡ": 92, "ㄒ": 93, "ㄔ": 94, "ㄕ": 95, "ㄗ": 96, "ㄘ": 97, "ㄙ": 98, "月": 99, "ㄚ": 100, "ɨ": 101, "ɪ": 102, "ʝ": 103, "ㄛ": 104,
        "ㄝ": 105, "ㄞ": 106, "ㄟ": 107, "ㄠ": 108, "ㄡ": 109, "ɯ": 110, "ɰ": 111, "ŋ": 112, "ɳ": 113, "ɲ": 114, "ɴ": 115, "ø": 116, "ㄢ": 117, "ɸ": 118, "θ": 119, "œ": 120, "ㄣ": 121, "ㄤ": 122, "ɹ": 123, "ㄥ": 124,
        "ɾ": 125, "ㄖ": 126, "ㄧ": 127, "ʁ": 128, "ɽ": 129, "ʂ": 130, "ʃ": 131, "ʈ": 132, "ʧ": 133, "ㄨ": 134, "ʊ": 135, "ʋ": 136, "ㄩ": 137, "ʌ": 138, "ɣ": 139, "ㄜ": 140, "ㄭ": 141, "χ": 142, "ʎ": 143, "十": 144,
        "压": 145, "言": 146, "ʒ": 147, "ʔ": 148, "阳": 149, "要": 150, "阴": 151, "应": 152, "用": 153, "又": 154, "中": 155, "ˈ": 156, "ˌ": 157, "ː": 158, "穵": 159, "外": 160, "万": 161, "ʰ": 162, "王": 163,
        "ʲ": 164, "为": 165, "文": 166, "瓮": 167, "我": 168, "3": 169, "5": 170, "1": 171, "2": 172, "4": 173, "元": 175, "云": 176, "ᵻ": 177
    }
    return dicts


VOCAB = get_vocab()
VOCAB_ZH = get_vocab_zh()
