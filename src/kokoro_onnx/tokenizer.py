import re
import phonemizer
from phonemizer.backend.espeak.wrapper import EspeakWrapper
import espeakng_loader
from .config import MAX_PHONEME_LENGTH, VOCAB, EspeakConfig
from .log import log
import ctypes
import platform
import sys
import os


class Tokenizer:
    def __init__(self, espeak_config: EspeakConfig | None = None):
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
            # Show OS information on error and try fallback to system wide
            environment_info = "OS: {}\nRelease: {}\nPython: {}".format(
                platform.platform(), platform.release(), sys.version
            )
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

        EspeakWrapper.set_data_path(espeak_config.data_path)
        EspeakWrapper.set_library(espeak_config.lib_path)

    @staticmethod
    def split_num(num):
        num = num.group()
        if "." in num:
            return num
        elif ":" in num:
            h, m = [int(n) for n in num.split(":")]
            if m == 0:
                return f"{h} o'clock"
            elif m < 10:
                return f"{h} oh {m}"
            return f"{h} {m}"
        year = int(num[:4])
        if year < 1100 or year % 1000 < 10:
            return num
        left, right = num[:2], int(num[2:4])
        s = "s" if num.endswith("s") else ""
        if 100 <= year % 1000 <= 999:
            if right == 0:
                return f"{left} hundred{s}"
            elif right < 10:
                return f"{left} oh {right}{s}"
        return f"{left} {right}{s}"

    @staticmethod
    def flip_money(m):
        m = m.group()
        bill = "dollar" if m[0] == "$" else "pound"
        if m[-1].isalpha():
            return f"{m[1:]} {bill}s"
        elif "." not in m:
            s = "" if m[1:] == "1" else "s"
            return f"{m[1:]} {bill}{s}"
        b, c = m[1:].split(".")
        s = "" if b == "1" else "s"
        c = int(c.ljust(2, "0"))
        coins = (
            f"cent{'' if c == 1 else 's'}"
            if m[0] == "$"
            else ("penny" if c == 1 else "pence")
        )
        return f"{b} {bill}{s} and {c} {coins}"

    @staticmethod
    def point_num(num) -> str:
        a, b = num.group().split(".")
        return " point ".join([a, " ".join(b)])

    @staticmethod
    def normalize_text(text) -> str:
        # remove leading and trailing whitespace and empty lines
        text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        # replace curly quotes with straight quotes
        text = text.replace(chr(8216), "'").replace(chr(8217), "'")
        # replace curly double quotes with straight double quotes
        text = text.replace("«", chr(8220)).replace("»", chr(8221))
        # replace other curly quotes with straight double quotes
        text = text.replace(chr(8220), '"').replace(chr(8221), '"')
        # replace other curly quotes with straight double quotes
        text = text.replace("(", "«").replace(")", "»")
        for a, b in zip("、。！，：；？", ",.!,:;?"):
            text = text.replace(a, b + " ")
        # replace ellipsis with three periods
        text = re.sub(r"[^\S \n]", " ", text)
        # replace multiple spaces with a single space
        text = re.sub(r"  +", " ", text)
        # replace multiple newlines with a single newline
        text = re.sub(r"(?<=\n) +(?=\n)", "", text)
        text = re.sub(r"\bD[Rr]\.(?= [A-Z])", "Doctor", text)
        text = re.sub(r"\b(?:Mr\.|MR\.(?= [A-Z]))", "Mister", text)
        text = re.sub(r"\b(?:Ms\.|MS\.(?= [A-Z]))", "Miss", text)
        text = re.sub(r"\b(?:Mrs\.|MRS\.(?= [A-Z]))", "Mrs", text)
        text = re.sub(r"\betc\.(?! [A-Z])", "etc", text)
        text = re.sub(r"(?i)\b(y)eah?\b", r"\1e'a", text)
        text = re.sub(
            r"\d*\.\d+|\b\d{4}s?\b|(?<!:)\b(?:[1-9]|1[0-2]):[0-5]\d\b(?!:)",
            Tokenizer.split_num,
            text,
        )
        text = re.sub(r"(?<=\d),(?=\d)", "", text)
        text = re.sub(
            r"(?i)[$£]\d+(?:\.\d+)?(?: hundred| thousand| (?:[bm]|tr)illion)*\b|[$£]\d+\.\d\d?\b",
            Tokenizer.flip_money,
            text,
        )
        text = re.sub(r"\d*\.\d+", Tokenizer.point_num, text)
        text = re.sub(r"(?<=\d)-(?=\d)", " to ", text)
        text = re.sub(r"(?<=\d)S", " S", text)
        text = re.sub(r"(?<=[BCDFGHJ-NP-TV-Z])'?s\b", "'S", text)
        text = re.sub(r"(?<=X')S\b", "s", text)
        text = re.sub(
            r"(?:[A-Za-z]\.){2,} [a-z]", lambda m: m.group().replace(".", "-"), text
        )
        text = re.sub(r"(?i)(?<=[A-Z])\.(?=[A-Z])", "-", text)
        return text.strip()

    def tokenize(self, phonemes):
        if len(phonemes) > MAX_PHONEME_LENGTH:
            raise ValueError(
                f"text is too long, must be less than {MAX_PHONEME_LENGTH} phonemes"
            )
        return [i for i in map(VOCAB.get, phonemes) if i is not None]

    def phonemize(self, text, lang="en-us", norm=True) -> str:
        """
        lang can be 'en-us' or 'en-gb'
        """
        if norm:
            text = Tokenizer.normalize_text(text)

        phonemes = phonemizer.phonemize(
            text, lang, preserve_punctuation=True, with_stress=True
        )

        # https://en.wiktionary.org/wiki/kokoro#English
        phonemes = phonemes.replace("kəkˈoːɹoʊ", "kˈoʊkəɹoʊ").replace(
            "kəkˈɔːɹəʊ", "kˈəʊkəɹəʊ"
        )
        phonemes = (
            phonemes.replace("ʲ", "j")
            .replace("r", "ɹ")
            .replace("x", "k")
            .replace("ɬ", "l")
        )
        phonemes = re.sub(r"(?<=[a-zɹː])(?=hˈʌndɹɪd)", " ", phonemes)
        phonemes = re.sub(r' z(?=[;:,.!?¡¿—…"«»“” ]|$)', "z", phonemes)
        if lang == "en-us":
            phonemes = re.sub(r"(?<=nˈaɪn)ti(?!ː)", "di", phonemes)
        phonemes = "".join(filter(lambda p: p in VOCAB, phonemes))
        return phonemes.strip()
