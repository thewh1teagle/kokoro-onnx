"""Tests for the optional max_phoneme_length cap on _split_phonemes.

Feature request https://github.com/thewh1teagle/kokoro-onnx/issues/130 :
allow smaller streaming batches so the first audio chunk is produced sooner
(lower time-to-first-audio). _split_phonemes now accepts an optional
max_phoneme_length that caps each batch, clamped to the model limit.

These call the pure string logic directly, so no model files are required.
"""

from kokoro_onnx import Kokoro
from kokoro_onnx.config import MAX_PHONEME_LENGTH

# _split_phonemes only uses string logic, not any model state, so we can invoke
# it on an uninitialised instance (no model files needed).
_kokoro = object.__new__(Kokoro)
_split = _kokoro._split_phonemes


def test_default_is_unchanged():
    text = "hello world. this is a test, of the splitter!"
    assert _split(text) == _split(text, None)


def test_smaller_cap_produces_smaller_batches():
    # A comma-separated list gives the splitter natural break points.
    text = ", ".join(["word"] * 40)
    small = _split(text, 20)
    assert small, "expected at least one batch"
    assert all(len(b) <= 20 for b in small)
    # A smaller cap should yield more (smaller) batches than the default.
    assert len(small) >= len(_split(text))


def test_cap_is_clamped_to_model_limit():
    text = ", ".join(["word"] * 40)
    # Asking for more than the model allows must not exceed the model limit.
    batches = _split(text, MAX_PHONEME_LENGTH * 10)
    assert all(len(b) <= MAX_PHONEME_LENGTH for b in batches)


def test_tiny_cap_still_progresses():
    # A cap of 0 / negative would be nonsensical; it is clamped to >= 1 so the
    # splitter always makes progress instead of looping or emptying output.
    text = "abc, def, ghi"
    assert _split(text, 0)
    assert _split(text, -5)


def test_no_phonemes_are_dropped():
    text = ", ".join(["abc"] * 30)
    batches = _split(text, 15)
    # Every "abc" survives the smaller batching (nothing truncated).
    assert sum(b.count("abc") for b in batches) == 30
