"""Tests for _split_phonemes: over-long segments must be split, not truncated.

Regression test for the silent phoneme truncation described in
https://github.com/thewh1teagle/kokoro-onnx/issues/184 : when a single
segment (no punctuation) was longer than MAX_PHONEME_LENGTH it was passed
downstream unchanged and truncated in _create_audio, dropping everything
past the limit. These tests check that _split_phonemes now keeps every
batch within the limit without losing any phonemes.

They call the pure string logic directly, so no model files are required.
"""

from kokoro_onnx import Kokoro
from kokoro_onnx.config import MAX_PHONEME_LENGTH

# _split_phonemes / _hard_split only use string logic, not any model state,
# so we can invoke them on an uninitialised instance (no model files needed).
_kokoro = object.__new__(Kokoro)
_split = _kokoro._split_phonemes


def _total_len(batches):
    return sum(len(b) for b in batches)


def test_long_unpunctuated_segment_is_split_not_truncated():
    text = "a" * 700
    batches = _split(text)
    assert all(len(b) <= MAX_PHONEME_LENGTH for b in batches)
    assert _total_len(batches) == 700  # nothing dropped


def test_multiple_long_words_without_punctuation():
    text = " ".join(["x" * 200] * 5)  # ~1000 chars, no punctuation
    batches = _split(text)
    assert all(len(b) <= MAX_PHONEME_LENGTH for b in batches)
    assert sum(b.count("x") for b in batches) == 1000


def test_single_token_no_spaces_hard_cut():
    text = "b" * 1200
    batches = _split(text)
    assert all(len(b) <= MAX_PHONEME_LENGTH for b in batches)
    assert _total_len(batches) == 1200


def test_normal_short_text_is_unchanged():
    text = "hello world. how are you? fine thanks."
    batches = _split(text)
    assert all(len(b) <= MAX_PHONEME_LENGTH for b in batches)
    assert batches  # produces at least one batch
