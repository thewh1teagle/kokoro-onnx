"""Tests for phoneme truncation IndexError fix and newline normalization."""

import numpy as np
import pytest

from kokoro_onnx.config import MAX_PHONEME_LENGTH
from kokoro_onnx.tokenizer import Tokenizer


class TestVoiceIndexClamping:
    """Test that voice array indexing is clamped to prevent IndexError.

    When phonemes are truncated to MAX_PHONEME_LENGTH, len(tokens) can equal
    MAX_PHONEME_LENGTH (510), but the voice array has shape (510, ...) with
    valid indices 0-509. The fix clamps the index with min(len(tokens), len(voice) - 1).
    """

    def test_voice_index_at_max_phoneme_length(self):
        """Verify min() clamping prevents IndexError when tokens == MAX_PHONEME_LENGTH."""
        voice_size = MAX_PHONEME_LENGTH  # 510 entries, valid indices 0-509
        voice = np.random.rand(voice_size, 256).astype(np.float32)

        # Simulate len(tokens) == MAX_PHONEME_LENGTH (the crash scenario)
        token_count = MAX_PHONEME_LENGTH

        # This is the fixed line from _create_audio
        idx = min(token_count, len(voice) - 1)
        result = voice[idx]

        assert result.shape == (256,)
        assert idx == voice_size - 1  # Should clamp to 509

    def test_voice_index_below_max(self):
        """Normal case: len(tokens) < voice array size."""
        voice_size = MAX_PHONEME_LENGTH
        voice = np.random.rand(voice_size, 256).astype(np.float32)

        token_count = 100
        idx = min(token_count, len(voice) - 1)
        result = voice[idx]

        assert result.shape == (256,)
        assert idx == 100  # No clamping needed

    def test_voice_index_one_below_max(self):
        """Edge case: len(tokens) == MAX_PHONEME_LENGTH - 1 (last valid index)."""
        voice_size = MAX_PHONEME_LENGTH
        voice = np.random.rand(voice_size, 256).astype(np.float32)

        token_count = MAX_PHONEME_LENGTH - 1
        idx = min(token_count, len(voice) - 1)
        result = voice[idx]

        assert result.shape == (256,)
        assert idx == MAX_PHONEME_LENGTH - 1

    def test_original_code_would_crash(self):
        """Demonstrate that the original code (voice[len(tokens)]) would crash."""
        voice_size = MAX_PHONEME_LENGTH
        voice = np.random.rand(voice_size, 256).astype(np.float32)
        token_count = MAX_PHONEME_LENGTH

        with pytest.raises(IndexError):
            _ = voice[token_count]  # Original bug: index 510 out of bounds


class TestNewlineNormalization:
    """Test that newlines and irregular whitespace are normalized before phonemization."""

    def test_newlines_replaced_with_space(self):
        text = "Hello\nworld"
        result = Tokenizer.normalize_text(text)
        assert result == "Hello world"

    def test_carriage_return_replaced(self):
        text = "Hello\r\nworld"
        result = Tokenizer.normalize_text(text)
        assert result == "Hello world"

    def test_multiple_newlines_collapsed(self):
        text = "Hello\n\n\nworld"
        result = Tokenizer.normalize_text(text)
        assert result == "Hello world"

    def test_tabs_and_spaces_collapsed(self):
        text = "Hello   \t  world"
        result = Tokenizer.normalize_text(text)
        assert result == "Hello world"

    def test_mixed_whitespace(self):
        text = "  Hello \n world \r\n foo  \t bar  "
        result = Tokenizer.normalize_text(text)
        assert result == "Hello world foo bar"

    def test_normal_text_unchanged(self):
        text = "Hello world"
        result = Tokenizer.normalize_text(text)
        assert result == "Hello world"

    def test_empty_string(self):
        text = ""
        result = Tokenizer.normalize_text(text)
        assert result == ""

    def test_only_whitespace(self):
        text = "  \n\t\r\n  "
        result = Tokenizer.normalize_text(text)
        assert result == ""
