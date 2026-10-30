import numpy as np
import pytest

from kokoro_onnx.pauses import insert
from kokoro_onnx.sliding import Timing

SAMPLE_RATE = 24000


def speech_with_gap(start: float, end: float) -> np.ndarray:
    audio = np.ones(SAMPLE_RATE, dtype=np.float32)
    audio[int(start * SAMPLE_RATE) : int(end * SAMPLE_RATE)] = 0
    return audio


def test_gap_between_speech_is_topped_up():
    audio = speech_with_gap(0.4, 0.5)
    timings = [Timing(".", 0.4, 0.5), Timing("a", 0.5, 1.0)]

    padded, moved = insert(audio, timings, SAMPLE_RATE, 0.5, 0.1)

    assert len(padded) == len(audio) + int(0.4 * SAMPLE_RATE)
    assert moved[1].start == pytest.approx(0.9)


@pytest.mark.parametrize("marks", [".", "?!", '."'])
def test_marks_after_the_last_spoken_phoneme_add_no_pause(marks):
    audio = speech_with_gap(0.8, 1.0)
    timings = [Timing("a", 0.0, 0.8)]
    timings += [Timing(mark, 0.8, 0.9) for mark in marks]

    padded, moved = insert(audio, timings, SAMPLE_RATE, 0.5, 0.1)

    assert len(padded) == len(audio)
    assert moved == timings
