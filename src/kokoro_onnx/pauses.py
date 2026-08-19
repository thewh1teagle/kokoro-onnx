"""
Silence at sentence and clause boundaries, placed with the model's timings.

The model leaves its own gap after a full stop, but a short one: around 0.1s
between the lines of a dialogue, which runs them together. Knowing when every
phoneme is spoken, the gap after each mark can be topped up to what the text
asks for, wherever the mark falls.
"""

import numpy as np
from numpy.typing import NDArray

from .chunker import CLAUSE_MARKS, SENTENCE_MARKS
from .sliding import Timing

# Anything this far below the loudest sample counts as silence already there
_QUIET_DB = -40
_STEP = 0.01
# A mark can be timed slightly before the gap it causes, so look a little past it
_REACH = 0.3


def wanted_after(phoneme: str, sentence: float, clause: float) -> float:
    """How long a pause the text asks for after this phoneme."""
    if phoneme in SENTENCE_MARKS:
        return sentence
    if phoneme in CLAUSE_MARKS:
        return clause
    return 0.0


def _silence_after(
    audio: NDArray[np.float32], at: int, quiet: float, step: int, reach: int
) -> int:
    """The longest run of near silence just after `at`, in samples.

    The gap a mark causes can start a frame or two after the mark itself ends,
    so the run is looked for in a window rather than required to start at `at`.
    """
    longest = run = 0
    for start in range(at, min(len(audio), at + reach) - step + 1, step):
        frame = audio[start : start + step]
        run = run + step if float(np.sqrt((frame**2).mean())) <= quiet else 0
        longest = max(longest, run)
    return longest


def insert(
    audio: NDArray[np.float32],
    timings: list[Timing],
    sample_rate: int,
    sentence: float,
    clause: float,
) -> tuple[NDArray[np.float32], list[Timing]]:
    """Lengthen the pause after every mark, and move later timings along."""
    if not timings or not (sentence or clause):
        return audio, timings

    quiet = float(np.abs(audio).max()) * 10 ** (_QUIET_DB / 20)
    step = max(1, int(_STEP * sample_rate))
    parts: list[NDArray[np.float32]] = []
    moved: list[Timing] = []
    cut, shift = 0, 0.0

    for timing in timings:
        moved.append(Timing(timing.phoneme, timing.start + shift, timing.end + shift))

        target = wanted_after(timing.phoneme, sentence, clause)
        if not target:
            continue

        at = min(len(audio), int(timing.end * sample_rate))
        reach = int((target + _REACH) * sample_rate)
        missing = target - _silence_after(audio, at, quiet, step, reach) / sample_rate
        if missing <= 0:
            continue

        parts += [
            audio[cut:at],
            np.zeros(int(missing * sample_rate), dtype=audio.dtype),
        ]
        cut = at
        shift += missing

    if not parts:
        return audio, timings

    parts.append(audio[cut:])
    return np.concatenate(parts), moved
