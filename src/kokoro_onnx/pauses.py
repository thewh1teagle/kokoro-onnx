"""
Silence at sentence and clause boundaries, placed with the model's timings.

The model leaves its own gap after a full stop, but not always a long one:
around 0.1s between the lines of a dialogue, which runs them together. Knowing
when every phoneme is spoken, the gap after a mark can be topped up to what the
text asks for, wherever the mark falls.
"""

import numpy as np
from numpy.typing import NDArray

from .chunker import CLAUSE_MARKS, SENTENCE_MARKS
from .sliding import Timing

# A frame this far below the loudest frame counts as part of a pause. Measuring
# against the loudest sample instead reads speech as near silence and pads gaps
# that were already long enough.
_QUIET_DB = -40
_FRAME = 0.01
# The model usually renders the gap a mark causes just before the mark's own
# timing ends, sometimes just after, so the pause is looked for on both sides
_REACH = 0.15


def wanted_after(phoneme: str, sentence: float, clause: float) -> float:
    """How long a pause the text asks for after this phoneme."""
    if phoneme in SENTENCE_MARKS:
        return sentence
    if phoneme in CLAUSE_MARKS:
        return clause
    return 0.0


def _quiet_frames(audio: NDArray[np.float32], frame: int) -> NDArray[np.bool_]:
    """Which frames of the audio are quiet enough to be part of a pause."""
    usable = len(audio) // frame * frame
    loudness = np.sqrt((audio[:usable].reshape(-1, frame) ** 2).mean(1))
    return loudness <= float(loudness.max()) * 10 ** (_QUIET_DB / 20)


def _run_around(quiet: NDArray[np.bool_], at: int, reach: int) -> tuple[int, int]:
    """The quiet run touching frame `at`, as (start, length).

    Only the run the mark itself sits in counts. Taking the longest run nearby
    instead lands the pause in the gap before the mark, lengthening the wrong
    silence and leaving the one the reader hears untouched.
    """
    inside = next(
        (
            index
            for index in sorted(
                range(at - reach, at + reach + 1), key=lambda i: abs(i - at)
            )
            if 0 <= index < len(quiet) and quiet[index]
        ),
        None,
    )
    if inside is None:
        return at, 0

    start = inside
    while start > 0 and quiet[start - 1]:
        start -= 1
    end = inside
    while end + 1 < len(quiet) and quiet[end + 1]:
        end += 1
    return start, end - start + 1


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

    frame = max(1, int(_FRAME * sample_rate))
    quiet = _quiet_frames(audio, frame)
    reach = int(_REACH / _FRAME)

    parts: list[NDArray[np.float32]] = []
    moved: list[Timing] = []
    cut, shift = 0, 0.0

    for timing in timings:
        moved.append(Timing(timing.phoneme, timing.start + shift, timing.end + shift))

        target = wanted_after(timing.phoneme, sentence, clause)
        if not target:
            continue

        at = int(timing.end * sample_rate) // frame
        start, length = _run_around(quiet, at, reach)
        missing = target - length * _FRAME

        # Splice inside the silence the model left. Cutting into sound to force
        # a gap is audible, and a mark the model ran straight through is a mark
        # it did not mean to stop on
        if not length or missing <= 0:
            continue

        middle = (start + length // 2) * frame
        parts += [
            audio[cut:middle],
            np.zeros(int(missing * sample_rate), dtype=audio.dtype),
        ]
        cut = middle
        shift += missing

    if not parts:
        return audio, timings

    parts.append(audio[cut:])
    return np.concatenate(parts), moved
