"""
Sliding window synthesis for models that report phoneme durations.

Every window is [context][keep][lookahead] and only the keep region is used, so
the cold onset and the sentence-final fall of each pass land in audio that is
discarded. Windows meet at a sentence end, or failing that at the quietest word
gap near the target length, which is where two takes join least audibly.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

KEEP = 300
CONTEXT = 60
LOOKAHEAD = 60
CROSSFADE = 0.025

_STOP_SEARCH = 70
_GAP_SEARCH = 12
_QUIET_WINDOW = 0.02

# Runs the model on one window of tokens, returning its audio and frame durations
Run = Callable[[Sequence[int]], tuple[NDArray[np.float32], NDArray[np.int64]]]


@dataclass(frozen=True)
class Timing:
    """When a phoneme is spoken, in seconds from the start of the audio."""

    phoneme: str
    start: float
    end: float


def token_edges(duration: NDArray[np.int64], samples: int) -> NDArray[np.int64]:
    """Sample offset of every token boundary, the leading pad included."""
    frames = np.concatenate([[0], np.cumsum(duration)])
    return np.round(frames * (samples / frames[-1])).astype(np.int64)


def timings(phonemes: str, edges: NDArray[np.int64], sample_rate: int) -> list[Timing]:
    """Pair each phoneme with the span of audio it occupies."""
    return [
        Timing(
            phoneme, float(edges[i] / sample_rate), float(edges[i + 1] / sample_rate)
        )
        for i, phoneme in enumerate(phonemes)
        if i + 1 < len(edges)
    ]


def _quietest(
    audio: NDArray[np.float32],
    edges: NDArray[np.int64],
    offset: int,
    candidates: list[int],
    sample_rate: int,
) -> int:
    """Of the candidate tokens, the one sitting in the quietest audio."""
    half = int(_QUIET_WINDOW * sample_rate)
    best, quietest = candidates[0], None
    for token in candidates:
        at = edges[token - offset + 1]
        frame = audio[max(0, at - half) : at + half]
        energy = float(np.sqrt((frame**2).mean())) if len(frame) else 1.0
        if quietest is None or energy < quietest:
            best, quietest = token, energy
    return best


def _join(
    pieces: list[NDArray[np.float32]], fade: int
) -> tuple[NDArray[np.float32], list[int]]:
    """Crossfade the kept regions together, reporting where each one starts."""
    audio, starts = pieces[0], [0]
    for piece in pieces[1:]:
        overlap = min(fade, len(audio), len(piece))
        starts.append(len(audio) - overlap)
        if overlap:
            ramp = np.linspace(0, 1, overlap, dtype=np.float32)
            blend = audio[-overlap:] * (1 - ramp) + piece[:overlap] * ramp
            audio = np.concatenate([audio[:-overlap], blend, piece[overlap:]])
        else:
            audio = np.concatenate([audio, piece])
    return audio, starts


def synthesize(
    run: Run,
    tokens: Sequence[int],
    *,
    sample_rate: int,
    stops: frozenset[int] = frozenset(),
    spaces: frozenset[int] = frozenset(),
    keep: int = KEEP,
    context: int = CONTEXT,
    lookahead: int = LOOKAHEAD,
    crossfade: float = CROSSFADE,
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    """Synthesize `tokens` as overlapping windows, keeping only their middles.

    Returns the audio and the sample offset of every token boundary, so the
    result carries exact timings for the whole text.
    """
    pieces: list[NDArray[np.float32]] = []
    spans: list[NDArray[np.int64]] = []
    start = 0

    while start < len(tokens):
        end = min(start + keep, len(tokens))
        first, last = max(0, start - context), min(len(tokens), end + lookahead)
        audio, duration = run(tokens[first:last])
        edges = token_edges(duration, len(audio))

        if end < len(tokens):
            end = _quietest(
                audio,
                edges,
                first,
                _candidates(tokens, stops, spaces, end, start, last),
                sample_rate,
            )

        if end <= start:  # a window that does not advance would loop forever
            raise RuntimeError(f"window {start}:{end} makes no progress")

        begin, finish = edges[start - first + 1], edges[end - first + 1]
        pieces.append(audio[begin:finish])
        spans.append(edges[start - first + 1 : end - first + 2] - begin)
        start = end

    audio, starts = _join(pieces, int(crossfade * sample_rate))
    offsets = [span[:-1] + base for span, base in zip(spans, starts)]
    return audio, np.concatenate([*offsets, [len(audio)]])


def _candidates(
    tokens: Sequence[int],
    stops: frozenset[int],
    spaces: frozenset[int],
    target: int,
    floor: int,
    limit: int,
) -> list[int]:
    """Tokens the window may end on, sentence ends preferred over word gaps.

    A candidate has to lie past `floor`, the first token this window keeps, or
    the window would end before it starts and the walk would never advance.
    """

    def within(reach: int, wanted: frozenset[int], shift: int) -> list[int]:
        reach = min(reach, max(1, target - floor))
        return [
            index + shift
            for index in range(target - reach, target + reach)
            if floor < index + shift < limit and tokens[index] in wanted
        ]

    return within(_STOP_SEARCH, stops, 1) or within(_GAP_SEARCH, spaces, 0) or [target]
