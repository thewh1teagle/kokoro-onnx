"""
Split a phoneme string into batches the model can synthesize in one pass.

The model context is fixed at MAX_PHONEME_LENGTH phonemes, so long inputs are
cut at the least disruptive boundary available: sentence, clause, word, and
only as a last resort mid-word. Every batch is non-empty and within the limit.
"""

import re

from .config import MAX_PHONEME_LENGTH

SENTENCE_MARKS = ".!?…"
CLAUSE_MARKS = ",;:"

# Least to most disruptive place to cut. Punctuation stays with the text
# before it, which is how the model was trained.
_BOUNDARIES = (
    re.compile(rf"(?<=[{re.escape(SENTENCE_MARKS)}])\s+"),
    re.compile(rf"(?<=[{re.escape(CLAUSE_MARKS)}])\s+"),
    re.compile(r"\s+"),
)


def _atoms(phonemes: str, max_length: int, level: int = 0):
    """Yield pieces of `phonemes`, each no longer than `max_length`."""
    if len(phonemes) <= max_length:
        if phonemes:
            yield phonemes
        return

    for index in range(level, len(_BOUNDARIES)):
        pieces = _BOUNDARIES[index].split(phonemes)
        if len(pieces) > 1:
            for piece in pieces:
                yield from _atoms(piece.strip(), max_length, index + 1)
            return

    # A single unbroken run longer than the context: slice it rather than
    # dropping the tail.
    for start in range(0, len(phonemes), max_length):
        yield phonemes[start : start + max_length]


def _pack(lengths: list[int], limit: int) -> list[tuple[int, int]]:
    """Group consecutive atoms into batches within `limit`, as index ranges."""
    batches: list[tuple[int, int]] = []
    start, size = 0, 0

    for index, length in enumerate(lengths):
        candidate = length if index == start else size + 1 + length
        if candidate > limit and index > start:
            batches.append((start, index))
            start, size = index, length
        else:
            size = candidate

    if lengths:
        batches.append((start, len(lengths)))
    return batches


def split_phonemes(phonemes: str, max_length: int = MAX_PHONEME_LENGTH) -> list[str]:
    """Split `phonemes` into evenly sized batches of at most `max_length`.

    Filling each batch to the limit would leave a short remainder, and a short
    batch is spoken at a different rate and loudness than its neighbours. The
    batches are therefore balanced: the smallest limit that still needs no
    extra pass over the text.
    """
    atoms = list(_atoms(phonemes.strip(), max_length))
    if not atoms:
        return []

    lengths = [len(atom) for atom in atoms]
    fewest = len(_pack(lengths, max_length))

    low, high = max(lengths), max_length
    while low < high:
        middle = (low + high) // 2
        if len(_pack(lengths, middle)) <= fewest:
            high = middle
        else:
            low = middle + 1

    return [" ".join(atoms[start:end]) for start, end in _pack(lengths, low)]


def pause_after(phonemes: str, sentence: float, clause: float) -> float:
    """Seconds of silence a batch ending with this text should be followed by."""
    mark = phonemes.rstrip()[-1:]
    if mark in SENTENCE_MARKS:
        return sentence
    if mark in CLAUSE_MARKS:
        return clause
    return 0.0
