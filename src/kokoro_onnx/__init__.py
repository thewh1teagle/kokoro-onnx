import asyncio
import contextlib
import importlib
import importlib.metadata
import json
import platform
import time
from collections.abc import AsyncGenerator

import numpy as np
import onnxruntime as rt
from numpy.typing import NDArray

from .chunker import pause_after, split_phonemes
from .config import MAX_PHONEME_LENGTH, SAMPLE_RATE, EspeakConfig, KoKoroConfig
from .log import log
from .pauses import insert as insert_pauses
from .session import create_session, embedded_vocab, input_dtypes
from .sliding import Timing, synthesize, timings, token_edges
from .tokenizer import Tokenizer
from .trim import trim as trim_audio


class Kokoro:
    def __init__(
        self,
        model_path: str,
        voices_path: str,
        espeak_config: EspeakConfig | None = None,
        vocab_config: dict | str | None = None,
    ):
        # Show useful information for bug reports
        log.debug(
            f"koko-onnx version {importlib.metadata.version('kokoro-onnx')} on {platform.platform()} {platform.version()}"
        )
        self._setup(
            session=create_session(model_path),
            model_path=model_path,
            voices_path=voices_path,
            espeak_config=espeak_config,
            vocab_config=vocab_config,
        )

    @classmethod
    def from_session(
        cls,
        session: rt.InferenceSession,
        voices_path: str,
        espeak_config: EspeakConfig | None = None,
        vocab_config: dict | str | None = None,
    ):
        instance = cls.__new__(cls)
        instance._setup(
            session=session,
            model_path=session._model_path,
            voices_path=voices_path,
            espeak_config=espeak_config,
            vocab_config=vocab_config,
        )
        return instance

    def _setup(
        self,
        session: rt.InferenceSession,
        model_path: str,
        voices_path: str,
        espeak_config: EspeakConfig | None,
        vocab_config: dict | str | None,
    ) -> None:
        self.config = KoKoroConfig(model_path, voices_path, espeak_config)
        self.config.validate()

        self.sess = session
        self.voices: np.ndarray = np.load(voices_path)
        # An explicit config wins, then the one the model was exported with,
        # and the vocabulary shipped with the package is the last resort
        vocab = self._load_vocab(vocab_config) or embedded_vocab(session)
        self.tokenizer = Tokenizer(espeak_config, vocab=vocab)

        self._input_dtypes = input_dtypes(session)
        # Older exports name the token input "tokens", newer ones "input_ids"
        self._tokens_input = (
            "input_ids" if "input_ids" in self._input_dtypes else "tokens"
        )
        # Only exports that report durations can place phonemes in time
        self.has_timings = "duration" in {o.name for o in session.get_outputs()}
        vocab = self.tokenizer.vocab
        self._stops = frozenset(vocab[mark] for mark in ".!?" if mark in vocab)
        self._spaces = frozenset(vocab[mark] for mark in " " if mark in vocab)

    def _load_vocab(self, vocab_config: dict | str | None) -> dict:
        """Load vocabulary from config file or dictionary.

        Args:
            vocab_config: Path to vocab config file or dictionary containing vocab.

        Returns:
            Loaded vocabulary dictionary or empty dictionary if no config provided.
        """

        if isinstance(vocab_config, str):
            with open(vocab_config, encoding="utf-8") as fp:
                config = json.load(fp)
                return config["vocab"]
        if isinstance(vocab_config, dict):
            return vocab_config["vocab"]
        return {}

    def _speed_value(self, speed: float) -> float:
        """Adapt the requested speed to what the graph accepts."""
        dtype = self._input_dtypes.get("speed", np.dtype(np.float32))
        if dtype.kind in "iu" and float(speed) != int(speed):
            rounded = max(1, round(speed))
            log.warning(
                f"This model was exported with an integer speed input, "
                f"rounding speed {speed} to {rounded}"
            )
            return rounded
        return speed

    def _infer(
        self, tokens: list[int], style: NDArray[np.float32], speed: float
    ) -> tuple[NDArray[np.float32], NDArray[np.int64] | None]:
        """Run the model on one window of tokens, with its frame durations."""
        start_t = time.time()
        inputs = {
            self._tokens_input: np.array(
                [[0, *tokens, 0]], dtype=self._input_dtypes[self._tokens_input]
            ),
            "style": np.asarray(style, dtype=self._input_dtypes["style"]),
            "speed": np.array(
                [self._speed_value(speed)], dtype=self._input_dtypes["speed"]
            ),
        }

        outputs = self.sess.run(None, inputs)
        audio = np.asarray(outputs[0]).ravel()
        duration = np.asarray(outputs[1]).ravel() if self.has_timings else None

        audio_duration = len(audio) / SAMPLE_RATE
        create_duration = time.time() - start_t
        rtf = create_duration / audio_duration
        log.debug(
            f"Created audio in length of {audio_duration:.2f}s for {len(tokens)} phonemes in {create_duration:.2f}s (RTF: {rtf:.2f}"
        )
        return audio, duration

    def _style_for(
        self, voice: NDArray[np.float32], length: int
    ) -> NDArray[np.float32]:
        """One style vector per phoneme count, so n phonemes use row n - 1."""
        return voice[min(length, len(voice)) - 1]

    def _create_audio(
        self, phonemes: str, voice: NDArray[np.float32], speed: float
    ) -> tuple[NDArray[np.float32], int]:
        log.debug(f"Phonemes: {phonemes}")
        tokens = self.tokenizer.tokenize(phonemes)
        if not tokens:
            raise ValueError(f"No phonemes of {phonemes!r} are in the model vocabulary")
        audio, _ = self._infer(tokens, self._style_for(voice, len(tokens)), speed)
        return audio, SAMPLE_RATE

    def get_voice_style(self, name: str) -> NDArray[np.float32]:
        return self.voices[name]

    def _split_phonemes(self, phonemes: str) -> list[str]:
        """
        Split phonemes into batches of at most MAX_PHONEME_LENGTH.
        Prefer splitting at punctuation marks, then at word boundaries.
        """
        return split_phonemes(phonemes, MAX_PHONEME_LENGTH)

    def _prepare(
        self,
        text: str,
        voice: str | NDArray[np.float32],
        speed: float,
        lang: str,
        is_phonemes: bool,
        sentence_pause: float,
        clause_pause: float,
    ) -> tuple[NDArray[np.float32], str, list[tuple[str, float]]]:
        """Resolve the voice style and split the text into batches and pauses."""
        if not 0.5 <= speed <= 2.0:
            raise ValueError(f"Speed should be between 0.5 and 2.0, got {speed}")

        if isinstance(voice, str):
            if voice not in self.voices:
                raise ValueError(f"Voice {voice} not found in available voices")
            voice = self.get_voice_style(voice)

        phonemes = text if is_phonemes else self.tokenizer.phonemize(text, lang)
        # Newlines are not in the vocabulary, so without this the lines of a
        # text run together with no gap for the model to pause on
        phonemes = " ".join(phonemes.split())
        batched_phonemes = self._split_phonemes(phonemes)
        if not batched_phonemes:
            raise ValueError(f"Nothing to synthesize, {text!r} produced no phonemes")

        log.debug(
            f"Creating audio for {len(batched_phonemes)} batches for {len(phonemes)} phonemes"
        )
        # Trimming removes the silence the model puts at a batch end, so the
        # pause the punctuation calls for is added back when joining
        batches = [
            (batch, pause_after(batch, sentence_pause, clause_pause))
            for batch in batched_phonemes[:-1]
        ]
        return voice, phonemes, [*batches, (batched_phonemes[-1], 0.0)]

    def _create_batch(
        self,
        phonemes: str,
        voice: NDArray[np.float32],
        speed: float,
        trim: bool,
        pause: float = 0.0,
    ) -> tuple[NDArray[np.float32], NDArray[np.int64] | None]:
        """Synthesize one batch, trimming silence and pausing after when trimming."""
        tokens = self.tokenizer.tokenize(phonemes)
        if not tokens:
            raise ValueError(f"No phonemes of {phonemes!r} are in the model vocabulary")

        audio, duration = self._infer(
            tokens, self._style_for(voice, len(tokens)), speed
        )
        # Drop the leading pad boundary so index i is where phoneme i starts
        edges = token_edges(duration, len(audio))[1:] if duration is not None else None

        if trim:
            # Trim leading and trailing silence for a more natural sound concatenation
            # (initial ~2s, subsequent ~0.02s)
            trimmed, (head, _) = trim_audio(audio)
            audio = trimmed
            if edges is not None:
                edges = np.clip(edges - head, 0, len(audio))
            if pause:
                silence = np.zeros(int(pause * SAMPLE_RATE), dtype=audio.dtype)
                audio = np.concatenate([audio, silence])
        return audio, edges

    def create(
        self,
        text: str,
        voice: str | NDArray[np.float32],
        speed: float = 1.0,
        lang: str = "en-us",
        is_phonemes: bool = False,
        trim: bool = True,
        sentence_pause: float = 0.25,
        clause_pause: float = 0.1,
        continuous: bool = False,
    ) -> tuple[NDArray[np.float32], int]:
        """
        Create audio from text using the specified voice and speed.
        """
        audio, sample_rate, _ = self.create_timed(
            text,
            voice,
            speed,
            lang,
            is_phonemes,
            trim,
            sentence_pause,
            clause_pause,
            continuous,
        )
        return audio, sample_rate

    def create_timed(
        self,
        text: str,
        voice: str | NDArray[np.float32],
        speed: float = 1.0,
        lang: str = "en-us",
        is_phonemes: bool = False,
        trim: bool = True,
        sentence_pause: float = 0.25,
        clause_pause: float = 0.1,
        continuous: bool = False,
    ) -> tuple[NDArray[np.float32], int, list[Timing]]:
        """
        Create audio and report when each phoneme is spoken.

        The timings are empty unless the model exposes a duration output.
        With `continuous` the text is synthesized as overlapping windows, which
        keeps prosody running across the joins but costs around 1.4x the work.
        """
        start_t = time.time()
        voice, phonemes, batches = self._prepare(
            text, voice, speed, lang, is_phonemes, sentence_pause, clause_pause
        )

        if continuous:
            audio, spoken = self._create_continuous(phonemes, voice, speed)
        else:
            audio, spoken = self._create_batches(batches, voice, speed, trim)

        # With timings a pause can be placed after every mark; without them
        # only the batch joins carry one, added while the batches were made
        audio, spoken = insert_pauses(
            audio, spoken, SAMPLE_RATE, sentence_pause, clause_pause
        )
        log.debug(f"Created audio in {time.time() - start_t:.2f}s")
        return audio, SAMPLE_RATE, spoken

    def _create_batches(
        self,
        batches: list[tuple[str, float]],
        voice: NDArray[np.float32],
        speed: float,
        trim: bool,
    ) -> tuple[NDArray[np.float32], list[Timing]]:
        """Synthesize each batch on its own and join the results."""
        parts, spoken, offset = [], [], 0
        for phonemes, pause in batches:
            audio, edges = self._create_batch(phonemes, voice, speed, trim, pause)
            if edges is not None:
                spoken += timings(
                    self.tokenizer.known(phonemes), edges + offset, SAMPLE_RATE
                )
            parts.append(audio)
            offset += len(audio)
        return np.concatenate(parts), spoken

    def _create_continuous(
        self, phonemes: str, voice: NDArray[np.float32], speed: float
    ) -> tuple[NDArray[np.float32], list[Timing]]:
        """Synthesize the whole text as overlapping windows."""
        if not self.has_timings:
            raise ValueError(
                "continuous synthesis needs a model that reports durations, "
                "export one with scripts/export.py"
            )

        tokens = self.tokenizer.tokenize(phonemes, limit=None)
        style = self._style_for(voice, len(tokens))
        audio, edges = synthesize(
            lambda window: self._infer(window, style, speed),
            tokens,
            sample_rate=SAMPLE_RATE,
            stops=self._stops,
            spaces=self._spaces,
        )
        return audio, timings(self.tokenizer.known(phonemes), edges, SAMPLE_RATE)

    async def create_stream(
        self,
        text: str,
        voice: str | NDArray[np.float32],
        speed: float = 1.0,
        lang: str = "en-us",
        is_phonemes: bool = False,
        trim: bool = True,
        sentence_pause: float = 0.25,
        clause_pause: float = 0.1,
    ) -> AsyncGenerator[tuple[NDArray[np.float32], int], None]:
        """
        Stream audio creation asynchronously in the background, yielding chunks as they are processed.

        Abandoning the generator stops the stream; the batch already running in
        the worker thread finishes, no further batch is started.
        """
        voice, _, batches = self._prepare(
            text, voice, speed, lang, is_phonemes, sentence_pause, clause_pause
        )

        # One slot for the batch being played, one for the batch synthesized
        # ahead of it, so the producer cannot run away with the whole text
        queue: asyncio.Queue[tuple[NDArray[np.float32], int] | Exception | None] = (
            asyncio.Queue(maxsize=1)
        )

        async def process_batches() -> None:
            """Process phoneme batches in the background."""
            loop = asyncio.get_running_loop()
            try:
                for i, (phonemes, pause) in enumerate(batches):
                    # Execute in separate thread since it's blocking operation
                    audio, _ = await loop.run_in_executor(
                        None, self._create_batch, phonemes, voice, speed, trim, pause
                    )
                    log.debug(f"Processed chunk {i} of stream")
                    await queue.put((audio, SAMPLE_RATE))
            except asyncio.CancelledError:
                raise
            except Exception as error:
                # Hand the failure to the consumer instead of leaving it waiting
                await queue.put(error)
            else:
                await queue.put(None)  # Signal the end of the stream

        task = asyncio.create_task(process_batches())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    def get_voices(self) -> list[str]:
        return sorted(self.voices.keys())
