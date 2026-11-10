"""LinnetTTS: lightweight on-device text-to-speech.

The heavy lifting happens in the native extension ``linnet_tts._native``
(the Rust runtime). This package wraps it in a small NumPy-friendly API::

    import linnet_tts, soundfile as sf

    model = linnet_tts.Model("linnet-en-35m.fp16.safetensors", "tokenizer.model")
    voice = model.load_voice("voice.npy")
    audio = model.synthesize("Hello there.", voice)          # float32, 24 kHz
    sf.write("hello.wav", audio, model.sample_rate)

    for chunk in model.session(voice).primed("Streaming works too."):
        play(chunk)                                          # 80 ms per chunk
"""

from __future__ import annotations

import os
from typing import Iterator, Optional, Union

import numpy as np

from . import _native

__version__: str = _native.__version__
__all__ = ["Model", "Voice", "Session", "__version__"]

_PathLike = Union[str, "os.PathLike[str]"]  # str or pathlib.Path


def _to_array(data: bytes) -> np.ndarray:
    # frombuffer views the bytes read-only; copy so callers get a writable array.
    return np.frombuffer(data, dtype=np.float32).copy()



def _threads(threads: int) -> int:
    """Validate a thread count: 0 = automatic, otherwise >= 1 (explicit values above 32 are capped by the runtime)."""
    t = int(threads)
    if t < 0:
        raise ValueError(f"threads must be >= 0 (0 = automatic), got {t}")
    return t

class Voice:
    """Voice prompt latents, a ``[frames, dim]`` float32 matrix, plus the prepared state for one model
    once ``Model.prepare_voice`` (or ``Model.load_voice``) ran. Shareable between threads and sessions."""

    __slots__ = ("_v",)

    def __init__(self, native: "_native.Voice"):
        self._v = native

    @classmethod
    def load(cls, path: _PathLike) -> "Voice":
        """Read a ``.npy`` file (format 1/2/3, C order, float32 or float64)."""
        return cls(_native.Voice.load(os.fspath(path)))

    @classmethod
    def from_array(cls, latents: np.ndarray) -> "Voice":
        """Wrap an in-memory ``[frames, dim]`` array (converted to float32, C order)."""
        arr = np.ascontiguousarray(latents, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"voice latents must be 2-D [frames, dim], got shape {arr.shape}")
        frames, dim = arr.shape
        return cls(_native.Voice.from_bytes(arr.tobytes(), frames, dim))

    @property
    def frames(self) -> int:
        return self._v.frames

    @property
    def dim(self) -> int:
        return self._v.dim

    def to_array(self) -> np.ndarray:
        """The latents as a new ``[frames, dim]`` float32 array."""
        return _to_array(self._v.to_bytes()).reshape(self.frames, self.dim)

    def __repr__(self) -> str:
        return repr(self._v)


class Session:
    """One utterance in progress.

    ``prime(text)`` runs the prefix; ``step()`` returns the next chunk of
    24 kHz float32 audio or ``None`` when the utterance is over. The object
    is also an iterator over the chunks. Per ``docs/streaming-contract.md``
    the first chunk is ``samples_per_step - output_delay`` samples long,
    later chunks ``samples_per_step``, and the last one ``output_delay``
    zeros, so ``np.concatenate(list(session))`` equals ``Model.synthesize``.
    """

    __slots__ = ("_s",)

    def __init__(self, native: "_native.Session"):
        self._s = native

    def prime(self, text: str) -> "Session":
        """Tokenize ``text`` and run the prefix. Resets any previous utterance.

        The voice part of the prefix stays in the session from its first prime
        (or from a prepared voice), so a re-prime runs only the text rows.
        """
        self._s.prime(text)
        return self

    primed = prime  # fluent alias: ``for chunk in model.session(voice).primed(text)``

    def prime_tokens(self, tokens) -> "Session":
        """Run the prefix from ready-made SentencePiece token ids."""
        self._s.prime_tokens([int(t) for t in tokens])
        return self

    def step(self) -> Optional[np.ndarray]:
        """Next chunk of audio, or ``None`` once the utterance is finished."""
        data = self._s.step()
        return None if data is None else _to_array(data)

    def latent(self) -> np.ndarray:
        """Codec-space latent of the most recent step (``[latent_dim]``)."""
        return _to_array(self._s.latent())

    @property
    def frames(self) -> int:
        """LM steps emitted so far."""
        return self._s.frames

    @property
    def finished(self) -> bool:
        return self._s.finished

    def __iter__(self) -> Iterator[np.ndarray]:
        return self

    def __next__(self) -> np.ndarray:
        chunk = self.step()
        if chunk is None:
            raise StopIteration
        return chunk


class Model:
    """Loaded weights plus tokenizer. Immutable and shareable between threads."""

    __slots__ = ("_m",)

    def __init__(self, weights_path: _PathLike, tokenizer_path: _PathLike):
        self._m = _native.Model(os.fspath(weights_path), os.fspath(tokenizer_path))

    @property
    def sample_rate(self) -> int:
        """Output sample rate in Hz (24000)."""
        return self._m.sample_rate

    @property
    def samples_per_step(self) -> int:
        """Samples per LM step (1920 = 80 ms at 24 kHz)."""
        return self._m.samples_per_step

    @property
    def latent_dim(self) -> int:
        return self._m.latent_dim

    @property
    def output_delay(self) -> int:
        """Decoder delay compensated by the session, in samples (720)."""
        return self._m.output_delay

    def tokenize(self, text: str) -> list:
        return list(self._m.tokenize(text))

    @property
    def threads(self) -> int:
        """Threads the model uses (1 = single-threaded)."""
        return self._m.threads

    def set_threads(self, threads: int) -> None:
        """Use ``threads`` threads: 1 = single-threaded, 0 = automatic (fast cores, at most 4).

        The audio is identical for every value; only the speed changes.
        """
        self._m.set_threads(_threads(threads))

    def load_voice(self, path: _PathLike) -> Voice:
        """Load a ``.npy`` voice file, check its width against the model and prepare it (``prepare_voice``)."""
        return Voice(self._m.load_voice(os.fspath(path)))

    def prepare_voice(self, voice: Voice) -> Voice:
        """Run ``voice`` through the model once and keep the result in the voice.

        Sessions created from it for this model (``session``, ``synthesize``) then
        skip the voice part of the prime pass: the first chunk costs the text rows
        plus one step instead of the whole prefix. The audio is identical. The
        state is tied to this model instance (about 2 MB per voice): a voice holds
        one state, so preparing it for another model replaces the state and the
        first model falls back to the latents. ``load_voice`` does this for you
        (and therefore runs the model), ``Voice.load`` and ``Voice.from_array``
        do not.
        """
        self._m.prepare_voice(voice._v)
        return voice

    def is_prepared(self, voice: Voice) -> bool:
        """True when ``voice`` was prepared for this model."""
        return self._m.is_prepared(voice._v)

    def synthesize(
        self,
        text: str,
        voice: Voice,
        temp: float = 0.3,
        cfg: float = 2.0,
        max_frames: int = 300,
        eos_threshold: float = -1.0,
        seed: int = 0,
        threads: int = 0,
    ) -> np.ndarray:
        """Whole utterance as a float32 array at ``sample_rate``.

        ``threads``: 1 = single-threaded, 0 = automatic; the result does not depend on it.
        """
        data = self._m.synthesize(
            text,
            voice._v,
            temp=temp,
            cfg=cfg,
            max_frames=max_frames,
            eos_threshold=eos_threshold,
            seed=seed,
            threads=_threads(threads),
        )
        return _to_array(data)

    def session(
        self,
        voice: Voice,
        temp: float = 0.3,
        cfg: float = 2.0,
        max_frames: int = 300,
        eos_threshold: float = -1.0,
        seed: int = 0,
        threads: int = 0,
    ) -> Session:
        """Streaming session for ``voice``; call ``prime(text)`` then iterate."""
        return Session(
            self._m.session(
                voice._v,
                temp=temp,
                cfg=cfg,
                max_frames=max_frames,
                eos_threshold=eos_threshold,
                seed=seed,
                threads=_threads(threads),
            )
        )

    def decode(self, latents: np.ndarray) -> np.ndarray:
        """Offline decode of ``[T, latent_dim]`` codec latents (delay compensated)."""
        arr = np.ascontiguousarray(latents, dtype=np.float32)
        return _to_array(self._m.decode(arr.tobytes()))

    def __repr__(self) -> str:
        return repr(self._m)
