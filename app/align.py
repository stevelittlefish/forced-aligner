"""The alignment core.

Modelled on nightingale's app-core/analyzer/align.py: trim the vocals, feed the
whole known lyric as one segment, run wav2vec2 CTC forced alignment via WhisperX,
group characters into words, and map the words back onto the original lines.

WhisperX is imported lazily inside functions so the module imports (and the app
starts, and `--help`/config errors surface) without a GPU or the heavy stack.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from threading import Lock

from .config import Config


class LanguageUnsupported(Exception):
    """No wav2vec2 model maps to the requested language."""


class AudioDecodeError(Exception):
    """The uploaded bytes could not be decoded as audio (bad/unsupported file).

    Separate from a model/alignment failure so the HTTP layer can answer 415
    (unsupported media) rather than 500 — the spec distinguishes them because a
    bad upload is the caller's problem, not the service's."""


def supported_languages() -> list[str]:
    """Language codes WhisperX has a default wav2vec2 model for. Used to fill the
    422 response body so a caller that sent a bad code learns what *is* possible
    without reading our source. Imported lazily; falls back to [] if the map
    isn't reachable (old WhisperX), which is honest rather than a lie."""
    try:
        from whisperx import alignment as _a

        codes = set()
        for name in ("DEFAULT_ALIGN_MODELS_TORCH", "DEFAULT_ALIGN_MODELS_HF"):
            codes.update(getattr(_a, name, {}).keys())
        return sorted(codes)
    except Exception:
        return []


@dataclass
class Word:
    text: str
    start: float | None
    end: float | None


@dataclass
class Line:
    text: str
    start: float | None
    end: float | None
    words: list[Word]


@dataclass
class AlignResult:
    language: str
    duration: float
    lines: list[Line]


class Aligner:
    """Holds the per-language wav2vec2 model cache. One instance per process.

    Models load lazily on first use for a language and stay resident. Loading is
    guarded by a lock because a GPU model load is not something to run twice
    concurrently for the same language.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._models: dict[str, tuple] = {}  # language -> (model, metadata)
        self._lock = Lock()

    def loaded_languages(self) -> list[str]:
        return sorted(self._models.keys())

    def preload(self, language: str) -> None:
        """Load a language's model now and keep it resident, so the first /align
        for it isn't slow. Just warms the same cache _get_model uses; raises
        LanguageUnsupported if no model maps to the code."""
        self._get_model(language)

    def _get_model(self, language: str):
        with self._lock:
            if language not in self._models:
                import whisperx  # heavy; imported on first real use

                try:
                    model, metadata = whisperx.load_align_model(
                        language_code=language,
                        device=self.cfg.device,
                        model_dir=self.cfg.model_cache_dir or None,
                    )
                except (ValueError, KeyError) as e:
                    # WhisperX raises when no model maps to the language code.
                    raise LanguageUnsupported(str(e)) from e
                self._models[language] = (model, metadata)
            return self._models[language]

    def align(self, audio_path: str, text: str, language: str) -> AlignResult:
        import whisperx

        try:
            audio = whisperx.load_audio(audio_path)  # 16 kHz mono float32
        except Exception as e:
            # ffmpeg couldn't decode it: not an audio file, or a corrupt one.
            raise AudioDecodeError(str(e)) from e
        duration = len(audio) / 16000.0

        start, end = self._sung_region(audio, duration)

        lines_text = [ln.strip() for ln in text.splitlines() if ln.strip()]
        flat = " ".join(lines_text)

        model, metadata = self._get_model(language)

        # One segment spanning the sung region: the aligner sees all the words
        # and all the audio at once and places them monotonically.
        segments = [{"text": flat, "start": start, "end": end}]
        result = None
        try:
            result = whisperx.align(
                segments,
                model,
                metadata,
                audio,
                self.cfg.device,
                return_char_alignments=False,
            )

            aligned = _collect_words(result)
            lines = _map_words_to_lines(lines_text, aligned)
            return AlignResult(language=language, duration=duration, lines=lines)
        finally:
            # A long song spikes wav2vec2 activations into several GB, and
            # PyTorch's caching allocator keeps that reserved after the job. This
            # GPU is shared with Demucs and Stable Audio, and the reserved pool
            # is not visible to those separate processes — so after aligning a
            # ~10 min track the aligner sat at ~9.9 GB and the next Demucs run
            # OOM'd on a full card. Drop this job's big tensors and hand the
            # cache back to the driver, where the co-tenants can claim it. The
            # per-language model cache (self._models) is untouched — only the
            # audio and alignment tensors go, so the next /align is still warm.
            del audio, segments, result
            self._release_vram()

    def _release_vram(self) -> None:
        """Return the caching allocator's free blocks to the CUDA driver so other
        processes on the shared GPU can use them. A no-op off CUDA. empty_cache()
        only releases already-freed cache, so the del above must happen first."""
        if not str(self.cfg.device).startswith("cuda"):
            return
        import gc

        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            # Freeing cache is best-effort housekeeping; never fail a good align
            # because the release hiccuped.
            pass

    def _sung_region(self, audio, duration: float) -> tuple[float, float]:
        """Where singing actually is. A cheap energy-threshold VAD is enough for
        the common case (silence/instrumental lead-in and tail-out); WhisperX's
        own alignment handles gaps within the sung region. Returns the full clip
        when trimming is off or finds nothing."""
        if not self.cfg.vad_trim:
            return 0.0, duration
        import numpy as np

        # 20 ms frames, RMS energy, threshold at a fraction of the peak.
        frame = int(0.02 * 16000)
        if frame <= 0 or len(audio) < frame:
            return 0.0, duration
        n = len(audio) // frame
        frames = audio[: n * frame].reshape(n, frame)
        rms = np.sqrt(np.mean(frames.astype("float32") ** 2, axis=1))
        peak = float(rms.max()) if len(rms) else 0.0
        if peak <= 0:
            return 0.0, duration
        active = np.where(rms > 0.05 * peak)[0]
        if len(active) == 0:
            return 0.0, duration
        start = float(active[0] * frame) / 16000.0
        end = float((active[-1] + 1) * frame) / 16000.0
        # Pad a little so we don't clip the attack of the first/last word.
        start = max(0.0, start - 0.2)
        end = min(duration, end + 0.2)
        return start, end


def _collect_words(result: dict) -> list[Word]:
    """Every aligned word in input-text order. WhisperX groups characters into
    words by a monotonically increasing index, so segment order is text order.
    A word it could not place lacks start/end — we keep it, with None, as the
    honest 'unresolved' signal rather than dropping or guessing it."""
    words: list[Word] = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            text = (w.get("word") or "").strip()
            if not text:
                continue
            s = w.get("start")
            e = w.get("end")
            words.append(
                Word(
                    text=text,
                    start=float(s) if s is not None else None,
                    end=float(e) if e is not None else None,
                )
            )
    return words


def _norm(s: str) -> str:
    return re.sub(r"[^\w]", "", s, flags=re.UNICODE).lower()


def _map_words_to_lines(lines_text: list[str], aligned: list[Word]) -> list[Line]:
    """Walk aligned words back onto the original lines. Each line consumed as
    many aligned words as it has tokens; line start/end come from its first/last
    resolved word. This mirrors nightingale's approach and tolerates the aligner
    dropping punctuation-only tokens by matching on normalized text loosely."""
    lines: list[Line] = []
    i = 0
    for lt in lines_text:
        tokens = [t for t in lt.split() if _norm(t)]
        line_words: list[Word] = []
        for tok in tokens:
            if i < len(aligned):
                aw = aligned[i]
                line_words.append(Word(text=tok, start=aw.start, end=aw.end))
                i += 1
            else:
                line_words.append(Word(text=tok, start=None, end=None))
        starts = [w.start for w in line_words if w.start is not None]
        ends = [w.end for w in line_words if w.end is not None]
        lines.append(
            Line(
                text=lt,
                start=min(starts) if starts else None,
                end=max(ends) if ends else None,
                words=line_words,
            )
        )
    return lines
