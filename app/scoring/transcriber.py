"""
Speech-to-Text via faster-whisper (CTranslate2, GPU-beschleunigt).

Liefert zeitgestempelte Transkript-Segmente. Diese dienen als zusaetzlicher
Kontext fuer das LLM-Wertigkeits-Scoring (Humor/Interessantheit anhand von
Sprache/Kommentar/Lachen-Beschreibung im Transkript), unabhaengig von den
YOLO-/Audio-Peak-Signalen.
"""

from dataclasses import dataclass

import torch

_MODEL_CACHE: dict = {}


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


def _get_model(model_size: str):
    if model_size not in _MODEL_CACHE:
        from faster_whisper import WhisperModel

        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        print(f"[STT] Lade Whisper-Modell '{model_size}' ({device}/{compute_type}) ...")
        _MODEL_CACHE[model_size] = WhisperModel(model_size, device=device, compute_type=compute_type)
    return _MODEL_CACHE[model_size]


def transcribe_audio(
    video_path: str,
    model_size: str = "small",
    language: str | None = None,
    progress_callback=None,
    cancel_event=None,
) -> list[TranscriptSegment]:
    """Transkribiert die Audiospur von video_path. Dekodiert die Audiospur selbst (ffmpeg/av)."""
    model = _get_model(model_size)

    print("[STT] Transkribiere Audiospur ...")
    segments_iter, info = model.transcribe(
        video_path,
        language=language,
        vad_filter=True,
        beam_size=1,
    )

    segments: list[TranscriptSegment] = []
    total_duration = max(info.duration, 1.0)

    for seg in segments_iter:
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("Vorgang abgebrochen.")

        text = seg.text.strip()
        if text:
            segments.append(TranscriptSegment(start=seg.start, end=seg.end, text=text))

        if progress_callback is not None:
            progress_callback("Transkription", min(100.0, seg.end / total_duration * 100))

    print(f"[STT] {len(segments)} Transkript-Segmente erkannt.")
    if progress_callback is not None:
        progress_callback("Transkription", 100.0)
    return segments
