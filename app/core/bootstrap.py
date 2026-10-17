from pathlib import Path

import cv2

from app.extractor.frame_extractor import FrameExtractor

BOOTSTRAP_FRAME_TARGET = 100
FIRST_MODEL_MIN_LABELED = 20  # mind. so viele Bilder mit Boxen, damit das erste Training Sinn ergibt


def plan_bootstrap_timestamps(durations: list[float], target: int) -> list[list[float]]:
    """
    Verteilt `target` Frames proportional zur Laenge auf die Videos und gibt pro Video
    gleichmaessig verteilte Zeitstempel zurueck (Anfang/Ende der Videos ausgespart).
    """
    total = sum(durations)
    if total <= 0 or target <= 0:
        return [[] for _ in durations]

    plan = []
    remaining = target
    for i, d in enumerate(durations):
        n = remaining if i == len(durations) - 1 else round(target * d / total)
        n = max(0, min(n, remaining))
        remaining -= n
        plan.append([d * (k + 0.5) / n for k in range(n)] if n else [])
    return plan


def extract_bootstrap_frames(
    extractor: FrameExtractor,
    video_paths: list[str],
    target: int = BOOTSTRAP_FRAME_TARGET,
    duplicate_threshold: float = 3.0,
    progress_callback=None,
    cancel_event=None,
) -> dict:
    """Legt aus den Videos ca. `target` ungelabelte Frames in der Review Queue des Spiels ab."""
    durations = []
    for path in video_paths:
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        cap.release()
        durations.append(frames / fps if fps > 0 else 0.0)

    plan = plan_bootstrap_timestamps(durations, target)
    extracted = 0
    duplicates = 0
    for path, timestamps in zip(video_paths, plan):
        if not timestamps:
            continue
        print(f"[Bootstrap] {Path(path).name}: {len(timestamps)} Frames gleichmaessig verteilt")
        ids, dups = extractor.extract_frames(
            video_path=str(path),
            timestamps=timestamps,
            model_version="bootstrap",
            duplicate_threshold=duplicate_threshold,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )
        extracted += len(ids)
        duplicates += dups
    return {"extracted": extracted, "duplicates": duplicates}
