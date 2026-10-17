from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

BATCH_SIZE = 8
CONFIDENCE_BUCKETS = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.01)]
PALETTE = [(60, 76, 231), (219, 152, 52), (113, 204, 46), (18, 156, 243),
           (182, 89, 155), (156, 188, 26), (34, 126, 230), (94, 73, 52)]  # BGR


def _bucket_label(lo: float, hi: float) -> str:
    return f"{lo:.2f}-{min(hi, 1.0):.2f}"


def test_model_on_video(
    model_path: str,
    video_path: str,
    output_path: str,
    confidence: float = 0.4,
    sample_every_n: int = 10,
    max_minutes: float = 0.0,
    progress_callback=None,
    cancel_event=None,
) -> dict:
    """
    Prueft jeden N-ten Frame mit dem Modell, zeichnet alle Erkennungen (Klasse + Konfidenz)
    ein und schreibt daraus ein Video. Gibt Statistiken pro Klasse und Konfidenz zurueck.
    """
    sample_every_n = max(1, int(sample_every_n))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path).to(device)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Video konnte nicht geladen werden: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    if max_minutes and max_minutes > 0:
        total_frames = min(total_frames, int(max_minutes * 60 * fps))
    total_samples = max(1, total_frames // sample_every_n)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), max(1.0, fps / sample_every_n), (width, height))
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Ausgabe-Video konnte nicht erstellt werden: {output_path}")

    scale = max(1.0, height / 720)
    thickness = max(2, int(round(2 * scale)))
    font_scale = 0.6 * scale

    per_class: dict[str, dict] = {}
    buckets = {_bucket_label(lo, hi): 0 for lo, hi in CONFIDENCE_BUCKETS}
    all_confs = []
    frames_checked = 0
    frames_with_detection = 0
    first_times = []

    def process_batch(batch):
        nonlocal frames_checked, frames_with_detection
        frames = [f for _, f in batch]
        results = model.predict(frames, conf=confidence, verbose=False)
        for (frame_idx, frame), r in zip(batch, results):
            t = frame_idx / fps
            frames_checked += 1
            count_here = 0
            for box in r.boxes:
                cls_id = int(box.cls[0])
                name = str(model.names.get(cls_id, cls_id))
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = PALETTE[cls_id % len(PALETTE)]

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                label = f"{name} {conf:.2f}"
                (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                top = max(0, y1 - th - base - 4)
                cv2.rectangle(frame, (x1, top), (x1 + tw + 6, top + th + base + 4), color, -1)
                cv2.putText(frame, label, (x1 + 3, top + th + 1), cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale, (255, 255, 255), max(1, thickness - 1), cv2.LINE_AA)

                stats = per_class.setdefault(name, {"count": 0, "conf_sum": 0.0, "min": 1.0, "max": 0.0})
                stats["count"] += 1
                stats["conf_sum"] += conf
                stats["min"] = min(stats["min"], conf)
                stats["max"] = max(stats["max"], conf)
                all_confs.append(conf)
                for lo, hi in CONFIDENCE_BUCKETS:
                    if lo <= conf < hi:
                        buckets[_bucket_label(lo, hi)] += 1
                        break
                count_here += 1

            if count_here:
                frames_with_detection += 1
                if len(first_times) < 20:
                    first_times.append(round(t, 1))

            cv2.putText(frame, f"t={t:.1f}s  Erkennungen im Bild: {count_here}  (gesamt: {len(all_confs)})",
                        (20, int(40 * scale)), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.2,
                        (0, 0, 0), thickness + 3, cv2.LINE_AA)
            cv2.putText(frame, f"t={t:.1f}s  Erkennungen im Bild: {count_here}  (gesamt: {len(all_confs)})",
                        (20, int(40 * scale)), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.2,
                        (0, 255, 255), thickness, cv2.LINE_AA)
            out.write(frame)

    print(f"[Modelltest] Video: {Path(video_path).name} ({width}x{height} @ {fps:.1f} fps)")
    print(f"[Modelltest] Pruefe jeden {sample_every_n}. Frame, Mindest-Konfidenz {confidence:.2f}")

    try:
        batch = []
        frame_idx = 0
        while frame_idx < total_frames:
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("Vorgang abgebrochen.")
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % sample_every_n == 0:
                batch.append((frame_idx, frame))
                if len(batch) >= BATCH_SIZE:
                    process_batch(batch)
                    batch = []
                    if progress_callback is not None:
                        progress_callback("Modell-Test", min(100.0, frames_checked / total_samples * 100))
            frame_idx += 1
        if batch:
            process_batch(batch)
        if progress_callback is not None:
            progress_callback("Modell-Test", 100.0)
    finally:
        cap.release()
        out.release()

    return {
        "outputPath": output_path,
        "framesChecked": frames_checked,
        "framesWithDetection": frames_with_detection,
        "detections": len(all_confs),
        "avgConfidence": (sum(all_confs) / len(all_confs)) if all_confs else 0.0,
        "classes": [
            {"name": n, "count": s["count"], "avg": s["conf_sum"] / s["count"], "min": s["min"], "max": s["max"]}
            for n, s in sorted(per_class.items(), key=lambda kv: -kv[1]["count"])
        ],
        "buckets": [{"label": k, "count": v} for k, v in buckets.items()],
        "firstTimes": first_times,
        "confidence": confidence,
        "sampleEveryN": sample_every_n,
    }
