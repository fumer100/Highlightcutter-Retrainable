from pathlib import Path
import json
import cv2
import torch

from ultralytics import YOLO

from app.dataset.dataset_manager import DatasetManager
from app.dataset.sample_metadata import SampleMetadata


class AutoAnnotator:

    """
    Fuehrt YOLO auf Frames aus und erzeugt Pseudo-Labels.
    Entscheidet anhand der SCHWAECHSTEN Box im Bild:
    - min_conf >= confidence_threshold          -> automatisch ins Training (train/val)
    - review_threshold <= min_conf < threshold  -> Review Queue (unsicher)
    - keine Box >= review_threshold             -> verworfen (nichts erkannt)
    """

    def __init__(
        self,
        dataset_manager: DatasetManager,
        model_path: str,
        confidence_threshold: float = 0.85,
        review_threshold: float = 0.25
    ):

        self.dm = dataset_manager

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path).to(self.device)

        self.confidence_threshold = confidence_threshold

        self.review_threshold = review_threshold


    def annotate_sample(
        self,
        sample_id: str,
        model_version: str = "v1",
        auto_threshold: float | None = None,
        min_confidence: float | None = None,
        keep_empty: bool = False,
    ):
        auto_threshold = self.confidence_threshold if auto_threshold is None else auto_threshold
        min_confidence = self.review_threshold if min_confidence is None else min_confidence

        image_path = self.dm.review_images / f"{sample_id}.jpg"
        label_path = self.dm.review_labels / f"{sample_id}.txt"
        meta_path = self.dm.review_metadata / f"{sample_id}.json"

        if not image_path.exists():
            return None

        results = self.model(image_path, conf=min_confidence, verbose=False)[0]
        boxes = results.boxes

        confidences = []
        label_lines = []

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x, y, w, h = box.xywhn[0].tolist()
                confidences.append(conf)
                label_lines.append(f"{cls_id} {x} {y} {w} {h}")

        if not label_lines and keep_empty:
            # Fruehes Modell erkennt noch nichts: Bild ungelabelt zur Review, damit der Nutzer selbst labelt
            return {
                "sample_id": sample_id,
                "confidence": 0.0,
                "max_confidence": 0.0,
                "outcome": "review",
                "needs_review": True,
                "label_count": 0,
                "split": None,
            }

        if not label_lines:
            # Nichts erkannt: kein Trainingswert, nicht in die Review Queue
            for p in (image_path, label_path, meta_path):
                p.unlink(missing_ok=True)
            return {
                "sample_id": sample_id,
                "confidence": 0.0,
                "max_confidence": 0.0,
                "outcome": "discarded",
                "needs_review": False,
                "label_count": 0,
                "split": None,
            }

        min_conf = min(confidences)
        max_conf = max(confidences)
        outcome = "auto" if min_conf >= auto_threshold else "review"

        label_path.write_text("\n".join(label_lines))

        # Metadata laden (wurde vom FrameExtractor bereits angelegt) und updaten
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta_data = json.load(f)
        else:
            meta_data = {"sample_id": sample_id}

        meta_data["confidence"] = min_conf
        meta_data["max_confidence"] = max_conf
        meta_data["model_version"] = model_version
        meta_data["reviewed"] = outcome == "auto"
        meta_data["accepted"] = outcome == "auto"

        split = None
        if outcome == "auto":
            # Hohe Konfidenz -> direkt ins Trainingsset (inkl. val-Anteil)
            img_dst, label_dst = self.dm.route_promoted_sample(image_path, label_path)
            split = "val" if img_dst.parent == self.dm.val_images else "train"
            meta_data["image_path"] = str(img_dst)
            meta_data["label_path"] = str(label_dst)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=4, ensure_ascii=False)

        return {
            "sample_id": sample_id,
            "confidence": min_conf,
            "max_confidence": max_conf,
            "outcome": outcome,
            "needs_review": outcome == "review",
            "label_count": len(label_lines),
            "split": split,
        }
