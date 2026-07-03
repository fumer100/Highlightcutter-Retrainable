from pathlib import Path
from Lib import json
import cv2

from ultralytics import YOLO

from app.dataset.dataset_manager import DatasetManager
from app.dataset.sample_metadata import SampleMetadata


class AutoAnnotator:

    """
    Führt YOLO auf Frames aus und erzeugt Pseudo-Labels.
    Entscheidet automatisch:
    - Train (high confidence)
    - Review (low confidence)
    """

    def __init__(
        self,
        dataset_manager: DatasetManager,
        model_path: str,
        confidence_threshold: float = 0.6,
        review_threshold: float = 0.4
    ):

        self.dm = dataset_manager

        self.model = YOLO(model_path)

        self.confidence_threshold = confidence_threshold

        self.review_threshold = review_threshold


    def annotate_sample(self, sample_id: str, model_version: str = "v1"):
        image_path = self.dm.review_images / f"{sample_id}.jpg"
        label_path = self.dm.review_labels / f"{sample_id}.txt"
        meta_path = self.dm.review_metadata / f"{sample_id}.json"

        if not image_path.exists():
            return None

        results = self.model(image_path, verbose=False)[0]
        boxes = results.boxes

        best_conf = 0.0
        label_lines = []

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x, y, w, h = box.xywhn[0].tolist()
                best_conf = max(best_conf, conf)
                label_lines.append(f"{cls_id} {x} {y} {w} {h}")

        # Label in-place ueberschreiben (gleiche sample_id wie beim Extrahieren)
        label_path.write_text("\n".join(label_lines))

        needs_review = (not label_lines) or (best_conf < self.confidence_threshold)

        # Metadata laden (wurde vom FrameExtractor bereits angelegt) und updaten
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta_data = json.load(f)
        else:
            meta_data = {"sample_id": sample_id}

        meta_data["confidence"] = best_conf
        meta_data["model_version"] = model_version
        meta_data["reviewed"] = not needs_review
        meta_data["accepted"] = not needs_review

        if not needs_review:
            # Hohe Konfidenz -> direkt ins Trainingsset verschieben
            img_dst = self.dm.train_images / image_path.name
            label_dst = self.dm.train_labels / label_path.name
            image_path.rename(img_dst)
            label_path.rename(label_dst)
            meta_data["image_path"] = str(img_dst)
            meta_data["label_path"] = str(label_dst)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=4, ensure_ascii=False)

        return {
            "sample_id": sample_id,
            "confidence": best_conf,
            "needs_review": needs_review,
            "label_count": len(label_lines),
        }