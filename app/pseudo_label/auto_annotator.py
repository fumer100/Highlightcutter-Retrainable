from pathlib import Path
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


    def annotate_frame(
        self,
        image_path: str,
        video_name: str = "",
        frame_number: int = 0,
        model_version: str = "v1"
    ):

        image_path = Path(image_path)

        results = self.model(image_path, verbose=False)[0]

        boxes = results.boxes

        if boxes is None or len(boxes) == 0:
            return None

        best_conf = 0.0

        label_lines = []

        for box in boxes:

            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xywh = box.xywh[0].tolist()

            best_conf = max(best_conf, conf)

            # YOLO format: class x y w h (normalized später optional)
            x, y, w, h = xywh

            label_lines.append(
                f"{cls_id} {x} {y} {w} {h}"
            )

        sample_id = self.dm.create_sample_id()

        label_path = self.dm.review_labels / f"{sample_id}.txt"
        image_target = self.dm.review_images / f"{sample_id}.jpg"

        # Bild kopieren
        image_target.write_bytes(image_path.read_bytes())

        # Label schreiben
        label_path.write_text("\n".join(label_lines))

        # Entscheidung
        if best_conf >= self.confidence_threshold:

            destination = "train"

        elif best_conf >= self.review_threshold:

            destination = "review"

        else:

            destination = "review"

        # Metadata
        meta = SampleMetadata(
            sample_id=sample_id,
            game=self.dm.game.name,
            image_path=str(image_target),
            label_path=str(label_path),
            video_name=video_name,
            frame_number=frame_number,
            confidence=best_conf,
            model_version=model_version,
            reviewed=False,
            accepted=False
        )

        meta.save(self.dm.review_metadata / f"{sample_id}.json")

        return {
            "sample_id": sample_id,
            "confidence": best_conf,
            "destination": destination,
            "label_count": len(label_lines)
        }