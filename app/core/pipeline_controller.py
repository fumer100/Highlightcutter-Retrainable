from pathlib import Path

from app.dataset.dataset_manager import DatasetManager
from app.extractor.frame_extractor import FrameExtractor
from app.pseudo_label.auto_annotator import AutoAnnotator


class PipelineController:

    def __init__(self, game_name: str, model_path: str):
        self.dm = DatasetManager(game_name)
        self.extractor = FrameExtractor(self.dm)
        self.annotator = AutoAnnotator(self.dm, model_path=model_path)

    def run_full_cycle(self, video_path: str, events: list[float], model_version: str = "v1") -> dict:
        video_path = Path(video_path)

        if not events:
            print(f"[Pipeline] Keine Events fuer {video_path.name}, ueberspringe Extraktion.")
            return {"video": video_path.name, "extracted": 0, "annotated": 0, "needs_review": 0}

        extracted_samples = self.extractor.extract_from_video(
            video_path=str(video_path),
            event_timestamps=events,
            model_version=model_version,
        )

        annotated = 0
        needs_review = 0

        for sample_id in extracted_samples:
            result = self.annotator.annotate_sample(
                sample_id=sample_id,
                model_version=model_version,
            )
            if result is None:
                continue
            annotated += 1
            if result["needs_review"]:
                needs_review += 1