from app.dataset.dataset_manager import DatasetManager
from app.extractor.frame_extractor import FrameExtractor
from app.pseudo_label.auto_annotator import AutoAnnotator
from app.training.trainer import Trainer


class Pipeline:

    """
    Orchestriert das komplette Active Learning System:

    Video → Frames → Pseudo Labels → Review → Training
    """

    def __init__(
        self,
        game_name: str,
        model_path: str
    ):

        self.dm = DatasetManager(game_name)

        self.frame_extractor = FrameExtractor(self.dm)

        self.annotator = AutoAnnotator(
            dataset_manager=self.dm,
            model_path=model_path
        )

        self.trainer = Trainer(
            dataset_manager=self.dm,
            model_path=model_path
        )

    # -----------------------------
    # FULL PIPELINE
    # -----------------------------

    def run_video(self, video_path: str, events: list[float]):

        print("🎬 Pipeline gestartet")

        # 1. Frames extrahieren
        print("📦 Extrahiere Frames...")

        self.frame_extractor.extract_from_video(
            video_path=video_path,
            event_timestamps=events
        )

        # 2. Pseudo Labels erzeugen
        print("🧠 Erstelle Pseudo Labels...")

        review_images = list(self.dm.review_images.glob("*.jpg"))

        for img in review_images:

            self.annotator.annotate_frame(
                image_path=str(img),
                video_name=video_path,
                frame_number=0
            )

        print("📊 Pseudo Labeling abgeschlossen")

        return True

    # -----------------------------
    # TRAINING PIPELINE
    # -----------------------------

    def train_model(self, epochs: int = 20):

        print("🚀 Starte Training Pipeline")

        result = self.trainer.train(epochs=epochs)

        print("✅ Training abgeschlossen")

        return result

    # -----------------------------
    # FULL AUTO MODE
    # -----------------------------

    def run_full_cycle(self, video_path: str, events: list[float]):

        print("🔥 FULL ACTIVE LEARNING CYCLE START")

        self.run_video(video_path, events)

        print("⏳ Warten auf Review Queue (optional manuell)")

        result = self.train_model()

        print("🎯 Cycle abgeschlossen")

        return result