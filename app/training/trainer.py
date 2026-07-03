from pathlib import Path
import shutil
import time
from ultralytics import YOLO

from app.dataset.dataset_manager import DatasetManager


class Trainer:

    """
    Trainiert YOLO Modelle aus dem aktuellen Dataset
    und ersetzt automatisch das Live-Modell.
    """

    def __init__(
        self,
        dataset_manager: DatasetManager,
        model_path: str
    ):

        self.dm = dataset_manager

        self.model_path = Path(model_path)

        self.version_dir = self.model_path.parent / "versions"

        self.version_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # MAIN TRAIN FUNCTION
    # -------------------------

    def train(self, epochs: int = 20, img_size: int = 640):

        print("🚀 Training gestartet...")

        dataset_yaml = self._create_dataset_yaml()

        model = YOLO(self.model_path)

        results = model.train(
            data=dataset_yaml,
            epochs=epochs,
            imgsz=img_size,
            project=str(self.version_dir),
            name=f"run_{int(time.time())}"
        )

        self._update_model(results)

        print("✅ Training abgeschlossen")

        return results

    # -------------------------
    # DATASET YAML
    # -------------------------

    def _create_dataset_yaml(self):

        dataset_root = self.dm.dataset

        yaml_path = dataset_root / "dataset.yaml"

        classes = self.dm.game.classes

        yaml_content = f"""
path: {dataset_root}
train: images/train
val: images/val

names:
"""

        for i, c in enumerate(classes):
            yaml_content += f"  {i}: {c}\n"

        yaml_path.write_text(yaml_content)

        return str(yaml_path)

    # -------------------------
    # MODEL UPDATE
    # -------------------------

    def _update_model(self, results):

        timestamp = time.strftime("%Y%m%d_%H%M%S")

        backup_path = self.model_path.parent / f"backup_{timestamp}.pt"

        # Backup old model
        if self.model_path.exists():
            shutil.copy(self.model_path, backup_path)

        # Neue best.pt übernehmen
        best_model = Path(results.save_dir) / "weights" / "best.pt"

        if best_model.exists():

            shutil.copy(best_model, self.model_path)

        print(f"📦 Modell aktualisiert: {self.model_path}")