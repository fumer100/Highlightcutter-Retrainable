from pathlib import Path
import shutil
import time
import torch
from ultralytics import YOLO

from app.dataset.dataset_manager import DatasetManager
from app.training import evaluation


class Trainer:

    """
    Trainiert YOLO Modelle aus dem aktuellen Dataset
    und ersetzt automatisch das Live-Modell.
    """

    def __init__(
        self,
        dataset_manager: DatasetManager,
        model_path: str,
        base_weights: str = "yolo26n.pt"
    ):

        self.dm = dataset_manager

        self.model_path = Path(model_path)

        # Startgewichte fuers ERSTE Modell eines Spiels (current.pt existiert noch nicht)
        self.base_weights = base_weights

        self.version_dir = self.model_path.parent / "versions"

        self.version_dir.mkdir(parents=True, exist_ok=True)

        # Ergebnis der letzten Bewertung (Golden-Vergleich), None solange nicht trainiert wurde
        self.last_evaluation: dict | None = None

    # -------------------------
    # MAIN TRAIN FUNCTION
    # -------------------------

    def train(self, epochs: int = 20, img_size: int = 640, cancel_event=None):

        print("🚀 Training gestartet...")

        self.last_evaluation = None

        dataset_yaml = self._create_dataset_yaml()

        if self.model_path.exists():
            model = YOLO(self.model_path)
        else:
            print(f"[Training] Erstes Modell: starte von {self.base_weights}")
            model = YOLO(self.base_weights)

        def stop_if_cancelled(_trainer):
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("Training abgebrochen.")

        if cancel_event is not None:
            model.add_callback("on_train_epoch_end", stop_if_cancelled)

        device = 0 if torch.cuda.is_available() else "cpu"  # 0 = erste CUDA-GPU, explizit statt Auto-Erkennung

        results = model.train(
            data=dataset_yaml,
            epochs=epochs,
            imgsz=img_size,
            device=device,
            project=str(self.version_dir),
            name=f"run_{int(time.time())}"
        )

        best_model = Path(results.save_dir) / "weights" / "best.pt"
        self.last_evaluation = self._evaluate_candidate(best_model, img_size, device)

        if self.last_evaluation["accepted"]:
            self._update_model(results)
            print("✅ Training abgeschlossen")
        else:
            print(f"[Training] Training abgeschlossen, aber neues Modell ABGELEHNT: {self.last_evaluation['reason']}")
            print(f"   Das bisherige Modell bleibt aktiv ({self.model_path}).")

        return results

    # -------------------------
    # BEWERTUNG: NEUES VS. BISHERIGES MODELL
    # -------------------------

    def _evaluate_candidate(self, best_model: Path, img_size: int, device) -> dict:
        """Vergleicht best.pt mit current.pt auf dem festen Pruefset und entscheidet ueber das Live-Schalten."""
        if not best_model.exists():
            return {"accepted": False, "reason": "best.pt wurde nicht erzeugt.", "eval_set": None}

        eval_set, images_dir, labels_dir = evaluation.choose_eval_set(self.dm)
        image_count = len(list(images_dir.glob("*.jpg")))
        print(f"[Bewertung] Pruefset: {eval_set} ({image_count} Bilder)"
              + (" - Golden Set noch zu klein, Fallback auf val" if eval_set == "val" else ""))

        try:
            data_yaml = evaluation.write_eval_yaml(self.dm, eval_set, images_dir)
            candidate = evaluation.evaluate_weights(best_model, data_yaml, img_size, device)
            baseline = None
            reset = self.dm.baseline_reset_flag.exists()
            if reset:
                print("[Bewertung] Golden Set wurde neu aufgeteilt: dieses Training setzt die Vergleichs-Baseline neu "
                      "(kein Vergleich mit dem bisherigen Modell, es kennt die Golden-Bilder aus dem Training).")
            elif self.model_path.exists():
                try:
                    baseline = evaluation.evaluate_weights(self.model_path, data_yaml, img_size, device)
                except Exception as e:
                    print(f"[Bewertung] Bisheriges Modell nicht bewertbar ({e}) - kein Vergleich.")
        except Exception as e:
            # Fail-open wie im restlichen Projekt: lieber das neue Modell uebernehmen als stillschweigend blockieren
            print(f"[Bewertung] Bewertung fehlgeschlagen ({e}) - neues Modell wird ohne Vergleich uebernommen.")
            return {"accepted": True, "reason": f"Bewertung fehlgeschlagen: {e}", "eval_set": eval_set}

        instances = evaluation.instances_per_class(self.dm, labels_dir)
        accepted, reason = evaluation.decide(candidate, baseline, instances)
        if reset and accepted:
            self.dm.baseline_reset_flag.unlink(missing_ok=True)
        print(evaluation.format_comparison(candidate, baseline, instances))

        return {
            "accepted": accepted,
            "reason": reason,
            "eval_set": eval_set,
            "eval_images": image_count,
            "candidate": candidate,
            "baseline": baseline,
        }

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