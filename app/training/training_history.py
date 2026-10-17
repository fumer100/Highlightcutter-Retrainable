import json
import time
from pathlib import Path

from app.dataset.dataset_manager import DatasetManager

HISTORY_FILE_NAME = "training_history.json"


def history_path(model_path: Path) -> Path:
    return Path(model_path).parent / HISTORY_FILE_NAME


def compute_dataset_snapshot(dm: DatasetManager) -> dict:
    """Aktueller Datenstand eines Spiels: Bilder in train/val, Boxen insgesamt und pro Klasse."""
    train = sorted(p.name for p in dm.train_images.glob("*.jpg"))
    val = sorted(p.name for p in dm.val_images.glob("*.jpg"))

    per_class: dict[int, int] = {}
    with_boxes = 0
    for label_dir in (dm.train_labels, dm.val_labels):
        for label in label_dir.glob("*.txt"):
            has_box = False
            for line in label.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.split()
                if not parts:
                    continue
                try:
                    cls_id = int(float(parts[0]))
                except ValueError:
                    continue
                per_class[cls_id] = per_class.get(cls_id, 0) + 1
                has_box = True
            with_boxes += has_box

    classes = dm.game.classes
    return {
        "train": len(train),
        "val": len(val),
        "total": len(train) + len(val),
        "with_boxes": with_boxes,
        "boxes": sum(per_class.values()),
        "per_class": {
            (classes[i] if i < len(classes) else f"cls{i}"): n for i, n in sorted(per_class.items())
        },
        "images": train + val,
    }


def load_history(model_path: Path) -> list[dict]:
    path = history_path(model_path)
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def append_history(model_path: Path, snapshot: dict, epochs: int, first_model: bool):
    history = load_history(model_path)
    history.append({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "epochs": epochs,
        "first_model": first_model,
        "snapshot": snapshot,
    })
    path = history_path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")


def build_training_overview(dm: DatasetManager, min_first_labeled: int) -> dict:
    """Vergleich 'letztes Training' vs. 'jetzt' fuer die Anzeige vor dem Trainingsstart."""
    now = compute_dataset_snapshot(dm)
    history = load_history(dm.game.model_path)
    last_entry = history[-1] if history else None
    last = last_entry["snapshot"] if last_entry else None

    added = removed = None
    if last is not None:
        before, current = set(last.get("images", [])), set(now["images"])
        added, removed = len(current - before), len(before - current)

    first_model = not dm.game.model_path.is_file()
    problem = None
    if first_model and (now["with_boxes"] < min_first_labeled or now["val"] < 1):
        problem = (
            f"Fuer das erste Modell werden mindestens {min_first_labeled} gelabelte Bilder mit Boxen benoetigt "
            f"(aktuell {now['with_boxes']}, davon {now['val']} in val)."
        )

    def strip(s):
        return None if s is None else {k: v for k, v in s.items() if k != "images"}

    return {
        "game": dm.game.name,
        "firstModel": first_model,
        "now": strip(now),
        "last": strip(last),
        "lastTimestamp": last_entry["timestamp"] if last_entry else None,
        "added": added,
        "removed": removed,
        "reviewPending": len(list(dm.review_images.glob("*.jpg"))),
        "trainingCount": len(history),
        "problem": problem,
    }
