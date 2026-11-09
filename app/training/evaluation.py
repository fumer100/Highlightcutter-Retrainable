"""
Faire Modell-Bewertung vor dem Live-Schalten.

Das neue Modell (best.pt) ersetzt current.pt nur, wenn es auf DEMSELBEN festen Pruefset
nicht schlechter ist als das bisherige Modell. Pruefset = Golden Set (wird nie trainiert);
solange es zu klein ist, ersatzweise das aktuelle val-Set (weniger aussagekraeftig).
"""

from pathlib import Path

import yaml

from app.dataset.dataset_manager import DatasetManager, count_boxes_per_class

GOLDEN_MIN_IMAGES = 30  # darunter ist das Golden Set zu klein -> Fallback auf val
MAX_OVERALL_DROP = 0.005  # so viel mAP50 darf das neue Modell gegenueber dem alten verlieren (Rauschen)
MAX_CLASS_DROP = 0.15  # so viel AP50 darf eine einzelne Klasse verlieren ...
MIN_CLASS_INSTANCES = 10  # ... sofern sie im Pruefset mindestens so viele Boxen hat (sonst zu verrauscht)


def choose_eval_set(dm: DatasetManager) -> tuple[str, Path, Path]:
    """Gibt (Name, Bilder-Ordner, Label-Ordner) des Pruefsets zurueck."""
    if len(list(dm.golden_images.glob("*.jpg"))) >= GOLDEN_MIN_IMAGES:
        return "golden", dm.golden_images, dm.golden_labels
    return "val", dm.val_images, dm.val_labels


def write_eval_yaml(dm: DatasetManager, name: str, images_dir: Path) -> str:
    """YAML, das den Pruefset sowohl als train als auch als val angibt (Ultralytics braucht beides)."""
    rel = images_dir.relative_to(dm.dataset).as_posix()
    path = dm.dataset / f"eval_{name}.yaml"
    path.write_text(
        yaml.safe_dump({
            "path": str(dm.dataset),
            "train": rel,
            "val": rel,
            "names": dict(enumerate(dm.game.classes)),
        }, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return str(path)


def evaluate_weights(weights: Path, data_yaml: str, img_size: int, device) -> dict:
    """mAP, Precision/Recall und AP50 pro Klasse eines Modells auf dem Pruefset."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    metrics = model.val(data=data_yaml, imgsz=img_size, device=device, verbose=False, plots=False)
    names = getattr(model, "names", {}) or {}

    per_class = {
        str(names.get(int(cls_id), cls_id)): float(ap)
        for cls_id, ap in zip(metrics.box.ap_class_index, metrics.box.ap50)
    }
    return {
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "class_names": [str(names[i]) for i in sorted(names)],
        "per_class_ap50": per_class,
    }


def instances_per_class(dm: DatasetManager, labels_dir: Path) -> dict[str, int]:
    classes = dm.game.classes
    return {
        (classes[i] if i < len(classes) else f"cls{i}"): n
        for i, n in count_boxes_per_class(labels_dir).items()
    }


def decide(candidate: dict, baseline: dict | None, instances: dict[str, int]) -> tuple[bool, str]:
    """Entscheidet, ob das neue Modell das bisherige ersetzen darf. Gibt (akzeptiert, Begruendung) zurueck."""
    if baseline is None:
        return True, "Kein Vergleichsmodell vorhanden."
    if candidate["class_names"] != baseline["class_names"]:
        return True, "Klassen haben sich geaendert - kein fairer Vergleich moeglich."

    cand, base = candidate["map50"], baseline["map50"]
    if cand < base - MAX_OVERALL_DROP:
        return False, f"mAP50 sinkt von {base:.3f} auf {cand:.3f}."

    for name, base_ap in baseline["per_class_ap50"].items():
        if instances.get(name, 0) < MIN_CLASS_INSTANCES:
            continue
        cand_ap = candidate["per_class_ap50"].get(name, 0.0)
        if base_ap - cand_ap > MAX_CLASS_DROP:
            return False, f"Klasse '{name}' faellt von AP50 {base_ap:.2f} auf {cand_ap:.2f}."

    return True, f"mAP50 {base:.3f} -> {cand:.3f}."


def format_comparison(candidate: dict, baseline: dict | None, instances: dict[str, int]) -> str:
    """Lesbare Tabelle (AP50 pro Klasse, bisher vs. neu) fuers Log."""
    names = sorted(set(candidate["per_class_ap50"]) | set((baseline or {}).get("per_class_ap50", {})))
    lines = [f"{'Klasse':<26}{'Boxen':>6}{'bisher':>9}{'neu':>9}"]
    for name in names:
        old = (baseline or {}).get("per_class_ap50", {}).get(name)
        new = candidate["per_class_ap50"].get(name)
        lines.append(
            f"{name:<26}{instances.get(name, 0):>6}"
            f"{('-' if old is None else f'{old:.2f}'):>9}{('-' if new is None else f'{new:.2f}'):>9}"
        )
    base_map = "-" if baseline is None else f"{baseline['map50']:.3f}"
    lines.append(f"{'mAP50 gesamt':<32}{base_map:>9}{candidate['map50']:>9.3f}")
    return "\n".join(lines)
