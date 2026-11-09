from pathlib import Path
from itertools import count
import json
import re

from config.games import GAMES

SPLIT_NAMES = ("train", "val", "golden")
# Ziel-Anteile pro Split, gemessen in VIDEOS (nicht Bildern): ein Video gehoert komplett
# zu einem Split, damit sich fast identische Frames nie ueber Train und Val/Golden verteilen.
SPLIT_TARGETS = {"train": 0.8, "val": 0.1, "golden": 0.1}
# Solange weniger Videos bekannt sind, ist keine saubere Aufteilung moeglich (sonst waere z.B. val leer).
# Diese Videos heissen "mixed" und werden wie frueher pro Bild nach VAL_SPLIT_RATIO verteilt.
MIN_VIDEOS_FOR_VIDEO_SPLIT = 5
VIDEO_SPLITS_FILE = "video_splits.json"
_DUP_SUFFIX_RE = re.compile(r"_dup\d+$")


def count_boxes_per_class(label_dir: Path) -> dict[int, int]:
    """Boxen pro Klassen-ID in allen YOLO-Label-Dateien eines Ordners."""
    counts: dict[int, int] = {}
    if not label_dir.is_dir():
        return counts
    for label in label_dir.glob("*.txt"):
        for line in label.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split()
            if not parts:
                continue
            try:
                cls_id = int(float(parts[0]))
            except ValueError:
                continue
            counts[cls_id] = counts.get(cls_id, 0) + 1
    return counts


class DatasetManager:

    """
    Verwaltet sämtliche Trainingsdaten.

    Diese Klasse ist die EINZIGE Stelle,
    die Dateien verschiebt oder kopiert.

    Splits (alle pro VIDEO, siehe SPLIT_TARGETS):
      train   - Trainingsdaten (hier duerfen auch automatische Pseudo-Labels landen)
      val     - Validierung waehrend des Trainings (nur von Menschen geprueft)
      golden  - festes Vergleichs-Set, wird NIE trainiert (liegt ausserhalb von images/train|val)
    """
    VAL_SPLIT_RATIO = 0.1  # Ziel fuer "mixed"-Videos: ~10% der Bilder landen in val/
    def __init__(self, game_name: str):

        if game_name not in GAMES:

            raise ValueError(f"Unbekanntes Spiel: {game_name}")

        self.game = GAMES[game_name]

        self.dataset = self.game.dataset_path

        self.review = self.game.review_queue_path

        self.train_images = self.dataset / "images" / "train"

        self.train_labels = self.dataset / "labels" / "train"

        self.val_images = self.dataset / "images" / "val"

        self.val_labels = self.dataset / "labels" / "val"

        self.golden_images = self.dataset / "golden" / "images"

        self.golden_labels = self.dataset / "golden" / "labels"

        self.review_images = self.review / "images"

        self.review_labels = self.review / "labels"

        self.review_metadata = self.review / "metadata"

        self.ensure_directories()

        self._counter = count(
            self._find_highest_existing_id() + 1
        )

    def ensure_directories(self):

        folders = [

            self.train_images,
            self.train_labels,

            self.val_images,
            self.val_labels,

            self.golden_images,
            self.golden_labels,

            self.review_images,
            self.review_labels,
            self.review_metadata

        ]

        for folder in folders:

            folder.mkdir(
                parents=True,
                exist_ok=True
            )

    @property
    def discarded_dir(self):
        return self.review / "discarded"

    def _scan_ids(self) -> set[int]:
        """Alle bereits vergebenen Nummern: Bilder UND Metadaten in train/val/review/discarded."""
        numbers = set()
        sources = (
            (self.train_images, "*.jpg"),
            (self.val_images, "*.jpg"),
            (self.golden_images, "*.jpg"),
            (self.review_images, "*.jpg"),
            (self.review_metadata, "*.json"),
            (self.discarded_dir, "*.jpg"),
            (self.discarded_dir, "*.json"),
        )
        for folder, pattern in sources:
            if not folder.is_dir():
                continue
            for path in folder.glob(pattern):
                try:
                    numbers.add(int(path.stem.split("_")[-1]))
                except ValueError:
                    pass
        return numbers

    def _find_highest_existing_id(self):
        # Verworfene Samples zaehlen mit, sonst werden IDs nach einem Neustart doppelt vergeben
        return max(self._scan_ids(), default=0)

    def sample_id_in_use(self, sample_id: str) -> bool:
        name = sample_id
        return any(
            p.exists() for p in (
                self.train_images / f"{name}.jpg",
                self.val_images / f"{name}.jpg",
                self.golden_images / f"{name}.jpg",
                self.review_images / f"{name}.jpg",
                self.review_metadata / f"{name}.json",
                self.discarded_dir / f"{name}.jpg",
                self.discarded_dir / f"{name}.json",
            )
        )

    def create_sample_id(self):

        prefix = self.game.name.upper().replace(" ", "_")

        while True:
            sample_id = f"{prefix}_{next(self._counter):08d}"
            if not self.sample_id_in_use(sample_id):
                return sample_id

    def get_statistics(self):

        stats = {

            "train_images":
                len(list(self.train_images.glob("*"))),

            "train_labels":
                len(list(self.train_labels.glob("*"))),

            "review_images":
                len(list(self.review_images.glob("*"))),

            "review_labels":
                len(list(self.review_labels.glob("*"))),

            "validation_images":
                len(list(self.val_images.glob("*"))),

            "validation_labels":
                len(list(self.val_labels.glob("*"))),

            "golden_images":
                len(list(self.golden_images.glob("*"))),

        }

        return stats

    def labeled_stats(self):
        """Anzahl Trainings-/Validierungsbilder und wie viele davon mindestens eine Box haben."""
        with_boxes = 0
        for label_dir in (self.train_labels, self.val_labels):
            for label in label_dir.glob("*.txt"):
                if label.stat().st_size > 0:
                    with_boxes += 1

        return {
            "train": len(list(self.train_images.glob("*.jpg"))),
            "val": len(list(self.val_images.glob("*.jpg"))),
            "golden": len(list(self.golden_images.glob("*.jpg"))),
            "with_boxes": with_boxes,
            "review": len(list(self.review_images.glob("*.jpg"))),
        }

    def class_box_counts(self) -> dict[int, int]:
        """Boxen pro Klassen-ID ueber train, val und golden (fuer die Seltenheit einer Klasse)."""
        counts: dict[int, int] = {}
        for label_dir in (self.train_labels, self.val_labels, self.golden_labels):
            for cls_id, n in count_boxes_per_class(label_dir).items():
                counts[cls_id] = counts.get(cls_id, 0) + n
        return counts

    # -------------------------
    # VIDEO-BASIERTER SPLIT
    # -------------------------

    @property
    def video_splits_path(self) -> Path:
        return self.dataset / VIDEO_SPLITS_FILE

    @property
    def baseline_reset_flag(self) -> Path:
        """Marker: das aktuelle Modell hat Golden-Bilder gesehen und taugt nicht als Vergleichswert."""
        return self.dataset / "golden_baseline_reset.flag"

    def video_key(self, sample_id: str) -> str | None:
        """Quellvideo eines Samples (aus den Metadaten, die im Review-Ordner liegen bleiben) oder None."""
        for name in (sample_id, _DUP_SUFFIX_RE.sub("", sample_id)):
            meta_file = self.review_metadata / f"{name}.json"
            if not meta_file.is_file():
                continue
            try:
                video = json.loads(meta_file.read_text(encoding="utf-8")).get("video_name")
            except (OSError, ValueError):
                return None
            return video or None
        return None

    def _seed_video_splits(self) -> dict[str, str]:
        """Erstellt die Video->Split-Zuordnung aus dem vorhandenen Datenstand."""
        seen: dict[str, set] = {}
        for split, folder in (("train", self.train_images), ("val", self.val_images),
                              ("golden", self.golden_images)):
            for img in folder.glob("*.jpg"):
                key = self.video_key(img.stem)
                if key:
                    seen.setdefault(key, set()).add(split)
        # Videos mit Bildern in mehreren Splits (Altbestand mit Zufalls-Split) bleiben "mixed"
        return {key: (next(iter(s)) if len(s) == 1 else "mixed") for key, s in seen.items()}

    def load_video_splits(self) -> dict[str, str]:
        path = self.video_splits_path
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return {str(k): str(v) for k, v in data.items()}
            except (OSError, ValueError):
                pass
        splits = self._seed_video_splits()
        self.save_video_splits(splits)
        return splits

    def save_video_splits(self, splits: dict[str, str]):
        self.video_splits_path.write_text(
            json.dumps(splits, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )

    @staticmethod
    def choose_split_for_new_video(splits: dict[str, str]) -> str:
        """Waehlt den Split, dessen Video-Anteil am weitesten unter dem Ziel liegt."""
        if len(splits) < MIN_VIDEOS_FOR_VIDEO_SPLIT:
            return "mixed"
        n = len(splits)
        counts = {name: 0 for name in SPLIT_NAMES}
        for value in splits.values():
            # "mixed" (Altbestand) besteht ueberwiegend aus Trainingsdaten
            counts["train" if value == "mixed" else value] += 1
        # Bei Gleichstand: golden vor val vor train, damit die Pruefmengen zuerst entstehen
        return max(("golden", "val", "train"), key=lambda s: SPLIT_TARGETS[s] - counts[s] / n)

    def split_for_sample(self, sample_id: str) -> str:
        """Ziel-Split eines Samples: 'train' | 'val' | 'golden' | 'mixed'. Neue Videos werden hier verteilt."""
        key = self.video_key(sample_id)
        if key is None:
            return "mixed"
        splits = self.load_video_splits()
        if key not in splits:
            splits[key] = self.choose_split_for_new_video(splits)
            self.save_video_splits(splits)
        return splits[key]

    def can_auto_promote(self, sample_id: str) -> bool:
        """Automatische Pseudo-Labels duerfen nur ins Training, nie nach val/golden."""
        return self.split_for_sample(sample_id) in ("train", "mixed")

    def split_dirs(self, split: str) -> tuple[Path, Path]:
        if split == "train":
            return self.train_images, self.train_labels
        if split == "val":
            return self.val_images, self.val_labels
        if split == "golden":
            return self.golden_images, self.golden_labels
        raise ValueError(f"Unbekannter Split: {split}")
    

    @staticmethod
    def _free_stem(stem: str, targets: list) -> str:
        """Gibt `stem` zurueck oder `stem_dupN`, falls einer der Zieldateinamen schon belegt ist."""
        candidate, n = stem, 1
        while any((folder / f"{candidate}{ext}").exists() for folder, ext in targets):
            n += 1
            candidate = f"{stem}_dup{n}"
        return candidate

    def route_promoted_sample(self, image_path: Path, label_path: Path, human_reviewed: bool = True):
        """
        Verschiebt ein akzeptiertes Review-Sample in den Split seines Videos (train/val/golden).
        human_reviewed=False (automatisches Pseudo-Label) ist nur fuer train erlaubt.
        """
        split = self.split_for_sample(image_path.stem)

        if split == "mixed":
            # Altbestand / zu wenige Videos: pro Bild nach VAL_SPLIT_RATIO, Automatik nie nach val
            train_count = len(list(self.train_images.glob("*.jpg")))
            val_count = len(list(self.val_images.glob("*.jpg")))
            target_val_count = round((train_count + val_count + 1) * self.VAL_SPLIT_RATIO)
            split = "val" if human_reviewed and val_count < target_val_count else "train"
        elif not human_reviewed and split != "train":
            raise ValueError(f"Automatisches Label darf nicht nach '{split}' (nur menschlich geprueft).")

        img_dir, lbl_dir = self.split_dirs(split)

        # Bereits belegter Name (z.B. alte doppelte ID): neuen Namen nehmen statt zu scheitern
        stem = self._free_stem(image_path.stem, [(img_dir, ".jpg"), (lbl_dir, ".txt")])
        img_dst = img_dir / f"{stem}.jpg"
        lbl_dst = lbl_dir / f"{stem}.txt"

        image_path.rename(img_dst)
        if label_path.exists():
            label_path.rename(lbl_dst)

        return img_dst, lbl_dst

    def discard_sample(self, sample_id: str) -> Path:
        """Verschiebt Bild/Label/Metadaten eines Review-Samples nach discarded/ (ohne Namenskollision)."""
        self.discarded_dir.mkdir(parents=True, exist_ok=True)
        sources = [
            self.review_images / f"{sample_id}.jpg",
            self.review_labels / f"{sample_id}.txt",
            self.review_metadata / f"{sample_id}.json",
        ]
        stem = self._free_stem(sample_id, [(self.discarded_dir, p.suffix) for p in sources])
        for src in sources:
            if src.exists():
                src.rename(self.discarded_dir / f"{stem}{src.suffix}")
        return self.discarded_dir / f"{stem}.jpg"

    def discard_dataset_sample(self, split: str, sample_id: str) -> Path:
        """Verschiebt ein Bild aus train/val/golden samt Label (und evtl. Review-Metadaten) nach discarded/."""
        img_dir, lbl_dir = self.split_dirs(split)

        self.discarded_dir.mkdir(parents=True, exist_ok=True)
        sources = [
            img_dir / f"{sample_id}.jpg",
            lbl_dir / f"{sample_id}.txt",
            self.review_metadata / f"{sample_id}.json",
        ]
        stem = self._free_stem(sample_id, [(self.discarded_dir, p.suffix) for p in sources])
        for src in sources:
            if src.exists():
                src.rename(self.discarded_dir / f"{stem}{src.suffix}")
        return self.discarded_dir / f"{stem}.jpg"