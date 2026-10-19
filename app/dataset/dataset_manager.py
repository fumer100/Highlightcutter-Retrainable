from pathlib import Path
from itertools import count

from config.games import GAMES


class DatasetManager:

    """
    Verwaltet sämtliche Trainingsdaten.

    Diese Klasse ist die EINZIGE Stelle,
    die Dateien verschiebt oder kopiert.
    """
    VAL_SPLIT_RATIO = 0.1  # Ziel: ~10% der Daten landen in val/
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
            "with_boxes": with_boxes,
            "review": len(list(self.review_images.glob("*.jpg"))),
        }
    
    

    @staticmethod
    def _free_stem(stem: str, targets: list) -> str:
        """Gibt `stem` zurueck oder `stem_dupN`, falls einer der Zieldateinamen schon belegt ist."""
        candidate, n = stem, 1
        while any((folder / f"{candidate}{ext}").exists() for folder, ext in targets):
            n += 1
            candidate = f"{stem}_dup{n}"
        return candidate

    def route_promoted_sample(self, image_path: Path, label_path: Path):
        """
        Verschiebt ein akzeptiertes Review-Sample nach train/ oder val/,
        je nachdem was das aktuelle Verhaeltnis noch braucht.
        """
        train_count = len(list(self.train_images.glob("*.jpg")))
        val_count = len(list(self.val_images.glob("*.jpg")))
        total = train_count + val_count

        target_val_count = round((total + 1) * self.VAL_SPLIT_RATIO)

        if val_count < target_val_count:
            img_dir, lbl_dir = self.val_images, self.val_labels
        else:
            img_dir, lbl_dir = self.train_images, self.train_labels

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