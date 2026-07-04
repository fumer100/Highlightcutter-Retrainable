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

    def _find_highest_existing_id(self):

        highest = 0

        for image in self.train_images.glob("*.jpg"):

            try:

                number = int(
                    image.stem.split("_")[-1]
                )

                highest = max(highest, number)

            except:

                pass

        for image in self.review_images.glob("*.jpg"):

            try:

                number = int(
                    image.stem.split("_")[-1]
                )

                highest = max(highest, number)

            except:

                pass

        return highest

    def create_sample_id(self):

        prefix = self.game.name.upper().replace(" ", "_")

        return f"{prefix}_{next(self._counter):08d}"

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
            img_dst = self.val_images / image_path.name
            lbl_dst = self.val_labels / label_path.name
        else:
            img_dst = self.train_images / image_path.name
            lbl_dst = self.train_labels / label_path.name

        image_path.rename(img_dst)
        if label_path.exists():
            label_path.rename(lbl_dst)

        return img_dst, lbl_dst