from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Game:

    name: str

    model_path: Path

    dataset_path: Path

    review_queue_path: Path

    classes: list[str]


PROJECT_ROOT = Path(__file__).resolve().parent.parent


GAMES = {

    "The Finals": Game(

        name="The Finals",

        model_path=PROJECT_ROOT / "models" / "THE FINALS" / "current.pt",

        dataset_path=PROJECT_ROOT / "datasets" / "THE FINALS",

        review_queue_path=PROJECT_ROOT / "review_queue" / "THE FINALS",

        classes=[
            "event",
            "hitmarker"
        ]
    )

}