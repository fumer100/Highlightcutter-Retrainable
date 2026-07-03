from dataclasses import dataclass, asdict
from pathlib import Path
import json


@dataclass
class SampleMetadata:

    sample_id: str

    game: str

    image_path: str

    label_path: str

    video_name: str

    frame_number: int

    confidence: float

    model_version: str

    reviewed: bool = False

    accepted: bool = False

    source_clip: str = ""


    def save(self, output_file: Path):

        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:

            json.dump(
                asdict(self),
                f,
                indent=4,
                ensure_ascii=False
            )