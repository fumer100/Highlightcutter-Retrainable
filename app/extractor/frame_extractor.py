from pathlib import Path
import cv2

from app.dataset.dataset_manager import DatasetManager


class FrameExtractor:

    """
    Extrahiert Frames aus Highlight-Clips
    und schiebt sie direkt in die Review Queue.
    """

    def __init__(
        self,
        dataset_manager: DatasetManager,
        fps_sample_rate: int = 5,
        context_frames: int = 8
    ):

        self.dm = dataset_manager
        self.fps_sample_rate = fps_sample_rate
        self.context_frames = context_frames

    def extract_from_video(
        self,
        video_path: str,
        event_timestamps: list[float],
        model_version: str = "v1",
        confidence: float = 0.0
    ):

        video_path = Path(video_path)

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise RuntimeError(f"Video konnte nicht geladen werden: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)

        frame_count = 0

        extracted = 0

        for event_time in event_timestamps:

            event_frame = int(event_time * fps)

            start_frame = max(0, event_frame - self.context_frames)
            end_frame = event_frame + self.context_frames

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            current_frame = start_frame

            while current_frame <= end_frame:

                success, frame = cap.read()

                if not success:
                    break

                if (current_frame - start_frame) % self.fps_sample_rate == 0:

                    sample_id = self.dm.create_sample_id()

                    image_name = f"{sample_id}.jpg"
                    label_name = f"{sample_id}.txt"

                    image_path = self.dm.review_images / image_name
                    label_path = self.dm.review_labels / label_name

                    cv2.imwrite(str(image_path), frame)

                    # leeres Label erstmal (Pseudo später)
                    label_path.write_text("")

                    # Metadata später erweitern
                    from app.dataset.sample_metadata import SampleMetadata

                    meta = SampleMetadata(
                        sample_id=sample_id,
                        game=self.dm.game.name,
                        image_path=str(image_path),
                        label_path=str(label_path),
                        video_name=video_path.name,
                        frame_number=current_frame,
                        confidence=confidence,
                        model_version=model_version,
                        reviewed=False,
                        accepted=False
                    )

                    meta_path = self.dm.review_metadata / f"{sample_id}.json"
                    meta.save(meta_path)

                    extracted += 1

                current_frame += 1

        cap.release()

        return extracted