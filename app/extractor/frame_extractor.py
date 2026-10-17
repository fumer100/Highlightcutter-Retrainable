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

        extracted_samples = []

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

                    extracted_samples.append(sample_id)

                current_frame += 1

        cap.release()

        return extracted_samples

    def extract_frames(
        self,
        video_path: str,
        timestamps: list[float],
        model_version: str = "v1",
        duplicate_threshold: float = 3.0,
        progress_callback=None,
        cancel_event=None,
    ):
        """
        Extrahiert genau EINEN Frame pro Zeitstempel (keine Nachbarframes) und
        ueberspringt Frames, die einem bereits extrahierten Frame dieses Videos
        fast gleichen (mittlere Grauwert-Abweichung auf 32x32 < duplicate_threshold).

        Gibt (sample_ids, duplicates_skipped) zurueck.
        """
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Video konnte nicht geladen werden: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total = len(timestamps)
        kept_thumbs = []
        sample_ids = []
        duplicates = 0

        try:
            for i, t in enumerate(sorted(timestamps)):
                if cancel_event is not None and cancel_event.is_set():
                    raise RuntimeError("Vorgang abgebrochen.")

                frame_number = max(0, int(round(t * fps)))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                success, frame = cap.read()
                if not success:
                    continue

                thumb = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (32, 32)).astype("float32")
                if any(abs(thumb - k).mean() < duplicate_threshold for k in kept_thumbs):
                    duplicates += 1
                else:
                    kept_thumbs.append(thumb)
                    sample_ids.append(self._write_sample(frame, video_path, frame_number, t, model_version))

                if progress_callback is not None:
                    progress_callback("Frames extrahieren", (i + 1) / total * 100)
        finally:
            cap.release()

        return sample_ids, duplicates

    def _write_sample(self, frame, video_path: Path, frame_number: int, timestamp_sec: float,
                      model_version: str, confidence: float = 0.0) -> str:
        from app.dataset.sample_metadata import SampleMetadata

        sample_id = self.dm.create_sample_id()
        image_path = self.dm.review_images / f"{sample_id}.jpg"
        label_path = self.dm.review_labels / f"{sample_id}.txt"

        cv2.imwrite(str(image_path), frame)
        label_path.write_text("")

        SampleMetadata(
            sample_id=sample_id,
            game=self.dm.game.name,
            image_path=str(image_path),
            label_path=str(label_path),
            video_name=video_path.name,
            frame_number=frame_number,
            confidence=confidence,
            model_version=model_version,
            reviewed=False,
            accepted=False,
            timestamp_sec=timestamp_sec,
        ).save(self.dm.review_metadata / f"{sample_id}.json")

        return sample_id
