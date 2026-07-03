from app.dataset.dataset_manager import DatasetManager
from app.extractor.frame_extractor import FrameExtractor


dm = DatasetManager("The Finals")

extractor = FrameExtractor(dm)

video = "test.mp4"

events = [
    12.5,
    25.0,
    40.2
]

count = extractor.extract_from_video(
    video_path=video,
    event_timestamps=events,
    confidence=0.55,
    model_version="v1"
)

print("Extrahierte Frames:", count)