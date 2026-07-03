from multiprocessing import freeze_support
from app.core.pipeline import Pipeline


def main():

    pipeline = Pipeline(
        game_name="THE FINALS",
        model_path="models/THE FINALS/current.pt"
    )

    pipeline.run_full_cycle(
        video_path="D:\\OBS CLIPS\\Alle Aufnahmen\\Used\\Test Fight.mp4",
        events=[12.5, 25.0, 40.0]
    )


if __name__ == "__main__":

    freeze_support()  # wichtig für Windows + YOLO

    main()