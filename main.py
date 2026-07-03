from app.core.pipeline import Pipeline


pipeline = Pipeline(
    game_name="The Finals",
    model_path="models/THE FINALS/current.pt"
)

pipeline.run_full_cycle(
    video_path="test.mp4",
    events=[12.5, 25.0, 40.0]
)