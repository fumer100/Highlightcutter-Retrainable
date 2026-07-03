from app.dataset.dataset_manager import DatasetManager

manager = DatasetManager("The Finals")

print(manager.create_sample_id())

print(manager.get_statistics())