from pathlib import Path
import json


class GameConfig:

    def __init__(self, config_file="config/games.json"):

        config_file = Path(config_file)

        if not config_file.exists():
            raise FileNotFoundError(config_file)

        with open(config_file, encoding="utf-8") as f:
            self.data = json.load(f)

    def get(self, game_name: str):

        games = self.data["games"]

        if game_name not in games:
            raise ValueError(f"{game_name} existiert nicht.")

        return games[game_name]