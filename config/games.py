from dataclasses import dataclass
from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parent.parent
GAMES_CONFIG_FILE = Path(__file__).resolve().parent / "games.json"


@dataclass(frozen=True)
class Game:
    name: str
    model_path: Path
    dataset_path: Path
    review_queue_path: Path
    classes: list[str]


REQUIRED_FIELDS = ["model_path", "dataset_path", "review_queue_path", "classes"]


class GamesConfigError(ValueError):
    """Wird bei ungueltiger games.json ausgeloest, mit klarer Fehlermeldung."""
    pass


def _validate_game_entry(name: str, cfg: dict):
    if not isinstance(cfg, dict):
        raise GamesConfigError(
            f"Spiel '{name}': Eintrag muss ein Objekt sein, ist aber {type(cfg).__name__}."
        )

    missing = [field for field in REQUIRED_FIELDS if field not in cfg]
    if missing:
        raise GamesConfigError(
            f"Spiel '{name}': fehlende Felder in games.json: {', '.join(missing)}"
        )

    for field in ["model_path", "dataset_path", "review_queue_path"]:
        value = cfg[field]
        if not isinstance(value, str) or not value.strip():
            raise GamesConfigError(
                f"Spiel '{name}': Feld '{field}' muss ein nicht-leerer Text sein, "
                f"ist aber {value!r}."
            )

    classes = cfg["classes"]
    if not isinstance(classes, list) or len(classes) == 0:
        raise GamesConfigError(
            f"Spiel '{name}': 'classes' muss eine nicht-leere Liste sein, "
            f"ist aber {classes!r}."
        )

    for i, cls_name in enumerate(classes):
        if not isinstance(cls_name, str) or not cls_name.strip():
            raise GamesConfigError(
                f"Spiel '{name}': Klasse an Position {i} ist ungueltig: {cls_name!r} "
                f"(muss nicht-leerer Text sein)."
            )

    if len(classes) != len(set(classes)):
        duplicates = [c for c in classes if classes.count(c) > 1]
        raise GamesConfigError(
            f"Spiel '{name}': doppelte Klassennamen gefunden: {set(duplicates)}"
        )


def _load_games() -> dict[str, Game]:
    if not GAMES_CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"Spiele-Konfiguration nicht gefunden: {GAMES_CONFIG_FILE}"
        )

    with open(GAMES_CONFIG_FILE, encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise GamesConfigError(
                f"games.json ist kein gueltiges JSON: {e}"
            ) from e

    if "games" not in data or not isinstance(data["games"], dict):
        raise GamesConfigError(
            "games.json muss ein Objekt 'games' mit mindestens einem Eintrag enthalten."
        )

    if len(data["games"]) == 0:
        raise GamesConfigError(
            "games.json enthaelt keine Spiele unter 'games'."
        )

    games = {}
    for name, cfg in data["games"].items():
        _validate_game_entry(name, cfg)

        games[name] = Game(
            name=name,
            model_path=PROJECT_ROOT / cfg["model_path"],
            dataset_path=PROJECT_ROOT / cfg["dataset_path"],
            review_queue_path=PROJECT_ROOT / cfg["review_queue_path"],
            classes=cfg["classes"],
        )

    return games


GAMES = _load_games()


def reload_games():
    """Erlaubt Neuladen der Konfiguration zur Laufzeit, ohne die App neu zu starten."""
    global GAMES
    GAMES = _load_games()
    return GAMES