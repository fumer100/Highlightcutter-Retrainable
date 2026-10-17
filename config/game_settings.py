"""
Speichert Schneide-/ML-Einstellungen pro Spiel in config/game_settings.json.

Gespeichert werden nur Werte, die vom Standard abweichen. Spiele ohne Eintrag
laufen mit den Standardwerten der Config-Klasse, ihre Eintraege in games.json
bleiben unangetastet.
"""

import json
import os
import shutil
from pathlib import Path

SETTINGS_FILE = Path(__file__).resolve().parent / "game_settings.json"


def _read_all() -> dict:
    if not SETTINGS_FILE.is_file():
        return {}
    try:
        data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    games = data.get("games") if isinstance(data, dict) else None
    return games if isinstance(games, dict) else {}


def _write_all(games: dict):
    # Defekte Datei vor dem Ueberschreiben sichern, damit keine Einstellungen still verloren gehen
    if SETTINGS_FILE.is_file():
        try:
            json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            shutil.copy2(SETTINGS_FILE, SETTINGS_FILE.with_suffix(".json.corrupt"))

    tmp = SETTINGS_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"games": games}, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, SETTINGS_FILE)


def get_game_settings(name: str) -> dict:
    value = _read_all().get(name)
    return dict(value) if isinstance(value, dict) else {}


def save_game_settings(name: str, values: dict):
    """Leere Werte entfernen den Eintrag des Spiels (= Standardwerte)."""
    games = _read_all()
    if values:
        games[name] = values
    else:
        games.pop(name, None)
    _write_all(games)


def pop_game_settings(name: str) -> dict:
    games = _read_all()
    removed = games.pop(name, None)
    if removed is not None:
        _write_all(games)
    return dict(removed) if isinstance(removed, dict) else {}
