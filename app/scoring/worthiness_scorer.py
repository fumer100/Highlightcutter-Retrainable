"""
LLM-basiertes Wertigkeits-Scoring fuer Highlight-Kandidaten-Fenster.

Bewertet strukturierte Fenster-Beschreibungen (YOLO-Events, Audio-Kennzahlen,
Transkript-Ausschnitt, Spiel-Glossar) per LLM und liefert pro Fenster:
  - score:          Gesamt-Wertigkeit (0-100) -> entscheidet rein/raus
  - humor_score:    wie lustig/unterhaltsam (0-100, aus Transkript/Kontext)
  - interest_score: wie spielerisch relevant/spannend (0-100)
  - reason:         kurze Begruendung (Deutsch)

Design-Prinzip: Das LLM entscheidet NUR ueber Wertigkeit, NICHT ueber
Zeitgrenzen - die kommen weiterhin aus der deterministischen Cluster-Logik.

Laeuft standardmaessig komplett LOKAL ueber Ollama (http://localhost:11434),
kein API-Key, keine Internetverbindung, keine Daten verlassen den Rechner.
Cloud-APIs (z.B. OpenAI) sind optional ueber llm_provider="openai" weiterhin
nutzbar, aber nicht der Standard.

Fail-open: Wenn der lokale Ollama-Server nicht laeuft oder die Anfrage
fehlschlaegt, werden neutrale/hohe Scores zurueckgegeben (lieber ein Clip zu
viel als ein guter Moment verloren) und das im Log klar markiert.
"""

import json
import os
import urllib.request

_FALLBACK_SCORE = 100.0
_FALLBACK_REASON = "LLM-Scoring nicht verfuegbar (Ollama nicht erreichbar/Fehler) - Fenster sicherheitshalber behalten."
_DEFAULT_OLLAMA_URL = "http://localhost:11434/v1"

_SYSTEM_PROMPT = """Du bewertest Ausschnitte aus Gaming-Highlight-Videos fuer einen \
automatischen Video-Cutter. Du bekommst pro Fenster strukturierte Fakten \
(Spiel, erkannte Spiel-Events/Trigger, Lautstaerke-Kennzahlen, ein \
Transkript-Ausschnitt der gesprochenen Kommentare). Du bekommst NIE das \
Video oder Audio selbst.

Bewerte pro Fenster:
- score (0-100): Gesamt-Wertigkeit als Highlight. 0 = komplett irrelevant/langweilig, \
100 = eindeutig ein starkes Highlight.
- humor_score (0-100): wie lustig/unterhaltsam (aus dem Transkript, z.B. Lachen, \
witzige Kommentare, Ueberraschung). 0 falls kein Transkript vorhanden.
- interest_score (0-100): wie spielerisch relevant/spannend (basierend auf den \
Spiel-Events, nicht auf Humor).
- reason: EIN kurzer deutscher Satz als Begruendung.

Antworte NUR mit JSON in exakt diesem Format, ohne zusaetzlichen Text:
{"results": [{"index": 0, "score": 0-100, "humor_score": 0-100, "interest_score": 0-100, "reason": "..."}]}

Die Liste "results" muss GENAU so viele Eintraege haben wie Fenster im Input, \
in der gleichen Reihenfolge (index = Position im Input, beginnend bei 0)."""


def _chunked(items: list, size: int):
    for i in range(0, len(items), size):
        yield i, items[i:i + size]


def _fallback_results(count: int) -> list[dict]:
    return [
        {"score": _FALLBACK_SCORE, "humor_score": 0.0, "interest_score": 0.0, "reason": _FALLBACK_REASON}
        for _ in range(count)
    ]


def _ollama_reachable(base_url: str) -> bool:
    host_url = base_url.rsplit("/v1", 1)[0]
    try:
        urllib.request.urlopen(host_url, timeout=2)
        return True
    except Exception:
        return False


def _extract_json(text: str) -> dict:
    """Robust gegen lokale Modelle, die trotz JSON-Modus z.B. Markdown-Codefences
    drumherum bauen (```json ... ```)."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end + 1]
    return json.loads(text)


def _call_llm(client, model: str, payloads: list[dict]) -> list[dict]:
    user_content = json.dumps({"windows": payloads}, ensure_ascii=False)

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )

    data = _extract_json(response.choices[0].message.content)
    raw_results = data.get("results", [])

    def _parse_entry(entry: dict) -> dict:
        return {
            "score": float(entry.get("score", _FALLBACK_SCORE)),
            "humor_score": float(entry.get("humor_score", 0.0)),
            "interest_score": float(entry.get("interest_score", 0.0)),
            "reason": str(entry.get("reason", "")).strip() or "(keine Begruendung)",
        }

    # Lokale Modelle (z.B. llama3.1:8b) halten das "index"-Feld bei groesseren
    # Batches nicht immer zuverlaessig ein (fehlt, doppelt, falsch gezaehlt).
    # Deshalb: erst per Index zuordnen, alles andere danach der Reihe nach
    # (Antwort-Reihenfolge) auf die verbleibenden Luecken verteilen, statt
    # bei jedem kaputten Index sofort in den Fallback zu fallen.
    by_index: dict[int, dict] = {}
    unindexed: list[dict] = []

    for entry in raw_results:
        if not isinstance(entry, dict):
            continue
        try:
            parsed = _parse_entry(entry)
        except (TypeError, ValueError):
            continue

        idx = entry.get("index")
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            idx = None

        if idx is not None and 0 <= idx < len(payloads) and idx not in by_index:
            by_index[idx] = parsed
        else:
            unindexed.append(parsed)

    unindexed_iter = iter(unindexed)
    results = []
    for i in range(len(payloads)):
        if i in by_index:
            results.append(by_index[i])
            continue
        fallback_entry = next(unindexed_iter, None)
        if fallback_entry is not None:
            results.append(fallback_entry)
        else:
            results.append({
                "score": _FALLBACK_SCORE,
                "humor_score": 0.0,
                "interest_score": 0.0,
                "reason": "LLM-Antwort fuer dieses Fenster fehlte/ungueltig - Fallback.",
            })
    return results


def score_windows(
    payloads: list[dict],
    llm_model: str,
    batch_size: int = 10,
    provider: str = "ollama",
    ollama_base_url: str = _DEFAULT_OLLAMA_URL,
) -> list[dict]:
    """
    payloads: Liste strukturierter Fenster-Beschreibungen (siehe
    highlight_cutter_pointbased._build_scoring_payload).

    provider: "ollama" (Standard, komplett lokal, kein API-Key noetig) oder
    "openai" (Cloud, benoetigt OPENAI_API_KEY als Umgebungsvariable).

    Rueckgabe: Liste GLEICHER Laenge/Reihenfolge mit
    {"score", "humor_score", "interest_score", "reason"}.
    """
    if not payloads:
        return []

    try:
        from openai import OpenAI
    except ImportError:
        print("[Scoring] Paket 'openai' nicht installiert - LLM-Scoring uebersprungen.")
        return _fallback_results(len(payloads))

    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("[Scoring] Kein OPENAI_API_KEY gesetzt - LLM-Scoring uebersprungen (Fail-Open, alle Fenster behalten).")
            return _fallback_results(len(payloads))
        client = OpenAI(api_key=api_key)
    else:
        if not _ollama_reachable(ollama_base_url):
            print(f"[Scoring] Ollama-Server unter {ollama_base_url} nicht erreichbar "
                  f"(ist 'ollama serve' / die Ollama-App gestartet?) - Fallback, alle Fenster behalten.")
            return _fallback_results(len(payloads))
        # Ollamas OpenAI-kompatible Schnittstelle ignoriert den API-Key inhaltlich,
        # verlangt aber einen nicht-leeren String.
        client = OpenAI(base_url=ollama_base_url, api_key="ollama-local")

    results: list[dict] = [None] * len(payloads)

    for offset, chunk in _chunked(payloads, max(1, batch_size)):
        try:
            chunk_results = _call_llm(client, llm_model, chunk)
        except Exception as e:
            print(f"[Scoring] LLM-Aufruf fehlgeschlagen ({e}) - Fallback fuer dieses Batch.")
            chunk_results = _fallback_results(len(chunk))

        for i, r in enumerate(chunk_results):
            results[offset + i] = r

    return results
