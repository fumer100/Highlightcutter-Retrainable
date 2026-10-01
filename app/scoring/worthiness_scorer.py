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

Fail-open: Wenn kein API-Key gesetzt ist oder die Anfrage fehlschlaegt,
werden neutrale/hohe Scores zurueckgegeben (lieber ein Clip zu viel als ein
guter Moment verloren) und das im Log klar markiert.
"""

import json
import os

_FALLBACK_SCORE = 100.0
_FALLBACK_REASON = "LLM-Scoring nicht verfuegbar (kein API-Key/Fehler) - Fenster sicherheitshalber behalten."

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

    data = json.loads(response.choices[0].message.content)
    raw_results = data.get("results", [])

    by_index = {}
    for entry in raw_results:
        try:
            idx = int(entry["index"])
            by_index[idx] = {
                "score": float(entry.get("score", _FALLBACK_SCORE)),
                "humor_score": float(entry.get("humor_score", 0.0)),
                "interest_score": float(entry.get("interest_score", 0.0)),
                "reason": str(entry.get("reason", "")).strip() or "(keine Begruendung)",
            }
        except (KeyError, TypeError, ValueError):
            continue

    results = []
    for i in range(len(payloads)):
        results.append(by_index.get(i, {
            "score": _FALLBACK_SCORE,
            "humor_score": 0.0,
            "interest_score": 0.0,
            "reason": "LLM-Antwort fuer dieses Fenster fehlte/ungueltig - Fallback.",
        }))
    return results


def score_windows(payloads: list[dict], llm_model: str, batch_size: int = 25) -> list[dict]:
    """
    payloads: Liste strukturierter Fenster-Beschreibungen (siehe
    highlight_cutter_pointbased._build_scoring_payload).

    Rueckgabe: Liste GLEICHER Laenge/Reihenfolge mit
    {"score", "humor_score", "interest_score", "reason"}.
    """
    if not payloads:
        return []

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[Scoring] Kein OPENAI_API_KEY gesetzt - LLM-Scoring uebersprungen (Fail-Open, alle Fenster behalten).")
        return _fallback_results(len(payloads))

    try:
        from openai import OpenAI
    except ImportError:
        print("[Scoring] Paket 'openai' nicht installiert - LLM-Scoring uebersprungen.")
        return _fallback_results(len(payloads))

    client = OpenAI(api_key=api_key)
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
