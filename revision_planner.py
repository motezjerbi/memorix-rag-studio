"""
revision_planner.py
--------------------
Génère un plan de révision jour par jour à partir de l'historique de
progression (progress_tracker.py) et de la liste des sections disponibles.

AUCUN appel LLM ici : calcul pur sur des données déjà en mémoire, donc
quasi instantané et sans coût pour Ollama — pensé pour une machine avec
des ressources limitées.
"""

# Priorité assumée pour une section jamais testée : ni "faible" (on ne sait
# pas qu'elle pose problème), ni "forte" (on n'a aucune preuve qu'elle est
# maîtrisée) — un score neutre-prudent qui la fait apparaître avant les
# sections confirmées comme solides, mais après les échecs confirmés.
UNTESTED_PRIORITY = 0.4


def build_plan(all_sections, stats, num_days, sections_per_day=2):
    """
    all_sections : liste de dicts {"book_id", "book_label", "chapter"} —
        TOUTES les sections connues (testées ou non) de tous les cours
        indexés.
    stats : résultat de progress_tracker.get_section_stats() — scores
        moyens déjà calculés pour les sections testées.
    num_days : nombre de jours disponibles avant l'échéance (>= 1).
    sections_per_day : nombre de sections à réviser par jour (hors dernier
        jour, réservé à la révision flash).

    Retourne une liste de jours : [{"day_label", "items", "is_recap"}, ...].
    Chaque item est un dict {"book_id", "book_label", "chapter", "avg_score",
    "attempts", "status"} où status ∈ {"tested", "untested"}.
    """
    stats_lookup = {(s["book_id"], s["chapter"]): s for s in stats}

    scored_sections = []
    for sec in all_sections:
        key = (sec["book_id"], sec["chapter"])
        if key in stats_lookup:
            s = stats_lookup[key]
            scored_sections.append({
                **sec,
                "avg_score": s["avg_score"],
                "attempts": s["attempts"],
                "status": "tested",
            })
        else:
            scored_sections.append({
                **sec,
                "avg_score": UNTESTED_PRIORITY,
                "attempts": 0,
                "status": "untested",
            })

    # Priorité : score croissant (le plus faible / le moins connu = le plus urgent).
    scored_sections.sort(key=lambda s: s["avg_score"])

    if not scored_sections:
        return []

    if num_days <= 1:
        return [{"day_label": "Aujourd'hui", "items": scored_sections[:sections_per_day], "is_recap": False}]

    study_days = num_days - 1  # le dernier jour est réservé à la révision flash
    days = []
    idx = 0
    for d in range(study_days):
        chunk = scored_sections[idx: idx + sections_per_day]
        idx += sections_per_day
        if not chunk:
            break
        days.append({"day_label": f"J-{study_days - d}", "items": chunk, "is_recap": False})

    # Jour de révision flash : reprend les sections les plus faibles du plan entier,
    # tous jours confondus (utile même si elles ont déjà été vues plus tôt).
    recap_items = scored_sections[:min(4, len(scored_sections))]
    days.append({"day_label": "Jour J // Révision flash", "items": recap_items, "is_recap": True})

    return days