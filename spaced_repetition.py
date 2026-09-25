"""
spaced_repetition.py
---------------------
Algorithme SM-2 (SuperMemo 2) pour la répétition espacée des flashcards.
Logique pure, sans dépendance : facile à tester isolément, sans base de données.
"""

from datetime import datetime, timedelta

# Qualité de rappel (échelle SM-2 simplifiée à 4 boutons)
AGAIN = 0   # "Je ne savais pas" -> la carte revient dès demain
HARD = 3    # "Difficile, mais j'ai trouvé"
GOOD = 4    # "Correct, sans effort particulier"
EASY = 5    # "Trop facile" -> l'intervalle grandit plus vite

EASE_FACTOR_MIN = 1.3
EASE_FACTOR_DEFAULT = 2.5


def schedule(quality: int, repetitions: int, ease_factor: float, interval_days: float):
    """
    Calcule le nouvel état d'une carte après une révision (algorithme SM-2).

    quality       : 0 (Again) à 5 (Easy), voir les constantes ci-dessus.
    repetitions   : nombre de révisions consécutives réussies AVANT cette révision.
    ease_factor   : facilité de la carte (>= 1.3, 2.5 par défaut : plus haut = intervalles
                    qui grandissent plus vite).
    interval_days : intervalle actuel en jours, AVANT cette révision.

    Retourne (nouvelles_repetitions, nouveau_ease_factor, nouvel_intervalle_jours).
    """
    if quality < 3:
        # Échec : on repart de zéro, la carte revient dès le lendemain.
        return 0, ease_factor, 1.0

    if repetitions == 0:
        nouvel_intervalle = 1.0
    elif repetitions == 1:
        nouvel_intervalle = 6.0
    else:
        nouvel_intervalle = round(interval_days * ease_factor, 1)

    nouvel_ease = ease_factor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    nouvel_ease = max(EASE_FACTOR_MIN, round(nouvel_ease, 2))

    return repetitions + 1, nouvel_ease, nouvel_intervalle


def next_review_date(interval_days: float, from_dt: datetime | None = None) -> str:
    """Date ISO de la prochaine révision, `interval_days` jours après `from_dt` (défaut : maintenant)."""
    base = from_dt or datetime.now()
    return (base + timedelta(days=interval_days)).isoformat()