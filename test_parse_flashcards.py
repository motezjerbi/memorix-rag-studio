"""Vérifie isolément parse_flashcards_text sur un échantillon réel (sans Ollama)."""
import rag_core as core

echantillon = """1. FRONT: Qu'est-ce que Python?
   BACK: Python est un langage de programmation très utilisé en Data Science grâce à son écosystème.

2. FRONT: Quels sont les types courants en Python?
   BACK: Les types courants en Python sont int, float, str, bool, list, tuple, dict et set.

3. FRONT: Qu'est-ce qu'une fonction en Python?
   BACK: Une fonction en Python est une série d'instructions qui peuvent être exécutées plusieurs fois avec des arguments différents."""

cards = core.parse_flashcards_text(echantillon)
print(f"{len(cards)} carte(s) extraite(s) (attendu : 3)\n")
for c in cards:
    print(f"FRONT: {c['front']}")
    print(f"BACK : {c['back']}\n")