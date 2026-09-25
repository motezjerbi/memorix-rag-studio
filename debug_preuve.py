"""
debug_preuve.py
---------------
Montre, étape par étape, comment la preuve d'une question est trouvée ou rejetée.
Utilise le même juge (température 0) et le même mode que l'application. Ollama doit être lancé.

Usage :
    python debug_preuve.py "Qu'est-ce que la p-value ?"
    python debug_preuve.py "Qu'est-ce que la p-value ?" recopie     (ancienne méthode)
"""

import sys

import rag_core as core

QUESTION_PAR_DEFAUT = "Comment utiliser pandas pour lire un CSV ?"


def main():
    question = sys.argv[1] if len(sys.argv) > 1 else QUESTION_PAR_DEFAUT
    if len(sys.argv) > 2:
        core.PREUVE_MODE = sys.argv[2]

    vs = core.load_vectorstore()
    judge = core.load_judge_llm()

    results = core.retrieve_with_scores(vs, question)
    docs = [d for d, s in results if s <= core.DISTANCE_MAX]

    print(f"Question : {question}")
    print(f"Mode de preuve : {core.PREUVE_MODE}")
    print(f"{len(docs)} extrait(s) sous le seuil de distance ({core.DISTANCE_MAX})\n")
    if not docs:
        print("Aucun extrait assez proche : réponse générale directe.")
        return

    print("=== Extraits retrouvés ===")
    for i, (d, s) in enumerate(results, start=1):
        apercu = d.page_content[:150].replace("\n", " ")
        print(f"[{i}] distance {s:.3f} | {d.metadata.get('source')}\n    {apercu}")

    details = {}
    preuve, support = core.analyser_preuve(
        judge, docs, question, details=details, embeddings=getattr(vs, "embeddings", None),
    )

    print(f"\n=== Ce que le juge a vu ({details.get('extraits', '?')}) ===")
    if "classement" in details:
        print(f"Classement des phrases : {details['classement']} | question d'usage : {details.get('usage')}")
    for k, ph in enumerate(details.get("phrases", []), start=1):
        print(f"[{k}] {ph}")

    print("\n=== Réponse brute du juge ===")
    print(repr(details.get("reponse", "")))

    if "verification" in details:
        print("\n=== Vérification (2e étape, une seule phrase) ===")
        print(repr(details["verification"]))

    print("\n=== Verdict ===")
    for phrase, motif in details.get("examens", []):
        apercu = phrase if len(phrase) <= 110 else phrase[:107] + "..."
        print(f"- {motif}" + (f"\n     {apercu!r}" if phrase else ""))

    print()
    if preuve:
        sources = sorted({d.metadata.get("source") for d in support})
        print(f"➡️  Preuve VALIDE -> 📚 documents. Sources : {', '.join(sources)}")
    else:
        print("➡️  Preuve REJETÉE -> 🌐 général.")


if __name__ == "__main__":
    main()