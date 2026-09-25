"""
test_couverture.py
-------------------
Pour une question en échec (REJET_PRUDENT), montre TOUS les candidats extraits
(pas seulement les 6 premiers) avec leur rang de similarité. Permet de distinguer :
  - la bonne réponse existe mais est mal classée (correctif : augmenter PREUVE_NB_PHRASES)
  - la bonne réponse n'existe pas dans le corpus (correctif : enrichir les documents)

Usage : python test_couverture.py "Comment remplacer les valeurs manquantes d'une colonne avec pandas ?"
"""

import sys
import rag_core as core

QUESTION_PAR_DEFAUT = "Comment remplacer les valeurs manquantes d'une colonne avec pandas ?"


def main():
    question = sys.argv[1] if len(sys.argv) > 1 else QUESTION_PAR_DEFAUT

    vs = core.load_vectorstore()
    results = core.retrieve_with_scores(vs, question)
    docs = [d for d, s in results if s <= core.DISTANCE_MAX]

    print(f"Question : {question}")
    print(f"{len(docs)} extrait(s) retrouvé(s) sous le seuil ({core.DISTANCE_MAX})\n")

    usage = bool(core.USAGE_QUESTION_PATTERN.search(question))
    candidats = core._extraire_candidats(docs, avec_code=usage)
    print(f"Question d'usage : {usage} | {len(candidats)} candidat(s) extrait(s) AU TOTAL (avant le plafond de {core.PREUVE_NB_PHRASES})\n")

    embeddings = getattr(vs, "embeddings", None)
    classes, methode = core._classer_candidats(embeddings, question, candidats)

    print(f"Classement : {methode}\n")
    print("=== TOUS les candidats, classés du plus proche au moins proche ===")
    for i, c in enumerate(classes, start=1):
        limite = " <-- dans le top actuel (montré au juge)" if i <= core.PREUVE_NB_PHRASES else ""
        score = f"{c['score']:.3f}" if c.get("score") is not None else "?"
        tag = "[code]" if c["code"] else "[prose]"
        print(f"[{i:2d}] sim={score} {tag} {c['texte'][:120]}{limite}")


if __name__ == "__main__":
    main()