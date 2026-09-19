"""Affiche la distance du meilleur extrait pour des questions dans / hors cours."""

import rag_core as core

DANS_LES_COURS = [
    "Qu'est-ce que la descente de gradient ?",
    "Comment fonctionne une convolution ?",
    "Comment utiliser pandas pour lire un CSV ?",
    "Qu'est-ce que le surapprentissage ?",
    "À quoi sert la régularisation L2 ?",
]

HORS_DES_COURS = [
    "Qu'est-ce que XGBoost ?",
    "Comment fonctionne un algorithme de Random Forest ?",
    "Explique le test A/B en statistiques.",
    "Qu'est-ce qu'un modèle de diffusion ?",
    "Quelle est la capitale de la France ?",
]


def meilleure_distance(vectorstore, question):
    results = core.retrieve_with_scores(vectorstore, question)
    return min(score for _, score in results) if results else None


if __name__ == "__main__":
    vs = core.load_vectorstore()
    for q in ["Qu'est-ce que XGBoost ?", "Comment fonctionne un algorithme de Random Forest ?",
              "Qu'est-ce qu'un modèle de diffusion ?"]:
        doc, score = core.retrieve_with_scores(vs, q)[0]
        print(f"\n### {q}  (distance {score:.3f})")
        print(f"Source : {doc.metadata.get('source')}")
        print(doc.page_content[:300].replace("\n", " "))