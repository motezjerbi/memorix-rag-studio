"""
test_cascade.py
---------------
Teste la cascade de réponse : documents -> preuve vérifiée -> réponse générale.
Pour chaque question, affiche l'origine obtenue, l'origine attendue (✅ / ❌),
la preuve trouvée dans le cours et un début de réponse.

Prérequis : Ollama doit être lancé.
Usage : python test_cascade.py
"""

import rag_core as core
from langchain_core.prompts import PromptTemplate

# (question, origine attendue) : "docs" = 📚 cours, "general" = 🌐 réponse générale
CAS = [
    ("Comment utiliser pandas pour lire un CSV ?", "docs"),
    ("Qu'est-ce que le surapprentissage ?", "docs"),
    ("Qu'est-ce que XGBoost ?", "general"),
    ("Comment fonctionne un algorithme de Random Forest ?", "general"),
    ("Quelle est la capitale de la France ?", "general"),
]


def main():
    vs = core.load_vectorstore()
    llm = core.load_llm()
    qa = PromptTemplate(template=core.QA_PROMPT_TEMPLATE, input_variables=["context", "question"])

    reussis = 0
    for question, attendu in CAS:
        # La preuve est recalculée ici uniquement pour l'afficher (un appel de plus au modèle).
        candidats = [d for d, s in core.retrieve_with_scores(vs, question) if s <= core.DISTANCE_MAX]
        preuve = core.trouver_preuve(llm, candidats, question) if candidats else None

        answer, docs, _warning = core.answer_question(vs, llm, qa, question)
        obtenu = "docs" if docs else "general"
        ok = obtenu == attendu
        reussis += ok

        etiquette = "📚 documents" if docs else "🌐 général"
        print(f"\n### {question}")
        print(f"[{etiquette}] {'✅' if ok else '❌ (attendu : ' + attendu + ')'}")
        if docs:
            sources = sorted({d.metadata.get("source", "?") for d in docs})
            print(f"Sources : {', '.join(sources)}")
        print(f"Preuve  : {preuve or '(aucune)'}")
        print(f"Réponse : {answer[:300].strip()}")

    print(f"\n=== Résultat : {reussis}/{len(CAS)} conformes ===")


if __name__ == "__main__":
    main()