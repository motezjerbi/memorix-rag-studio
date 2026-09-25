"""
test_cascade.py
---------------
Teste la cascade de réponse : documents -> preuve vérifiée -> réponse générale.
Pour chaque question : origine obtenue vs attendue (✅ / ❌), raison de la décision,
distances des extraits, preuve, sources réellement utilisées et début de réponse.
Chaque question est posée REPETITIONS fois : si l'origine change d'une fois à l'autre, ⚠️ instable.

Prérequis : Ollama doit être lancé.
Usage : python test_cascade.py
"""

import rag_core as core
from langchain_core.prompts import PromptTemplate

REPETITIONS = 2  # mettre 1 pour aller plus vite

# (question, origine attendue) : "documents" = 📚 cours, "general" = 🌐 réponse générale
CAS = [
    ("Comment utiliser pandas pour lire un CSV ?", "documents"),
    ("Qu'est-ce que le surapprentissage ?", "documents"),
    ("Qu'est-ce que la p-value ?", "documents"),  # fiche 05_Statistiques_Bases.md
    ("Qu'est-ce que XGBoost ?", "general"),
    ("Comment fonctionne un algorithme de Random Forest ?", "general"),
    ("Quelle est la capitale de la France ?", "general"),
]


def main():
    vs = core.load_vectorstore()
    llm = core.load_llm()
    judge = core.load_judge_llm()
    qa = PromptTemplate(template=core.QA_PROMPT_TEMPLATE, input_variables=["context", "question"])

    conformes = 0
    instables = 0
    for question, attendu in CAS:
        traces = [core.answer_with_trace(vs, llm, qa, question, judge_llm=judge) for _ in range(REPETITIONS)]
        t = traces[0]
        origines = {x["origine"] for x in traces}
        stable = len(origines) == 1
        ok = stable and t["origine"] == attendu
        conformes += ok
        instables += not stable

        etiquette = "📚 documents" if t["origine"] == "documents" else "🌐 général"
        verdict = "✅" if ok else f"❌ (attendu : {attendu})"
        if not stable:
            verdict += "  ⚠️ instable : " + " / ".join(x["origine"] for x in traces)

        print(f"\n### {question}")
        print(f"[{etiquette}] {verdict}")
        print(f"Raison    : {t['raison']}")
        print(f"Distances : {t['distances']}")
        print(f"Preuve    : {t['preuve'] or '(aucune)'}")
        if t["docs"]:
            sources = sorted({d.metadata.get('source', '?') for d in t["docs"]})
            print(f"Sources   : {', '.join(sources)}")
        print(f"Réponse   : {t['answer'][:250].strip()}")

    print(f"\n=== Résultat : {conformes}/{len(CAS)} conformes, {instables} instable(s) ===")


if __name__ == "__main__":
    main()