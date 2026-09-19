"""
debug_preuve.py
---------------
Montre, étape par étape, pourquoi la preuve d'une question est acceptée ou rejetée.
Change QUESTION pour tester une autre question. Ollama doit être lancé.

Usage : python debug_preuve.py
"""

import rag_core as core

QUESTION = "Comment utiliser pandas pour lire un CSV ?"


def main():
    vs = core.load_vectorstore()
    llm = core.load_llm()

    results = core.retrieve_with_scores(vs, QUESTION)
    docs = [d for d, s in results if s <= core.DISTANCE_MAX]

    print(f"Question : {QUESTION}")
    print(f"{len(docs)} extrait(s) sous le seuil de distance ({core.DISTANCE_MAX})\n")

    print("=== Extraits montrés à Mistral ===")
    for i, (d, s) in enumerate(results[: core.JUDGE_MAX_EXTRACTS], start=1):
        apercu = d.page_content[:200].replace("\n", " ")
        print(f"[{i}] distance {s:.3f} | {d.metadata.get('source')}\n    {apercu}\n")

    if not docs:
        print("Aucun extrait assez proche : réponse générale directe.")
        return

    context = "\n\n".join(d.page_content for d in docs[: core.JUDGE_MAX_EXTRACTS])
    reponse = llm.invoke(
        core.PREUVE_PROMPT_TEMPLATE.format(context=context, question=QUESTION)
    ).strip()

    print("=== Réponse brute de Mistral ===")
    print(reponse, "\n")

    print("=== Vérifications du code ===")
    if core.re.match(r"\W*aucune", reponse.lower()):
        print("❌ Mistral a répondu AUCUNE.")
        return

    match = core.QUOTE_PATTERN.search(reponse)
    if not match:
        print("❌ Pas de phrase entre guillemets dans la réponse.")
        return

    phrase = match.group(1).strip()
    print(f"Phrase extraite : {phrase}\n")

    ok_longueur = len(phrase) >= core.PREUVE_MIN_CHARS
    ok_pas_code = not core.CODE_OR_LIST_PATTERN.search(phrase)
    ok_existe = core._norm(phrase) in core._norm(context)

    print(f"{'✅' if ok_longueur else '❌'} Longueur ({len(phrase)} caractères, minimum {core.PREUVE_MIN_CHARS})")
    print(f"{'✅' if ok_pas_code else '❌'} Pas de code ni de liste")
    print(f"{'✅' if ok_existe else '❌'} La phrase existe vraiment dans les extraits")

    if ok_longueur and ok_pas_code and ok_existe:
        print("\n➡️  Preuve VALIDE : la question serait en 📚 documents.")
    else:
        print("\n➡️  Preuve REJETÉE : la question part en 🌐 général.")


if __name__ == "__main__":
    main()