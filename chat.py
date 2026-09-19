"""
chat.py
-------
Chatbot en ligne de commande orienté études Data Science, IA & MLOps :
répond à des questions techniques, génère des résumés d'architecture,
des fiches mémo et des quiz spécialisés sur un cours entier ou une section précise.
Toute la logique métier est centralisée dans rag_core.py.

Usage:
    python chat.py
"""

import os

os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")

from langchain.prompts import PromptTemplate
import rag_core as core

QUIZ_KEYWORDS = ["quiz", "qcm", "teste-moi", "teste moi", "questionnaire", "test"]
FICHE_KEYWORDS = [
    "fiche de révision", "fiche de revision", "fiche", "récapitulatif",
    "recapitulatif", "aide-mémoire", "aide memoire", "formulaire", "memo"
]
SUMMARY_KEYWORDS = ["résume", "resume", "synthèse", "synthese", "summary", "résumé", "vue d'ensemble"]


def print_sources(docs):
    sources = sorted(set(d.metadata.get("source", "inconnu") for d in docs))
    if sources:
        print("📚 Sources techniques indexées :")
        for src in sources:
            print(f"   - {src}")


def resolve_book_or_explain(question, chapter_number, vectorstore, action_hint):
    """
    Identifie le module/livre Data Science visé par la question.
    En cas d'ambiguïté, guide l'utilisateur vers les cours indexés.
    """
    book_id = core.detect_book_in_question(question)
    if book_id is not None:
        return book_id

    if chapter_number:
        candidates = core.get_books_with_chapter(vectorstore, chapter_number)
        if len(candidates) == 0:
            print(f"\n⚠️  Aucun cours indexé ne contient la section {chapter_number}.\n")
            return None
        if len(candidates) == 1:
            return candidates[0]
        labels = ", ".join(core.BOOK_LABELS.get(b, b) for b in candidates)
        print(
            f"\n🤔 Plusieurs cours comportent une section {chapter_number} : {labels}.\n"
            f"   Précise par exemple : '{action_hint} {chapter_number} de "
            f"{core.BOOK_LABELS.get(candidates[0], candidates[0])}'\n"
        )
        return None

    labels = ", ".join(core.BOOK_LABELS.values())
    print(
        f"\n🤔 Spécifie le domaine Data Science concerné. Cours disponibles : {labels}.\n"
        f"   Indique le nom ou le numéro de cours dans ta commande.\n"
    )
    return None


def main():
    if not core.is_vectorstore_ready():
        print("❌ Aucune base vectorielle trouvée.")
        print("   Exécute d'abord l'indexation : python ingest.py")
        return

    print("🧠 Chargement des tenseurs et de l'index Chroma...")
    vectorstore = core.load_vectorstore()

    print(f"🤖 Initialisation du moteur LLM '{core.LLM_MODEL}' (Ollama)...")
    llm = core.load_llm()

    qa_prompt = PromptTemplate(
        template=core.QA_PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    print("\n✅ Cockpit MEMORIX (Data Science Hub) opérationnel ! Tape 'exit' pour quitter.")
    print("💡 Commandes rapides :")
    print("   - 'résume le cours 2' (synthèse Machine Learning)")
    print("   - 'fiche de révision sur computer vision' (définitions, tenseurs et loss)")
    print("   - 'quiz sur le cours 1' (validation interactive)")
    print("   - 'résume la section 3 de machine learning' (zoom ciblé)\n")

    while True:
        question = input("❓ Requête technique : ").strip()
        if question.lower() in ("exit", "quit", "q"):
            print("👋 Session terminée.")
            break
        if not question:
            continue

        chapter_number = core.detect_chapter_request(question)
        question_lower = question.lower()

        wants_quiz = any(kw in question_lower for kw in QUIZ_KEYWORDS)
        wants_fiche = (not wants_quiz) and any(kw in question_lower for kw in FICHE_KEYWORDS)
        wants_summary = (not wants_quiz) and (not wants_fiche) and any(kw in question_lower for kw in SUMMARY_KEYWORDS)

        # --- Mode 1 : Quiz / QCM ---
        if wants_quiz:
            book_id = resolve_book_or_explain(question, chapter_number, vectorstore, "quiz sur la section")
            if book_id is None:
                print("-" * 60)
                continue

            book_label = core.BOOK_LABELS.get(book_id, book_id)
            if chapter_number:
                print(f"\n📝 Élaboration du protocole QCM — section {chapter_number} ({book_label})...")
                docs = core.get_chapter_chunks(vectorstore, chapter_number, book_id)
            else:
                print(f"\n📝 Élaboration du protocole QCM — {book_label} (complet)...")
                docs = core.get_book_chunks(vectorstore, book_id)

            questions, raw_text = core.generate_quiz(llm, docs, progress_callback=lambda m: print(f"   {m}"))

            if not questions:
                print("\n⚠️  Impossible de compiler le quiz structuré. Sortie brute du modèle :\n")
                print(raw_text or "(aucun contenu)")
                print("-" * 60)
                continue

            print(f"\n📝 Quiz Data Science — {book_label}" + (f" (section {chapter_number})" if chapter_number else ""))
            score = 0
            for i, q in enumerate(questions, start=1):
                print(f"\n❓ Question {i}/{len(questions)} : {q['question']}")
                for letter in ("a", "b", "c", "d"):
                    print(f"   {letter.upper()}) {q[letter]}")
                user_answer = input("   Ta réponse (A/B/C/D) : ").strip().upper()
                if user_answer == q["correct"]:
                    print("   ✅ Exact !")
                    score += 1
                else:
                    correct_letter = q["correct"].lower()
                    print(f"   ❌ Erreur. Réponse attendue : {q['correct']}) {q.get(correct_letter, '?')}")
                print(f"   💡 Analyse : {q['explanation']}")
            print(f"\n🏆 Résultat final : {score}/{len(questions)} ({(score / len(questions)) * 100:.1f}%)")
            print("-" * 60)
            continue

        # --- Mode 2 : Fiche mémo technique ---
        if wants_fiche:
            book_id = resolve_book_or_explain(question, chapter_number, vectorstore, "fiche de révision sur la section")
            if book_id is None:
                print("-" * 60)
                continue

            book_label = core.BOOK_LABELS.get(book_id, book_id)
            if chapter_number:
                print(f"\n📋 Extraction de la fiche technique — section {chapter_number} ({book_label})...")
                docs = core.get_chapter_chunks(vectorstore, chapter_number, book_id)
                empty_msg = f"Aucun extrait localisé pour la section {chapter_number} dans '{book_label}'."
            else:
                print(f"\n📋 Extraction de la fiche technique — {book_label} (module entier)...")
                docs = core.get_book_chunks(vectorstore, book_id)
                empty_msg = f"Aucun contenu indexé pour '{book_label}'."

            fiche = core.summarize_documents(
                llm, docs, empty_msg, core.FICHE_EXTRACT_PROMPT, core.COMBINE_FICHE_SECTION_PROMPT,
                progress_callback=lambda m: print(f"   {m}"),
            )
            print(f"\n📋 Fiche Technique — {book_label} :\n{fiche}\n")
            print_sources(docs)
            print("-" * 60)
            continue

        # --- Mode 3 : Synthèse analytique ---
        if wants_summary:
            book_id = resolve_book_or_explain(question, chapter_number, vectorstore, "résume la section")
            if book_id is None:
                print("-" * 60)
                continue

            book_label = core.BOOK_LABELS.get(book_id, book_id)
            if chapter_number:
                print(f"\n📖 Synthèse approfondie — section {chapter_number} ({book_label})...")
                docs = core.get_chapter_chunks(vectorstore, chapter_number, book_id)
                empty_msg = f"Aucun contenu trouvé pour la section {chapter_number} dans '{book_label}'."
            else:
                print(f"\n📖 Synthèse générale — {book_label} (toutes sections)...")
                docs = core.get_book_chunks(vectorstore, book_id)
                empty_msg = f"Aucun document indexé pour '{book_label}'."

            summary = core.summarize_documents(
                llm, docs, empty_msg, core.CHAPTER_SUMMARY_PROMPT, core.COMBINE_SUMMARIES_PROMPT,
                progress_callback=lambda m: print(f"   {m}"),
            )
            print(f"\n💬 Synthèse — {book_label} :\n{summary}\n")
            print_sources(docs)
            print("-" * 60)
            continue

        # --- Mode 4 : Recherche augmentée (QA classique) ---
        answer, docs, warning = core.answer_question(vectorstore, llm, qa_prompt, question)
        print(f"\n💬 Réponse technique :\n{answer}\n")
        if warning:
            print(f"   ⚠️  {warning}")
        if docs:
            print_sources(docs)
        print("-" * 60)


if __name__ == "__main__":
    main()