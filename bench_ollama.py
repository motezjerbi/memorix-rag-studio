"""
bench_ollama.py
---------------
Deux mesures complémentaires :

1. Coût élémentaire d'un rechargement de modèle (comme avant) : chronomètre séparément
   chaque étape d'une question isolée.
2. Comparaison réelle sur un petit lot de questions : ancienne méthode (une bascule
   embeddings/juge par question) vs nouvelle méthode en deux phases (core.preparer_lot
   puis core.juger_lot : une seule bascule pour tout le lot).

Ne modifie rien de façon persistante. Dure environ 2 à 5 minutes. Ollama doit être lancé.
À lancer quand aucune évaluation ne tourne (sinon les mesures sont faussées).

Usage :
    python bench_ollama.py                 (les deux mesures)
    python bench_ollama.py --rapide        (uniquement la comparaison en deux phases)
"""

import argparse
import subprocess
import time

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

import rag_core as core

QUESTION = "À quoi sert le stride dans une convolution ?"
SEUIL_RECHARGEMENT = 8.0  # secondes : au-delà, on soupçonne un rechargement de modèle

# Petit lot représentatif pour la comparaison ancienne / nouvelle méthode.
QUESTIONS_LOT = [
    "Qu'est-ce que le surapprentissage ?",
    "Comment utiliser pandas pour lire un CSV ?",
    "Qu'est-ce que XGBoost ?",
    "À quoi sert le stride dans une convolution ?",
    "Quelle est la capitale de la France ?",
]


def chrono(nom, fonction):
    debut = time.perf_counter()
    resultat = fonction()
    duree = time.perf_counter() - debut
    print(f"  {nom:64s} {duree:6.1f} s")
    return resultat, duree


def ollama_ps(titre):
    print(f"\n--- ollama ps : {titre} ---")
    try:
        sortie = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=15)
        print(sortie.stdout.strip() or "(aucun modèle chargé)")
    except Exception as e:  # ollama absent du PATH, etc.
        print(f"(impossible de lancer 'ollama ps' : {e})")


# ---------------------------------------------------------------------------
# Mesure 1 : coût élémentaire d'un rechargement (étape par étape, question isolée)
# ---------------------------------------------------------------------------

def mesure_elementaire():
    print("=" * 78)
    print("MESURE 1 — coût élémentaire d'un rechargement de modèle")
    print("=" * 78)

    vs = core.load_vectorstore()
    emb = vs.embeddings
    if emb is None:
        print("vectorstore.embeddings est absent : le classement des phrases utilise l'ordre des extraits.")
        return
    judge = core.load_judge_llm()

    brut = vs.get(include=["documents", "metadatas"])
    docs = [Document(page_content=t, metadata=m) for t, m in zip(brut["documents"], brut["metadatas"])]
    phrases = [c["texte"] for c in core._extraire_candidats(docs)][:45]
    six = phrases[:6]
    prompt_choix = core.CHOIX_PROMPT_TEMPLATE.format(
        phrases="\n".join(f"[{i}] {p}" for i, p in enumerate(six, start=1)), question=QUESTION)
    prompt_verif = core.VERIF_PROMPT_TEMPLATE.format(
        consigne=core.VERIF_CONSIGNE_CONCEPT, question=QUESTION, phrase=six[0])

    print(f"{len(phrases)} phrases de cours, prompt de choix ≈ {len(prompt_choix) // 4} mots-pièces, "
          f"prompt de vérification ≈ {len(prompt_verif) // 4}\n")
    print("Mesures (dans l'ordre où une vraie question les enchaîne) :")

    _, emb_1 = chrono("1. embedding de la question (1re fois)", lambda: emb.embed_query(QUESTION))
    _, emb_2 = chrono("2. embedding de la question (2e fois : modèle déjà chargé)", lambda: emb.embed_query(QUESTION))
    _, phr_1 = chrono(f"3. embedding de {len(phrases)} phrases du cours", lambda: emb.embed_documents(phrases))
    ollama_ps("après les embeddings")

    _, jug_1 = chrono("4. juge : prompt de choix (1re fois, après les embeddings)", lambda: judge.invoke(prompt_choix))
    ollama_ps("après le premier appel du juge")
    _, jug_2 = chrono("5. juge : prompt de choix (2e fois : modèle déjà chargé)", lambda: judge.invoke(prompt_choix))
    _, ver_1 = chrono("6. juge : prompt de vérification (une seule phrase)", lambda: judge.invoke(prompt_verif))
    _, emb_3 = chrono("7. embedding de la question, APRÈS un appel du juge", lambda: emb.embed_query(QUESTION))
    _, jug_3 = chrono("8. juge : prompt de choix, APRÈS un embedding", lambda: judge.invoke(prompt_choix))
    ollama_ps("fin")

    print("\n=== Lecture ===")
    surcout_emb = emb_3 - emb_2
    surcout_jug = jug_3 - jug_2
    print(f"Coût de calcul d'une question : embeddings ≈ {emb_2 + phr_1:.0f} s, "
          f"juge choix ≈ {jug_2:.0f} s, vérification ≈ {ver_1:.0f} s "
          f"(total ≈ {emb_2 + phr_1 + jug_2 + ver_1:.0f} s si tout reste chargé)")
    if surcout_emb > SEUIL_RECHARGEMENT or surcout_jug > SEUIL_RECHARGEMENT:
        print(f"⚠️  Rechargement de modèle détecté : +{surcout_emb:.0f} s pour les embeddings après le juge, "
              f"+{surcout_jug:.0f} s pour le juge après les embeddings.")
        print("   Ollama décharge un modèle pour charger l'autre (mémoire insuffisante pour les deux).")
    else:
        print("✅ Pas de rechargement détecté : les deux modèles restent en mémoire.")

    return {"cout_reel": emb_2 + phr_1 + jug_2 + ver_1, "surcout_emb": surcout_emb, "surcout_jug": surcout_jug}


# ---------------------------------------------------------------------------
# Mesure 2 : comparaison réelle ancienne méthode vs nouvelle méthode en deux phases
# ---------------------------------------------------------------------------

def mesure_ancienne_methode(vs, llm, judge, qa):
    """Une question à la fois : embeddings puis juge, en boucle (comme avant)."""
    print(f"\n--- Ancienne méthode : {len(QUESTIONS_LOT)} question(s), une par une ---")
    t0 = time.perf_counter()
    for q in QUESTIONS_LOT:
        core.answer_with_trace(vs, llm, qa, q, judge_llm=judge, generer_reponses=False)
    duree = time.perf_counter() - t0
    print(f"  Total (ancienne méthode)                                        {duree:6.1f} s")
    return duree


def mesure_nouvelle_methode(vs, llm, judge, qa):
    """Toutes les recherches d'abord (embeddings), puis tout le jugement (mistral)."""
    print(f"\n--- Nouvelle méthode : {len(QUESTIONS_LOT)} question(s), en deux phases ---")
    t0 = time.perf_counter()
    lot = core.preparer_lot(vs, QUESTIONS_LOT)
    t_phase1 = time.perf_counter() - t0
    print(f"  Phase 1 (embeddings, {len(QUESTIONS_LOT)} question(s))                          {t_phase1:6.1f} s")

    t1 = time.perf_counter()
    core.juger_lot(judge, llm, qa, lot, generer_reponses=False)
    t_phase2 = time.perf_counter() - t1
    print(f"  Phase 2 (juge, {len(QUESTIONS_LOT)} question(s))                                {t_phase2:6.1f} s")

    duree = t_phase1 + t_phase2
    print(f"  Total (nouvelle méthode)                                        {duree:6.1f} s")
    return duree


def mesure_comparaison():
    print("\n" + "=" * 78)
    print(f"MESURE 2 — comparaison sur {len(QUESTIONS_LOT)} questions : ancienne vs nouvelle méthode")
    print("=" * 78)

    if core.PREUVE_MODE != "phrases":
        print(f"⚠️  PREUVE_MODE = '{core.PREUVE_MODE}' : la nouvelle méthode (juger_lot) exige 'phrases'. "
              "Comparaison ignorée.")
        return

    vs = core.load_vectorstore()
    llm = core.load_llm()
    judge = core.load_judge_llm()
    qa = PromptTemplate(template=core.QA_PROMPT_TEMPLATE, input_variables=["context", "question"])

    ollama_ps("avant la comparaison")

    duree_ancienne = mesure_ancienne_methode(vs, llm, judge, qa)
    ollama_ps("après l'ancienne méthode")

    duree_nouvelle = mesure_nouvelle_methode(vs, llm, judge, qa)
    ollama_ps("après la nouvelle méthode")

    print("\n=== Lecture ===")
    print(f"Ancienne méthode (question par question) : {duree_ancienne:.0f} s pour {len(QUESTIONS_LOT)} questions "
          f"({duree_ancienne / len(QUESTIONS_LOT):.0f} s/question)")
    print(f"Nouvelle méthode (deux phases)            : {duree_nouvelle:.0f} s pour {len(QUESTIONS_LOT)} questions "
          f"({duree_nouvelle / len(QUESTIONS_LOT):.0f} s/question)")
    if duree_ancienne > 0:
        gain = 100 * (1 - duree_nouvelle / duree_ancienne)
        facteur = duree_ancienne / duree_nouvelle if duree_nouvelle > 0 else float("inf")
        print(f"➡️  Gain : {gain:.0f} % plus rapide (×{facteur:.1f})")
    print("Regarde les sorties 'ollama ps' ci-dessus : avec l'ancienne méthode, le modèle chargé "
          "alterne à chaque question ; avec la nouvelle, il ne change qu'une seule fois.")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Ollama : rechargements de modèle")
    parser.add_argument("--rapide", action="store_true",
                        help="saute la mesure élémentaire (mesure 1), ne garde que la comparaison en deux phases")
    args = parser.parse_args()

    if not args.rapide:
        mesure_elementaire()

    mesure_comparaison()


if __name__ == "__main__":
    main()