"""
evaluate.py
-----------
Mesure la qualité de la cascade (📚 documents / 🌐 réponse générale) sur eval_set.json.

Pour chaque question : origine attendue vs obtenue, et pour les réponses 📚 la bonne source.
Erreurs distinguées (de la plus grave à la moins grave) :
  FAUSSE_ETIQUETTE  attendu 🌐 mais obtenu 📚 : le système prétend citer le cours à tort. À garder à ZÉRO.
  MAUVAISE_SOURCE   📚 attendu et obtenu, mais le fichier cité n'est pas le bon.
  REJET_PRUDENT     attendu 📚 mais obtenu 🌐 : la réponse est correcte, seule l'étiquette est trop sévère.
  ERREUR            Ollama n'a pas répondu (deux tentatives) : la question est à refaire.

Chaque exécution est ajoutée à resultats_eval.csv (nom de l'essai, embeddings, seuil, mode...)
pour comparer avant / après un changement.

Le juge tourne à température 0 : sa réponse ne dépend que de son prompt. Elle est donc gardée dans
.eval_cache.json ; relancer l'évaluation (ou la reprendre après un plantage ou Ctrl+C) est instantané
pour les questions déjà jugées. Changer de modèle, d'extraits ou de prompt invalide le cache tout seul.

Prérequis : Ollama lancé.
Usage :
    python evaluate.py --nom baseline --rapide                 (aucune réponse rédigée : le plus court)
    python evaluate.py --nom test --rapide --un-sur 3          (une question sur trois : aperçu rapide)
    python evaluate.py --nom diag --rapide --numeros 3,6,7     (seulement ces questions, numérotées comme dans la liste)
    python evaluate.py --nom sans --rapide --sans-verification (sans la 2e étape OUI/NON : pour comparer)
    python evaluate.py --nom ancien --rapide --mode recopie    (ancienne méthode : Mistral recopie une phrase)
    python evaluate.py --nom bge --rapide --seuil 1.1          (autre seuil de distance)
    python evaluate.py --nom sim --rapide --sim-min 0.6        (rejette les phrases choisies trop éloignées)
    Ctrl+C : arrête proprement, résume et enregistre ce qui est déjà fait.
"""

import argparse
import csv
import hashlib
import json
import os
import time
from collections import defaultdict
from datetime import datetime

import rag_core as core
from langchain_core.prompts import PromptTemplate

EVAL_FILE = "eval_set.json"
CSV_FILE = "resultats_eval.csv"
CACHE_FILE = ".eval_cache.json"
PAUSE_REPRISE = 20          # secondes d'attente avant la 2e tentative si Ollama ne répond pas
ERREURS_MAX_DE_SUITE = 2    # au-delà, on arrête (Ollama est probablement tombé)

ICONES = {"OK": "✅", "FAUSSE_ETIQUETTE": "🚨", "REJET_PRUDENT": "⚠️ ", "MAUVAISE_SOURCE": "❌", "ERREUR": "💥"}


class JugeEnCache:
    """Enveloppe le juge : la même invite donne la même réponse, sans refaire le calcul."""

    def __init__(self, judge, path=CACHE_FILE):
        self.judge = judge
        self.path = path
        self.hits = 0
        self.data = {}
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as f:
                    self.data = json.load(f)
            except (OSError, json.JSONDecodeError):
                self.data = {}

    def invoke(self, prompt):
        key = hashlib.sha256(f"{core.LLM_MODEL}\x00{prompt}".encode("utf-8")).hexdigest()
        if key in self.data:
            self.hits += 1
            return self.data[key]
        reply = self.judge.invoke(prompt)
        self.data[key] = reply
        with open(self.path, "w", encoding="utf-8") as f:  # enregistré tout de suite : rien n'est perdu
            json.dump(self.data, f, ensure_ascii=False)
        return reply


def statut_du_cas(cas, trace):
    """Retourne le statut d'une question."""
    attendu, obtenu = cas["attendu"], trace["origine"]
    if attendu == "general" and obtenu == "documents":
        return "FAUSSE_ETIQUETTE"
    if attendu == "documents" and obtenu == "general":
        return "REJET_PRUDENT"
    if attendu == "documents" and obtenu == "documents":
        sources = " ".join(d.metadata.get("source", "") for d in trace["docs"])
        if cas.get("source") and cas["source"] not in sources:
            return "MAUVAISE_SOURCE"
    return "OK"


def appeler_avec_reprise(*args, **kwargs):
    """Une question ; en cas d'erreur d'Ollama, une seconde tentative après une pause."""
    try:
        return core.answer_with_trace(*args, **kwargs)
    except Exception as err:  # KeyboardInterrupt n'est pas une Exception : Ctrl+C reste possible
        print(f"   ⚠️  {type(err).__name__} : nouvelle tentative dans {PAUSE_REPRISE}s...")
        time.sleep(PAUSE_REPRISE)
        return core.answer_with_trace(*args, **kwargs)


def appeler_avec_reprise_lot(executer):
    """Comme appeler_avec_reprise, mais pour un lot entier (phase 2) : une seule reprise possible."""
    try:
        return executer()
    except Exception as err:
        print(f"   ⚠️  {type(err).__name__} : nouvelle tentative dans {PAUSE_REPRISE}s...")
        time.sleep(PAUSE_REPRISE)
        return executer()


def nouvelle_ligne(numero, cas, statut, trace=None, erreur=""):
    """Une ligne du CSV (mêmes colonnes pour une réponse et pour une erreur)."""
    trace = trace or {"origine": "", "raison": erreur, "distances": [], "preuve": None, "docs": [],
                      "temps": {"juge": 0.0}, "juge": {}}
    juge = trace["juge"]
    return {
        "date": datetime.now().isoformat(timespec="seconds"),
        "essai": "",
        "embeddings": core.EMBEDDING_MODEL,
        "llm": core.LLM_MODEL,
        "seuil": core.DISTANCE_MAX,
        "mode_preuve": core.PREUVE_MODE,
        "verification": "oui" if core.VERIF_ACTIVE else "non",
        "numero": numero,
        "question": cas["question"],
        "categorie": cas["categorie"],
        "attendu": cas["attendu"],
        "obtenu": trace["origine"],
        "statut": statut,
        "raison": trace["raison"],
        "distance_min": min(trace["distances"]) if trace["distances"] else "",
        "preuve": trace["preuve"] or "",
        "sources": " | ".join(sorted({d.metadata.get("source", "") for d in trace["docs"]})),
        "secondes_juge": round(trace["temps"]["juge"], 1),
        "extraits_montres": juge.get("extraits", ""),
        "classement": juge.get("classement", ""),
        "similarite": juge.get("similarite", ""),
        "secondes_classement": round(juge.get("secondes_classement", 0), 1),
        "phrases_montrees": " || ".join(juge.get("phrases", [])),
        "reponse_juge": juge.get("reponse", ""),
        "verdict_verification": juge.get("verification", ""),
        "motifs": " || ".join(f"{m} :: {p[:90]}" for p, m in juge.get("examens", [])),
    }


def main():
    parser = argparse.ArgumentParser(description="Évalue la cascade sur eval_set.json")
    parser.add_argument("--nom", default="", help="nom de l'essai (ex. 'baseline-nomic')")
    parser.add_argument("--seuil", type=float, default=None, help="remplace DISTANCE_MAX pour cet essai")
    parser.add_argument("--budget", type=int, default=None,
                        help="remplace JUDGE_MAX_CHARS (mode 'recopie') : caractères d'extraits montrés au juge")
    parser.add_argument("--mode", choices=["phrases", "recopie"], default=None,
                        help="méthode de la preuve : 'phrases' (Mistral choisit un numéro) ou 'recopie'")
    parser.add_argument("--sim-min", type=float, default=None, dest="sim_min",
                        help="similarité minimale (cosinus) de la phrase choisie ; rejette les choix trop éloignés")
    parser.add_argument("--sans-verification", action="store_true", dest="sans_verif",
                        help="désactive la 2e étape (le juge vérifie par OUI/NON la phrase choisie)")
    parser.add_argument("--numeros", default="",
                        help="ne garde que ces questions, ex. 3,6,7 (numéros affichés par une exécution complète)")
    parser.add_argument("--rapide", action="store_true", help="ne rédige aucune réponse (origine seulement)")
    parser.add_argument("--un-sur", type=int, default=1, dest="un_sur",
                        help="ne garde qu'une question sur N (aperçu rapide)")
    parser.add_argument("--sans-cache", action="store_true", help="ignore le cache du juge")
    args = parser.parse_args()

    if args.seuil is not None:
        core.DISTANCE_MAX = args.seuil
    if args.budget is not None:
        core.JUDGE_MAX_CHARS = args.budget
    if args.sim_min is not None:
        core.PHRASE_SIM_MIN = args.sim_min
    if args.sans_verif:
        core.VERIF_ACTIVE = False
    if args.mode is not None:
        core.PREUVE_MODE = args.mode   # avant load_judge_llm : la longueur de sortie en dépend

    with open(EVAL_FILE, encoding="utf-8-sig") as f:
        cas_list = json.load(f)

    total_complet = len(cas_list)
    numeros = list(range(1, total_complet + 1))          # numéro d'origine de chaque question
    if args.numeros.strip():
        voulus = {int(n) for n in args.numeros.replace(" ", "").split(",") if n.isdigit()}
        numeros = [n for n in numeros if n in voulus]
    elif args.un_sur > 1:
        numeros = numeros[:: args.un_sur]
    selection = [(n, cas_list[n - 1]) for n in numeros]
    cas_list = [c for _, c in selection]

    vs = core.load_vectorstore()
    llm = core.load_llm()
    judge = core.load_judge_llm()
    if not args.sans_cache:
        judge = JugeEnCache(judge)
    qa = PromptTemplate(template=core.QA_PROMPT_TEMPLATE, input_variables=["context", "question"])

    print(f"Essai : {args.nom or '(sans nom)'} | embeddings : {core.EMBEDDING_MODEL} | LLM : {core.LLM_MODEL} | "
          f"seuil : {core.DISTANCE_MAX} | preuve : {core.PREUVE_MODE} | "
          f"vérification : {'oui' if core.VERIF_ACTIVE else 'non'} | "
          f"{len(cas_list)} questions{' | rapide' if args.rapide else ''}\n")

    lignes, par_categorie, compteurs = [], defaultdict(lambda: [0, 0]), defaultdict(int)
    t_debut = time.time()
    interrompu = arret_ollama = False

    deux_phases = core.PREUVE_MODE == "phrases"
    if not deux_phases:
        print("ℹ️  PREUVE_MODE = 'recopie' : le traitement en deux phases n'est pas disponible, "
              "retour à l'ancien mode question par question (bascules de modèle à chaque question).\n")

    try:
        if deux_phases:
            # --- PHASE 1 : recherche + classement pour TOUTES les questions (embeddings seuls) ---
            print(f"Phase 1/2 — recherche vectorielle ({core.EMBEDDING_MODEL}) sur {len(selection)} question(s)...")
            t_phase1 = time.time()
            questions = [cas["question"] for _, cas in selection]
            lot = core.preparer_lot(vs, questions)
            print(f"   terminé en {time.time() - t_phase1:.0f}s\n")

            # --- PHASE 2 : jugement pour TOUTES les questions (mistral seul, en continu) ---
            print(f"Phase 2/2 — jugement ({core.LLM_MODEL}) sur {len(selection)} question(s)...")

            def executer():
                return core.juger_lot(judge, llm, qa, lot, generer_reponses=not args.rapide)

            try:
                traces = appeler_avec_reprise_lot(executer)
            except Exception as err:
                print(f"\n💥 Échec de la phase de jugement : {type(err).__name__} : {str(err)[:150]}")
                traces = None

            if traces is None:
                for numero, cas in selection:
                    ligne = nouvelle_ligne(numero, cas, "ERREUR", erreur="échec de la phase de jugement (lot)")
                    ligne["essai"] = args.nom
                    lignes.append(ligne)
                    compteurs["ERREUR"] += 1
                    par_categorie[cas["categorie"]][1] += 1
                arret_ollama = True
            else:
                for (numero, cas), trace in zip(selection, traces):
                    par_categorie[cas["categorie"]][1] += 1
                    statut = statut_du_cas(cas, trace)
                    compteurs[statut] += 1
                    par_categorie[cas["categorie"]][0] += statut == "OK"

                    obtenu = "📚" if trace["origine"] == "documents" else "🌐"
                    attendu = "📚" if cas["attendu"] == "documents" else "🌐"
                    tps = trace["temps"]
                    print(f"[{numero:2d}/{total_complet}] {ICONES[statut]} {statut:16s} attendu {attendu} obtenu {obtenu}  "
                          f"{cas['question'][:52]}  (juge {tps['juge']:.0f}s)")

                    ligne = nouvelle_ligne(numero, cas, statut, trace)
                    ligne["essai"] = args.nom
                    lignes.append(ligne)
        else:
            # --- Ancien mode : question par question (mode "recopie" uniquement) ---
            erreurs_de_suite = 0
            for i, (numero, cas) in enumerate(selection, start=1):
                t0 = time.time()
                par_categorie[cas["categorie"]][1] += 1
                try:
                    trace = appeler_avec_reprise(
                        vs, llm, qa, cas["question"], judge_llm=judge, generer_reponses=not args.rapide,
                    )
                    erreurs_de_suite = 0
                except Exception as err:
                    erreurs_de_suite += 1
                    compteurs["ERREUR"] += 1
                    message = f"{type(err).__name__} : {str(err)[:100]}"
                    print(f"[{numero:2d}/{total_complet}] {ICONES['ERREUR']} ERREUR            "
                          f"{cas['question'][:52]}  ({message})")
                    ligne = nouvelle_ligne(numero, cas, "ERREUR", erreur=message)
                    ligne["essai"] = args.nom
                    lignes.append(ligne)
                    if erreurs_de_suite >= ERREURS_MAX_DE_SUITE:
                        arret_ollama = True
                        print("\n⏹  Ollama ne répond plus : arrêt.")
                        break
                    continue

                statut = statut_du_cas(cas, trace)
                compteurs[statut] += 1
                par_categorie[cas["categorie"]][0] += statut == "OK"

                obtenu = "📚" if trace["origine"] == "documents" else "🌐"
                attendu = "📚" if cas["attendu"] == "documents" else "🌐"
                tps = trace["temps"]
                print(f"[{numero:2d}/{total_complet}] {ICONES[statut]} {statut:16s} attendu {attendu} obtenu {obtenu}  "
                      f"{cas['question'][:52]}  ({time.time() - t0:.0f}s : juge {tps['juge']:.0f}s)")

                ligne = nouvelle_ligne(numero, cas, statut, trace)
                ligne["essai"] = args.nom
                lignes.append(ligne)
    except KeyboardInterrupt:
        interrompu = True
        print("\n⏹  Interrompu : résumé des questions déjà traitées.")

    total = len(lignes)
    if total == 0:
        print("Aucune question traitée.")
        return

    print("\n" + "=" * 78)
    partiel = interrompu or arret_ollama or total < len(cas_list)
    print(f"Score global : {compteurs['OK']}/{total} ({100 * compteurs['OK'] / total:.0f} %)"
          f"{'   [PARTIEL]' if partiel else ''}   durée : {time.time() - t_debut:.0f}s")
    print(f"🚨 Fausses étiquettes 📚 : {compteurs['FAUSSE_ETIQUETTE']}   (doit rester à 0)")
    print(f"❌ Mauvaises sources     : {compteurs['MAUVAISE_SOURCE']}")
    print(f"⚠️  Rejets prudents       : {compteurs['REJET_PRUDENT']}   (réponse correcte, étiquette 🌐 trop sévère)")
    if compteurs["ERREUR"]:
        print(f"💥 Erreurs Ollama        : {compteurs['ERREUR']}   (questions à refaire)")
    if not args.sans_cache:
        print(f"⚡ Juge : {judge.hits} réponse(s) reprise(s) du cache sur {total}")
    mesures = [l for l in lignes if l["statut"] != "ERREUR"]
    cl = [l["secondes_classement"] for l in mesures if l["classement"]]
    if cl:
        print(f"⏱  Dont classement des phrases (embeddings) : {sum(cl) / len(cl):.0f}s en moyenne")
    jg = [l["secondes_juge"] for l in mesures if l["secondes_juge"]]
    if jg:
        print(f"⏱  Temps moyen du juge (hors cache) : {sum(jg) / len(jg):.0f}s")

    sim_ok = sorted(l["similarite"] for l in lignes if l["statut"] == "OK" and l["obtenu"] == "documents"
                    and l["similarite"] not in ("", None))
    sim_faux = [l["similarite"] for l in lignes if l["statut"] == "FAUSSE_ETIQUETTE"
                and l["similarite"] not in ("", None)]
    if sim_ok or sim_faux:
        print("\nCalibration de la similarité (cosinus de la phrase choisie avec la question) :")
        if sim_ok:
            print(f"   📚 corrects : min {sim_ok[0]} | médiane {sim_ok[len(sim_ok) // 2]} | max {sim_ok[-1]}   ({len(sim_ok)} cas)")
        if sim_faux:
            print(f"   🚨 fausses étiquettes : {sorted(sim_faux)}")
        if sim_ok and sim_faux and max(sim_faux) < sim_ok[0]:
            print(f"   ➡️  un seuil entre {max(sim_faux)} et {sim_ok[0]} (--sim-min) écarterait les fausses "
                  f"étiquettes sans perdre les corrects")

    print("\nPar catégorie :")
    for cat, (ok, n) in sorted(par_categorie.items()):
        print(f"   {cat:30s} {ok}/{n}")

    ecarts = [l for l in lignes if l["statut"] not in ("OK", "ERREUR")]
    if ecarts:
        print("\nDétail des écarts :")
        for l in ecarts:
            print(f"\n - [{l['statut']}] {l['question']}")
            print(f"     raison    : {l['raison']}")
            print(f"     distance  : {l['distance_min']}")
            if l["preuve"]:
                print(f"     preuve    : {l['preuve'][:140]}")
            if l["classement"]:
                print(f"     classement: {l['classement']} | similarité du choix : {l['similarite']}")
            if l["phrases_montrees"]:
                for k, ph in enumerate(l["phrases_montrees"].split(" || "), start=1):
                    print(f"     [{k}] {ph[:110]}")
            if l["reponse_juge"]:
                print(f"     juge dit  : {l['reponse_juge'][:260]!r}")
            if l["verdict_verification"]:
                print(f"     vérif.    : {l['verdict_verification'][:120]!r}")
            if l["motifs"]:
                print(f"     verdicts  : {l['motifs'][:400]}")

    colonnes = list(lignes[0].keys())
    if os.path.isfile(CSV_FILE):
        with open(CSV_FILE, encoding="utf-8-sig") as f:
            entete = f.readline().strip().split(",")
        if entete != colonnes:  # anciennes colonnes : on garde l'ancien fichier de côté au lieu de décaler les données
            sauvegarde = CSV_FILE.replace(".csv", f"_ancien_{datetime.now():%Y%m%d_%H%M%S}.csv")
            os.replace(CSV_FILE, sauvegarde)
            print(f"\nℹ️  Les colonnes ont changé : l'ancien {CSV_FILE} est conservé sous {sauvegarde}")
    nouveau = not os.path.isfile(CSV_FILE)
    with open(CSV_FILE, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=colonnes)
        if nouveau:
            writer.writeheader()
        writer.writerows(lignes)
    print(f"\nRésultats ajoutés à {CSV_FILE}")


if __name__ == "__main__":
    main()