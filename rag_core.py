"""
rag_core.py
-----------
Logique métier du cockpit MEMORIX orienté Data Science, IA et MLOps :
- Chargement des modèles Ollama (LLM & Embeddings) et Chroma
- Détection fine des domaines (Python DS, Machine Learning, Computer Vision)
- Map-Reduce pédagogique spécialisé pour les résumés, fiches techniques et quiz
- Évaluation d'argumentation orale technique et simulation de jury Data Science
- Cascade de réponse : documents -> preuve vérifiée -> réponse générale étiquetée
"""

import os

# Désactive la télémétrie Chroma AVANT tout import
os.environ["ANONYMIZED_TELEMETRY"] = "FALSE"
os.environ["CHROMA_TELEMETRY_ENABLED"] = "FALSE"

import re
import tempfile
import chromadb
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.documents import Document

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma

from ingest import (
    BOOK_KEYWORDS,
    BOOK_LABELS,
    split_text_by_sections,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    DATA_SCIENCE_SEPARATORS,
    load_pdf_with_ocr_fallback,
)

PERSIST_DIR = "chroma_db"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "mistral"
TOP_K = 6
MAX_CHARS_PER_BATCH = 3500

# Cascade : au-delà de cette distance, on ne consulte même pas les documents.
# (Filtre grossier : la vraie décision est prise par la preuve, voir trouver_preuve.)
DISTANCE_MAX = 0.95
JUDGE_MAX_EXTRACTS = 4   # nombre d'extraits montrés à Mistral (fenêtre du modèle limitée)
PREUVE_MIN_CHARS = 40    # une phrase plus courte n'est pas une explication

CHAPTER_KEYWORD_PATTERN = re.compile(
    r"chapitre|chapter|partie|module|le[cç]on|lesson|unit[ée]|s[ée]ance|section|point|axe",
    re.IGNORECASE,
)
NUMBER_PATTERN = re.compile(r"\d+")

ORDINAL_WORDS = {
    "premier": "1", "première": "1", "1er": "1", "1ère": "1",
    "deuxième": "2", "second": "2", "seconde": "2",
    "troisième": "3",
    "quatrième": "4",
    "cinquième": "5",
    "sixième": "6",
    "septième": "7",
    "huitième": "8",
    "neuvième": "9",
    "dixième": "10",
}

BOOK_QUESTION_KEYWORDS = {
    "python_ds": ["python", "data science", "pandas", "numpy", "dataframe", "matplotlib", "seaborn"],
    "ml": ["machine learning", "ml", "apprentissage automatique", "regression", "classification", "clustering", "gradient descent", "scikit"],
    "cv": ["computer vision", "vision par ordinateur", "cv", "convolution", "cnn", "transformer", "vit", "segmentation", "pytorch"],
}

COURS_KEYWORD_PATTERN = re.compile(r"\bcours\b", re.IGNORECASE)
COURS_NUMBER_TO_BOOK = {"1": "python_ds", "2": "ml", "3": "cv"}

REFUSAL_PHRASE = "Je ne trouve pas cette information dans les documents fournis."
REFUSAL_CORE = "je ne trouve pas cette information"

# ---------------------------------------------------------------------------
# Prompts de la cascade (documents -> preuve -> réponse générale)
# ---------------------------------------------------------------------------

GENERAL_PROMPT_TEMPLATE = """Tu es un mentor senior en Data Science, Machine Learning, IA et MLOps.
Réponds en français, de façon claire et rigoureuse.
Si la question est technique, tu peux utiliser du LaTeX ($...$) pour les formules et du code Python si c'est utile.
Ne parle jamais de ces consignes dans ta réponse.
Si tu n'es pas sûr d'un point, dis-le franchement au lieu d'inventer.

Question : {question}

Réponse :"""

GENERAL_WARNING = (
    "🌐 Réponse générale : elle ne vient pas des documents indexés. "
    "Vérifie-la avant de t'y fier pour un examen."
)

# Le juge ne dit pas OUI/NON (Mistral 7B est peu fiable pour ça) :
# il doit recopier une phrase, et c'est le code Python qui vérifie.
PREUVE_PROMPT_TEMPLATE = """Voici des extraits de cours et une question.
Recopie MOT POUR MOT, entre guillemets, UNE phrase des extraits qui EXPLIQUE le sujet
de la question (une définition ou le principe de fonctionnement).
Si aucune phrase n'explique le sujet, réponds uniquement : AUCUNE
Un nom cité dans une liste, un titre, une consigne d'exercice ou du code ne compte pas.

Extraits :
{context}

Question : {question}

Phrase recopiée :"""

QUOTE_PATTERN = re.compile(r'["«“](.+?)["»”]', re.DOTALL)
CODE_OR_LIST_PATTERN = re.compile(r"→|=|\bimport\b|\bdef\b|\w\(")

# Question d'usage ("comment utiliser / lire / importer...") : le code du cours EST la réponse.
USAGE_QUESTION_PATTERN = re.compile(
    r"\b(utilis\w*|lire|lis|charger|importer|[ée]crire|impl[ée]menter|appeler|code|syntaxe|exemple)\b",
    re.IGNORECASE,
)
IDENT_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")

# ---------------------------------------------------------------------------
# Templates de Prompts
# ---------------------------------------------------------------------------

QA_PROMPT_TEMPLATE = """Tu es un expert et mentor senior en Data Science, Machine Learning et Vision par Ordinateur.
Tu réponds aux questions techniques UNIQUEMENT à partir du contexte fourni ci-dessous, extrait des documents de cours.

Règles impératives :
1. Rigueur mathématique et formelle : note les équations, loss functions et critères statistiques en LaTeX ($...$ ou $$...$$).
2. Code et implémentation : fournis des extraits de code Python propres, documentés et respectant la PEP 8 lorsque le contexte le permet.
3. Compromis d'ingénierie : détaille les nuances techniques (biais/variance, complexité mémoire et calculatoire) mentionnées dans le cours.
4. Si l'information demandée n'est PAS clairement présente dans le contexte, réponds EXACTEMENT et UNIQUEMENT :
   "{refusal_phrase}"
   N'ajoute aucune spéculation ni connaissance extérieure après cette phrase.

Contexte :
{{context}}

Question : {{question}}

Réponse technique détaillée et rigoureuse :""".format(refusal_phrase=REFUSAL_PHRASE)

CHAPTER_SUMMARY_PROMPT = """Tu es un chercheur et pédagogue en Data Science.
Voici un extrait de cours technique. Résume-le de façon claire et méthodique en conservant :
- Les concepts mathématiques, statistiques et algorithmiques clés
- Le fonctionnement des modèles, fonctions de coût et architectures
- Les métriques d'évaluation et cas d'usage pratiques

Extrait :
{text}

Synthèse analytique de cet extrait :"""

COMBINE_SUMMARIES_PROMPT = """Voici plusieurs synthèses partielles ordonnées d'un même module de Data Science.
Fusionne-les en une synthèse générale continue, parfaitement articulée, sans omettre aucune formule, métrique ou modèle majeur. Utilise des sous-titres clairs et le formalisme LaTeX.

Synthèses partielles :
{summaries}

Synthèse consolidée du module :"""

FICHE_EXTRACT_PROMPT = """Tu es un lead data scientist préparant une fiche de révision technique condensée.
Voici un extrait de cours. Extrais-en sous forme de liste ultra-précise :
- Définitions mathématiques & théorèmes indispensables
- Algorithmes, hyperparamètres critiques et signatures de fonctions clés (Python/Scikit-Learn/PyTorch)
- Métriques d'évaluation adaptées (ex: Precision/Recall/F1, ROC-AUC, BLEU, mAP)
- Pièges techniques (data leakage, surapprentissage, instabilités numériques)

Extrait :
{text}

Éléments clés extraits :"""

COMBINE_FICHE_SECTION_PROMPT = """Voici plusieurs extraits de fiche technique d'une même section de cours.
Assemble-les en une fiche de synthèse définitive, sans répétition et au format standard suivant (conserve ces titres exacts) :

🔑 Définitions & Fondements Mathématiques
📐 Algorithmes, Métriques & Implémentation
⚠️ Pièges, Fuites de données & Diagnostic de Modèle

Extraits :
{summaries}

Fiche technique consolidée :"""

QUIZ_NUM_QUESTIONS = 6

QUIZ_QUESTION_FORMAT = """Q: <question technique précise>
A) <option A>
B) <option B>
C) <option C>
D) <option D>
CORRECT: <A, B, C ou D>
EXPLICATION EN FRANÇAIS: <justification technique et mathématique concise en français>"""

QUIZ_MAP_PROMPT = """Tu es un enseignant en Data Science qui conçoit des examens rigoureux.
À partir de cet extrait de cours, génère 2 questions techniques à choix multiples (QCM) portant UNIQUEMENT sur le texte.
Inclus des calculs de dimension, des choix de métriques ou des comportements d'optimiseurs.
Une seule option valide, 3 distracteurs plausibles.

Format strict à respecter :
""" + QUIZ_QUESTION_FORMAT + """

Extrait :
{text}

Questions :"""

QUIZ_BLOCK_PATTERN = re.compile(
    r"Q:\s*(?P<question>.+?)\s*\n"
    r"A\)\s*(?P<a>.+?)\s*\n"
    r"B\)\s*(?P<b>.+?)\s*\n"
    r"C\)\s*(?P<c>.+?)\s*\n"
    r"D\)\s*(?P<d>.+?)\s*\n"
    r"CORRECT:\s*(?P<correct>[ABCD])\s*\n"
    r"EXPLICATION[^:]*:\s*(?P<explanation>.+?)(?=\n\s*Q:|\Z)",
    re.IGNORECASE | re.DOTALL,
)

ORAL_QUESTION_FORMAT = "Q: <problématique technique ouverte>"

ORAL_PROFILES = {
    "lead_ds": {
        "label": "🧪 Lead Data Scientist (R&D)",
        "context": "une revue d'architecture algorithmique menée par un Lead Data Scientist exigeant",
        "question_style": (
            "une question conceptuelle approfondie sur les limites d'un modèle, le comportement des gradients, "
            "ou l'impact mathématique d'une régularisation"
        ),
        "grading_focus": "la maîtrise théorique fondamentale, la clarté mathématique et l'esprit critique",
    },
    "mlops": {
        "label": "⚙️ Entretien Technique MLOps & Production",
        "context": "un entretien technique orienté déploiement, robustesse en production et monitoring de pipelines",
        "question_style": (
            "une mise en situation sur la latence d'inférence, la dérive des données (data drift), "
            "les pipelines CI/CD de modèles ou la conteneurisation"
        ),
        "grading_focus": "les contraintes de scalabilité, la sécurité des pipelines et le pragmatisme en ingénierie logicielle",
    },
    "soutenance": {
        "label": "🎓 Soutenance d'Ingénierie & Examen Académique",
        "context": "une soutenance officielle de projet Data Science devant un jury académique universitaire",
        "question_style": (
            "une question d'examen académique demandant de justifier le protocole expérimental, "
            "la validation croisée et le choix de la fonction objectif"
        ),
        "grading_focus": "la rigueur de la méthode scientifique, l'exhaustivité et le vocabulaire formel",
    },
    "business": {
        "label": "💼 Restitution Métier & Décideurs (Data Storytelling)",
        "context": "une présentation d'arbitrage technologique devant des décideurs métier et un Product Owner",
        "question_style": (
            "une question portant sur la rentabilité (ROI), l'explicabilité du modèle (XAI) ou la traduction "
            "d'une métrique technique (ex. AUC) en impact business concret"
        ),
        "grading_focus": "la clarté pédagogique sans jargon inutile tout en restant fidèle à la vérité scientifique",
    },
}

ORAL_MAP_PROMPT_TEMPLATE = """Tu prépares {context}.
Voici un extrait de cours en Data Science. Génère 1 question ouverte (sans QCM) avec le style : {question_style}.
La question doit solliciter l'analyse et la réflexion technique. Réponds en français.

Format :
""" + ORAL_QUESTION_FORMAT + """

Extrait :
{{text}}

Question :"""

ORAL_GRADE_PROMPT_TEMPLATE = """Tu es dans le contexte suivant : {context}.
Voici le cours de référence, la problématique soumise et l'argumentation orale formulée par l'étudiant.

Évalue la réponse UNIQUEMENT au regard du contenu de référence. Concentre ton analyse sur :
{grading_focus}.

Vérifie attentivement si les idées maîtresses sont évoquées, même formulées différemment.

Format strict requis :
NIVEAU: <Insuffisant, Partiel, Bon, ou Excellent>
FEEDBACK: <3 à 5 phrases d'analyse constructive s'adressant directement à l'étudiant avec "tu">

Contenu de référence :
{{reference}}

Question posée : {{question}}

Réponse de l'étudiant : {{answer}}

Évaluation :"""

ORAL_LEVEL_TO_SCORE = {
    "insuffisant": 0.0,
    "partiel": 0.5,
    "bon": 0.8,
    "excellent": 1.0,
}

ORAL_BLOCK_PATTERN = re.compile(
    r"NIVEAU:\s*(?P<level>\w+)\s*\n"
    r"FEEDBACK:\s*(?P<feedback>.+)",
    re.IGNORECASE | re.DOTALL,
)

SYNTHESIS_PROMPT = """Voici les résumés par section d'un cours de Data Science.
Rédige une vue d'ensemble panoramique très concise (3 à 4 phrases) soulignant l'articulation logique globale du domaine.

Résumés :
{summaries}

Introduction panoramique :"""

QA_SYNTHESIS_PROMPT = """Voici les analyses obtenues à travers différentes sections d'un cours pour répondre à la question ci-dessous.
Rédige une synthèse globale brève (2 à 3 phrases) synthétisant la réponse technique.

Question : {question}

Analyses par section :
{answers}

Synthèse globale :"""

QA_BATCH_PROMPT = """Tu es un assistant de recherche technique qui analyse un extrait de cours de Data Science.
Règles :
- Si l'extrait permet de répondre précisément à la question, formule une explication rigoureuse en français en te limitant STRICTEMENT aux informations présentes.
- Si l'extrait ne contient pas la solution, réponds UNIQUEMENT par le mot :
  PAS_TROUVE
- Ne fais aucune déduction hors contexte.

Extrait :
{text}

Question : {question}

Réponse :"""

QA_SECTION_COMBINE_PROMPT = """Voici les éléments techniques collectés dans une même section d'un cours pour répondre à la question.
Consolide-les en une réponse fluide, détaillée et techniquement exacte.

Question : {question}

Éléments :
{answers}

Réponse consolidée :"""

NOT_FOUND_PATTERN = re.compile(r"pas[_\s]?trouv[eé]", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Initialisation des modèles
# ---------------------------------------------------------------------------

def load_vectorstore():
    """Charge l'index vectoriel Chroma local de manière standardisée."""
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    return Chroma(
        client=client,
        collection_name="memorix_ds_collection",
        embedding_function=embeddings,
    )


def load_llm():
    """Instancie le LLM avec paramètres optimisés pour les réponses techniques."""
    return OllamaLLM(model=LLM_MODEL, temperature=0.2, num_ctx=3072)


def is_vectorstore_ready():
    """Valide l'existence physique du dossier Chroma."""
    return os.path.isdir(PERSIST_DIR)


# ---------------------------------------------------------------------------
# Résolution contextuelle
# ---------------------------------------------------------------------------

def normalize_ordinals(text: str) -> str:
    result = text
    for word, digit in ORDINAL_WORDS.items():
        result = re.sub(rf"\b{word}\b", digit, result, flags=re.IGNORECASE)
    return result


def detect_book_in_question(question: str) -> str | None:
    normalized = normalize_ordinals(question)

    if COURS_KEYWORD_PATTERN.search(normalized):
        number_match = NUMBER_PATTERN.search(normalized)
        if number_match and number_match.group(0) in COURS_NUMBER_TO_BOOK:
            return COURS_NUMBER_TO_BOOK[number_match.group(0)]

    question_lower = question.lower()
    for book_id, keywords in BOOK_QUESTION_KEYWORDS.items():
        if any(k in question_lower for k in keywords):
            return book_id
    return None


def detect_chapter_request(question: str) -> str | None:
    normalized = normalize_ordinals(question)
    has_keyword = CHAPTER_KEYWORD_PATTERN.search(normalized)
    number_match = NUMBER_PATTERN.search(normalized)
    if has_keyword and number_match:
        return number_match.group(0)
    return None


def get_all_books(vectorstore):
    raw = vectorstore.get(include=["metadatas"])
    book_ids = sorted(set(m.get("book", "unknown") for m in raw["metadatas"]))
    return {b: BOOK_LABELS.get(b, b) for b in book_ids}


def get_sections_for_book(vectorstore, book_id: str):
    raw = vectorstore.get(where={"book": book_id}, include=["metadatas"])
    chapters = set(m.get("chapter", "0") for m in raw["metadatas"])
    chapters.discard("0")
    return sorted(chapters, key=lambda x: (len(x), x))


def get_books_with_chapter(vectorstore, chapter_number: str):
    raw = vectorstore.get(where={"chapter": chapter_number}, include=["metadatas"])
    return sorted(set(m.get("book", "unknown") for m in raw["metadatas"]))


def get_book_chunks(vectorstore, book_id: str):
    raw = vectorstore.get(where={"book": book_id}, include=["documents", "metadatas"])
    docs = [
        Document(page_content=content, metadata=meta)
        for content, meta in zip(raw["documents"], raw["metadatas"])
        if meta.get("chapter", "0") != "0"
    ]
    docs.sort(
        key=lambda d: (
            int(d.metadata.get("chapter", "0")) if d.metadata.get("chapter", "0").isdigit() else 999,
            d.metadata.get("chunk_index", 0),
        )
    )
    return docs


def get_chapter_chunks(vectorstore, chapter_number: str, book_id: str):
    raw = vectorstore.get(
        where={"$and": [{"chapter": chapter_number}, {"book": book_id}]},
        include=["documents", "metadatas"],
    )
    docs = [Document(page_content=content, metadata=meta) for content, meta in zip(raw["documents"], raw["metadatas"])]
    docs.sort(key=lambda d: (d.metadata.get("source", ""), d.metadata.get("chunk_index", 0)))
    return docs


# ---------------------------------------------------------------------------
# Moteur Map-Reduce par Chapitre
# ---------------------------------------------------------------------------

def batch_documents(docs, max_chars=MAX_CHARS_PER_BATCH):
    batches = []
    current_batch = []
    current_len = 0

    for doc in docs:
        doc_len = len(doc.page_content)
        if current_len + doc_len > max_chars and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_len = 0
        current_batch.append(doc)
        current_len += doc_len

    if current_batch:
        batches.append(current_batch)
    return batches


def group_docs_by_chapter(docs):
    order = []
    by_chapter = {}
    for d in docs:
        ch = d.metadata.get("chapter", "0")
        if ch not in by_chapter:
            by_chapter[ch] = []
            order.append(ch)
        by_chapter[ch].append(d)

    for ch in order:
        by_chapter[ch].sort(key=lambda d: d.metadata.get("chunk_index", 0))

    return [(ch, by_chapter[ch]) for ch in order]


def _run_by_chapter(llm, docs, extract_prompt_template, combine_prompt_template,
                     synthesis_prompt_template, progress_callback=None, section_label="Section"):
    def log(msg):
        if progress_callback:
            progress_callback(msg)

    chapters = group_docs_by_chapter(docs)
    sections = []

    for chapter_number, chapter_docs in chapters:
        batches = batch_documents(chapter_docs)
        log(f"{section_label} {chapter_number} : analyse en cours ({len(batches)} lot(s))...")

        source_text = "\n\n".join(d.page_content for d in chapter_docs)
        partials = []
        for i, batch in enumerate(batches, start=1):
            text = "\n\n".join(d.page_content for d in batch)
            prompt = extract_prompt_template.format(text=text)
            if len(batches) > 1:
                log(f"{section_label} {chapter_number} — traitement lot {i}/{len(batches)}...")
            partials.append(llm.invoke(prompt))

        if len(partials) == 1:
            result = partials[0]
        else:
            combined = "\n\n".join(f"[Segment {i + 1}]\n{p}" for i, p in enumerate(partials))
            result = llm.invoke(combine_prompt_template.format(summaries=combined))

        sections.append({
            "chapter": chapter_number,
            "label": f"{section_label} {chapter_number}" if chapter_number != "0" else None,
            "text": result.strip(),
            "source": source_text,
        })

    synthesis = None
    if len(sections) > 1:
        log("Synthèse globale du module...")
        section_summaries_text = "\n\n".join(
            f"[{s['label'] or 'Introduction'}] {s['text']}" for s in sections
        )
        synthesis = llm.invoke(synthesis_prompt_template.format(summaries=section_summaries_text)).strip()

    return {"sections": sections, "synthesis": synthesis}


def summarize_documents(llm, docs, empty_message, extract_prompt_template,
                        combine_prompt_template, progress_callback=None, section_label="Section"):
    if not docs:
        return empty_message

    result = _run_by_chapter(
        llm, docs, extract_prompt_template, combine_prompt_template, SYNTHESIS_PROMPT,
        progress_callback=progress_callback, section_label=section_label,
    )
    sections = result["sections"]
    if len(sections) == 1:
        return sections[0]["text"]

    parts = [f"## {s['label']}\n\n{s['text']}" if s["label"] else s["text"] for s in sections]
    detail = "\n\n---\n\n".join(parts)
    return f"## Vue d'ensemble\n\n{result['synthesis']}\n\n---\n\n{detail}"


def summarize_documents_structured(llm, docs, empty_message, extract_prompt_template,
                                   combine_prompt_template, progress_callback=None, section_label="Section"):
    if not docs:
        return {"empty": True, "message": empty_message, "sections": [], "synthesis": None}

    result = _run_by_chapter(
        llm, docs, extract_prompt_template, combine_prompt_template, SYNTHESIS_PROMPT,
        progress_callback=progress_callback, section_label=section_label,
    )
    result["empty"] = False
    result["message"] = None
    return result


# ---------------------------------------------------------------------------
# Évaluations & Entraînements
# ---------------------------------------------------------------------------

def parse_quiz_text(text: str):
    questions = []
    for match in QUIZ_BLOCK_PATTERN.finditer(text):
        questions.append({
            "question": match.group("question").strip(),
            "a": match.group("a").strip(),
            "b": match.group("b").strip(),
            "c": match.group("c").strip(),
            "d": match.group("d").strip(),
            "correct": match.group("correct").strip().upper(),
            "explanation": match.group("explanation").strip(),
        })
    return questions


def generate_quiz(llm, docs, num_questions=QUIZ_NUM_QUESTIONS, progress_callback=None):
    def log(msg):
        if progress_callback:
            progress_callback(msg)

    if not docs:
        return [], None

    chapters = group_docs_by_chapter(docs)
    all_questions = []
    raw_parts = []

    for chapter_number, chapter_docs in chapters:
        batches = batch_documents(chapter_docs)
        log(f"Section {chapter_number} : génération d'items QCM ({len(batches)} lot(s))...")

        for batch in batches:
            text = "\n\n".join(d.page_content for d in batch)
            prompt = QUIZ_MAP_PROMPT.format(text=text)
            result = llm.invoke(prompt)
            raw_parts.append(result)
            for q in parse_quiz_text(result):
                q["chapter"] = chapter_number
                all_questions.append(q)

    raw_text = "\n\n".join(raw_parts)
    if len(all_questions) <= num_questions:
        return all_questions, raw_text

    by_chapter = {}
    for q in all_questions:
        by_chapter.setdefault(q["chapter"], []).append(q)

    selected = []
    chapters_cycle = list(by_chapter.keys())
    idx = 0
    while len(selected) < num_questions and any(by_chapter.values()):
        ch = chapters_cycle[idx % len(chapters_cycle)]
        if by_chapter[ch]:
            selected.append(by_chapter[ch].pop(0))
        idx += 1

    return selected, raw_text


def generate_oral_questions(llm, docs, num_questions=3, profile_key="lead_ds", progress_callback=None):
    def log(msg):
        if progress_callback:
            progress_callback(msg)

    if not docs:
        return []

    profile = ORAL_PROFILES.get(profile_key, ORAL_PROFILES["lead_ds"])
    map_prompt_template = ORAL_MAP_PROMPT_TEMPLATE.format(
        context=profile["context"], question_style=profile["question_style"],
    )

    chapters = group_docs_by_chapter(docs)
    all_questions = []

    for chapter_number, chapter_docs in chapters:
        batches = batch_documents(chapter_docs)
        log(f"Section {chapter_number} : élaboration du sujet oral...")
        text = "\n\n".join(d.page_content for d in batches[0])
        result = llm.invoke(map_prompt_template.format(text=text)).strip()
        question_text = re.sub(r"^Q:\s*", "", result, flags=re.IGNORECASE).strip()

        if question_text:
            all_questions.append({
                "chapter": chapter_number,
                "question": question_text,
                "reference": text,
            })

    if len(all_questions) <= num_questions:
        return all_questions

    step = len(all_questions) / num_questions
    indices = [int(i * step) for i in range(num_questions)]
    return [all_questions[i] for i in indices]


def grade_oral_answer(llm, question: str, reference: str, answer: str, profile_key="lead_ds"):
    profile = ORAL_PROFILES.get(profile_key, ORAL_PROFILES["lead_ds"])
    grade_prompt_template = ORAL_GRADE_PROMPT_TEMPLATE.format(
        context=profile["context"], grading_focus=profile["grading_focus"],
    )
    prompt = grade_prompt_template.format(reference=reference, question=question, answer=answer)
    result = llm.invoke(prompt).strip()

    match = ORAL_BLOCK_PATTERN.search(result)
    if not match:
        return {"level": "Partiel", "feedback": result, "score": 0.5}

    level = match.group("level").strip().lower()
    feedback = match.group("feedback").strip()
    score = ORAL_LEVEL_TO_SCORE.get(level, 0.5)
    return {"level": match.group("level").strip().capitalize(), "feedback": feedback, "score": score}


def grade_quiz(questions, user_answers):
    results = []
    score = 0
    for q, user_answer in zip(questions, user_answers):
        is_correct = user_answer == q["correct"]
        if is_correct:
            score += 1
        results.append({
            **q,
            "user_answer": user_answer,
            "is_correct": is_correct,
        })
    return score, results


# ---------------------------------------------------------------------------
# Mode Chat & Recherche Augmentée
# ---------------------------------------------------------------------------

def enforce_refusal(answer: str) -> str:
    """Normalise toute réponse de refus en une phrase unique et reconnaissable."""
    if REFUSAL_CORE in answer.lower():
        return REFUSAL_PHRASE
    return answer


def _run_qa_by_chapter(llm, docs, question: str, progress_callback=None):
    def log(msg):
        if progress_callback:
            progress_callback(msg)

    chapters = group_docs_by_chapter(docs)
    sections = []

    for chapter_number, chapter_docs in chapters:
        batches = batch_documents(chapter_docs)
        log(f"Section {chapter_number} : recoupement ({len(batches)} lot(s))...")
        source_text = "\n\n".join(d.page_content for d in chapter_docs)

        found_in_section = []
        for i, batch in enumerate(batches, start=1):
            text = "\n\n".join(d.page_content for d in batch)
            prompt = QA_BATCH_PROMPT.format(text=text, question=question)
            if len(batches) > 1:
                log(f"Section {chapter_number} — lot {i}/{len(batches)}...")
            result = llm.invoke(prompt).strip()
            if not NOT_FOUND_PATTERN.search(result):
                found_in_section.append(result)

        if not found_in_section:
            continue

        if len(found_in_section) == 1:
            text = found_in_section[0]
        else:
            combined = "\n\n".join(f"[Extrait {i + 1}] {a}" for i, a in enumerate(found_in_section))
            text = llm.invoke(QA_SECTION_COMBINE_PROMPT.format(question=question, answers=combined)).strip()

        sections.append({
            "chapter": chapter_number,
            "label": f"Section {chapter_number}" if chapter_number != "0" else None,
            "text": text,
            "source": source_text,
        })

    synthesis = None
    if len(sections) > 1:
        log("Synthèse multi-sections...")
        answers_text = "\n\n".join(f"[{s['label'] or 'Introduction'}] {s['text']}" for s in sections)
        synthesis = llm.invoke(QA_SYNTHESIS_PROMPT.format(question=question, answers=answers_text)).strip()

    return {"sections": sections, "synthesis": synthesis}


def answer_question_full_book(llm, docs, question: str, progress_callback=None):
    if not docs:
        return REFUSAL_PHRASE

    result = _run_qa_by_chapter(llm, docs, question, progress_callback=progress_callback)
    sections = result["sections"]

    if not sections:
        return REFUSAL_PHRASE
    if len(sections) == 1:
        return sections[0]["text"]

    parts = [f"**D'après la section {s['chapter']} :** {s['text']}" if s["label"] else s["text"] for s in sections]
    detail = "\n\n".join(parts)
    return f"**Synthèse Data Science :** {result['synthesis']}\n\n---\n\n{detail}"


def answer_question_full_book_structured(llm, docs, question: str, progress_callback=None):
    if not docs:
        return {"empty": True, "sections": [], "synthesis": None}

    result = _run_qa_by_chapter(llm, docs, question, progress_callback=progress_callback)
    result["empty"] = len(result["sections"]) == 0
    return result


# ---------------------------------------------------------------------------
# Cascade de réponse : documents -> preuve vérifiée -> réponse générale
# ---------------------------------------------------------------------------

def retrieve_with_scores(vectorstore, question: str, book_id: str | None = None):
    """Retourne [(document, distance), ...]. Distance basse = extrait proche de la question."""
    kwargs = {"k": TOP_K}
    if book_id is not None:
        kwargs["filter"] = {"book": book_id}
    return vectorstore.similarity_search_with_score(question, **kwargs)


def _norm(text: str) -> str:
    """Minuscules + espaces/retours à la ligne aplatis, pour comparer deux textes."""
    return re.sub(r"\s+", " ", text).strip().lower()


def trouver_preuve(llm, docs, question: str) -> str | None:
    """
    Demande à Mistral de recopier UNE phrase du cours qui explique le sujet,
    puis vérifie en Python que :
      - la phrase existe vraiment dans les extraits (pas inventée),
      - elle est assez longue pour être une explication,
      - ce n'est ni du code ni une liste.
    Retourne la phrase, ou None si aucune preuve valide.
    """
    context = "\n\n".join(d.page_content for d in docs[:JUDGE_MAX_EXTRACTS])
    reponse = llm.invoke(PREUVE_PROMPT_TEMPLATE.format(context=context, question=question)).strip()

    if re.match(r"\W*aucune", reponse.lower()):
        return None

    match = QUOTE_PATTERN.search(reponse)
    if not match:
        return None

    phrase = match.group(1).strip()
    if len(phrase) < PREUVE_MIN_CHARS:
        return None

    est_code = bool(CODE_OR_LIST_PATTERN.search(phrase))

    # Cas 1 : phrase en prose -> elle doit exister telle quelle dans les extraits.
    if not est_code:
        return phrase if _norm(phrase) in _norm(context) else None

    # Cas 2 : code. Accepté seulement pour une question d'usage, jamais pour une ligne de liste (→).
    # Mistral reformate souvent le code (guillemets, retours à la ligne) : on ne compare pas
    # la phrase mot pour mot, on vérifie que chaque nom (pandas, read_csv...) existe dans les extraits.
    if USAGE_QUESTION_PATTERN.search(question) and "→" not in phrase:
        noms = IDENT_PATTERN.findall(phrase)
        if noms and all(n in context for n in noms):
            return phrase
    return None


def documents_suffisent(llm, docs, question: str) -> bool:
    """True seulement si une preuve valide a été trouvée dans les extraits."""
    return trouver_preuve(llm, docs, question) is not None


def answer_question(vectorstore, llm, qa_prompt, question: str, progress_callback=None):
    """
    Cascade :
    1. Recherche dans les documents (filtre grossier par distance).
    2. Mistral doit citer une phrase du cours, vérifiée par le code (trouver_preuve).
    3. Si la preuve est valide, Mistral rédige la réponse à partir du cours.
    4. Sinon, réponse générale clairement étiquetée (docs = []).

    Retourne (réponse, documents_utilisés, avertissement).
    """
    book_id = detect_book_in_question(question)
    results = retrieve_with_scores(vectorstore, question, book_id)
    close = [(d, s) for d, s in results if s <= DISTANCE_MAX]

    # Niveau 1 : les documents
    if close:
        docs = [d for d, _ in close]
        if documents_suffisent(llm, docs, question):
            context = "\n\n".join(d.page_content for d in docs)
            answer = enforce_refusal(llm.invoke(qa_prompt.format(context=context, question=question)))
            if answer != REFUSAL_PHRASE:
                warning = None
                if book_id is None:
                    books_in_result = sorted(set(d.metadata.get("book", "unknown") for d in docs))
                    if len(books_in_result) > 1:
                        labels = ", ".join(BOOK_LABELS.get(b, b) for b in books_in_result)
                        warning = (
                            f"Question transversale : les résultats recoupent plusieurs cours ({labels}). "
                            f"Précisez la matière ou le modèle pour une réponse isolée."
                        )
                return answer, docs, warning

    # Niveau 2 : connaissance générale, étiquetée
    answer = llm.invoke(GENERAL_PROMPT_TEMPLATE.format(question=question))
    return answer, [], GENERAL_WARNING


# ---------------------------------------------------------------------------
# Sessions Éphémères (Documents déposés en mémoire)
# ---------------------------------------------------------------------------

def process_uploaded_pdf(file_bytes: bytes, filename: str):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        pages = load_pdf_with_ocr_fallback(tmp_path)
    finally:
        os.remove(tmp_path)

    pages.sort(key=lambda d: d.metadata.get("page", 0))
    full_text = "\n".join(p.page_content for p in pages)
    segments = split_text_by_sections(full_text)

    splitter = RecursiveCharacterTextSplitter(
        separators=DATA_SCIENCE_SEPARATORS,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    docs = []
    chunk_index = 0
    for chapter_number, segment_text in segments:
        pieces = [segment_text] if len(segment_text) <= CHUNK_SIZE else splitter.split_text(segment_text)
        for piece in pieces:
            chunk_index += 1
            docs.append(Document(
                page_content=piece,
                metadata={
                    "source": filename,
                    "book": "upload",
                    "chapter": chapter_number,
                    "chunk_index": chunk_index,
                },
            ))

    return docs


def get_sections_in_docs(docs):
    chapters = set(d.metadata.get("chapter", "0") for d in docs)
    chapters.discard("0")
    return sorted(chapters, key=lambda x: (len(x), x))


def filter_whole_document(docs):
    filtered = [d for d in docs if d.metadata.get("chapter", "0") != "0"]
    filtered.sort(
        key=lambda d: (
            int(d.metadata.get("chapter", "0")) if d.metadata.get("chapter", "0").isdigit() else 999,
            d.metadata.get("chunk_index", 0),
        )
    )
    return filtered


def filter_document_section(docs, chapter_number: str):
    filtered = [d for d in docs if d.metadata.get("chapter") == chapter_number]
    filtered.sort(key=lambda d: d.metadata.get("chunk_index", 0))
    return filtered