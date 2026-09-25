"""
rag_core.py
-----------
Logique métier du cockpit MEMORIX orienté Data Science, IA et MLOps :
- Chargement des modèles Ollama (LLM & Embeddings) et Chroma
- Cours reconnus automatiquement depuis les métadonnées de l'index (aucune liste écrite en dur)
- Map-Reduce pédagogique spécialisé pour les résumés, fiches techniques, quiz, flashcards et podcast
- Évaluation d'argumentation orale technique et simulation de jury Data Science
- Cascade de réponse : documents -> preuve vérifiée -> réponse générale étiquetée
"""

import os

# Désactive la télémétrie Chroma AVANT tout import
os.environ["ANONYMIZED_TELEMETRY"] = "FALSE"
os.environ["CHROMA_TELEMETRY_ENABLED"] = "FALSE"

import re
import tempfile
import time
import unicodedata
import chromadb
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.documents import Document

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma

from ingest import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    PERSIST_DIR,
    split_text_by_sections,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    DATA_SCIENCE_SEPARATORS,
    load_pdf_with_ocr_fallback,
)

LLM_MODEL = "mistral"
TOP_K = 6
MAX_CHARS_PER_BATCH = 3500

# Cascade : au-delà de cette distance, on ne consulte même pas les documents.
# (Filtre grossier : la vraie décision est prise par la preuve, voir trouver_preuve.)
DISTANCE_MAX = 0.95
JUDGE_MAX_EXTRACTS = 6   # nombre maximal d'extraits montrés à Mistral
JUDGE_MAX_CHARS = 5000   # budget de caractères (la fenêtre du modèle est limitée à 3072 tokens)
JUDGE_MAX_TOKENS = 150   # la phrase de preuve est courte : on coupe le bavardage de Mistral
PREUVE_MIN_CHARS = 40    # une phrase plus courte n'est pas une explication (mode "recopie")

# Mode de la preuve :
#   "phrases" : le code extrait les phrases du cours, les classe par proximité avec la question,
#               et Mistral choisit un NUMÉRO (la phrase est exacte par construction, prompt court).
#   "recopie" : Mistral recopie une phrase, le code vérifie (Mistral reformule souvent : peu fiable).
PREUVE_MODE = "phrases"
PREUVE_NB_PHRASES = 10   # nombre de phrases montrées à Mistral
PHRASE_MIN_CHARS = 30    # une phrase de cours plus courte n'est pas une explication
CODE_MIN_CHARS = 12      # ligne de code minimale (questions d'usage)
CHOIX_MAX_TOKENS = 8     # Mistral ne répond qu'un numéro : inutile de le laisser justifier (lent)
PHRASE_SIM_MIN = 0.0     # similarité minimale (cosinus) de la phrase choisie ; 0 = désactivé (à calibrer)
VERIF_ACTIVE = True      # 2e étape : le juge vérifie par OUI/NON que la phrase choisie répond vraiment

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

# Mots qui signalent qu'on parle d'un document précis ("le cours de ML", "dans ce pdf"...).
COURS_KEYWORD_PATTERN = re.compile(r"\b(?:cours|documents?|supports?|livres?|pdf)\b", re.IGNORECASE)
# "cours 2", "document n°3", "le 2e cours" (le numéro est le préfixe du fichier : 02_...)
COURS_NUMBER_PATTERN = re.compile(
    r"\b(?:cours|documents?|supports?|livres?)\s*(?:n[°o]\s*)?(\d+)\b"
    r"|\b(\d+)\s*(?:er|e|ème|eme)?\s*(?:cours|documents?|supports?|livres?)\b",
    re.IGNORECASE,
)
# Séparateurs dans un libellé : "Computer Vision & Deep Learning" -> 2 noms utilisables
LABEL_SPLIT_PATTERN = re.compile(r"\s*(?:&|/|,|;|\bet\b|\band\b)\s*", re.IGNORECASE)

REFUSAL_PHRASE = "Je ne trouve pas cette information dans les documents fournis."
REFUSAL_CORE = "je ne trouve pas cette information"

# Une ligne de code (ou de liste, ou de titre Markdown) : jamais une explication en prose.
CODE_LINE_PATTERN = re.compile(
    r"^\s*(?:#|import\b|from\b|def\b|class\b|for\b|if\b|elif\b|else\b|while\b|with\b|try\b|except\b|return\b|print\b|[\]\)\}])"
    r"|=|→|\w\("
)
SENTENCE_SPLIT_PATTERN = re.compile(r'(?<=[.!?])\s+(?=[A-ZÀ-ÖØ-Ý"«(0-9])')

QUOTE_PATTERN = re.compile(r'["«“](.+?)["»”]', re.DOTALL)
LABEL_PREFIX_PATTERN = re.compile(r"^\s*(?:phrase\s+recopi[ée]e|r[ée]ponse)\s*:\s*", re.IGNORECASE)
CODE_OR_LIST_PATTERN = re.compile(r"→|=|\bimport\b|\bdef\b|\w\(")

# Question d'usage ("comment utiliser / lire / importer...") : le code du cours EST la réponse.
USAGE_QUESTION_PATTERN = re.compile(
    r"\b(utilis\w*|lire|lis|charger|importer|[ée]crire|impl[ée]menter|appeler|code|syntaxe|exemple)\b"
    r"|^\s*comment\s+(?:[\w'-]+\s+){0,2}?(?:s[ée]parer|remplacer|calculer|cr[ée]er|normaliser|standardiser|"
    r"entra[iî]ner|[ée]valuer|afficher|tracer|s[ée]lectionner|filtrer|trier|fusionner|supprimer|ajouter|"
    r"convertir|sauvegarder|enregistrer|appliquer|d[ée]finir|choisir|construire|fai(?:re|t)|installer|"
    r"traiter|nettoyer|encoder|pr[ée]dire|d[ée]couper)\b",
    re.IGNORECASE,
)
IDENT_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")

# Détecte une liste énumérative entre parenthèses : "(ResNet, EfficientNet, etc.)".
PARENTHESE_LISTE_PATTERN = re.compile(r"\(([^()]*,[^()]*)\)")

# Mots trop génériques pour servir de "sujet" d'une question
_STOPWORDS_QUESTION = {
    "que", "qui", "quoi", "comment", "quel", "quelle", "quels", "quelles",
    "est", "ce", "cette", "ces", "un", "une", "des", "les", "le", "la",
    "dans", "pour", "sur", "avec", "sans", "du", "de", "et", "ou", "en",
    "fait", "faire", "sert", "utilise", "utiliser", "fonctionne",
}


def _mots_cles(text: str) -> set:
    """Mots significatifs (≥4 lettres, hors mots vides) d'un texte, en minuscules."""
    return {
        m.lower() for m in IDENT_PATTERN.findall(text)
        if len(m) >= 4 and m.lower() not in _STOPWORDS_QUESTION
    }


def _sujet_seulement_en_liste(phrase: str, question: str) -> bool:
    """
    Vrai si le sujet de la question n'apparaît dans la phrase QUE dans une liste entre
    parenthèses : la phrase cite le sujet sans l'expliquer.
    """
    mots_question = _mots_cles(question)
    if not mots_question:
        return False
    listes = PARENTHESE_LISTE_PATTERN.findall(phrase)
    if not listes:
        return False
    mots_liste = set()
    for groupe in listes:
        mots_liste |= _mots_cles(groupe)
    if not (mots_question & mots_liste):
        return False
    hors_parentheses = PARENTHESE_LISTE_PATTERN.sub(" ", phrase)
    mots_hors = _mots_cles(hors_parentheses)
    return not (mots_question & mots_hors)


# ---------------------------------------------------------------------------
# Templates de Prompts de la cascade (documents -> preuve -> réponse générale)
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

PREUVE_PROMPT_TEMPLATE = """Voici des extraits de cours et une question.
Recopie MOT POUR MOT, entre guillemets, UNE phrase des extraits qui EXPLIQUE le sujet
de la question (une définition ou le principe de fonctionnement).
Si aucune phrase n'explique le sujet, réponds uniquement : AUCUNE
Un nom cité dans une liste, un titre, une consigne d'exercice ou du code ne compte pas.

Extraits :
{context}

Question : {question}

Phrase recopiée :"""

CHOIX_PROMPT_TEMPLATE = """Voici une question et des phrases extraites d'un cours, numérotées.
Choisis la phrase qui EXPLIQUE directement le sujet de la question (une définition, le principe, ou la façon de faire).
La phrase doit parler du sujet lui-même, sous le même nom ou son équivalent dans une autre langue.
Si aucune phrase ne le fait, réponds 0. Il vaut mieux répondre 0 que choisir une phrase qui parle d'autre chose.
Réponds uniquement par le numéro.

Exemple 1 :
[1] Le gradient indique la direction de plus forte pente.
[2] La variance mesure la dispersion des valeurs autour de leur moyenne.
Question : À quoi sert le gradient ?
Numéro : 1

Exemple 2 :
[1] Le gradient indique la direction de plus forte pente.
[2] La variance mesure la dispersion des valeurs autour de leur moyenne.
Question : Qu'est-ce que l'entropie ?
Numéro : 0

Phrases :
{phrases}

Question : {question}
Numéro :"""

VERIF_CONSIGNE_CONCEPT = "L'extrait donne-t-il la réponse, c'est-à-dire une définition ou une explication du sujet lui-même ?"
VERIF_CONSIGNE_USAGE = "La ligne de code montre-t-elle comment faire ce que demande la question ?"

VERIF_PROMPT_TEMPLATE = """Tu vérifies si un extrait de cours répond à une question.
{consigne}
Un extrait qui mentionne le sujet seulement en passant, ou qui parle d'un autre sujet, ne compte pas.
Réponds par un seul mot : OUI ou NON.

Exemple 1 :
Question : À quoi sert le gradient ?
Extrait : Le gradient indique la direction de plus forte pente.
Réponse : OUI

Exemple 2 :
Question : Qu'est-ce que l'entropie ?
Extrait : La variance mesure la dispersion des valeurs autour de leur moyenne.
Réponse : NON

Exemple 3 :
Question : Qu'est-ce que BERT ?
Extrait : On peut utiliser un modèle pré-entraîné (BERT, GPT, etc.) puis remplacer sa tête de classification.
Réponse : NON

Question : {question}
Extrait : {phrase}
Réponse :"""

# ---------------------------------------------------------------------------
# Templates de Prompts Pédagogiques
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

FLASHCARD_NUM_DEFAULT = 10

FLASHCARD_QUESTION_FORMAT = """FRONT: <question courte ou terme technique>
BACK: <réponse concise, 2 à 3 phrases maximum>"""

FLASHCARD_MAP_PROMPT = """Tu es un pédagogue en Data Science qui prépare des flashcards de révision.
À partir de cet extrait de cours, génère 3 flashcards distinctes portant UNIQUEMENT sur le texte.
Le recto (FRONT) est une question courte ou un terme technique à définir.
Le verso (BACK) est la réponse, concise et précise (2 à 3 phrases maximum).

Format strict à respecter, une flashcard après l'autre :
""" + FLASHCARD_QUESTION_FORMAT + """

Extrait :
{text}

Flashcards :"""

FLASHCARD_BLOCK_PATTERN = re.compile(
    r"FRONT:\s*(?P<front>.+?)\s*\n\s*"
    r"BACK:\s*(?P<back>.+?)(?=\n\s*(?:\d+[.)]\s*)?FRONT:|\Z)",
    re.IGNORECASE | re.DOTALL,
)

PODCAST_DIALOGUE_PROMPT = """Tu écris le script d'un podcast pédagogique à deux voix sur un extrait
de cours en Data Science. Deux personnages : PROF (expert, précis, pédagogue) et ETUDIANT (curieux,
pose des questions simples, reformule pour vérifier sa compréhension).
Le dialogue doit expliquer les notions clés de l'extrait de façon naturelle et vivante, en 4 à 6
répliques alternées, en français. Chaque réplique est concise (1 à 3 phrases courtes, adaptées à
l'oral : pas de formules mathématiques complexes, pas de symboles).
Format strict, une réplique par ligne, alternée strictement PROF puis ETUDIANT :
PROF: <réplique>
ETUDIANT: <réplique>
PROF: <réplique>
ETUDIANT: <réplique>

Extrait :
{text}

Script :"""

PODCAST_BLOCK_PATTERN = re.compile(
    r"(PROF|ETUDIANT)\s*:\s*(.+?)(?=\n\s*(?:PROF|ETUDIANT)\s*:|\Z)",
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
        collection_name=COLLECTION_NAME,
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


def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def _label_names(book_id: str, label: str):
    """Noms sous lesquels on peut désigner un cours : parties du libellé + identifiant."""
    names = set()
    for part in LABEL_SPLIT_PATTERN.split(_strip_accents(label).lower()):
        part = part.strip()
        if len(part) >= 4:
            names.add(part)
    slug = book_id.replace("_", " ").lower()
    if len(slug) >= 4:
        names.add(slug)
    return names


def detect_book_in_question(question: str, books_info: dict | None, strict: bool = True) -> str | None:
    if not books_info:
        return None
    normalized = normalize_ordinals(question)

    for m in COURS_NUMBER_PATTERN.finditer(normalized):
        number = int(m.group(1) or m.group(2))
        for book_id, info in books_info.items():
            if info.get("number") and info["number"] == number:
                return book_id

    if strict and not COURS_KEYWORD_PATTERN.search(normalized):
        return None

    lowered = _strip_accents(question).lower()
    for book_id, info in books_info.items():
        for name in _label_names(book_id, info["label"]):
            if re.search(rf"\b{re.escape(name)}\b", lowered):
                return book_id
    return None


def detect_chapter_request(question: str) -> str | None:
    normalized = normalize_ordinals(question)
    has_keyword = CHAPTER_KEYWORD_PATTERN.search(normalized)
    number_match = NUMBER_PATTERN.search(normalized)
    if has_keyword and number_match:
        return number_match.group(0)
    return None


_BOOKS_CACHE: dict = {}


def get_books_info(vectorstore, refresh: bool = False) -> dict:
    key = id(vectorstore)
    if refresh or key not in _BOOKS_CACHE:
        raw = vectorstore.get(include=["metadatas"])
        info = {}
        for m in raw["metadatas"]:
            book_id = m.get("book", "unknown")
            if book_id not in info:
                info[book_id] = {
                    "label": m.get("book_label") or book_id,
                    "number": int(m.get("course_no") or 0),
                }
        ordered = sorted(info.items(), key=lambda kv: (kv[1]["number"] == 0, kv[1]["number"], kv[0]))
        _BOOKS_CACHE[key] = dict(ordered)
    return _BOOKS_CACHE[key]


def get_all_books(vectorstore):
    return {book_id: info["label"] for book_id, info in get_books_info(vectorstore).items()}


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
    all_docs = [
        Document(page_content=content, metadata=meta)
        for content, meta in zip(raw["documents"], raw["metadatas"])
    ]
    numbered = [d for d in all_docs if d.metadata.get("chapter", "0") != "0"]
    docs = numbered or all_docs
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
# Évaluations, Quiz, Flashcards & Podcast
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


def parse_flashcards_text(text: str):
    cards = []
    for match in FLASHCARD_BLOCK_PATTERN.finditer(text):
        front = match.group("front").strip()
        back = match.group("back").strip()
        if front and back:
            cards.append({"front": front, "back": back})
    return cards


def generate_flashcards(llm, docs, num_cards=FLASHCARD_NUM_DEFAULT, progress_callback=None):
    def log(msg):
        if progress_callback:
            progress_callback(msg)

    if not docs:
        return [], None

    chapters = group_docs_by_chapter(docs)
    all_cards = []
    raw_parts = []

    for chapter_number, chapter_docs in chapters:
        batches = batch_documents(chapter_docs)
        log(f"Section {chapter_number} : génération de flashcards ({len(batches)} lot(s))...")

        for batch in batches:
            text = "\n\n".join(d.page_content for d in batch)
            result = llm.invoke(FLASHCARD_MAP_PROMPT.format(text=text))
            raw_parts.append(result)
            for c in parse_flashcards_text(result):
                c["chapter"] = chapter_number
                all_cards.append(c)

    raw_text = "\n\n".join(raw_parts)
    if len(all_cards) <= num_cards:
        return all_cards, raw_text

    # Répartit les cartes retenues entre les sections (comme generate_quiz), pour éviter
    # qu'une seule section très fournie monopolise le paquet.
    by_chapter = {}
    for c in all_cards:
        by_chapter.setdefault(c["chapter"], []).append(c)

    selected = []
    chapters_cycle = list(by_chapter.keys())
    idx = 0
    while len(selected) < num_cards and any(by_chapter.values()):
        ch = chapters_cycle[idx % len(chapters_cycle)]
        if by_chapter[ch]:
            selected.append(by_chapter[ch].pop(0))
        idx += 1

    return selected, raw_text


def parse_podcast_text(text: str):
    turns = []
    for match in PODCAST_BLOCK_PATTERN.finditer(text):
        speaker = match.group(1).strip().lower()
        contenu = re.sub(r"\s+", " ", match.group(2).strip())
        if contenu:
            turns.append({"speaker": speaker, "text": contenu})
    return turns


def generate_podcast_script(llm, docs, progress_callback=None):
    """
    Génère un dialogue Prof/Étudiant par section du cours, à lire à voix haute côté
    navigateur. Retourne une liste plate de répliques : [{"speaker", "text", "chapter"}].
    """
    def log(msg):
        if progress_callback:
            progress_callback(msg)

    if not docs:
        return []

    chapters = group_docs_by_chapter(docs)
    script = []
    for chapter_number, chapter_docs in chapters:
        batches = batch_documents(chapter_docs)
        log(f"Section {chapter_number} : écriture du dialogue...")
        text = "\n\n".join(d.page_content for d in batches[0])
        result = llm.invoke(PODCAST_DIALOGUE_PROMPT.format(text=text))
        turns = parse_podcast_text(result)
        for t in turns:
            t["chapter"] = chapter_number
        script.extend(turns)
    return script


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


def load_judge_llm():
    """LLM du juge : température 0, pour que la même question donne toujours le même verdict."""
    tokens = CHOIX_MAX_TOKENS if PREUVE_MODE == "phrases" else JUDGE_MAX_TOKENS
    return OllamaLLM(model=LLM_MODEL, temperature=0, num_ctx=3072, num_predict=tokens)


def _judge_docs(docs):
    """Extraits montrés à Mistral : par ordre de proximité, dans la limite d'un budget de caractères."""
    selected, total = [], 0
    for d in docs[:JUDGE_MAX_EXTRACTS]:
        size = len(d.page_content)
        if selected and total + size > JUDGE_MAX_CHARS:
            break
        selected.append(d)
        total += size
    return selected


def _candidats_preuve(reponse: str):
    """
    Phrases candidates dans la réponse de Mistral :
    - celles entre guillemets s'il y en a ;
    - sinon (Mistral oublie souvent les guillemets) la réponse entière, puis chacune de ses lignes.
    Aucune de ces phrases n'est crue sur parole : _verifier_phrase les contrôle toutes.
    """
    reponse = LABEL_PREFIX_PATTERN.sub("", reponse.strip())
    candidats = [m.group(1).strip() for m in QUOTE_PATTERN.finditer(reponse)]
    if not candidats:
        candidats = [reponse] + [ligne.strip() for ligne in reponse.splitlines() if ligne.strip()]
    uniques, vus = [], set()
    for c in candidats:
        if c and c not in vus:
            vus.add(c)
            uniques.append(c)
    return uniques


def _verifier_phrase(phrase: str, question: str, context: str, shown):
    """
    Contrôle une phrase candidate. Retourne (extraits_qui_la_contiennent, motif) :
    la liste est vide (None) si la phrase est rejetée, et le motif dit pourquoi.

    - Prose : elle doit exister telle quelle dans les extraits (sinon reformulée ou inventée).
    - Code : accepté seulement pour une question d'usage, jamais pour une ligne de liste (→) ;
      Mistral reformate souvent le code, donc on vérifie que chaque nom (pandas, read_csv...)
      existe dans les extraits au lieu de comparer mot pour mot.
    """
    if len(phrase) < PREUVE_MIN_CHARS:
        return None, f"trop courte ({len(phrase)} caractères, minimum {PREUVE_MIN_CHARS})"

    if not CODE_OR_LIST_PATTERN.search(phrase):
        if _norm(phrase) not in _norm(context):
            return None, "absente des extraits (reformulée ou inventée)"
        if _sujet_seulement_en_liste(phrase, question):
            return None, "sujet cité seulement entre parenthèses (liste), pas expliqué"
        support = [d for d in shown if _norm(phrase) in _norm(d.page_content)]
        return (support or shown), "valide (prose)"

    if "→" in phrase:
        return None, "ligne de liste (→)"
    if not USAGE_QUESTION_PATTERN.search(question):
        return None, "code refusé : la question demande un concept, pas un usage"

    noms = IDENT_PATTERN.findall(phrase)
    manquants = [n for n in noms if n not in context]
    if not noms or manquants:
        return None, f"noms absents des extraits : {manquants or 'aucun nom détecté'}"

    hits = [sum(n in d.page_content for n in noms) for d in shown]
    best = max(hits)
    return [d for d, h in zip(shown, hits) if h == best], "valide (code)"


def _analyser_preuve_recopie(llm, docs, question: str, details: dict | None = None):
    """
    Mode "recopie" : demande à Mistral de recopier UNE phrase du cours qui explique le sujet, puis la vérifie
    en Python. Retourne (phrase, extraits_qui_la_contiennent), ou (None, []) si aucune preuve valide.

    details (facultatif) : dictionnaire rempli pour le diagnostic avec la réponse brute de Mistral,
    le nombre d'extraits montrés et le motif d'acceptation / de rejet de chaque phrase candidate.
    """
    shown = _judge_docs(docs)
    context = "\n\n".join(d.page_content for d in shown)
    reponse_brute = llm.invoke(PREUVE_PROMPT_TEMPLATE.format(context=context, question=question)).strip()
    reponse = LABEL_PREFIX_PATTERN.sub("", reponse_brute)

    if details is not None:
        details.update(reponse=reponse_brute, extraits=len(shown), caracteres=len(context), examens=[])

    if re.match(r"\W*aucune", reponse.lower()):
        if details is not None:
            details["examens"].append(("", "Mistral a répondu AUCUNE"))
        return None, []

    for phrase in _candidats_preuve(reponse):
        support, motif = _verifier_phrase(phrase, question, context, shown)
        if details is not None:
            details["examens"].append((phrase, motif))
        if support:
            return phrase, support
    return None, []


def _ajouter_candidat(candidats: dict, texte: str, est_code: bool, index_doc: int):
    """Ajoute une phrase/ligne candidate (dédoublonnée : les chunks se chevauchent)."""
    cle = _norm(texte)
    if cle in candidats:
        candidats[cle]["docs"].add(index_doc)
    else:
        candidats[cle] = {"texte": texte, "code": est_code, "docs": {index_doc}}


def _est_liste(phrase: str) -> bool:
    """
    Vrai pour une phrase qui ÉNUMÈRE des noms (« Solutions : plus de données, régularisation, validation
    croisée, … ») : elle cite les sujets sans les expliquer. Critère : au moins 3 virgules et une majorité
    de segments de 3 mots ou moins.
    """
    segments = phrase.split(",")
    if len(segments) - 1 < 3:
        return False
    courts = sum(1 for s in segments if len(re.findall(r"\w+", s)) <= 3)
    return courts / len(segments) >= 0.6


def _est_titre(ligne: str) -> bool:
    """Ligne courte sans ponctuation finale (« Pooling », « 2. NumPy ») : un titre, pas du texte."""
    return len(ligne) <= 45 and ligne[-1] not in ".!?:;,"


def _extraire_candidats(docs, avec_code: bool = False):
    """
    Découpe les extraits en phrases de prose (et, pour les questions d'usage, en lignes de code).
    Les lignes de code, de liste (→) et les titres Markdown ne sont JAMAIS des phrases de prose.
    Chaque candidat garde les positions des extraits qui le contiennent.
    """
    candidats: dict = {}
    for index_doc, doc in enumerate(docs):
        prose: list = []

        def vider():
            if not prose:
                return
            texte = " ".join(prose)
            prose.clear()
            for phrase in SENTENCE_SPLIT_PATTERN.split(texte):
                phrase = re.sub(r"\s+", " ", phrase).strip()
                if (len(phrase) >= PHRASE_MIN_CHARS
                        and (phrase[0].isupper() or phrase[0].isdigit())
                        and phrase[-1] in ".!?"
                        and not _est_liste(phrase)):
                    _ajouter_candidat(candidats, phrase, False, index_doc)

        for ligne in doc.page_content.split("\n"):
            ligne = ligne.strip()
            if not ligne:
                vider()
            elif CODE_LINE_PATTERN.search(ligne):
                vider()
                if avec_code and len(ligne) >= CODE_MIN_CHARS and not ligne.startswith("#"):
                    _ajouter_candidat(candidats, ligne, True, index_doc)
            elif _est_titre(ligne):
                vider()
            else:
                prose.append(ligne)
        vider()
    return list(candidats.values())


def _classer_candidats(embeddings, question: str, candidats):
    """
    Classe les candidats du plus proche au moins proche de la question (similarité cosinus des embeddings).
    Sans embeddings disponibles : ordre des extraits (déjà classés par proximité). Retourne (liste, méthode).
    """
    if embeddings is None or not candidats:
        return candidats, "ordre des extraits"
    try:
        q = embeddings.embed_query(question)
        vecteurs = embeddings.embed_documents([c["texte"] for c in candidats])
    except Exception:
        return candidats, "ordre des extraits"

    def cosinus(u, v):
        num = sum(x * y for x, y in zip(u, v))
        den = (sum(x * x for x in u) ** 0.5) * (sum(y * y for y in v) ** 0.5)
        return num / den if den else 0.0

    scores = [cosinus(q, v) for v in vecteurs]
    for c, s in zip(candidats, scores):
        c["score"] = s
    ordre = sorted(range(len(candidats)), key=lambda i: scores[i], reverse=True)
    return [candidats[i] for i in ordre], "embeddings"


def _verifier_par_le_juge(llm, question: str, phrase: str, est_code: bool):
    """
    2e étape : le juge répond OUI ou NON sur UNE seule phrase.
    Seul un « OUI » valide ; tout le reste rejette.
    """
    consigne = VERIF_CONSIGNE_USAGE if est_code else VERIF_CONSIGNE_CONCEPT
    reponse = llm.invoke(
        VERIF_PROMPT_TEMPLATE.format(consigne=consigne, question=question, phrase=phrase)
    ).strip()
    m = re.match(r"\W*(oui|non)\b", reponse.lower())
    return bool(m and m.group(1) == "oui"), reponse


def _preparer_candidats_phrases(docs, question: str, embeddings=None):
    """
    Partie 'embeddings' du mode phrases : extraction + classement des candidats.
    N'appelle JAMAIS le juge (mistral) — uniquement le modèle d'embeddings.
    Retourne (retenus, methode, usage, secondes_classement).
    """
    usage = bool(USAGE_QUESTION_PATTERN.search(question))
    candidats = _extraire_candidats(docs, avec_code=usage)
    # Écarte les phrases de prose qui ne font que CITER le sujet dans une liste entre
    # parenthèses (ex. "modèle pré-entraîné (ResNet, EfficientNet, etc.)") : le code
    # (avec_code) n'est pas concerné, cette règle ne vaut que pour la prose.
    candidats = [
        c for c in candidats
        if c["code"] or not _sujet_seulement_en_liste(c["texte"], question)
    ]
    t0 = time.perf_counter()
    classes, methode = _classer_candidats(embeddings, question, candidats)
    duree = time.perf_counter() - t0
    return classes[:PREUVE_NB_PHRASES], methode, usage, duree


def _juger_candidats_phrases(llm, docs, question: str, retenus, methode, usage,
                             secondes_classement=0.0, details: dict | None = None):
    """
    Partie 'juge' du mode phrases : choix + vérification par mistral.
    N'appelle JAMAIS les embeddings — uniquement le juge.
    Reprend le résultat de _preparer_candidats_phrases (calculé à l'avance, en phase 1).
    """
    if details is not None:
        details.update(
            reponse="", extraits=len(retenus), classement=methode, usage=usage,
            phrases=[c["texte"] for c in retenus], examens=[], secondes_classement=secondes_classement,
            scores=[round(c["score"], 3) if c.get("score") is not None else None for c in retenus],
        )

    if not retenus:
        if details is not None:
            details["examens"].append(("", "aucune phrase exploitable dans les extraits"))
        return None, []

    liste = "\n".join(f"[{i}] {c['texte']}" for i, c in enumerate(retenus, start=1))
    reponse = llm.invoke(CHOIX_PROMPT_TEMPLATE.format(phrases=liste, question=question)).strip()
    if details is not None:
        details["reponse"] = reponse

    m = re.match(r"\s*\[?\s*(\d+)\s*\]?", reponse)
    numero = int(m.group(1)) if m else -1
    if not 1 <= numero <= len(retenus):
        if details is not None:
            motif = "le juge a répondu 0 (aucune phrase n'explique le sujet)" if numero == 0 \
                else "réponse illisible ou numéro hors liste"
            details["examens"].append(("", motif))
        return None, []

    choisi = retenus[numero - 1]
    # Tout texte que le juge ajoute après le numéro (écho du prompt, justification,
    # continuation hallucinée) est ignoré : seul le numéro compte pour identifier la
    # phrase. La 2e étape (vérification OUI/NON) reste le vrai garde-fou si le juge
    # s'est trompé de numéro.

    sim = choisi.get("score")
    if details is not None:
        details["similarite"] = round(sim, 3) if sim is not None else None
    if PHRASE_SIM_MIN > 0 and sim is not None and sim < PHRASE_SIM_MIN:
        if details is not None:
            details["examens"].append((choisi["texte"], f"rejetée : similarité {sim:.2f} < {PHRASE_SIM_MIN:.2f}"))
        return None, []
    if VERIF_ACTIVE:
        accepte, verdict = _verifier_par_le_juge(llm, question, choisi["texte"], choisi["code"])
        if details is not None:
            details["verification"] = verdict
        if not accepte:
            if details is not None:
                details["examens"].append((choisi["texte"], f"rejetée à la vérification (le juge répond : {verdict[:30]!r})"))
            return None, []

    if details is not None:
        details["examens"].append((choisi["texte"], f"choisie par le juge (n° {numero})" + (" et vérifiée (OUI)" if VERIF_ACTIVE else "")))
    return choisi["texte"], [docs[i] for i in sorted(choisi["docs"])]


def _analyser_preuve_phrases(llm, docs, question: str, details: dict | None = None, embeddings=None):
    """Mode 'phrases' en un seul appel (usage normal, question par question — voir app.py/chat.py)."""
    retenus, methode, usage, duree = _preparer_candidats_phrases(docs, question, embeddings)
    return _juger_candidats_phrases(llm, docs, question, retenus, methode, usage, duree, details)


def analyser_preuve(llm, docs, question: str, details: dict | None = None, embeddings=None):
    """
    Cherche dans les extraits UNE phrase du cours qui explique le sujet de la question.
    Retourne (phrase, extraits_qui_la_contiennent), ou (None, []) si aucune preuve valide.
    Le mode dépend de PREUVE_MODE ("phrases" par défaut, ou "recopie").
    """
    if PREUVE_MODE == "recopie":
        return _analyser_preuve_recopie(llm, docs, question, details)
    return _analyser_preuve_phrases(llm, docs, question, details, embeddings)


def trouver_preuve(llm, docs, question: str) -> str | None:
    """La phrase de preuve seule (voir analyser_preuve), ou None."""
    return analyser_preuve(llm, docs, question)[0]


def documents_suffisent(llm, docs, question: str) -> bool:
    """True seulement si une preuve valide a été trouvée dans les extraits."""
    return trouver_preuve(llm, docs, question) is not None


def answer_with_trace(vectorstore, llm, qa_prompt, question: str, judge_llm=None,
                      generer_reponses: bool = True):
    """
    Cascade complète, avec le détail de ce qui s'est passé :
    1. Recherche dans les documents (filtre grossier par distance).
    2. Le juge (température 0) doit citer une phrase du cours, vérifiée par le code.
    3. Si la preuve est valide : Mistral répond UNIQUEMENT à partir des extraits qui la contiennent.
    4. Sinon : réponse générale clairement étiquetée (docs = []).
    """
    t0 = time.perf_counter()
    books = get_books_info(vectorstore)
    book_id = detect_book_in_question(question, books, strict=True)
    results = retrieve_with_scores(vectorstore, question, book_id)
    close = [(d, s) for d, s in results if s <= DISTANCE_MAX]

    trace = {
        "question": question,
        "book_id": book_id,
        "distances": [round(s, 3) for _, s in results],
        "preuve": None,
        "docs": [],
        "warning": GENERAL_WARNING,
        "origine": "general",
        "temps": {"recherche": time.perf_counter() - t0, "juge": 0.0, "reponse": 0.0},
        "juge": {},
    }

    if not close:
        trace["raison"] = f"aucun extrait sous le seuil de distance ({DISTANCE_MAX})"
    else:
        docs = [d for d, _ in close]
        judge = judge_llm if judge_llm is not None else load_judge_llm()
        t1 = time.perf_counter()
        preuve, support = analyser_preuve(
            judge, docs, question, details=trace["juge"], embeddings=getattr(vectorstore, "embeddings", None),
        )
        trace["temps"]["juge"] = time.perf_counter() - t1

        if not preuve:
            trace["raison"] = "aucune preuve valide dans les extraits"
        else:
            trace["preuve"] = preuve
            if not generer_reponses:
                trace.update(
                    answer="", docs=support, warning=None, origine="documents",
                    raison="preuve valide (réponse non générée)",
                )
                return trace

            context = "\n\n".join(d.page_content for d in support)
            t2 = time.perf_counter()
            answer = enforce_refusal(llm.invoke(qa_prompt.format(context=context, question=question)))
            trace["temps"]["reponse"] = time.perf_counter() - t2
            if answer != REFUSAL_PHRASE:
                trace.update(
                    answer=answer, docs=support, warning=None, origine="documents",
                    raison="preuve valide dans les documents",
                )
                return trace
            trace["raison"] = "preuve valide, mais le modèle a refusé de répondre"

    if generer_reponses:
        t3 = time.perf_counter()
        trace["answer"] = llm.invoke(GENERAL_PROMPT_TEMPLATE.format(question=question))
        trace["temps"]["reponse"] += time.perf_counter() - t3
    else:
        trace["answer"] = ""
    return trace


def answer_question(vectorstore, llm, qa_prompt, question: str, progress_callback=None):
    """Version simple de answer_with_trace : retourne (réponse, documents_utilisés, avertissement)."""
    t = answer_with_trace(vectorstore, llm, qa_prompt, question)
    return t["answer"], t["docs"], t["warning"]


# ---------------------------------------------------------------------------
# Traitement par lot (évaluation) : deux phases pour éviter les rechargements
# de modèle Ollama (embeddings puis juge, chacun appelé une seule fois en continu).
# ---------------------------------------------------------------------------

def preparer_lot(vectorstore, questions, book_ids=None):
    """
    PHASE 1 (embeddings uniquement) : recherche vectorielle + extraction/classement des
    candidats pour toute une liste de questions. N'appelle jamais le juge (mistral).
    questions : liste de questions.
    book_ids  : liste de même longueur (ou None) : book_id ciblé pour chaque question.
    Retourne une liste de dict (un par question), à passer ensuite à juger_lot().
    """
    if book_ids is None:
        book_ids = [None] * len(questions)
    embeddings = getattr(vectorstore, "embeddings", None)
    lot = []
    for question, book_id in zip(questions, book_ids):
        t0 = time.perf_counter()
        results = retrieve_with_scores(vectorstore, question, book_id)
        t_recherche = time.perf_counter() - t0
        close = [(d, s) for d, s in results if s <= DISTANCE_MAX]
        docs = [d for d, _ in close]
        item = {
            "question": question,
            "book_id": book_id,
            "distances": [round(s, 3) for _, s in results],
            "docs": docs,
            "temps_recherche": t_recherche,
        }
        if docs:
            retenus, methode, usage, duree = _preparer_candidats_phrases(docs, question, embeddings)
            item.update(retenus=retenus, methode=methode, usage=usage, secondes_classement=duree)
        else:
            item.update(retenus=[], methode=None, usage=False, secondes_classement=0.0)
        lot.append(item)
    return lot


def juger_lot(judge, llm, qa_prompt, lot, generer_reponses=True):
    """
    PHASE 2 (juge uniquement) : à partir d'un lot préparé par preparer_lot(), exécute le
    choix + la vérification + (si demandé) la réponse rédigée, pour toutes les questions
    d'affilée. N'appelle jamais les embeddings.
    Retourne une liste de traces, dans le même format que answer_with_trace().
    NB : ne fonctionne qu'avec PREUVE_MODE == "phrases" (le mode par défaut). En mode
    "recopie", utilise answer_with_trace() question par question comme avant.
    """
    if PREUVE_MODE != "phrases":
        raise ValueError(
            "juger_lot() ne prend en charge que PREUVE_MODE == 'phrases'. "
            "En mode 'recopie', utilise answer_with_trace() question par question."
        )
    traces = []
    for item in lot:
        question = item["question"]
        trace = {
            "question": question,
            "book_id": item["book_id"],
            "distances": item["distances"],
            "preuve": None,
            "docs": [],
            "warning": GENERAL_WARNING,
            "origine": "general",
            "temps": {"recherche": item["temps_recherche"], "juge": 0.0, "reponse": 0.0},
            "juge": {},
        }
        docs = item["docs"]
        if not docs:
            trace["raison"] = f"aucun extrait sous le seuil de distance ({DISTANCE_MAX})"
        else:
            t1 = time.perf_counter()
            preuve, support = _juger_candidats_phrases(
                judge, docs, question, item["retenus"], item["methode"], item["usage"],
                item["secondes_classement"], details=trace["juge"],
            )
            trace["temps"]["juge"] = time.perf_counter() - t1
            if not preuve:
                trace["raison"] = "aucune preuve valide dans les extraits"
            else:
                trace["preuve"] = preuve
                if not generer_reponses:
                    trace.update(
                        answer="", docs=support, warning=None, origine="documents",
                        raison="preuve valide (réponse non générée)",
                    )
                    traces.append(trace)
                    continue

                context = "\n\n".join(d.page_content for d in support)
                t2 = time.perf_counter()
                answer = enforce_refusal(llm.invoke(qa_prompt.format(context=context, question=question)))
                trace["temps"]["reponse"] = time.perf_counter() - t2
                if answer != REFUSAL_PHRASE:
                    trace.update(
                        answer=answer, docs=support, warning=None, origine="documents",
                        raison="preuve valide dans les documents",
                    )
                    traces.append(trace)
                    continue
                trace["raison"] = "preuve valide, mais le modèle a refusé de répondre"

        if generer_reponses:
            t3 = time.perf_counter()
            trace["answer"] = llm.invoke(GENERAL_PROMPT_TEMPLATE.format(question=question))
            trace["temps"]["reponse"] += time.perf_counter() - t3
        else:
            trace["answer"] = ""
        traces.append(trace)
    return traces


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
    filtered = [d for d in docs if d.metadata.get("chapter", "0") != "0"] or list(docs)
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