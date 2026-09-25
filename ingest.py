"""
ingest.py
---------
Lit tous les documents du dossier `documents/` (PDF, TXT, Markdown), les découpe en morceaux
(chunks) adaptés aux contenus techniques (Data Science, code Python, formules),
détecte les chapitres/sections ainsi que l'identité de chaque document,
et les indexe dans une base vectorielle Chroma locale via Ollama.

Identité d'un document (« cours ») : automatique, à partir du nom de fichier.
    01_Python_Data_Science.pdf  ->  id "python_data_science", libellé "Python Data Science", numéro 1
    fiche_xgboost.md            ->  id "fiche_xgboost",       libellé "fiche xgboost",       numéro 0

Pour choisir soi-même l'id et le libellé, créer `documents/catalog.json` :
    {
      "01_Python_Data_Science.pdf": {"id": "python_ds", "label": "Python & Data Science"}
    }
(Garder les anciens id permet de conserver l'historique de progression de progress.db.)

L'index est reconstruit à zéro à chaque lancement : relancer ce script ne crée jamais de doublons.

Usage:
    python ingest.py
"""

import os

# Désactive la télémétrie Chroma AVANT tout import
os.environ["ANONYMIZED_TELEMETRY"] = "FALSE"
os.environ["CHROMA_TELEMETRY_ENABLED"] = "FALSE"

import json
import re
import unicodedata
import chromadb
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# --- Support OCR (optionnel) -------------------------------------------
try:
    import pytesseract
    from PIL import Image
    import pymupdf
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

OCR_MIN_CHARS_PER_PAGE = 30
OCR_LANG = "fra"
OCR_DPI = 200

DOCUMENTS_DIR = "documents"
CATALOG_FILE = os.path.join(DOCUMENTS_DIR, "catalog.json")
PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "memorix_ds_collection"
EMBEDDING_MODEL = "nomic-embed-text"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

# Séparateurs ordonnés pour respecter les blocs de code Python, les équations et les sections
DATA_SCIENCE_SEPARATORS = [
    "\n## ",
    "\n### ",
    "\n#### ",
    "\n```python",
    "\n```",
    "\nclass ",
    "\ndef ",
    "\n\n",
    "\n",
    " ",
    "",
]

SECTION_NUMBER_PATTERN = re.compile(
    r"(?<![\d./])(\d{1,2})\.\s+[A-ZÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ]"
)

CHAPTER_PATTERNS = [
    SECTION_NUMBER_PATTERN,
    re.compile(r"chapitre\s+(\d+)", re.IGNORECASE),
    re.compile(r"chapter\s+(\d+)", re.IGNORECASE),
    re.compile(r"partie\s+(\d+)", re.IGNORECASE),
    re.compile(r"module\s+(\d+)", re.IGNORECASE),
    re.compile(r"le[cç]on\s+(\d+)", re.IGNORECASE),
    re.compile(r"lesson\s+(\d+)", re.IGNORECASE),
    re.compile(r"unit[ée]\s+(\d+)", re.IGNORECASE),
    re.compile(r"s[ée]ance\s+(\d+)", re.IGNORECASE),
]

# Préfixe numérique d'un nom de fichier : "01_", "2-", "03 " ...
NUMBER_PREFIX_PATTERN = re.compile(r"^(\d+)[\s_.\-]+")


# ---------------------------------------------------------------------------
# Identité des documents (plus aucune liste de cours écrite en dur)
# ---------------------------------------------------------------------------

def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def slugify(text: str) -> str:
    """'Machine Learning (v2)' -> 'machine_learning_v2'."""
    text = _strip_accents(text).lower()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def load_catalog(path: str = CATALOG_FILE) -> dict:
    """Lit documents/catalog.json ({nom_de_fichier: {"id", "label"}}). {} si absent ou illisible."""
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8-sig") as f:  # utf-8-sig : tolère le BOM du Bloc-notes Windows
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"⚠️  {path} illisible ({e}) : identités déduites des noms de fichiers.")
        return {}
    if not isinstance(data, dict):
        print(f"⚠️  {path} doit contenir un objet JSON : identités déduites des noms de fichiers.")
        return {}
    return {k: v for k, v in data.items() if isinstance(v, dict)}


def book_identity(source: str, catalog: dict | None = None):
    """
    Retourne (book_id, label, number) pour un fichier source.
    Le catalogue, s'il mentionne le fichier, a priorité sur la déduction automatique.
    """
    filename = os.path.basename(source)
    stem = os.path.splitext(filename)[0]

    number = 0
    stem_clean = stem
    m = NUMBER_PREFIX_PATTERN.match(stem)
    if m:
        number = int(m.group(1))
        stem_clean = stem[m.end():]

    book_id = slugify(stem_clean) or slugify(stem) or "document"
    label = re.sub(r"[_\-]+", " ", stem_clean).strip() or stem

    catalog_lc = {k.lower(): v for k, v in (catalog or {}).items()}
    entry = catalog_lc.get(filename.lower())
    if entry:
        book_id = slugify(str(entry.get("id", ""))) or book_id
        label = str(entry.get("label") or label)

    return book_id, label, number


def assign_identities(sources, catalog: dict | None = None):
    """
    Retourne {source: (book_id, label, number, origine)} pour tous les fichiers.
    Les fichiers du catalogue sont traités en premier ; deux fichiers non catalogués
    qui donneraient le même id reçoivent un suffixe (_2, _3...) pour ne pas se mélanger.
    """
    catalog_lc = {k.lower() for k in (catalog or {})}
    ordered = sorted(sources, key=lambda s: (os.path.basename(s).lower() not in catalog_lc, s))

    identities = {}
    taken = set()
    for source in ordered:
        book_id, label, number = book_identity(source, catalog)
        if os.path.basename(source).lower() in catalog_lc:
            origin = "catalogue"
        else:
            origin = "déduit du nom"
            base, n = book_id, 2
            while book_id in taken:
                book_id = f"{base}_{n}"
                n += 1
        taken.add(book_id)
        identities[source] = (book_id, label, number, origin)
    return identities


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------

def _ocr_page_text(pdf_path: str, page_number: int, dpi: int = OCR_DPI, lang: str = OCR_LANG) -> str:
    """Applique l'OCR Tesseract sur une page PDF scannée si disponible."""
    if not OCR_AVAILABLE:
        return ""
    try:
        doc = pymupdf.open(pdf_path)
        page = doc[page_number]
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()
        return pytesseract.image_to_string(img, lang=lang)
    except Exception as e:
        print(f"  ! OCR impossible sur la page {page_number + 1} de {os.path.basename(pdf_path)} : {e}")
        return ""


def load_pdf_with_ocr_fallback(pdf_path: str):
    """Charge un PDF page par page avec fallback OCR si le texte natif est insuffisant."""
    pages = PyPDFLoader(pdf_path).load()
    ocr_used_on = []

    for page_doc in pages:
        if len(page_doc.page_content.strip()) < OCR_MIN_CHARS_PER_PAGE:
            page_number = page_doc.metadata.get("page", 0)
            ocr_text = _ocr_page_text(pdf_path, page_number)
            if len(ocr_text.strip()) > len(page_doc.page_content.strip()):
                page_doc.page_content = ocr_text
                ocr_used_on.append(page_number + 1)

    if ocr_used_on:
        print(f"  🔎 OCR appliqué sur {len(ocr_used_on)} page(s) de {os.path.basename(pdf_path)} : {ocr_used_on}")
    elif not OCR_AVAILABLE:
        empty_pages = [
            p.metadata.get("page", 0) + 1 for p in pages if len(p.page_content.strip()) < OCR_MIN_CHARS_PER_PAGE
        ]
        if empty_pages:
            print(
                f"  ⚠️  {len(empty_pages)} page(s) de {os.path.basename(pdf_path)} semblent sans texte "
                f"(scan/image) mais Tesseract OCR n'est pas configuré."
            )

    return pages


def load_documents():
    """Charge tous les documents PDF, TXT et Markdown du dossier cible (sous-dossiers inclus)."""
    docs = []

    pdf_paths = []
    for root, _, files in os.walk(DOCUMENTS_DIR):
        for name in files:
            if name.lower().endswith(".pdf"):
                pdf_paths.append(os.path.join(root, name))
    pdf_paths.sort()

    pdf_pages = []
    for path in pdf_paths:
        try:
            pdf_pages.extend(load_pdf_with_ocr_fallback(path))
        except Exception as e:
            print(f"  ! Erreur de chargement {path} : {e}")
    docs.extend(pdf_pages)
    print(f"  -> {len(pdf_paths)} fichier(s) PDF chargé(s) ({len(pdf_pages)} page(s))")

    # Markdown lu comme du texte brut (UTF-8) : garde les titres "##" et les blocs de code,
    # que les séparateurs du découpage exploitent, et n'exige aucune dépendance supplémentaire.
    loaders_config = [
        ("**/*.txt", TextLoader),
        ("**/*.md", TextLoader),
    ]

    for glob_pattern, loader_cls in loaders_config:
        loader = DirectoryLoader(
            DOCUMENTS_DIR,
            glob=glob_pattern,
            loader_cls=loader_cls,
            loader_kwargs={"encoding": "utf-8"},
            silent_errors=True,  # un fichier illisible est signalé mais ne bloque pas les autres
            show_progress=True,
        )
        try:
            loaded = loader.load()
            docs.extend(loaded)
            print(f"  -> {len(loaded)} fichier(s) chargé(s) pour {glob_pattern}")
        except Exception as e:
            print(f"  ! Erreur de chargement {glob_pattern} : {e}")

    return docs


# ---------------------------------------------------------------------------
# Sections et découpage
# ---------------------------------------------------------------------------

def find_section_matches(text: str):
    """Localise toutes les têtes de sections/chapitres dans le contenu brut."""
    matches = []
    for pattern in CHAPTER_PATTERNS:
        for m in pattern.finditer(text):
            matches.append((m.start(), m.group(1)))
    matches.sort(key=lambda item: item[0])

    deduped = []
    for pos, num in matches:
        if deduped and pos - deduped[-1][0] < 3:
            continue
        deduped.append((pos, num))
    return deduped


def split_text_by_sections(text: str):
    """Découpe le texte d'un document complet selon les frontières de sections."""
    matches = find_section_matches(text)

    if not matches:
        return [("0", text)]

    segments = []
    first_pos = matches[0][0]
    if first_pos > 0:
        intro = text[:first_pos].strip()
        if intro:
            segments.append(("0", intro))

    for i, (pos, num) in enumerate(matches):
        end = matches[i + 1][0] if i + 1 < len(matches) else len(text)
        segment_text = text[pos:end].strip()
        if segment_text:
            segments.append((num, segment_text))

    return segments


def group_documents_by_source(documents):
    """Concatène les pages extraites par document source pour maintenir la continuité textuelle."""
    groups = {}
    for doc in documents:
        source = doc.metadata.get("source", "inconnu")
        groups.setdefault(source, []).append(doc)

    full_texts = {}
    for source, docs in groups.items():
        docs.sort(key=lambda d: d.metadata.get("page", 0))
        full_texts[source] = "\n".join(d.page_content for d in docs)

    return full_texts


def build_tagged_chunks(full_texts, catalog: dict | None = None):
    """
    Découpe le contenu par section puis applique un text splitter adapté
    à la Data Science et au code pour préserver les blocs logiques.
    Chaque chunk porte : source, book (id), book_label, course_no, chapter, chunk_index.
    """
    if catalog is None:
        catalog = load_catalog()

    splitter = RecursiveCharacterTextSplitter(
        separators=DATA_SCIENCE_SEPARATORS,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    identities = assign_identities(list(full_texts), catalog)

    print("📇 Documents reconnus :")
    for source in sorted(identities):
        book_id, label, number, origin = identities[source]
        print(f"   - {source}  ->  id '{book_id}' | « {label} » | {origin}")

    final_docs = []

    for source, full_text in full_texts.items():
        book_id, label, number, _origin = identities[source]

        segments = split_text_by_sections(full_text)

        chunk_index = 0
        for chapter_number, segment_text in segments:
            if len(segment_text) <= CHUNK_SIZE:
                pieces = [segment_text]
            else:
                pieces = splitter.split_text(segment_text)

            for piece in pieces:
                chunk_index += 1
                final_docs.append(
                    Document(
                        page_content=piece,
                        metadata={
                            "source": source,
                            "book": book_id,
                            "book_label": label,
                            "course_no": number,
                            "chapter": chapter_number,
                            "chunk_index": chunk_index,
                        },
                    )
                )

    _warn_if_no_chapter_detected(final_docs)
    return final_docs


def _warn_if_no_chapter_detected(chunks):
    """Signale les documents sans section numérotée (ils restent interrogeables)."""
    labels = {}
    for c in chunks:
        labels.setdefault(c.metadata.get("book"), c.metadata.get("book_label", c.metadata.get("book")))

    for book_id, label in sorted(labels.items(), key=lambda kv: str(kv[0])):
        book_chunks = [c for c in chunks if c.metadata.get("book") == book_id]
        chapters = set(c.metadata.get("chapter", "0") for c in book_chunks)
        if chapters == {"0"}:
            print(
                f"\nℹ️  Aucune section numérotée détectée pour « {label} » : le document reste interrogeable, "
                f"et les synthèses, fiches et quiz travailleront sur le document entier."
            )


def main():
    if not os.path.isdir(DOCUMENTS_DIR) or not os.listdir(DOCUMENTS_DIR):
        print(f"⚠️  Le dossier '{DOCUMENTS_DIR}/' est vide.")
        print("   Ajoutez des fichiers .pdf, .txt ou .md avant de lancer l'indexation.")
        return

    print(f"📂 Chargement des documents depuis '{DOCUMENTS_DIR}/'...")
    documents = load_documents()

    if not documents:
        print("❌ Aucun document n'a pu être chargé.")
        return

    print(f"✅ {len(documents)} page(s)/fichier(s) chargé(s).")
    print("📎 Regroupement des pages par document source...")
    full_texts = group_documents_by_source(documents)

    print("🏷️  Découpage respectueux du code et des concepts Data Science...")
    chunks = build_tagged_chunks(full_texts)

    books_found = sorted(set(c.metadata["book"] for c in chunks))
    print(f"\n📘 {len(books_found)} document(s) indexé(s) :")
    for book_id in books_found:
        label = next(c.metadata["book_label"] for c in chunks if c.metadata["book"] == book_id)
        chapters = sorted(
            set(c.metadata["chapter"] for c in chunks if c.metadata["book"] == book_id),
            key=lambda x: (len(x), x),
        )
        print(f"   - {label} ({book_id}) : sections {chapters}")

    print(f"\n🧠 Vectorisation avec '{EMBEDDING_MODEL}' (Ollama)...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    print(f"💾 Stockage Chroma dans '{PERSIST_DIR}/'...")
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)  # reconstruction à zéro : jamais de doublons
    except Exception:
        pass  # première indexation : rien à supprimer

    Chroma.from_documents(
        client=client,
        collection_name=COLLECTION_NAME,
        documents=chunks,
        embedding=embeddings,
    )

    print("\n🎉 Indexation Data Science terminée avec succès !")
    print(f"   {len(chunks)} fragments indexés dans '{PERSIST_DIR}/'.")


if __name__ == "__main__":
    main()