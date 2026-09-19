"""
ingest.py
---------
Lit tous les documents du dossier `documents/`, les découpe en morceaux
(chunks) adaptés aux contenus techniques (Data Science, code Python, formules),
détecte les chapitres/sections ainsi que le domaine source, et les indexe
dans une base vectorielle Chroma locale via Ollama.

Usage:
    python ingest.py
"""

import os

# Désactive la télémétrie Chroma AVANT tout import
os.environ["ANONYMIZED_TELEMETRY"] = "FALSE"
os.environ["CHROMA_TELEMETRY_ENABLED"] = "FALSE"

import re
import chromadb
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
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
PERSIST_DIR = "chroma_db"
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

BOOK_KEYWORDS = {
    "python_ds": ["01_python", "data science", "data_science", "python", "pyhton", "pandas", "numpy"],
    "ml": ["02_machine", "machine learning", "machine_learning", "ml", "scikit", "sklearn"],
    "cv": ["03_computer", "computer vision", "computer_vision", "cv", "vision", "deep learning", "pytorch"],
}

BOOK_LABELS = {
    "python_ds": "Python & Data Science",
    "ml": "Machine Learning",
    "cv": "Computer Vision & Deep Learning",
}


def detect_book(source: str) -> str:
    """Déduit l'identifiant du cours à partir du nom de fichier source."""
    source_lower = os.path.basename(source).lower()
    for book_id, keywords in BOOK_KEYWORDS.items():
        if any(k in source_lower for k in keywords):
            return book_id
    return "unknown"


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
    """Charge tous les documents PDF, TXT et Markdown du dossier cible."""
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

    loaders_config = [
        ("**/*.txt", TextLoader),
        ("**/*.md", UnstructuredMarkdownLoader),
    ]

    for glob_pattern, loader_cls in loaders_config:
        loader = DirectoryLoader(
            DOCUMENTS_DIR,
            glob=glob_pattern,
            loader_cls=loader_cls,
            show_progress=True,
        )
        try:
            loaded = loader.load()
            docs.extend(loaded)
            print(f"  -> {len(loaded)} fichier(s) chargé(s) pour {glob_pattern}")
        except Exception as e:
            print(f"  ! Erreur de chargement {glob_pattern} : {e}")

    return docs


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


def build_tagged_chunks(full_texts):
    """
    Découpe le contenu par section puis applique un text splitter adapté
    à la Data Science et au code pour préserver les blocs logiques.
    """
    splitter = RecursiveCharacterTextSplitter(
        separators=DATA_SCIENCE_SEPARATORS,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    final_docs = []
    unknown_sources = set()

    for source, full_text in full_texts.items():
        book_id = detect_book(source)
        if book_id == "unknown":
            unknown_sources.add(source)

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
                            "chapter": chapter_number,
                            "chunk_index": chunk_index,
                        },
                    )
                )

    if unknown_sources:
        print("⚠️  Attention : ces fichiers n'ont pas été rattachés à un livre connu :")
        for src in sorted(unknown_sources):
            print(f"   - {src}")
        print("   -> Ajoute un mot-clé dans BOOK_KEYWORDS si nécessaire.")

    _warn_if_no_chapter_detected(final_docs)
    return final_docs


def _warn_if_no_chapter_detected(chunks):
    """Vérifie si des cours restent sans chapitre identifié (chapitre '0')."""
    books = sorted(set(c.metadata.get("book", "unknown") for c in chunks))
    for book_id in books:
        book_chunks = [c for c in chunks if c.metadata.get("book") == book_id]
        chapters = set(c.metadata.get("chapter", "0") for c in book_chunks)
        if chapters == {"0"}:
            label = BOOK_LABELS.get(book_id, book_id)
            print(f"\n⚠️  Aucune sous-section détectée pour '{label}'. Aperçu des premiers extraits :")
            for c in book_chunks[:2]:
                preview = c.page_content[:140].replace("\n", " ")
                print(f"   ... {preview} ...")


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
    print(f"📘 Domaines indexés : {[BOOK_LABELS.get(b, b) for b in books_found]}")

    for book_id in books_found:
        chapters = sorted(
            set(c.metadata["chapter"] for c in chunks if c.metadata["book"] == book_id),
            key=lambda x: (len(x), x),
        )
        print(f"   - {BOOK_LABELS.get(book_id, book_id)} : sections {chapters}")

    print(f"🧠 Vectorisation avec '{EMBEDDING_MODEL}' (Ollama)...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    print(f"💾 Stockage Chroma dans '{PERSIST_DIR}/'...")
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    Chroma.from_documents(
        client=client,
        collection_name="memorix_ds_collection",
        documents=chunks,
        embedding=embeddings,
    )

    print("\n🎉 Indexation Data Science terminée avec succès !")
    print(f"   {len(chunks)} fragments indexés dans '{PERSIST_DIR}/'.")


if __name__ == "__main__":
    main()