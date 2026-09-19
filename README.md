# RAG Chatbot — Ollama + LangChain + Chroma

Chatbot qui répond à des questions à partir de tes propres documents (cours, PDF, notes...)
en utilisant la récupération augmentée par génération (RAG), 100% en local.

## Stack
- **LLM** : Mistral (via Ollama, local)
- **Embeddings** : nomic-embed-text (via Ollama, local)
- **Orchestration** : LangChain
- **Base vectorielle** : Chroma (locale, gratuite)

## Prérequis
- Python 3.10+
- [Ollama](https://ollama.com) installé sur ta machine

## Installation

```bash
# 1. Cloner / se placer dans le dossier du projet
cd rag-chatbot

# 2. Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate      # Windows : venv\Scripts\activate

# 3. Installer les dépendances Python
pip install -r requirements.txt

# 4. Télécharger les modèles Ollama nécessaires
ollama pull nomic-embed-text
ollama pull mistral

# 5. Lancer le serveur Ollama (dans un terminal séparé, laisse-le tourner)
ollama serve
```

## Utilisation

```bash
# 1. Mets tes documents (.pdf, .txt, .md) dans le dossier documents/

# 2. Indexe-les dans la base vectorielle
python ingest.py

# 3. Lance le chatbot
python chat.py
```

Tape ensuite tes questions. Le chatbot répond en citant les documents sources
utilisés, et refuse d'inventer une réponse s'il ne trouve rien de pertinent.

## Structure du projet

```
rag-chatbot/
├── documents/        # tes fichiers sources (non versionnés)
├── chroma_db/         # base vectorielle générée par ingest.py (non versionnée)
├── ingest.py          # découpe + indexe les documents
├── chat.py            # boucle de chat avec récupération de contexte
├── requirements.txt
└── README.md
```

## Points techniques à retenir (pour un entretien)
- **Chunking** avec chevauchement (`RecursiveCharacterTextSplitter`, 1000/200)
  pour éviter de couper une idée au milieu.
- **Prompt anti-hallucination** : le LLM est explicitement instruit de dire
  qu'il ne sait pas si l'information n'est pas dans le contexte récupéré.
- **Traçabilité** : chaque réponse liste les fichiers sources utilisés.
- **Architecture découplée** : l'indexation (`ingest.py`) est séparée du
  chat (`chat.py`), comme en production.

## Améliorations possibles
- Interface web avec Streamlit
- Support d'autres formats (docx, HTML)
- Ré-indexation incrémentale (ne pas tout refaire à chaque nouveau fichier)
- Évaluation de la qualité des réponses (RAGAS)
