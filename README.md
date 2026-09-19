# Memorix RAG Studio — Data Science Study Hub

Assistant d'études pour la Data Science qui répond à des questions à partir de tes propres documents (cours, PDF, notes...) en utilisant la récupération augmentée par génération (RAG), 100% en local.

## Stack
- **LLM** : Mistral (via Ollama, local)
- **Embeddings** : nomic-embed-text (via Ollama, local)
- **Orchestration** : LangChain
- **Base vectorielle** : Chroma (locale, gratuite)

## Prérequis
- Python 3.10+
- [Ollama](https://ollama.com) installé sur la machine

## Installation

```bash
# 1. Cloner / se placer dans le dossier du projet
cd rag-chatbot

# 2. Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate       # Windows : venv\Scripts\activate

# 3. Installer les dépendances Python
pip install -r requirements.txt

# 4. Télécharger les modèles Ollama nécessaires
ollama pull nomic-embed-text
ollama pull mistral

# 5. Lancer le serveur Ollama (dans un terminal séparé)
ollama serve