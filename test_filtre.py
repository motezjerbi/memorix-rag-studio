"""Vérifie isolément _sujet_seulement_en_liste, sans Ollama."""
import rag_core as core

phrase = ("Au lieu d'entraîner un CNN depuis zéro, on peut utiliser un modèle "
          "pré-entraîné (ResNet, EfficientNet, etc.) puis remplacer sa tête de classification.")
question = "Qu'est-ce que ResNet ?"

if not hasattr(core, "_sujet_seulement_en_liste"):
    print("❌ _sujet_seulement_en_liste n'existe pas dans rag_core.py : le correctif n'a pas été ajouté.")
else:
    resultat = core._sujet_seulement_en_liste(phrase, question)
    print(f"_sujet_seulement_en_liste(...) = {resultat}")
    print("✅ Filtre actif, la phrase sera écartée." if resultat else "❌ Le filtre ne détecte pas ce cas.")