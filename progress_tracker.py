"""
progress_tracker.py
--------------------
Suivi de progression local (SQLite) pour les quiz QCM et les examens oraux.
Permet de repérer les sections où l'utilisateur a des difficultés
récurrentes, pour lui proposer une fiche de révision ciblée.

Toutes les données restent en local (fichier progress.db), rien n'est
envoyé nulle part.
"""

import os
import sqlite3
from datetime import datetime

DB_PATH = "progress.db"


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Crée la table de suivi si elle n'existe pas déjà. Appelée automatiquement."""
    conn = _connect()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            book_id TEXT NOT NULL,
            book_label TEXT NOT NULL,
            chapter TEXT NOT NULL,
            mode TEXT NOT NULL,          -- 'qcm' ou 'oral'
            question TEXT,
            score REAL NOT NULL          -- 0.0 à 1.0
        )
    """)
    conn.commit()
    conn.close()


def record_attempt(book_id, book_label, chapter, mode, score, question=None):
    """Enregistre un résultat (une question répondue) dans l'historique."""
    init_db()
    conn = _connect()
    conn.execute(
        "INSERT INTO attempts (timestamp, book_id, book_label, chapter, mode, question, score) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (datetime.now().isoformat(), book_id, book_label, chapter, mode, question, score),
    )
    conn.commit()
    conn.close()


def record_quiz_results(book_id, book_label, results):
    """
    Enregistre en une fois les résultats d'un quiz QCM déjà corrigé
    (voir rag_core.grade_quiz). Chaque élément de `results` doit avoir
    "chapter", "question" et "is_correct".
    """
    for r in results:
        record_attempt(
            book_id, book_label, r.get("chapter", "0"), "qcm",
            score=1.0 if r.get("is_correct") else 0.0,
            question=r.get("question"),
        )


def get_section_stats(book_id=None):
    """
    Retourne les statistiques par (book_id, book_label, chapter) :
    nombre de tentatives et score moyen (0 à 1). Trié par score croissant
    (les sections les plus faibles en premier).
    """
    init_db()
    conn = _connect()
    query = "SELECT book_id, book_label, chapter, COUNT(*) as attempts, AVG(score) as avg_score FROM attempts"
    params = ()
    if book_id:
        query += " WHERE book_id = ?"
        params = (book_id,)
    query += " GROUP BY book_id, chapter ORDER BY avg_score ASC"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_weak_sections(threshold=0.6, min_attempts=1, book_id=None):
    """
    Retourne les sections où la moyenne est sous `threshold` (0.6 = 60%
    par défaut), avec au moins `min_attempts` tentatives enregistrées
    (pour éviter de juger sur une seule question ratée par hasard).
    """
    stats = get_section_stats(book_id=book_id)
    return [s for s in stats if s["attempts"] >= min_attempts and s["avg_score"] < threshold]


def has_any_history():
    """True si au moins une tentative a été enregistrée (pour afficher ou non le tableau de bord)."""
    if not os.path.isfile(DB_PATH):
        return False
    init_db()
    conn = _connect()
    count = conn.execute("SELECT COUNT(*) as c FROM attempts").fetchone()["c"]
    conn.close()
    return count > 0


def reset_history():
    """Efface tout l'historique de progression (remise à zéro complète)."""
    if os.path.isfile(DB_PATH):
        os.remove(DB_PATH)