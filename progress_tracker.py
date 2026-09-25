"""
progress_tracker.py
--------------------
Suivi de progression local (SQLite) pour les quiz QCM, les examens oraux,
le Boss Fight, les flashcards (SM-2), le Duel de Révision et la calibration
de confiance métacognitive (score de Brier).
Permet de repérer les sections où l'utilisateur a des difficultés
récurrentes, pour lui proposer une fiche de révision ciblée.

Toutes les données restent en local (fichier progress.db), rien n'est
envoyé nulle part.
"""

import os
import sqlite3
from datetime import datetime

import spaced_repetition as sr

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


# ---------------------------------------------------------------------------
# Boss Fight : XP, niveau et historique des victoires
# ---------------------------------------------------------------------------

def _init_boss_tables():
    conn = _connect()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS player_xp (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            xp INTEGER NOT NULL DEFAULT 0
        )
    """)
    conn.execute("INSERT OR IGNORE INTO player_xp (id, xp) VALUES (1, 0)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS boss_victories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            profile_key TEXT NOT NULL,
            profile_label TEXT NOT NULL,
            book_id TEXT NOT NULL,
            book_label TEXT NOT NULL,
            xp_gagne INTEGER NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def get_xp() -> int:
    _init_boss_tables()
    conn = _connect()
    row = conn.execute("SELECT xp FROM player_xp WHERE id = 1").fetchone()
    conn.close()
    return row["xp"] if row else 0


def add_xp(amount: int) -> int:
    """Ajoute de l'XP (peut être négatif) et retourne le nouveau total."""
    _init_boss_tables()
    conn = _connect()
    conn.execute("UPDATE player_xp SET xp = MAX(0, xp + ?) WHERE id = 1", (amount,))
    conn.commit()
    conn.close()
    return get_xp()


def level_from_xp(xp: int):
    """Retourne (niveau, xp_dans_le_niveau, xp_requis_pour_le_niveau_suivant)."""
    seuil = 100
    niveau = xp // seuil + 1
    xp_dans_niveau = xp % seuil
    return niveau, xp_dans_niveau, seuil


def record_boss_victory(profile_key, profile_label, book_id, book_label, xp_gagne):
    _init_boss_tables()
    conn = _connect()
    conn.execute(
        "INSERT INTO boss_victories (timestamp, profile_key, profile_label, book_id, book_label, xp_gagne) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (datetime.now().isoformat(), profile_key, profile_label, book_id, book_label, xp_gagne),
    )
    conn.commit()
    conn.close()


def get_boss_victories(limit=10):
    if not os.path.isfile(DB_PATH):
        return []
    _init_boss_tables()
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM boss_victories ORDER BY timestamp DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def count_boss_victories(profile_key=None) -> int:
    if not os.path.isfile(DB_PATH):
        return 0
    _init_boss_tables()
    conn = _connect()
    if profile_key:
        n = conn.execute(
            "SELECT COUNT(*) as c FROM boss_victories WHERE profile_key = ?", (profile_key,)
        ).fetchone()["c"]
    else:
        n = conn.execute("SELECT COUNT(*) as c FROM boss_victories").fetchone()["c"]
    conn.close()
    return n


# ---------------------------------------------------------------------------
# Flashcards : génération, stockage et répétition espacée (SM-2)
# ---------------------------------------------------------------------------

def _init_flashcard_tables():
    conn = _connect()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS flashcards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            book_id TEXT NOT NULL,
            book_label TEXT NOT NULL,
            chapter TEXT NOT NULL,
            front TEXT NOT NULL,
            back TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS flashcard_review (
            card_id INTEGER PRIMARY KEY,
            repetitions INTEGER NOT NULL DEFAULT 0,
            ease_factor REAL NOT NULL DEFAULT 2.5,
            interval_days REAL NOT NULL DEFAULT 0,
            next_review TEXT NOT NULL,
            last_reviewed TEXT,
            FOREIGN KEY (card_id) REFERENCES flashcards(id)
        )
    """)
    conn.commit()
    conn.close()


def add_flashcards(book_id, book_label, cards):
    """
    Ajoute un lot de flashcards (voir rag_core.generate_flashcards) : chaque élément
    doit avoir "chapter", "front", "back". Les nouvelles cartes sont dues immédiatement.
    """
    _init_flashcard_tables()
    conn = _connect()
    now = datetime.now().isoformat()
    for c in cards:
        cur = conn.execute(
            "INSERT INTO flashcards (book_id, book_label, chapter, front, back, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (book_id, book_label, c.get("chapter", "0"), c["front"], c["back"], now),
        )
        conn.execute(
            "INSERT INTO flashcard_review (card_id, repetitions, ease_factor, interval_days, next_review) "
            "VALUES (?, 0, ?, 0, ?)",
            (cur.lastrowid, sr.EASE_FACTOR_DEFAULT, now),
        )
    conn.commit()
    conn.close()


def get_due_flashcards(book_id=None, limit=20):
    """Cartes dont next_review <= maintenant, les plus en retard en premier."""
    if not os.path.isfile(DB_PATH):
        return []
    _init_flashcard_tables()
    conn = _connect()
    now = datetime.now().isoformat()
    query = """
        SELECT f.id, f.book_id, f.book_label, f.chapter, f.front, f.back,
               r.repetitions, r.ease_factor, r.interval_days, r.next_review
        FROM flashcards f JOIN flashcard_review r ON f.id = r.card_id
        WHERE r.next_review <= ?
    """
    params = [now]
    if book_id:
        query += " AND f.book_id = ?"
        params.append(book_id)
    query += " ORDER BY r.next_review ASC LIMIT ?"
    params.append(limit)
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def record_flashcard_review(card_id, quality):
    """Applique SM-2 à une carte après une révision et enregistre le nouvel état."""
    _init_flashcard_tables()
    conn = _connect()
    row = conn.execute(
        "SELECT repetitions, ease_factor, interval_days FROM flashcard_review WHERE card_id = ?",
        (card_id,),
    ).fetchone()
    if row is None:
        conn.close()
        return
    reps, ease, interval = sr.schedule(quality, row["repetitions"], row["ease_factor"], row["interval_days"])
    now = datetime.now()
    conn.execute(
        "UPDATE flashcard_review SET repetitions = ?, ease_factor = ?, interval_days = ?, "
        "next_review = ?, last_reviewed = ? WHERE card_id = ?",
        (reps, ease, interval, sr.next_review_date(interval, now), now.isoformat(), card_id),
    )
    conn.commit()
    conn.close()


def get_flashcard_deck_stats(book_id=None):
    """{"total": N, "dues": N} pour un cours donné (ou l'ensemble si book_id=None)."""
    if not os.path.isfile(DB_PATH):
        return {"total": 0, "dues": 0}
    _init_flashcard_tables()
    conn = _connect()
    now = datetime.now().isoformat()
    where = "WHERE f.book_id = ?" if book_id else ""
    params_total = (book_id,) if book_id else ()
    total = conn.execute(f"SELECT COUNT(*) as c FROM flashcards f {where}", params_total).fetchone()["c"]
    where_due = (where + " AND" if where else "WHERE") + " r.next_review <= ?"
    params_due = (*params_total, now) if book_id else (now,)
    dues = conn.execute(
        f"SELECT COUNT(*) as c FROM flashcards f JOIN flashcard_review r ON f.id = r.card_id {where_due}",
        params_due,
    ).fetchone()["c"]
    conn.close()
    return {"total": total, "dues": dues}


def reset_flashcards(book_id=None):
    """Supprime les flashcards (et leur historique SM-2) d'un cours, ou de tous si book_id=None."""
    if not os.path.isfile(DB_PATH):
        return
    _init_flashcard_tables()
    conn = _connect()
    if book_id:
        ids = [r["id"] for r in conn.execute("SELECT id FROM flashcards WHERE book_id = ?", (book_id,)).fetchall()]
        conn.executemany("DELETE FROM flashcard_review WHERE card_id = ?", [(i,) for i in ids])
        conn.execute("DELETE FROM flashcards WHERE book_id = ?", (book_id,))
    else:
        conn.execute("DELETE FROM flashcard_review")
        conn.execute("DELETE FROM flashcards")
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Duel de Révision : historique des matchs et tableau des vainqueurs
# ---------------------------------------------------------------------------

def _init_duel_tables():
    conn = _connect()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS duel_matches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            book_id TEXT NOT NULL,
            book_label TEXT NOT NULL,
            player1 TEXT NOT NULL,
            player2 TEXT NOT NULL,
            score1 INTEGER NOT NULL,
            score2 INTEGER NOT NULL,
            winner TEXT
        )
    """)
    conn.commit()
    conn.close()


def record_duel_result(book_id, book_label, player1, player2, score1, score2):
    _init_duel_tables()
    winner = None
    if score1 > score2:
        winner = player1
    elif score2 > score1:
        winner = player2
    conn = _connect()
    conn.execute(
        "INSERT INTO duel_matches (timestamp, book_id, book_label, player1, player2, score1, score2, winner) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (datetime.now().isoformat(), book_id, book_label, player1, player2, score1, score2, winner),
    )
    conn.commit()
    conn.close()
    return winner


def get_duel_leaderboard(limit=10):
    if not os.path.isfile(DB_PATH):
        return []
    _init_duel_tables()
    conn = _connect()
    rows = conn.execute(
        "SELECT winner, COUNT(*) as victoires FROM duel_matches "
        "WHERE winner IS NOT NULL GROUP BY winner ORDER BY victoires DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_duel_history(limit=10):
    if not os.path.isfile(DB_PATH):
        return []
    _init_duel_tables()
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM duel_matches ORDER BY timestamp DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Calibration de confiance : prédictions vs réalité (score de Brier)
# ---------------------------------------------------------------------------

def _init_calibration_table():
    conn = _connect()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS calibration_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            book_id TEXT NOT NULL,
            book_label TEXT NOT NULL,
            chapter TEXT NOT NULL,
            confidence INTEGER NOT NULL,   -- 0 à 100
            correct INTEGER NOT NULL       -- 0 ou 1
        )
    """)
    conn.commit()
    conn.close()


def record_calibration(book_id, book_label, chapter, confidence, correct):
    _init_calibration_table()
    conn = _connect()
    conn.execute(
        "INSERT INTO calibration_records (timestamp, book_id, book_label, chapter, confidence, correct) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (datetime.now().isoformat(), book_id, book_label, chapter, confidence, int(correct)),
    )
    conn.commit()
    conn.close()


def get_brier_score(book_id=None):
    """Score de Brier : moyenne((confiance/100 - résultat)²). 0 = parfait, 0.25 = hasard à 50%."""
    if not os.path.isfile(DB_PATH):
        return None
    _init_calibration_table()
    conn = _connect()
    query = "SELECT confidence, correct FROM calibration_records"
    params = ()
    if book_id:
        query += " WHERE book_id = ?"
        params = (book_id,)
    rows = conn.execute(query, params).fetchall()
    conn.close()
    if not rows:
        return None
    erreurs = [((r["confidence"] / 100) - r["correct"]) ** 2 for r in rows]
    return sum(erreurs) / len(erreurs)


def get_calibration_buckets(book_id=None):
    """
    Regroupe les prédictions en 5 tranches de confiance (0-20, 20-40, ..., 80-100)
    et calcule le taux de réussite réel dans chaque tranche.
    Retourne [{"label", "predicted_mid", "actual_pct", "n"}, ...] (tranches non vides seulement).
    """
    if not os.path.isfile(DB_PATH):
        return []
    _init_calibration_table()
    conn = _connect()
    query = "SELECT confidence, correct FROM calibration_records"
    params = ()
    if book_id:
        query += " WHERE book_id = ?"
        params = (book_id,)
    rows = conn.execute(query, params).fetchall()
    conn.close()

    tranches = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 101)]
    resultats = []
    for lo, hi in tranches:
        subset = [r for r in rows if lo <= r["confidence"] < hi]
        if not subset:
            continue
        actual_pct = 100 * sum(r["correct"] for r in subset) / len(subset)
        resultats.append({
            "label": f"{lo}-{min(hi, 100)}%",
            "predicted_mid": (lo + min(hi, 100)) / 2,
            "actual_pct": round(actual_pct, 1),
            "n": len(subset),
        })
    return resultats


def get_calibration_insight(buckets):
    """Phrase de diagnostic simple : surconfiance / sous-confiance / bien calibré."""
    if not buckets:
        return None
    total_n = sum(b["n"] for b in buckets)
    ecart_pondere = sum((b["predicted_mid"] - b["actual_pct"]) * b["n"] for b in buckets) / total_n
    if ecart_pondere > 12:
        return f"Tu es en **surconfiance** en moyenne : tu annonces environ {ecart_pondere:.0f} points de plus que ta réussite réelle. Prends un instant de plus avant de répondre."
    if ecart_pondere < -12:
        return f"Tu es en **sous-confiance** : tu réussis en moyenne {abs(ecart_pondere):.0f} points de plus que ce que tu annonces. Fais-toi davantage confiance !"
    return "Ta calibration est plutôt bonne : ta confiance annoncée reflète assez fidèlement ta réussite réelle."