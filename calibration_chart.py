"""
calibration_chart.py
--------------------
Rendu SVG du diagramme de fiabilité (courbe de calibration métacognitive)
et évaluation du score de Brier.
"""

VIOLET = "#7C5CFC"
BLUE = "#4F8CFF"
GREEN = "#2FD6A0"
RED = "#F0506E"
BG = "#0D0D1A"
GRID = "#1D212E"
TEXT = "#E8E8F5"
MUTED = "#797996"


def brier_label(score: float) -> tuple[str, str]:
    """Retourne une appréciation concise et la couleur associée."""
    if score is None:
        return "Non évalué", MUTED
    if score <= 0.10:
        return "Excellente calibration — tes estimations sont très fiables.", GREEN
    if score <= 0.18:
        return "Bonne calibration — tes annonces reflètent globalement ta réussite.", BLUE
    if score <= 0.25:
        return "Calibration moyenne — niveau comparable à une prédiction aléatoire à 50%.", RED
    return "Mauvaise calibration — fort décalage entre ta certitude et la réalité.", RED


def render_calibration_curve(buckets: list[dict], width=640, height=340):
    """
    buckets : [{"label": "60-80%", "predicted_mid": 70, "actual_pct": 55, "n": 4}, ...]
    Trace confiance annoncée (x) vs taux de réussite réel (y), avec la diagonale
    de calibration parfaite en référence. Le score de Brier n'est PAS dessiné ici
    (il est affiché séparément, en dehors du graphique, pour ne jamais chevaucher
    un point de données).
    """
    margin_left, margin_bottom, margin_top, margin_right = 50, 40, 20, 20
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    def px(v):
        return margin_left + (v / 100) * plot_w

    def py(v):
        return margin_top + plot_h - (v / 100) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" '
        f'font-family="\'Space Grotesk\', Inter, Arial, sans-serif">',
        f'<rect width="{width}" height="{height}" rx="14" fill="{BG}"/>',
    ]

    for v in (0, 20, 40, 60, 80, 100):
        parts.append(
            f'<line x1="{px(v):.1f}" y1="{margin_top}" x2="{px(v):.1f}" y2="{py(0):.1f}" '
            f'stroke="{GRID}" stroke-width="1"/>'
        )
        parts.append(
            f'<line x1="{margin_left}" y1="{py(v):.1f}" x2="{width-margin_right}" y2="{py(v):.1f}" '
            f'stroke="{GRID}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{px(v):.1f}" y="{height-margin_bottom+16}" text-anchor="middle" '
            f'font-size="10" fill="{MUTED}">{v}%</text>'
        )
        parts.append(
            f'<text x="{margin_left-8}" y="{py(v)+3:.1f}" text-anchor="end" '
            f'font-size="10" fill="{MUTED}">{v}%</text>'
        )

    parts.append(
        f'<text x="{width/2}" y="{height-6}" text-anchor="middle" font-size="10" '
        f'fill="{MUTED}">Confiance annoncée</text>'
    )

    # Label d'axe Y centré verticalement sur le tracé
    axis_y_center = margin_top + plot_h / 2
    parts.append(
        f'<text x="16" y="{axis_y_center:.1f}" text-anchor="middle" font-size="10" '
        f'fill="{MUTED}" transform="rotate(-90 16 {axis_y_center:.1f})">Réussite réelle</text>'
    )

    # Diagonale parfaite
    parts.append(
        f'<line x1="{px(0):.1f}" y1="{py(0):.1f}" x2="{px(100):.1f}" y2="{py(100):.1f}" '
        f'stroke="{MUTED}" stroke-width="1.4" stroke-dasharray="5 4"/>'
    )

    if buckets:
        pts = [(px(b["predicted_mid"]), py(b["actual_pct"])) for b in buckets]
        chemin = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        parts.append(f'<polyline points="{chemin}" fill="none" stroke="{VIOLET}" stroke-width="2.4"/>')
        for b, (x, y) in zip(buckets, pts):
            couleur = GREEN if b["actual_pct"] >= b["predicted_mid"] - 8 else RED
            r = 4 + min(b["n"], 8) * 0.8
            label_y = y - r - 6 if y - r - 16 > margin_top else y + r + 14
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{couleur}" fill-opacity="0.85"/>')
            parts.append(
                f'<text x="{x:.1f}" y="{label_y:.1f}" text-anchor="middle" font-size="9" '
                f'fill="{TEXT}" font-family="\'JetBrains Mono\', monospace">n={b["n"]}</text>'
            )

    parts.append("</svg>")
    return "".join(parts)