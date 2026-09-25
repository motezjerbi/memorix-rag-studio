"""
landing_visual.py
------------------
Composition SVG unique pour le fond de la page d'accueil : constellation de
données + halos de couleur + pictogrammes (succès, progression, tendance).
Déterministe (seed fixe) : le rendu ne change pas d'un rechargement à l'autre.
"""

import random

VIOLET = "#7C5CFC"
BLUE = "#4F8CFF"
GREEN = "#2FD6A0"


def _constellation(rng, width, height, n_nodes):
    nodes = [
        (rng.uniform(0, width), rng.uniform(0, height), rng.uniform(1.4, 3.0))
        for _ in range(n_nodes)
    ]
    edges = []
    for i, (x1, y1, _) in enumerate(nodes):
        proches = sorted(
            range(len(nodes)),
            key=lambda j: (nodes[j][0] - x1) ** 2 + (nodes[j][1] - y1) ** 2,
        )
        for j in proches[1:3]:
            paire = (min(i, j), max(i, j))
            if paire not in edges:
                edges.append(paire)
    parts = []
    for i, j in edges:
        x1, y1, _ = nodes[i]
        x2, y2, _ = nodes[j]
        parts.append(
            f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" '
            f'stroke="{VIOLET}" stroke-width="0.6" stroke-opacity="0.28"/>'
        )
    for x, y, r in nodes:
        color = VIOLET if rng.random() > 0.35 else BLUE
        parts.append(
            f'<circle cx="{x:.0f}" cy="{y:.0f}" r="{r:.1f}" fill="{color}" fill-opacity="0.55"/>'
        )
    return "".join(parts)


def _pictogramme_succes(x, y, scale=1.0, color=GREEN):
    """Un badge check discret, façon 'objectif atteint'."""
    return f"""<g transform="translate({x},{y}) scale({scale})" opacity="0.5">
      <circle cx="0" cy="0" r="22" fill="none" stroke="{color}" stroke-width="1.4"/>
      <path d="M -9 0 L -2 8 L 11 -9" fill="none" stroke="{color}" stroke-width="2.4"
            stroke-linecap="round" stroke-linejoin="round"/>
    </g>"""


def _pictogramme_barres(x, y, scale=1.0, color=BLUE):
    """Mini-histogramme croissant, façon 'progression'."""
    barres = [10, 18, 14, 26, 22]
    parts = [f'<g transform="translate({x},{y}) scale({scale})" opacity="0.4">']
    for i, h in enumerate(barres):
        parts.append(
            f'<rect x="{i*10}" y="{-h}" width="6" height="{h}" fill="{color}" rx="1"/>'
        )
    parts.append("</g>")
    return "".join(parts)


def _pictogramme_courbe(x, y, scale=1.0, color=VIOLET):
    """Courbe ascendante avec point final, façon 'tendance positive'."""
    return f"""<g transform="translate({x},{y}) scale({scale})" opacity="0.45">
      <polyline points="0,20 18,8 34,14 52,-10 70,-4" fill="none"
                stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
      <circle cx="70" cy="-4" r="3.5" fill="{color}"/>
    </g>"""


def render_landing_backdrop(width=1600, height=1000, seed=11):
    """
    Grand fond illustré, pensé pour tenir toute la page d'accueil : constellation
    de données en trame de fond + halos de couleur doux + 3 pictogrammes
    (réussite, progression, tendance) placés à des endroits calmes de la
    composition. Un seul rendu, pas un motif répété.
    """
    rng = random.Random(seed)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'preserveAspectRatio="xMidYMid slice" width="100%" height="100%">',
        "<defs>",
        f'<radialGradient id="glowA" cx="20%" cy="15%" r="55%">'
        f'<stop offset="0%" stop-color="{VIOLET}" stop-opacity="0.16"/>'
        f'<stop offset="100%" stop-color="{VIOLET}" stop-opacity="0"/></radialGradient>',
        f'<radialGradient id="glowB" cx="85%" cy="75%" r="50%">'
        f'<stop offset="0%" stop-color="{BLUE}" stop-opacity="0.13"/>'
        f'<stop offset="100%" stop-color="{BLUE}" stop-opacity="0"/></radialGradient>',
        "</defs>",
        f'<rect width="{width}" height="{height}" fill="url(#glowA)"/>',
        f'<rect width="{width}" height="{height}" fill="url(#glowB)"/>',
        _constellation(rng, width, height, n_nodes=42),
        _pictogramme_succes(width * 0.86, height * 0.18),
        _pictogramme_barres(width * 0.10, height * 0.78),
        _pictogramme_courbe(width * 0.72, height * 0.86),
        "</svg>",
    ]
    return "".join(parts)