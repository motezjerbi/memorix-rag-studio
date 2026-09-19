"""
knowledge_graph.py
------------------
Graphe de connaissances MEMORIX : cours -> sections -> concepts clés.

100 % Python standard + SVG. Aucun appel LLM, aucune dépendance système.
Les concepts sont extraits par fréquence (TF-IDF simplifié) sur les chunks
déjà indexés ; la disposition des noeuds vient d'un algorithme "force-directed"
(Fruchterman-Reingold) écrit à la main.
"""

import math
import random
import re
from collections import Counter
from html import escape

# ---------------------------------------------------------------------------
# Palette (cohérente avec le thème MEMORIX)
# ---------------------------------------------------------------------------

VIOLET = "#7C5CFC"
BLUE = "#4F8CFF"
TEXT = "#E8E8F5"
MUTED = "#797996"
BG = "#0D0D1A"
NODE_DARK = "#1B1B33"

MASTERY_COLORS = {
    "insuffisant": "#F0506E",   # < 40 %
    "partiel": "#F5A524",       # 40 - 60 %
    "bon": BLUE,                # 60 - 85 %
    "excellent": "#2FD6A0",     # >= 85 %
    "inconnu": MUTED,           # jamais testée
}

# ---------------------------------------------------------------------------
# Extraction des concepts
# ---------------------------------------------------------------------------

TOKEN_PATTERN = re.compile(r"[a-zA-ZÀ-ÿ][a-zA-ZÀ-ÿ0-9_\-]{2,}")

STOPWORDS = set("""
le la les un une des du de d l et ou mais donc or ni car que qui quoi dont où
ce cet cette ces se sa son ses leur leurs notre nos votre vos mon ma mes ton ta tes
je tu il elle on nous vous ils elles y en au aux par pour sur sous dans avec sans
entre vers chez comme si plus moins très trop aussi ainsi alors puis encore déjà
est sont être était été sera ont avoir avait fait faire peut peuvent doit doivent
il_y_a cela ceci celui celle ceux celles tout tous toute toutes autre autres même
non pas ne bien lors quand donc selon lorsque afin dont sera seront
the and or but of to in on at by for with without from as is are was were be been
this that these those it its can may will would should could has have had not
cours chapitre section partie exemple exemples figure page cas voir note notes
etc via peut-être ensuite ici là
""".split())


def _tokens(text: str):
    return [t.lower().strip("-_") for t in TOKEN_PATTERN.findall(text)]


def _terms(text: str):
    """Mots utiles + bigrammes (deux mots utiles collés dans le texte)."""
    toks = _tokens(text)
    terms = []
    for i, t in enumerate(toks):
        if t in STOPWORDS or len(t) < 3:
            continue
        terms.append(t)
        if i + 1 < len(toks) and toks[i + 1] not in STOPWORDS and len(toks[i + 1]) >= 3:
            terms.append(f"{t} {toks[i + 1]}")
    return terms


def build_graph(docs_by_book, book_labels, per_section=3, max_concepts=40, min_count=2):
    """
    docs_by_book : {book_id: [Document, ...]} (voir core.get_book_chunks)
    book_labels  : {book_id: "Libellé lisible"}
    Retourne {"nodes": [...], "edges": [...]}.
    """
    # 1. Regrouper les chunks par section
    sections = {}
    for book_id, docs in docs_by_book.items():
        for d in docs:
            chapter = d.metadata.get("chapter", "0")
            if chapter == "0":
                continue
            sections.setdefault((book_id, chapter), []).append(d.page_content)

    if not sections:
        return {"nodes": [], "edges": []}

    n_sec = len(sections)
    sec_counts = {
        key: Counter(t for text in texts for t in _terms(text))
        for key, texts in sections.items()
    }
    doc_freq = Counter()
    for c in sec_counts.values():
        doc_freq.update(c.keys())

    # 2. Meilleurs termes par section (fréquent ici, plus rare ailleurs)
    sec_top = {}
    concept_score = Counter()
    for key, counts in sec_counts.items():
        scored = []
        for term, tf in counts.items():
            if tf < min_count:
                continue
            idf = 1 + math.log(n_sec / doc_freq[term])
            bonus = 1.5 if " " in term else 1.0
            scored.append((tf * idf * bonus, term))
        scored.sort(reverse=True)

        picked = []
        for score, term in scored:
            if len(picked) >= per_section:
                break
            words = set(term.split())
            if any(words & set(p.split()) for _, p in picked):
                continue  # évite "gradient" + "gradient descent"
            picked.append((score, term))
            concept_score[term] += score
        sec_top[key] = picked

    kept = {t for t, _ in concept_score.most_common(max_concepts)}

    # 3. Noeuds
    nodes, edges = [], []
    book_ids = sorted({b for b, _ in sections})
    for b in book_ids:
        nodes.append({
            "id": f"b:{b}", "kind": "book", "label": book_labels.get(b, b),
            "size": 24, "book_id": b, "chapter": None,
            "tooltip": f"Cours : {book_labels.get(b, b)}",
        })

    for (b, ch), texts in sorted(sections.items(), key=lambda kv: (kv[0][0], len(kv[0][1]), kv[0][1])):
        nodes.append({
            "id": f"s:{b}:{ch}", "kind": "section", "label": f"S{ch}",
            "size": 11 + min(len(texts), 12) * 0.5, "book_id": b, "chapter": ch,
            "tooltip": f"{book_labels.get(b, b)} — Section {ch} ({len(texts)} extraits)",
        })
        edges.append({"source": f"b:{b}", "target": f"s:{b}:{ch}", "kind": "hier", "weight": 1.0})

    for term in sorted(kept):
        nodes.append({
            "id": f"c:{term}", "kind": "concept", "label": term,
            "size": 4 + min(doc_freq[term], 5), "book_id": None, "chapter": None,
            "tooltip": f"Concept « {term} » — présent dans {doc_freq[term]} section(s)",
        })

    # 4. Liens section -> concept
    for (b, ch), picked in sec_top.items():
        top_score = max((s for s, _ in picked), default=1.0)
        for score, term in picked:
            if term in kept:
                edges.append({
                    "source": f"s:{b}:{ch}", "target": f"c:{term}",
                    "kind": "has", "weight": 0.4 + 0.6 * (score / top_score),
                })

    # 5. Liens concept <-> concept (apparaissent dans les mêmes extraits)
    pair_counts = Counter()
    for texts in sections.values():
        for text in texts:
            present = sorted(set(_terms(text)) & kept)
            for i in range(len(present)):
                for j in range(i + 1, len(present)):
                    if set(present[i].split()) & set(present[j].split()):
                        continue
                    pair_counts[(present[i], present[j])] += 1

    top_pairs = [(p, c) for p, c in pair_counts.most_common(30) if c >= 2]
    max_pair = max((c for _, c in top_pairs), default=1)
    for (a, b), c in top_pairs:
        edges.append({
            "source": f"c:{a}", "target": f"c:{b}",
            "kind": "co", "weight": 0.3 + 0.7 * (c / max_pair),
        })

    return {"nodes": nodes, "edges": edges}


# ---------------------------------------------------------------------------
# Disposition (Fruchterman-Reingold, déterministe)
# ---------------------------------------------------------------------------

def compute_layout(graph, width=1100, height=680, iterations=220, seed=7):
    """Retourne {node_id: (x, y)}. Même graphe -> même dessin."""
    nodes = graph["nodes"]
    n = len(nodes)
    if n == 0:
        return {}

    rng = random.Random(seed)
    idx = {nd["id"]: i for i, nd in enumerate(nodes)}
    cx, cy = width / 2, height / 2
    pos = [
        [cx + rng.uniform(-1, 1) * width * 0.3, cy + rng.uniform(-1, 1) * height * 0.3]
        for _ in nodes
    ]
    edges = [
        (idx[e["source"]], idx[e["target"]], e["weight"], e["kind"])
        for e in graph["edges"]
        if e["source"] in idx and e["target"] in idx
    ]

    k = math.sqrt(width * height / n) * 0.75
    temp0 = width / 8
    margin = 55

    for it in range(iterations):
        temp = temp0 * (1 - it / iterations) + 1
        disp = [[0.0, 0.0] for _ in range(n)]

        # Répulsion : tous les noeuds se repoussent
        for i in range(n):
            xi, yi = pos[i]
            for j in range(i + 1, n):
                dx, dy = xi - pos[j][0], yi - pos[j][1]
                d2 = dx * dx + dy * dy
                if d2 < 0.01:
                    dx, dy, d2 = rng.uniform(-1, 1), rng.uniform(-1, 1), 1.0
                d = math.sqrt(d2)
                f = k * k / d
                fx, fy = dx / d * f, dy / d * f
                disp[i][0] += fx
                disp[i][1] += fy
                disp[j][0] -= fx
                disp[j][1] -= fy

        # Attraction : les noeuds reliés se rapprochent
        for a, b, w, kind in edges:
            dx, dy = pos[a][0] - pos[b][0], pos[a][1] - pos[b][1]
            d = math.sqrt(dx * dx + dy * dy) or 0.01
            strength = (1.6 if kind == "hier" else 1.0) * (0.5 + w)
            f = d * d / k * strength
            fx, fy = dx / d * f, dy / d * f
            disp[a][0] -= fx
            disp[a][1] -= fy
            disp[b][0] += fx
            disp[b][1] += fy

        # Gravité douce vers le centre + déplacement limité par la "température"
        for i in range(n):
            disp[i][0] -= (pos[i][0] - cx) * 0.9
            disp[i][1] -= (pos[i][1] - cy) * 0.9
            length = math.hypot(disp[i][0], disp[i][1]) or 0.01
            step = min(length, temp)
            pos[i][0] += disp[i][0] / length * step
            pos[i][1] += disp[i][1] / length * step
            pos[i][0] = min(max(pos[i][0], margin), width - margin)
            pos[i][1] = min(max(pos[i][1], margin), height - margin)

    return {nd["id"]: (pos[i][0], pos[i][1]) for i, nd in enumerate(nodes)}


# ---------------------------------------------------------------------------
# Rendu SVG
# ---------------------------------------------------------------------------

def mastery_level(score):
    if score is None:
        return "inconnu"
    if score < 0.4:
        return "insuffisant"
    if score < 0.6:
        return "partiel"
    if score < 0.85:
        return "bon"
    return "excellent"


def render_svg(graph, positions, stats=None, width=1100, height=680):
    """
    stats : résultat de progress_tracker.get_section_stats() (facultatif).
    Retourne une chaîne SVG autonome (téléchargeable telle quelle).
    """
    stats_lookup = {(s["book_id"], s["chapter"]): s for s in (stats or [])}
    nodes = {nd["id"]: nd for nd in graph["nodes"]}

    neighbors = {nid: set() for nid in nodes}
    for e in graph["edges"]:
        if e["source"] in nodes and e["target"] in nodes:
            neighbors[e["source"]].add(e["target"])
            neighbors[e["target"]].add(e["source"])

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="100%" font-family="\'Space Grotesk\', Inter, Arial, sans-serif">',
        "<style>"
        ".node{cursor:pointer;transition:opacity .15s}"
        ".edge{transition:opacity .15s}"
        ".dim{opacity:.10}"
        ".node:hover circle{stroke:#fff;stroke-width:2.5}"
        "</style>",
        "<defs>"
        f'<linearGradient id="gBook" x1="0" y1="0" x2="1" y2="1">'
        f'<stop offset="0" stop-color="{VIOLET}"/><stop offset="1" stop-color="{BLUE}"/>'
        "</linearGradient>"
        '<filter id="glow" x="-50%" y="-50%" width="200%" height="200%">'
        '<feGaussianBlur stdDeviation="4" result="b"/>'
        '<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>'
        "</filter>"
        "</defs>",
        f'<rect width="{width}" height="{height}" rx="14" fill="{BG}"/>',
    ]

    # Liens (dessinés en premier, sous les noeuds)
    for e in graph["edges"]:
        if e["source"] not in positions or e["target"] not in positions:
            continue
        x1, y1 = positions[e["source"]]
        x2, y2 = positions[e["target"]]
        if e["kind"] == "hier":
            stroke, sw, op, dash = VIOLET, 2.2, 0.75, ""
        elif e["kind"] == "has":
            stroke, sw, op, dash = BLUE, 1.2, 0.45, ""
        else:
            stroke, sw, op, dash = MUTED, 1.0 + e["weight"], 0.5, ' stroke-dasharray="4 4"'
        parts.append(
            f'<line class="edge" data-s="{escape(e["source"], True)}" data-t="{escape(e["target"], True)}" '
            f'x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-width="{sw:.1f}" stroke-opacity="{op}"{dash}/>'
        )

    # Noeuds
    for nid, nd in nodes.items():
        if nid not in positions:
            continue
        x, y = positions[nid]
        r = nd["size"]
        tooltip = nd["tooltip"]

        if nd["kind"] == "book":
            fill, stroke, extra = "url(#gBook)", TEXT, ' filter="url(#glow)"'
            label_y, font, weight, color = y + r + 16, 13, "700", TEXT
        elif nd["kind"] == "section":
            s = stats_lookup.get((nd["book_id"], nd["chapter"]))
            level = mastery_level(s["avg_score"] if s else None)
            fill, stroke, extra = MASTERY_COLORS[level], BG, ""
            if s:
                tooltip += f" — maîtrise {s['avg_score'] * 100:.0f}% ({s['attempts']} tentative(s))"
            else:
                tooltip += " — jamais testée"
            label_y, font, weight, color = y + r + 13, 10.5, "600", TEXT
        else:
            fill, stroke, extra = NODE_DARK, VIOLET, ""
            label_y, font, weight, color = y + r + 12, 10, "400", MUTED

        parts.append(
            f'<g class="node" data-id="{escape(nid, True)}" '
            f'data-nb="{escape("|".join(sorted(neighbors[nid])), True)}">'
            f"<title>{escape(tooltip)}</title>"
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="1.5"{extra}/>'
            f'<text x="{x:.1f}" y="{label_y:.1f}" text-anchor="middle" font-size="{font}" '
            f'font-weight="{weight}" fill="{color}" stroke="{BG}" stroke-width="3" '
            f'paint-order="stroke">{escape(nd["label"])}</text>'
            "</g>"
        )

    # Légende
    legend = [
        ("Insuffisant (<40%)", MASTERY_COLORS["insuffisant"]),
        ("Partiel (40-60%)", MASTERY_COLORS["partiel"]),
        ("Bon (60-85%)", MASTERY_COLORS["bon"]),
        ("Excellent (≥85%)", MASTERY_COLORS["excellent"]),
        ("Jamais testée", MASTERY_COLORS["inconnu"]),
    ]
    ly = height - 20 - 16 * len(legend)
    for i, (txt, col) in enumerate(legend):
        yy = ly + i * 16
        parts.append(f'<circle cx="26" cy="{yy}" r="5" fill="{col}"/>')
        parts.append(f'<text x="38" y="{yy + 4}" font-size="10" fill="{MUTED}">{escape(txt)}</text>')

    parts.append("</svg>")
    return "".join(parts)


def wrap_html(svg: str) -> str:
    """Enrobe le SVG avec un peu de JS : survol d'un noeud = ses voisins restent visibles."""
    script = """
<script>
const nodes = document.querySelectorAll('.node');
const edges = document.querySelectorAll('.edge');
nodes.forEach(n => {
  n.addEventListener('mouseenter', () => {
    const id = n.dataset.id;
    const nb = new Set((n.dataset.nb || '').split('|'));
    nb.add(id);
    nodes.forEach(m => m.classList.toggle('dim', !nb.has(m.dataset.id)));
    edges.forEach(e => e.classList.toggle('dim', !(e.dataset.s === id || e.dataset.t === id)));
  });
  n.addEventListener('mouseleave', () => {
    nodes.forEach(m => m.classList.remove('dim'));
    edges.forEach(e => e.classList.remove('dim'));
  });
});
</script>
"""
    return f'<body style="margin:0;background:transparent">{svg}{script}</body>'