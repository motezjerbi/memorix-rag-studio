"""
duel_arena.py
--------------
Rendu SVG animé + sons synthétisés du mode Duel : face-à-face 1v1 en local,
réutilise la même approche que boss_battle.py (SVG + Web Audio, sans fichier).
"""

from html import escape

VIOLET = "#7C5CFC"
BLUE = "#4F8CFF"
RED = "#F0506E"
GREEN = "#2FD6A0"
AMBER = "#F5A524"
BG = "#0D0D1A"
NODE_BG = "#1B1B33"
TEXT = "#E8E8F5"
MUTED = "#797996"


def _initiales(nom: str) -> str:
    mots = nom.strip().split()
    if not mots:
        return "?"
    if len(mots) == 1:
        return mots[0][:2].upper()
    return (mots[0][0] + mots[-1][0]).upper()


def render_vs_intro(nom1: str, nom2: str, width=640, height=180):
    """Écran d'introduction, affiché une fois avant le début du duel."""
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%"
      font-family="'Space Grotesk', Inter, Arial, sans-serif">
      <style>
        @keyframes slide-left {{ from {{ transform: translateX(-40px); opacity: 0; }} to {{ transform: translateX(0); opacity: 1; }} }}
        @keyframes slide-right {{ from {{ transform: translateX(40px); opacity: 0; }} to {{ transform: translateX(0); opacity: 1; }} }}
        @keyframes vs-pop {{ 0% {{ transform: scale(0); }} 60% {{ transform: scale(1.3); }} 100% {{ transform: scale(1); }} }}
        .p1 {{ animation: slide-left 0.5s ease-out; }}
        .p2 {{ animation: slide-right 0.5s ease-out; }}
        .vs {{ animation: vs-pop 0.6s ease-out 0.3s backwards; transform-origin: center; }}
      </style>
      <rect width="{width}" height="{height}" rx="14" fill="{BG}"/>
      <g class="p1">
        <circle cx="130" cy="80" r="46" fill="{NODE_BG}" stroke="{VIOLET}" stroke-width="2.5"/>
        <text x="130" y="90" text-anchor="middle" font-size="26" font-weight="700" fill="{VIOLET}">{escape(_initiales(nom1))}</text>
        <text x="130" y="145" text-anchor="middle" font-size="14" font-weight="600" fill="{TEXT}">{escape(nom1)}</text>
      </g>
      <g class="vs">
        <circle cx="{width/2}" cy="80" r="26" fill="{BG}" stroke="{AMBER}" stroke-width="2"/>
        <text x="{width/2}" y="88" text-anchor="middle" font-size="18" font-weight="800" fill="{AMBER}">VS</text>
      </g>
      <g class="p2">
        <circle cx="{width-130}" cy="80" r="46" fill="{NODE_BG}" stroke="{BLUE}" stroke-width="2.5"/>
        <text x="{width-130}" y="90" text-anchor="middle" font-size="26" font-weight="700" fill="{BLUE}">{escape(_initiales(nom2))}</text>
        <text x="{width-130}" y="145" text-anchor="middle" font-size="14" font-weight="600" fill="{TEXT}">{escape(nom2)}</text>
      </g>
    </svg>'''


def render_scoreboard(nom1, score1, nom2, score2, manche, total_manches,
                      event=None, points_gagnes1=0, points_gagnes2=0, width=640, height=150):
    """
    event : None, "p1_marque", "p2_marque", "egalite", "aucun" — anime le score qui vient
    de changer (pulse + nombre flottant), déclenché une seule fois par appel.
    """
    max_score = max(score1, score2, 10, 1)
    largeur_barre = width / 2 - 60

    def barre(score, x, couleur, ancrage):
        pct = min(1.0, score / (max_score * 1.15)) if max_score else 0
        w = largeur_barre * pct
        rx = x if ancrage == "start" else x - w
        return f'''<rect x="{x if ancrage=="start" else x-largeur_barre}" y="60" width="{largeur_barre}" height="18" rx="9"
              fill="{NODE_BG}" stroke="#2A2A45" stroke-width="1.5"/>
              <rect x="{rx}" y="60" width="{w:.1f}" height="18" rx="9" fill="{couleur}"/>'''

    classe1 = "pulse" if event == "p1_marque" else ""
    classe2 = "pulse" if event == "p2_marque" else ""

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" '
        f'font-family="\'Space Grotesk\', Inter, Arial, sans-serif">',
        '<style>@keyframes pulse-sb {0%,100%{transform:scale(1);}50%{transform:scale(1.06);}} '
        '.pulse{animation:pulse-sb 0.4s ease-in-out;transform-origin:center;} '
        '@keyframes float-up-sb {0%{transform:translateY(0);opacity:1;}100%{transform:translateY(-30px);opacity:0;}} '
        '.floatnum{animation:float-up-sb 1s ease-out forwards;}</style>',
        f'<rect width="{width}" height="{height}" rx="14" fill="{BG}"/>',
        f'<text x="20" y="30" font-size="13" font-weight="700" fill="{VIOLET}">{escape(nom1)}</text>',
        f'<text x="{width-20}" y="30" text-anchor="end" font-size="13" font-weight="700" fill="{BLUE}">{escape(nom2)}</text>',
        f'<g class="{classe1}" transform-origin="20 69">',
        barre(score1, 20, VIOLET, "start"),
        f'<text x="20" y="53" font-family="\'JetBrains Mono\', monospace" font-size="15" font-weight="700" fill="{TEXT}">{score1:.0f}</text>',
        "</g>",
        f'<g class="{classe2}">',
        barre(score2, width - 20, BLUE, "end"),
        f'<text x="{width-20}" y="53" text-anchor="end" font-family="\'JetBrains Mono\', monospace" font-size="15" font-weight="700" fill="{TEXT}">{score2:.0f}</text>',
        "</g>",
        f'<text x="{width/2}" y="100" text-anchor="middle" font-size="12" font-weight="700" fill="{MUTED}">MANCHE {manche} / {total_manches}</text>',
    ]

    if points_gagnes1:
        parts.append(f'<text class="floatnum" x="{20+largeur_barre}" y="55" font-size="16" font-weight="800" '
                     f'fill="{GREEN}" text-anchor="end">+{points_gagnes1:.0f}</text>')
    if points_gagnes2:
        parts.append(f'<text class="floatnum" x="{width-20-largeur_barre}" y="55" font-size="16" font-weight="800" '
                     f'fill="{GREEN}">+{points_gagnes2:.0f}</text>')

    parts.append("</svg>")
    return "".join(parts)


def render_victory_banner(gagnant, score1, score2, width=640, height=110):
    if score1 == score2:
        texte, couleur = "ÉGALITÉ PARFAITE", AMBER
    else:
        texte, couleur = f"🏆 {gagnant} REMPORTE LE DUEL", GREEN
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%"
      font-family="'Space Grotesk', Inter, Arial, sans-serif">
      <rect width="{width}" height="{height}" rx="14" fill="{BG}"/>
      <text x="{width/2}" y="{height/2+8}" text-anchor="middle" font-size="20" font-weight="800"
            fill="{couleur}">{escape(texte)}</text>
    </svg>'''


def render_duel_sound(event: str, nonce) -> str:
    """Effet sonore synthétisé (Web Audio, aucun fichier). À afficher via components.html(height=0)."""
    return f"""
    <div style="display:none">{nonce}</div>
    <script>
    (function() {{
        try {{
            const ctx = new (window.AudioContext || window.webkitAudioContext)();
            function tone(freq, start, dur, type, gain) {{
                const osc = ctx.createOscillator(); const g = ctx.createGain();
                osc.type = type; osc.frequency.value = freq;
                osc.connect(g); g.connect(ctx.destination);
                g.gain.setValueAtTime(gain, ctx.currentTime + start);
                g.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + start + dur);
                osc.start(ctx.currentTime + start); osc.stop(ctx.currentTime + start + dur);
            }}
            const evt = "{event}";
            if (evt === "reveal") {{ tone(440, 0, 0.1, 'square', 0.15); }}
            else if (evt === "point") {{ tone(660, 0, 0.08, 'triangle', 0.18); tone(880, 0.06, 0.1, 'triangle', 0.15); }}
            else if (evt === "steal") {{ tone(300, 0, 0.1, 'sawtooth', 0.2); tone(500, 0.1, 0.15, 'triangle', 0.18); }}
            else if (evt === "victory") {{
                tone(523, 0, 0.15, 'triangle', 0.2); tone(659, 0.15, 0.15, 'triangle', 0.2); tone(784, 0.3, 0.4, 'triangle', 0.22);
            }}
        }} catch (e) {{}}
    }})();
    </script>
    """