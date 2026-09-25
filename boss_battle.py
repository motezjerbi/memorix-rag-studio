"""
boss_battle.py
--------------
Rendu visuel (SVG animé) + effets sonores (Web Audio API, synthétisés, sans fichier
externe) du mode "Boss Fight" : transforme la simulation orale en combat RPG.

100 % Python + SVG + JS minimal, aucune dépendance ni appel réseau.
"""

from html import escape

VIOLET = "#7C5CFC"
BLUE = "#4F8CFF"
RED = "#F0506E"
ORANGE = "#F5A524"
GREEN = "#2FD6A0"
BG = "#0D0D1A"
NODE_BG = "#1B1B33"
TEXT = "#E8E8F5"
MUTED = "#797996"

BOSS_HP_MAX = 100
PLAYER_HP_MAX = 100

BOSS_AVATARS = {
    "lead_ds": "🧪",
    "mlops": "⚙️",
    "soutenance": "🎓",
    "business": "💼",
}


def scale_boss_hp(num_manches: int) -> float:
    """PV du boss proportionnels au nombre de manches choisies (gagnable en jouant bien)."""
    return min(BOSS_HP_MAX, num_manches * 20)


def score_to_damage(score: float, combo: int) -> float:
    base = score * 22
    bonus_combo = min(combo, 5) * 2
    return round(base + bonus_combo, 1)


def score_to_counter(score: float) -> float:
    if score >= 0.5:
        return 0.0
    return round((0.5 - score) * 30, 1)


def next_combo(combo: int, score: float) -> int:
    return combo + 1 if score >= 0.6 else 0


def xp_reward(boss_hp_remaining_pct: float, player_hp_remaining_pct: float, combo_max: int) -> int:
    base = 40
    bonus_pv = round(player_hp_remaining_pct * 30)
    bonus_combo = min(combo_max, 8) * 3
    return base + bonus_pv + bonus_combo


def _bar_svg(current, maximum, width, height, color_full, color_low, label, x=0, y=0):
    pct = max(0.0, min(1.0, current / maximum)) if maximum else 0
    fill_w = width * pct
    color = color_low if pct < 0.3 else color_full
    return f'''<g transform="translate({x},{y})">
      <rect width="{width}" height="{height}" rx="{height/2}" fill="{NODE_BG}" stroke="#2A2A45" stroke-width="1.5"/>
      <rect width="{fill_w:.1f}" height="{height}" rx="{height/2}" fill="{color}"/>
      <text x="{width/2}" y="{height/2 + 4}" text-anchor="middle" font-size="11" font-weight="700"
            fill="{TEXT}" stroke="{BG}" stroke-width="2.5" paint-order="stroke">{escape(label)} — {current:.0f} / {maximum:.0f}</text>
    </g>'''


def render_battle_svg(profile_key, boss_hp, player_hp, combo, event=None,
                      dmg_boss=None, dmg_player=None, banner=None,
                      boss_hp_max=None, width=640, height=220):
    """
    event  : événement du tour JUSTE écoulé (None, "hit_boss", "hit_player", "combo_up",
             "victory", "defeat") — déclenche une animation ÉPHÉMÈRE (tremblement, flash,
             emoji, nombre flottant). Ne doit être passé qu'une seule fois par action.
    banner : "victory" ou "defeat", persiste tant que le combat reste dans cet état
             (affiche le bandeau "🏆 VICTOIRE" / "💀 DÉFAITE" en continu).
    """
    boss_hp_max = boss_hp_max or BOSS_HP_MAX
    avatar = BOSS_AVATARS.get(profile_key, "🧑‍⚖️")
    combo_color = ORANGE if combo >= 3 else MUTED
    combo_class = "pulse-combo" if event == "combo_up" else ""
    boss_class = "shake flash-red" if event in ("hit_boss", "victory") else ""
    player_class = "shake flash-red" if event in ("hit_player", "defeat") else ""

    emoji_map = {"hit_boss": "💥", "hit_player": "😵", "combo_up": "⚡", "victory": "🎉", "defeat": "💀"}
    emoji = emoji_map.get(event)
    emoji_boss = emoji if event in ("hit_boss", "victory", "combo_up") else None
    emoji_player = emoji if event in ("hit_player", "defeat") else None

    style = f"""<style>
      @keyframes shake {{
        0%, 100% {{ transform: translateX(0); }}
        20% {{ transform: translateX(-6px); }}
        40% {{ transform: translateX(6px); }}
        60% {{ transform: translateX(-4px); }}
        80% {{ transform: translateX(4px); }}
      }}
      @keyframes flash-red {{
        0% {{ filter: brightness(1); }}
        25% {{ filter: brightness(1.9) saturate(2.2); }}
        100% {{ filter: brightness(1); }}
      }}
      @keyframes float-up {{
        0% {{ transform: translateY(0); opacity: 1; }}
        100% {{ transform: translateY(-42px); opacity: 0; }}
      }}
      @keyframes pop-fade {{
        0% {{ transform: scale(0.3); opacity: 0; }}
        30% {{ transform: scale(1.3); opacity: 1; }}
        70% {{ transform: scale(1); opacity: 1; }}
        100% {{ transform: scale(1); opacity: 0; }}
      }}
      @keyframes pulse-combo {{
        0%, 100% {{ transform: scale(1); }}
        50% {{ transform: scale(1.35); }}
      }}
      .shake {{ animation: shake 0.35s ease-in-out; }}
      .flash-red {{ animation: flash-red 0.4s ease-in-out; }}
      .dmg-float {{ animation: float-up 1.1s ease-out forwards; }}
      .pop-emoji {{ animation: pop-fade 1s ease-out forwards; transform-origin: center; }}
      .pulse-combo {{ animation: pulse-combo 0.45s ease-in-out; transform-origin: center; }}
    </style>"""

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" '
        f'font-family="\'Space Grotesk\', Inter, Arial, sans-serif">',
        style,
        f'<rect width="{width}" height="{height}" rx="14" fill="{BG}"/>',

        f'<g class="{boss_class}">',
        f'<text x="24" y="42" font-size="34">{avatar}</text>',
        _bar_svg(boss_hp, boss_hp_max, width - 90, 22, VIOLET, RED, "JURY", x=70, y=24),
        "</g>",

        f'<g class="{player_class}">',
        '<text x="24" y="120" font-size="34">🧑\u200d🎓</text>',
        _bar_svg(player_hp, PLAYER_HP_MAX, width - 90, 22, BLUE, RED, "TOI", x=70, y=100),
        "</g>",

        f'<text x="{width/2}" y="{height - 44}" text-anchor="middle" font-size="14" '
        f'font-weight="700" fill="{combo_color}" class="{combo_class}">COMBO x{combo}</text>',
    ]

    if dmg_boss:
        parts.append(
            f'<text class="dmg-float" x="{width - 20}" y="46" text-anchor="end" '
            f'font-size="20" font-weight="800" fill="{RED}" stroke="{BG}" stroke-width="2" '
            f'paint-order="stroke">-{dmg_boss:.0f}</text>'
        )
    if dmg_player:
        parts.append(
            f'<text class="dmg-float" x="{width - 20}" y="122" text-anchor="end" '
            f'font-size="20" font-weight="800" fill="{RED}" stroke="{BG}" stroke-width="2" '
            f'paint-order="stroke">-{dmg_player:.0f}</text>'
        )
    if emoji_boss:
        parts.append(f'<text class="pop-emoji" x="130" y="20" font-size="26">{emoji_boss}</text>')
    if emoji_player:
        parts.append(f'<text class="pop-emoji" x="130" y="98" font-size="26">{emoji_player}</text>')

    if banner == "victory":
        parts.append(f'<text x="{width/2}" y="{height - 16}" text-anchor="middle" font-size="16" '
                      f'font-weight="800" fill="{GREEN}">🏆 VICTOIRE 🏆</text>')
    elif banner == "defeat":
        parts.append(f'<text x="{width/2}" y="{height - 16}" text-anchor="middle" font-size="16" '
                      f'font-weight="800" fill="{RED}">💀 DÉFAITE 💀</text>')

    parts.append("</svg>")
    return "".join(parts)


def render_xp_bar_svg(xp, niveau, xp_dans_niveau, xp_requis, width=280, height=54):
    pct = max(0.0, min(1.0, xp_dans_niveau / xp_requis)) if xp_requis else 0
    fill_w = width * pct
    bar_y, bar_h = 26, 16
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" '
        f'font-family="\'Space Grotesk\', Inter, Arial, sans-serif">',
        f'<text x="0" y="14" font-size="12" font-weight="700" fill="{TEXT}">NIVEAU {niveau}</text>',
        f'<text x="{width}" y="14" text-anchor="end" font-size="11" fill="{MUTED}">{xp} XP au total</text>',
        f'<rect x="0" y="{bar_y}" width="{width}" height="{bar_h}" rx="{bar_h/2}" fill="{NODE_BG}" stroke="#2A2A45" stroke-width="1.5"/>',
        f'<rect x="0" y="{bar_y}" width="{fill_w:.1f}" height="{bar_h}" rx="{bar_h/2}" fill="{GREEN}"/>',
        f'<text x="{width/2}" y="{bar_y + bar_h/2 + 4}" text-anchor="middle" font-size="10" font-weight="700" '
        f'fill="{TEXT}" stroke="{BG}" stroke-width="2.5" paint-order="stroke">{xp_dans_niveau} / {xp_requis} XP</text>',
        "</svg>",
    ]
    return "".join(parts)


def render_sound_effect(event: str, nonce) -> str:
    """
    Fragment HTML/JS : joue un effet sonore SYNTHÉTISÉ (Web Audio API — aucun fichier
    audio requis). À afficher via st.components.v1.html(..., height=0).
    `nonce` doit être unique à chaque appel (ex. time.time()) pour forcer la ré-exécution
    du script par l'iframe Streamlit (sinon le navigateur ne rejoue pas un script identique).
    """
    return f"""
    <div id="snd-{nonce}" style="display:none"></div>
    <script>
    (function() {{
        try {{
            const ctx = new (window.AudioContext || window.webkitAudioContext)();
            function tone(freq, start, dur, type, gain) {{
                const osc = ctx.createOscillator();
                const g = ctx.createGain();
                osc.type = type; osc.frequency.value = freq;
                osc.connect(g); g.connect(ctx.destination);
                g.gain.setValueAtTime(gain, ctx.currentTime + start);
                g.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + start + dur);
                osc.start(ctx.currentTime + start);
                osc.stop(ctx.currentTime + start + dur);
            }}
            const evt = "{event}";
            if (evt === "hit_boss") {{
                tone(220, 0, 0.08, 'square', 0.20);
                tone(160, 0.06, 0.12, 'square', 0.15);
            }} else if (evt === "hit_player") {{
                tone(110, 0, 0.18, 'sawtooth', 0.22);
            }} else if (evt === "combo_up") {{
                tone(880, 0, 0.06, 'triangle', 0.16);
                tone(1046, 0.05, 0.08, 'triangle', 0.13);
            }} else if (evt === "victory") {{
                tone(523, 0, 0.15, 'triangle', 0.22);
                tone(659, 0.15, 0.15, 'triangle', 0.22);
                tone(784, 0.30, 0.35, 'triangle', 0.24);
            }} else if (evt === "defeat") {{
                tone(200, 0, 0.30, 'sawtooth', 0.18);
                tone(140, 0.28, 0.45, 'sawtooth', 0.18);
            }}
        }} catch (e) {{ /* audio indisponible (autoplay bloqué, etc.) : on ignore */ }}
    }})();
    </script>
    """