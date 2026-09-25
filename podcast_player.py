"""
podcast_player.py
------------------
Lecteur de podcast intégré : lit un script Prof/Étudiant à voix haute dans le
navigateur via l'API SpeechSynthesis native (aucun fichier audio, aucune
dépendance externe). Rendu comme composant HTML/JS autonome.
"""

import json

VIOLET = "#7C5CFC"
BLUE = "#4F8CFF"
BG = "#0D0D1A"
NODE_BG = "#1B1B33"
TEXT = "#E8E8F5"
MUTED = "#797996"


def render_podcast_player(script: list[dict], height: int = 420) -> str:
    """
    script : liste de {"speaker": "prof"|"etudiant", "text": str, "chapter": str}.
    Retourne le HTML complet à afficher via st.components.v1.html(..., height=height).
    """
    data = json.dumps(script, ensure_ascii=False)

    return f"""
<div id="podcast-root" style="font-family: 'Space Grotesk', Inter, Arial, sans-serif;
     background: {BG}; border-radius: 14px; padding: 18px; color: {TEXT};">

  <div id="pod-now-playing" style="min-height: 90px; display: flex; align-items: center;
       gap: 14px; padding: 10px 6px; border-bottom: 1px solid #2A2A45; margin-bottom: 12px;">
    <div id="pod-avatar" style="font-size: 34px;">🎧</div>
    <div>
      <div id="pod-speaker-name" style="font-weight: 700; font-size: 13px; color: {MUTED};">
        Prêt à démarrer
      </div>
      <div id="pod-current-text" style="font-size: 15px; line-height: 1.4; max-width: 480px;">
        Clique sur lecture pour commencer l'écoute.
      </div>
    </div>
  </div>

  <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 14px;">
    <button id="pod-play" style="background: {VIOLET}; color: #fff; border: none;
        border-radius: 6px; padding: 8px 18px; font-weight: 700; cursor: pointer;">▶ Lecture</button>
    <button id="pod-pause" style="background: {NODE_BG}; color: {TEXT}; border: 1px solid #2A2A45;
        border-radius: 6px; padding: 8px 14px; cursor: pointer;">⏸ Pause</button>
    <button id="pod-restart" style="background: {NODE_BG}; color: {TEXT}; border: 1px solid #2A2A45;
        border-radius: 6px; padding: 8px 14px; cursor: pointer;">⟲ Reprendre</button>
    <label style="font-size: 12px; color: {MUTED}; margin-left: auto; display: flex; align-items: center; gap: 6px;">
      Vitesse
      <input id="pod-speed" type="range" min="0.7" max="1.4" step="0.1" value="1"
             style="accent-color: {VIOLET};">
      <span id="pod-speed-val" style="font-family: monospace; width: 32px;">1.0x</span>
    </label>
  </div>

  <div id="pod-transcript" style="max-height: 220px; overflow-y: auto; padding-right: 6px;"></div>
</div>

<script>
(function() {{
  const script = {data};
  const root = document.getElementById("podcast-root");
  const transcriptEl = document.getElementById("pod-transcript");
  const nameEl = document.getElementById("pod-speaker-name");
  const textEl = document.getElementById("pod-current-text");
  const avatarEl = document.getElementById("pod-avatar");
  const speedInput = document.getElementById("pod-speed");
  const speedVal = document.getElementById("pod-speed-val");

  let idx = 0;
  let playing = false;
  let voices = [];
  let voxProf = null, voxEtudiant = null;

  function loadVoices() {{
    voices = window.speechSynthesis.getVoices();
    const fr = voices.filter(v => v.lang && v.lang.toLowerCase().startsWith("fr"));
    voxProf = fr[0] || voices[0] || null;
    voxEtudiant = fr[1] || fr[0] || voices[1] || voices[0] || null;
  }}
  loadVoices();
  if (window.speechSynthesis) {{
    window.speechSynthesis.onvoiceschanged = loadVoices;
  }}

  // Construit la transcription complète (tout le script, ligne active surlignée)
  script.forEach((turn, i) => {{
    const row = document.createElement("div");
    row.id = "pod-line-" + i;
    row.style.padding = "6px 8px";
    row.style.borderRadius = "6px";
    row.style.marginBottom = "3px";
    row.style.fontSize = "13px";
    row.style.color = "{MUTED}";
    const who = turn.speaker === "prof" ? "🧑\\u200d🏫 Prof" : "🙋 Étudiant";
    row.innerHTML = "<strong>" + who + " :</strong> " + turn.text;
    transcriptEl.appendChild(row);
  }});

  function highlight(i) {{
    script.forEach((_, j) => {{
      const el = document.getElementById("pod-line-" + j);
      if (!el) return;
      el.style.background = (j === i) ? "rgba(124,92,252,0.18)" : "transparent";
      el.style.color = (j === i) ? "{TEXT}" : "{MUTED}";
    }});
    const active = document.getElementById("pod-line-" + i);
    if (active) active.scrollIntoView({{ block: "nearest", behavior: "smooth" }});
  }}

  function speakCurrent() {{
    if (idx >= script.length) {{
      nameEl.textContent = "Terminé";
      textEl.textContent = "Fin du dialogue. Relance avec « Reprendre ».";
      avatarEl.textContent = "✅";
      playing = false;
      return;
    }}
    const turn = script[idx];
    const isProf = turn.speaker === "prof";
    nameEl.textContent = isProf ? "Prof" : "Étudiant";
    textEl.textContent = turn.text;
    avatarEl.textContent = isProf ? "🧑\\u200d🏫" : "🙋";
    highlight(idx);

    try {{
      const utt = new SpeechSynthesisUtterance(turn.text);
      utt.lang = "fr-FR";
      utt.voice = isProf ? voxProf : voxEtudiant;
      utt.pitch = isProf ? 0.95 : 1.15;
      utt.rate = parseFloat(speedInput.value);
      utt.onend = function() {{
        if (playing) {{ idx += 1; speakCurrent(); }}
      }};
      window.speechSynthesis.speak(utt);
    }} catch (e) {{
      textEl.textContent = "La synthèse vocale n'est pas disponible sur ce navigateur.";
    }}
  }}

  document.getElementById("pod-play").addEventListener("click", function() {{
    if (playing) return;
    playing = true;
    window.speechSynthesis.cancel();
    speakCurrent();
  }});
  document.getElementById("pod-pause").addEventListener("click", function() {{
    playing = false;
    window.speechSynthesis.cancel();
  }});
  document.getElementById("pod-restart").addEventListener("click", function() {{
    playing = false;
    window.speechSynthesis.cancel();
    idx = 0;
    highlight(-1);
    nameEl.textContent = "Prêt à démarrer";
    textEl.textContent = "Clique sur lecture pour commencer l'écoute.";
    avatarEl.textContent = "🎧";
  }});
  speedInput.addEventListener("input", function() {{
    speedVal.textContent = parseFloat(speedInput.value).toFixed(1) + "x";
  }});
}})();
</script>
"""