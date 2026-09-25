"""
app.py
------
Interface cockpit pour MEMORIX : Studio RAG d'études et révisions.
Design cinématique New Yorker, typographie d'ingénierie, chatbot de pointe sans scroll.
"""

import base64
import pathlib
import time
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components
from langchain_core.prompts import PromptTemplate

import boss_battle
import calibration_chart
import duel_arena
import export_utils
import knowledge_graph as kg
import landing_visual
import podcast_player
import progress_tracker as tracker
import rag_core as core
import revision_planner

st.set_page_config(
    page_title="MEMORIX — Data Science Study Hub",
    page_icon="⚡",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Optimisation des assets statiques en cache
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def get_cached_css() -> str:
    css_path = pathlib.Path(__file__).parent / "assets" / "style.css"
    return css_path.read_text(encoding="utf-8") if css_path.exists() else ""


@st.cache_data(show_spinner=False)
def get_cached_logo_b64() -> str | None:
    logo_path = pathlib.Path(__file__).parent / "assets" / "logo.png"
    return base64.b64encode(logo_path.read_bytes()).decode() if logo_path.exists() else None


def load_css():
    css_content = get_cached_css()
    if css_content:
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)


load_css()

# État de navigation global
if "app_started" not in st.session_state:
    st.session_state.app_started = False

if "initial_prefill_query" not in st.session_state:
    st.session_state.initial_prefill_query = None


# ---------------------------------------------------------------------------
# Fonctions d'authentification et d'identité
# ---------------------------------------------------------------------------

def _salutation() -> str:
    heure = datetime.now().hour
    if heure < 5:
        return "Bonne nuit"
    if heure < 12:
        return "Bonjour"
    if heure < 18:
        return "Bon après-midi"
    return "Bonsoir"


def _avatar_tag(picture: str | None, name: str, size: int = 42) -> str:
    """Avatar en cercle parfait avec bordure néon violette."""
    if picture:
        return f'<img src="{picture}" referrerpolicy="no-referrer" class="top-avatar-circle-img" style="width:{size}px;height:{size}px;min-width:{size}px;"/>'
    initiales = "".join(m[0] for m in (name or "?").split()[:2]).upper()
    return f'<div class="top-avatar-circle-fallback" style="width:{size}px;height:{size}px;min-width:{size}px;font-size:{int(size*0.38)}px;">{initiales}</div>'


def render_login_gate():
    st.markdown(
        f'<div class="landing-backdrop">{landing_visual.render_landing_backdrop()}</div>',
        unsafe_allow_html=True,
    )
    encoded_logo = get_cached_logo_b64()
    if encoded_logo:
        st.markdown(
            f'<div class="login-logo-box"><img src="data:image/png;base64,{encoded_logo}" class="top-logo-img" alt="MEMORIX"/></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="login-logo-box"><span class="top-brand-text">⚡ MEMORIX</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="login-gate-card">'
        '<div class="poster-kicker">AUTHENTIFICATION SÉCURISÉE</div>'
        '<h2 class="login-gate-title">Connecte-toi pour ouvrir ton cockpit de révision</h2>'
        '<p class="poster-body">Ta progression, tes flashcards et tes résultats sont reliés à ton compte Google.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    _, col_btn, _ = st.columns([1.3, 1.4, 1.3])
    with col_btn:
        if st.button("Se connecter avec Google", type="primary", use_container_width=True):
            st.login()


# ---------------------------------------------------------------------------
# Landing Page : Format Cinématique New Yorker (Full-Screen Fit)
# ---------------------------------------------------------------------------

def render_landing_page():
    encoded_logo = get_cached_logo_b64()

    # Arrière-plan cinématique global animé : LLM, tenseurs & synapses
    bg_llm_canvas = (
        '<div class="global-llm-backdrop">'
        '  <div class="llm-watermark-term term-1">TRANSFORMERS</div>'
        '  <div class="llm-watermark-term term-2">LATENT SPACE</div>'
        '  <div class="llm-watermark-term term-3">ATTENTION HEADS</div>'
        '  <div class="llm-watermark-term term-4">EMBEDDINGS</div>'
        '  <div class="llm-watermark-term term-5">GRADIENT DESCENT</div>'
        '  <svg class="global-llm-svg" viewBox="0 0 1440 900" preserveAspectRatio="none">'
        '    <defs>'
        '      <linearGradient id="llm-line-grad" x1="0%" y1="0%" x2="100%" y2="100%">'
        '        <stop offset="0%" stop-color="#7C5CFC" stop-opacity="0.35"/>'
        '        <stop offset="50%" stop-color="#4F8CFF" stop-opacity="0.15"/>'
        '        <stop offset="100%" stop-color="#2FD6A0" stop-opacity="0.05"/>'
        '      </linearGradient>'
        '    </defs>'
        '    <line x1="100" y1="120" x2="380" y2="280" stroke="url(#llm-line-grad)" stroke-width="1.6" class="llm-flow-line l1"/>'
        '    <line x1="380" y1="280" x2="720" y2="150" stroke="url(#llm-line-grad)" stroke-width="1.6" class="llm-flow-line l2"/>'
        '    <line x1="720" y1="150" x2="1100" y2="340" stroke="url(#llm-line-grad)" stroke-width="1.6" class="llm-flow-line l3"/>'
        '    <line x1="1100" y1="340" x2="1350" y2="180" stroke="url(#llm-line-grad)" stroke-width="1.6" class="llm-flow-line l4"/>'
        '    <line x1="250" y1="460" x2="560" y2="320" stroke="url(#llm-line-grad)" stroke-width="1.2" class="llm-flow-line l5"/>'
        '    <line x1="560" y1="320" x2="940" y2="480" stroke="url(#llm-line-grad)" stroke-width="1.2" class="llm-flow-line l6"/>'
        '    <circle cx="100" cy="120" r="5" fill="#4F8CFF" class="llm-node n1"/>'
        '    <circle cx="380" cy="280" r="6" fill="#7C5CFC" class="llm-node n2"/>'
        '    <circle cx="720" cy="150" r="5" fill="#2FD6A0" class="llm-node n3"/>'
        '    <circle cx="1100" cy="340" r="5.5" fill="#4F8CFF" class="llm-node n4"/>'
        '    <circle cx="1350" cy="180" r="4.5" fill="#F5A524" class="llm-node n5"/>'
        '    <circle cx="560" cy="320" r="4" fill="#7C5CFC" class="llm-node n6"/>'
        '  </svg>'
        '</div>'
    )
    st.markdown(bg_llm_canvas, unsafe_allow_html=True)

    # 1. En-tête supérieur pro : Logo officiel à gauche, Profil & Quitter à droite
    nav_l, nav_r = st.columns([3.0, 1.4], vertical_alignment="center")

    with nav_l:
        if encoded_logo:
            brand_inner = f'<img src="data:image/png;base64,{encoded_logo}" class="top-logo-img" alt="MEMORIX"/>'
        else:
            brand_inner = '<span class="top-brand-text">⚡ MEMORIX</span>'

        brand_html = (
            f'<div class="top-navbar-left">'
            f'  {brand_inner}'
            f'  <span class="top-badge-core">DATA SCIENCE CORE</span>'
            f'  <span class="top-badge-live"><span class="live-pulse"></span>CHROMA PRÊT</span>'
            f'</div>'
        )
        st.markdown(brand_html, unsafe_allow_html=True)

    with nav_r:
        u_box, u_btn = st.columns([2.3, 1.0], vertical_alignment="center")
        with u_box:
            user_chip_html = (
                f'<div class="top-user-chip">'
                f'  {_avatar_tag(st.user.picture, st.user.name, 42)}'
                f'  <div class="top-user-meta">'
                f'    <span class="top-user-greeting">{_salutation()}</span>'
                f'    <span class="top-user-name">{st.user.name}</span>'
                f'  </div>'
                f'</div>'
            )
            st.markdown(user_chip_html, unsafe_allow_html=True)
        with u_btn:
            if st.button("Quitter", key="top_logout_btn", use_container_width=True):
                st.session_state.pop("welcome_shown", None)
                st.logout()

    # 2. Grand Billboard Cinématique avec Slogan & Livre animé
    hero_billboard_html = (
        '<div class="ny-hero-billboard">'
        '  <div class="ny-hero-bg-overlay"></div>'
        '  <div class="ny-slogan-watermark">VOTRE SOURCE ULTIME EN DATA SCIENCE</div>'
        '  <div class="ny-holo-book">'
        '    <svg viewBox="0 0 200 120" xmlns="http://www.w3.org/2000/svg">'
        '      <path d="M 15 85 C 50 80, 85 86, 100 100 C 115 86, 150 80, 185 85 L 180 25 C 145 20, 115 26, 100 40 C 85 26, 55 20, 20 25 Z" '
        '            fill="none" stroke="var(--signal-violet)" stroke-width="1.8" class="book-glow"/>'
        '      <path d="M 100 40 L 100 100" stroke="var(--signal-blue)" stroke-width="1.4" stroke-dasharray="3 3"/>'
        '      <circle cx="100" cy="22" r="2.5" fill="var(--signal-green)" class="data-particle p1"/>'
        '      <circle cx="82" cy="14" r="2" fill="var(--signal-violet)" class="data-particle p2"/>'
        '      <circle cx="118" cy="12" r="2.2" fill="var(--signal-blue)" class="data-particle p3"/>'
        '    </svg>'
        '  </div>'
        '  <div class="ny-hero-content">'
        '    <div class="ny-slogan-pill">⚡ VOTRE SOURCE ULTIME EN DATA SCIENCE &amp; IA</div>'
        '    <h1 class="ny-hero-title">PROPULSEZ VOS RÉVISIONS VERS L\'EXCELLENCE</h1>'
        '    <p class="ny-hero-sub">'
        '      La plateforme de référence locale pour dominer les modèles neuronaux, l\'algèbre tensorielle et '
        '      les architectures de production en Data Science & IA.'
        '    </p>'
        '    <div class="ny-hero-specs">'
        '      <span class="ny-spec-item"><strong>100%</strong> LOCAL &amp; CONFIDENTIEL</span>'
        '      <span class="ny-spec-sep">/</span>'
        '      <span class="ny-spec-item"><strong>&lt; 100ms</strong> LATENCE DE RETRIEVAL</span>'
        '      <span class="ny-spec-sep">/</span>'
        '      <span class="ny-spec-item"><strong>SM-2</strong> MÉMORISATION ACTIVE</span>'
        '    </div>'
        '  </div>'
        '</div>'
    )
    st.markdown(hero_billboard_html, unsafe_allow_html=True)

    # 3. Ruban Ticker Défilant
    ticker_html = (
        '<div class="announcement-marquee-wrapper">'
        '  <div class="announcement-marquee-track">'
        '    <span class="marquee-item"><span class="badge-dot-green">●</span> NOYAU RAG ACTIF — ChromaDB synchronisé</span>'
        '    <span class="marquee-separator">///</span>'
        '    <span class="marquee-item"><span class="badge-tag">FLASHCARDS</span> Mémorisation SM-2 calibrée</span>'
        '    <span class="marquee-separator">///</span>'
        '    <span class="marquee-item"><span class="badge-dot-violet">●</span> JURY TECHNIQUE — Simulations d\'oraux actives</span>'
        '    <span class="marquee-separator">///</span>'
        '    <span class="marquee-item"><span class="badge-tag">DUEL</span> Mode Face-à-face local disponible</span>'
        '    <span class="marquee-separator">///</span>'
        '    <span class="marquee-item"><span class="badge-dot-blue">●</span> PODCAST IA — Synthèse vocale interactive</span>'
        '    <span class="marquee-separator">///</span>'
        '    <span class="marquee-item"><span class="badge-dot-green">●</span> NOYAU RAG ACTIF — ChromaDB synchronisé</span>'
        '    <span class="marquee-separator">///</span>'
        '    <span class="marquee-item"><span class="badge-tag">FLASHCARDS</span> Mémorisation SM-2 calibrée</span>'
        '  </div>'
        '</div>'
    )
    st.markdown(ticker_html, unsafe_allow_html=True)

    # 4. Grille de 4 Affiches Thématiques (Format New Yorker)
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            """
            <div class="ny-card">
              <div class="ny-card-media" style="background-image: url('https://images.unsplash.com/photo-1507146426996-ef05306b995a?auto=format&fit=crop&w=400&q=80');">
                <span class="ny-card-badge">VISION</span>
              </div>
              <div class="ny-card-content">
                <h4>COMPUTER VISION</h4>
                <p>Convolutions, ViT, backbones et segmentation.</p>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Explorer Vision →", key="btn_cv", use_container_width=True):
            st.session_state.initial_prefill_query = (
                "Détaille les principes des convolutions, des Vision Transformers "
                "et des loss functions de segmentation abordés dans les documents."
            )
            st.session_state.app_started = True
            st.rerun()

    with c2:
        st.markdown(
            """
            <div class="ny-card">
              <div class="ny-card-media" style="background-image: url('https://images.unsplash.com/photo-1620712943543-bcc4688e7485?auto=format&fit=crop&w=400&q=80');">
                <span class="ny-card-badge">NLP</span>
              </div>
              <div class="ny-card-content">
                <h4>NLP & LLMS</h4>
                <p>Attention multi-têtes, tokens et fine-tuning LoRA.</p>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Explorer NLP →", key="btn_nlp", use_container_width=True):
            st.session_state.initial_prefill_query = (
                "Présente le formalisme de l'attention multi-têtes, la tokenisation "
                "et les méthodes de fine-tuning par adaptation de bas rang (LoRA)."
            )
            st.session_state.app_started = True
            st.rerun()

    with c3:
        st.markdown(
            """
            <div class="ny-card">
              <div class="ny-card-media" style="background-image: url('https://images.unsplash.com/photo-1558494949-ef010cbdcc31?auto=format&fit=crop&w=400&q=80');">
                <span class="ny-card-badge">MLOPS</span>
              </div>
              <div class="ny-card-content">
                <h4>MLOPS & PIPELINES</h4>
                <p>Data drift, registries, Docker et CI/CD.</p>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Explorer MLOps →", key="btn_mlops", use_container_width=True):
            st.session_state.initial_prefill_query = (
                "Structure les étapes d'un pipeline MLOps : ingestion, validation des données, "
                "registre de conteneurs et politiques de déploiement continu."
            )
            st.session_state.app_started = True
            st.rerun()

    with c4:
        st.markdown(
            """
            <div class="ny-card">
              <div class="ny-card-media" style="background-image: url('https://images.unsplash.com/photo-1509228468518-180dd4864904?auto=format&fit=crop&w=400&q=80');">
                <span class="ny-card-badge">MATHS</span>
              </div>
              <div class="ny-card-content">
                <h4>STATS & FONDEMENTS</h4>
                <p>Gradients, régularisation L1/L2 et probabilités.</p>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Explorer Maths →", key="btn_stats", use_container_width=True):
            st.session_state.initial_prefill_query = (
                "Établis la fiche de synthèse des critères d'optimisation (descente de gradient, "
                "régularisations L1/L2) et des distributions statistiques utiles."
            )
            st.session_state.app_started = True
            st.rerun()

    # 5. Bouton central d'entrée cockpit
    st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
    _, col_action, _ = st.columns([1.5, 2.2, 1.5])
    with col_action:
        if st.button("⚡ ACCÉDER AU COCKPIT STUDIO →", type="primary", use_container_width=True):
            st.session_state.app_started = True
            st.rerun()


# Contrôle d'accès : connexion Google obligatoire avant tout le reste
if not st.user.is_logged_in:
    render_login_gate()
    st.stop()

# Notification discrète à la connexion
if not st.session_state.get("welcome_shown"):
    st.toast(f"{_salutation()} {st.user.name} 👋 Prêt à réviser sur MEMORIX.")
    st.session_state.welcome_shown = True

# Contrôle d'affichage de la landing page
if not st.session_state.app_started:
    render_landing_page()
    st.stop()


# ---------------------------------------------------------------------------
# Composants Cockpit
# ---------------------------------------------------------------------------

def render_brand():
    encoded = get_cached_logo_b64()
    if encoded:
        st.sidebar.markdown(
            f'<div class="brand-container"><img src="data:image/png;base64,{encoded}" alt="MEMORIX"/></div>',
            unsafe_allow_html=True,
        )
    else:
        brand_html = (
            '<div class="brand-fallback">'
            '<div class="brand-title">MEMORIX</div>'
            '<div class="brand-tag">RAG ENGINE // DATA SCIENCE</div>'
            '</div>'
        )
        st.sidebar.markdown(brand_html, unsafe_allow_html=True)

    if st.user.is_logged_in:
        st.sidebar.markdown(
            f'<div class="pilot-badge">{_avatar_tag(st.user.picture, st.user.name, 36)}'
            f'<div><div class="pilot-name">{st.user.name}</div>'
            f'<div class="pilot-status">Session active</div></div></div>',
            unsafe_allow_html=True,
        )
        if st.sidebar.button("Se déconnecter", key="sidebar_logout", use_container_width=True):
            st.session_state.pop("welcome_shown", None)
            st.logout()


def render_cockpit_header(title: str, subtitle: str, active_context: str = "ChromaDB Ready"):
    st.markdown(
        f"""
        <div class="hero-cockpit">
            <div class="hero-cockpit-content">
                <h1>{title}</h1>
                <p>{subtitle}</p>
            </div>
            <div class="engine-status-tag">
                <div class="pulse-led"></div>
                <span>{active_context}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_broadcast_bar(tag: str, text: str):
    st.markdown(
        f"""
        <div class="broadcast-bar">
            <span class="broadcast-badge">{tag}</span>
            <span class="broadcast-text">{text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def stream_text_generator(text: str):
    words = text.split(" ")
    for idx, word in enumerate(words):
        yield word + (" " if idx < len(words) - 1 else "")
        time.sleep(0.012)


@st.cache_data(show_spinner="Extraction des concepts et calcul de la carte...")
def get_cached_graph(_vectorstore, labels: tuple, per_section: int, max_concepts: int):
    book_labels = dict(labels)
    docs_by_book = {b: core.get_book_chunks(_vectorstore, b) for b in book_labels}
    graph = kg.build_graph(docs_by_book, book_labels, per_section=per_section, max_concepts=max_concepts)
    positions = kg.compute_layout(graph)
    return graph, positions


# ---------------------------------------------------------------------------
# Initialisation & Chargement RAG
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Initialisation des embeddings et des tenseurs...")
def get_resources():
    vectorstore = core.load_vectorstore()
    llm = core.load_llm()
    qa_prompt = PromptTemplate(
        template=core.QA_PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )
    return vectorstore, llm, qa_prompt


def show_sources(docs):
    sources = sorted(set(d.metadata.get("source", "inconnu") for d in docs))
    if sources:
        with st.expander(f"INDEXATION SOURCE // {len(sources)} FICHIER(S)"):
            for src in sources:
                st.markdown(f"`{src}`")


def show_structured_result(result, empty_fallback_message=None, export_title=None):
    if result.get("empty"):
        st.warning(result.get("message") or empty_fallback_message or "Aucun segment trouvé.")
        return

    sections = result.get("sections", [])
    if not sections:
        st.warning(empty_fallback_message or "Aucun contenu identifié.")
        return

    if result.get("synthesis"):
        st.markdown(f"**Synthèse globale**\n\n{result['synthesis']}")
        st.markdown("---")

    for s in sections:
        with st.container(border=True):
            if s.get("label"):
                st.markdown(f"**{s['label']}**")
            st.markdown(s["text"])
            with st.expander("Consulter l'extrait source brut"):
                st.text(s["source"])

    if export_title:
        safe_filename = "".join(c if c.isalnum() or c in " -_" else "_" for c in export_title).strip()
        c1, c2 = st.columns(2)
        with c1:
            pdf_bytes = export_utils.export_to_pdf_bytes(export_title, result)
            st.download_button(
                "Exporter en PDF",
                data=pdf_bytes,
                file_name=f"{safe_filename}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        with c2:
            docx_bytes = export_utils.export_to_docx_bytes(export_title, result)
            st.download_button(
                "Exporter en Word",
                data=docx_bytes,
                file_name=f"{safe_filename}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
            )


# ---------------------------------------------------------------------------
# Vérification du VectorStore
# ---------------------------------------------------------------------------

if not core.is_vectorstore_ready():
    st.error("Index vectoriel introuvable. Exécutez `python ingest.py` au préalable.")
    st.stop()

vectorstore, llm, qa_prompt = get_resources()
books = core.get_all_books(vectorstore)


# ---------------------------------------------------------------------------
# Barre Latérale
# ---------------------------------------------------------------------------

render_brand()

nav_options = [
    "🔎 Recherche augmentée",
    "📖 Synthèse générale",
    "🧾 Fiche mémo technique",
    "🧪 Atelier Code & Diagnostics",
    "✅ Validation QCM",
    "🗂️ Flashcards",
    "🎙️ Simulation orale",
    "⚔️ Boss Fight",
    "🤺 Duel de Révision",
    "🎧 Podcast Révision",
    "🔮 Calibration",
    "🕸️ Graphe de connaissances",
    "📊 Métriques & Suivi",
    "🗓️ Plan de révision",
    "📄 Session PDF éphémère",
]

mode_raw = st.sidebar.radio("Navigation", nav_options)
_SANS_ICONE = {"🗂️ Flashcards", "⚔️ Boss Fight", "🤺 Duel de Révision"}
mode = mode_raw if mode_raw in _SANS_ICONE else mode_raw.split(" ", 1)[1]

st.sidebar.markdown("---")
if st.sidebar.button("← Accueil vitrine", use_container_width=True):
    st.session_state.app_started = False
    st.rerun()

st.sidebar.markdown(
    '<div style="font-family:\'JetBrains Mono\',monospace; font-size:0.7rem; color:#797996; margin-bottom:0.5rem; letter-spacing:0.06em;">BASES VECTORISÉES</div>',
    unsafe_allow_html=True,
)

for book_id, label in books.items():
    st.sidebar.markdown(
        f'<div class="course-indicator"><span>{label}</span><div class="dot-online"></div></div>',
        unsafe_allow_html=True,
    )

requires_picker = mode not in (
    "Recherche augmentée",
    "Atelier Code & Diagnostics",
    "Session PDF éphémère",
    "Métriques & Suivi",
    "Plan de révision",
    "Graphe de connaissances",
)
selected_book_id = None
selected_scope = "Cours entier"

if requires_picker:
    st.sidebar.markdown("---")
    label_to_id = {label: bid for bid, label in books.items()}
    selected_label = st.sidebar.selectbox("Cours actif", list(label_to_id.keys()))
    selected_book_id = label_to_id[selected_label]

    sections = core.get_sections_for_book(vectorstore, selected_book_id)
    scope_options = ["Cours entier"] + [f"Section {s}" for s in sections]
    selected_scope = st.sidebar.selectbox("Périmètre", scope_options)


# ---------------------------------------------------------------------------
# Mode 1 : Recherche augmentée (Chatbot Pro Cybernétique)
# ---------------------------------------------------------------------------

@st.fragment
def render_chat_cockpit(vectorstore, llm, qa_prompt):
    prompt_templates = [
        {
            "badge": "Architecture",
            "title": "Architectures & Modèles",
            "desc": "Compare les fonctions de coût, tenseurs et structures neuronales.",
            "query": "Synthétise les architectures de modèles, leurs fonctions d'activation et loss functions clés présentées dans ce cours.",
        },
        {
            "badge": "Formules",
            "title": "Aide-mémoire mathématique",
            "desc": "Recense les formules LaTeX, gradients et critères statistiques.",
            "query": "Dresse un aide-mémoire exhaustif avec les formulations mathématiques, gradients et critères d'optimisation.",
        },
        {
            "badge": "Diagnostic",
            "title": "Diagnostics & MLOps",
            "desc": "Analyse du surapprentissage, data leakage et compromis de production.",
            "query": "Expose les pièges de modélisation (data leakage, surapprentissage, biais/variance) et les protocoles de validation croisée.",
        },
    ]

    cols = st.columns(len(prompt_templates))
    selected_card_query = None
    for idx, (col, c) in enumerate(zip(cols, prompt_templates)):
        with col:
            st.markdown(
                f"""
                <div class="action-card">
                    <div class="card-top-meta">
                        <span class="card-category-pill">{c['badge']}</span>
                    </div>
                    <div class="card-headline">{c['title']}</div>
                    <div class="card-description">{c['desc']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Lancer la requête →", key=f"btn_card_{idx}", use_container_width=True):
                selected_card_query = c["query"]

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Rendu des messages avec des avatars professionnels dédiés
    for turn in st.session_state.chat_history:
        if turn["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(f'<div class="chat-sender-label user-label">PILOTE // {st.user.name}</div>', unsafe_allow_html=True)
                st.markdown(turn["content"])
        else:
            with st.chat_message("assistant", avatar="⚡"):
                st.markdown('<div class="chat-sender-label ai-label">MEMORIX // RAG CORE INFERENCE</div>', unsafe_allow_html=True)
                st.markdown(turn["content"])
                if turn.get("warning"):
                    st.warning(turn["warning"])
                if turn.get("docs"):
                    show_sources(turn["docs"])

    auto_query = None
    if st.session_state.get("initial_prefill_query"):
        auto_query = st.session_state.initial_prefill_query
        st.session_state.initial_prefill_query = None

    user_input = st.chat_input("Formuler une requête technique en Data Science...")
    question = selected_card_query or auto_query or user_input

    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user", avatar="👤"):
            st.markdown(f'<div class="chat-sender-label user-label">PILOTE // {st.user.name}</div>', unsafe_allow_html=True)
            st.markdown(question)

        with st.chat_message("assistant", avatar="⚡"):
            st.markdown('<div class="chat-sender-label ai-label">MEMORIX // RAG CORE INFERENCE</div>', unsafe_allow_html=True)
            with st.spinner("Recherche vectorielle et recoupement dans le corpus..."):
                answer_raw, docs, warning = core.answer_question(vectorstore, llm, qa_prompt, question)

            full_rendered_answer = st.write_stream(stream_text_generator(answer_raw))

            if warning:
                st.warning(warning)
            if docs:
                show_sources(docs)

        st.session_state.chat_history.append(
            {"role": "assistant", "content": full_rendered_answer, "docs": docs, "warning": warning}
        )


if mode == "Recherche augmentée":
    render_cockpit_header("Recherche augmentée", "Interrogation sémantique directe des bases locales.")
    render_broadcast_bar("RAG PIPELINE", "Vector retrieval actif avec recoupement mathématique et métadonnées.")
    render_chat_cockpit(vectorstore, llm, qa_prompt)


# ---------------------------------------------------------------------------
# Mode 2 : Synthèse générale
# ---------------------------------------------------------------------------

elif mode == "Synthèse générale":
    book_label = books[selected_book_id]
    render_cockpit_header(f"Synthèse // {book_label}", f"Périmètre : {selected_scope}")
    render_broadcast_bar("COMPILATION", "Extraction hiérarchique des points essentiels par réduction de contexte.")

    if st.button("Générer la synthèse", type="primary"):
        progress = st.empty()

        if selected_scope == "Cours entier":
            docs = core.get_book_chunks(vectorstore, selected_book_id)
            empty_msg = f"Aucun segment trouvé pour '{book_label}'."
        else:
            chapter_number = selected_scope.replace("Section ", "")
            docs = core.get_chapter_chunks(vectorstore, chapter_number, selected_book_id)
            empty_msg = f"Section {chapter_number} introuvable dans '{book_label}'."

        with st.spinner("Synthèse en cours..."):
            result = core.summarize_documents_structured(
                llm,
                docs,
                empty_msg,
                core.CHAPTER_SUMMARY_PROMPT,
                core.COMBINE_SUMMARIES_PROMPT,
                progress_callback=lambda m: progress.info(m),
            )

        progress.empty()
        show_structured_result(
            result,
            empty_fallback_message=empty_msg,
            export_title=f"Synthese_{book_label}_{selected_scope}",
        )
        show_sources(docs)


# ---------------------------------------------------------------------------
# Mode 3 : Fiche mémo technique
# ---------------------------------------------------------------------------

elif mode == "Fiche mémo technique":
    book_label = books[selected_book_id]
    render_cockpit_header(f"Fiche mémo // {book_label}", f"Périmètre : {selected_scope}")
    render_broadcast_bar("RÉVISION CIBLÉE", "Extraction des points clés prête à l'exportation.")

    if st.button("Compiler la fiche mémo", type="primary"):
        progress = st.empty()

        if selected_scope == "Cours entier":
            docs = core.get_book_chunks(vectorstore, selected_book_id)
            empty_msg = f"Données absentes pour '{book_label}'."
        else:
            chapter_number = selected_scope.replace("Section ", "")
            docs = core.get_chapter_chunks(vectorstore, chapter_number, selected_book_id)
            empty_msg = f"Section {chapter_number} introuvable dans '{book_label}'."

        with st.spinner("Extraction analytique en cours..."):
            result = core.summarize_documents_structured(
                llm,
                docs,
                empty_msg,
                core.FICHE_EXTRACT_PROMPT,
                core.COMBINE_FICHE_SECTION_PROMPT,
                progress_callback=lambda m: progress.info(m),
            )

        progress.empty()
        show_structured_result(
            result,
            empty_fallback_message=empty_msg,
            export_title=f"FicheMemo_{book_label}_{selected_scope}",
        )
        show_sources(docs)


# ---------------------------------------------------------------------------
# Mode 4 : Atelier Code & Diagnostics
# ---------------------------------------------------------------------------

elif mode == "Atelier Code & Diagnostics":
    render_cockpit_header("Atelier Code & Diagnostics", "Audit d'implémentations, détection de fuites de données et optimisation.")
    render_broadcast_bar("CODE REVIEW", "Vérification des algorithmes et scripts Python par rapport aux documents indexés.")

    col_code, col_settings = st.columns([3, 1])

    with col_settings:
        audit_type = st.selectbox(
            "Type d'analyse",
            [
                "Audit de fuite de données (Data Leakage)",
                "Optimisation vectorielle (NumPy / Pandas)",
                "Explication algorithmique pas-à-pas",
                "Validation des métriques d'évaluation",
            ]
        )
        target_course = st.selectbox("Référentiel de validation", ["Tous les cours"] + list(books.values()))

    with col_code:
        default_snippet = (
            "# Exemple : Pipeline d'entraînement potentiellement biaisé\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "from sklearn.linear_model import LogisticRegression\n\n"
            "scaler = StandardScaler()\n"
            "X_scaled = scaler.fit_transform(X)  # Attention : fit avant le split\n"
            "X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)\n"
            "model = LogisticRegression().fit(X_train, y_train)\n"
        )
        code_input = st.text_area("Snippet Python / Pipeline à auditer :", value=default_snippet, height=220)
        run_audit = st.button("Lancer l'audit de code", type="primary")

    if run_audit and code_input:
        with st.spinner("Confrontation avec le corpus de cours..."):
            inv_map = {v: k for k, v in books.items()}
            target_book_id = inv_map.get(target_course) if target_course != "Tous les cours" else None

            audit_query = (
                f"Analyse ce code sous le prisme '{audit_type}'. "
                f"Confronte-le aux bonnes pratiques et formules vues dans le cours. "
                f"Donne les corrections en code Python PEP 8 :\n\n```python\n{code_input}\n```"
            )

            answer_raw, docs, warning = core.answer_question(vectorstore, llm, qa_prompt, audit_query)

            st.markdown("### Rapport d'audit technique")
            st.markdown(answer_raw)
            if warning:
                st.warning(warning)
            if docs:
                show_sources(docs)


# ---------------------------------------------------------------------------
# Mode 5 : Validation QCM
# ---------------------------------------------------------------------------

elif mode == "Validation QCM":
    book_label = books[selected_book_id]
    render_cockpit_header(f"Validation interactive // {book_label}", f"Périmètre : {selected_scope}")
    render_broadcast_bar("ÉVALUATION CONTINUE", "Génération dynamique de QCM pour valider la rétention active.")

    num_questions = st.slider("Volume de questions", min_value=2, max_value=10, value=5)
    generate = st.button("Générer le protocole de test", type="primary")

    if generate:
        progress = st.empty()
        if selected_scope == "Cours entier":
            docs = core.get_book_chunks(vectorstore, selected_book_id)
        else:
            chapter_number = selected_scope.replace("Section ", "")
            docs = core.get_chapter_chunks(vectorstore, chapter_number, selected_book_id)

        with st.spinner("Génération des questions et distracteurs..."):
            questions, raw_text = core.generate_quiz(
                llm,
                docs,
                num_questions=num_questions,
                progress_callback=lambda m: progress.info(m),
            )

        progress.empty()
        st.session_state.quiz_questions = questions
        st.session_state.quiz_raw = raw_text
        st.session_state.quiz_submitted = False

    questions = st.session_state.get("quiz_questions")

    if questions is not None and len(questions) == 0:
        st.warning("Formatage incomplet de la réponse du modèle.")
        st.text(st.session_state.get("quiz_raw") or "(aucune réponse)")

    elif questions:
        with st.form("quiz_terminal_form"):
            user_answers = []
            for i, q in enumerate(questions, start=1):
                st.markdown(f"**Item {i:02d} // {q['question']}**")
                opts = [
                    "— Choisir l'option valide —",
                    f"A) {q['a']}",
                    f"B) {q['b']}",
                    f"C) {q['c']}",
                    f"D) {q['d']}",
                ]
                choice = st.radio(
                    f"item_{i}",
                    opts,
                    key=f"q_{i}",
                    label_visibility="collapsed",
                )
                user_answers.append(choice[0] if choice != opts[0] else None)
                st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)

            submitted = st.form_submit_button("Soumettre pour évaluation", type="primary")

        if submitted:
            score, results = core.grade_quiz(questions, user_answers)
            st.session_state.quiz_results = results
            st.session_state.quiz_score = score
            st.session_state.quiz_submitted = True
            tracker.record_quiz_results(selected_book_id, book_label, results)

        if st.session_state.get("quiz_submitted"):
            sc = st.session_state.quiz_score
            tot = len(questions)
            st.markdown(
                f'<div class="metric-terminal-pill">RÉSULTAT DU CONTRÔLE // {sc} / {tot} ({(sc/tot)*100:.1f}%)</div>',
                unsafe_allow_html=True,
            )
            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

            for i, r in enumerate(st.session_state.quiz_results, start=1):
                if r["user_answer"] is None:
                    st.warning(f"Item {i:02d} : non renseigné.")
                elif r["is_correct"]:
                    st.success(f"Item {i:02d} : validation confirmée.")
                else:
                    correct_val = r[r["correct"].lower()]
                    st.error(f"Item {i:02d} : incorrect — Réponse attendue : [{r['correct']}] {correct_val}")
                st.caption(f"Analyse contextuelle : {r['explanation']}")


# ---------------------------------------------------------------------------
# Mode : Flashcards (répétition espacée SM-2)
# ---------------------------------------------------------------------------

elif mode == "🗂️ Flashcards":
    book_label = books[selected_book_id]
    render_cockpit_header(f"🗂️ Flashcards // {book_label}", f"Périmètre : {selected_scope}")
    render_broadcast_bar("RÉPÉTITION ESPACÉE", "Algorithme SM-2 : les cartes reviennent plus ou moins souvent selon ta mémorisation.")

    stats = tracker.get_flashcard_deck_stats(selected_book_id)
    col1, col2 = st.columns(2)
    col1.markdown(f'<div class="metric-terminal-pill">{stats["total"]} CARTE(S) AU TOTAL</div>', unsafe_allow_html=True)
    col2.markdown(f'<div class="metric-terminal-pill">{stats["dues"]} À RÉVISER MAINTENANT</div>', unsafe_allow_html=True)
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    with st.expander("➕ Générer de nouvelles flashcards"):
        num_cards = st.slider("Nombre de cartes à générer", min_value=5, max_value=20, value=10, key="fc_num")
        if st.button("Générer", type="primary", key="fc_generate"):
            progress = st.empty()
            if selected_scope == "Cours entier":
                docs = core.get_book_chunks(vectorstore, selected_book_id)
            else:
                chapter_number = selected_scope.replace("Section ", "")
                docs = core.get_chapter_chunks(vectorstore, chapter_number, selected_book_id)

            with st.spinner("Extraction des notions clés..."):
                cards, raw = core.generate_flashcards(
                    llm, docs, num_cards=num_cards, progress_callback=lambda m: progress.info(m),
                )
            progress.empty()

            if not cards:
                st.warning("Aucune flashcard n'a pu être générée. Réponse brute du modèle :")
                st.text(raw or "(aucun contenu)")
            else:
                tracker.add_flashcards(selected_book_id, book_label, cards)
                st.success(f"{len(cards)} flashcard(s) ajoutée(s) au paquet « {book_label} ».")
                st.rerun()

        if stats["total"] > 0 and st.button("🗑️ Réinitialiser ce paquet", key="fc_reset"):
            tracker.reset_flashcards(selected_book_id)
            for k in ("fc_session", "fc_index", "fc_flipped", "fc_stats"):
                st.session_state.pop(k, None)
            st.success("Paquet réinitialisé.")
            st.rerun()

    st.markdown("---")

    if stats["dues"] == 0:
        if stats["total"] == 0:
            st.info("Aucune flashcard dans ce paquet. Génère-en avec le bouton ci-dessus.")
        else:
            st.success("✅ Aucune carte à réviser pour l'instant. Reviens plus tard !")
    else:
        if st.session_state.get("fc_session") is None:
            if st.button(f"▶️ Réviser maintenant ({stats['dues']} carte(s))", type="primary"):
                st.session_state.fc_session = tracker.get_due_flashcards(selected_book_id, limit=stats["dues"])
                st.session_state.fc_index = 0
                st.session_state.fc_flipped = False
                st.session_state.fc_stats = {"again": 0, "hard": 0, "good": 0, "easy": 0}
                st.rerun()
        else:
            session = st.session_state.fc_session
            idx = st.session_state.fc_index

            if idx >= len(session):
                st.success("🎉 Session terminée !")
                s = st.session_state.fc_stats
                st.markdown(
                    f'<div class="metric-terminal-pill">'
                    f'✅ {s["easy"] + s["good"]} bien sues · 🟠 {s["hard"]} difficiles · 🔴 {s["again"]} à revoir'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if st.button("Terminer"):
                    for k in ("fc_session", "fc_index", "fc_flipped", "fc_stats"):
                        st.session_state.pop(k, None)
                    st.rerun()
            else:
                card = session[idx]
                st.caption(f"Carte {idx + 1} / {len(session)}")
                with st.container(border=True):
                    if card.get("chapter") and card["chapter"] != "0":
                        st.caption(f"Section {card['chapter']}")
                    st.markdown(f"### {card['front']}")
                    if st.session_state.fc_flipped:
                        st.markdown("---")
                        st.markdown(card["back"])

                if not st.session_state.fc_flipped:
                    if st.button("👁️ Retourner la carte", type="primary", use_container_width=True):
                        st.session_state.fc_flipped = True
                        st.rerun()
                else:
                    st.markdown("**As-tu bien répondu ?**")
                    b1, b2, b3, b4 = st.columns(4)
                    quality = None
                    if b1.button("🔴 À revoir", use_container_width=True, key=f"fc_again_{idx}"):
                        quality = 0
                    if b2.button("🟠 Difficile", use_container_width=True, key=f"fc_hard_{idx}"):
                        quality = 3
                    if b3.button("🟢 Correct", use_container_width=True, key=f"fc_good_{idx}"):
                        quality = 4
                    if b4.button("🔵 Facile", use_container_width=True, key=f"fc_easy_{idx}"):
                        quality = 5

                    if quality is not None:
                        stats_key = {0: "again", 3: "hard", 4: "good", 5: "easy"}[quality]
                        st.session_state.fc_stats[stats_key] += 1
                        tracker.record_flashcard_review(card["id"], quality)
                        st.session_state.fc_index = idx + 1
                        st.session_state.fc_flipped = False
                        st.rerun()


# ---------------------------------------------------------------------------
# Mode 6 : Simulation orale
# ---------------------------------------------------------------------------

elif mode == "Simulation orale":
    book_label = books[selected_book_id]
    render_cockpit_header(f"Simulation orale // {book_label}", f"Périmètre : {selected_scope}")
    render_broadcast_bar("JURY TECHNIQUE", "Contrôle de la rigueur conceptuelle, des formules et de l'argumentation.")

    profile_labels = {v["label"]: k for k, v in core.ORAL_PROFILES.items()}
    selected_profile_label = st.selectbox("Contexte du jury", list(profile_labels.keys()))
    selected_profile_key = profile_labels[selected_profile_label]

    num_oral = st.slider("Nombre de questions", min_value=2, max_value=6, value=3)

    if st.button("Initialiser la session", type="primary"):
        progress = st.empty()
        if selected_scope == "Cours entier":
            docs = core.get_book_chunks(vectorstore, selected_book_id)
        else:
            chapter_number = selected_scope.replace("Section ", "")
            docs = core.get_chapter_chunks(vectorstore, chapter_number, selected_book_id)

        with st.spinner("Formulation des problématiques techniques..."):
            oral_questions = core.generate_oral_questions(
                llm,
                docs,
                num_questions=num_oral,
                profile_key=selected_profile_key,
                progress_callback=lambda m: progress.info(m),
            )

        progress.empty()
        st.session_state.oral_questions = oral_questions
        st.session_state.oral_profile_key = selected_profile_key
        st.session_state.oral_current_index = 0
        st.session_state.oral_history = []

    oral_questions = st.session_state.get("oral_questions")

    if not oral_questions:
        st.info("Prêt pour l'interrogation. Cliquez sur 'Initialiser la session'.")
    else:
        active_profile = core.ORAL_PROFILES.get(
            st.session_state.get("oral_profile_key", "lead_ds"),
            core.ORAL_PROFILES["lead_ds"],
        )
        st.caption(f"Jury actif : {active_profile['label']}")

        current_idx = st.session_state.get("oral_current_index", 0)
        total_q = len(oral_questions)

        if current_idx < total_q:
            item = oral_questions[current_idx]
            st.markdown(f"**Question {current_idx + 1:02d} / {total_q:02d}**")
            if item.get("chapter") and item["chapter"] != "0":
                st.caption(f"Section de référence : {item['chapter']}")
            st.markdown(f"> *{item['question']}*")

            ans = st.text_area("Votre argumentation technique :", key=f"oral_input_{current_idx}", height=140)

            if st.button("Soumettre au jury", type="primary", key=f"btn_eval_{current_idx}"):
                with st.spinner("Évaluation en cours par le jury..."):
                    evaluation = core.grade_oral_answer(
                        llm,
                        item["question"],
                        item["reference"],
                        ans,
                        profile_key=st.session_state.get("oral_profile_key", "lead_ds"),
                    )

                tracker.record_attempt(
                    selected_book_id,
                    book_label,
                    item.get("chapter", "0"),
                    "oral",
                    score=evaluation["score"],
                    question=item["question"],
                )

                st.session_state.oral_history.append({
                    "question": item["question"],
                    "chapter": item.get("chapter", "0"),
                    "answer": ans,
                    "level": evaluation["level"],
                    "feedback": evaluation["feedback"],
                    "reference": item["reference"],
                })
                st.session_state.oral_current_index = current_idx + 1
                st.rerun()
        else:
            st.success("Session terminée. Synthèse des appréciations ci-dessous.")

        for i, h in enumerate(st.session_state.get("oral_history", []), start=1):
            with st.expander(f"Résultat Question {i:02d} // Mention : {h['level']}"):
                st.markdown(f"**Énoncé :** {h['question']}")
                st.markdown(f"**Réponse fournie :** {h['answer'] or 'Aucune réponse'}")
                st.markdown(
                    f'<span class="level-badge level-{h["level"]}">{h["level"].upper()}</span>',
                    unsafe_allow_html=True,
                )
                st.markdown(f"**Critique du jury :** {h['feedback']}")
                with st.expander("Contenu de référence utilisé pour cette correction"):
                    st.caption(
                        "Extrait exact du cours ayant servi de support d'évaluation :"
                    )
                    st.text(h.get("reference", "(non disponible)"))


# ---------------------------------------------------------------------------
# Mode : Boss Fight
# ---------------------------------------------------------------------------

elif mode == "⚔️ Boss Fight":
    book_label = books[selected_book_id]
    render_cockpit_header(f"⚔️ Boss Fight // {book_label}", f"Périmètre : {selected_scope}")
    render_broadcast_bar("COMBAT TECHNIQUE", "Chaque bonne réponse inflige des dégâts au jury. Chaque erreur en coûte.")

    xp_actuel = tracker.get_xp()
    niveau, xp_niveau, xp_requis = tracker.level_from_xp(xp_actuel)
    st.markdown(boss_battle.render_xp_bar_svg(xp_actuel, niveau, xp_niveau, xp_requis), unsafe_allow_html=True)
    if st.session_state.pop("battle_level_up", False):
        st.success(f"🌟 Niveau supérieur ! Tu es maintenant niveau {niveau}.")
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns([2, 1, 1])
    with col_a:
        profile_labels = {v["label"]: k for k, v in core.ORAL_PROFILES.items()}
        selected_profile_label = st.selectbox("Choisis ton adversaire", list(profile_labels.keys()))
        selected_profile_key = profile_labels[selected_profile_label]
    with col_b:
        num_oral = st.slider("Manches", min_value=3, max_value=8, value=5)
    with col_c:
        son_actif = st.checkbox("🔊 Sons", value=st.session_state.get("battle_son_actif", True), key="battle_son_actif")

    if st.button("⚔️ Lancer le combat", type="primary"):
        progress = st.empty()
        if selected_scope == "Cours entier":
            docs = core.get_book_chunks(vectorstore, selected_book_id)
        else:
            chapter_number = selected_scope.replace("Section ", "")
            docs = core.get_chapter_chunks(vectorstore, chapter_number, selected_book_id)

        with st.spinner("Le jury se prépare..."):
            questions = core.generate_oral_questions(
                llm, docs, num_questions=num_oral, profile_key=selected_profile_key,
                progress_callback=lambda m: progress.info(m),
            )
        progress.empty()
        st.session_state.battle_questions = questions
        st.session_state.battle_profile_key = selected_profile_key
        st.session_state.battle_book_hp_max = boss_battle.scale_boss_hp(num_oral)
        st.session_state.battle_index = 0
        st.session_state.battle_boss_hp = st.session_state.battle_book_hp_max
        st.session_state.battle_player_hp = boss_battle.PLAYER_HP_MAX
        st.session_state.battle_combo = 0
        st.session_state.battle_combo_max = 0
        st.session_state.battle_log = []
        st.session_state.battle_status = "en_cours"
        for k in ("battle_event", "battle_dmg_boss", "battle_dmg_player", "battle_sound_nonce", "battle_result"):
            st.session_state.pop(k, None)
        st.rerun()

    questions = st.session_state.get("battle_questions")
    if not questions:
        st.info("Choisis ton adversaire et clique sur « Lancer le combat ».")
    else:
        boss_hp = st.session_state.battle_boss_hp
        boss_hp_max = st.session_state.get("battle_book_hp_max", boss_battle.BOSS_HP_MAX)
        player_hp = st.session_state.battle_player_hp
        combo = st.session_state.battle_combo
        status = st.session_state.get("battle_status", "en_cours")
        banner = st.session_state.get("battle_result")

        event = st.session_state.pop("battle_event", None)
        dmg_boss = st.session_state.pop("battle_dmg_boss", None)
        dmg_player = st.session_state.pop("battle_dmg_player", None)
        nonce = st.session_state.pop("battle_sound_nonce", None)

        st.markdown(
            boss_battle.render_battle_svg(
                st.session_state.battle_profile_key, boss_hp, player_hp, combo,
                event=event, dmg_boss=dmg_boss, dmg_player=dmg_player, banner=banner,
                boss_hp_max=boss_hp_max,
            ),
            unsafe_allow_html=True,
        )

        if event and nonce and son_actif:
            components.html(boss_battle.render_sound_effect(event, nonce), height=0)

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

        idx = st.session_state.battle_index
        total_q = len(questions)

        if status == "en_cours" and boss_hp > 0 and player_hp > 0 and idx < total_q:
            item = questions[idx]
            st.markdown(f"**Manche {idx + 1} / {total_q}**")
            st.markdown(f"> *{item['question']}*")

            ans = st.text_area("Ta réplique :", key=f"battle_input_{idx}", height=120)

            if st.button("Attaquer !", type="primary", key=f"battle_attack_{idx}"):
                with st.spinner("Le jury délibère..."):
                    evaluation = core.grade_oral_answer(
                        llm, item["question"], item["reference"], ans, profile_key=st.session_state.battle_profile_key,
                    )
                score = evaluation["score"]
                degats = boss_battle.score_to_damage(score, combo)
                contre = boss_battle.score_to_counter(score)
                nouveau_combo = boss_battle.next_combo(combo, score)

                nouveau_boss_hp = max(0.0, boss_hp - degats)
                nouveau_player_hp = max(0.0, player_hp - contre)
                st.session_state.battle_boss_hp = nouveau_boss_hp
                st.session_state.battle_player_hp = nouveau_player_hp
                st.session_state.battle_combo = nouveau_combo
                st.session_state.battle_combo_max = max(st.session_state.battle_combo_max, nouveau_combo)
                st.session_state.battle_index = idx + 1

                if nouveau_boss_hp <= 0:
                    tour_event = "victory"
                    st.session_state.battle_result = "victory"
                elif nouveau_player_hp <= 0:
                    tour_event = "defeat"
                    st.session_state.battle_result = "defeat"
                elif contre > 0:
                    tour_event = "hit_player"
                elif nouveau_combo > combo:
                    tour_event = "combo_up"
                else:
                    tour_event = "hit_boss"

                st.session_state.battle_event = tour_event
                st.session_state.battle_dmg_boss = degats
                st.session_state.battle_dmg_player = contre
                st.session_state.battle_sound_nonce = time.time()

                tracker.record_attempt(
                    selected_book_id, book_label, item.get("chapter", "0"), "oral",
                    score=score, question=item["question"],
                )
                st.session_state.battle_log.append({
                    "question": item["question"], "score": score, "degats": degats,
                    "contre": contre, "level": evaluation["level"], "feedback": evaluation["feedback"],
                })
                st.rerun()
        else:
            if boss_hp <= 0:
                pv_restants_pct = player_hp / boss_battle.PLAYER_HP_MAX
                xp = boss_battle.xp_reward(0, pv_restants_pct, st.session_state.battle_combo_max)
                if st.session_state.battle_status != "recompense_donnee":
                    niveau_avant, _, _ = tracker.level_from_xp(tracker.get_xp())
                    nouveau_total = tracker.add_xp(xp)
                    niveau_apres, _, _ = tracker.level_from_xp(nouveau_total)
                    tracker.record_boss_victory(
                        st.session_state.battle_profile_key,
                        core.ORAL_PROFILES[st.session_state.battle_profile_key]["label"],
                        selected_book_id, book_label, xp,
                    )
                    st.session_state.battle_status = "recompense_donnee"
                    st.session_state.battle_last_xp_gain = xp
                    if niveau_apres > niveau_avant:
                        st.session_state.battle_level_up = True
                    st.rerun()
                st.success(f"🏆 Victoire ! Le jury est convaincu. +{st.session_state.get('battle_last_xp_gain', xp)} XP")
                st.balloons()
            elif player_hp <= 0:
                st.error("💀 Défaite... Le jury n'est pas convaincu. Retente le combat après avoir révisé !")
            else:
                st.warning(f"⏱️ Manches épuisées. Le jury tient encore ({boss_hp:.0f} PV). Retente avec plus de manches !")

            if st.button("Nouveau combat"):
                for key in ("battle_questions", "battle_profile_key", "battle_index", "battle_boss_hp",
                            "battle_player_hp", "battle_combo", "battle_combo_max", "battle_log", "battle_status",
                            "battle_book_hp_max", "battle_last_xp_gain", "battle_event", "battle_dmg_boss",
                            "battle_dmg_player", "battle_sound_nonce", "battle_result"):
                    st.session_state.pop(key, None)
                st.rerun()

        if st.session_state.get("battle_log"):
            st.markdown("---")
            with st.expander("Journal de combat"):
                for i, entry in enumerate(st.session_state.battle_log, start=1):
                    st.markdown(
                        f"**Manche {i}** — {entry['level']} (score {entry['score']*100:.0f}%) "
                        f"— {entry['degats']:.0f} dégâts infligés"
                        + (f", {entry['contre']:.0f} subis" if entry["contre"] else "")
                    )
                    st.caption(entry["feedback"])


# ---------------------------------------------------------------------------
# Mode : Duel de Révision
# ---------------------------------------------------------------------------

elif mode == "🤺 Duel de Révision":
    book_label = books[selected_book_id]
    render_cockpit_header(f"🤺 Duel de Révision // {book_label}", f"Périmètre : {selected_scope}")
    render_broadcast_bar("FACE-À-FACE LOCAL", "Même appareil, même questions : que le meilleur gagne.")

    leaderboard = tracker.get_duel_leaderboard(5)
    if leaderboard:
        with st.expander("🏆 Tableau des vainqueurs"):
            for i, row in enumerate(leaderboard, start=1):
                st.markdown(f"**{i}.** {row['winner']} — {row['victoires']} victoire(s)")

    if not st.session_state.get("duel_questions"):
        col1, col2 = st.columns(2)
        nom1 = col1.text_input("Joueur 1", value="Joueur 1", key="duel_nom1")
        nom2 = col2.text_input("Joueur 2", value="Joueur 2", key="duel_nom2")
        num_manches = st.slider("Nombre de manches", min_value=3, max_value=10, value=5)

        if st.button("🤺 Lancer le duel", type="primary"):
            progress = st.empty()
            if selected_scope == "Cours entier":
                docs = core.get_book_chunks(vectorstore, selected_book_id)
            else:
                chapter_number = selected_scope.replace("Section ", "")
                docs = core.get_chapter_chunks(vectorstore, chapter_number, selected_book_id)

            with st.spinner("Préparation des questions..."):
                questions, _ = core.generate_quiz(
                    llm, docs, num_questions=num_manches, progress_callback=lambda m: progress.info(m),
                )
            progress.empty()

            if not questions:
                st.warning("Impossible de générer des questions pour ce cours.")
            else:
                st.session_state.duel_questions = questions
                st.session_state.duel_nom1_actif = nom1 or "Joueur 1"
                st.session_state.duel_nom2_actif = nom2 or "Joueur 2"
                st.session_state.duel_index = 0
                st.session_state.duel_score1 = 0
                st.session_state.duel_score2 = 0
                st.session_state.duel_stage = "p1"
                st.session_state.duel_intro_vue = False
                st.rerun()
    else:
        questions = st.session_state.duel_questions
        nom1 = st.session_state.duel_nom1_actif
        nom2 = st.session_state.duel_nom2_actif
        idx = st.session_state.duel_index
        total = len(questions)
        stage = st.session_state.duel_stage

        if not st.session_state.duel_intro_vue:
            st.markdown(duel_arena.render_vs_intro(nom1, nom2), unsafe_allow_html=True)
            if st.button("C'est parti !", type="primary"):
                st.session_state.duel_intro_vue = True
                st.rerun()
        elif idx >= total:
            score1, score2 = st.session_state.duel_score1, st.session_state.duel_score2
            st.markdown(duel_arena.render_victory_banner(
                nom1 if score1 > score2 else nom2, score1, score2), unsafe_allow_html=True)
            components.html(duel_arena.render_duel_sound("victory", time.time()), height=0)
            st.markdown(duel_arena.render_scoreboard(nom1, score1, nom2, score2, total, total), unsafe_allow_html=True)
            if st.session_state.get("duel_status") != "enregistre":
                tracker.record_duel_result(selected_book_id, book_label, nom1, nom2, score1, score2)
                st.session_state.duel_status = "enregistre"
            st.balloons()
            if st.button("Nouveau duel"):
                for k in list(st.session_state.keys()):
                    if k.startswith("duel_"):
                        del st.session_state[k]
                st.rerun()
        else:
            q = questions[idx]
            event = st.session_state.pop("duel_event", None)
            pts1 = st.session_state.pop("duel_pts1", 0)
            pts2 = st.session_state.pop("duel_pts2", 0)
            nonce = st.session_state.pop("duel_sound_nonce", None)

            st.markdown(
                duel_arena.render_scoreboard(
                    nom1, st.session_state.duel_score1, nom2, st.session_state.duel_score2,
                    idx + 1, total, event=event, points_gagnes1=pts1, points_gagnes2=pts2,
                ),
                unsafe_allow_html=True,
            )
            if event and nonce:
                son = "steal" if event in ("p1_marque", "p2_marque") and (pts1 == 15 or pts2 == 15) else \
                      "point" if event in ("p1_marque", "p2_marque", "egalite") else "reveal"
                components.html(duel_arena.render_duel_sound(son, nonce), height=0)

            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

            if stage == "p1":
                st.markdown(f"### Tour de {nom1}")
                st.markdown(f"**{q['question']}**")
                choix = st.radio(
                    "Ta réponse :",
                    [f"A) {q['a']}", f"B) {q['b']}", f"C) {q['c']}", f"D) {q['d']}"],
                    key=f"duel_choix1_{idx}", label_visibility="collapsed",
                )
                if st.button("Valider ma réponse", type="primary", key=f"duel_valid1_{idx}"):
                    st.session_state.duel_p1_answer = choix[0]
                    st.session_state.duel_stage = "transition"
                    st.rerun()
            elif stage == "transition":
                st.info(f"📱 Passe l'écran à **{nom2}**. {nom1}, ne regarde pas !")
                if st.button(f"{nom2}, je suis prêt(e)", type="primary"):
                    st.session_state.duel_stage = "p2"
                    st.rerun()
            elif stage == "p2":
                st.markdown(f"### Tour de {nom2}")
                st.markdown(f"**{q['question']}**")
                choix = st.radio(
                    "Ta réponse :",
                    [f"A) {q['a']}", f"B) {q['b']}", f"C) {q['c']}", f"D) {q['d']}"],
                    key=f"duel_choix2_{idx}", label_visibility="collapsed",
                )
                if st.button("Valider ma réponse", type="primary", key=f"duel_valid2_{idx}"):
                    p2_answer = choix[0]
                    p1_answer = st.session_state.duel_p1_answer
                    correct = q["correct"]
                    c1, c2 = p1_answer == correct, p2_answer == correct
                    gain1 = gain2 = 0
                    if c1 and c2:
                        gain1, gain2 = 10, 10
                        evt = "egalite"
                    elif c1:
                        gain1 = 15
                        evt = "p1_marque"
                    elif c2:
                        gain2 = 15
                        evt = "p2_marque"
                    else:
                        evt = "aucun"

                    st.session_state.duel_score1 += gain1
                    st.session_state.duel_score2 += gain2
                    st.session_state.duel_reveal = {
                        "p1": p1_answer, "p2": p2_answer, "correct": correct,
                        "explanation": q["explanation"], "gain1": gain1, "gain2": gain2,
                    }
                    st.session_state.duel_event = evt
                    st.session_state.duel_pts1 = gain1
                    st.session_state.duel_pts2 = gain2
                    st.session_state.duel_sound_nonce = time.time()
                    st.session_state.duel_stage = "reveal"
                    st.rerun()
            elif stage == "reveal":
                r = st.session_state.duel_reveal
                st.markdown(f"**{q['question']}**")
                st.markdown(f"✅ Bonne réponse : **{r['correct']}) {q[r['correct'].lower()]}**")
                st.caption(r["explanation"])
                c1, c2 = st.columns(2)
                icone1 = "✅" if r["p1"] == r["correct"] else "❌"
                icone2 = "✅" if r["p2"] == r["correct"] else "❌"
                c1.markdown(f"{icone1} **{nom1}** a répondu {r['p1']}" + (f" (+{r['gain1']})" if r['gain1'] else ""))
                c2.markdown(f"{icone2} **{nom2}** a répondu {r['p2']}" + (f" (+{r['gain2']})" if r['gain2'] else ""))
                if st.button("Manche suivante →", type="primary"):
                    st.session_state.duel_index = idx + 1
                    st.session_state.duel_stage = "p1"
                    st.rerun()


# ---------------------------------------------------------------------------
# Mode : Podcast Révision
# ---------------------------------------------------------------------------

elif mode == "Podcast Révision":
    book_label = books[selected_book_id]
    render_cockpit_header(f"🎧 Podcast Révision // {book_label}", f"Périmètre : {selected_scope}")
    render_broadcast_bar("ÉCOUTE ACTIVE", "Un dialogue Prof/Étudiant généré à partir du cours, lu à voix haute.")

    if st.button("🎬 Générer l'épisode", type="primary"):
        progress = st.empty()
        if selected_scope == "Cours entier":
            docs = core.get_book_chunks(vectorstore, selected_book_id)
        else:
            chapter_number = selected_scope.replace("Section ", "")
            docs = core.get_chapter_chunks(vectorstore, chapter_number, selected_book_id)

        with st.spinner("Écriture du script..."):
            script = core.generate_podcast_script(llm, docs, progress_callback=lambda m: progress.info(m))
        progress.empty()

        if not script:
            st.warning("Impossible de générer un dialogue pour ce contenu.")
        else:
            st.session_state.podcast_script = script
            st.rerun()

    script = st.session_state.get("podcast_script")
    if not script:
        st.info("Clique sur « Générer l'épisode » pour créer un dialogue audio à partir du cours.")
    else:
        st.caption(f"{len(script)} répliques · lu directement par ton navigateur, aucun fichier téléchargé.")
        components.html(podcast_player.render_podcast_player(script), height=460)
        if st.button("🗑️ Nouvel épisode"):
            st.session_state.pop("podcast_script", None)
            st.rerun()


# ---------------------------------------------------------------------------
# Mode : Calibration de confiance
# ---------------------------------------------------------------------------

elif mode == "Calibration":
    book_label = books[selected_book_id]
    render_cockpit_header(f"🔮 Calibration // {book_label}", f"Périmètre : {selected_scope}")
    render_broadcast_bar("MÉTACOGNITION", "Annonce ta confiance avant chaque réponse : sais-tu quand tu sais ?")

    brier_global = tracker.get_brier_score(selected_book_id)
    buckets_global = tracker.get_calibration_buckets(selected_book_id)

    if buckets_global:
        if brier_global is not None:
            libelle, _ = calibration_chart.brier_label(brier_global)
            col_score, col_libelle = st.columns([1, 3])
            col_score.markdown(
                f'<div class="metric-terminal-pill">Brier {brier_global:.3f}</div>',
                unsafe_allow_html=True,
            )
            col_libelle.caption(libelle)

        st.markdown(calibration_chart.render_calibration_curve(buckets_global), unsafe_allow_html=True)
        insight = tracker.get_calibration_insight(buckets_global)
        if insight:
            st.info(insight)
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    else:
        st.caption("Aucune donnée de calibration pour ce cours pour l'instant — lance une session ci-dessous.")

    num_q = st.slider("Nombre de questions", min_value=3, max_value=10, value=5)

    if st.button("🔮 Lancer une session de calibration", type="primary"):
        progress = st.empty()
        if selected_scope == "Cours entier":
            docs = core.get_book_chunks(vectorstore, selected_book_id)
        else:
            chapter_number = selected_scope.replace("Section ", "")
            docs = core.get_chapter_chunks(vectorstore, chapter_number, selected_book_id)

        with st.spinner("Génération des questions..."):
            questions, _ = core.generate_quiz(
                llm, docs, num_questions=num_q, progress_callback=lambda m: progress.info(m),
            )
        progress.empty()

        if not questions:
            st.warning("Impossible de générer des questions pour ce cours.")
        else:
            st.session_state.calib_questions = questions
            st.session_state.calib_index = 0
            st.session_state.calib_records = []
            st.session_state.calib_stage = "question"
            st.rerun()

    questions = st.session_state.get("calib_questions")

    if questions:
        idx = st.session_state.calib_index
        total = len(questions)
        stage = st.session_state.calib_stage

        if idx >= total:
            st.success("🎉 Session de calibration terminée !")
            records = st.session_state.calib_records
            n_correct = sum(1 for r in records if r["correct"])
            st.markdown(
                f'<div class="metric-terminal-pill">✅ {n_correct}/{total} bonnes réponses</div>',
                unsafe_allow_html=True,
            )
            if st.button("Voir ma courbe de calibration mise à jour"):
                for k in ("calib_questions", "calib_index", "calib_records", "calib_stage"):
                    st.session_state.pop(k, None)
                st.rerun()
        else:
            q = questions[idx]
            st.caption(f"Question {idx + 1} / {total}")
            st.markdown(f"**{q['question']}**")
            opts = [f"A) {q['a']}", f"B) {q['b']}", f"C) {q['c']}", f"D) {q['d']}"]

            if stage == "question":
                choix = st.radio("Ta réponse :", opts, key=f"calib_choix_{idx}", label_visibility="collapsed")
                confiance = st.slider(
                    "À quel point es-tu confiant(e) dans cette réponse ?",
                    min_value=0, max_value=100, value=50, step=5, key=f"calib_conf_{idx}",
                    format="%d%%",
                )
                if st.button("Valider", type="primary", key=f"calib_valid_{idx}"):
                    correct = choix[0] == q["correct"]
                    st.session_state.calib_last = {
                        "choix": choix[0], "confiance": confiance, "correct": correct,
                    }
                    tracker.record_calibration(
                        selected_book_id, book_label, q.get("chapter", "0"), confiance, correct,
                    )
                    st.session_state.calib_records.append({"correct": correct, "confiance": confiance})
                    st.session_state.calib_stage = "reveal"
                    st.rerun()
            elif stage == "reveal":
                r = st.session_state.calib_last
                if r["correct"]:
                    st.success(f"✅ Bonne réponse ! Tu avais annoncé {r['confiance']}% de confiance.")
                else:
                    correct_opt = q['correct'].lower()
                    st.error(
                        f"❌ Ce n'était pas ça — la bonne réponse était {q['correct']}) {q[correct_opt]}. "
                        f"Tu avais annoncé {r['confiance']}% de confiance."
                    )
                st.caption(q["explanation"])
                if st.button("Question suivante →", type="primary"):
                    st.session_state.calib_index = idx + 1
                    st.session_state.calib_stage = "question"
                    st.rerun()


# ---------------------------------------------------------------------------
# Mode : Graphe de connaissances
# ---------------------------------------------------------------------------

elif mode == "Graphe de connaissances":
    render_cockpit_header(
        "Graphe de connaissances",
        "Carte des cours, sections et concepts clés — calcul local, sans appel au modèle.",
    )
    render_broadcast_bar("GRAPH ENGINE", "Cours → sections → concepts. La couleur d'une section reflète ta maîtrise.")

    selected_labels = st.multiselect("Cours à afficher", list(books.values()), default=list(books.values()))
    col1, col2 = st.columns(2)
    with col1:
        per_section = st.slider("Concepts par section", min_value=1, max_value=6, value=3)
    with col2:
        max_concepts = st.slider("Concepts au total", min_value=10, max_value=80, value=40, step=5)

    if st.button("Recalculer depuis l'index"):
        get_cached_graph.clear()

    if not selected_labels:
        st.info("Sélectionne au moins un cours.")
    else:
        chosen = tuple((bid, lbl) for bid, lbl in books.items() if lbl in selected_labels)
        graph, positions = get_cached_graph(vectorstore, chosen, per_section, max_concepts)

        if not graph["nodes"]:
            st.warning("Aucune section détectée dans ces cours. Vérifie l'indexation (`python ingest.py`).")
        else:
            stats = tracker.get_section_stats() if tracker.has_any_history() else []
            svg = kg.render_svg(graph, positions, stats=stats)

            n_books = sum(1 for n in graph["nodes"] if n["kind"] == "book")
            n_secs = sum(1 for n in graph["nodes"] if n["kind"] == "section")
            n_concepts = sum(1 for n in graph["nodes"] if n["kind"] == "concept")

            st.markdown(
                f'<div class="metric-terminal-pill">{n_books} COURS // {n_secs} SECTIONS // {n_concepts} CONCEPTS</div>',
                unsafe_allow_html=True,
            )
            components.html(kg.wrap_html(svg), height=700, scrolling=False)
            st.caption("Survole un noeud pour isoler ses voisins. Les traits pointillés relient des concepts vus ensemble dans le cours.")
            st.download_button(
                "Exporter en SVG",
                data=svg,
                file_name="graphe_memorix.svg",
                mime="image/svg+xml",
            )


# ---------------------------------------------------------------------------
# Mode 7 : Métriques & Suivi
# ---------------------------------------------------------------------------

elif mode == "Métriques & Suivi":
    render_cockpit_header("Métriques d'apprentissage", "Analyse télémétrique locale de progression (progress.db).")
    render_broadcast_bar("DIAGNOSTIC", "Détection automatique des modules sous le seuil de maîtrise (60%).")

    if not tracker.has_any_history():
        st.info("Aucune donnée enregistrée. Effectuez des quiz ou examens pour alimenter les métriques.")
    else:
        stats = tracker.get_section_stats()

        st.markdown("**Indice de maîtrise par section**")
        for s in stats:
            pct = s["avg_score"] * 100
            sec_title = f"{s['book_label']} — Section {s['chapter']}" if s["chapter"] != "0" else s["book_label"]
            st.markdown(f"`{sec_title}` // **{pct:.1f}%** ({s['attempts']} session(s))")
            st.progress(min(max(s["avg_score"], 0.0), 1.0))

        st.markdown("---")
        st.markdown("**Points d'attention prioritaire (< 60%)**")

        weak = tracker.get_weak_sections(threshold=0.6, min_attempts=1)
        if not weak:
            st.success("Toutes les sections dépassent les exigences minimales.")
        else:
            for w in weak:
                lbl = f"{w['book_label']} — Section {w['chapter']}" if w["chapter"] != "0" else w["book_label"]
                st.warning(f"`{lbl}` // Score moyen : {w['avg_score']*100:.0f}% sur {w['attempts']} passage(s)")

                if w["book_id"] in books and st.button(
                    "Générer une fiche ciblée sur cette lacune",
                    key=f"remed_{w['book_id']}_{w['chapter']}",
                ):
                    with st.spinner("Compilation du mémo de renfort..."):
                        if w["chapter"] == "0":
                            t_docs = core.get_book_chunks(vectorstore, w["book_id"])
                        else:
                            t_docs = core.get_chapter_chunks(vectorstore, w["chapter"], w["book_id"])
                        res = core.summarize_documents_structured(
                            llm,
                            t_docs,
                            "Contenu non localisé.",
                            core.FICHE_EXTRACT_PROMPT,
                            core.COMBINE_FICHE_SECTION_PROMPT,
                        )
                    show_structured_result(res, export_title=f"Remediation_{lbl}")

        st.markdown("---")
        if st.button("Réinitialiser l'historique de suivi"):
            tracker.reset_history()
            st.success("Base de télémétrie remise à zéro.")
            st.rerun()


# ---------------------------------------------------------------------------
# Mode 8 : Plan de révision
# ---------------------------------------------------------------------------

elif mode == "Plan de révision":
    render_cockpit_header(
        "Plan de révision",
        "Planning généré à partir de ton historique de progression — sans appel au modèle.",
    )
    render_broadcast_bar(
        "SCHEDULER",
        "Priorise les sections faibles ou jamais testées, réparties sur ton calendrier.",
    )

    if not tracker.has_any_history():
        st.info(
            "Aucune donnée de progression pour le moment. Fais au moins un quiz ou un examen "
            "oral pour que le planificateur ait de quoi travailler."
        )
    else:
        col1, col2 = st.columns(2)
        with col1:
            num_days = st.slider("Jours avant l'échéance", min_value=1, max_value=14, value=5)
        with col2:
            sections_per_day = st.slider("Sections par jour", min_value=1, max_value=4, value=2)

        if st.button("Générer mon plan", type="primary"):
            all_sections = []
            for book_id, book_label in books.items():
                for chapter in core.get_sections_for_book(vectorstore, book_id):
                    all_sections.append({"book_id": book_id, "book_label": book_label, "chapter": chapter})

            stats = tracker.get_section_stats()
            plan = revision_planner.build_plan(all_sections, stats, num_days, sections_per_day)
            st.session_state.revision_plan = plan

        plan = st.session_state.get("revision_plan")

        if plan:
            for day in plan:
                st.markdown(f"**{day['day_label']}**")
                for item in day["items"]:
                    if item["status"] == "tested":
                        pct = item["avg_score"] * 100
                        status_txt = f"{pct:.0f}% de moyenne ({item['attempts']} tentative(s))"
                        level_class = (
                            "level-Insuffisant" if pct < 40 else
                            "level-Partiel" if pct < 60 else
                            "level-Bon" if pct < 85 else "level-Excellent"
                        )
                    else:
                        status_txt = "Jamais testée"
                        level_class = "level-Partiel"

                    sec_label = (
                        f"{item['book_label']} — Section {item['chapter']}"
                        if item["chapter"] != "0" else item["book_label"]
                    )

                    c1, c2 = st.columns([4, 1])
                    with c1:
                        st.markdown(
                            f'`{sec_label}` — <span class="level-badge {level_class}">{status_txt}</span>',
                            unsafe_allow_html=True,
                        )
                    with c2:
                        btn_key = f"plan_fiche_{day['day_label']}_{item['book_id']}_{item['chapter']}"
                        if st.button("Fiche", key=btn_key, use_container_width=True):
                            with st.spinner("Génération de la fiche ciblée..."):
                                if item["chapter"] == "0":
                                    t_docs = core.get_book_chunks(vectorstore, item["book_id"])
                                else:
                                    t_docs = core.get_chapter_chunks(vectorstore, item["chapter"], item["book_id"])
                                res = core.summarize_documents_structured(
                                    llm,
                                    t_docs,
                                    "Contenu non localisé.",
                                    core.FICHE_EXTRACT_PROMPT,
                                    core.COMBINE_FICHE_SECTION_PROMPT,
                                )
                            show_structured_result(res, export_title=f"Fiche_{sec_label}")
                st.markdown("---")
        else:
            st.info("Règle le nombre de jours et clique sur « Générer mon plan ».")


# ---------------------------------------------------------------------------
# Mode 9 : Session PDF éphémère
# ---------------------------------------------------------------------------

elif mode == "Session PDF éphémère":
    render_cockpit_header("Session PDF éphémère", "Document chargé uniquement en mémoire vive (RAM volatile).")
    render_broadcast_bar("ISOLATION", "Aucune donnée n'est inscrite dans la base vectorielle persistante.")

    file_up = st.file_uploader("Document PDF", type=["pdf"])

    if file_up is not None:
        if st.session_state.get("uploaded_filename") != file_up.name:
            with st.spinner("Indexation temporaire..."):
                d_up = core.process_uploaded_pdf(file_up.getvalue(), file_up.name)
            st.session_state.uploaded_docs = d_up
            st.session_state.uploaded_filename = file_up.name

        docs_session = st.session_state.uploaded_docs
        sections = core.get_sections_in_docs(docs_session)
        st.markdown(
            f'<div class="metric-terminal-pill">{len(docs_session)} CHUNKS EXTRAITS // {len(sections)} SECTION(S)</div>',
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)

        action = st.radio("Action :", ["Recherche", "Synthèse", "Fiche mémo"], horizontal=True)

        if action == "Recherche":
            q_pdf = st.text_input("Requête sur ce document")
            if st.button("Lancer la recherche", type="primary") and q_pdf:
                with st.spinner("Analyse du document..."):
                    res = core.answer_question_full_book_structured(
                        llm,
                        core.filter_whole_document(docs_session),
                        q_pdf,
                    )
                show_structured_result(res, export_title=f"Resultat_{file_up.name}")

        elif action == "Synthèse":
            if st.button("Générer la synthèse", type="primary"):
                with st.spinner("Synthèse en cours..."):
                    res = core.summarize_documents_structured(
                        llm,
                        core.filter_whole_document(docs_session),
                        "Document vide.",
                        core.CHAPTER_SUMMARY_PROMPT,
                        core.COMBINE_SUMMARIES_PROMPT,
                    )
                show_structured_result(res, export_title=f"Synthese_{file_up.name}")

        elif action == "Fiche mémo":
            if st.button("Générer la fiche mémo", type="primary"):
                with st.spinner("Extraction..."):
                    res = core.summarize_documents_structured(
                        llm,
                        core.filter_whole_document(docs_session),
                        "Document vide.",
                        core.FICHE_EXTRACT_PROMPT,
                        core.COMBINE_FICHE_SECTION_PROMPT,
                    )
                show_structured_result(res, export_title=f"Fiche_{file_up.name}")