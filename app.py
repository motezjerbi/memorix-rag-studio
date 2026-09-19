"""
app.py
------
Interface cockpit pour MEMORIX : Studio RAG d'études et révisions.
Design technique épuré, typographie Space Grotesk/Inter, vitrine Data Science immersive.
"""

import base64
import pathlib
import time
import streamlit as st
import streamlit.components.v1 as components
from langchain_core.prompts import PromptTemplate

import export_utils
import knowledge_graph as kg
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
    """Lit et met en cache la feuille de style pour éviter les accès disque récurrents."""
    css_path = pathlib.Path(__file__).parent / "assets" / "style.css"
    return css_path.read_text(encoding="utf-8") if css_path.exists() else ""


@st.cache_data(show_spinner=False)
def get_cached_logo_b64() -> str | None:
    """Encode le logo en mémoire tampon base64."""
    logo_path = pathlib.Path(__file__).parent / "assets" / "logo.png"
    return base64.b64encode(logo_path.read_bytes()).decode() if logo_path.exists() else None


def load_css():
    """Injecte les règles CSS préchargées."""
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
# Landing Page : Affiche Data Science & Badges Thématiques
# ---------------------------------------------------------------------------

def render_landing_page():
    """Vitrine d'accueil orientée réussite académique et ingénierie Data Science."""
    encoded_logo = get_cached_logo_b64()

    # 1. Logo centré ou fallback
    if encoded_logo:
        logo_html = f'<div class="landing-logo-container"><img src="data:image/png;base64,{encoded_logo}" class="landing-logo-img" alt="MEMORIX Logo"/></div>'
    else:
        logo_html = '<div class="landing-logo-container"><h1 class="landing-brand-text">MEMORIX</h1><p class="landing-brand-sub">STUDIO RAG // DATA SCIENCE & IA</p></div>'
    st.markdown(logo_html, unsafe_allow_html=True)

    # 2. Hero Poster rendu en un seul bloc HTML sans rupture de balises
    poster_html = (
        '<div class="landing-hero-poster">'
        '<div class="poster-kicker">ENGINEERING & RESEARCH // DATA SCIENCE CORE</div>'
        '<h1 class="poster-title">Propulsez vos Révisions vers l\'Excellence</h1>'
        '<p class="poster-body">'
        'Maîtrisez les architectures complexes, les modèles statistiques et les pipelines de production. '
        'Une infrastructure RAG locale dédiée à la réussite des examens, soutenances techniques et entretiens de haut niveau.'
        '</p>'
        '<div class="poster-grid-stats">'
        '<div class="poster-stat-item"><div class="stat-number">100%</div><div class="stat-desc">Indexation locale & confidentielle</div></div>'
        '<div class="poster-stat-item"><div class="stat-number">&lt; 100ms</div><div class="stat-desc">Retrieval vectoriel instantané</div></div>'
        '<div class="poster-stat-item"><div class="stat-number">Spaced</div><div class="stat-desc">Répétition active & simulations orales</div></div>'
        '</div>'
        '</div>'
    )
    st.markdown(poster_html, unsafe_allow_html=True)

    # 3. Badges thématiques Data Science cliquables
    st.markdown(
        '<div class="domain-pills-label">SÉLECTIONNER UN AXE PRIORITAIRE DE TRAVAIL</div>',
        unsafe_allow_html=True,
    )

    b1, b2, b3, b4 = st.columns(4)
    with b1:
        if st.button("👁️ Computer Vision", key="pill_cv", use_container_width=True):
            st.session_state.initial_prefill_query = (
                "Détaille les principes des convolutions, des Vision Transformers "
                "et des loss functions de segmentation abordés dans les documents."
            )
            st.session_state.app_started = True
            st.rerun()
    with b2:
        if st.button("💬 NLP & LLMs", key="pill_nlp", use_container_width=True):
            st.session_state.initial_prefill_query = (
                "Présente le formalisme de l'attention multi-têtes, la tokenisation "
                "et les méthodes de fine-tuning par adaptation de bas rang (LoRA)."
            )
            st.session_state.app_started = True
            st.rerun()
    with b3:
        if st.button("⚙️ MLOps & Pipelines", key="pill_mlops", use_container_width=True):
            st.session_state.initial_prefill_query = (
                "Structure les étapes d'un pipeline MLOps : ingestion, validation des données, "
                "registre de conteneurs et politiques de déploiement continu."
            )
            st.session_state.app_started = True
            st.rerun()
    with b4:
        if st.button("📐 Stats & Maths", key="pill_stats", use_container_width=True):
            st.session_state.initial_prefill_query = (
                "Établis la fiche de synthèse des critères d'optimisation (descente de gradient, "
                "régularisations L1/L2) et des distributions statistiques utiles."
            )
            st.session_state.app_started = True
            st.rerun()

    st.markdown("<div style='height: 1.4rem;'></div>", unsafe_allow_html=True)

    # 4. Actions principales d'accès
    col_gauche, col_enter, col_google, col_droite = st.columns([1.2, 2.2, 2.2, 1.2])
    with col_enter:
        if st.button("Accéder au Studio RAG →", type="primary", use_container_width=True):
            st.session_state.app_started = True
            st.rerun()

    with col_google:
        if st.button("Se connecter avec Google", key="btn_google_login", use_container_width=True):
            st.info("Passerelle Google OAuth 2.0 active sur le serveur.")


# Contrôle d'affichage de la landing page
if not st.session_state.app_started:
    render_landing_page()
    st.stop()


# ---------------------------------------------------------------------------
# Composants Cockpit
# ---------------------------------------------------------------------------

def render_brand():
    """Affiche le logo MEMORIX en barre latérale."""
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
    """Construit le graphe et sa disposition une seule fois par combinaison de réglages."""
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
    "Recherche augmentée",
    "Synthèse générale",
    "Fiche mémo technique",
    "Atelier Code & Diagnostics",
    "Validation QCM",
    "Simulation orale",
    "Graphe de connaissances",
    "Métriques & Suivi",
    "Plan de révision",
    "Session PDF éphémère",
]
mode = st.sidebar.radio("Navigation", nav_options)

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
# Mode 1 : Recherche augmentée
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

    for turn in st.session_state.chat_history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn["role"] == "assistant":
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
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Recherche vectorielle et inférence technique..."):
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