"""
Insyte — Document Intelligence Platform (Research Prototype)
============================================================
Run:  streamlit run appv2.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from collections import defaultdict

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ─── Page config — must be the very first Streamlit call ─────────────────────
st.set_page_config(page_title="Insyte", layout="wide")

# ─── Password gate ────────────────────────────────────────────────────────────
from auth import check_password  # noqa: E402

if not check_password():
    st.stop()

# ─── Brand CSS ────────────────────────────────────────────────────────────────
# Palette: deep midnight plum · warm cream · bronze/gold
# #0C0820  background          #F5EFE0  cream text
# #160F2E  sidebar             #C4B89A  muted cream
# #1C1438  card surface        #C9A84C  gold accent
# #35265A  border              #D9BF6E  gold hover

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Cormorant+Garamond:wght@400;600&display=swap');

    /* ── Typography ── */
    html, body, [class*="css"], p, li, span, div {
        font-family: 'DM Sans', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'Cormorant Garamond', serif !important;
        letter-spacing: 0.02em;
    }

    /* ── App canvas ── */
    .stApp { background-color: #0C0820; }
    .main  { background-color: #0C0820; }
    .main .block-container {
        background-color: #0C0820;
        padding-top: 1.5rem;
        max-width: 1100px;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div:first-child {
        background-color: #160F2E !important;
        border-right: 1px solid #35265A;
    }

    /* ── Global text ── */
    .stMarkdown p, label, .stSelectbox label, [data-testid="stWidgetLabel"] p {
        color: #F5EFE0;
    }
    .stCaption p, [data-testid="stCaptionContainer"] p {
        color: #C4B89A !important;
    }

    /* ── Bordered containers / cards ── */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #1C1438 !important;
        border: 1px solid #35265A !important;
        border-radius: 8px !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background-color: #1C1438;
        color: #F5EFE0;
        border: 1px solid #35265A;
        border-radius: 6px;
        font-family: 'DM Sans', sans-serif;
        transition: background 0.15s, border-color 0.15s;
    }
    .stButton > button:hover {
        background-color: rgba(201,168,76,0.12);
        border-color: #C9A84C;
        color: #F5EFE0;
    }
    .stButton > button:focus {
        border-color: #C9A84C;
        box-shadow: 0 0 0 2px rgba(201,168,76,0.22);
    }

    /* ── Primary button (Analyse language) ── */
    .stButton > button[kind="primary"] {
        background-color: #C9A84C;
        color: #0C0820;
        border: none;
        font-weight: 600;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #D9BF6E;
        color: #0C0820;
    }

    /* ── Sidebar nav radio ── */
    [data-testid="stSidebar"] .stRadio > label { display: none; }
    [data-testid="stSidebar"] [data-testid="stRadio"] > div {
        display: flex;
        flex-direction: column;
        gap: 2px;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        display: flex;
        align-items: center;
        padding: 8px 12px 8px 14px;
        border-radius: 4px;
        color: #C4B89A;
        font-size: 0.88rem;
        cursor: pointer;
        border-left: 3px solid transparent;
        transition: background 0.15s, color 0.15s;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background: rgba(201,168,76,0.07);
        color: #F5EFE0;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] input[type="radio"] {
        display: none;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
        background: rgba(201,168,76,0.10);
        border-left: 3px solid #C9A84C;
        color: #F5EFE0;
        font-weight: 500;
    }

    /* ── Selectbox ── */
    .stSelectbox > div > div {
        background-color: #1C1438;
        border: 1px solid #35265A;
        color: #F5EFE0;
        border-radius: 6px;
    }
    .stSelectbox > div > div:focus-within {
        border-color: #C9A84C;
        box-shadow: 0 0 0 2px rgba(201,168,76,0.2);
    }

    /* ── File uploader ── */
    [data-testid="stFileUploader"] section {
        background-color: #1C1438;
        border: 1px dashed #35265A;
        border-radius: 8px;
    }
    [data-testid="stFileUploader"] section:hover {
        border-color: #C9A84C;
    }

    /* ── Text input ── */
    .stTextInput > div > div > input {
        background-color: #1C1438;
        border: 1px solid #35265A;
        color: #F5EFE0;
        border-radius: 6px;
    }
    .stTextInput > div > div > input:focus {
        border-color: #C9A84C;
        box-shadow: 0 0 0 2px rgba(201,168,76,0.2);
    }
    .stTextInput > div > div > input::placeholder {
        color: #C4B89A;
    }

    /* ── Expanders ── */
    [data-testid="stExpander"] {
        border: 1px solid #35265A !important;
        background-color: #160F2E !important;
        border-radius: 6px !important;
    }
    [data-testid="stExpander"] summary { color: #C4B89A; }
    [data-testid="stExpander"] summary:hover { color: #F5EFE0; }

    /* ── Progress bars ── */
    .stProgress > div > div > div { background-color: #C9A84C; }

    /* ── Dividers ── */
    hr { border-color: #35265A !important; }

    /* ── Alert colours ── */
    div[data-testid="stAlert"] { border-radius: 6px; }
    div[data-testid="stAlert"][data-baseweb="notification"] > div { border-radius: 6px; }

    /* ── Insyte wordmark ── */
    .insyte-wordmark {
        font-family: 'Cormorant Garamond', serif !important;
        font-size: 1.4rem;
        font-weight: 600;
        color: #F5EFE0 !important;
        letter-spacing: 0.04em;
        margin: 4px 0 2px 0;
        line-height: 1.3;
    }
    .insyte-tagline {
        font-size: 0.68rem;
        color: #C9A84C !important;
        letter-spacing: 0.10em;
        text-transform: uppercase;
        margin: 0 0 4px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Model loading ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading the tools...")
def load_embedding_model():
    import numpy as np
    from fastembed import TextEmbedding

    class _EmbedWrapper:
        """Drop-in replacement for SentenceTransformer using fastembed + ONNX.
        Uses ~80 MB RAM versus ~400 MB for torch-backed sentence-transformers."""
        def __init__(self):
            self._model = TextEmbedding(
                model_name="BAAI/bge-small-en-v1.5",
                cache_dir="./.model_cache",
            )

        def encode(self, texts, convert_to_numpy=True,
                   batch_size=64, show_progress_bar=False):
            return np.array(list(self._model.embed(texts)))

    return _EmbedWrapper()


# ─── Pipeline imports ─────────────────────────────────────────────────────────
from pipeline import (  # noqa: E402
    extract_documents,
    summarize_documents,
    contextual_search,
    exact_search,
    analyze_language,
    pattern_search,
    scan_risks,
)

# ─── UI helpers ───────────────────────────────────────────────────────────────

SECTION_ICONS = {
    "Presenting Concerns":        "›",
    "Subject's Account":          "›",
    "Risk Indicators":            "▲",
    "Assessments & Observations": "›",
    "Stressors & Context":        "›",
    "Actions & Plans":            "›",
    "Notable Absences":           "–",
    "Clinical Notes":             "›",
    "Client Overview":            "›",
}


def _source_tag(filename: str, page_num: int) -> str:
    return f"**{filename}** — p. {page_num}"



def _render_risk_banner(risk_results: list[dict]) -> None:
    """
    Persistent risk banner — only rendered when risk_results is non-empty.
    Groups flags by severity so clinicians see context, not just alarms.
    """
    acute     = [r for r in risk_results if r["severity"] == "acute"]
    unassessed = [r for r in risk_results if r["severity"] == "unassessed"]
    assessed  = [r for r in risk_results if r["severity"] == "assessed"]

    if acute:
        st.error(
            f"**High-priority risk language detected** — {len(acute)} passage(s) contain "
            "active or unresolved indicators with no safety plan noted. Immediate review recommended."
        )
    if unassessed:
        st.warning(
            f"**Risk language detected — no clinical assessment found** in uploaded documents "
            f"({len(unassessed)} passage(s)). Closer review recommended."
        )
    if assessed and not acute and not unassessed:
        st.info(
            f"**Risk language present — clinician assessment on file** ({len(assessed)} passage(s)). "
            "Review for context."
        )
    elif assessed:
        st.info(
            f"Additionally, {len(assessed)} passage(s) contain risk language with a "
            "clinician assessment documented."
        )


def _render_risk_detail(risk_results: list[dict]) -> None:
    if not risk_results:
        st.success("No risk language detected in the uploaded documents.")
        return

    order = ["acute", "unassessed", "assessed"]
    labels = {
        "acute":     "Acute / Unresolved",
        "unassessed": "Unassessed",
        "assessed":  "Assessed by Clinician",
    }
    for sev in order:
        items = [r for r in risk_results if r["severity"] == sev]
        if not items:
            continue
        st.markdown(f"### {labels[sev]}")
        for item in items:
            with st.container(border=True):
                st.markdown(f"> {item['sentence']}")
                c1, c2, c3 = st.columns([3, 2, 2])
                c1.caption(_source_tag(item["filename"], item["page_num"]))
                c2.caption(f"Match: **{item['match_type']}**")
                c3.caption(f"Phrase: *{item['matched_phrase']}*")


_CONFIDENCE_COLOURS = {
    "Strong":   ("#0A1F10", "#6ee7a0"),
    "Moderate": ("#2A1E00", "#C9A84C"),
    "Weak":     ("#2A1010", "#e09090"),
    "Exact":    ("#0F1E38", "#8ab8e0"),
}


def _confidence_chip_html(label: str) -> str:
    bg, fg = _CONFIDENCE_COLOURS.get(label, ("#1A1035", "#A89BC2"))
    return (
        f'<span style="background:{bg};color:{fg};padding:1px 8px;'
        f'border-radius:10px;font-size:0.75rem;font-weight:600;">'
        f"{label}</span>"
    )


_INTENSITY_STYLE = {
    "absent":   ("#1C1438", "#C4B89A"),
    "mild":     ("#2A1E00", "#C9A84C"),
    "moderate": ("#0F1E38", "#8ab8e0"),
    "strong":   ("#1A0D30", "#D9BF6E"),
}

_INTENSITY_LABEL = {
    "absent":   "ABSENT",
    "mild":     "MILD",
    "moderate": "MODERATE",
    "strong":   "STRONG",
}


def _intensity_badge(intensity: str) -> str:
    bg, fg = _INTENSITY_STYLE.get(intensity, ("#1A1035", "#A89BC2"))
    label = _INTENSITY_LABEL.get(intensity, intensity.upper())
    return (
        f'<span style="background:{bg};color:{fg};padding:3px 14px;'
        f'border-radius:10px;font-size:0.8rem;font-weight:700;letter-spacing:0.06em;">'
        f"{label}</span>"
    )


_QUICK_LANG_QUERIES = [
    "Depressive language",
    "Anxiety indicators",
    "Trauma indicators",
    "Treatment satisfaction",
    "Functional capacity",
    "Social isolation",
    "Anger or hostility",
    "Substance use",
]


def _render_context_match(match: dict) -> None:
    """Render a single search result with surrounding context (no confidence chip)."""
    with st.container(border=True):
        if match.get("context_before"):
            st.caption(f"…{match['context_before']}")
        st.markdown(match["sentence"])
        if match.get("context_after"):
            st.caption(f"…{match['context_after']}")
        cols = st.columns([4, 1])
        cols[0].caption(f"p. {match['page_num']}")
        if match.get("score") is not None and match["score"] < 1.0:
            cols[1].caption(f"Score: {match['score']:.2f}")


# ─── Summary card helpers ─────────────────────────────────────────────────────

_PROSE_SECTION_PRIORITY = [
    "Risk Indicators",
    "Presenting Concerns",
    "Subject's Account",
    "Assessments & Observations",
    "Stressors & Context",
    "Actions & Plans",
    "Notable Absences",
    # sumy fallback names
    "Clinical Notes",
    "Client Overview",
]

_TAG_RULES: list[tuple[str, str, list[str] | None]] = [
    ("Risk language present",       "risk",     None),
    ("Suicidal ideation noted",     "risk",     ["suicid", "ideation", "self-harm", "self harm"]),
    ("Safety concern present",      "risk",     ["safety concern", "risk assessment", "safety plan"]),
    ("Medication mentioned",        "clinical", ["medication", "prescribed", "dosage", "mg", "drug"]),
    ("Diagnosis referenced",        "clinical", ["diagnos", "disorder", "condition", "symptoms"]),
    ("Treatment goal identified",   "clinical", ["goal", "objective", "working toward", "treatment plan"]),
    ("Therapy or referral noted",   "clinical", ["therapy", "counsell", "counsel", "referral", "session"]),
    ("Presenting concerns noted",   "clinical", None),
    ("Housing stressor noted",      "stressor", ["housing", "homeless", "shelter", "evict"]),
    ("Financial stressor noted",    "stressor", ["financial", "money", "debt", "income", "unemploy"]),
    ("Social isolation noted",      "stressor", ["isolat", "alone", "withdraw", "no support"]),
    ("Family conflict noted",       "stressor", ["family conflict", "domestic", "relationship", "abuse"]),
    ("Substance use noted",         "stressor", ["substance", "alcohol", "drug use", "cannabis", "opioid"]),
]

_CHIP_STYLES = {
    "risk":     ("#2A1010",          "#e09090"),
    "clinical": ("#0F1E38",          "#8ab8e0"),
    "stressor": ("rgba(42,30,0,0.9)", "#C9A84C"),
}


def _build_prose_summary(sections: dict) -> str:
    chosen: list[str] = []
    for section_name in _PROSE_SECTION_PRIORITY:
        items = sections.get(section_name, [])
        if items and len(chosen) < 3:
            sentence = items[0]["sentence"]
            # Truncate very long verbatim sentences (sumy fallback) for card readability
            if len(sentence) > 220:
                sentence = sentence[:217].rstrip() + "…"
            chosen.append(sentence)
    return "  ".join(chosen)


def _detect_tags(doc_summary: dict, risk_results: list[dict]) -> list[tuple[str, str]]:
    sections = doc_summary.get("sections", {})
    filename = doc_summary["filename"]
    all_text = " ".join(
        item["sentence"].lower()
        for items in sections.values()
        for item in items
    )
    found: list[tuple[str, str]] = []
    for label, chip_type, keywords in _TAG_RULES:
        if len(found) >= 3:
            break
        if label == "Risk language present":
            if any(r["filename"] == filename for r in risk_results):
                found.append((label, chip_type))
        elif label == "Presenting concerns noted":
            if sections.get("Presenting Concerns") or sections.get("Subject's Account"):
                found.append((label, chip_type))
        elif keywords and any(kw in all_text for kw in keywords):
            found.append((label, chip_type))
    return found[:3]


def _chip_html(label: str, chip_type: str) -> str:
    bg, fg = _CHIP_STYLES.get(chip_type, ("#1A1035", "#A89BC2"))
    return (
        f'<span style="background:{bg};color:{fg};padding:3px 10px;'
        f'border-radius:12px;font-size:0.75rem;font-weight:600;'
        f'margin-right:6px;white-space:nowrap;">'
        f"{label}</span>"
    )


def _render_summary_card(doc_summary: dict, risk_results: list[dict]) -> None:
    sections = doc_summary.get("sections", {})
    insufficient = doc_summary.get("insufficient_clinical_content", False)
    llm_used  = doc_summary.get("llm_used", False)
    model_used = doc_summary.get("model_used")

    with st.container(border=True):
        left, right = st.columns([1, 2], gap="large")

        with left:
            st.markdown(f"**{doc_summary['filename']}**")
            st.caption(doc_summary["doc_type"])
            if doc_summary.get("client_name"):
                st.caption(f"Client: {doc_summary['client_name']}")
            if doc_summary.get("date"):
                st.caption(doc_summary["date"])
            # LLM / fallback badge
            if llm_used and model_used:
                st.caption(f"AI · `{model_used}`")

        with right:
            if insufficient:
                st.warning(
                    "This document appears to contain primarily administrative content. "
                    "No clinical summary generated."
                )
            elif not sections:
                st.caption("No content could be extracted from this document.")
            else:
                prose = _build_prose_summary(sections)
                if prose:
                    st.markdown(prose)

                tags = _detect_tags(doc_summary, risk_results)
                if tags:
                    chips_html = "".join(_chip_html(label, ctype) for label, ctype in tags)
                    st.markdown(
                        f'<div style="margin-top:8px;">{chips_html}</div>',
                        unsafe_allow_html=True,
                    )

                # Show all sections returned by the synthesizer (LLM or sumy)
                all_section_names = [s for s in _PROSE_SECTION_PRIORITY if s in sections]
                # also include any section names not in the priority list (future-proof)
                for s in sections:
                    if s not in all_section_names:
                        all_section_names.append(s)

                with st.expander("View source passages"):
                    for section_name in all_section_names:
                        items = sections.get(section_name, [])
                        if not items:
                            continue
                        icon = SECTION_ICONS.get(section_name, "•")
                        st.markdown(f"**{icon} {section_name}**")
                        for item in items:
                            st.markdown(
                                f'<div style="border-left:2px solid #C9A84C;'
                                f'padding:4px 10px;margin:4px 0;'
                                f'font-size:0.9rem;color:#E8D9BA;">'
                                f'{item["sentence"]}'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                            st.caption(f"p. {item['page_num']}")


def _render_pattern_match(match: dict) -> None:
    confidence = match.get("match_confidence", "")
    with st.container(border=True):
        if match.get("context_before"):
            st.caption(f"…{match['context_before']}")
        st.markdown(match["sentence"])
        if match.get("context_after"):
            st.caption(f"…{match['context_after']}")
        cols = st.columns([3, 2, 1])
        cols[0].caption(f"p. {match['page_num']}")
        if confidence:
            cols[1].markdown(_confidence_chip_html(confidence), unsafe_allow_html=True)
        if match.get("score") is not None:
            cols[2].caption(f"{match['score']:.2f}")


# ─── Sidebar: header + file upload ───────────────────────────────────────────

with st.sidebar:
    st.markdown(
        '<p class="insyte-wordmark">Insyte</p>'
        '<p class="insyte-tagline">Document Intelligence</p>',
        unsafe_allow_html=True,
    )

    st.divider()
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Typed PDFs only. Scanned/image PDFs are not supported.",
    )

# ─── Disclaimer (always visible in main area) ─────────────────────────────────

st.warning(
    "Research prototype — submit synthetic or anonymised documents only. "
    "Not for use with real client data."
)

if not uploaded_files:
    st.markdown(" ")
    st.info("Upload one or more PDFs using the sidebar to begin.")
    st.stop()

# ─── Processing (cached by file set) ─────────────────────────────────────────

_upload_key = tuple(sorted(f.name for f in uploaded_files))

if st.session_state.get("_upload_key") != _upload_key:
    st.session_state["_upload_key"] = _upload_key
    for key in ("docs", "synthesis", "risk_results"):
        st.session_state.pop(key, None)

if "docs" not in st.session_state:
    embed_model = load_embedding_model()

    with st.spinner("Extracting text from PDFs…"):
        docs, errors = extract_documents(uploaded_files)
        st.session_state["docs"] = docs
        for err in errors:
            st.error(err)
        if not docs:
            st.error("No text could be extracted from the uploaded files.")
            st.stop()

    with st.spinner("Generating intelligence summary…"):
        try:
            synthesis = summarize_documents(docs)
        except Exception as exc:  # noqa: BLE001
            synthesis = []
            st.warning(f"Summarization failed: {exc}")
        st.session_state["synthesis"] = synthesis

    with st.spinner("Scanning for risk language…"):
        st.session_state["risk_results"] = scan_risks(docs, embed_model)

docs: list[dict]       = st.session_state["docs"]
synthesis: list[dict]  = st.session_state["synthesis"]
risk_results: list[dict] = st.session_state["risk_results"]

# ─── Sidebar: document selector + navigation ─────────────────────────────────

with st.sidebar:
    # Document filter
    all_filenames = [d["filename"] for d in docs]
    if len(all_filenames) > 1:
        doc_choice = st.selectbox(
            "Document",
            ["All documents"] + all_filenames,
            key="doc_filter",
        )
    else:
        doc_choice = "All documents"
        st.caption(all_filenames[0])

    st.divider()

    # Navigation
    nav_selection = st.radio(
        "Navigation",
        ["Summary", "Keyword Search", "Language Analysis", "Pattern Search", "Risk Flags"],
        key="nav_radio",
        label_visibility="collapsed",
    )

    # Risk indicator badge (only if flags exist)
    if risk_results:
        st.divider()
        acute_count = sum(1 for r in risk_results if r["severity"] == "acute")
        unassessed_count = sum(1 for r in risk_results if r["severity"] == "unassessed")
        if acute_count:
            st.error(f"High priority — {acute_count} risk flag(s)")
        elif unassessed_count:
            st.warning(f"Review needed — {unassessed_count} flag(s)")
        else:
            st.info(f"{len(risk_results)} assessed flag(s)")

# ─── Apply document filter ────────────────────────────────────────────────────

if doc_choice == "All documents":
    active_docs    = docs
    active_synth   = synthesis
    active_risks   = risk_results
else:
    active_docs  = [d for d in docs if d["filename"] == doc_choice]
    active_synth = [s for s in synthesis if s["filename"] == doc_choice]
    active_risks = [r for r in risk_results if r["filename"] == doc_choice]

# Map nav label → panel key
_NAV_MAP = {
    "Summary":           "summary",
    "Keyword Search":    "keyword",
    "Language Analysis": "sentiment",
    "Pattern Search":    "patterns",
    "Risk Flags":        "risk",
}
active = _NAV_MAP.get(nav_selection, "summary")

# ─── Risk banner (only when flags exist for current doc filter) ───────────────

if active_risks:
    _render_risk_banner(active_risks)

# ─── Panel: Summary ───────────────────────────────────────────────────────────

if active == "summary":
    st.subheader("Intelligence Summary")
    st.caption(
        f"{len(active_docs)} document(s) — "
        f"{sum(len(d['pages']) for d in active_docs)} pages total"
    )
    if active_synth:
        for doc_summary in active_synth:
            _render_summary_card(doc_summary, active_risks)
    else:
        st.info("No summary sentences could be extracted.")

# ─── Panel: Keyword Search ────────────────────────────────────────────────────

elif active == "keyword":
    st.subheader("Keyword Search")
    query = st.text_input(
        "Search across documents",
        placeholder="e.g. housing instability, medication, family conflict…",
    )

    if query:
        embed_model = load_embedding_model()
        with st.spinner("Searching…"):
            results = contextual_search(query, active_docs, embed_model, top_n=12)
            fallback_used = False
            if not results:
                results = exact_search(query, active_docs)
                fallback_used = True

        if fallback_used:
            st.caption("No strong semantic matches — showing exact matches instead.")

        if results:
            grouped: dict[str, list[dict]] = defaultdict(list)
            for r in results:
                grouped[r["filename"]].append(r)
            for filename, matches in grouped.items():
                st.markdown(f"**{filename}** — {len(matches)} match(es)")
                for match in matches:
                    _render_context_match(match)
        else:
            st.info("No matches found.")

# ─── Panel: Linguistic Analysis ───────────────────────────────────────────────

elif active == "sentiment":
    st.subheader("Linguistic Analysis")
    st.caption(
        "Describe the type of language you want to find. "
        "The tool retrieves relevant passages and characterises their presence. "
        "**Not a clinical assessment.**"
    )

    # ── Quick-select presets ──
    st.caption("Quick queries:")
    chip_cols = st.columns(4)
    for _i, _qq in enumerate(_QUICK_LANG_QUERIES):
        if chip_cols[_i % 4].button(_qq, key=f"langchip_{_i}", use_container_width=True):
            st.session_state["lang_preset"] = _qq
            st.session_state["lang_query_input"] = _qq

    # ── Query input ──
    _preset = st.session_state.get("lang_preset", "")
    lang_query = st.text_input(
        "What type of language are you looking for?",
        value=_preset,
        placeholder=(
            "e.g. depressive language  ·  satisfaction with treatment  ·  "
            "trauma indicators  ·  functional capacity"
        ),
        key="lang_query_input",
    )
    # Keep preset in sync so next rerun preserves the typed value
    st.session_state["lang_preset"] = lang_query

    # ── Document scope ──
    scope_choice = st.radio(
        "Analyse",
        ["All documents", "One document"],
        horizontal=True,
        key="lang_scope",
    )
    if scope_choice == "One document":
        scope_docs = [
            d for d in active_docs
            if d["filename"] == st.selectbox(
                "Document",
                [d["filename"] for d in active_docs],
                key="lang_doc_select",
            )
        ]
    else:
        scope_docs = active_docs

    if st.button(
        "Analyse language",
        disabled=not bool(lang_query.strip()),
        type="primary",
    ):
        embed_model = load_embedding_model()
        results: list[dict] = []
        prog = st.progress(0, text="Analysing…")
        for _n, _doc in enumerate(scope_docs):
            prog.progress(
                (_n + 1) / len(scope_docs),
                text=f"Analysing {_doc['filename']}…",
            )
            results.append(analyze_language(lang_query.strip(), _doc, embed_model))
        prog.empty()
        st.session_state["lang_results"] = results
        st.session_state["lang_results_query"] = lang_query.strip()

    # ── Display results ──
    results = st.session_state.get("lang_results", [])
    last_query = st.session_state.get("lang_results_query", "")

    if results and last_query:
        st.markdown("---")
        st.caption(f'Results for: *"{last_query}"*')

        for res in results:
            with st.container(border=True):
                # Header row: filename + intensity badge
                h_left, h_right = st.columns([3, 1])
                h_left.markdown(f"**{res['filename']}**")
                h_right.markdown(
                    _intensity_badge(res.get("intensity", "absent")),
                    unsafe_allow_html=True,
                )

                # Verdict
                st.markdown(
                    f'<p style="color:#E8E0F5;font-size:0.95rem;margin:6px 0 10px 0;">'
                    f'{res["verdict"]}</p>',
                    unsafe_allow_html=True,
                )

                # Evidence quotes
                evidence = res.get("evidence", [])
                if evidence and res.get("intensity", "absent") != "absent":
                    for ev in evidence:
                        quote = ev.get("quote", "").strip().strip('"')
                        note  = ev.get("note", "")
                        page  = ev.get("page_num", "?")
                        st.markdown(
                            f'<div style="border-left:3px solid #C9A84C;'
                            f'padding:8px 14px;margin:6px 0;'
                            f'background:#160F2E;border-radius:0 6px 6px 0;">'
                            f'<span style="color:#E8D9BA;font-style:italic;">'
                            f'&ldquo;{quote}&rdquo;</span>'
                            f'<br><span style="color:#C4B89A;font-size:0.78rem;">'
                            f'p.&nbsp;{page}&ensp;·&ensp;{note}</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                # Summary paragraph
                summary = res.get("summary", "")
                if summary and summary != res.get("verdict", ""):
                    with st.expander("Full analysis"):
                        st.markdown(
                            f'<p style="color:#E8E0F5;font-size:0.9rem;">{summary}</p>',
                            unsafe_allow_html=True,
                        )
                        if not res.get("llm_used"):
                            st.caption(
                                "Extractive mode — enable Ollama for AI-assisted analysis."
                            )

# ─── Panel: Pattern Search ────────────────────────────────────────────────────

elif active == "patterns":
    st.subheader("Pattern Search")
    st.caption(
        "Describe the type of information you want to find across all uploaded documents."
    )

    pattern_query = st.text_input(
        "What pattern are you looking for?",
        placeholder=(
            "e.g. Medications mentioned · Medical assessments completed · "
            "Stated stressors or life pressures · Referrals made or recommended · "
            "Risk language or safety concerns"
        ),
    )

    if st.button("Search for pattern", disabled=not bool(pattern_query)):
        embed_model = load_embedding_model()
        with st.spinner("Searching for pattern…"):
            result = pattern_search(pattern_query, active_docs, embed_model, top_n=20)

        groups        = result["groups"]
        expanded_query = result["expanded_query"]
        low_conf      = result["low_confidence_warning"]
        fallback_used = result["fallback_used"]

        if expanded_query.lower() != pattern_query.lower():
            st.caption(f"Query expanded to: *{expanded_query}*")

        if not groups:
            st.info("No relevant passages found across the uploaded documents.")
        else:
            if low_conf:
                st.warning(
                    "Low-confidence matches — results may not closely match your query. "
                    "Try rephrasing or using more specific terminology."
                )
            if fallback_used:
                st.info("No semantic matches found. Showing exact keyword matches instead.")
            total = sum(len(g["matches"]) for g in groups)
            st.caption(f"{total} passage(s) across {len(groups)} document(s)")
            for group in groups:
                st.markdown(f"**{group['filename']}** — {len(group['matches'])} match(es)")
                for match in group["matches"]:
                    _render_pattern_match(match)

# ─── Panel: Risk Flags ────────────────────────────────────────────────────────

elif active == "risk":
    st.subheader("Risk Flags — Detail View")
    _render_risk_detail(active_risks)
