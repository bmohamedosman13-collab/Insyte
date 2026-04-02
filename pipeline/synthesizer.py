"""
Document summarization using a local Ollama LLM.

Designed for diverse organisations:
  Mental health clinics       — psychiatric notes, intake forms, progress notes
  Housing / homelessness      — case notes, referral letters, housing assessments
  Fraud investigation         — audit records, case files, incident reports
  Youth services & charities  — programme records, safeguarding notes, plans

Architecture
------------
Primary:  Ollama LLM (any locally-installed model)
          - Reads the document, does not count words
          - Negation-aware: "client denies SI" ≠ "client has SI"
          - Voice-aware: subject's own account vs. professional's observations
          - Returns structured JSON; mapped back to source pages via keyword overlap

Fallback: centroid-based extractive summarizer (fastembed ONNX)
          - Activated automatically when Ollama is not running
          - Uses the BAAI/bge-small-en-v1.5 model already in memory
          - No additional dependencies beyond what the app already loads

Setup
-----
1. Install Ollama:  https://ollama.com
2. Pull a model:    ollama pull llama3.2
3. Start server:    ollama serve   (or it starts automatically on most installs)

Environment variables (optional):
  OLLAMA_URL    default: http://localhost:11434
  OLLAMA_MODEL  default: auto-detected from available models
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from difflib import SequenceMatcher

# ─── Section names ────────────────────────────────────────────────────────────
# Domain-agnostic: used across mental health, housing, fraud, youth services.

SECTION_ORDER = [
    "Presenting Concerns",
    "Subject's Account",
    "Risk Indicators",
    "Assessments & Observations",
    "Stressors & Context",
    "Actions & Plans",
    "Notable Absences",
]

# LLM JSON key → UI section name
_LLM_KEY_TO_SECTION: dict[str, str] = {
    "presenting_concerns":       "Presenting Concerns",
    "subject_account":           "Subject's Account",
    "risk_and_safety":           "Risk Indicators",
    "professional_observations": "Assessments & Observations",
    "context_and_stressors":     "Stressors & Context",
    "actions_and_plans":         "Actions & Plans",
    "notable_absences":          "Notable Absences",
}

# ─── Document type detection ──────────────────────────────────────────────────

_DOC_TYPE_MAP: list[tuple[list[str], str]] = [
    (["intake", "intake form", "initial assessment", "registration", "onboarding"],
     "Intake Form"),
    (["progress note", "session note", "session record", "progress"],
     "Progress Note"),
    (["survey", "questionnaire", "screening", "self-report", "self report"],
     "Survey / Questionnaire"),
    (["referral", "referral letter", "refer"],
     "Referral"),
    (["discharge", "closing", "closure", "termination"],
     "Discharge Summary"),
    (["assessment", "evaluation", "psychological", "psychosocial"],
     "Clinical Assessment"),
    (["case note", "case file", "case record", "case summary"],
     "Case Note"),
    (["incident report", "incident"],
     "Incident Report"),
    (["audit", "fraud", "investigation", "review", "compliance"],
     "Audit / Investigation Record"),
    (["housing", "shelter", "residential"],
     "Housing Record"),
    (["safeguarding", "protection", "risk assessment"],
     "Safeguarding Record"),
]


def _detect_doc_type(doc: dict) -> str:
    probe = doc["filename"].lower() + " " + doc["full_text"][:600].lower()
    for keywords, label in _DOC_TYPE_MAP:
        if any(kw in probe for kw in keywords):
            return label
    return "Document"


def _detect_client_name(doc: dict) -> str | None:
    # Name word: "Smith" or hyphenated "Osei-Mensah"
    _NW = r"[A-Z][a-z]+(?:-[A-Z][a-z]+)?"
    # Name part: full word OR middle initial like "T."
    _NP = rf"(?:{_NW}|[A-Z]\.)"
    patterns = [
        # "Client: Darnell T. Okafor" — requires at least first + last word
        rf"(?:client|patient|participant|resident|youth|subject)\s*[:\-]\s*"
        rf"({_NW}(?:\s+{_NP})*\s+{_NW})",
        # "Name: Abena Osei-Mensah"
        rf"^Name\s*[:\-]\s*({_NW}(?:\s+{_NP})*\s+{_NW})",
    ]
    for pattern in patterns:
        m = re.search(pattern, doc["full_text"][:1500], re.MULTILINE)
        if m:
            candidate = m.group(1).strip()
            # Allow 2–5 parts (first [middle] last, or hyphenated combos)
            if 2 <= len(candidate.split()) <= 5:
                return candidate
    return None


def _detect_date(doc: dict) -> str | None:
    patterns = [
        r"\b((?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+\d{1,2},?\s+\d{4})\b",
        r"\b(\d{4}[\/\-]\d{2}[\/\-]\d{2})\b",
        r"\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})\b",
    ]
    for pattern in patterns:
        m = re.search(pattern, doc["full_text"][:800], re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None


# ─── Source attribution ───────────────────────────────────────────────────────

def _attribution_page(fragment: str, doc: dict) -> int:
    """
    Find the page most likely to contain this text fragment.

    Pass 1: exact substring match (fast, reliable for direct quotes).
    Pass 2: key-word overlap scoring (handles paraphrases and near-quotes).
    """
    frag_lower = fragment.lower().strip()

    # Pass 1 — exact
    for page in doc["pages"]:
        if frag_lower in page["text"].lower():
            return page["page_num"]

    # Pass 1b — first 60 chars
    frag_short = frag_lower[:60]
    if len(frag_short) > 15:
        for page in doc["pages"]:
            if frag_short in page["text"].lower():
                return page["page_num"]

    # Pass 2 — keyword overlap
    key_words = {w.lower() for w in re.findall(r"[a-zA-Z]{4,}", fragment)}
    if key_words:
        best_score, best_page = 0.0, doc["pages"][0]["page_num"]
        for page in doc["pages"]:
            page_words = {w.lower() for w in re.findall(r"[a-zA-Z]{4,}", page["text"])}
            if page_words:
                score = len(key_words & page_words) / len(key_words)
                if score > best_score:
                    best_score, best_page = score, page["page_num"]
        if best_score > 0.3:
            return best_page

    return doc["pages"][0]["page_num"] if doc["pages"] else 1


# ─── Ollama integration ───────────────────────────────────────────────────────

from .ollama_client import (
    ollama_available as _ollama_available,
    get_best_model as _get_best_model,
    call_ollama as _call_ollama,
    parse_llm_json as _parse_llm_json,
)




def _build_prompt(doc_text: str, doc_type: str) -> list[dict]:
    """
    Build the Ollama /api/chat messages list.

    Prompt design principles (grounded in clinical NLP literature):
    - Negation must be explicit: a denial is different from an absence
    - Voice must be separated: what the subject says ≠ what the professional records
    - No hallucination: only extract what is in the document
    - Domain-agnostic: works for mental health, housing, fraud, youth services
    - Direct quotes preferred for source attribution accuracy
    """
    # Truncate very long documents to fit model context
    # ~4 chars per token; target ~3500 tokens of document text
    max_chars = 14000
    truncated = False
    if len(doc_text) > max_chars:
        doc_text = doc_text[:max_chars]
        truncated = True

    system = (
        "You are a document analyst assisting social service and nonprofit professionals. "
        "You extract factual information from case documents, clinical notes, referral letters, "
        "intake forms, assessments, housing records, audit files, and programme records. "
        "You work across mental health services, housing and homelessness support, "
        "fraud and compliance investigations, youth services, and charities. "
        "You never diagnose, never assess clinical risk, and never add information "
        "that is not explicitly present in the document."
    )

    user = f"""Analyse the following {doc_type} and extract its substantive content.

CRITICAL RULES — read carefully before extracting:

1. ACCURACY: Extract only what the document explicitly states. Do not infer, speculate, or add context from outside the document.

2. NEGATION (most common error): When something is explicitly denied, ruled out, or stated as absent — for example "denies suicidal ideation", "no fixed address", "no prior criminal history", "client reports no current substance use" — record that denial as a separate item. A denial is clinically and legally distinct from the condition being present. Do NOT silently omit denials.

3. VOICE DISTINCTION: Separate (a) what the person/client/subject says about themselves from (b) what the professional, worker, or author records as their own observation or assessment. These carry different evidentiary weight.

4. OMIT BOILERPLATE: Skip administrative text, consent language, mailing addresses, phone numbers, checkbox fields, form labels, legal disclaimers, and repeated headers.

5. SYNTHESIZE: Write clear, readable summaries in plain language — do NOT copy sentences verbatim from the document. Paraphrase and synthesize the content into concise, easy-to-read statements that a practitioner can quickly scan.{"" if not truncated else chr(10) + "NOTE: This document was truncated due to length. Only the first portion has been analysed."}

Return ONLY a valid JSON object. Include only the keys where content exists — omit keys entirely when there is nothing to say. Do not include explanatory text outside the JSON. Each array should contain 1-3 SHORT, SYNTHESIZED summary statements (not direct quotes).

{{
  "presenting_concerns": [
    "A concise synthesized statement of the primary reason this person is being seen. What brought them here."
  ],
  "subject_account": [
    "A synthesized summary of what the person (client, resident, youth, subject) says about themselves — paraphrased clearly."
  ],
  "risk_and_safety": [
    "A concise synthesized statement of any risk or safety matters — whether confirmed, denied, or assessed (e.g. 'Client denies current suicidal ideation; safety plan in place')."
  ],
  "professional_observations": [
    "A synthesized summary of the worker or clinician's own assessment or professional judgement."
  ],
  "context_and_stressors": [
    "A synthesized summary of relevant life circumstances and social factors: housing, finances, relationships, employment, trauma."
  ],
  "actions_and_plans": [
    "A synthesized summary of referrals, goals, interventions, and next steps."
  ],
  "notable_absences": [
    "A concise note on anything the document explicitly rules out that would be significant if present."
  ]
}}

DOCUMENT ({doc_type}):
{doc_text}"""

    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]




def _llm_response_to_sections(parsed: dict, doc: dict) -> dict[str, list[dict]]:
    """
    Convert the parsed LLM JSON into the internal sections format.
    Each item becomes {sentence, page_num}.
    Page numbers are recovered via keyword-overlap attribution.
    """
    sections: dict[str, list[dict]] = {}

    for llm_key, section_name in _LLM_KEY_TO_SECTION.items():
        items = parsed.get(llm_key, [])
        if not items:
            continue
        # Accept either a list of strings or a list of dicts
        section_items = []
        for item in items:
            if isinstance(item, dict):
                text = item.get("text") or item.get("quote") or str(item)
            else:
                text = str(item)
            text = text.strip()
            if len(text) < 10:
                continue
            page_num = _attribution_page(text, doc)
            section_items.append({"sentence": text, "page_num": page_num})
        if section_items:
            sections[section_name] = section_items

    return sections


# ─── Fallback: centroid extractive summarizer ────────────────────────────────

_NOISE_PHRASES: list[str] = [
    "i consent", "i agree to", "by signing", "authorized to release",
    "protected under", "pipa", "personal information protection",
    "privacy act", "freedom of information", "confidentiality agreement",
    "release of information", "authorization form", "signature required",
    "terms and conditions", "all rights reserved", "© ", "copyright",
    "fax:", "toll-free", "website:", "www.", "office hours",
    "please contact us", "for more information", "thank you for",
    "if you have any questions", "please check", "check all that apply",
    "circle one", "select one", "leave blank", "for office use only",
    "do not write", "☐", "□", "[ ]",
]

_PHONE_RE  = re.compile(r"\(?\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4}")
_POSTAL_RE = re.compile(r"\b[A-Z]\d[A-Z]\s?\d[A-Z]\d\b")
_ZIP_RE    = re.compile(r"\b\d{5}(?:-\d{4})?\b")
_ADDR_RE   = re.compile(
    r"\b\d{1,5}\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+"
    r"(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Boulevard|Blvd|Lane|Ln|Way|Court|Ct)\b",
    re.IGNORECASE,
)
_CHECKBOX_RE   = re.compile(r"^\s*(?:yes|no|n\/a)\s*[\/|]\s*(?:yes|no|n\/a)\s*$", re.IGNORECASE)
_FIELDLABEL_RE = re.compile(r"^[A-Za-z /\(\)]{3,40}:\s*(?:_{2,}|\.{2,})?\s*$")

_VERB_INDICATORS = {
    "is", "are", "was", "were", "be", "been", "being",
    "has", "have", "had", "do", "does", "did",
    "feel", "feels", "felt", "report", "reports", "reported",
    "states", "stated", "describes", "described", "noted",
    "presents", "presented", "experiences", "expressed",
    "diagnosed", "prescribed", "referred", "completed",
}

_CLINICAL_SIGNALS: list[list[str]] = [
    ["feel", "feels", "feeling", "felt", "emotion", "mood", "distress",
     "anxious", "depressed", "hopeless", "overwhelmed", "afraid", "angry", "sad"],
    ["behav", "function", "daily", "sleep", "appetite", "concentrat",
     "memory", "withdraw", "isolat", "engage", "avoid", "difficul"],
    ["history of", "previous", "prior", "past", "experienced", "childhood",
     "trauma", "abuse", "loss", "grief"],
    ["concern", "worry", "presenting", "referred", "seeking", "complaint",
     "issue", "struggle", "challeng", "difficul", "problem"],
    ["risk", "harm", "suicid", "ideation", "safety", "crisis", "danger",
     "threat", "attempt", "plan", "intent", "fraud", "irregularity"],
    ["diagnos", "disorder", "condition", "symptom", "episode",
     "housing", "homeless", "shelter", "evict", "subsid"],
    ["medication", "prescribed", "dosage", "therapy", "counsell",
     "treatment", "intervention", "session", "referral", "discharge"],
    ["goal", "objective", "plan", "target", "aim", "intend",
     "wants to", "working toward", "next steps", "follow"],
]

_STRONG_SINGLES = [
    "suicid", "self-harm", "self harm", "ideation", "overdose",
    "diagnos", "prescribed", "medication", "hospitali", "psychiat",
    "homeless", "evict", "fraud", "irregularity", "safeguarding",
]


def _is_noise(sentence: str) -> bool:
    lower = sentence.lower().strip()
    if any(p in lower for p in _NOISE_PHRASES):
        return True
    if _PHONE_RE.search(sentence):
        return True
    if _POSTAL_RE.search(sentence) or _ZIP_RE.search(sentence):
        return True
    if _ADDR_RE.search(sentence):
        return True
    if _CHECKBOX_RE.match(sentence):
        return True
    if _FIELDLABEL_RE.match(sentence):
        return True
    words = sentence.split()
    if len(words) < 12:
        if sentence.strip().endswith(":"):
            return True
        if not any(w.lower() in _VERB_INDICATORS for w in words):
            return True
    return False


def _is_substantive(sentence: str) -> bool:
    lower = sentence.lower()
    if any(sig in lower for sig in _STRONG_SINGLES):
        return True
    matched = sum(1 for grp in _CLINICAL_SIGNALS if any(sig in lower for sig in grp))
    return matched >= 2


def _relevance_score(sentence: str) -> float:
    lower = sentence.lower()
    all_sigs = [w for grp in _CLINICAL_SIGNALS for w in grp] + _STRONG_SINGLES
    hits = sum(1 for sig in all_sigs if sig in lower)
    strong = sum(1 for sig in _STRONG_SINGLES if sig in lower)
    return (hits + strong * 2) * min(1.0, len(sentence.split()) / 20)


def _classify_fallback(sentence: str) -> str:
    lower = sentence.lower()
    scores: dict[str, int] = {
        "Presenting Concerns":       sum(1 for kw in [
            "presenting", "reason for referral", "chief complaint", "concern",
            "referred for", "reason for visit", "purpose of",
        ] if kw in lower),
        "Subject's Account":         sum(1 for kw in [
            "stated", "reported", "described", "expressed", "said",
            "told", "mentioned", "felt", "feel", "i feel", "i have",
            "i don't", "i can't", "i am", "i was", "i've", "my ",
            "client stated", "client reported", "client described",
            "patient stated", "patient reported", "resident reported",
        ] if kw in lower),
        "Risk Indicators":           sum(1 for kw in [
            "suicid", "self-harm", "ideation", "harm", "crisis", "safety",
            "risk", "hopeless", "danger", "threat", "fraud", "irregularity",
            "safeguarding",
        ] if kw in lower),
        "Assessments & Observations": sum(1 for kw in [
            "assessment", "diagnos", "clinical", "observation", "noted",
            "mental status", "therapy", "intervention", "recommendation",
            "housing assessment", "audit finding",
        ] if kw in lower),
        "Stressors & Context":       sum(1 for kw in [
            "stress", "financial", "housing", "relationship", "family",
            "work", "employment", "isolation", "trauma", "grief", "loss",
            "abuse", "conflict", "barrier", "substance",
        ] if kw in lower),
        "Actions & Plans":           sum(1 for kw in [
            "plan", "goal", "referral", "prescribed", "medication",
            "follow-up", "schedule", "next", "recommend", "connect",
        ] if kw in lower),
    }
    best = max(scores, key=scores.__getitem__)
    return best if scores[best] > 0 else "Assessments & Observations"


def _find_page(sentence: str, doc: dict) -> int:
    needle = sentence.lower().strip()
    for page in doc["pages"]:
        if needle in page["text"].lower():
            return page["page_num"]
    needle_short = needle[:60]
    for page in doc["pages"]:
        if needle_short in page["text"].lower():
            return page["page_num"]
    return doc["pages"][0]["page_num"] if doc["pages"] else 1


def _extractive_summarize(doc: dict, num_sentences: int = 8, embed_model=None) -> dict[str, list[dict]]:
    """
    Centroid-based extractive summarizer using the fastembed ONNX model.

    Replaces the LSA fallback. Uses no new dependencies — the same
    BAAI/bge-small-en-v1.5 model already loaded for search is reused here.

    Algorithm:
      1. Split doc text into candidate sentences (>40 chars, noise/substance filtered)
      2. Embed all candidates with fastembed
      3. Rank by dot product with centroid (mean embedding)
      4. Classify top sentences into sections with the existing keyword classifier
    """
    import numpy as np

    max_chars = 14000
    text = doc["full_text"][:max_chars]

    raw = re.split(r"(?<=[.!?])\s+", text)
    candidates: list[tuple[str, int]] = [
        (s.strip(), _find_page(s.strip(), doc))
        for s in raw
        if len(s.strip()) > 40
    ]

    filtered = [
        (s, p) for s, p in candidates
        if not _is_noise(s) and _is_substantive(s)
    ]

    if not filtered:
        # Nothing passed the filters — rank by relevance score and take top sentences
        candidates.sort(key=lambda pair: _relevance_score(pair[0]), reverse=True)
        filtered = candidates[:num_sentences * 4]

    if not filtered:
        return {}

    sentences = [s for s, _ in filtered]

    # Rank by centroid similarity using the already-loaded embed_model.
    # Never instantiate a new model here — that would load a second ONNX
    # session alongside the cached one in appv2.py, doubling memory usage.
    try:
        if embed_model is None:
            raise ValueError("no model")
        embs = embed_model.encode(sentences, convert_to_numpy=True)
        centroid = embs.mean(axis=0)
        scores = embs.dot(centroid)
        order = np.argsort(scores)[::-1]
        ranked = [(sentences[i], filtered[i][1]) for i in order]
    except Exception:
        # Model not available — fall back to relevance score ranking
        ranked = sorted(filtered, key=lambda pair: _relevance_score(pair[0]), reverse=True)

    section_counts: dict[str, int] = {}
    sections: dict[str, list[dict]] = defaultdict(list)
    for sentence, page_num in ranked:
        sec = _classify_fallback(sentence)
        if section_counts.get(sec, 0) >= 3:
            continue
        sections[sec].append({"sentence": sentence, "page_num": page_num})
        section_counts[sec] = section_counts.get(sec, 0) + 1
        if sum(section_counts.values()) >= num_sentences:
            break

    return dict(sections)


# ─── Public API ───────────────────────────────────────────────────────────────

def get_ollama_status() -> dict:
    """
    Return current Ollama availability and selected model.
    Called by the UI to show status to the user.
    """
    if not _ollama_available():
        return {"available": False, "model": None}
    model = _get_best_model()
    return {"available": True, "model": model}


def summarize_documents(docs: list[dict], num_sentences_per_doc: int = 8, embed_model=None) -> list[dict]:
    """
    Summarize each document independently.

    Tries Ollama LLM first; falls back to centroid extractive summarizer if Ollama is unavailable.

    Returns list of per-document objects:
      {
        filename, doc_type, client_name, date,
        insufficient_clinical_content: bool,
        llm_used: bool,
        model_used: str | None,
        sections: {section_name: [{sentence, page_num}]}
      }
    """
    # Resolve Ollama once for the whole batch
    ollama_model = _get_best_model() if _ollama_available() else None

    results: list[dict] = []

    for doc in docs:
        doc_type    = _detect_doc_type(doc)
        client_name = _detect_client_name(doc)
        date        = _detect_date(doc)
        llm_used    = False
        model_used  = None
        sections: dict[str, list[dict]] = {}

        if ollama_model:
            try:
                messages = _build_prompt(doc["full_text"], doc_type)
                raw_response = _call_ollama(messages, ollama_model)
                parsed = _parse_llm_json(raw_response)
                if parsed:
                    sections = _llm_response_to_sections(parsed, doc)
                    llm_used = True
                    model_used = ollama_model
            except Exception:
                # LLM call failed — fall through to extractive summarizer
                pass

        if not sections:
            sections = _extractive_summarize(doc, num_sentences_per_doc, embed_model)

        insufficient = sum(len(v) for v in sections.values()) == 0

        results.append({
            "filename":                    doc["filename"],
            "doc_type":                    doc_type,
            "client_name":                 client_name,
            "date":                        date,
            "insufficient_clinical_content": insufficient,
            "llm_used":                    llm_used,
            "model_used":                  model_used,
            "truncated":                   len(doc["full_text"]) > 14000,
            "sections":                    sections,
        })

    return results


def summarize(docs: list[dict], num_sentences: int = 12) -> list[dict]:
    """Flat list for backward compatibility."""
    per_doc = summarize_documents(
        docs,
        num_sentences_per_doc=max(3, num_sentences // max(1, len(docs))),
    )
    flat: list[dict] = []
    for doc_summary in per_doc:
        for section_items in doc_summary["sections"].values():
            for item in section_items:
                flat.append({
                    "sentence": item["sentence"],
                    "filename": doc_summary["filename"],
                    "page_num": item["page_num"],
                })
    return flat
