"""
Risk / crisis language detection with severity classification.

Runs automatically on every document upload — cannot be suppressed.

Two-pass detection:
  1. Exact keyword/phrase match
  2. Semantic similarity via sentence-transformers

Severity levels:
  assessed   — risk language is present AND a clinical assessment/safety plan is documented
               on the same or adjacent page. Amber / informational.
  unassessed — risk language present, no clinical assessment context found.
               Amber / caution.
  acute      — acute or unresolved indicators: active ideation language, no safety plan
               language detected nearby. Red / high priority.
"""

from __future__ import annotations

import re

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

SEMANTIC_THRESHOLD = 0.80

# ─── Risk vocabulary ─────────────────────────────────────────────────────────

RISK_KEYWORDS: list[str] = [
    # Suicidality
    "kill myself", "end my life", "want to die", "suicidal", "suicide",
    "no reason to live", "don't want to be here", "don't want to live",
    "better off dead", "wish i was dead", "not worth living",
    "end it all", "take my own life",
    # Self-harm
    "self-harm", "self harm", "cutting myself", "hurt myself", "hurting myself",
    "harming myself", "burning myself",
    # Homicidal / harm to others
    "hurt someone", "harm others", "want to hurt", "kill someone",
    "homicidal", "thoughts of hurting",
    # Crisis / acute distress
    "crisis", "breaking point", "falling apart", "losing control",
    "can't go on", "cannot go on",
    # Hopelessness
    "hopeless", "no hope", "pointless", "nothing to look forward to",
    "no future", "give up", "giving up", "can't cope", "cannot cope",
    "overwhelmed",
]

REFERENCE_RISK_PHRASES: list[str] = [
    "I want to kill myself",
    "I am thinking about suicide",
    "I want to end my life",
    "I feel hopeless and see no way out",
    "I have been hurting myself",
    "I am in a crisis right now",
    "I cannot cope anymore",
    "I feel like giving up on everything",
    "I do not want to be alive",
    "I am going to harm myself",
    "I feel completely worthless",
    "There is no point in continuing",
    "I want to hurt someone",
    "I have thoughts of harming myself",
]

# ─── Severity classifiers ────────────────────────────────────────────────────

# Presence of these near a flagged sentence → clinician has already assessed
_ASSESSED_MARKERS: list[str] = [
    "risk assessment", "safety assessment", "assessed as", "risk level",
    "low risk", "moderate risk", "high risk", "minimal risk", "no risk",
    "safety plan", "safety planning", "plan in place", "has a safety plan",
    # present tense denials
    "denies suicidal", "denies si", "denies any ideation",
    "denies homicidal", "no active ideation", "no current suicidal",
    "no imminent risk", "not at risk", "denies intent", "denies plan",
    # past tense denials — common in clinical notes ("Client denied suicidal ideation")
    "denied suicidal", "denied si", "denied any ideation",
    "denied homicidal", "denied intent", "denied plan",
    "denied any", "no suicidal ideation", "no homicidal ideation",
    # assessment documentation phrases
    "completed risk assessment", "risk screened", "screened for",
    "risk assessment completed", "verbal risk assessment",
    "no acute risk", "no acute safety concern", "no immediate safety concern",
    "no current risk", "no imminent safety concern",
    "ruled out", "risk: low", "risk: moderate", "risk: high",
    # historical / past framing markers
    "historical", "not recent", "in the past", "previously", "one-time",
    "no current", "not current",
]

# Presence of these IN the flagged sentence itself → acute
_ACUTE_SENTENCE_MARKERS: list[str] = [
    "right now", "tonight", "this evening", "today i am", "right at this moment",
    "active plan", "has a plan to", "intent to", "means to act", "will act",
    "plan to end", "planning to end", "going to do it", "going to hurt",
]

# Presence near the flagged sentence — these do NOT override to assessed
# (no safety plan language = higher concern)
_SAFETY_PLAN_MARKERS: list[str] = [
    "safety plan", "plan in place", "crisis line", "emergency contact",
    "follow-up scheduled", "follow up scheduled",
]


def _get_page_context(filename: str, page_num: int, docs: list[dict]) -> str:
    """Return lowercased text of the flagged page plus one page either side."""
    texts: list[str] = []
    for doc in docs:
        if doc["filename"] != filename:
            continue
        for page in doc["pages"]:
            if abs(page["page_num"] - page_num) <= 1:
                texts.append(page["text"].lower())
    return " ".join(texts)


def _classify_severity(item: dict, docs: list[dict]) -> str:
    context = _get_page_context(item["filename"], item["page_num"], docs)
    sentence_lower = item["sentence"].lower()

    # If the context contains documented assessment language → assessed
    if any(marker in context for marker in _ASSESSED_MARKERS):
        return "assessed"

    # If the sentence itself has acute/active language and no safety plan nearby → acute
    has_acute_language = any(marker in sentence_lower for marker in _ACUTE_SENTENCE_MARKERS)
    has_safety_plan = any(marker in context for marker in _SAFETY_PLAN_MARKERS)
    if has_acute_language and not has_safety_plan:
        return "acute"

    return "unassessed"


# ─── Sentence extraction ─────────────────────────────────────────────────────

def _all_sentences(docs: list[dict]) -> list[dict]:
    results: list[dict] = []
    for doc in docs:
        for page in doc["pages"]:
            raw = re.split(r"(?<=[.!?])\s+", page["text"])
            for sent in raw:
                sent = sent.strip()
                if len(sent) > 10:
                    results.append(
                        {
                            "sentence": sent,
                            "filename": doc["filename"],
                            "page_num": page["page_num"],
                        }
                    )
    return results


def _exact_matches(sentences: list[dict]) -> list[dict]:
    flagged: list[dict] = []
    for s in sentences:
        text_lower = s["sentence"].lower()
        for kw in RISK_KEYWORDS:
            if kw in text_lower:
                flagged.append({**s, "match_type": "exact", "matched_phrase": kw})
                break
    return flagged


def _semantic_matches(
    sentences: list[dict],
    model,
    already_flagged_texts: set[str],
) -> list[dict]:
    candidates = [s for s in sentences if s["sentence"] not in already_flagged_texts]
    if not candidates:
        return []

    candidate_embs = model.encode(
        [s["sentence"] for s in candidates],
        convert_to_numpy=True,
        batch_size=64,
        show_progress_bar=False,
    )
    reference_embs = model.encode(
        REFERENCE_RISK_PHRASES,
        convert_to_numpy=True,
        batch_size=64,
        show_progress_bar=False,
    )

    sim = cosine_similarity(candidate_embs, reference_embs)
    max_sims = sim.max(axis=1)
    best_ref_idxs = sim.argmax(axis=1)

    flagged: list[dict] = []
    for i, (score, ref_idx) in enumerate(zip(max_sims, best_ref_idxs)):
        if float(score) >= SEMANTIC_THRESHOLD:
            flagged.append(
                {
                    **candidates[i],
                    "match_type": "semantic",
                    "matched_phrase": REFERENCE_RISK_PHRASES[ref_idx],
                    "score": float(score),
                }
            )
    return flagged


# ─── Public API ──────────────────────────────────────────────────────────────

def scan_risks(docs: list[dict], model) -> list[dict]:
    """
    Scan all documents for risk / crisis language.

    Always runs on upload. Results must always be displayed if non-empty.

    Returns list of flagged items:
      {sentence, filename, page_num, match_type, matched_phrase, severity}

    severity: "assessed" | "unassessed" | "acute"
    """
    sentences = _all_sentences(docs)
    if not sentences:
        return []

    exact_flagged = _exact_matches(sentences)
    already_flagged = {s["sentence"] for s in exact_flagged}
    semantic_flagged = _semantic_matches(sentences, model, already_flagged)

    all_flagged = exact_flagged + semantic_flagged

    # Classify severity for each flag
    for item in all_flagged:
        item["severity"] = _classify_severity(item, docs)

    return all_flagged
