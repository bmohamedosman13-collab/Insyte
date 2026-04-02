"""
Prompt-driven pattern search across all uploaded documents.

Features:
  - Clinical abbreviation expansion before embedding
  - Semantic cosine similarity (all-MiniLM-L6-v2)
  - Exact substring fallback when semantic returns no results
  - Per-match confidence label: Strong / Moderate / Weak
  - Low-confidence warning when best score < 0.55
"""

from __future__ import annotations

import re

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ─── Thresholds ───────────────────────────────────────────────────────────────

THRESHOLD = 0.42          # hard minimum — results below this are not returned
LOW_CONFIDENCE_CUTOFF = 0.50   # warn the user if the best match is below this
STRONG_CUTOFF = 0.60
MODERATE_CUTOFF = 0.50    # [0.50, 0.60) = Moderate; [0.42, 0.50) = Weak

# ─── Clinical abbreviation expansion ─────────────────────────────────────────
# Applied as whole-word substitutions (case-sensitive keys where case matters,
# case-insensitive fallback for single-letter abbreviations).
# Ordered longest-first to avoid partial collisions.

_EXPANSIONS: list[tuple[str, str]] = [
    # Multi-letter — match as whole words, case-insensitive
    ("ADLs",  "activities of daily living"),
    ("ADL",   "activities of daily living"),
    ("PTSD",  "post-traumatic stress disorder"),
    ("MDD",   "major depressive disorder"),
    ("GAD",   "generalized anxiety disorder"),
    ("OCD",   "obsessive compulsive disorder"),
    ("PMH",   "past medical history"),
    ("MSE",   "mental status examination"),
    ("CBT",   "cognitive behavioural therapy"),
    ("DBT",   "dialectical behaviour therapy"),
    ("ACT",   "acceptance and commitment therapy"),
    ("GAF",   "global assessment of functioning"),
    ("CDSS",  "calgary depression scale for schizophrenia"),
    ("PHQ",   "patient health questionnaire"),
    ("GAD-7", "generalized anxiety disorder scale"),
    ("PCL",   "PTSD checklist"),
    ("SA",    "substance abuse"),
    ("MH",    "mental health"),
    # Single-letter clinical abbreviations (context-sensitive — ordered carefully)
    ("SI",    "suicidal ideation"),
    ("HI",    "homicidal ideation"),
    ("Hx",    "history"),
    ("hx",    "history"),
    ("Dx",    "diagnosis"),
    ("dx",    "diagnosis"),
    ("Tx",    "treatment"),
    ("tx",    "treatment"),
    ("Rx",    "prescription medication"),
    ("rx",    "prescription medication"),
    ("Sx",    "symptoms"),
    ("sx",    "symptoms"),
    ("Fx",    "fracture"),
    ("Ax",    "assessment"),
]


def expand_query(query: str) -> str:
    """
    Replace clinical abbreviations in the query with their full phrases.
    Returns the expanded query string (original kept if no match found).
    """
    result = query
    for abbrev, expansion in _EXPANSIONS:
        # Whole-word boundary replacement, case-sensitive for the abbreviation
        result = re.sub(
            rf"\b{re.escape(abbrev)}\b",
            expansion,
            result,
        )
    return result


# ─── Meta-language stripping ──────────────────────────────────────────────────
# Users often phrase queries as instructions ("Find the plans for the patient
# across the documents") rather than as content descriptors ("treatment plans").
# Stripping the instructional wrapper improves embedding quality.

_META_RE = re.compile(
    r"\b(?:find|show|get|identify|locate|search\s+for|look\s+for|retrieve|"
    r"list|extract|give\s+me|tell\s+me)\b"
    r"|(?:\b(?:across|throughout|within|in|from)\s+(?:the\s+)?(?:all\s+)?documents?\b)"
    r"|(?:\bfor\s+the\s+(?:patient|client|subject|individual|person|resident|youth)\b)"
    r"|(?:\bthe\s+(?:patient|client|subject|individual|person|resident|youth)\b)",
    re.IGNORECASE,
)


def _strip_meta(query: str) -> str:
    """Remove instructional wrapper words so the embedding targets content."""
    cleaned = _META_RE.sub("", query)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip().strip(",").strip()
    return cleaned if len(cleaned) > 5 else query


# ─── Sentence index ───────────────────────────────────────────────────────────

def _build_sentence_index(docs: list[dict]) -> list[dict]:
    items: list[dict] = []
    for doc in docs:
        if "sentences" in doc:
            from collections import defaultdict as _dd
            by_page: dict = _dd(list)
            for s in doc["sentences"]:
                if len(s["text"]) > 20:  # patterns uses a slightly higher threshold
                    by_page[s["page_num"]].append(s["text"])
            for pn, page_sents in by_page.items():
                for idx, sent in enumerate(page_sents):
                    items.append({
                        "sentence": sent,
                        "filename": doc["filename"],
                        "page_num": pn,
                        "_page_sents": page_sents,
                        "_idx": idx,
                    })
        else:
            for page in doc["pages"]:
                raw = re.split(r"(?<=[.!?])\s+", page["text"])
                page_sents = [s.strip() for s in raw if len(s.strip()) > 20]
                for idx, sent in enumerate(page_sents):
                    items.append({
                        "sentence": sent,
                        "filename": doc["filename"],
                        "page_num": page["page_num"],
                        "_page_sents": page_sents,
                        "_idx": idx,
                    })
    return items


def _get_context(item: dict) -> tuple[str | None, str | None]:
    sents = item["_page_sents"]
    idx = item["_idx"]
    return (
        sents[idx - 1] if idx > 0 else None,
        sents[idx + 1] if idx < len(sents) - 1 else None,
    )


def _confidence_label(score: float) -> str:
    if score >= STRONG_CUTOFF:
        return "Strong"
    if score >= MODERATE_CUTOFF:
        return "Moderate"
    return "Weak"


# ─── Exact fallback ───────────────────────────────────────────────────────────

_STOPWORDS: frozenset[str] = frozenset({
    "mentioned", "completed", "described", "found", "noted", "recorded",
    "documented", "identified", "indicated", "reported", "the", "and",
    "for", "with", "from", "that", "this", "are", "was", "were", "been",
    "have", "has", "had", "any", "all", "about", "into", "or", "not",
    "what", "how", "when", "where", "which", "who",
})


def _key_nouns(query: str) -> list[str]:
    """Extract content words from the (already expanded) query for substring matching."""
    words = re.findall(r"[a-zA-Z]{3,}", query.lower())
    return [w for w in words if w not in _STOPWORDS]


def _exact_fallback(
    expanded_query: str,
    index: list[dict],
) -> list[dict]:
    """
    Substring match: return sentences containing any key noun from the query.
    Used only when semantic search yields no results above THRESHOLD.
    """
    nouns = _key_nouns(expanded_query)
    if not nouns:
        return []

    seen: set[str] = set()
    matches: list[dict] = []

    for item in index:
        lower = item["sentence"].lower()
        if any(noun in lower for noun in nouns):
            key = (item["filename"], item["sentence"])
            if key in seen:
                continue
            seen.add(key)
            before, after = _get_context(item)
            matches.append(
                {
                    "sentence": item["sentence"],
                    "context_before": before,
                    "context_after": after,
                    "filename": item["filename"],
                    "page_num": item["page_num"],
                    "score": None,           # no cosine score for exact matches
                    "match_confidence": "Exact",
                    "match_source": "exact",
                }
            )

    return matches


# ─── Public API ───────────────────────────────────────────────────────────────

def pattern_search(
    query: str,
    docs: list[dict],
    model,
    top_n: int = 20,
    precomputed_index: list[dict] | None = None,
    precomputed_embeddings=None,
) -> dict:
    """
    Search for a user-described pattern across all documents.

    Returns:
      {
        "groups": [{filename, matches: [{sentence, context_before, context_after,
                    page_num, score, match_confidence, match_source}]}],
        "low_confidence_warning": bool,
        "expanded_query": str,         # shown in UI if expansion changed the query
        "fallback_used": bool,
      }

    match_confidence: "Strong" | "Moderate" | "Weak" | "Exact"
    match_source:     "semantic" | "exact"
    """
    expanded = expand_query(query)
    search_query = _strip_meta(expanded)   # strip "Find X across the documents" wrapper
    index = precomputed_index if precomputed_index is not None else _build_sentence_index(docs)

    if not index:
        return {"groups": [], "low_confidence_warning": False,
                "expanded_query": expanded, "fallback_used": False}

    # ── Semantic search ──
    query_emb = model.encode([search_query], convert_to_numpy=True)
    if precomputed_embeddings is not None:
        doc_embs = precomputed_embeddings
    else:
        texts = [item["sentence"] for item in index]
        doc_embs = model.encode(texts, convert_to_numpy=True, batch_size=64, show_progress_bar=False)
    scores = cosine_similarity(query_emb, doc_embs)[0]

    ranked = [
        (i, float(scores[i]))
        for i in np.argsort(scores)[::-1]
        if float(scores[i]) >= THRESHOLD
    ][:top_n]

    # ── Exact fallback if semantic returns nothing ──
    fallback_used = len(ranked) == 0
    if fallback_used:
        flat_matches = _exact_fallback(search_query, index)
        groups: dict[str, list[dict]] = {}
        for m in flat_matches:
            groups.setdefault(m["filename"], []).append(m)
        return {
            "groups": [
                {"filename": fname, "matches": matches}
                for fname, matches in groups.items()
            ],
            "low_confidence_warning": False,
            "expanded_query": expanded,
            "fallback_used": True,
        }

    # ── Build result groups from semantic matches ──
    best_score = ranked[0][1] if ranked else 0.0
    low_confidence_warning = best_score < LOW_CONFIDENCE_CUTOFF

    groups_dict: dict[str, list[dict]] = {}
    for idx, score in ranked:
        item = index[idx]
        before, after = _get_context(item)
        match = {
            "sentence": item["sentence"],
            "context_before": before,
            "context_after": after,
            "page_num": item["page_num"],
            "score": score,
            "match_confidence": _confidence_label(score),
            "match_source": "semantic",
        }
        groups_dict.setdefault(item["filename"], []).append(match)

    return {
        "groups": [
            {"filename": fname, "matches": sorted(ms, key=lambda m: -(m["score"] or 0))}
            for fname, ms in groups_dict.items()
        ],
        "low_confidence_warning": low_confidence_warning,
        "expanded_query": expanded,
        "fallback_used": False,
    }
