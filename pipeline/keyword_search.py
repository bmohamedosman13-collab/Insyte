"""
Keyword / semantic search across all uploaded documents.

Returns contextual excerpts: the matched sentence plus 1 sentence before and after,
with query terms bolded in the matched sentence.

semantic_search  — cosine similarity via sentence-transformers
exact_search     — case-insensitive substring fallback
contextual_search — semantic search with surrounding context (primary search entry point)
"""

from __future__ import annotations

import re

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# ─── Sentence extraction with positional index ───────────────────────────────

def _build_sentence_index(docs: list[dict]) -> list[dict]:
    """
    Return all sentences with their neighbours on the same page.
    Each item includes private _page_sentences and _idx used to fetch context.
    """
    items: list[dict] = []
    for doc in docs:
        for page in doc["pages"]:
            raw = re.split(r"(?<=[.!?])\s+", page["text"])
            page_sents = [s.strip() for s in raw if len(s.strip()) > 10]
            for idx, sent in enumerate(page_sents):
                items.append(
                    {
                        "sentence": sent,
                        "filename": doc["filename"],
                        "page_num": page["page_num"],
                        "_page_sents": page_sents,
                        "_idx": idx,
                    }
                )
    return items


def _extract_context(item: dict) -> tuple[str | None, str | None]:
    page_sents = item["_page_sents"]
    idx = item["_idx"]
    before = page_sents[idx - 1] if idx > 0 else None
    after = page_sents[idx + 1] if idx < len(page_sents) - 1 else None
    return before, after


def _highlight(sentence: str, query: str) -> str:
    """Bold any query word that appears in the sentence (case-insensitive)."""
    query_words = [w for w in re.split(r"\W+", query) if len(w) > 2]
    result = sentence
    for word in query_words:
        result = re.sub(
            rf"(?i)({re.escape(word)})",
            r"**\1**",
            result,
        )
    return result


def _clean(item: dict) -> dict:
    """Return a copy without private _page_sents / _idx fields."""
    return {k: v for k, v in item.items() if not k.startswith("_")}


# ─── Search functions ────────────────────────────────────────────────────────

def contextual_search(
    query: str,
    docs: list[dict],
    model,
    top_n: int = 10,
) -> list[dict]:
    """
    Semantic search with surrounding context.

    Returns list of:
      {sentence (with keywords bolded), context_before, context_after,
       filename, page_num, score}

    Grouped by filename in the UI — raw list returned here.
    """
    index = _build_sentence_index(docs)
    if not index:
        return []

    texts = [item["sentence"] for item in index]
    query_emb = model.encode([query], convert_to_numpy=True)
    doc_embs = model.encode(texts, convert_to_numpy=True, batch_size=64, show_progress_bar=False)
    scores = cosine_similarity(query_emb, doc_embs)[0]

    THRESHOLD = 0.28  # below this all matches are noise; triggers exact_search fallback

    results: list[dict] = []
    for idx in np.argsort(scores)[::-1]:
        if len(results) >= top_n:
            break
        if float(scores[idx]) < THRESHOLD:
            break
        item = index[idx]
        before, after = _extract_context(item)
        results.append(
            {
                "sentence": _highlight(item["sentence"], query),
                "context_before": before,
                "context_after": after,
                "filename": item["filename"],
                "page_num": item["page_num"],
                "score": float(scores[idx]),
            }
        )
    return results


def semantic_search(
    query: str,
    docs: list[dict],
    model,
    top_n: int = 10,
) -> list[dict]:
    """Semantic search without context (retained for backward compat)."""
    index = _build_sentence_index(docs)
    if not index:
        return []

    texts = [item["sentence"] for item in index]
    query_emb = model.encode([query], convert_to_numpy=True)
    doc_embs = model.encode(texts, convert_to_numpy=True, batch_size=64, show_progress_bar=False)
    scores = cosine_similarity(query_emb, doc_embs)[0]
    top_indices = np.argsort(scores)[::-1][:top_n]

    return [
        {**_clean(index[i]), "score": float(scores[i])}
        for i in top_indices
    ]


def exact_search(query: str, docs: list[dict]) -> list[dict]:
    """
    Case-insensitive substring match with surrounding context.
    Returns all matching sentences.
    """
    needle = query.lower()
    index = _build_sentence_index(docs)
    results: list[dict] = []
    for item in index:
        if needle in item["sentence"].lower():
            before, after = _extract_context(item)
            results.append(
                {
                    "sentence": _highlight(item["sentence"], query),
                    "context_before": before,
                    "context_after": after,
                    "filename": item["filename"],
                    "page_num": item["page_num"],
                    "score": 1.0,
                }
            )
    return results
