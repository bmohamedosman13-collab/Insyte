"""
Query-driven linguistic analysis.

The user describes the type of language to find (e.g. "depressive language",
"satisfaction with treatment") and the tool:
  1. Expands the query with cognitive-linguistic markers for known themes
  2. Retrieves the most semantically relevant passages from the document
  3. Synthesises a qualitative narrative with direct quotes (via Ollama)
  4. Falls back to a scored, template-based summary when Ollama is unavailable

No HuggingFace classifier required — uses only the shared embedding model
and the already-running Ollama instance.
"""

from __future__ import annotations

import re

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .ollama_client import (
    call_ollama as _call_ollama,
    get_best_model as _get_best_model,
    ollama_available as _ollama_available,
    parse_llm_json as _parse_llm_json,
)

# ─── Intensity thresholds ─────────────────────────────────────────────────────
# Tuned for all-MiniLM-L6-v2 cosine similarity scores.
# The *mean* score of the top-N retrieved passages drives the intensity label.

_ABSENT_CUTOFF   = 0.30
_MILD_CUTOFF     = 0.38
_MODERATE_CUTOFF = 0.48

_INTENSITY_LABELS = {
    "absent":   "No clear indicators found",
    "mild":     "Mild indicators present",
    "moderate": "Moderate indicators present",
    "strong":   "Strong indicators present",
}

# ─── Cognitive-linguistic marker expansion ────────────────────────────────────
# For well-known clinical / psychosocial themes, appending domain markers to
# the query improves retrieval because the embedding model was trained on
# general web text, not clinical language.
# Key = substring match; value = marker string appended to the query.

_EXPANSIONS: dict[str, str] = {
    "depress": (
        "hopelessness worthlessness no motivation emptiness low mood cannot cope "
        "withdrawn given up no energy anhedonia guilt burden helpless meaningless "
        "nothing to look forward to sad tearful crying fatigued"
    ),
    "anxiet": (
        "worry fear panic overwhelmed nervous dread cannot relax on edge "
        "racing thoughts avoidance hypervigilance restless tension apprehension "
        "anticipatory fear generalised worry"
    ),
    "suicid": (
        "want to die end my life no reason to live self-harm hopeless "
        "better off dead plan to hurt myself passive death wish ideation"
    ),
    "satisf": (
        "happy pleased helpful improved grateful working well positive change "
        "better recommend effective useful progress beneficial met expectations"
    ),
    "trauma": (
        "flashback nightmare avoidance hypervigilance intrusive triggered "
        "reliving detachment numbing re-experiencing dissociation startle response"
    ),
    "substance": (
        "drinking alcohol drug use cannabis opioid using again relapse "
        "withdrawal craving intoxicated sober abstinence dependence"
    ),
    "function": (
        "activities of daily living able to work managing tasks independent "
        "self-care cooking hygiene employment capacity impairment occupational"
    ),
    "social": (
        "isolated no friends withdrawn support network lonely "
        "family relationships disconnected no contact estranged"
    ),
    "anger": (
        "frustrated angry rage irritable hostile resentful aggressive "
        "outburst conflict violent temper dysregulation"
    ),
    "grief": (
        "loss bereavement mourning died death missing someone grieving "
        "sadness after loss memorial anniversary complicated grief"
    ),
    "hope": (
        "future plans goals optimism looking forward recovery progress "
        "positive outlook change motivation engaged"
    ),
    "comply": (
        "attending sessions following treatment plan medication adherence "
        "engaged cooperative motivated participating consistent"
    ),
}


def _expand_query(query: str) -> str:
    """Append cognitive-linguistic marker phrases for recognised themes."""
    q_lower = query.lower()
    for keyword, markers in _EXPANSIONS.items():
        if keyword in q_lower:
            return f"{query}. {markers}"
    return query


# ─── Passage retrieval ────────────────────────────────────────────────────────

def _split_sentences(doc: dict) -> list[dict]:
    sentences: list[dict] = []
    for page in doc["pages"]:
        raw = re.split(r"(?<=[.!?])\s+", page["text"])
        for s in raw:
            s = s.strip()
            if len(s) > 25:
                sentences.append({"text": s, "page_num": page["page_num"]})
    return sentences


def retrieve_passages(
    query: str,
    doc: dict,
    embed_model,
    top_n: int = 12,
) -> tuple[list[dict], float]:
    """
    Return the top-N most semantically relevant passages from a single document,
    plus the mean similarity score used for intensity classification.
    """
    sentences = _split_sentences(doc)
    if not sentences:
        return [], 0.0

    expanded = _expand_query(query)
    q_emb = embed_model.encode([expanded], convert_to_numpy=True)
    d_embs = embed_model.encode(
        [s["text"] for s in sentences],
        convert_to_numpy=True,
        batch_size=64,
        show_progress_bar=False,
    )
    scores = cosine_similarity(q_emb, d_embs)[0]

    top_idx = np.argsort(scores)[::-1][:top_n]
    passages = [
        {
            "text": sentences[i]["text"],
            "page_num": sentences[i]["page_num"],
            "score": float(scores[i]),
        }
        for i in top_idx
        if float(scores[i]) > 0.22
    ]

    mean_score = float(np.mean([p["score"] for p in passages])) if passages else 0.0
    return passages, mean_score


def _score_to_intensity(mean_score: float) -> str:
    if mean_score < _ABSENT_CUTOFF:
        return "absent"
    if mean_score < _MILD_CUTOFF:
        return "mild"
    if mean_score < _MODERATE_CUTOFF:
        return "moderate"
    return "strong"


# ─── Ollama synthesis ─────────────────────────────────────────────────────────

_SYS_PROMPT = (
    "You are a careful document analyst. Your task is to characterise the presence "
    "and nature of a specific type of language or theme within extracted document "
    "passages. Be evidence-based, precise, and non-judgmental. Do not diagnose. "
    "Do not speculate beyond what is written. "
    "Output ONLY valid JSON — no prose, no markdown fences."
)


def _build_prompt(query: str, filename: str, passages: list[dict]) -> list[dict]:
    passage_block = "\n\n".join(
        f"[p.{p['page_num']}] {p['text']}" for p in passages[:10]
    )
    user = (
        f'Document: "{filename}"\n\n'
        f"Relevant passages:\n{passage_block}\n\n"
        f'Looking for: "{query}"\n\n'
        "Return a JSON object with exactly these keys:\n"
        '  "verdict"   — one sentence characterising presence/absence/nature of '
        'this language IN THE SUBJECT\'S OWN EXPERIENCE AND EXPRESSION '
        '(e.g. "The subject expresses moderate depressive language, '
        'particularly around themes of hopelessness and withdrawal.")\n'
        '  "intensity" — exactly one of: "absent", "mild", "moderate", "strong"\n'
        '  "evidence"  — array of up to 4 objects, each:\n'
        '                  {"quote": str, "page_num": int, "note": str}\n'
        '                  quote = exact excerpt no longer than 50 words '
        '(PREFER the subject\'s own words over clinician notes); '
        'note = one brief phrase on relevance\n'
        '  "summary"   — 2-3 sentence paragraph starting with "The subject..." '
        'describing what the INDIVIDUAL expressed or experienced. '
        'If absent/minimal, say so explicitly.\n\n'
        "CRITICAL RULES:\n"
        "- Focus on the SUBJECT'S OWN language, feelings, and self-reported experiences.\n"
        "- The 'intensity' rating must reflect what the SUBJECT expresses — NOT what a "
        "clinician writes in their notes or assessment.\n"
        "- If a passage is a clinician's observation rather than the subject's own words, "
        "label it clearly as such in the 'note' field and give it lower weight.\n"
        "- Only quote text present in the passages provided.\n"
        "- Honour negation: 'denies X' means X is NOT present for the subject.\n"
        "- If passages are clinical/administrative text unrelated to the subject's "
        "own experience, set intensity to 'absent'."
    )
    return [
        {"role": "system", "content": _SYS_PROMPT},
        {"role": "user", "content": user},
    ]


def _ollama_analyze(query: str, filename: str, passages: list[dict]) -> dict | None:
    """Try Ollama synthesis; return None on failure or unavailability."""
    if not _ollama_available():
        return None
    model = _get_best_model()
    if not model:
        return None
    try:
        messages = _build_prompt(query, filename, passages)
        raw = _call_ollama(messages, model)
        parsed = _parse_llm_json(raw)
        if all(k in parsed for k in ("verdict", "intensity", "evidence", "summary")):
            return parsed
    except Exception:  # noqa: BLE001
        pass
    return None


# ─── Rule-based fallback ──────────────────────────────────────────────────────

def _fallback_analyze(query: str, passages: list[dict], intensity: str) -> dict:
    label = _INTENSITY_LABELS[intensity]
    if intensity == "absent":
        verdict = (
            f"No clear indicators of {query.lower()} were identified in this document."
        )
        summary = (
            verdict
            + " The retrieved passages did not contain language consistent with the query."
        )
    else:
        verdict = (
            f"This document contains {label.lower()} consistent with {query.lower()}."
        )
        summary = (
            f"{verdict} "
            f"{len(passages)} relevant passage(s) were identified. "
            "The evidence below reflects the closest matches found. "
            "Enable Ollama for richer AI-assisted analysis with direct quotes."
        )

    evidence = [
        {
            "quote": p["text"][:280],
            "page_num": p["page_num"],
            "note": f"Relevance score: {p['score']:.2f}",
        }
        for p in passages[:4]
    ]

    return {
        "verdict": verdict,
        "intensity": intensity,
        "evidence": evidence,
        "summary": summary,
        "llm_used": False,
    }


# ─── Public API ───────────────────────────────────────────────────────────────

def analyze_language(
    query: str,
    doc: dict,
    embed_model,
    top_n: int = 12,
) -> dict:
    """
    Analyse a single document for the presence and nature of user-queried language.

    Parameters
    ----------
    query       : free-text description of what to look for
    doc         : document dict from extract_documents
    embed_model : loaded SentenceTransformer instance
    top_n       : number of passages to retrieve for analysis

    Returns
    -------
    {
      "query":     str,
      "filename":  str,
      "verdict":   str,          # one-sentence characterisation
      "intensity": str,          # "absent" | "mild" | "moderate" | "strong"
      "evidence":  list[dict],   # [{"quote", "page_num", "note"}]
      "summary":   str,          # narrative paragraph
      "llm_used":  bool,
    }
    """
    passages, mean_score = retrieve_passages(query, doc, embed_model, top_n=top_n)
    intensity = _score_to_intensity(mean_score)

    llm_result = _ollama_analyze(query, doc["filename"], passages)
    if llm_result is not None:
        llm_result.update({
            "query": query,
            "filename": doc["filename"],
            "llm_used": True,
        })
        return llm_result

    result = _fallback_analyze(query, passages, intensity)
    result.update({"query": query, "filename": doc["filename"]})
    return result
