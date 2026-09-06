"""
faq_cache.py
Provides fast FAQ matching using embeddings, with unmatched questions handled by the main agent.
"""

from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np

SIMILARITY_THRESHOLD = 0.80

FAQS = [
    {
        "question": "What are the hospital's visiting hours?",
        "answer": (
            "General visiting hours are typically in the afternoon and evening. "
            "For the exact current visiting hours for a specific ward, please check "
            "the hospital's website or call reception, as this can vary by ward."
        ),
    },
    {
        "question": "How do I book a doctor's channeling appointment?",
        "answer": (
            "You can book a channeling appointment by asking me for a doctor's "
            "available sessions (I can look up live schedule, room, and fee info), "
            "or by calling the hospital's channeling center directly."
        ),
    },
    {
        "question": "Do you accept health insurance?",
        "answer": (
            "Most private hospitals in Sri Lanka work with major local and "
            "international insurance providers, but coverage depends on your policy. "
            "Please check with the billing/insurance desk to confirm your specific plan."
        ),
    },
    {
        "question": "Is there an emergency department available 24/7?",
        "answer": (
            "Most full-service private hospitals maintain a 24-hour emergency unit. "
            "For a life-threatening emergency, please call ahead or go directly to "
            "the nearest emergency department."
        ),
    },
    {
        "question": "How can I get my lab test results?",
        "answer": (
            "Lab report delivery times vary by test (I can look up the exact turnaround "
            "time for a specific test). Reports are usually collected in person, or "
            "increasingly are available via the hospital's online patient portal, if offered."
        ),
    },
]

_embeddings = None
_faq_vectors = None


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def _get_embeddings():
    global _embeddings, _faq_vectors
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        _faq_vectors = [
            np.array(_embeddings.embed_query(faq["question"])) for faq in FAQS
        ]
    return _embeddings, _faq_vectors


def match_faq(user_message: str) -> str | None:
    """Return a canonical FAQ answer if the message closely matches a known FAQ, else None."""
    embeddings, faq_vectors = _get_embeddings()
    query_vec = np.array(embeddings.embed_query(user_message))

    best_score, best_idx = -1.0, -1
    for i, vec in enumerate(faq_vectors):
        score = _cosine_sim(query_vec, vec)
        if score > best_score:
            best_score, best_idx = score, i

    if best_score >= SIMILARITY_THRESHOLD:
        return FAQS[best_idx]["answer"]
    return None
