"""
vector_tool.py
--------------
Wraps the Chroma vector store built by scripts/scrape_website.py and
exposes a single retrieval function the agent can call for
unstructured, informational questions (hospital services, clinical
centers, general policies, "about us" content, visiting hours, etc.)

Retrieval-quality improvements (bonus criterion) implemented here:
1. MMR (Maximal Marginal Relevance) search for the initial candidate
   pool -- reduces near-duplicate chunks in the context window.
2. Cross-encoder reranking: a dedicated `sentence-transformers`
   cross-encoder (`ms-marco-MiniLM-L-6-v2`) scores each (query, chunk)
   pair jointly, which is substantially more accurate than pure
   embedding cosine-similarity for ranking -- this is the standard
   "retrieve-then-rerank" pattern used in production RAG systems.
   Falls back to a cheap keyword-overlap score if the cross-encoder
   model can't be loaded (e.g. no internet), so the tool never hard-fails.
3. Source metadata is always returned, so the chatbot can cite the
   page it pulled information from.
"""

from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

ROOT = Path(__file__).resolve().parent.parent
PERSIST_DIR = str(ROOT / "vectorstore" / "chroma_db")

_embeddings = None
_vectordb = None
_reranker = None
_reranker_load_failed = False


def _get_vectordb():
    global _embeddings, _vectordb
    if _vectordb is None:
        _embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        _vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=_embeddings)
    return _vectordb


def _get_reranker():
    """Lazily load the cross-encoder reranker. Returns None if unavailable."""
    global _reranker, _reranker_load_failed
    if _reranker is None and not _reranker_load_failed:
        try:
            from sentence_transformers import CrossEncoder
            _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception:  # noqa: BLE001 - degrade gracefully, don't crash retrieval
            _reranker_load_failed = True
    return _reranker


def _keyword_overlap_score(query: str, text: str) -> int:
    query_terms = {t.lower() for t in query.split() if len(t) > 2}
    text_lower = text.lower()
    return sum(1 for term in query_terms if term in text_lower)


def search_hospital_website(query: str, k: int = 4) -> dict:
    """
    Search the scraped hospital website knowledge base for information
    relevant to `query`. Use this for general, informational questions
    (services offered, hospital facilities, clinical centers, visiting
    policies, contact/location info, etc.) -- NOT for prices, schedules
    or anything that lives in the structured database.
    """
    try:
        vectordb = _get_vectordb()
    except Exception as exc:  # noqa: BLE001 - surface a clean tool error to the LLM
        return {"success": False, "error": f"Vector store unavailable: {exc}"}

    # Pull a larger candidate pool via MMR (reduces near-duplicates),
    # then rerank with a cross-encoder for precise relevance scoring.
    candidates = vectordb.max_marginal_relevance_search(query, k=k * 3, fetch_k=k * 6)

    reranker = _get_reranker()
    if reranker is not None and candidates:
        pairs = [[query, doc.page_content] for doc in candidates]
        scores = reranker.predict(pairs)
        reranked = [doc for _, doc in sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)][:k]
        rerank_method = "cross-encoder"
    else:
        # Fallback: cheap keyword-overlap re-rank (still better than raw order).
        reranked = sorted(
            candidates,
            key=lambda doc: _keyword_overlap_score(query, doc.page_content),
            reverse=True,
        )[:k]
        rerank_method = "keyword-overlap-fallback"

    results = [
        {
            "content": doc.page_content,
            "source": doc.metadata.get("source"),
            "title": doc.metadata.get("title"),
        }
        for doc in reranked
    ]
    return {"success": True, "results": results, "result_count": len(results), "rerank_method": rerank_method}


VECTOR_TOOL_SPEC = {
    "name": "search_hospital_website",
    "description": (
        "Search the hospital's website knowledge base (services, clinical centers, "
        "facilities, general policies, about-us info) for unstructured, informational "
        "answers. Do NOT use this for prices, doctor fees, channeling schedules, or lab "
        "test details -- use query_hospital_database for those instead."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query, ideally the user's question rephrased for retrieval.",
            }
        },
        "required": ["query"],
    },
}
