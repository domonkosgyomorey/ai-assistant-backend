from core.models.models import RelevanceDecision
from langchain.schema import Document


def simple_relevance_checker(docs: list[Document], query: str) -> RelevanceDecision:
    """
    Simple heuristic-based relevance checker.
    Can be replaced with more sophisticated checkers.
    """
    if not docs:
        return RelevanceDecision(relevant=False, reason="No documents retrieved")

    # Simple keyword overlap heuristic
    query_words = set(query.lower().split())

    total_overlap = 0
    for doc in docs[:3]:  # Check top 3 docs
        doc_words = set(doc.page_content.lower().split())
        overlap = len(query_words.intersection(doc_words))
        total_overlap += overlap

    avg_overlap = total_overlap / min(len(docs), 3)

    if avg_overlap >= 3:
        return RelevanceDecision(relevant=True, reason=f"Good keyword overlap (avg: {avg_overlap:.1f})")
    elif avg_overlap >= 1:
        return RelevanceDecision(relevant=True, reason=f"Moderate keyword overlap (avg: {avg_overlap:.1f})")
    else:
        return RelevanceDecision(relevant=False, reason=f"Low keyword overlap (avg: {avg_overlap:.1f})")
