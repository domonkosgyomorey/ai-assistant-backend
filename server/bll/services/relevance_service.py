from core.models.models import RelevanceDecision
from core.utils.relevance import simple_relevance_checker
from langchain.schema import Document
from langchain_core.language_models import BaseLanguageModel


def create_structured_relevance_checker(llm: BaseLanguageModel):
    """
    Modern LangChain structured output relevance checker.
    Uses with_structured_output() for cleaner implementation.
    Requires LangChain models that support structured output (like OpenAI, Anthropic).
    """

    def check_relevance(docs: list[Document], query: str) -> RelevanceDecision:
        if not docs:
            return RelevanceDecision(relevant=False, reason="No documents retrieved")

        doc_excerpts = "\n".join([f"Doc {i + 1}: {doc.page_content[:300]}..." for i, doc in enumerate(docs[:3])])

        try:
            structured_llm = llm.with_structured_output(RelevanceDecision)

            prompt = f"""Given the following question and document excerpts, determine if the documents contain relevant information to answer the question.

Question: {query}

Documents:
{doc_excerpts}

Analyze whether the documents contain information that could help answer the question. Consider partial relevance as well.
Provide your decision as a structured response with 'relevant' (boolean) and 'reason' (string explaining your decision)."""

            result = structured_llm.invoke(prompt)

            if isinstance(result, RelevanceDecision):
                return result
            else:
                return RelevanceDecision(relevant=False, reason="Invalid structured output format")

        except Exception:
            return simple_relevance_checker(docs, query)

    return check_relevance
