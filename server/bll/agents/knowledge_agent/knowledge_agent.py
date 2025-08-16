from typing import Callable

from bll.agents.base_agent import BaseAgent
from bll.agents.knowledge_agent.prompts import get_knowledge_prompt
from core.interfaces import BaseRetriever
from core.models.models import RelevanceDecision
from langchain.schema import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda


class KnowledgeAgent(BaseAgent):
    """
    Knowledge agent that leverages internal knowledge and optional web search
    to provide comprehensive answers with proper source handling.
    """

    def __init__(
        self,
        llm: BaseLanguageModel,
        domain_context: str,
        db_retriever: BaseRetriever,
        relevance_checker: Callable[[list[Document], str], RelevanceDecision],
        web_search_retriever: BaseRetriever | None = None,
        db_top_k: int = 5,
        web_max_k: int = 5,
        web_supplement_k: int = 2,
        verbose: bool = False,
    ):
        """
        Initialize knowledge agent.

        Args:
            llm: LangChain-compatible language model
            domain_context: Required domain context for the agent
            db_retriever: Enhanced database retriever
            relevance_checker: Function to check document relevance
            web_search_retriever: Optional web search retriever
            db_top_k: Number of top documents to retrieve from the database
            web_max_k: Maximum number of web search results to retrieve
            web_supplement_k: Number of web search results to use for supplementation
            verbose: Enable detailed logging
        """
        self.domain_context = domain_context
        self.db_retriever = db_retriever
        self.relevance_checker = relevance_checker
        self.web_search_retriever = web_search_retriever
        self.db_top_k = db_top_k
        self.web_max_k = web_max_k
        self.web_supplement_k = web_supplement_k

        super().__init__(llm, verbose)

    def _retrieve_db_docs(self, input_dict: dict) -> dict:
        """Retrieve documents from the database."""
        query = input_dict["question"]
        docs = self.db_retriever.retrieve(
            query=query,
            k=self.db_top_k,
        )
        self._log(f"Retrieved {len(docs)} docs from database")
        return {**input_dict, "db_docs": docs}

    def _check_relevance(self, input_dict: dict) -> dict:
        """Check if retrieved documents are relevant to the query."""
        query = input_dict["question"]
        docs = input_dict["db_docs"]
        decision = self.relevance_checker(docs, query)
        self._log("Relevance decision", {"relevant": decision.relevant, "reason": decision.reason})
        return {**input_dict, "relevance_decision": decision}

    def _conditional_web_search(self, input_dict: dict) -> dict:
        """Conditionally perform web search based on relevance and configuration."""
        web_docs = []

        if self.web_search_retriever:
            query = input_dict["question"]
            decision = input_dict["relevance_decision"]

            if decision.relevant:
                web_k = min(self.web_supplement_k, self.web_max_k)
                self._log(f"Supplementing with {web_k} web results")
            else:
                web_k = self.web_max_k
                self._log(f"DB docs not relevant, using {web_k} web results")

            web_docs = self.web_search_retriever.retrieve(query, k=web_k)
            self._log(f"Retrieved {len(web_docs)} web docs")

        return {**input_dict, "web_docs": web_docs}

    def _merge_context(self, input_dict: dict) -> dict:
        """Merge all retrieved documents into a single context."""
        db_docs = input_dict.get("db_docs", [])
        web_docs = input_dict.get("web_docs", [])

        all_docs = []
        for doc in db_docs:
            doc.metadata = doc.metadata or {}
            doc.metadata["source_type"] = "internal"
            all_docs.append(doc)

        for doc in web_docs:
            doc.metadata = doc.metadata or {}
            doc.metadata["source_type"] = "web"
            all_docs.append(doc)

        context_parts = []
        for doc in all_docs:
            source_type = doc.metadata.get("source_type", "unknown")
            source_info = doc.metadata.get("source", doc.metadata.get("url", "Unknown source"))
            context_parts.append(f"[{source_type.upper()} - {source_info}]\n{doc.page_content}")

        merged_context = "\n\n---\n\n".join(context_parts)
        self._log(f"Merged context from {len(all_docs)} total documents")

        return {**input_dict, "context": merged_context, "domain_context": self.domain_context}

    def _build_chain(self) -> Runnable:
        """Build the LangChain Runnable chain for the knowledge agent."""

        retrieval_chain = (
            RunnableLambda(self._retrieve_db_docs)
            | RunnableLambda(self._check_relevance)
            | RunnableLambda(self._conditional_web_search)
            | RunnableLambda(self._merge_context)
        )

        answer_chain = get_knowledge_prompt() | self.llm | StrOutputParser()

        return retrieval_chain | answer_chain
