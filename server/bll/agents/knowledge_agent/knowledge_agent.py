from bll.agents.base_agent import BaseAgent
from bll.agents.contextualizer_agent import ContextualizerAgent
from bll.agents.knowledge_agent.prompts import KNOWLEDGE_SYSTEM_PROMPT
from core.interfaces import BaseRetriever
from core.logger import logger
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough


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
            web_search_retriever: Optional web search retriever
            db_top_k: Number of top documents to retrieve from the database
            web_max_k: Maximum number of web search results to retrieve
            web_supplement_k: Number of web search results to use for supplementation
            verbose: Enable detailed logging
        """
        self.domain_context = domain_context
        self.db_retriever = db_retriever
        self.web_search_retriever = web_search_retriever
        self.db_top_k = db_top_k
        self.web_max_k = web_max_k
        self.web_supplement_k = web_supplement_k

        self.db_docs_min_tolerance = 1

        self.contextualizer = ContextualizerAgent()

        super().__init__(llm, verbose)

    def _transform_input(self, input_dict: dict) -> dict:
        messages: list[AIMessage | HumanMessage] = input_dict["messages"]
        history = "\n".join(
            [("User: " if isinstance(msg, HumanMessage) else "AI: ") + msg.content for msg in messages[:-1]]
        )
        del input_dict["messages"]
        return input_dict | {"prompt": messages[-1].content, "history": history, "domain_context": self.domain_context}

    def _retrieve_db_docs(self, input_dict: dict) -> dict:
        """Retrieve documents from the database."""
        query = input_dict["contextual_prompt"]
        docs = self.db_retriever.retrieve(
            query=query,
            k=self.db_top_k,
        )
        logger.debug(f"Retrieved {len(docs)} docs from database")
        return input_dict | {"db_docs": docs}

    def _conditional_web_search(self, input_dict: dict) -> dict:
        """Conditionally perform web search based on relevance and configuration."""
        web_docs = []

        if self.web_search_retriever:
            query = input_dict["contextual_prompt"]
            db_docs = input_dict["db_docs"]

            if len(db_docs) >= self.db_docs_min_tolerance:
                web_k = min(self.web_supplement_k, self.web_max_k)
            else:
                web_k = self.web_max_k
                logger.debug(f"DB docs not relevant, using {web_k} web results")

            web_docs = self.web_search_retriever.retrieve(query, k=web_k)
            logger.debug(f"Retrieved {len(web_docs)} web docs")

        logger.debug(
            f"Tavily retrieved docs: {[str(doc.metadata['source']) + ' : ' + str(doc.metadata['score']) for doc in web_docs]}"
        )
        return input_dict | {"web_docs": web_docs}

    def _merge_context(self, input_dict: dict) -> dict:
        """Merge all retrieved documents into a single context and extract references."""
        db_docs = input_dict.get("db_docs", [])
        web_docs = input_dict.get("web_docs", [])

        all_docs = []

        for doc in db_docs:
            doc.metadata = doc.metadata or {}
            doc.metadata["type"] = "internal"
            all_docs.append(doc)

        for doc in web_docs:
            doc.metadata = doc.metadata or {}
            doc.metadata["type"] = "web"
            all_docs.append(doc)

        context_parts = []
        for doc in all_docs:
            source_type: str = doc.metadata["type"]
            source_info: str = str(doc.metadata.get("source"))
            if source_type == "internal":
                source_info += f"#page={doc.metadata.get('page_number', 1)}"
            context_parts.append(f"[{source_type.upper()}]\n{doc.page_content}\nReference: {source_info}")

        merged_context: str = "\n\n---\n\n".join(context_parts)

        return input_dict | {
            "context": merged_context,
        }

    def _extract_references(self, input_dict: dict) -> list[str]:
        """Extract references from documents in the state."""
        references = []
        db_docs = input_dict.get("db_docs", [])
        web_docs = input_dict.get("web_docs", [])

        for doc in db_docs:
            source = doc.metadata.get("source", "Unknown internal source")
            if source not in references:
                references.append(source)

        for doc in web_docs:
            url = doc.metadata.get("url") or doc.metadata.get("source", "Unknown web source")
            if url not in references:
                references.append(url)

        return references

    def _build_chain(self):
        chain = (
            RunnableLambda(self._transform_input)
            | RunnableLambda(lambda input: self.contextualizer.chain.invoke(input))
            | RunnableLambda(self._retrieve_db_docs)
            | RunnableLambda(self._conditional_web_search)
            | RunnableLambda(self._merge_context)
            | RunnablePassthrough.assign(
                answer=KNOWLEDGE_SYSTEM_PROMPT | self.llm | StrOutputParser(),
            )
        )
        return chain
