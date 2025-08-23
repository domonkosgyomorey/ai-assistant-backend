from core.config.config import config
from core.utils.components import get_llm
from dal.mongo_db import MongoRetriever
from dal.tavily_websearch import TavilyRetriever

from bll.agents.knowledge_agent.knowledge_agent import KnowledgeAgent


class Knowledge(KnowledgeAgent):
    def __init__(self):
        super().__init__(
            llm=get_llm(),
            domain_context="University of Obuda, student administration, graduate programm and etc.",
            db_retriever=MongoRetriever(config.mongo.COLLECTION_NAME),
            web_search_retriever=TavilyRetriever(
                include_domains=[
                    "uni-obuda.hu",
                    "uni-obuda.hu/en",
                    "nik.uni-obuda.hu",
                    "neptun.uni-obuda.hu",
                    "en.wikipedia.org",
                    "hu.wikipedia.org",
                ],
            ),
            db_top_k=6,
            web_max_k=3,
            web_supplement_k=2,
            verbose=True,
        )
