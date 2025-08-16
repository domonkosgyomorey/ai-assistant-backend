from core.config.config import config
from core.utils.components import get_llm
from dal.mongo_db import MongoRetriever

from bll.agents import KnowledgeAgent
from bll.services.relevance_service import create_structured_relevance_checker


class NIKObudaKnowledgeAgent(KnowledgeAgent):
    def __init__(self):
        super().__init__(
            get_llm(),
            "University of Obuda where you can find information about administration students right about the exams and etc.",
            MongoRetriever(config.mongo.COLLECTION_NAME),
            create_structured_relevance_checker(get_llm()),
            verbose=True,
        )
