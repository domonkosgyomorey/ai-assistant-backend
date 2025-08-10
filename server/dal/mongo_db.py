from typing import Any

from core.components import get_embedding
from core.config import config
from core.logger import logger
from langchain.schema import Document
from langchain_core.retrievers import RetrieverLike
from langchain_core.runnables import RunnableConfig
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database


class MongoDB:
    client: MongoClient = MongoClient(config.mongo.URI, connect=True)

    @classmethod
    def close(cls) -> None:
        """Close the MongoDB connection."""
        cls.client.close()


def get_db_client(db_name: str) -> Database:
    if db_name in MongoDB.client.list_database_names():
        return MongoDB.client[db_name]
    else:
        logger.error(f"Database with '{db_name}' does not exists!")
        raise Exception("Database with '{db_name}' does not exists!")


def get_collection(db_name: str, collection_name: str) -> Collection:
    db = get_db_client(db_name)
    if collection_name in db.list_collection_names():
        return db[collection_name]
    else:
        logger.error(f"Collection with '{db_name}' name does not exists in the {db_name} database!")
        raise Exception("Collection with '{db_name}' name does not exists in the {db_name} database!")


class MongoRetriever(RetrieverLike):
    def __init__(self, collection_name: str, top_k: int):
        super().__init__()
        self.top_k = top_k
        self.retriever = self.get_retriever(collection_name)

    def get_relevant_documents(self, query: str) -> list[Document]:
        return self.retriever.similarity_search(query, k=self.top_k)

    def invoke(self, input: str, config: RunnableConfig | None = None, **kwargs: Any) -> list[Document]:
        return self.retriever.similarity_search(input, k=self.top_k)

    def get_retriever(self, collection_name: str) -> MongoDBAtlasVectorSearch:
        return MongoDBAtlasVectorSearch(
            collection=get_collection(config.mongo.DB_NAME, collection_name),
            embedding=get_embedding(),
            index_name=config.mongo.VECTOR_SRACH_INDEX_NAME,
            relevance_score_fn=config.mongo.RELEVANCE_SCORE_FN,
        )
