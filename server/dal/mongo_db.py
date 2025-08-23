from core.config.config import config
from core.interfaces import BaseRetriever
from core.logger import logger
from core.utils.components import get_embedding
from langchain.schema import Document
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


class MongoRetriever(BaseRetriever):
    """MongoDB Atlas Vector Search implementation of BaseRetriever interface."""

    def __init__(self, collection_name: str, top_k: int = 5, relevance_tolerance: float = 0.4):
        super().__init__()
        self.top_k = top_k
        self.collection_name = collection_name
        self.relevance_tolerance = relevance_tolerance
        self.vector_store = self.get_retriever(collection_name)

    def retrieve(self, query: str, k: int = 5, **kwargs) -> list[Document]:
        """Implement BaseRetriever interface method."""
        docs_n_scores = self.vector_store.similarity_search_with_relevance_scores(query, k=k)
        logger.info(
            f"Retrieved from MongoDB: {[doc.metadata['source'] + ' : ' + str(score) for doc, score in docs_n_scores]}"
        )
        return [doc for doc, _ in docs_n_scores if _ > self.relevance_tolerance]

    def get_retriever(self, collection_name: str) -> MongoDBAtlasVectorSearch:
        """Create MongoDB Atlas Vector Search instance."""
        return MongoDBAtlasVectorSearch(
            collection=get_collection(config.mongo.DB_NAME, collection_name),
            embedding=get_embedding(),
            index_name=config.mongo.VECTOR_SRACH_INDEX_NAME,
            relevance_score_fn=config.mongo.RELEVANCE_SCORE_FN,
        )
