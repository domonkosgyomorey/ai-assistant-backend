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
        pipeline = [
            {"$addFields": {"relevance_score": {"$meta": "vectorSearchScore"}}},
            {"$match": {"relevance_score": {"$gte": self.relevance_tolerance}}},
        ]
        self.get_all_similarity_scores(query)
        return self.vector_store.similarity_search(query, k=k, post_filter_pipeline=pipeline)

    def retrieve_with_scores(self, query: str, k: int = 5, **kwargs) -> list[tuple[Document, float]]:
        """Retrieve documents with their similarity scores."""
        pipeline = [
            {"$addFields": {"relevance_score": {"$meta": "vectorSearchScore"}}},
            {"$match": {"relevance_score": {"$gte": self.relevance_tolerance}}},
        ]

        return self.vector_store.similarity_search_with_score(query, k=k, post_filter_pipeline=pipeline)

    def retrieve_all_with_scores(
        self, query: str, score_threshold: float = None, **kwargs
    ) -> list[tuple[Document, float]]:
        """Retrieve ALL documents with similarity scores above threshold, not limited by k."""
        threshold = score_threshold if score_threshold is not None else self.relevance_tolerance

        large_k = kwargs.get("max_results", 10000)  # Adjust based on your collection size

        pipeline = [
            {"$addFields": {"relevance_score": {"$meta": "vectorSearchScore"}}},
            {"$match": {"relevance_score": {"$gte": threshold}}},
        ]
        return self.vector_store.similarity_search_with_score(query, k=large_k, post_filter_pipeline=pipeline)

    def get_all_similarity_scores(self, query: str) -> list[dict]:
        """Get all similarity scores with document IDs."""
        import time

        t0 = time.perf_counter()
        embedding_vector = self.vector_store._embedding.embed_query(query)
        pipeline = [
            {
                "$vectorSearch": {
                    "index": config.mongo.VECTOR_SRACH_INDEX_NAME,
                    "path": "embedding",
                    "queryVector": embedding_vector,
                    "numCandidates": 10000,
                    "limit": 10000,
                }
            },
            {"$addFields": {"similarity_score": {"$meta": "vectorSearchScore"}}},
            {"$project": {"_id": 1, "similarity_score": 1}},
            {"$sort": {"similarity_score": -1}},
        ]

        collection = get_collection(config.mongo.DB_NAME, self.collection_name)
        return list(collection.aggregate(pipeline))

    def get_retriever(self, collection_name: str) -> MongoDBAtlasVectorSearch:
        """Create MongoDB Atlas Vector Search instance."""
        return MongoDBAtlasVectorSearch(
            collection=get_collection(config.mongo.DB_NAME, collection_name),
            embedding=get_embedding(),
            index_name=config.mongo.VECTOR_SRACH_INDEX_NAME,
            relevance_score_fn=config.mongo.RELEVANCE_SCORE_FN,
        )
