import os
import sys

from bson import ObjectId
from pymongo import MongoClient
from pymongo.collection import Collection

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "server")))
from chunk_gatherer.chunk_gatherer import ChunkGatherer
from core.config.config import Config


class MongoDB:
    _client: MongoClient | None = None

    @classmethod
    def get_client(cls) -> MongoClient:
        if cls._client is None:
            cls._client = MongoClient(Config.mongo.URI, connect=True)
        return cls._client

    @classmethod
    def get_collection(cls, db_name: str, collection_name: str) -> Collection:
        client = cls.get_client()
        return client[db_name][collection_name]

    @classmethod
    def close(cls):
        if cls._client is not None:
            cls._client.close()
            cls._client = None


class DBGatherer(ChunkGatherer):
    def __init__(
        self,
        collection_name: str,
        vector_index_name: str = "vector_index",
        similarity_threshold: float = 0.7,
        top_k: int = 7,
        page_content_key: str = "text",
        embedding_key: str = "embedding",
    ):
        self.collection = MongoDB.get_collection(Config.mongo.DB_NAME, collection_name)
        self.vector_index_name = vector_index_name
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self.page_content_key = page_content_key
        self.embedding_key = embedding_key

    def close(self):
        MongoDB.close()

    def gather(
        self,
        key: str,
    ) -> str:
        try:
            object_id = ObjectId(key)
            projection = {self.page_content_key: 1, "_id": 0}
            doc = self.collection.find_one({"_id": object_id}, projection)

            if doc and self.page_content_key in doc:
                return doc[self.page_content_key]
            else:
                return ""

        except Exception as e:
            print(f"Error gathering document with id {id}: {e}")
            return ""
