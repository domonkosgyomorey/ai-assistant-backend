import logging

from langchain.schema import Document
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient, errors
from pymongo.errors import ConnectionFailure

from ingestion.config import config
from ingestion.interfaces import DocumentStore
from ingestion.utils.components import get_embeddings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class MongoAtlasVectorStore(DocumentStore):
    def __init__(self, collection_name: str = config.mongo.COLLECTION_NAME):
        self.uri = config.mongo.URI
        self.db_name = config.mongo.DB_NAME
        self.collection_name = collection_name

        try:
            self.client: MongoClient = MongoClient(self.uri)
            self.db = self.client[self.db_name]
            logger.info(f"Connected to MongoDB Atlas database: {self.db_name}")
        except ConnectionFailure as e:
            logger.error(f"Could not connect to MongoDB: {e}")
            raise

        self.collection = self._get_or_throw_collection(self.collection_name)
        embedding, embed_dimension = get_embeddings()
        self.vector_store = MongoDBAtlasVectorSearch(
            collection=self.collection,
            embedding=embedding,
            index_name=config.mongo.VECTOR_SRACH_INDEX_NAME,
            relevance_score_fn="cosine",
        )

        self.vector_store.create_vector_search_index(dimensions=embed_dimension)

    def _get_or_throw_collection(self, collection_name: str):
        existing_collections = self.db.list_collection_names()
        if collection_name in existing_collections:
            logger.info(f"Collection '{collection_name}' already exists. Please delete before continuing.")
            raise ValueError(f"Collection '{collection_name}' already exists. Please delete before continuing.")
        try:
            logger.info(f"Creating collection '{collection_name}'.")
            return self.db.create_collection(collection_name)
        except errors.CollectionInvalid:
            logger.warning(f"Collection '{collection_name}' already exists (race condition).")
            return self.db[collection_name]
        except Exception as e:
            logger.error(f"Error creating collection '{collection_name}': {e}")
            raise

    def save(self, docs: list[Document]):
        if not docs:
            logger.warning("No documents provided to save.")
            return

        logger.info(f"Saving {len(docs)} documents to MongoDB collection '{self.collection_name}'")
        try:
            self.vector_store.add_documents(docs)
            logger.info(f"Successfully saved {len(docs)} documents to MongoDB collection '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Error saving documents to MongoDB: {e}")
            raise

    def list_source_keys(self) -> list[str]:
        pipeline = [
            {"$group": {"_id": "$metadata.file_path"}},  # Group by file_path in metadata
            {"$project": {"file_path": "$_id", "_id": 0}},  # Project only the file_path field
        ]

        result = self.collection.aggregate(pipeline)
        return [doc["file_path"] for doc in result]
