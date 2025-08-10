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
    def __init__(
        self,
        collection_name: str = config.mongo.COLLECTION_NAME,
        use_existing_collection: bool = False,
        clear_collection_before: bool = False,
    ):
        self.uri = config.mongo.URI
        self.db_name = config.mongo.DB_NAME
        self.collection_name = collection_name
        self.use_existing_collection = use_existing_collection
        self.clear_collection_before = clear_collection_before

        try:
            self.client: MongoClient = MongoClient(self.uri)
            self.db = self.client[self.db_name]
            logger.info(f"Connected to MongoDB Atlas database: {self.db_name}")
        except ConnectionFailure as e:
            logger.error(f"Could not connect to MongoDB: {e}")
            raise

        self.collection = self.get_collection()
        if clear_collection_before:
            logger.info(f"Clearing collection '{self.collection_name}' before running the pipeline.")
            self.collection.delete_many({})

        embedding, embed_dimension = get_embeddings()
        self.vector_store = MongoDBAtlasVectorSearch(
            collection=self.collection,
            embedding=embedding,
            index_name=config.mongo.VECTOR_SRACH_INDEX_NAME,
            relevance_score_fn="cosine",
        )

        self.vector_store.create_vector_search_index(dimensions=embed_dimension)

    def get_collection(self):
        existing_collections = self.db.list_collection_names()
        if self.collection_name in existing_collections:
            logger.info(f"Collection '{self.collection_name}' already exists.")
            if not self.use_existing_collection:
                exit(1)
            else:
                logger.warning(f"Using existing collection '{self.collection_name}'.")
            return self.db[self.collection_name]
        try:
            logger.info(f"Creating collection '{self.collection_name}'.")
            return self.db.create_collection(self.collection_name)
        except errors.CollectionInvalid:
            logger.warning(f"Collection '{self.collection_name}' already exists (race condition).")
            return self.db[self.collection_name]
        except Exception as e:
            logger.error(f"Error creating collection '{self.collection_name}': {e}")
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
            {"$group": {"_id": "$file_path"}},  # Group by file_path in metadata
            {"$project": {"file_path": "$_id", "_id": 0}},  # Project only the file_path field
        ]

        result = self.collection.aggregate(pipeline)
        return [doc["file_path"] for doc in result]
