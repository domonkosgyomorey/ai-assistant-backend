import pickle
from typing import Optional

import pandas as pd
from giskard.llm.client.base import LLMClient
from giskard.llm.embeddings.base import BaseEmbedding
from giskard.rag.testset_generation import KnowledgeBase


class PersistentKnowledgeBase:
    def __init__(
        self,
        data: pd.DataFrame,
        columns: list[str],
        llm_client: LLMClient,
        embedding_model: BaseEmbedding,
        cache_path: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the PersistentKnowledgeBase.

        Args:
            data: DataFrame containing the knowledge base data
            columns: List of columns to use from the DataFrame
            llm_client: LLM client for question generation
            embedding_model: Embedding model for vector operations
            cache_path: Path to save/load the knowledge base cache
            **kwargs: Additional arguments passed to KnowledgeBase
        """
        self.data = data
        self.columns = columns
        self.llm_client = llm_client
        self.embedding_model = embedding_model
        self.cache_path = cache_path
        self.kwargs = kwargs
        self._knowledge_base: Optional[KnowledgeBase] = None

    def save_to_cache(self, file_path: str) -> None:
        if not self._knowledge_base:
            return

        try:
            savable_data = self._knowledge_base.get_savable_data()

            cache_data = {
                "data_shape": self.data.shape,
                "columns": self.columns,
                "kwargs": self.kwargs,
                "data_csv": self.data.to_csv(index=False),
                "giskard_savable_data": savable_data,
                "topics": dict(self._knowledge_base.topics),
                "computed_embeddings": None,
                "computed_reduced_embeddings": None,
                "computed_documents": None,
            }

            try:
                if hasattr(self._knowledge_base, "_embeddings_inst"):
                    embeddings = getattr(self._knowledge_base, "_embeddings_inst", None)
                    if embeddings is not None:
                        cache_data["computed_embeddings"] = embeddings

                if hasattr(self._knowledge_base, "_reduced_embeddings_inst"):
                    reduced = getattr(self._knowledge_base, "_reduced_embeddings_inst", None)
                    if reduced is not None:
                        cache_data["computed_reduced_embeddings"] = reduced

                if hasattr(self._knowledge_base, "_documents"):
                    docs = getattr(self._knowledge_base, "_documents", None)
                    if docs is not None:
                        # Convert to serializable format
                        cache_data["computed_documents"] = [
                            {"content": doc.content, "metadata": dict(doc.metadata) if hasattr(doc, "metadata") else {}}
                            for doc in docs
                        ]
            except Exception as e:
                print(f"Could not extract some computed data: {e}")

            with open(file_path, "wb") as f:
                pickle.dump(cache_data, f)
        except Exception as e:
            print(f"Warning: Could not save computed state: {e}")
            return

    def _restore_computed_state(self, cache_data: dict) -> bool:
        try:
            # Create a fresh knowledge base with minimal computation
            print("Creating KnowledgeBase with cached computed state...")

            self._knowledge_base = KnowledgeBase(
                self.data, self.columns, llm_client=self.llm_client, embedding_model=self.embedding_model, **self.kwargs
            )

            # Now restore the computed state BEFORE accessing .topics
            savable_data = cache_data.get("giskard_savable_data", {})

            # Restore the internal computed attributes
            components_restored = []

            if hasattr(self._knowledge_base, "_topics_inst") and "topics" in savable_data:
                self._knowledge_base._topics_inst = savable_data["topics"]
                components_restored.append("topics")

            if hasattr(self._knowledge_base, "_documents_topics") and "documents_topics" in savable_data:
                self._knowledge_base._documents_topics = savable_data["documents_topics"]
                components_restored.append("document mappings")

            # Restore embeddings if available
            if cache_data.get("computed_embeddings") is not None:
                self._knowledge_base._embeddings_inst = cache_data["computed_embeddings"]
                components_restored.append("embeddings")

            if cache_data.get("computed_reduced_embeddings") is not None:
                self._knowledge_base._reduced_embeddings_inst = cache_data["computed_reduced_embeddings"]
                components_restored.append("reduced embeddings")

            # Restore documents if available
            if cache_data.get("computed_documents") is not None:
                from types import SimpleNamespace

                docs = []
                for doc_data in cache_data["computed_documents"]:
                    doc = SimpleNamespace()
                    doc.content = doc_data["content"]
                    doc.metadata = doc_data["metadata"]
                    docs.append(doc)
                self._knowledge_base._documents = docs
                components_restored.append("documents")

            print(f"Restored components: {', '.join(components_restored)}")
            return True

        except Exception as e:
            print(f"Failed to restore computed state: {e}")
            self._knowledge_base = None
            return False

    def load_from_cache(self, cache_file_path: str) -> bool:
        try:
            with open(cache_file_path, "rb") as f:
                cache_data = pickle.load(f)

            # Validate that the current data matches the cached data
            if (
                "data_csv" in cache_data
                and cache_data["data_csv"] == self.data.to_csv(index=False)
                and cache_data["columns"] == self.columns
                and cache_data["kwargs"] == self.kwargs
            ):
                topics = cache_data.get("topics", {})
                print(f"Cache found: {len(topics)} topics")

                return self._restore_computed_state(cache_data)
            else:
                print("Cache invalid: data has changed")
                return False

        except Exception as e:
            print(f"Cache error: {e}")
            return False

    def get_knowledge_base(self) -> KnowledgeBase:
        if self._knowledge_base is not None:
            return self._knowledge_base

        print("Creating fresh knowledge base (this may take some time)...")
        self._knowledge_base = KnowledgeBase(
            self.data, self.columns, llm_client=self.llm_client, embedding_model=self.embedding_model, **self.kwargs
        )

        print(f"Knowledge base created with {len(self._knowledge_base.topics)} topics")
        return self._knowledge_base

    @property
    def topics(self):
        """Get topics from the knowledge base."""
        kb = self.get_knowledge_base()
        return kb.topics

    def __getattr__(self, name):
        """Delegate attribute access to the underlying KnowledgeBase."""
        kb = self.get_knowledge_base()
        return getattr(kb, name)
