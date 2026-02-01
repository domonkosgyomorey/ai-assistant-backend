import math
from typing import Dict, List

import numpy as np
from cluster_descriptor.interfaces import ClusterDescriptor
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from utils.common_types import CustomDocument

ALGORITHM_NAME = "tfidf_discriminative"


class TfidfDiscriminativeDescriptor(ClusterDescriptor):
    def __init__(
        self,
        max_features: int = 1000,
        top_terms: int = 10,
        max_df: float = 0.8,
        cluster_frequency_penalty: float = 0.5,
        stop_words: List[str] = ENGLISH_STOP_WORDS,
    ):
        self.max_features = max_features
        self.top_terms = top_terms
        self.max_df = max_df
        self.cluster_frequency_penalty = cluster_frequency_penalty
        self.stop_words = stop_words

        self._tfidf_matrix = None
        self._feature_names = None
        self._doc_labels: np.ndarray = np.array([])

    def describe(self, clustered_docs: dict[str, list[CustomDocument]]) -> Dict[str, str]:
        self._tfidf_matrix = None
        self._feature_names = None
        self._doc_labels = np.array([])

        number_of_docs = 0
        for _, docs in clustered_docs.items():
            number_of_docs += len(docs)
        avg_docs_per_cluster = number_of_docs / len(clustered_docs.keys())

        min_df = max(2, math.floor(0.1 * avg_docs_per_cluster))
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words=self.stop_words,
            ngram_range=(2, 3),
            min_df=min_df,
            max_df=self.max_df,
            token_pattern=r"\b[a-zA-Z]{3,}\b",
        )

        all_docs, doc_labels = self._prepare_documents(clustered_docs)
        self._fit_tfidf(all_docs, doc_labels)

        descriptions = {}
        for cluster_id in clustered_docs.keys():
            doc_count = len(clustered_docs[cluster_id])
            discriminative_terms = self._find_discriminative_terms(cluster_id)
            description = self._create_description(discriminative_terms, doc_count)
            descriptions[cluster_id] = description

        return descriptions

    def _prepare_documents(self, clustered_docs: Dict[str, List]) -> tuple[List[str], List[str]]:
        all_docs = []
        doc_labels = []

        for cluster_id, docs in clustered_docs.items():
            for doc in docs:
                if doc["content"].strip():
                    all_docs.append(doc["content"].strip())
                    doc_labels.append(cluster_id)

        return all_docs, doc_labels

    def _fit_tfidf(self, all_docs: List[str], doc_labels: List[str]) -> None:
        try:
            self._tfidf_matrix = self.vectorizer.fit_transform(all_docs)
            self._feature_names = self.vectorizer.get_feature_names_out()
            self._doc_labels = np.array(doc_labels)
        except ValueError as e:
            print(f"Warning: TF-IDF fitting failed: {e}")
            self._tfidf_matrix = None
            self._feature_names = None
            self._doc_labels = np.array([])

    def _find_discriminative_terms(self, cluster_id: str) -> List[tuple[str, float]]:
        if self._tfidf_matrix is None or self._feature_names is None:
            return []

        cluster_mask = self._doc_labels == cluster_id
        other_mask = ~cluster_mask

        if not np.any(cluster_mask) or not np.any(other_mask):
            return []

        # Mean TF-IDF for cluster and others
        cluster_tfidf = self._tfidf_matrix[cluster_mask].mean(axis=0).A1
        other_tfidf = self._tfidf_matrix[other_mask].mean(axis=0).A1
        discrimination_scores = cluster_tfidf - other_tfidf

        # Apply frequency penalty to reduce common terms
        cluster_means = np.array(
            [self._tfidf_matrix[self._doc_labels == idx].mean(axis=0).A1 for idx in np.unique(self._doc_labels)]
        )
        cluster_counts = (cluster_means > 0.01).sum(axis=0)
        frequency_penalty = (cluster_counts / cluster_means.shape[0]) * self.cluster_frequency_penalty
        discrimination_scores -= frequency_penalty

        # Get top terms with positive score
        top_indices = np.argsort(discrimination_scores)[-self.top_terms :][::-1]
        return [
            (self._feature_names[idx], discrimination_scores[idx])
            for idx in top_indices
            if discrimination_scores[idx] > 0
        ]

    def _create_description(
        self,
        discriminative_terms: List[tuple[str, float]],
        doc_count: int,
        max_terms_in_description: int = 3,
    ) -> str:
        if not discriminative_terms:
            return f"Cluster with {doc_count} documents (no distinctive terms)"

        terms = [term for term, _ in discriminative_terms[:max_terms_in_description]]

        if len(terms) == 1:
            description = f"Documents distinguished by '{terms[0]}'"
        else:
            joined_terms = "', '".join(terms[:-1])
            description = f"Documents distinguished by '{joined_terms}' and '{terms[-1]}'"

        return description
