from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..features.text_cleaning import build_preprocessor
from ..utils import get_project_root, _resolve_path


@dataclass
class RetrievalIndex:
    """Serialized retrieval index with TF-IDF representations and metadata."""

    vectorizer: TfidfVectorizer
    matrix: Any  # typically scipy.sparse.csr_matrix
    metadata: List[Dict[str, Any]]
    text_columns: List[str]
    join_with: str


def retrieval_enabled(cfg: Dict[str, Any]) -> bool:
    """Return True if retrieval is enabled in config."""
    return bool(cfg.get('retrieval', {}).get('enabled', False))


def _build_corpus(df, text_cols: List[str], join_with: str) -> List[str]:
    """Join configured text columns."""
    text_df = df[text_cols].fillna('')
    return text_df.apply(lambda row: join_with.join(map(str, row)), axis=1).tolist()


def build_retrieval_index(df, cfg: Dict[str, Any]) -> RetrievalIndex:
    """Create a TF-IDF retrieval index that can be reused offline."""
    retrieval_cfg = cfg.get('retrieval', {})
    if not retrieval_cfg.get('enabled', False):
        raise ValueError('Retrieval is disabled in the config.')

    text_cols = retrieval_cfg.get('text_columns') or cfg['preprocessing']['text_columns']
    join_with = retrieval_cfg.get('join_with', cfg['preprocessing'].get('join_with', ' '))
    vectorizer_cfg = retrieval_cfg.get('vectorizer', {})

    prep_cfg = cfg['preprocessing']['text_cleaning']
    preprocessor = build_preprocessor(prep_cfg)

    ngram = vectorizer_cfg.get('ngram_range', [1, 2])
    ngram_range = (int(ngram[0]), int(ngram[1]))
    max_features = vectorizer_cfg.get('max_features', None)
    stop_words = 'english' if vectorizer_cfg.get('remove_stopwords', True) else None

    vectorizer = TfidfVectorizer(
        preprocessor=preprocessor,
        ngram_range=ngram_range,
        max_features=max_features,
        stop_words=stop_words,
    )

    corpus = _build_corpus(df, text_cols, join_with)
    matrix = vectorizer.fit_transform(corpus)

    metadata_cols = [
        'headline',
        'description',
        'source',
        'link',
        'date',
        'categories',
        'broad_category',
    ]
    use_cols = [c for c in metadata_cols if c in df.columns]
    metadata = df[use_cols].fillna('').to_dict('records')
    for record, combined in zip(metadata, corpus):
        record['combined_text'] = combined

    return RetrievalIndex(
        vectorizer=vectorizer,
        matrix=matrix,
        metadata=metadata,
        text_columns=list(text_cols),
        join_with=join_with,
    )


def _get_index_path(cfg: Dict[str, Any], root: Optional[str | Path] = None) -> Path:
    root_path = Path(root) if root is not None else get_project_root()
    retrieval_cfg = cfg.get('retrieval', {})
    artifacts = retrieval_cfg.get('artifacts', {})
    dir_path = _resolve_path(artifacts.get('index_dir', 'data/processed/retrieval_index'), root_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    filename = artifacts.get('filename', 'retrieval_index.joblib')
    return dir_path / filename


def save_retrieval_index(index: RetrievalIndex, cfg: Dict[str, Any], root: Optional[str | Path] = None) -> Path:
    """Persist retrieval index to disk."""
    path = _get_index_path(cfg, root=root)
    joblib.dump(index, path)
    return path


def load_retrieval_index(cfg: Dict[str, Any], root: Optional[str | Path] = None) -> RetrievalIndex:
    """Load retrieval index from disk."""
    path = _get_index_path(cfg, root=root)
    return joblib.load(path)


def retrieve_similar_articles(
    text: str,
    index: RetrievalIndex,
    *,
    top_k: int = 3,
    min_similarity: float = 0.0,
) -> List[Dict[str, Any]]:
    """Return top-k similar articles from the retrieval index."""
    if not text or not str(text).strip():
        return []

    query = index.vectorizer.transform([text])
    if query.nnz == 0:
        return []

    scores = cosine_similarity(query, index.matrix).ravel()
    if scores.size == 0:
        return []

    order = np.argsort(scores)[::-1]
    results: List[Dict[str, Any]] = []

    for idx in order:
        score = float(scores[idx])
        if score < min_similarity:
            break
        record = dict(index.metadata[idx])
        record['score'] = score
        results.append(record)
        if len(results) >= top_k:
            break

    return results
