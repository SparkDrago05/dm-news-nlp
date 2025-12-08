from __future__ import annotations

from typing import Dict, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from .text_cleaning import build_preprocessor


def build_vectorizer(cfg: Dict) -> TfidfVectorizer | CountVectorizer:
    """Build a TF-IDF or Bag-of-Words vectorizer from config."""
    vec_cfg = cfg['vectorizer']
    prep_cfg = cfg['preprocessing']['text_cleaning']

    vec_type = vec_cfg.get('type', 'tfidf')
    ngram_range_list = vec_cfg.get('ngram_range', [1, 1])
    ngram_range: Tuple[int, int] = (int(ngram_range_list[0]), int(ngram_range_list[1]))
    max_features = vec_cfg.get('max_features', None)

    preprocessor = build_preprocessor(prep_cfg)

    common_kwargs = {
        'preprocessor': preprocessor,
        'ngram_range': ngram_range,
        'max_features': max_features,
    }

    if prep_cfg.get('remove_stopwords', True):
        common_kwargs['stop_words'] = 'english'

    if vec_type == 'bow':
        return CountVectorizer(**common_kwargs)
    return TfidfVectorizer(**common_kwargs)
