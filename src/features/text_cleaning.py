from __future__ import annotations

import re
from functools import partial
from typing import Dict

try:
    import nltk
    from nltk.corpus import stopwords

    _NLTK_AVAILABLE = True
except ImportError:
    _NLTK_AVAILABLE = False

_PUNCT_RE = re.compile(r'[^\w\s]')
_DIGIT_RE = re.compile(r'\d+')


def _get_stopwords() -> set[str]:
    """Return a set of English stopwords, if NLTK is available."""
    if not _NLTK_AVAILABLE:
        return set()
    try:
        _ = stopwords.words('english')
    except LookupError:
        import nltk as _nltk

        _nltk.download('stopwords')
    return set(stopwords.words('english'))


_STOPWORDS = _get_stopwords()


def clean_text(text: str, cfg: Dict) -> str:
    """Clean a single text string based on configuration."""
    if not isinstance(text, str):
        return ''

    lower = cfg.get('lower', True)
    remove_punct = cfg.get('remove_punctuation', True)
    remove_numbers = cfg.get('remove_numbers', True)
    remove_stopwords = cfg.get('remove_stopwords', True)
    lemmatize = cfg.get('lemmatize', False)

    if lower:
        text = text.lower()

    if remove_punct:
        text = _PUNCT_RE.sub(' ', text)

    if remove_numbers:
        text = _DIGIT_RE.sub(' ', text)

    # Token-based processing
    tokens = text.split()

    if remove_stopwords and _STOPWORDS:
        tokens = [t for t in tokens if t not in _STOPWORDS]

    if lemmatize and _NLTK_AVAILABLE:
        try:
            from nltk.stem import WordNetLemmatizer
        except LookupError:
            import nltk as _nltk

            _nltk.download('wordnet')
            from nltk.stem import WordNetLemmatizer

        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return ' '.join(tokens)


def _preprocessor(text: str, *, cfg: Dict) -> str:
    """Pickle-safe preprocessor wrapper for sklearn vectorizers."""
    return clean_text(text, cfg)


def build_preprocessor(cfg: Dict):
    """Return a pickle-safe callable suitable for sklearn Vectorizer.preprocessor."""
    return partial(_preprocessor, cfg=cfg)
