from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Any, List

import math

try:
    from transformers import pipeline as hf_pipeline

    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False


@dataclass
class SummarizerConfig:
    """Configuration for summarizer."""
    summarizer_type: str
    params: Dict[str, Any]


class BaseSummarizer:
    """Base class summarizer interface."""

    def summarize(self, text: str, category: Optional[str] = None) -> str:
        """Return a summary or improved version of text."""
        raise NotImplementedError


class NoOpSummarizer(BaseSummarizer):
    """Return original text (no summarization)."""

    def summarize(self, text: str, category: Optional[str] = None) -> str:
        return text


class ExtractiveSummarizer(BaseSummarizer):
    """Very simple extractive summarizer based on word frequency."""

    def __init__(self, max_sentences: int = 3) -> None:
        self.max_sentences = max_sentences

    def summarize(self, text: str, category: Optional[str] = None) -> str:
        sentences = self._split_sentences(text)
        if len(sentences) <= self.max_sentences:
            return text

        word_freq = self._compute_word_frequencies(text)
        scored = []
        for i, sent in enumerate(sentences):
            score = sum(word_freq.get(w.lower(), 0.0) for w in sent.split())
            scored.append((score, i, sent))

        scored.sort(reverse=True, key=lambda x: x[0])
        top = scored[: self.max_sentences]
        top_sorted = sorted(top, key=lambda x: x[1])

        summary = ' '.join(s for _, _, s in top_sorted)
        return summary

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Naive sentence split using punctuation."""
        raw = text.replace('\n', ' ')
        parts = []
        current = []
        for ch in raw:
            current.append(ch)
            if ch in '.!?':
                s = ''.join(current).strip()
                if s:
                    parts.append(s)
                current = []
        tail = ''.join(current).strip()
        if tail:
            parts.append(tail)
        return parts

    @staticmethod
    def _compute_word_frequencies(text: str) -> Dict[str, float]:
        """Compute simple term frequencies."""
        text = text.lower()
        tokens = [t for t in text.split() if t.isalpha() and len(t) > 2]
        freq: Dict[str, float] = {}
        for t in tokens:
            freq[t] = freq.get(t, 0.0) + 1.0
        if not freq:
            return freq
        max_freq = max(freq.values())
        for t in list(freq.keys()):
            freq[t] = freq[t] / max_freq
        return freq


class TransformersSummarizer(BaseSummarizer):
    """Wrapper around HuggingFace transformers summarization pipeline."""

    def __init__(
            self,
            model_name: str,
            max_length: int,
            min_length: int,
            device: Optional[int] = None,
    ) -> None:
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError(
                'transformers is not installed. Install with `pip install transformers`.',
            )
        self.model_name = model_name
        self.max_length = max_length
        self.min_length = min_length

        self.pipe = hf_pipeline(
            'summarization',
            model=model_name,
            device=device if device is not None else -1,
        )

    def summarize(self, text: str, category: Optional[str] = None) -> str:
        if not text or not text.strip():
            return text

        prefix = ''
        if category:
            prefix = f'[Category: {category}] '

        input_text = prefix + text
        # Some models have max input length; we can truncate
        max_input_len = 1024
        if len(input_text.split()) > max_input_len:
            tokens = input_text.split()
            input_text = ' '.join(tokens[:max_input_len])

        out = self.pipe(
            input_text,
            max_length=self.max_length,
            min_length=self.min_length,
            truncation=True,
        )
        if isinstance(out, list) and out:
            return out[0].get('summary_text', text)
        return text


def build_summarizer(cfg: Dict) -> BaseSummarizer:
    """Build a summarizer object based on config."""
    sum_cfg = cfg['summarizer']
    s_type = sum_cfg.get('type', 'none')

    if s_type == 'none':
        return NoOpSummarizer()

    if s_type == 'extractive':
        params = sum_cfg.get('extractive', {})
        max_sentences = int(params.get('max_sentences', 3))
        return ExtractiveSummarizer(max_sentences=max_sentences)

    if s_type == 't5':
        params = sum_cfg.get('t5', {})
        return TransformersSummarizer(
            model_name=params.get('model_name', 't5-small'),
            max_length=int(params.get('max_length', 120)),
            min_length=int(params.get('min_length', 40)),
        )

    if s_type == 'bart':
        params = sum_cfg.get('bart', {})
        return TransformersSummarizer(
            model_name=params.get('model_name', 'facebook/bart-large-cnn'),
            max_length=int(params.get('max_length', 120)),
            min_length=int(params.get('min_length', 40)),
        )

    raise ValueError(f'Unknown summarizer type: {s_type}')
