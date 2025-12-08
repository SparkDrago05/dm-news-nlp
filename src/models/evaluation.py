from __future__ import annotations

from typing import Dict, List, Optional, Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

try:
    from rouge_score import rouge_scorer

    _ROUGE_AVAILABLE = True
except ImportError:
    _ROUGE_AVAILABLE = False

try:
    import textstat

    _TEXTSTAT_AVAILABLE = True
except ImportError:
    _TEXTSTAT_AVAILABLE = False


def evaluate_classifier(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute standard classification metrics."""
    metrics: Dict[str, Any] = {}
    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    metrics['macro_f1'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))

    if labels is None:
        labels = sorted(list(set(y_true)))

    metrics['classification_report'] = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    metrics['labels'] = labels

    return metrics


def evaluate_summaries(
        originals: List[str],
        summaries: List[str],
        use_rouge: bool = False,
        compute_readability: bool = True,
) -> Dict[str, Any]:
    """Compute basic summarization metrics. ROUGE is optional."""
    if len(originals) != len(summaries):
        raise ValueError('originals and summaries must have the same length')

    n = len(originals)
    length_ratios = []

    for orig, summ in zip(originals, summaries):
        o_len = max(len(orig.split()), 1)
        s_len = max(len(summ.split()), 1)
        length_ratios.append(s_len / o_len)

    metrics: Dict[str, Any] = {
        'num_samples': n,
        'avg_length_ratio': float(np.mean(length_ratios)),
    }

    if compute_readability and _TEXTSTAT_AVAILABLE:
        orig_scores = [textstat.flesch_reading_ease(o) for o in originals]
        summ_scores = [textstat.flesch_reading_ease(s) for s in summaries]
        metrics['orig_readability_mean'] = float(np.mean(orig_scores))
        metrics['summ_readability_mean'] = float(np.mean(summ_scores))

    if use_rouge and _ROUGE_AVAILABLE:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge1_f = []
        rouge2_f = []
        rougeL_f = []

        for orig, summ in zip(originals, summaries):
            scores = scorer.score(orig, summ)
            rouge1_f.append(scores['rouge1'].fmeasure)
            rouge2_f.append(scores['rouge2'].fmeasure)
            rougeL_f.append(scores['rougeL'].fmeasure)

        metrics['rouge1_f'] = float(np.mean(rouge1_f))
        metrics['rouge2_f'] = float(np.mean(rouge2_f))
        metrics['rougeL_f'] = float(np.mean(rougeL_f))

    return metrics
