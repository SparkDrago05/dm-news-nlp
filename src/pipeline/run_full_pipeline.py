from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Any

import yaml
import pandas as pd

from ..data.load import load_all_sources, add_broad_category
from ..models.classifier import load_classifier, prepare_text_and_labels
from ..models.summarizer import build_summarizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load YAML config file."""
    path = Path(path)
    with path.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main() -> None:
    """Run end-to-end pipeline: classify and summarize some sample articles."""
    parser = argparse.ArgumentParser(description='Run full pipeline on sample articles.')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/base.yaml',
        help='Path to YAML config.',
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=5,
        help='Number of random samples to run.',
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    df = load_all_sources(cfg)
    df = add_broad_category(df, cfg)

    model = load_classifier(cfg)
    summarizer = build_summarizer(cfg)

    # Sample rows
    sample = df.sample(n=args.num_samples, random_state=cfg['project']['random_seed']).reset_index(drop=True)

    text_cols = cfg['preprocessing']['text_columns']
    join_with = cfg['preprocessing'].get('join_with', ' ')
    text_df = sample[text_cols].fillna('')
    texts = text_df.apply(lambda row: join_with.join(map(str, row)), axis=1).values

    preds = model.predict(texts)

    for idx, (orig_row, text, pred_cat) in enumerate(zip(sample.to_dict('records'), texts, preds), start=1):
        logger.info('--- SAMPLE %d ---', idx)
        logger.info('Headline: %s', orig_row.get('headline', '')[:200])
        logger.info('True broad_category: %s', orig_row.get('broad_category', 'N/A'))
        logger.info('Predicted category: %s', pred_cat)

        summary = summarizer.summarize(orig_row.get('description', ''), category=pred_cat)
        logger.info('Original description (first 400 chars): %s', orig_row.get('description', '')[:400])
        logger.info('Summary (first 400 chars): %s', summary[:400])


if __name__ == '__main__':
    main()
