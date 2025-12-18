from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from ..data.load import load_yaml, load_all_sources, add_broad_category
from ..models.classifier import load_classifier
from ..models.rewriter import build_rewriter
from ..models.summarizer import build_summarizer
from ..retrieval.indexer import load_retrieval_index, retrieve_similar_articles, retrieval_enabled

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
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
    parser.add_argument(
        '--root',
        type=str,
        default=None,
        help='Override project root used to resolve relative data paths (e.g., data/raw).',
    )
    return parser.parse_args()


def main() -> None:
    """Run end-to-end pipeline: classify and summarize some sample articles."""
    args = parse_args()

    cfg = load_yaml(args.config)

    root_path = Path(args.root).expanduser().resolve() if args.root else None

    df = load_all_sources(cfg, root=root_path)
    df = add_broad_category(df, cfg, root=root_path)

    model = load_classifier(cfg, root=root_path)
    summarizer = build_summarizer(cfg)
    rewriter = build_rewriter(cfg)

    retrieval_index: Optional[Any] = None
    retrieval_cfg = cfg.get('retrieval', {})
    if retrieval_enabled(cfg):
        try:
            retrieval_index = load_retrieval_index(cfg, root=root_path)
            logger.info('Loaded retrieval index for contextual expansion.')
        except FileNotFoundError:
            logger.warning(
                'Retrieval index not found. Run build_retrieval_index.py --config %s to enable context-aware rewrites.',
                args.config,
            )

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

        retrieved = []
        if retrieval_index is not None:
            retrieved = retrieve_similar_articles(
                text,
                retrieval_index,
                top_k=retrieval_cfg.get('top_k', 3),
                min_similarity=retrieval_cfg.get('min_similarity', 0.0),
            )

        rewrite_result = rewriter.rewrite(
            headline=orig_row.get('headline', ''),
            description=orig_row.get('description', ''),
            category=pred_cat,
            retrieved=retrieved,
        )

        logger.info('Improved article:\n%s', rewrite_result.compose_text())
        if retrieved:
            logger.info('Context articles used:')
            for ctx in retrieved:
                logger.info(
                    ' - [%s | score=%.3f] %s',
                    ctx.get('source', 'source'),
                    ctx.get('score', 0.0),
                    ctx.get('headline', '')[:160],
                )


if __name__ == '__main__':
    main()
