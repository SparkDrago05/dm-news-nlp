from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

from ..data.load import add_broad_category, load_all_sources, load_yaml
from ..retrieval.indexer import build_retrieval_index, retrieval_enabled, save_retrieval_index

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build TF-IDF retrieval index for news improver.')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/base.yaml',
        help='Path to YAML config file.',
    )
    parser.add_argument(
        '--root',
        type=str,
        default=None,
        help='Optional project root override.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    if not retrieval_enabled(cfg):
        logger.warning('Retrieval disabled in %s. Enable retrieval.enabled before building.', args.config)
        return

    root_path: Optional[Path] = Path(args.root).resolve() if args.root else None

    logger.info('Loading dataset to build retrieval index...')
    df = load_all_sources(cfg, root=root_path)
    df = add_broad_category(df, cfg, root=root_path)
    logger.info('Dataset size for retrieval index: %s', df.shape)

    index = build_retrieval_index(df, cfg)
    path = save_retrieval_index(index, cfg, root=root_path)
    logger.info('Saved retrieval index to %s', path)


if __name__ == '__main__':
    main()
