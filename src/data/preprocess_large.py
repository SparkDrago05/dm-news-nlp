from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load YAML config."""
    path = Path(path)
    with path.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def preprocess_business_reorder(config: Dict[str, Any]) -> None:
    """Preprocess the large Business Recorder CSV in chunks and write a cleaned version."""
    data_cfg = config['data']
    sources_cfg = data_cfg['sources']
    src_cfg = sources_cfg['business_reorder']

    raw_dir = Path(data_cfg['raw_dir'])
    interim_dir = Path(data_cfg['interim_dir'])
    interim_dir.mkdir(parents=True, exist_ok=True)

    input_path = raw_dir / src_cfg['filename']
    encoding = src_cfg.get('encoding', 'latin1')
    chunksize = int(data_cfg['large_file']['business_reorder_chunksize'])

    output_path = interim_dir / 'business_reorder_clean.csv'
    if output_path.exists():
        logger.warning('Output file %s already exists and will be overwritten.', output_path)
        output_path.unlink()

    logger.info('Preprocessing Business Recorder from %s', input_path)

    chunk_iter = pd.read_csv(
        input_path,
        encoding=encoding,
        on_bad_lines='skip',
        low_memory=False,
        chunksize=chunksize,
    )

    total_rows = 0
    for idx, chunk in enumerate(chunk_iter, start=1):
        logger.info('Processing chunk %d (%d rows)', idx, len(chunk))

        # Drop unnamed columns
        cols = [c for c in chunk.columns if not c.startswith('Unnamed')]
        chunk = chunk.loc[:, cols]

        # Ensure expected columns
        expected = ['headline', 'date', 'link', 'source', 'categories', 'description']
        missing = [c for c in expected if c not in chunk.columns]
        if missing:
            logger.warning('Chunk %d is missing columns: %s', idx, missing)
            for c in missing:
                chunk[c] = ''

        chunk = chunk[expected]

        # Fill NAs
        chunk = chunk.fillna('')

        # Append to CSV
        mode = 'a'
        header = not output_path.exists()
        chunk.to_csv(output_path, index=False, mode=mode, header=header)
        total_rows += len(chunk)

    logger.info('Finished preprocessing Business Recorder. Total rows written: %d', total_rows)

    # Optional parquet version
    try:
        logger.info('Loading cleaned CSV to write Parquet (may take some time)...')
        df = pd.read_csv(output_path)
        parquet_path = output_path.with_suffix('.parquet')
        df.to_parquet(parquet_path, index=False)
        logger.info('Wrote Parquet to %s', parquet_path)
    except Exception as exc:
        logger.warning('Failed to write Parquet version: %s', exc)


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description='Preprocess large Business Recorder dataset.')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/base.yaml',
        help='Path to YAML config file.',
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    preprocess_business_reorder(config)


if __name__ == '__main__':
    main()
