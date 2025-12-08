from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """Infer project root as two levels above this file: src/data/load.py -> project/"""
    return Path(__file__).resolve().parents[2]


def _resolve_path(p: str | Path, root: Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (root / p)


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    path = Path(path)
    with path.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _drop_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns whose names start with 'Unnamed'."""
    cols = [c for c in df.columns if not c.startswith('Unnamed')]
    return df.loc[:, cols]


def _load_single_source(
        name: str,
        source_cfg: Dict[str, Any],
        data_cfg: Dict[str, Any],
        root: Path,
) -> pd.DataFrame:
    """Load a single news source CSV."""
    raw_dir = _resolve_path(data_cfg['raw_dir'], root)
    filename = source_cfg['filename']
    encoding = source_cfg.get('encoding', 'utf-8')
    path = raw_dir / filename

    logger.info('Loading source %s from %s (encoding=%s)', name, path, encoding)
    df = pd.read_csv(
        path,
        encoding=encoding,
        on_bad_lines='skip',
        low_memory=False,
    )
    df = _drop_unnamed_columns(df)

    # Basic normalization of string columns
    for col in ['source', 'categories', 'headline', 'description']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()  # Removes whitespaces

    # Ensure expected columns exist
    expected = ['headline', 'date', 'link', 'source', 'categories', 'description']
    missing = [c for c in expected if c not in df.columns]
    if missing:
        logger.warning('Source %s is missing columns: %s', name, missing)

    # If 'source' column is missing, set it to this source name
    if 'source' not in df.columns:
        df['source'] = name

    # Keep only relevant columns if they exist
    keep = [c for c in expected if c in df.columns]
    df = df[keep]

    df['__file__'] = name
    return df


def load_all_sources(config: Dict[str, Any], root: str | Path | None = None) -> pd.DataFrame:
    """Load all configured news sources into a single DataFrame."""
    root_path = Path(root) if root is not None else get_project_root()

    data_cfg = config['data']
    sources_cfg = data_cfg['sources']

    all_dfs: List[pd.DataFrame] = []

    for name, source_cfg in sources_cfg.items():
        if name == 'business_reorder':
            # Prefer loading preprocessed version if available
            interim_dir = _resolve_path(data_cfg['interim_dir'], root_path)
            processed_path_parquet = interim_dir / 'business_reorder_clean.parquet'
            processed_path_csv = interim_dir / 'business_reorder_clean.csv'

            if processed_path_parquet.exists():
                logger.info('Loading preprocessed business_reorder from %s', processed_path_parquet)
                df = pd.read_parquet(processed_path_parquet)
            elif processed_path_csv.exists():
                logger.info('Loading preprocessed business_reorder from %s', processed_path_csv)
                df = pd.read_csv(processed_path_csv, low_memory=False)
            else:
                logger.warning(
                    'No preprocessed business_reorder file found, loading raw (may be heavy).',
                )
                df = _load_single_source(name, source_cfg, data_cfg, root_path)
        else:
            df = _load_single_source(name, source_cfg, data_cfg, root_path)

        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)

    valid_sources = {
        'Daily Times',
        'Pakistan Today',
        'Tribune',
        'Dawn',
        'Business Recorder',
    }

    # Filter to valid sources only
    before_shape = combined.shape
    combined = combined[combined['source'].isin(valid_sources)].copy()  # Take a copy of the filtered DataFrame and include only valid sources
    logger.info('Filtered invalid sources: %s -> %s', before_shape, combined.shape)

    logger.info('Combined dataset shape: %s', combined.shape)

    # Optional sampling per source
    if data_cfg.get('use_sample', False):
        per_source = int(data_cfg['sample']['per_source'])
        logger.info('Sampling up to %d rows per source (__file__ column).', per_source)
        combined = (
            combined.groupby('__file__', group_keys=False)
            .apply(lambda g: g.sample(min(len(g), per_source), random_state=config['project']['random_seed']))
        )

    return combined


def load_category_mapping(mapping_path: str | Path, root: str | Path | None = None) -> Dict[str, List[str]]:
    """Load category mapping YAML file."""
    root_path = Path(root) if root is not None else get_project_root()
    mapping_path = _resolve_path(mapping_path, root_path)

    mapping = load_yaml(mapping_path)
    # Normalize keys and lists to strings
    norm_mapping: Dict[str, List[str]] = {}
    for broad, raw_list in mapping.items():
        norm_mapping[str(broad)] = [str(x) for x in raw_list]
    return norm_mapping


def add_broad_category(
        df: pd.DataFrame,
        config: Dict[str, Any],
        category_inference: bool = True,
        root: str | Path | None = None
) -> pd.DataFrame:
    """Add a 'broad_category' column to the DataFrame based on mapping and optional heuristics."""
    root_path = Path(root) if root is not None else get_project_root()

    cat_cfg = config['categories']
    mapping_file = cat_cfg['mapping_file']
    unknown_to = cat_cfg.get('unknown_to', 'Other')

    mapping = load_category_mapping(mapping_file, root=root_path)

    # Reverse mapping: raw_category -> broad_category
    raw_to_broad: Dict[str, str] = {}
    for broad, raw_list in mapping.items():
        for raw in raw_list:
            raw_to_broad[raw.lower()] = broad

    def map_category(raw: Optional[str]) -> str:
        if not isinstance(raw, str) or not raw.strip():
            return unknown_to
        key = raw.strip().lower()

        if key in raw_to_broad:
            return raw_to_broad[key]
        # Fuzzy / contains-style matching
        for stored, broad in raw_to_broad.items():
            if stored in key:
                return broad

        return unknown_to

    df = df.copy()

    # Convert characters like & etc which appear &amp; in the data
    df['categories'] = df['categories'].str.replace('&amp;', '&', regex=False)

    # Convert to replace and with &
    df['categories'] = df['categories'].str.replace(' and ', ' & ', regex=False)

    # Convert to title case for better matching
    df['categories'] = df['categories'].str.title()

    df['broad_category'] = df['categories'].apply(map_category)

    if category_inference:
        df = _infer_broad_category_from_text(df, unknown_to)

    return df


def _infer_broad_category_from_text(df: pd.DataFrame, unknown_label: str) -> pd.DataFrame:
    """Apply simple keyword heuristics to fill unknown broad categories."""
    df = df.copy()

    def infer(row: pd.Series) -> str:
        broad = row['broad_category']
        if broad != unknown_label:
            return broad

        text_parts = []
        for col in ['headline', 'description']:
            if col in row and isinstance(row[col], str):
                text_parts.append(row[col])
        text = ' '.join(text_parts).lower()

        if any(k in text for k in ['stock market', 'gdp', 'imf', 'economic', 'economy', 'budget', 'tax']):
            return 'Business'
        if 'world cup' in text or any(k in text for k in ['tournament', 'match', 'innings', 'wicket', 'goal']):
            return 'Sports'
        if any(k in text for k in ['smartphone', 'telecom', 'technology', 'internet', '5g', 'software']):
            return 'Technology'
        if any(k in text for k in ['editorial', 'op-ed', 'opinion', 'column']):
            return 'Opinion'
        if 'pakistan' in text and any(k in text for k in ['prime minister', 'pm ', 'government', 'assembly']):
            return 'Pakistan'
        if any(k in text for k in ['united states', 'us ', 'china', 'india', 'united nations', 'eu ', 'europe']):
            return 'World'

        return broad

    df['broad_category'] = df.apply(infer, axis=1)
    return df
