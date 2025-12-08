from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Any

import yaml

from ..data.load import load_all_sources, add_broad_category
from ..models.classifier import train_text_classifier
from ..models.evaluation import evaluate_classifier
from ..models.classifier import save_classifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load YAML config file."""
    path = Path(path)
    with path.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main() -> None:
    """Train text classifier according to config."""
    parser = argparse.ArgumentParser(description='Train news classification model.')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/base.yaml',
        help='Path to YAML config.',
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    logger.info('Loaded config from %s', args.config)

    df = load_all_sources(cfg)
    logger.info('Loaded combined dataset shape: %s', df.shape)

    df = add_broad_category(df, cfg)
    logger.info('After broad_category mapping: %s', df['broad_category'].value_counts())

    model, X_train, X_test, y_train, y_test, report_dict = train_text_classifier(df, cfg)
    logger.info('Training complete.')

    # Basic evaluation
    from sklearn.metrics import classification_report

    y_pred = model.predict(X_test)
    metrics = evaluate_classifier(y_test, y_pred)
    logger.info('Accuracy: %.4f, Macro F1: %.4f', metrics['accuracy'], metrics['macro_f1'])
    logger.info('Classification report:\n%s', classification_report(y_test, y_pred, zero_division=0))

    save_classifier(model, cfg)
    logger.info('Model saved.')


if __name__ == '__main__':
    main()
