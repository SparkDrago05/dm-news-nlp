from __future__ import annotations

from typing import Dict, Tuple, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from ..features.vectorizers import build_vectorizer


def _build_base_classifier(cfg: Dict) -> Any:
    """Build a base sklearn classifier from config."""
    clf_cfg = cfg['classifier']
    clf_type = clf_cfg.get('type', 'logreg')

    if clf_type == 'logreg':
        params = clf_cfg.get('logreg', {})
        return LogisticRegression(
            C=float(params.get('C', 1.0)),
            class_weight=params.get('class_weight', None),
            max_iter=200,
            n_jobs=-1,
        )
    if clf_type == 'svm':
        params = clf_cfg.get('svm', {})
        return LinearSVC(
            C=float(params.get('C', 1.0)),
        )
    if clf_type == 'nb':
        params = clf_cfg.get('nb', {})
        return MultinomialNB(alpha=float(params.get('alpha', 1.0)))
    if clf_type == 'rf':
        params = clf_cfg.get('rf', {})
        return RandomForestClassifier(
            n_estimators=int(params.get('n_estimators', 200)),
            max_depth=params.get('max_depth', None),
            n_jobs=-1,
        )

    raise ValueError(f'Unknown classifier type: {clf_type}')


def build_text_classifier_pipeline(cfg: Dict) -> Pipeline:
    """Build a sklearn Pipeline: vectorizer + classifier."""
    vectorizer = build_vectorizer(cfg)
    classifier = _build_base_classifier(cfg)

    pipeline = Pipeline(
        steps=[
            ('vect', vectorizer),
            ('clf', classifier),
        ],
    )
    return pipeline


def prepare_text_and_labels(df: pd.DataFrame, cfg: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare text (X) and labels (y) from DataFrame based on config."""
    text_cols = cfg['preprocessing']['text_columns']
    join_with = cfg['preprocessing'].get('join_with', ' ')

    # Ensure missing texts are filled with ''
    text_df = df[text_cols].fillna('')
    X_text = text_df.apply(lambda row: join_with.join(map(str, row)), axis=1).values
    y = df['broad_category'].values

    return X_text, y


def train_text_classifier(
        df: pd.DataFrame,
        cfg: Dict,
) -> Tuple[Pipeline, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Train classifier and return model + splits + basic metrics."""
    test_size = cfg['evaluation'].get('test_size', 0.2)
    random_seed = cfg['project']['random_seed']

    X_text, y = prepare_text_and_labels(df, cfg)

    X_train, X_test, y_train, y_test = train_test_split(
        X_text,
        y,
        test_size=test_size,
        random_state=random_seed,
        stratify=y,
    )

    pipeline = build_text_classifier_pipeline(cfg)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    report_dict = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    return pipeline, X_train, X_test, y_train, y_test, report_dict


def save_classifier(model: Pipeline, cfg: Dict) -> None:
    """Save trained classifier to disk."""
    art_cfg = cfg['artifacts']
    model_dir = art_cfg.get('model_dir', 'models')
    filename = art_cfg.get('classifier_filename', 'news_classifier.joblib')

    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)

    full_path = model_path / filename
    joblib.dump(model, full_path)


def load_classifier(cfg: Dict) -> Pipeline:
    """Load a trained classifier from disk."""
    art_cfg = cfg['artifacts']
    model_dir = art_cfg.get('model_dir', 'models')
    filename = art_cfg.get('classifier_filename', 'news_classifier.joblib')

    full_path = Path(model_dir) / filename
    model: Pipeline = joblib.load(full_path)
    return model
