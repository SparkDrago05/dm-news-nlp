from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import streamlit as st
import yaml

from ..models.classifier import load_classifier
from ..models.summarizer import build_summarizer


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load YAML config file."""
    path = Path(path)
    with path.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main() -> None:
    """Streamlit app for interactive classification and summarization."""
    st.set_page_config(page_title='News Improvement Demo', layout='wide')

    st.title('📰 News Classification & Professional Rewrite')

    cfg = load_yaml('configs/base.yaml')
    classifier = load_classifier(cfg)
    summarizer = build_summarizer(cfg)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Input Article')
        headline = st.text_input('Headline', value='')
        description = st.text_area('Full article text', height=300)

        if st.button('Run'):
            if not description.strip() and not headline.strip():
                st.warning('Please enter at least a headline or description.')
                return

            text_cols = cfg['preprocessing']['text_columns']
            join_with = cfg['preprocessing'].get('join_with', ' ')
            values = []
            for col in text_cols:
                if col == 'headline':
                    values.append(headline)
                elif col == 'description':
                    values.append(description)
                else:
                    values.append('')
            combined_text = join_with.join(values)

            pred_cat = classifier.predict([combined_text])[0]

            with col2:
                st.subheader('Results')
                st.write(f'**Predicted Category:** {pred_cat}')

                summary = summarizer.summarize(description or headline, category=pred_cat)
                st.markdown('**Improved / Summarized Version:**')
                st.write(summary)

                # Small stats
                orig_len = len((description or headline).split())
                summ_len = len(summary.split())
                st.write(f'Original length: {orig_len} words')
                st.write(f'Summary length: {summ_len} words')


if __name__ == '__main__':
    main()
