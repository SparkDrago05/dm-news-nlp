from __future__ import annotations

import streamlit as st

from ..data.load import load_yaml
from ..models.classifier import load_classifier
from ..models.rewriter import build_rewriter
from ..models.summarizer import build_summarizer
from ..retrieval.indexer import load_retrieval_index, retrieve_similar_articles, retrieval_enabled


def main() -> None:
    """Streamlit app for interactive classification and summarization."""
    st.set_page_config(page_title='News Improvement Demo', layout='wide')

    st.title('📰 News Classification & Professional Rewrite')

    config_path = 'configs/prod_rewrite.yaml'
    try:
        cfg = load_yaml(config_path)
    except FileNotFoundError:
        cfg = load_yaml('configs/base.yaml')
        st.warning('prod_rewrite.yaml not found, falling back to base config.')

    classifier = load_classifier(cfg)
    summarizer = build_summarizer(cfg)
    rewriter = build_rewriter(cfg)

    retrieval_index = None
    retrieval_cfg = cfg.get('retrieval', {})
    if retrieval_enabled(cfg):
        try:
            retrieval_index = load_retrieval_index(cfg)
        except FileNotFoundError:
            st.info(
                'Retrieval index not found. Run `python -m src.pipeline.build_retrieval_index --config '
                f'{config_path}` to enable corpus context.',
            )

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

                retrieved = []
                if retrieval_index is not None:
                    retrieved = retrieve_similar_articles(
                        combined_text,
                        retrieval_index,
                        top_k=retrieval_cfg.get('top_k', 3),
                        min_similarity=retrieval_cfg.get('min_similarity', 0.0),
                    )

                rewrite_output = rewriter.rewrite(
                    headline=headline,
                    description=description,
                    category=pred_cat,
                    retrieved=retrieved,
                )

                st.markdown('**Professional Article Output**')
                if rewrite_output.headline:
                    st.markdown(f'### {rewrite_output.headline}')
                if rewrite_output.lead:
                    st.write(rewrite_output.lead)
                if rewrite_output.details:
                    st.write(rewrite_output.details)
                if rewrite_output.context:
                    st.info(rewrite_output.context)
                if rewrite_output.next_steps:
                    st.write(f'*What to watch next:* {rewrite_output.next_steps}')

                if retrieved:
                    st.markdown('**Retrieved sources used for context**')
                    for item in retrieved:
                        meta = [
                            f"{item.get('source', 'Source')}",
                            item.get('date', ''),
                            f"score={item.get('score', 0.0):.3f}",
                        ]
                        st.write(' - '.join(filter(None, meta)))
                        st.caption(item.get('headline', '') or item.get('description', '')[:200])

                with st.expander('Concise summary (BART)'):
                    summary = summarizer.summarize(description or headline, category=pred_cat)
                    st.write(summary)

                # Small stats
                orig_len = len((description or headline).split())
                improved_len = len(rewrite_output.compose_text().split())
                st.write(f'Original length: {orig_len} words')
                st.write(f'Improved article length: {improved_len} words')


if __name__ == '__main__':
    main()
