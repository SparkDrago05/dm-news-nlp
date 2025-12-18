from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RewriteOutput:
    """Structured response from the News Improver."""

    headline: str
    lead: str
    details: str
    context: str
    next_steps: str
    retrieved_sources: List[Dict[str, Any]]

    def compose_text(self) -> str:
        """Join non-empty sections into a readable article."""
        parts: List[str] = []
        for part in [self.headline, self.lead, self.details, self.context, self.next_steps]:
            part = (part or '').strip()
            if part:
                parts.append(part)
        return '\n\n'.join(parts)

    def as_dict(self) -> Dict[str, str]:
        return {
            'headline': self.headline,
            'lead': self.lead,
            'details': self.details,
            'context': self.context,
            'next_steps': self.next_steps,
        }


class BaseRewriter:
    """Interface for a news improver."""

    def rewrite(
        self,
        *,
        headline: str,
        description: str,
        category: Optional[str],
        retrieved: Optional[List[Dict[str, Any]]] = None,
    ) -> RewriteOutput:
        raise NotImplementedError


class IdentityRewriter(BaseRewriter):
    """Fallback rewriter that simply returns the original input."""

    def rewrite(
        self,
        *,
        headline: str,
        description: str,
        category: Optional[str],
        retrieved: Optional[List[Dict[str, Any]]] = None,
    ) -> RewriteOutput:
        text = (description or '').strip() or (headline or '').strip()
        return RewriteOutput(
            headline=(headline or '').strip(),
            lead=text,
            details='',
            context='',
            next_steps='',
            retrieved_sources=retrieved or [],
        )


_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')


def _normalize_ws(text: str) -> str:
    return re.sub(r'\s+', ' ', (text or '').replace('\n', ' ')).strip()


def _split_sentences(text: str) -> List[str]:
    """Basic sentence splitter (good enough for news paragraphs)."""
    text = _normalize_ws(text)
    if not text:
        return []
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p.strip()]
    return parts


def _clean_headline(text: str) -> str:
    """Normalize punctuation/casing for improved headline."""
    text = (text or '').strip()
    if not text:
        return ''
    text = text.rstrip('.')
    if len(text) <= 4:
        return text.title()
    return text[0].upper() + text[1:]


def _ensure_period(s: str) -> str:
    s = (s or '').strip()
    if not s:
        return ''
    if s.endswith(('.', '!', '?')):
        return s
    return s + '.'


def _dedupe_sentences(sentences: List[str]) -> List[str]:
    """Remove near duplicates by normalized form."""
    seen = set()
    out: List[str] = []
    for s in sentences:
        norm = re.sub(r'[^a-z0-9]+', '', s.lower())
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(s)
    return out


def _safe_join(sentences: List[str], max_sentences: int) -> str:
    sentences = [s.strip() for s in sentences if s.strip()]
    sentences = _dedupe_sentences(sentences)
    return ' '.join(_ensure_period(s) for s in sentences[:max_sentences]).strip()


def _default_why_it_matters(category: str) -> str:
    category = (category or 'Other').strip()
    templates = {
        'Business': (
            'The development matters for consumers and businesses because energy and input costs can flow into '
            'prices, investment decisions, and overall market confidence.'
        ),
        'Pakistan': (
            'The issue is significant because administrative decisions and public-sector responses can shape '
            'service delivery, public confidence, and institutional accountability.'
        ),
        'World': (
            'The situation matters because shifts in diplomatic positioning and regional coordination can affect '
            'security, trade, and humanitarian outcomes beyond national borders.'
        ),
        'Sports': (
            'The outcome matters for team strategy and competition standings, while fans and management watch for '
            'performance trends and fitness updates.'
        ),
        'Technology': (
            'The development matters because product choices, regulation, and investment cycles can influence '
            'consumer adoption and industry competition.'
        ),
        'Lifestyle': (
            'The story matters because community behaviour, public awareness, and cultural trends can shape '
            'everyday choices and social outcomes.'
        ),
        'Opinion': (
            'The debate matters because policy and public discourse often shift when competing narratives are '
            'tested against evidence and practical constraints.'
        ),
    }
    return templates.get(category, 'The development matters because it can influence public decisions, sentiment, and follow-up actions.')


class ArticleExpander(BaseRewriter):
    """
    Offline article-style expander.

    Key idea:
    - Do NOT invent specific facts.
    - Expand using:
      (1) the user input (headline/description)
      (2) retrieved similar articles from the dataset (extractive context)
      (3) safe newsroom structure + neutral explanatory framing.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

        self.include_headline = bool(cfg.get('include_headline', True))
        self.include_context = bool(cfg.get('include_context', True))
        self.include_next_steps = bool(cfg.get('include_next_steps', True))
        self.include_evidence = bool(cfg.get('include_evidence', True))

        self.lede_sentences = int(cfg.get('lede_sentences', 3))
        self.user_detail_sentences = int(cfg.get('user_detail_sentences', 6))

        self.context_max_sources = int(cfg.get('context_max_sources', 3))
        self.context_sentences_per_source = int(cfg.get('context_sentences_per_source', 2))
        self.context_max_total_sentences = int(cfg.get('context_max_total_sentences', 6))
        self.context_min_sentence_chars = int(cfg.get('context_min_sentence_chars', 45))

        self.target_min_words = int(cfg.get('target_min_words', 220))
        self.target_max_words = int(cfg.get('target_max_words', 420))

        self.background_prefix = cfg.get('background_prefix', 'Related coverage in our corpus indicates:')
        self.next_steps_templates = cfg.get('next_steps_templates', {})

    def rewrite(
        self,
        *,
        headline: str,
        description: str,
        category: Optional[str],
        retrieved: Optional[List[Dict[str, Any]]] = None,
    ) -> RewriteOutput:
        category = (category or 'Other').strip() or 'Other'
        headline = (headline or '').strip()
        description = (description or '').strip()

        base_text = description or headline
        base_text = _normalize_ws(base_text)

        user_sents = _split_sentences(base_text)
        user_sents = _dedupe_sentences(user_sents)

        # Build headline
        improved_headline = ''
        if self.include_headline:
            improved_headline = _clean_headline(headline) if headline else self._headline_from_text(user_sents, category)

        # Build lede (lead paragraph)
        lede = self._build_lede(improved_headline, user_sents, category)

        # Build user detail body (expanded but grounded)
        details = self._build_details(user_sents, category)

        # Build context paragraph from retrieved docs
        retrieved = retrieved or []
        context = ''
        if self.include_context and retrieved:
            context = self._build_context(base_text, retrieved)

        # Build “what to watch next”
        next_steps = ''
        if self.include_next_steps:
            next_steps = self._build_next_steps(category, retrieved=retrieved)

        # Ensure overall length is “article-like”
        output = RewriteOutput(
            headline=improved_headline,
            lead=lede,
            details=details,
            context=context,
            next_steps=next_steps,
            retrieved_sources=retrieved,
        )

        output = self._pad_if_too_short(output, category)
        output = self._trim_if_too_long(output)

        return output

    def _headline_from_text(self, user_sents: List[str], category: str) -> str:
        if user_sents:
            # Take first sentence fragment as headline-like
            h = user_sents[0]
            h = re.sub(r'^(according to|reportedly|in a statement)\s+', '', h.strip(), flags=re.I)
            h = h[:110].rstrip(' ,;:-')
            return _clean_headline(h)
        return _clean_headline(f'{category} update')

    def _build_lede(self, headline: str, user_sents: List[str], category: str) -> str:
        # Prefer extracting “what happened” from the first sentences
        core = _safe_join(user_sents, self.lede_sentences)
        if not core:
            core = 'Developments are emerging as more details become available.'

        # Add a neutral framing sentence to make it read like an article
        framing = {
            'Business': 'Market participants are watching closely as the implications extend to costs, investment, and policy signals.',
            'Pakistan': 'Officials and stakeholders are monitoring the situation, with public attention focused on accountability and next actions.',
            'World': 'Diplomatic and regional responses are in focus as the situation develops.',
            'Sports': 'Attention now shifts to preparation and upcoming fixtures as teams assess performance and fitness.',
            'Technology': 'Industry observers are tracking the impact on users, competition, and regulatory outlook.',
            'Lifestyle': 'Community response and practical impacts remain central as the story unfolds.',
            'Opinion': 'The debate is likely to continue as competing perspectives test policy choices and public expectations.',
        }.get(category, 'Further updates are expected as stakeholders respond and additional reporting clarifies the details.')

        # If core is already 3 sentences, keep framing short
        lede = core
        if len(_split_sentences(core)) < self.lede_sentences:
            lede = (core + ' ' + _ensure_period(framing)).strip()
        return lede

    def _build_details(self, user_sents: List[str], category: str) -> str:
        """
        Expand details while staying grounded:
        - include more user sentences
        - add safe “why it matters” paragraph
        - add safe connective paragraph
        """
        remaining = user_sents[self.lede_sentences : self.lede_sentences + self.user_detail_sentences]
        remaining_text = _safe_join(remaining, self.user_detail_sentences)

        why = _default_why_it_matters(category)
        why = _ensure_period(why)

        connector = (
            'Taken together, the available information suggests a developing situation where follow-through, implementation, '
            'and verified updates will determine the practical impact.'
        )
        connector = _ensure_period(connector)

        parts: List[str] = []
        if remaining_text:
            parts.append(remaining_text)

        # Always add two extra paragraphs to make it article-like (but still safe)
        parts.append(why)
        parts.append(connector)

        # Split into paragraphs
        return '\n\n'.join(p.strip() for p in parts if p.strip())

    def _build_context(self, query_text: str, retrieved: List[Dict[str, Any]]) -> str:
        """
        Extract a few high-similarity sentences from retrieved articles (dataset RAG).
        We keep them short and attribute by source to avoid “hallucinated” context.
        """
        top_sources = retrieved[: self.context_max_sources]

        candidates: List[Tuple[str, str, float]] = []  # (source, sentence, score)
        for item in top_sources:
            src = str(item.get('source') or 'Source').strip()
            txt = str(item.get('description') or item.get('combined_text') or '').strip()
            for s in _split_sentences(txt):
                s = s.strip()
                if len(s) < self.context_min_sentence_chars:
                    continue
                candidates.append((src, s, 0.0))

        if not candidates:
            return ''

        # Rank sentences by similarity to query_text using a tiny TF-IDF model
        sent_texts = [c[1] for c in candidates]
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=12000, stop_words='english')
        X = vectorizer.fit_transform([query_text] + sent_texts)
        q = X[0:1]
        s_mat = X[1:]
        sims = cosine_similarity(q, s_mat).ravel()

        ranked: List[Tuple[str, str, float]] = []
        for (src, sent, _), sim in zip(candidates, sims):
            ranked.append((src, sent, float(sim)))
        ranked.sort(key=lambda x: x[2], reverse=True)

        # Select diverse sentences across sources
        picked: List[Tuple[str, str, float]] = []
        per_source_count: Dict[str, int] = {}
        for src, sent, sim in ranked:
            if len(picked) >= self.context_max_total_sentences:
                break
            if per_source_count.get(src, 0) >= self.context_sentences_per_source:
                continue
            picked.append((src, sent, sim))
            per_source_count[src] = per_source_count.get(src, 0) + 1

        if not picked:
            return ''

        lines = [f'{src}: {_ensure_period(sent)}' for src, sent, _ in picked]
        lines = _dedupe_sentences(lines)

        return f'{self.background_prefix}\n' + '\n'.join(f'- {ln}' for ln in lines)

    def _build_next_steps(self, category: str, *, retrieved: List[Dict[str, Any]]) -> str:
        template = self.next_steps_templates.get(category) or self.next_steps_templates.get('Other', '')
        template = (template or '').strip()

        evidence = ''
        if self.include_evidence and retrieved:
            # Keep evidence compact
            items = []
            for item in retrieved[:3]:
                src = item.get('source', 'source')
                score = float(item.get('score', 0.0))
                h = (item.get('headline') or '')[:140].strip()
                items.append(f'- [{src} | score={score:.3f}] {h}')
            evidence = 'Evidence used (top matches from dataset):\n' + '\n'.join(items)

        parts = []
        if template:
            parts.append(template)
        if evidence:
            parts.append(evidence)

        return '\n\n'.join(parts).strip()

    def _pad_if_too_short(self, output: RewriteOutput, category: str) -> RewriteOutput:
        """If output is too short, add one more safe paragraph (still no new facts)."""
        text = output.compose_text()
        wc = len(text.split())
        if wc >= self.target_min_words:
            return output

        padding = {
            'Business': (
                'In the near term, attention typically turns to how quickly policy signals translate into measurable changes in costs, '
                'supply conditions, and business sentiment. Clear communication and consistent implementation often shape whether markets '
                'price in stability or continued volatility.'
            ),
            'Pakistan': (
                'In similar cases, the public response often depends on transparency, timely clarification, and how quickly institutions '
                'address procedural questions. Observers also look for formal statements that confirm timelines and responsibilities.'
            ),
            'World': (
                'Observers commonly watch for coordinated statements, follow-up meetings, and any measurable shift in on-ground conditions '
                'that could indicate de-escalation or further strain. Communication between key parties is usually a key signal.'
            ),
            'Sports': (
                'Teams typically review match footage and training metrics to address weaknesses highlighted by recent performances. '
                'Selection decisions and fitness updates often provide early indicators of how the next contest may unfold.'
            ),
            'Technology': (
                'Product rollouts and policy decisions are often followed by user feedback, performance benchmarks, and competitor responses. '
                'How quickly issues are addressed can shape adoption and long-term trust.'
            ),
            'Lifestyle': (
                'Public interest in lifestyle stories often grows when practical guidance, local context, and community participation are clear. '
                'Organisers and stakeholders may adjust plans based on early feedback and outcomes.'
            ),
            'Opinion': (
                'Public debate often evolves as new evidence, expert commentary, and institutional responses emerge. Over time, attention tends to shift '
                'from broad claims to measurable outcomes and accountability.'
            ),
        }.get(category, 'Further clarity typically comes as stakeholders provide verified updates and details are confirmed through follow-up reporting.')

        padding = _ensure_period(padding)

        # Add as extra paragraph inside details
        details = (output.details or '').strip()
        if details:
            details = details + '\n\n' + padding
        else:
            details = padding

        output.details = details
        return output

    def _trim_if_too_long(self, output: RewriteOutput) -> RewriteOutput:
        """If output exceeds target_max_words, trim context first, then details."""
        text = output.compose_text()
        wc = len(text.split())
        if wc <= self.target_max_words:
            return output

        # 1) Trim context bullets
        if output.context:
            lines = output.context.splitlines()
            # keep header + first 3 bullets
            header = lines[:1]
            bullets = [ln for ln in lines[1:] if ln.strip().startswith('-')]
            bullets = bullets[:3]
            output.context = '\n'.join(header + bullets).strip()

        text = output.compose_text()
        wc = len(text.split())
        if wc <= self.target_max_words:
            return output

        # 2) Trim details by dropping the last paragraph
        if output.details and '\n\n' in output.details:
            paras = [p.strip() for p in output.details.split('\n\n') if p.strip()]
            if len(paras) > 2:
                output.details = '\n\n'.join(paras[:2])

        return output


class StructuredArticleRewriter(BaseRewriter):
    """Legacy template splitter (kept for compatibility)."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.include_headline = cfg.get('include_headline', True)
        self.include_context = cfg.get('include_context', True)
        self.include_next_steps = cfg.get('include_next_steps', True)
        self.lead_max_sentences = int(cfg.get('lead_max_sentences', 2))
        self.detail_max_sentences = int(cfg.get('detail_max_sentences', 4))
        self.context_sentences_per_article = int(cfg.get('context_sentences_per_article', 2))
        self.background_prefix = cfg.get('generic_background_prefix', 'Background:')
        self.next_steps_templates = cfg.get('next_steps_templates', {})

    def rewrite(
        self,
        *,
        headline: str,
        description: str,
        category: Optional[str],
        retrieved: Optional[List[Dict[str, Any]]] = None,
    ) -> RewriteOutput:
        base_text = (description or '').strip() or (headline or '').strip()
        sentences = _split_sentences(base_text)

        lead = ' '.join(sentences[: self.lead_max_sentences]).strip()
        details = ' '.join(sentences[self.lead_max_sentences : self.lead_max_sentences + self.detail_max_sentences]).strip()

        context = ''
        if self.include_context:
            context = self._build_context(retrieved or [])

        next_steps = ''
        if self.include_next_steps:
            next_steps = self._build_next_steps(category or 'Other')

        improved_headline = _clean_headline(headline) if self.include_headline else ''

        return RewriteOutput(
            headline=improved_headline,
            lead=lead or base_text,
            details=details,
            context=context,
            next_steps=next_steps,
            retrieved_sources=retrieved or [],
        )

    def _build_context(self, retrieved: List[Dict[str, Any]]) -> str:
        if not retrieved:
            return ''

        snippets: List[str] = []
        for item in retrieved:
            snippet_source = item.get('source') or 'Dataset source'
            snippet_text = item.get('description') or item.get('combined_text') or item.get('headline') or ''
            sentences = _split_sentences(str(snippet_text))
            text = ' '.join(sentences[: self.context_sentences_per_article]).strip()
            if text:
                snippets.append(f'{snippet_source}: {text}')

        if not snippets:
            return ''

        return f'{self.background_prefix} ' + ' '.join(snippets)

    def _build_next_steps(self, category: str) -> str:
        template = self.next_steps_templates.get(category)
        if not template:
            template = self.next_steps_templates.get('Other', '')
        return template


def build_rewriter(cfg: Dict[str, Any]) -> BaseRewriter:
    """Factory for rewriter implementations."""
    rewrite_cfg = cfg.get('rewrite', {})
    r_type = rewrite_cfg.get('type', 'article_expander')

    if r_type == 'article_expander':
        return ArticleExpander(rewrite_cfg)

    if r_type == 'structured_article':
        return StructuredArticleRewriter(rewrite_cfg)

    return IdentityRewriter()
