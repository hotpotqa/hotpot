"""
hotpot_main.py
==============
Extended entry-point for the HotpotQA baseline system.

Internet Retriever Functions
-----------------------------
  fetch_url_text(url, ...)
      Fetches a URL, strips HTML, returns clean text.

  wikipedia_search(query, top_k)
      OpenSearch API query returning title/snippet/URL tuples.

  wikipedia_page_extract(title, ...)
      REST summary API: returns the introductory plain-text extract.

  fetch_wikipedia_paragraphs(title, ...)          [NEW]
      Full-article plain-text paragraphs via MediaWiki action=query
      prop=extracts.  Returns a list of paragraph strings.

  duckduckgo_search(query, top_k)
      Scrapes DuckDuckGo HTML results for URLs.

  _extract_entities(question)                     [NEW]
      Uses spaCy NER (en_core_web_sm) to pull named entities from a
      question; falls back to noun-chunks when no NEs are found.

  retrieve_context_for_question(question, ...)    [NEW]
      End-to-end: extracts entities -> fetches Wikipedia paragraphs
      for each -> deduplicates -> returns ranked (title, para) list.

  build_context_from_internet(question, ...)      [NEW]
      Wraps retrieve_context_for_question and splits paragraphs into
      sentences, returning HotpotQA-style context dicts:
          {"title": ..., "sentences": [...]}
      Ready to be used directly as the "context" field of a HotpotQA
      sample for prepro.py.

  InternetRetriever (class)
      Object-oriented interface: search() / retrieve() / format_context().
      Supports Wikipedia and DuckDuckGo backends.

  build_internet_retriever_from_config(config)
      Constructs an InternetRetriever from CLI config flags.

  maybe_preview_internet_context(config)
      Convenience: runs retrieval for --internet_query and prints/saves
      the result.

  retrieve_mode(config)                           [NEW]
      --mode retrieve  entry-point.  Reads a JSONL question file
      ({"_id":..., "question":...} per line), fetches internet context
      for every question with build_context_from_internet(), and writes
      a HotpotQA-format JSON file ready for prepro.py.

Original modes (train / prepro / test / count) are preserved unchanged.

Dependencies
------------
  Standard library only for the original modes.
  For retrieve mode and NER-based retrieval:
      pip install requests spacy
      python -m spacy download en_core_web_sm
"""

import os
import re
import json
import time
import logging
import argparse
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import List, Dict, Tuple, Optional

from prepro import prepro
from run import train, test

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports (lazy) for new retriever functions
# ---------------------------------------------------------------------------

_nlp = None  # spaCy model, loaded on first use

def _get_nlp():
    """Return a cached spaCy NLP pipeline (loaded on first call)."""
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' not found.  "
                "Run:  python -m spacy download en_core_web_sm"
            )
    return _nlp


# ---------------------------------------------------------------------------
# Utility helpers (original)
# ---------------------------------------------------------------------------

class _HTMLTextExtractor(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self)
        self._chunks = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'):
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in ('script', 'style') and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._skip_depth == 0:
            text = data.strip()
            if text:
                self._chunks.append(text)

    def get_text(self):
        return ' '.join(self._chunks)


def _clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _safe_urlopen(url, timeout=10, headers=None):
    request = urllib.request.Request(url, headers=headers or {
        'User-Agent': 'hotpotqa-internet-retriever/1.0'
    })
    return urllib.request.urlopen(request, timeout=timeout)


# ---------------------------------------------------------------------------
# Low-level fetch helpers (original, kept intact)
# ---------------------------------------------------------------------------

def fetch_url_text(url, timeout=10, max_chars=5000):
    """Fetch a URL and return stripped plain text (up to max_chars)."""
    try:
        response = _safe_urlopen(url, timeout=timeout)
        content_type = response.headers.get('Content-Type', '')
        raw = response.read()
        charset = response.headers.get_content_charset() or 'utf-8'
        text = raw.decode(charset, errors='replace')

        if 'html' in content_type:
            parser = _HTMLTextExtractor()
            parser.feed(text)
            text = parser.get_text()

        text = _clean_text(text)
        if max_chars is not None:
            text = text[:max_chars]
        return text
    except Exception:
        return ''


def wikipedia_search(query, top_k=5):
    """OpenSearch API: return list of {title, snippet, url} dicts."""
    encoded = urllib.parse.quote(query)
    url = (
        'https://en.wikipedia.org/w/api.php'
        '?action=opensearch&search={}&limit={}&namespace=0&format=json'
        .format(encoded, top_k)
    )
    try:
        response = _safe_urlopen(url)
        data = json.loads(response.read().decode('utf-8'))
        titles = data[1]
        descriptions = data[2]
        links = data[3]
        results = []
        for idx, title in enumerate(titles):
            results.append({
                'title': title,
                'snippet': descriptions[idx] if idx < len(descriptions) else '',
                'url': links[idx] if idx < len(links) else ''
            })
        return results
    except Exception:
        return []


def wikipedia_page_extract(title, max_chars=3000):
    """REST summary API: return the introductory extract for *title*."""
    encoded = urllib.parse.quote(title)
    url = 'https://en.wikipedia.org/api/rest_v1/page/summary/{}'.format(encoded)
    try:
        response = _safe_urlopen(url)
        data = json.loads(response.read().decode('utf-8'))
        extract = data.get('extract', '')
        return _clean_text(extract)[:max_chars]
    except Exception:
        return ''


def duckduckgo_search(query, top_k=5):
    """Scrape DuckDuckGo HTML for URLs (no API key needed)."""
    encoded = urllib.parse.quote(query)
    url = 'https://duckduckgo.com/html/?q={}'.format(encoded)
    try:
        html = fetch_url_text(url, timeout=10, max_chars=20000)
        if not html:
            return []

        pattern = re.compile(r'(https?://[^\s]+)')
        urls = []
        for match in pattern.findall(html):
            if match not in urls:
                urls.append(match)
            if len(urls) >= top_k:
                break

        return [{'title': u, 'snippet': '', 'url': u} for u in urls]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# NEW: full-article paragraph retrieval via MediaWiki action=query
# ---------------------------------------------------------------------------

WIKIPEDIA_PARSE_API = 'https://en.wikipedia.org/w/api.php'


def fetch_wikipedia_paragraphs(
    title: str,
    timeout: int = 10,
    max_paragraphs: int = 10,
) -> List[str]:
    """Fetch plain-text paragraphs from a Wikipedia article.

    Uses the MediaWiki ``action=query&prop=extracts&explaintext`` endpoint
    to retrieve the full article text, then splits on double newlines and
    filters short/header-only blocks.

    Parameters
    ----------
    title : str
        Wikipedia article title (spaces or underscores both accepted).
    timeout : int
        HTTP request timeout in seconds.
    max_paragraphs : int
        Maximum number of paragraphs to return.

    Returns
    -------
    List[str]
        Plain-text paragraphs; empty list if the article is not found.
    """
    params = urllib.parse.urlencode({
        'action': 'query',
        'titles': title,
        'prop': 'extracts',
        'explaintext': '1',
        'exsectionformat': 'plain',
        'format': 'json',
        'redirects': '1',
    })
    api_url = '{}?{}'.format(WIKIPEDIA_PARSE_API, params)
    try:
        response = _safe_urlopen(api_url, timeout=timeout)
        data = json.loads(response.read().decode('utf-8'))
    except Exception as exc:
        _logger.warning('Wikipedia paragraph fetch failed for %r: %s', title, exc)
        return []

    pages = data.get('query', {}).get('pages', {})
    if not pages:
        return []

    page = next(iter(pages.values()))
    if 'missing' in page:
        _logger.debug('Wikipedia article not found: %r', title)
        return []

    raw_text = page.get('extract', '')
    if not raw_text:
        return []

    paragraphs = []
    for block in raw_text.split('\n\n'):
        block = block.strip()
        if not block:
            continue
        if block.startswith('=='):      # section header
            continue
        block = re.sub(r'\s+', ' ', block)
        if len(block) < 40:             # skip stubs
            continue
        paragraphs.append(block)
        if len(paragraphs) >= max_paragraphs:
            break

    return paragraphs


# ---------------------------------------------------------------------------
# NEW: NER-based entity extraction
# ---------------------------------------------------------------------------

_NER_LABELS = {
    'PERSON', 'ORG', 'GPE', 'LOC', 'PRODUCT',
    'WORK_OF_ART', 'EVENT', 'FAC', 'LAW', 'NORP',
}


def _extract_entities(question: str) -> List[str]:
    """Return a deduplicated list of named-entity strings from *question*.

    Uses spaCy NER (en_core_web_sm).  Falls back to noun-chunks when no
    recognised entities are found.
    """
    nlp = _get_nlp()
    doc = nlp(question)

    entities = [ent.text for ent in doc.ents if ent.label_ in _NER_LABELS]

    if not entities:
        entities = [chunk.text for chunk in doc.noun_chunks]

    # Deduplicate while preserving order
    seen: set = set()
    unique: List[str] = []
    for e in entities:
        key = e.lower()
        if key not in seen:
            seen.add(key)
            unique.append(e)
    return unique


# ---------------------------------------------------------------------------
# NEW: retrieve_context_for_question
# ---------------------------------------------------------------------------

def retrieve_context_for_question(
    question: str,
    top_k: int = 10,
    paragraphs_per_entity: int = 3,
    sleep_between_requests: float = 0.2,
) -> List[Tuple[str, str]]:
    """Retrieve Wikipedia context paragraphs relevant to *question*.

    Algorithm
    ---------
    1. Extract named entities (or noun-chunks) from *question* using spaCy.
    2. For each unique entity, fetch up to *paragraphs_per_entity*
       Wikipedia paragraphs via fetch_wikipedia_paragraphs().
    3. Return the first *top_k* (title, paragraph) pairs.

    Parameters
    ----------
    question : str
        The multi-hop question to retrieve context for.
    top_k : int
        Maximum number of (title, paragraph) pairs to return.
    paragraphs_per_entity : int
        Maximum paragraphs to keep per Wikipedia article.
    sleep_between_requests : float
        Polite delay (seconds) between Wikipedia API calls.

    Returns
    -------
    List[Tuple[str, str]]
        Each element is ``(article_title, paragraph_text)``.
    """
    entities = _extract_entities(question)
    _logger.debug('Entities for question %r: %s', question[:60], entities)

    results: List[Tuple[str, str]] = []
    seen_titles: set = set()

    for entity in entities:
        if len(results) >= top_k:
            break
        key = entity.lower()
        if key in seen_titles:
            continue
        seen_titles.add(key)

        paras = fetch_wikipedia_paragraphs(entity, max_paragraphs=paragraphs_per_entity)
        for para in paras:
            results.append((entity, para))
            if len(results) >= top_k:
                break

        if sleep_between_requests > 0:
            time.sleep(sleep_between_requests)

    return results


# ---------------------------------------------------------------------------
# NEW: build_context_from_internet
# ---------------------------------------------------------------------------

def build_context_from_internet(
    question: str,
    top_k: int = 10,
    paragraphs_per_entity: int = 3,
) -> List[Dict]:
    """Return HotpotQA-style context dicts retrieved from the internet.

    Each dict has the form::

        {
            "title": "<Wikipedia article title>",
            "sentences": ["sentence 1", "sentence 2", ...]
        }

    Paragraphs are sentence-split with spaCy to match the sentence-level
    supervision format expected by prepro.py.

    Parameters
    ----------
    question : str
        The multi-hop question.
    top_k : int
        Maximum number of (title, paragraph) pairs to retrieve.
    paragraphs_per_entity : int
        Maximum paragraphs per Wikipedia article.

    Returns
    -------
    List[Dict]
        HotpotQA context dicts, one per retrieved paragraph.
    """
    nlp = _get_nlp()
    raw_pairs = retrieve_context_for_question(
        question,
        top_k=top_k,
        paragraphs_per_entity=paragraphs_per_entity,
    )

    context_dicts: List[Dict] = []
    for title, paragraph in raw_pairs:
        doc = nlp(paragraph)
        sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
        context_dicts.append({'title': title, 'sentences': sentences})

    return context_dicts


# ---------------------------------------------------------------------------
# NEW: retrieve_mode  (--mode retrieve)
# ---------------------------------------------------------------------------

def retrieve_mode(config) -> None:
    """Batch retrieval: reads a JSONL question file, writes HotpotQA JSON.

    Input (JSONL, one object per line)::

        {"_id": "abc123", "question": "Who directed ...?"}

    Output (JSON at --retrieve_output_file)::

        [
          {
            "_id": "abc123",
            "question": "...",
            "answer": "",
            "type": "bridge",
            "level": "hard",
            "supporting_facts": [],
            "context": [
              {"title": "...", "sentences": ["...", ...]},
              ...
            ]
          },
          ...
        ]

    The output can be passed directly to prepro.py.
    """
    if not config.data_file:
        raise ValueError('--data_file must be provided for --mode retrieve')

    output_file = getattr(config, 'retrieve_output_file', 'retrieved_context.json')
    top_k = getattr(config, 'retrieve_top_k', 10)

    print('[retrieve] Reading questions from: {}'.format(config.data_file))
    print('[retrieve] Writing output to:      {}'.format(output_file))
    print('[retrieve] top_k paragraphs:        {}'.format(top_k))

    samples = []
    with open(config.data_file, 'r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    output_samples = []
    for idx, sample in enumerate(samples):
        qid = sample.get('_id', str(idx))
        question = sample.get('question', '')
        print('  [{}/{}] id={}  q={}'.format(idx + 1, len(samples), qid, question[:80]))

        context = build_context_from_internet(question, top_k=top_k)

        output_samples.append({
            '_id': qid,
            'question': question,
            'answer': sample.get('answer', ''),
            'type': sample.get('type', 'bridge'),
            'level': sample.get('level', 'hard'),
            'supporting_facts': sample.get('supporting_facts', []),
            'context': context,
        })

    with open(output_file, 'w', encoding='utf-8') as fh:
        json.dump(output_samples, fh, indent=2, ensure_ascii=False)

    print('[retrieve] Done. Wrote {} samples to {}'.format(len(output_samples), output_file))


# ---------------------------------------------------------------------------
# InternetRetriever class (original, kept intact)
# ---------------------------------------------------------------------------

class InternetRetriever(object):
    """Object-oriented internet retriever supporting Wikipedia and DuckDuckGo.

    Parameters
    ----------
    use_wikipedia : bool
        Query Wikipedia OpenSearch.
    use_duckduckgo : bool
        Query DuckDuckGo.
    fetch_pages : bool
        If True, fetch full page text in addition to snippets.
    top_k : int
        Maximum number of results to return per query.
    snippet_chars : int
        Maximum characters to keep from each snippet.
    page_chars : int
        Maximum characters to fetch from full pages.
    """

    def __init__(self, use_wikipedia=True, use_duckduckgo=False, fetch_pages=True,
                 top_k=5, snippet_chars=1200, page_chars=3000):
        self.use_wikipedia = use_wikipedia
        self.use_duckduckgo = use_duckduckgo
        self.fetch_pages = fetch_pages
        self.top_k = top_k
        self.snippet_chars = snippet_chars
        self.page_chars = page_chars

    def search(self, query):
        """Return raw search result dicts from enabled backends."""
        results = []
        seen = set()

        if self.use_wikipedia:
            for item in wikipedia_search(query, top_k=self.top_k):
                url = item.get('url', '')
                if url and url not in seen:
                    item['source'] = 'wikipedia'
                    results.append(item)
                    seen.add(url)

        if self.use_duckduckgo:
            for item in duckduckgo_search(query, top_k=self.top_k):
                url = item.get('url', '')
                if url and url not in seen:
                    item['source'] = 'duckduckgo'
                    results.append(item)
                    seen.add(url)

        return results[:self.top_k]

    def retrieve(self, query):
        """Search and optionally fetch full page text; return enriched dicts."""
        results = self.search(query)
        enriched = []
        for item in results:
            text = item.get('snippet', '') or ''
            if self.fetch_pages:
                if item.get('source') == 'wikipedia' and item.get('title'):
                    page_text = wikipedia_page_extract(item['title'], max_chars=self.page_chars)
                else:
                    page_text = fetch_url_text(item.get('url', ''), max_chars=self.page_chars)
                if page_text:
                    text = page_text
            text = _clean_text(text)[:self.snippet_chars]
            enriched.append({
                'title': item.get('title', ''),
                'url': item.get('url', ''),
                'source': item.get('source', ''),
                'text': text,
            })
        return enriched

    def format_context(self, query):
        """Return a human-readable string of retrieved context for *query*."""
        docs = self.retrieve(query)
        lines = []
        for idx, doc in enumerate(docs):
            lines.append('[{}] {} ({})'.format(idx + 1, doc['title'], doc['source']))
            if doc['url']:
                lines.append('URL: {}'.format(doc['url']))
            lines.append(doc['text'])
            lines.append('')
        return '\n'.join(lines).strip()


def build_internet_retriever_from_config(config):
    """Construct an InternetRetriever from CLI config flags."""
    return InternetRetriever(
        use_wikipedia=not config.disable_wikipedia_retriever,
        use_duckduckgo=config.enable_duckduckgo_retriever,
        fetch_pages=not config.disable_page_fetch,
        top_k=config.internet_top_k,
        snippet_chars=config.internet_snippet_chars,
        page_chars=config.internet_page_chars,
    )


def maybe_preview_internet_context(config):
    """Run retrieval for --internet_query and print/save the result."""
    if not config.internet_retriever:
        return
    if not config.internet_query:
        print('Internet retriever enabled, but no --internet_query was provided.')
        return

    retriever = build_internet_retriever_from_config(config)
    context = retriever.format_context(config.internet_query)

    if config.internet_output_file:
        with open(config.internet_output_file, 'w') as fh:
            fh.write(context)
        print('Saved internet retrieval context to {}'.format(config.internet_output_file))
    else:
        print('Internet retrieval context:')
        print(context)


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description='HotpotQA baseline -- extended with internet retrieval'
)

glove_word_file   = 'glove.840B.300d.txt'
word_emb_file     = 'word_emb.json'
char_emb_file     = 'char_emb.json'
train_eval        = 'train_eval.json'
dev_eval          = 'dev_eval.json'
test_eval         = 'test_eval.json'
word2idx_file     = 'word2idx.json'
char2idx_file     = 'char2idx.json'
idx2word_file     = 'idx2word.json'
idx2char_file     = 'idx2char.json'
train_record_file = 'train_record.pkl'
dev_record_file   = 'dev_record.pkl'
test_record_file  = 'test_record.pkl'

# ---- original arguments ---------------------------------------------------
parser.add_argument('--mode', type=str, default='train',
                    choices=['train', 'prepro', 'test', 'count', 'retrieve'],
                    help="Execution mode. Use 'retrieve' to fetch internet "
                         "context for a JSONL question file.")
parser.add_argument('--data_file',          type=str)
parser.add_argument('--glove_word_file',    type=str, default=glove_word_file)
parser.add_argument('--save',               type=str, default='HOTPOT')
parser.add_argument('--word_emb_file',      type=str, default=word_emb_file)
parser.add_argument('--char_emb_file',      type=str, default=char_emb_file)
parser.add_argument('--train_eval_file',    type=str, default=train_eval)
parser.add_argument('--dev_eval_file',      type=str, default=dev_eval)
parser.add_argument('--test_eval_file',     type=str, default=test_eval)
parser.add_argument('--word2idx_file',      type=str, default=word2idx_file)
parser.add_argument('--char2idx_file',      type=str, default=char2idx_file)
parser.add_argument('--idx2word_file',      type=str, default=idx2word_file)
parser.add_argument('--idx2char_file',      type=str, default=idx2char_file)
parser.add_argument('--train_record_file',  type=str, default=train_record_file)
parser.add_argument('--dev_record_file',    type=str, default=dev_record_file)
parser.add_argument('--test_record_file',   type=str, default=test_record_file)
parser.add_argument('--glove_char_size',    type=int, default=94)
parser.add_argument('--glove_word_size',    type=int, default=int(2.2e6))
parser.add_argument('--glove_dim',          type=int, default=300)
parser.add_argument('--char_dim',           type=int, default=8)
parser.add_argument('--para_limit',         type=int, default=1000)
parser.add_argument('--ques_limit',         type=int, default=80)
parser.add_argument('--sent_limit',         type=int, default=100)
parser.add_argument('--char_limit',         type=int, default=16)
parser.add_argument('--batch_size',         type=int, default=64)
parser.add_argument('--checkpoint',         type=int, default=1000)
parser.add_argument('--period',             type=int, default=100)
parser.add_argument('--init_lr',            type=float, default=0.5)
parser.add_argument('--keep_prob',          type=float, default=0.8)
parser.add_argument('--hidden',             type=int, default=80)
parser.add_argument('--char_hidden',        type=int, default=100)
parser.add_argument('--patience',           type=int, default=1)
parser.add_argument('--seed',               type=int, default=13)
parser.add_argument('--sp_lambda',          type=float, default=0.0)
parser.add_argument('--data_split',         type=str, default='train')
parser.add_argument('--fullwiki',           action='store_true')
parser.add_argument('--prediction_file',    type=str)
parser.add_argument('--sp_threshold',       type=float, default=0.3)

# ---- InternetRetriever (original) flags -----------------------------------
parser.add_argument('--internet_retriever',           action='store_true')
parser.add_argument('--internet_query',               type=str, default=None)
parser.add_argument('--internet_output_file',         type=str, default=None)
parser.add_argument('--internet_top_k',               type=int, default=5)
parser.add_argument('--internet_snippet_chars',       type=int, default=1200)
parser.add_argument('--internet_page_chars',          type=int, default=3000)
parser.add_argument('--enable_duckduckgo_retriever',  action='store_true')
parser.add_argument('--disable_wikipedia_retriever',  action='store_true')
parser.add_argument('--disable_page_fetch',           action='store_true')

# ---- retrieve_mode (new) flags --------------------------------------------
parser.add_argument(
    '--retrieve_output_file',
    type=str,
    default='retrieved_context.json',
    help='(retrieve mode) Path for the output HotpotQA-format JSON file.',
)
parser.add_argument(
    '--retrieve_top_k',
    type=int,
    default=10,
    help='(retrieve mode) Max (title, paragraph) pairs per question.',
)

config = parser.parse_args()


def _concat(filename):
    if config.fullwiki:
        return 'fullwiki.{}'.format(filename)
    return filename


config.dev_record_file  = _concat(config.dev_record_file)
config.test_record_file = _concat(config.test_record_file)
config.dev_eval_file    = _concat(config.dev_eval_file)
config.test_eval_file   = _concat(config.test_eval_file)

# ---------------------------------------------------------------------------
# Mode dispatch
# ---------------------------------------------------------------------------

if config.mode == 'retrieve':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s -- %(message)s',
    )
    retrieve_mode(config)
elif config.internet_retriever and config.mode == 'retrieve':
    maybe_preview_internet_context(config)
elif config.mode == 'train':
    train(config)
elif config.mode == 'prepro':
    prepro(config)
elif config.mode == 'test':
    if config.internet_retriever:
        maybe_preview_internet_context(config)
    test(config)
elif config.mode == 'count':
    cnt_len(config)
