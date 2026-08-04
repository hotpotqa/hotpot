"""hotpot_main.py

Extended entry-point for the HotpotQA baseline that adds an **Internet
Retriever** layer on top of the original train / prepro / test pipeline.

The retriever can be used in two ways:

1. Stand-alone interactive mode (``--mode retrieve``):
   Given a question it fetches candidate Wikipedia / web passages,
   ranks them, and prints the top-k results.

2. Augmented inference mode (``--mode retrieve_test``):
   Runs retrieval for every question in the test-eval file, injects the
   retrieved passages as additional context, then calls the trained model
   for prediction.  Useful for the *fullwiki* distractor setting where
   the gold paragraphs are NOT provided at test time.

Dependencies (beyond the original repo requirements):
    pip install requests beautifulsoup4 rank_bm25
"""

import os
import re
import time
import json
import logging
import argparse
import urllib.parse
from typing import List, Dict, Tuple, Optional

# ---------------------------------------------------------------------------
# Optional heavy imports -- only fail at call-time so the rest of the module
# can still be imported in environments where these are absent.
# ---------------------------------------------------------------------------
try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    requests = None  # type: ignore
    BeautifulSoup = None  # type: ignore

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None  # type: ignore

# ---------------------------------------------------------------------------
# Re-export the original pipeline so this file can replace main.py entirely.
# ---------------------------------------------------------------------------
from prepro import prepro
from run import train, test

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ===========================================================================
# Section 1 -- Low-level HTTP / Search helpers
# ===========================================================================

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
_SESSION: Optional["requests.Session"] = None  # re-used across calls


def _get_session() -> "requests.Session":
    """Return a lazily-created, keep-alive requests Session."""
    global _SESSION
    if requests is None:
        raise ImportError(
            "The 'requests' package is required for internet retrieval. "
            "Install it with:  pip install requests"
        )
    if _SESSION is None:
        _SESSION = requests.Session()
        _SESSION.headers.update(
            {"User-Agent": "HotpotQA-InternetRetriever/1.0 (research project)"}
        )
    return _SESSION


def wikipedia_search_titles(
    query: str,
    top_k: int = 10,
    timeout: float = 5.0,
) -> List[str]:
    """Return up to *top_k* Wikipedia article titles matching *query*.

    Uses the MediaWiki ``opensearch`` API endpoint which is fast and
    requires no authentication.
    """
    session = _get_session()
    params = {
        "action": "opensearch",
        "search": query,
        "limit": top_k,
        "namespace": 0,
        "format": "json",
    }
    try:
        resp = session.get(WIKIPEDIA_API_URL, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        # opensearch returns [query, [titles], [descriptions], [urls]]
        return data[1] if len(data) > 1 else []
    except Exception as exc:  # noqa: BLE001
        logger.warning("wikipedia_search_titles failed: %s", exc)
        return []


def fetch_wikipedia_intro(
    title: str,
    max_sentences: int = 10,
    timeout: float = 8.0,
) -> str:
    """Fetch the introductory paragraph of a Wikipedia article.

    Returns the text as a single string (sentences joined by spaces).
    Falls back to an empty string on any error.
    """
    session = _get_session()
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "exintro": True,
        "explaintext": True,
        "format": "json",
    }
    try:
        resp = session.get(WIKIPEDIA_API_URL, params=params, timeout=timeout)
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        for page in pages.values():
            extract = page.get("extract", "").strip()
            if extract:
                sentences = re.split(r"(?<=[.!?])\s+", extract)
                return " ".join(sentences[:max_sentences])
        return ""
    except Exception as exc:  # noqa: BLE001
        logger.warning("fetch_wikipedia_intro('%s') failed: %s", title, exc)
        return ""


def fetch_wikipedia_full_text(
    title: str,
    timeout: float = 10.0,
) -> str:
    """Fetch the full plain-text of a Wikipedia article via the parse API."""
    session = _get_session()
    params = {
        "action": "parse",
        "page": title,
        "prop": "wikitext",
        "format": "json",
    }
    try:
        resp = session.get(WIKIPEDIA_API_URL, params=params, timeout=timeout)
        resp.raise_for_status()
        wikitext = (
            resp.json().get("parse", {}).get("wikitext", {}).get("*", "")
        )
        # Strip wiki markup with a simple heuristic
        text = re.sub(r"\{\{[^}]*\}\}", "", wikitext)   # remove templates
        text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)  # links
        text = re.sub(r"<[^>]+>", "", text)              # html tags
        text = re.sub(r"={2,}[^=]+=+\n?", "", text)    # section headers
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text
    except Exception as exc:  # noqa: BLE001
        logger.warning("fetch_wikipedia_full_text('%s') failed: %s", title, exc)
        return ""


def search_duckduckgo(
    query: str,
    top_k: int = 5,
    timeout: float = 5.0,
) -> List[Tuple[str, str]]:
    """Return up to *top_k* search results (title, url) from DuckDuckGo.

    Uses DuckDuckGo HTML interface which is clean and requires no API key.
    """
    if requests is None or BeautifulSoup is None:
        logger.warning("requests or bs4 not installed. Cannot perform web search.")
        return []

    session = _get_session()
    url = "https://html.duckduckgo.com/html/"
    params = {"q": query}
    try:
        resp = session.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        results: List[Tuple[str, str]] = []
        for a in soup.find_all("a", class_="result__url"):
            href = a.get("href", "")
            title = a.get_text().strip()

            # Decode DDG redirection wrapper if present
            if "/l/?" in href:
                parsed = urllib.parse.urlparse(href)
                qs = urllib.parse.parse_qs(parsed.query)
                if "uddg" in qs:
                    href = qs["uddg"][0]

            if href and href.startswith("http") and "duckduckgo.com" not in href:
                results.append((title or "Web Result", href))
                if len(results) >= top_k:
                    break
        return results
    except Exception as exc:
        logger.warning("search_duckduckgo failed: %s", exc)
        return []


def fetch_general_url_text(
    url: str,
    timeout: float = 8.0,
) -> str:
    """Fetch and extract clean plain text from a general web page.

    Strips scripts, styles, navigation, footer, header, and sidebar elements,
    returning clean readable paragraphs separated by double newlines.
    """
    if requests is None or BeautifulSoup is None:
        logger.warning("requests or bs4 not installed. Cannot fetch URL content.")
        return ""

    session = _get_session()
    try:
        resp = session.get(url, timeout=timeout)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # Decompose non-content elements
        for element in soup(["script", "style", "nav", "footer", "header",
                             "aside", "form", "iframe"]):
            element.decompose()

        paragraphs = []
        for p in soup.find_all(["p", "article", "section", "div"]):
            if (p.name == "div" and
                    (len(p.find_all("div")) > 2 or
                     len(p.get_text(strip=True)) < 40)):
                continue
            text = p.get_text().strip()
            text = re.sub(r"\s+", " ", text)
            if len(text) > 40:
                paragraphs.append(text)

        # Remove duplicates while maintaining order
        seen = set()
        unique_paras = []
        for p in paragraphs:
            if p not in seen:
                seen.add(p)
                unique_paras.append(p)

        return "\n\n".join(unique_paras)
    except Exception as exc:
        logger.warning("fetch_general_url_text('%s') failed: %s", url, exc)
        return ""


# ===========================================================================
# Section 2 -- Passage splitting & ranking
# ===========================================================================

def split_into_sentences(text: str) -> List[str]:
    """Split *text* into sentences using a lightweight regex heuristic."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(
    text: str,
    chunk_size: int = 5,
    overlap: int = 1,
) -> List[str]:
    """Divide *text* into overlapping sentence-level chunks.

    Args:
        text:       Source text.
        chunk_size: Number of sentences per chunk.
        overlap:    Number of sentences shared between consecutive chunks.

    Returns:
        List of passage strings.
    """
    sentences = split_into_sentences(text)
    chunks: List[str] = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(sentences), step):
        chunk = " ".join(sentences[i : i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks


def bm25_rank(
    query: str,
    passages: List[str],
    top_k: int = 5,
) -> List[Tuple[float, str]]:
    """Rank *passages* against *query* using BM25.

    Returns a list of (score, passage) tuples sorted by descending score.
    Falls back to returning the first *top_k* passages if rank_bm25 is not
    installed.
    """
    if not passages:
        return []

    if BM25Okapi is None:
        logger.warning(
            "rank_bm25 is not installed; returning first %d passages unranked. "
            "Install with:  pip install rank_bm25",
            top_k,
        )
        return [(0.0, p) for p in passages[:top_k]]

    tokenized_corpus = [p.lower().split() for p in passages]
    bm25 = BM25Okapi(tokenized_corpus)
    query_tokens = query.lower().split()
    scores = bm25.get_scores(query_tokens)
    ranked = sorted(zip(scores, passages), key=lambda x: x[0], reverse=True)
    return ranked[:top_k]


# ===========================================================================
# Section 3 -- High-level retriever interface
# ===========================================================================

class InternetRetriever:
    """Retrieve and rank web / Wikipedia passages for a natural-language query.

    Supports both specialized Wikipedia searching and broader general web
    searching.

    Usage::

        retriever = InternetRetriever(
            top_k_titles=5, top_k_passages=3, retriever_source="all"
        )
        results = retriever.retrieve(
            "Who was the first person to walk on the Moon?"
        )
        for passage in results:
            print(passage["title"], "|", passage["score"])
            print(passage["text"])

    Each result dict has the keys:
        ``title``   -- Wikipedia article title or web page domain/title.
        ``text``    -- Retrieved passage text.
        ``score``   -- BM25 relevance score (float).
        ``url``     -- Canonical URL.
    """

    WIKI_BASE_URL = "https://en.wikipedia.org/wiki/"

    def __init__(
        self,
        top_k_titles: int = 5,
        top_k_passages: int = 3,
        chunk_size: int = 5,
        chunk_overlap: int = 1,
        use_full_text: bool = False,
        cache_dir: Optional[str] = None,
        retriever_source: str = "all",
    ) -> None:
        self.top_k_titles = top_k_titles
        self.top_k_passages = top_k_passages
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_full_text = use_full_text
        self._cache: Dict[str, str] = {}  # title or URL -> page/article text
        self.cache_dir = cache_dir
        self.retriever_source = retriever_source
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_path(self, title_or_url: str) -> str:
        safe = re.sub(r"[^\w\-]", "_", title_or_url)
        return os.path.join(self.cache_dir, f"{safe}.txt")  # type: ignore

    def _load_from_disk(self, title_or_url: str) -> Optional[str]:
        if not self.cache_dir:
            return None
        path = self._cache_path(title_or_url)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                return fh.read()
        return None

    def _save_to_disk(self, title_or_url: str, text: str) -> None:
        if not self.cache_dir:
            return
        path = self._cache_path(title_or_url)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)

    # ------------------------------------------------------------------
    # Article / Page fetching
    # ------------------------------------------------------------------

    def _get_page_text(self, title_or_url: str, is_url: bool = False) -> str:
        """Return article or webpage text, using in-memory then disk cache."""
        if title_or_url in self._cache:
            return self._cache[title_or_url]
        disk_text = self._load_from_disk(title_or_url)
        if disk_text is not None:
            self._cache[title_or_url] = disk_text
            return disk_text

        if is_url:
            text = fetch_general_url_text(title_or_url)
        else:
            if self.use_full_text:
                text = fetch_wikipedia_full_text(title_or_url)
            else:
                text = fetch_wikipedia_intro(title_or_url)

        self._cache[title_or_url] = text
        self._save_to_disk(title_or_url, text)
        return text

    # ------------------------------------------------------------------
    # Main retrieval method
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k_titles: Optional[int] = None,
        top_k_passages: Optional[int] = None,
    ) -> List[Dict]:
        """Retrieve the most relevant passages for *query*.

        Args:
            query:          Natural-language question or query string.
            top_k_titles:   Override instance default for number of sources
                            to search.
            top_k_passages: Override instance default for number of final
                            passages to return.

        Returns:
            List of dicts, each with keys: title, text, score, url.
        """
        n_titles = top_k_titles or self.top_k_titles
        n_passages = top_k_passages or self.top_k_passages

        all_passages: List[Tuple[str, str, str]] = []  # (title, passage, url)

        # 1. Wikipedia Retrieval
        if self.retriever_source in ["wikipedia", "all"]:
            titles = wikipedia_search_titles(query, top_k=n_titles)
            for title in titles:
                text = self._get_page_text(title, is_url=False)
                if not text:
                    continue
                chunks = chunk_text(
                    text, self.chunk_size, self.chunk_overlap
                )
                url = self.WIKI_BASE_URL + title.replace(" ", "_")
                for chunk in chunks:
                    all_passages.append((title, chunk, url))

        # 2. General Web Retrieval (DuckDuckGo)
        if self.retriever_source in ["web", "all"]:
            web_results = search_duckduckgo(query, top_k=n_titles)
            for title, url in web_results:
                text = self._get_page_text(url, is_url=True)
                if not text:
                    continue
                chunks = chunk_text(
                    text, self.chunk_size, self.chunk_overlap
                )
                for chunk in chunks:
                    all_passages.append((title, chunk, url))

        if not all_passages:
            logger.warning("No passages retrieved for query: %s", query)
            return []

        # 3. Rank all passages with BM25
        passage_texts = [p[1] for p in all_passages]
        ranked = bm25_rank(query, passage_texts, top_k=n_passages)

        # 4. Re-attach metadata and build result dicts
        text_to_meta: Dict[str, Tuple[str, str]] = {}
        for title, passage, url in all_passages:
            if passage not in text_to_meta:
                text_to_meta[passage] = (title, url)

        results: List[Dict] = []
        for score, passage in ranked:
            title, url = text_to_meta.get(passage, ("Unknown", ""))
            results.append(
                {
                    "title": title,
                    "text": passage,
                    "score": float(score),
                    "url": url,
                }
            )
        return results

    def retrieve_batch(
        self,
        queries: List[str],
        delay: float = 0.5,
        **kwargs,
    ) -> List[List[Dict]]:
        """Run retrieval for a list of queries with an inter-request delay."""
        all_results = []
        for i, query in enumerate(queries):
            logger.info("[%d/%d] Retrieving: %s", i + 1, len(queries), query)
            results = self.retrieve(query, **kwargs)
            all_results.append(results)
            if i < len(queries) - 1:
                time.sleep(delay)
        return all_results


# ===========================================================================
# Section 4 -- Augmented test pipeline
# ===========================================================================

def augment_eval_file_with_retrieval(
    eval_file: Dict,
    retriever: InternetRetriever,
    max_questions: Optional[int] = None,
    delay: float = 0.3,
) -> Dict:
    """Inject retrieved passages into a HotpotQA eval-file dict."""
    augmented: Dict = {}
    items = list(eval_file.items())
    if max_questions:
        items = items[:max_questions]

    for i, (qid, entry) in enumerate(items):
        question = entry.get("question", "")
        logger.info("[%d/%d] Augmenting qid %s", i + 1, len(items), qid)

        retrieved = retriever.retrieve(question)

        # Build new context paragraphs in HotpotQA format:
        # [[title, [sentence1, sentence2, ...]], ...]
        new_context = list(entry.get("context", []))
        for passage in retrieved:
            sentences = split_into_sentences(passage["text"])
            new_context.insert(0, [passage["title"], sentences])

        new_entry = dict(entry)
        new_entry["context"] = new_context
        augmented[qid] = new_entry

        if i < len(items) - 1:
            time.sleep(delay)

    return augmented


def retrieve_and_save(
    eval_file_path: str,
    output_path: str,
    retriever: InternetRetriever,
    max_questions: Optional[int] = None,
) -> None:
    """Load an eval file, augment it with retrieval, and write to output."""
    with open(eval_file_path, "r", encoding="utf-8") as fh:
        eval_file = json.load(fh)

    augmented = augment_eval_file_with_retrieval(
        eval_file, retriever, max_questions=max_questions
    )

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(augmented, fh, indent=2)

    logger.info("Augmented eval file saved to %s", output_path)


# ===========================================================================
# Section 5 -- Interactive retrieval mode
# ===========================================================================

def interactive_retrieve(retriever: InternetRetriever) -> None:
    """Simple REPL: type a question, get retrieved passages."""
    print("\nHotpotQA Internet Retriever  (type 'quit' to exit)\n")
    while True:
        try:
            query = input("Question> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query or query.lower() in {"quit", "exit", "q"}:
            break

        results = retriever.retrieve(query)
        if not results:
            print("  [No results found]\n")
            continue

        for rank, res in enumerate(results, 1):
            print(f"  [{rank}] {res['title']}  (score={res['score']:.3f})")
            print(f"      {res['url']}")
            snippet = (
                res['text'][:300] + '...'
                if len(res['text']) > 300
                else res['text']
            )
            print(f"      {snippet}")
            print()


# ===========================================================================
# Section 6 -- Argument parser & main
# ===========================================================================

glove_word_file    = "glove.840B.300d.txt"
word_emb_file      = "word_emb.json"
char_emb_file      = "char_emb.json"
train_eval         = "train_eval.json"
dev_eval           = "dev_eval.json"
test_eval          = "test_eval.json"
word2idx_file      = "word2idx.json"
char2idx_file      = "char2idx.json"
idx2word_file      = "idx2word.json"
idx2char_file      = "idx2char.json"
train_record_file  = "train_record.pkl"
dev_record_file    = "dev_record.pkl"
test_record_file   = "test_record.pkl"

parser = argparse.ArgumentParser(
    description="HotpotQA main with internet retriever augmentation."
)

# Original pipeline args
parser.add_argument("--mode", type=str, default="train",
                    choices=["train", "prepro", "test",
                             "retrieve", "retrieve_test"],
                    help="Pipeline mode.")
parser.add_argument("--data_file",        type=str)
parser.add_argument("--glove_word_file",  type=str, default=glove_word_file)
parser.add_argument("--save",             type=str, default="HOTPOT")
parser.add_argument("--word_emb_file",    type=str, default=word_emb_file)
parser.add_argument("--char_emb_file",    type=str, default=char_emb_file)
parser.add_argument("--train_eval_file",  type=str, default=train_eval)
parser.add_argument("--dev_eval_file",    type=str, default=dev_eval)
parser.add_argument("--test_eval_file",   type=str, default=test_eval)
parser.add_argument("--word2idx_file",    type=str, default=word2idx_file)
parser.add_argument("--char2idx_file",    type=str, default=char2idx_file)
parser.add_argument("--idx2word_file",    type=str, default=idx2word_file)
parser.add_argument("--idx2char_file",    type=str, default=idx2char_file)
parser.add_argument("--train_record_file",type=str, default=train_record_file)
parser.add_argument("--dev_record_file",  type=str, default=dev_record_file)
parser.add_argument("--test_record_file", type=str, default=test_record_file)
parser.add_argument("--glove_char_size",  type=int, default=94)
parser.add_argument("--glove_word_size",  type=int, default=int(2.2e6))
parser.add_argument("--glove_dim",        type=int, default=300)
parser.add_argument("--char_dim",         type=int, default=8)
parser.add_argument("--para_limit",       type=int, default=1000)
parser.add_argument("--ques_limit",       type=int, default=80)
parser.add_argument("--sent_limit",       type=int, default=100)
parser.add_argument("--char_limit",       type=int, default=16)
parser.add_argument("--batch_size",       type=int, default=64)
parser.add_argument("--checkpoint",       type=int, default=1000)
parser.add_argument("--period",           type=int, default=100)
parser.add_argument("--init_lr",          type=float, default=0.5)
parser.add_argument("--keep_prob",        type=float, default=0.8)
parser.add_argument("--hidden",           type=int, default=80)
parser.add_argument("--char_hidden",      type=int, default=100)
parser.add_argument("--patience",         type=int, default=1)
parser.add_argument("--seed",             type=int, default=13)
parser.add_argument("--sp_lambda",        type=float, default=0.0)
parser.add_argument("--data_split",       type=str, default="train")
parser.add_argument("--fullwiki",         action="store_true")
parser.add_argument("--prediction_file",  type=str)
parser.add_argument("--sp_threshold",     type=float, default=0.3)

# Retriever-specific args
parser.add_argument("--ret_source",         type=str,   default="all",
                    choices=["wikipedia", "web", "all"],
                    help="Retrieval source: wikipedia, web (DuckDuckGo), "
                         "or all.")
parser.add_argument("--ret_top_k_titles",   type=int,   default=5,
                    help="Number of source articles/websites to retrieve "
                         "per question.")
parser.add_argument("--ret_top_k_passages", type=int,   default=3,
                    help="Number of passages to keep after BM25 ranking.")
parser.add_argument("--ret_chunk_size",     type=int,   default=5,
                    help="Sentences per passage chunk.")
parser.add_argument("--ret_chunk_overlap",  type=int,   default=1,
                    help="Sentence overlap between consecutive chunks.")
parser.add_argument("--ret_use_full_text",  action="store_true",
                    help="Fetch full article text instead of intro only "
                         "(Wikipedia only).")
parser.add_argument("--ret_cache_dir",      type=str,   default=None,
                    help="Directory to cache fetched article/webpage text.")
parser.add_argument("--ret_augmented_eval", type=str,   default=None,
                    help="Output path for the retrieval-augmented eval file "
                         "(retrieve_test mode).")
parser.add_argument("--ret_max_questions",  type=int,   default=None,
                    help="Limit retrieval to this many questions (debug).")
parser.add_argument("--ret_delay",          type=float, default=0.3,
                    help="Seconds between HTTP/API calls.")

config = parser.parse_args()


def _concat(filename: str) -> str:
    if config.fullwiki:
        return "fullwiki.{}".format(filename)
    return filename


config.dev_record_file  = _concat(config.dev_record_file)
config.test_record_file = _concat(config.test_record_file)
config.dev_eval_file    = _concat(config.dev_eval_file)
config.test_eval_file   = _concat(config.test_eval_file)


def _build_retriever() -> InternetRetriever:
    return InternetRetriever(
        top_k_titles=config.ret_top_k_titles,
        top_k_passages=config.ret_top_k_passages,
        chunk_size=config.ret_chunk_size,
        chunk_overlap=config.ret_chunk_overlap,
        use_full_text=config.ret_use_full_text,
        cache_dir=config.ret_cache_dir,
        retriever_source=config.ret_source,
    )


if config.mode == "train":
    train(config)
elif config.mode == "prepro":
    prepro(config)
elif config.mode == "test":
    test(config)
elif config.mode == "retrieve":
    # Interactive retrieval REPL
    interactive_retrieve(_build_retriever())
elif config.mode == "retrieve_test":
    # Retrieve passages, augment the eval file, then run inference
    retriever = _build_retriever()
    augmented_eval_path = config.ret_augmented_eval or (
        config.test_eval_file.replace(".json", ".retrieved.json")
    )
    retrieve_and_save(
        config.test_eval_file,
        augmented_eval_path,
        retriever,
        max_questions=config.ret_max_questions,
    )
    # Swap in the augmented eval file and run the normal test pipeline
    config.test_eval_file = augmented_eval_path
    test(config)
