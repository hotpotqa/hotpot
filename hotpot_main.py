import os
import argparse
import urllib.request
import urllib.parse
import json
import re
import time
from collections import Counter
from prepro import prepro
from run import train, test

# ----------------------------------------------------------------------
# INTERNET RETRIEVER CLASS AND HELPERS
# ----------------------------------------------------------------------

def split_sentences(text):
    """
    Splits raw article text into sentences, handling common abbreviations
    so that they don't cause premature splits (e.g., 'U.S.', 'Mr.').
    """
    # Remove Wikipedia-style headings like == History ==
    text = re.sub(r'==+.*?==+', '', text)
    # Normalize multiple spaces and newlines
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Common abbreviations to ignore during splitting
    abbrevs = (
        r'(?<!\b(?:Mr|St|Dr|Ms|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec'
        r'|vs|ca|approx|est|etc|dept|univ|gen|capt|col|U\.S|U\.K|Inc|Co|e\.g|i\.e))\.'
    )
    # Sentence splitter pattern looking for end punctuation followed by spaces and a capital letter or number
    pattern = re.compile(abbrevs + r'(?=\s+[A-Z0-9]|\s*$)')
    
    sentences = []
    start = 0
    for match in pattern.finditer(text):
        end = match.end()
        sentences.append(text[start:end].strip())
        start = end
    if start < len(text):
        last_chunk = text[start:].strip()
        if last_chunk:
            sentences.append(last_chunk)
            
    # Filter out empty or too short sentences
    return [s for s in sentences if len(s) > 3]


class WikipediaRetriever:
    """
    Retrieves background context articles from Wikipedia using the official
    zero-authentication MediaWiki API. Highly effective for HotpotQA.
    """
    def __init__(self):
        self.api_url = "https://en.wikipedia.org/w/api.php"
        
    def search(self, query, limit=5):
        """Search Wikipedia for titles matching the query."""
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
            "format": "json",
            "utf8": 1
        }
        url = f"{self.api_url}?{urllib.parse.urlencode(params)}"
        headers = {"User-Agent": "HotpotQA-InternetRetriever/1.0 (academic-research@example.com)"}
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read().decode("utf-8"))
                search_results = data.get("query", {}).get("search", [])
                return [res["title"] for res in search_results]
        except Exception as e:
            print(f"[WikipediaRetriever] Error searching for query '{query}': {e}")
            return []

    def get_article_content(self, title):
        """Fetch the clean text of a Wikipedia article."""
        params = {
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "titles": title,
            "format": "json",
            "utf8": 1
        }
        url = f"{self.api_url}?{urllib.parse.urlencode(params)}"
        headers = {"User-Agent": "HotpotQA-InternetRetriever/1.0 (academic-research@example.com)"}
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read().decode("utf-8"))
                pages = data.get("query", {}).get("pages", {})
                for page_id, page in pages.items():
                    if "extract" in page:
                        return page["extract"]
        except Exception as e:
            print(f"[WikipediaRetriever] Error fetching content for '{title}': {e}")
        return ""


class DuckDuckGoRetriever:
    """
    A zero-key web retriever using DuckDuckGo HTML interface.
    Fetches snippets and follows search result links.
    """
    def __init__(self):
        self.search_url = "https://html.duckduckgo.com/html/"
        
    def search(self, query, limit=5):
        """Search DuckDuckGo and return list of result dicts containing title and url."""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        data = urllib.parse.urlencode({"q": query}).encode("utf-8")
        try:
            req = urllib.request.Request(self.search_url, data=data, headers=headers)
            with urllib.request.urlopen(req) as response:
                html_content = response.read().decode("utf-8", errors="ignore")
                
                # Simple regex parser for titles and URLs
                pattern = re.compile(r'<a class="result__a" href="([^"]+)">([^<]+)</a>')
                matches = pattern.findall(html_content)
                
                results = []
                for href, title in matches[:limit]:
                    url = urllib.parse.unquote(href)
                    if "/l/?" in url:
                        parsed = urllib.parse.urlparse(url)
                        queries = urllib.parse.parse_qs(parsed.query)
                        if "uddg" in queries:
                            url = queries["uddg"][0]
                    results.append({"title": title.strip(), "url": url})
                return results
        except Exception as e:
            print(f"[DuckDuckGoRetriever] Error searching for query '{query}': {e}")
            return []

    def get_page_content(self, url):
        """Fetch general page content, strip HTML tags, and return text."""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=8) as response:
                raw_html = response.read().decode("utf-8", errors="ignore")
                raw_html = re.sub(r'<(script|style).*?>.*?</\1>', '', raw_html, flags=re.DOTALL|re.IGNORECASE)
                text = re.sub(r'<[^>]+>', ' ', raw_html)
                text = re.sub(r'\s+', ' ', text)
                return text.strip()
        except Exception as e:
            return ""


class InternetRetriever:
    """
    Unified entry point for retrieving HotpotQA-compliant contexts
    from either Wikipedia or DuckDuckGo Web search.
    """
    def __init__(self, mode="wikipedia"):
        self.mode = mode.lower()
        if self.mode == "wikipedia":
            self.retriever = WikipediaRetriever()
        else:
            self.retriever = DuckDuckGoRetriever()
            
    def retrieve_context(self, question, limit=10):
        """
        Takes an input question, retrieves the top pages, segments them,
        and formats them into the standard HotpotQA context format.
        """
        formatted_context = []
        if self.mode == "wikipedia":
            print(f"[*] Retrieving up to {limit} pages from Wikipedia...")
            titles = self.retriever.search(question, limit=limit)
            for title in titles:
                content = self.retriever.get_article_content(title)
                if content:
                    sentences = split_sentences(content)
                    sentences = sentences[:25]  # Cap sentences per article
                    formatted_context.append([title, sentences])
        else:
            print(f"[*] Retrieving up to {limit} web pages via DuckDuckGo...")
            results = self.retriever.search(question, limit=limit)
            for res in results:
                title = res["title"]
                url = res["url"]
                content = self.retriever.get_page_content(url)
                if content:
                    sentences = split_sentences(content)
                    sentences = sentences[:25]
                    formatted_context.append([title, sentences])
                    
        return formatted_context


# ----------------------------------------------------------------------
# MULTI-HOP (TWO-HOP) INTERNET RETRIEVER FUNCTIONS
# ----------------------------------------------------------------------
# HotpotQA questions require reasoning over *two* Wikipedia articles
# (bridge: "the X that did Y"; comparison: "are A and B both Z?"). A
# single-hop search frequently retrieves only the seed article, missing
# the bridge entity's page. The functions below implement a lightweight
# two-hop retriever on top of the MediaWiki API:
#
#   hop 1: search -> fetch best page
#   hop 2: follow that page's outgoing links, re-rank them against the
#          question, fetch the best bridge pages
#
# All pages' paragraphs are ranked by question-term overlap and returned
# in the standard HotpotQA context shape: [[title, [paragraph, ...]], ...]
# Only the Python standard library is required.

WIKI_API_URL = "https://en.wikipedia.org/w/api.php"
WIKI_USER_AGENT = "HotpotQA-InternetRetriever/1.0 (academic-research@example.com)"
DEFAULT_TOP_K = 3        # paragraphs kept per retrieved page
DEFAULT_HOP_LINKS = 2    # follow-up pages fetched for the second hop

_STOPWORDS = {
    "the", "a", "an", "of", "in", "to", "and", "or", "was", "were", "is",
    "are", "who", "what", "when", "where", "which", "did", "do", "does",
    "by", "for", "on", "at", "with", "that", "this", "his", "her", "their",
    "its", "he", "she", "they", "as", "from", "be", "been", "has", "had",
}


def _wiki_request(params, timeout=15, retries=3):
    """Issue a GET request against the MediaWiki API; retry with backoff."""
    params = dict(params)
    params.setdefault("format", "json")
    params.setdefault("formatversion", "2")
    url = "{}?{}".format(WIKI_API_URL, urllib.parse.urlencode(params))
    req = urllib.request.Request(url, headers={"User-Agent": WIKI_USER_AGENT})
    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as err:
            last_err = err
            time.sleep(0.5 * (2 ** attempt))
    raise RuntimeError(f"Wikipedia API request failed after {retries} attempts: {last_err}")


def search_wikipedia_titles(query, limit=5):
    """Return a list of Wikipedia page titles matching `query`."""
    data = _wiki_request({
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": limit,
    })
    return [hit["title"] for hit in data.get("query", {}).get("search", [])]


def fetch_wikipedia_page(title, num_paragraphs=10):
    """Fetch a page's plain-text extract as (title, paragraphs).

    Returns (None, []) if the page does not exist. Paragraphs are the
    non-empty extract blocks in document order, capped at num_paragraphs.
    """
    data = _wiki_request({
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "redirects": 1,
        "titles": title,
    })
    pages = data.get("query", {}).get("pages", [])
    if not pages:
        return None, []
    page = pages[0]
    if "extract" not in page:
        return None, []
    paragraphs = [p.strip() for p in page["extract"].split("\n") if p.strip()]
    return page.get("title", title), paragraphs[:num_paragraphs]


def fetch_wikipedia_links(title, limit=10):
    """Return up to `limit` main-namespace outgoing link titles from a page.

    These are the candidate second-hop (bridge) pages.
    """
    data = _wiki_request({
        "action": "query",
        "prop": "links",
        "titles": title,
        "pllimit": "max",
        "plnamespace": 0,
    })
    pages = data.get("query", {}).get("pages", [])
    if not pages:
        return []
    return [l["title"] for l in pages[0].get("links", [])][:limit]


def term_overlap_score(text, question):
    """Content-word overlap count between `text` and `question` (stopwords removed)."""
    q_tokens = Counter(t for t in re.findall(r"\w+", question.lower())
                       if t not in _STOPWORDS and len(t) > 1)
    p_tokens = Counter(re.findall(r"\w+", text.lower()))
    if not q_tokens:
        return 0
    return sum(min(q_tokens[t], p_tokens.get(t, 0)) for t in q_tokens)


def rank_paragraphs(paragraphs, question, top_k=DEFAULT_TOP_K):
    """Rank paragraphs by question-term overlap and return the top-k."""
    scored = sorted(paragraphs, key=lambda p: term_overlap_score(p, question),
                    reverse=True)
    positive = [p for p in scored if term_overlap_score(p, question) > 0]
    return (positive or scored)[:top_k]


def extract_bridge_titles(question, first_hop_title, hop_links=DEFAULT_HOP_LINKS):
    """Pick second-hop page titles for a bridge/comparison question.

    Takes the first-hop page's outgoing Wikipedia links and re-ranks them
    by token overlap with the question (e.g. for "What position was held by
    the woman who voiced Anastasia in the Don Bluth film?", the first hop is
    the film/anastasia page and links like "Meg Ryan" / "John Cusack" rank
    highest).
    """
    candidates = fetch_wikipedia_links(first_hop_title, limit=hop_links + 8)
    candidates.sort(key=lambda t: term_overlap_score(t.replace("_", " "), question),
                    reverse=True)
    return candidates[:hop_links]


def retrieve_two_hop(question, num_hops=2, top_k=DEFAULT_TOP_K,
                     hop_links=DEFAULT_HOP_LINKS):
    """Run multi-hop internet retrieval for a single question.

    Returns a HotpotQA-style dict:
        {
          "question": str,
          "context": [[title, [paragraph, ...]], ...],
          "visited_titles": [str, ...],
        }
    """
    context, visited = [], []

    seed_titles = search_wikipedia_titles(question, limit=3)
    if not seed_titles:
        return {"question": question, "context": context, "visited_titles": visited}

    first_title, first_paras = fetch_wikipedia_page(seed_titles[0])
    if first_title:
        visited.append(first_title)
        context.append([first_title, rank_paragraphs(first_paras, question, top_k)])

    if num_hops >= 2 and first_title:
        for bridge in extract_bridge_titles(question, first_title, hop_links):
            if bridge in visited:
                continue
            t, paras = fetch_wikipedia_page(bridge)
            if t and t not in visited:
                visited.append(t)
                context.append([t, rank_paragraphs(paras, question, top_k)])

    return {"question": question, "context": context, "visited_titles": visited}


def retrieve_context_multihop(question, num_hops=2, top_k=DEFAULT_TOP_K):
    """Convenience wrapper: return (titles, paragraphs) parallel lists.

    Matches the (title, [sentences]) pairs the HotpotQA preprocessing
    pipeline consumes.
    """
    result = retrieve_two_hop(question, num_hops=num_hops, top_k=top_k)
    titles = [c[0] for c in result["context"]]
    paragraphs = [c[1] for c in result["context"]]
    return titles, paragraphs


def retrieve_batch_two_hop(questions, num_hops=2, top_k=DEFAULT_TOP_K,
                           output_file=None):
    """Retrieve two-hop contexts for multiple questions (1s polite pause)."""
    results = []
    for i, q in enumerate(questions):
        results.append(retrieve_two_hop(q, num_hops=num_hops, top_k=top_k))
        if i < len(questions) - 1:
            time.sleep(1.0)
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    return results


def run_live_retrieval(question, mode="wikipedia", limit=5, output_file=None,
                       num_hops=2, top_k=DEFAULT_TOP_K):
    """Runs retrieval on a single question, formats and prints/saves it.

    `mode="multihop"` (or "twohop") runs the two-hop Wikipedia retriever;
    "wikipedia"/"duckduckgo" run the original single-hop retrievers.
    """
    if mode in ("multihop", "twohop", "two_hop"):
        print(f"[*] Running {num_hops}-hop Wikipedia retrieval...")
        result = retrieve_two_hop(question, num_hops=num_hops, top_k=top_k)
        context = result["context"]
    else:
        retriever = InternetRetriever(mode=mode)
        context = retriever.retrieve_context(question, limit=limit)
        result = {
            "_id": "live_question_id",
            "question": question,
            "context": context,
            "supporting_facts": [],
            "visited_titles": [c[0] for c in context],
        }

    output_data = {
        "_id": "live_question_id",
        "question": question,
        "context": context,
        "supporting_facts": [],
        "visited_titles": result.get("visited_titles", [c[0] for c in context]),
    }

    print("\n" + "=" * 50)
    print(f"QUESTION: {question}")
    print("=" * 50)
    print(f"Retrieved {len(context)} documents:")
    for i, (title, sents) in enumerate(context, 1):
        print(f"  {i}. {title} ({len(sents)} passages)")
        if sents:
            print(f"     Snippet: {sents[0][:150]}...")
    print("=" * 50)

    if output_file:
        with open(output_file, "w") as f:
            json.dump([output_data], f, indent=2)
        print(f"[*] Saved retrieved context to {output_file}")

    return output_data


# ----------------------------------------------------------------------
# ARGUMENT PARSING AND MAIN PIPELINE INTEGRATION
# ----------------------------------------------------------------------

parser = argparse.ArgumentParser()

glove_word_file = "glove.840B.300d.txt"
word_emb_file = "word_emb.json"
char_emb_file = "char_emb.json"
train_eval = "train_eval.json"
dev_eval = "dev_eval.json"
test_eval = "test_eval.json"
word2idx_file = "word2idx.json"
char2idx_file = "char2idx.json"
idx2word_file = 'idx2word.json'
idx2char_file = 'idx2char.json'
train_record_file = 'train_record.pkl'
dev_record_file = 'dev_record.pkl'
test_record_file = 'test_record.pkl'

parser.add_argument('--mode', type=str, default='train', 
                    help='Execution mode: train, test, prepro, count, or retrieve')
parser.add_argument('--data_file', type=str)
parser.add_argument('--glove_word_file', type=str, default=glove_word_file)
parser.add_argument('--save', type=str, default='HOTPOT')

parser.add_argument('--word_emb_file', type=str, default=word_emb_file)
parser.add_argument('--char_emb_file', type=str, default=char_emb_file)
parser.add_argument('--train_eval_file', type=str, default=train_eval)
parser.add_argument('--dev_eval_file', type=str, default=dev_eval)
parser.add_argument('--test_eval_file', type=str, default=test_eval)
parser.add_argument('--word2idx_file', type=str, default=word2idx_file)
parser.add_argument('--char2idx_file', type=str, default=char2idx_file)
parser.add_argument('--idx2word_file', type=str, default=idx2word_file)
parser.add_argument('--idx2char_file', type=str, default=idx2char_file)

parser.add_argument('--train_record_file', type=str, default=train_record_file)
parser.add_argument('--dev_record_file', type=str, default=dev_record_file)
parser.add_argument('--test_record_file', type=str, default=test_record_file)

parser.add_argument('--glove_char_size', type=int, default=94)
parser.add_argument('--glove_word_size', type=int, default=int(2.2e6))
parser.add_argument('--glove_dim', type=int, default=300)
parser.add_argument('--char_dim', type=int, default=8)

parser.add_argument('--para_limit', type=int, default=1000)
parser.add_argument('--ques_limit', type=int, default=80)
parser.add_argument('--sent_limit', type=int, default=100)
parser.add_argument('--char_limit', type=int, default=16)

parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--checkpoint', type=int, default=1000)
parser.add_argument('--period', type=int, default=100)
parser.add_argument('--init_lr', type=float, default=0.5)
parser.add_argument('--keep_prob', type=float, default=0.8)
parser.add_argument('--hidden', type=int, default=80)
parser.add_argument('--char_hidden', type=int, default=100)
parser.add_argument('--patience', type=int, default=1)
parser.add_argument('--seed', type=int, default=13)

parser.add_argument('--sp_lambda', type=float, default=0.0)
parser.add_argument('--data_split', type=str, default='train')
parser.add_argument('--fullwiki', action='store_true')
parser.add_argument('--prediction_file', type=str)
parser.add_argument('--sp_threshold', type=float, default=0.3)

# Arguments for Internet Retrieval
parser.add_argument('--query', type=str, help='Question to query when mode is retrieve')
parser.add_argument('--retriever_type', type=str, default='multihop',
                    choices=['wikipedia', 'duckduckgo', 'multihop', 'twohop'],
                    help='Retriever backend: wikipedia, duckduckgo, or multihop '
                         '(two-hop Wikipedia retriever; default)')
parser.add_argument('--retrieve_limit', type=int, default=5,
                    help='Maximum number of web pages / articles to retrieve (single-hop modes)')
parser.add_argument('--num_hops', type=int, default=2,
                    help='Number of retrieval hops for the multihop retriever')
parser.add_argument('--top_k', type=int, default=DEFAULT_TOP_K,
                    help='Paragraphs to keep per retrieved page (multihop retriever)')
parser.add_argument('--questions_file', type=str, default=None,
                    help='JSON/JSON-lines file of questions for batch two-hop retrieval')

config = parser.parse_args()

def _concat(filename):
    if config.fullwiki:
        return 'fullwiki.{}'.format(filename)
    return filename

config.dev_record_file = _concat(config.dev_record_file)
config.test_record_file = _concat(config.test_record_file)
config.dev_eval_file = _concat(config.dev_eval_file)
config.test_eval_file = _concat(config.test_eval_file)

def main():
    if config.mode == 'train':
        train(config)
    elif config.mode == 'prepro':
        prepro(config)
    elif config.mode == 'test':
        test(config)
    elif config.mode == 'count':
        cnt_len(config)
    elif config.mode == 'retrieve':
        if config.questions_file:
            # Batch two-hop retrieval over a file of questions
            questions = []
            with open(config.questions_file, 'r', encoding='utf-8') as f:
                if config.questions_file.endswith('.jsonl'):
                    for line in f:
                        line = line.strip()
                        if line:
                            item = json.loads(line)
                            questions.append(item['question'] if isinstance(item, dict) else item)
                else:
                    for item in json.load(f):
                        questions.append(item['question'] if isinstance(item, dict) else item)
            results = retrieve_batch_two_hop(
                questions, num_hops=config.num_hops, top_k=config.top_k,
                output_file=config.data_file)
            print(json.dumps(results, ensure_ascii=False, indent=2))
        elif config.query:
            run_live_retrieval(
                question=config.query,
                mode=config.retriever_type,
                limit=config.retrieve_limit,
                output_file=config.data_file,
                num_hops=config.num_hops,
                top_k=config.top_k,
            )
        else:
            print("[-] Error: provide a query with --query or a file with --questions_file.")

if __name__ == '__main__':
    main()
