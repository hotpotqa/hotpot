import os
import argparse
import urllib.request
import urllib.parse
import json
import re
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
        data = urllib.parse.encode({"q": query}).encode("utf-8") if hasattr(urllib.parse, "encode") else urllib.parse.urlencode({"q": query}).encode("utf-8")
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
                content = self.get_page_content(url) if hasattr(self, 'get_page_content') else self.retriever.get_page_content(url)
                if content:
                    sentences = split_sentences(content)
                    sentences = sentences[:25]
                    formatted_context.append([title, sentences])
                    
        return formatted_context


def run_live_retrieval(question, mode="wikipedia", limit=5, output_file=None):
    """Runs retriever on a single question, formats and prints/saves it."""
    retriever = InternetRetriever(mode=mode)
    context = retriever.retrieve_context(question, limit=limit)
    
    output_data = {
        "_id": "live_question_id",
        "question": question,
        "context": context,
        "supporting_facts": []
    }
    
    print("\n" + "="*50)
    print(f"QUESTION: {question}")
    print("="*50)
    print(f"Retrieved {len(context)} documents:")
    for i, (title, sentences) in enumerate(context, 1):
        print(f"  {i}. {title} ({len(sentences)} sentences)")
        if sentences:
            print(f"     Snippet: {sentences[0][:150]}...")
    print("="*50)
    
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

# New arguments for Internet Retrieval
parser.add_argument('--query', type=str, help='Question to query when mode is retrieve')
parser.add_argument('--retriever_type', type=str, default='wikipedia', choices=['wikipedia', 'duckduckgo'],
                    help='Which retriever backend to use: wikipedia or duckduckgo')
parser.add_argument('--retrieve_limit', type=int, default=5,
                    help='Maximum number of web pages / articles to retrieve')

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
        if not config.query:
            print("[-] Error: Please provide a query using the --query argument.")
            return
        run_live_retrieval(
            question=config.query,
            mode=config.retriever_type,
            limit=config.retrieve_limit,
            output_file=config.data_file
        )

if __name__ == '__main__':
    main()
