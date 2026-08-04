"""Internet retrieval helpers for HotpotQA.

The retriever uses DuckDuckGo's HTML endpoint and the Python standard library,
so it does not require an API key or third-party package. For production use,
replace ``search_web`` with a licensed search provider as appropriate.
"""

import argparse
import json
import re
from html import unescape
from urllib.parse import parse_qs, quote_plus, unquote, urljoin, urlparse
from urllib.request import Request, urlopen


DEFAULT_USER_AGENT = "HotpotQA-internet-retriever/1.0"


def _visible_text(html):
    """Return readable text from an HTML document or fragment."""
    html = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\\1>", " ", html)
    html = re.sub(r"(?s)<[^>]+>", " ", html)
    return re.sub(r"\\s+", " ", unescape(html)).strip()


def _unwrap_search_url(url):
    """Unwrap DuckDuckGo redirect URLs when possible."""
    parsed = urlparse(url)
    target = parse_qs(parsed.query).get("uddg", [None])[0]
    return unquote(target) if target else url


def fetch_url_text(url, timeout=10, user_agent=DEFAULT_USER_AGENT):
    """Fetch a public URL and return its visible text.

    Raises the underlying URL/decoding exception so callers can decide whether
    to skip the page or use its search-result snippet as a fallback.
    """
    request = Request(url, headers={"User-Agent": user_agent})
    with urlopen(request, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return _visible_text(response.read().decode(charset, errors="replace"))


def search_web(query, top_k=5, timeout=10, user_agent=DEFAULT_USER_AGENT):
    """Search the web and return ``title``, ``url`` and ``snippet`` records."""
    if not query or top_k <= 0:
        return []

    endpoint = "https://html.duckduckgo.com/html/?q={}".format(quote_plus(query))
    request = Request(endpoint, headers={"User-Agent": user_agent})
    with urlopen(request, timeout=timeout) as response:
        html = response.read().decode("utf-8", errors="replace")

    results = []
    pattern = re.compile(
        r'''(?is)<a[^>]+class=["'][^"']*result__a[^"']*["'][^>]+href=["']([^"']+)["'][^>]*>(.*?)</a>'''
    )
    for match in pattern.finditer(html):
        url, title_html = match.groups()
        url = _unwrap_search_url(url)
        if url.startswith("//"):
            url = "https:" + url
        elif not url.startswith(("http://", "https://")):
            url = urljoin("https://html.duckduckgo.com", url)

        # Search result snippets usually occur immediately after the title.
        window = html[match.end():match.end() + 1800]
        snippet_match = re.search(
            r'''(?is)class=["'][^"']*result__snippet[^"']*["'][^>]*>(.*?)</''',
            window,
        )
        results.append({
            "title": _visible_text(title_html),
            "url": url,
            "snippet": _visible_text(snippet_match.group(1)) if snippet_match else "",
        })
        if len(results) >= top_k:
            break
    return results


def _split_sentences(text):
    return [part.strip() for part in re.split(r"(?<=[.!?])\\s+", text) if part.strip()]


def retrieve_internet(question, top_k=5, max_chars=12000, timeout=10):
    """Retrieve web pages as HotpotQA-style context records.

    Each record has ``title``, ``sentences`` and ``url`` fields. Pages that
    cannot be fetched are retained when a search snippet is available.
    """
    contexts = []
    for result in search_web(question, top_k=top_k, timeout=timeout):
        try:
            text = fetch_url_text(result["url"], timeout=timeout)
        except Exception:
            text = result.get("snippet", "")
        if not text:
            continue
        contexts.append({
            "title": result["title"] or result["url"],
            "sentences": _split_sentences(text[:max_chars]),
            "url": result["url"],
        })
    return contexts


def retrieve_to_hotpot_example(question, top_k=5, max_chars=12000, timeout=10):
    """Build a minimal dataset-compatible example from live search results."""
    contexts = retrieve_internet(question, top_k, max_chars, timeout)
    return {
        "_id": "internet",
        "question": question,
        "context": [[item["title"], item["sentences"]] for item in contexts],
        "retrieval_sources": [item["url"] for item in contexts],
    }


def main():
    parser = argparse.ArgumentParser(description="Retrieve internet context for HotpotQA")
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=10)
    parser.add_argument("--output", default="internet_retrieval.json")
    args = parser.parse_args()

    result = retrieve_to_hotpot_example(args.question, args.top_k, timeout=args.timeout)
    with open(args.output, "w") as output:
        json.dump(result, output, indent=2)
    print("Wrote {} contexts to {}".format(len(result["context"]), args.output))


if __name__ == "__main__":
    main()
