"""
Fetches and extracts text from a webpage using requests and BeautifulSoup.
"""

from bs4 import BeautifulSoup
import requests
from logging_manager import logger


def fetch_webpage_html(url: str) -> str:
    """
    The generic function to fetch a webpage in its pure HTML text form.
    """

    logger.debug("Fetching HTML for URL: %s", url)
    resp = requests.get(url, headers={
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "accept-encoding": "gzip, deflate, br, zstd",
        "accept-language": "en-US,en;q=0.7",
        "cache-control": "no-cache",
        "pragma": "no-cache",
        "priority": "u=0, i",
        "referer": "https://search.brave.com/",
        "sec-ch-ua": '"Brave";v="149", "Chromium";v="149", "Not)A;Brand";v="24"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Linux"',
        "sec-fetch-dest": "document",
        "sec-fetch-mode": "navigate",
        "sec-fetch-site": "cross-site",
        "sec-fetch-user": "?1",
        "sec-gpc": "1",
        "upgrade-insecure-requests": "1",
        "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36"
    })
    resp.raise_for_status()
    
    return resp.text


def extract_text_from_html(html: str) -> str:
    """
    Extracts text content from HTML using BeautifulSoup.
    """

    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n", strip=True)


def fetch_webpage_content_tool(args: dict[str, str]) -> dict:
    """
    Fetches a webpage using requests and extracts the text content only using
    BeautifulSoup.
    """

    if not isinstance(args, dict):
        logger.error("fetch_webpage: args not a dict: %s", type(args))
        raise ValueError("Arguments for the fetch_webpage tool must be a JSON object.")

    if "url" not in args:
        raise ValueError("Missing 'url' key in arguments for the fetch_webpage tool.")

    url = args.get("url", "").strip()

    if not url.startswith("http://") and not url.startswith("https://"):
        return {
            "error": "Invalid URL. Please provide a URL starting with http:// or https://."
        }

    try:
        html = fetch_webpage_html(url)
        text = extract_text_from_html(html)
        logger.debug(
            "Fetched and extracted text length=%d for url=%s", len(text or ""), url
        )

        # limit text length to 4096 characters to avoid overwhelming the model
        # with too much content
        if len(text) > 4096:
            text = text[:4096] + "... [truncated]"

        return {"text": text}
    except requests.RequestException as e:
        logger.exception("Failed to fetch webpage: %s", url)
        return {"error": f"Failed to fetch the webpage: {e}"}
