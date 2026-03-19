"""
Fetches and extracts text from a webpage using requests and BeautifulSoup.
"""

from bs4 import BeautifulSoup
import requests


def fetch_webpage_html(url: str) -> str:
    """
    The generic function to fetch a webpage in its pure HTML text form.
    """

    return requests.get(url).text


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

        return {"text": extract_text_from_html(html)}
    except requests.RequestException as e:
        return {"error": f"Failed to fetch the webpage: {e}"}
