"""FetchUrlTool: fetch and extract text content from a URL.

Pure fetch function — retrieves a web page or paper landing page, strips HTML to
readable text. Used to follow up on web_search hits (e.g., read a full abstract
or paper page). Returns truncated text for context economy.
"""

import json
import re
import logging

logger = logging.getLogger("retroagent")

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

_USER_AGENT = "RetroAgent/0.1 (chemistry retrosynthesis agent)"
_TIMEOUT = 20
_MAX_CHARS = 6000  # truncate to keep context manageable


class FetchUrlTool:
    name = "fetch_url"
    description = (
        "Fetch a URL and return its readable text content (HTML stripped). "
        "Use after web_search to read the full content of a promising hit "
        "(paper landing page, blog, doc). Returns up to ~6000 chars. Pure fetch."
    )

    def execute(self, parameters: dict) -> str:
        if not _HAS_REQUESTS:
            return json.dumps({"error": "requests library not installed"}, ensure_ascii=False)
        url = (parameters.get("url") or "").strip()
        if not url:
            return json.dumps({"error": "Missing 'url'"}, ensure_ascii=False)
        try:
            r = requests.get(url, headers={"User-Agent": _USER_AGENT}, timeout=_TIMEOUT)
            if not r.ok:
                return json.dumps({"error": f"HTTP {r.status_code}", "url": url}, ensure_ascii=False)
            content_type = r.headers.get("Content-Type", "")
            if "json" in content_type:
                text = r.text[:_MAX_CHARS]
            else:
                text = _html_to_text(r.text)[:_MAX_CHARS]
            return json.dumps({
                "url": url,
                "status": r.status_code,
                "content_type": content_type,
                "char_count": len(text),
                "content": text,
            }, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e), "url": url}, ensure_ascii=False)

    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to fetch (from a web_search result)"},
            },
            "required": ["url"],
        }


def _html_to_text(html: str) -> str:
    """Crude HTML → text: drop scripts/styles/tags, collapse whitespace."""
    # Remove script/style blocks
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
    # Drop all tags
    text = re.sub(r"<[^>]+>", " ", html)
    # Decode common entities
    text = (text.replace("&nbsp;", " ").replace("&amp;", "&")
                .replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"'))
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text
