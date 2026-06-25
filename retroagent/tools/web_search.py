"""WebSearchTool: literature and web search using FREE APIs only (no API keys).

Sources (all free, no key required for low volume):
  - Crossref (https://api.crossref.org) — 150M+ works, reliable, primary source
  - Semantic Scholar (https://api.semanticscholar.org) — abstracts enrichment
    (optional; rate-limited on shared IPs, degrades gracefully)
  - PubChem PUG REST — compound properties / basic synthesis info

Claude philosophy: this tool is a pure FETCH function. Given a query, it returns
results. The model reasons over relevance and how to cite them.
"""

import json
import urllib.parse
import logging

logger = logging.getLogger("retroagent")

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

_USER_AGENT = "RetroAgent/0.1 (chemistry retrosynthesis agent; mailto:noreply@example.com)"
_TIMEOUT = 20


class WebSearchTool:
    name = "web_search"
    description = (
        "Search the scientific literature and web using FREE APIs (no key needed). "
        "Searches Crossref (150M+ papers, reliable) and Semantic Scholar (abstracts) "
        "for papers, and PubChem for compound data. Use this to check literature "
        "precedent for a reaction/catalyst, find synthesis references, or verify "
        "whether a proposed route/ligand is known. Returns titles, years, authors, "
        "abstracts (when available), and DOIs/URLs. Pure fetch — you interpret."
    )

    def execute(self, parameters: dict) -> str:
        if not _HAS_REQUESTS:
            return json.dumps({"error": "requests library not installed"}, ensure_ascii=False)

        query = (parameters.get("query") or parameters.get("constraints") or "").strip()
        if not query:
            return json.dumps({"error": "Missing 'query'"}, ensure_ascii=False)
        search_type = parameters.get("search_type", "papers")
        limit = int(parameters.get("limit", 5))

        if search_type == "compound":
            results = self._search_pubchem(query, limit)
        else:
            # papers (default): Crossref primary, S2 enrichment
            results = self._search_papers(query, limit)

        return json.dumps({
            "query": query,
            "search_type": search_type,
            "count": len(results),
            "results": results,
        }, ensure_ascii=False)

    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query (reaction name, catalyst, molecule, topic)"},
                "search_type": {
                    "type": "string", "enum": ["papers", "compound"], "default": "papers",
                    "description": "'papers' = literature search (Crossref/S2); 'compound' = PubChem compound data",
                },
                "limit": {"type": "integer", "default": 5, "description": "Max results (1-10)"},
            },
            "required": ["query"],
        }

    # ------------------------------------------------------------------

    def _search_papers(self, query: str, limit: int) -> list[dict]:
        results: list[dict] = []
        # Primary: Crossref (reliable, free)
        cr = self._safe_call(self._crossref_search, query, limit)
        results.extend(cr)
        # Enrichment: Semantic Scholar abstracts (optional, may rate-limit)
        if len([r for r in results if r.get("abstract")]) < min(3, limit):
            s2 = self._safe_call(self._s2_search, query, limit)
            results = self._merge_by_title(results, s2)
        return results[:limit]

    def _crossref_search(self, query: str, limit: int) -> list[dict]:
        r = requests.get(
            "https://api.crossref.org/works",
            params={"query": query, "rows": min(limit, 10), "select": "title,author,published,DOI,URL,abstract,container-title"},
            headers={"User-Agent": _USER_AGENT},
            timeout=_TIMEOUT,
        )
        if not r.ok:
            logger.debug(f"Crossref HTTP {r.status_code}")
            return []
        items = r.json().get("message", {}).get("items", [])
        out = []
        for it in items:
            title = (it.get("title") or [""])[0]
            if not title:
                continue
            year = None
            try:
                year = it.get("published", {}).get("date-parts", [[None]])[0][0]
            except Exception:
                pass
            authors = []
            for a in (it.get("author") or [])[:5]:
                name = f"{a.get('given','')} {a.get('family','')}".strip()
                if name:
                    authors.append(name)
            abstract = it.get("abstract")
            if abstract:
                abstract = _strip_tags(abstract)[:600]
            out.append({
                "source": "Crossref",
                "title": title,
                "year": year,
                "authors": authors,
                "venue": (it.get("container-title") or [""])[0],
                "doi": it.get("DOI"),
                "url": it.get("URL"),
                "abstract": abstract,
            })
        return out

    def _s2_search(self, query: str, limit: int) -> list[dict]:
        try:
            r = requests.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={"query": query, "limit": min(limit, 10),
                        "fields": "title,year,abstract,authors,citationCount,externalIds"},
                headers={"User-Agent": _USER_AGENT},
                timeout=_TIMEOUT,
            )
        except requests.exceptions.RequestException:
            return []
        if not r.ok:
            logger.debug(f"Semantic Scholar HTTP {r.status_code}")
            return []
        data = r.json().get("data", [])
        out = []
        for p in data:
            if not p.get("title"):
                continue
            authors = [a.get("name", "") for a in (p.get("authors") or [])[:5] if a.get("name")]
            ext = p.get("externalIds") or {}
            doi = ext.get("DOI")
            url = f"https://doi.org/{doi}" if doi else p.get("url")
            out.append({
                "source": "Semantic Scholar",
                "title": p["title"],
                "year": p.get("year"),
                "authors": authors,
                "abstract": (p.get("abstract") or "")[:600] or None,
                "citation_count": p.get("citationCount"),
                "doi": doi,
                "url": url,
            })
        return out

    def _search_pubchem(self, query: str, limit: int) -> list[dict]:
        """PubChem compound search (free). Accepts name or SMILES-ish query."""
        # Try as name first
        r = requests.get(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{urllib.parse.quote(query)}/property/Title,MolecularFormula,MolecularWeight,IUPACName,CanonicalSMILES/JSON",
            timeout=_TIMEOUT,
        )
        if r.ok:
            props = r.json().get("PropertyTable", {}).get("Properties", [])
            return [{
                "source": "PubChem",
                "name": p.get("Title", query),
                "formula": p.get("MolecularFormula"),
                "mw": p.get("MolecularWeight"),
                "iupac": p.get("IUPACName"),
                "smiles": p.get("CanonicalSMILES"),
                "url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{p.get('CID','')}",
            } for p in props[:limit]]
        return []

    @staticmethod
    def _safe_call(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            logger.debug(f"{fn.__name__} failed: {e}")
            return []

    @staticmethod
    def _merge_by_title(primary: list[dict], secondary: list[dict]) -> list[dict]:
        """Merge S2 results in, enriching existing Crossref entries with abstracts."""
        norm = lambda s: (s or "").lower().strip()[:80]
        by_title = {norm(r.get("title")): r for r in primary}
        for s in secondary:
            key = norm(s.get("title"))
            if key in by_title and not by_title[key].get("abstract") and s.get("abstract"):
                by_title[key]["abstract"] = s["abstract"]
                by_title[key]["citation_count"] = s.get("citation_count")
            elif key not in by_title:
                primary.append(s)
        return primary


def _strip_tags(text: str) -> str:
    import re
    return re.sub(r"<[^>]+>", "", text)
