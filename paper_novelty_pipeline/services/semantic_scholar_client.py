"""
Semantic Scholar API client for academic paper search.

This module provides:
  - Academic paper search via the Semantic Scholar Graph API
  - Optional API key support for higher rate limits
  - Automatic retry with exponential backoff
  - Paper metadata normalization

API Docs: https://api.semanticscholar.org/graph/v1
Rate limits:
  - Without API key: ~100 requests per 5 minutes
  - With API key:    1 request per second (standard tier)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# Semantic Scholar Graph API endpoint
_S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

# Fields to request from Semantic Scholar
_S2_FIELDS = ",".join([
    "paperId",
    "externalIds",
    "url",
    "title",
    "abstract",
    "year",
    "authors",
    "venue",
    "publicationVenue",
    "openAccessPdf",
    "citationCount",
])

# Maximum results per single API request (S2 hard cap)
_S2_MAX_PER_PAGE = 100


def _extract_arxiv_id(external_ids: Optional[Dict[str, Any]]) -> Optional[str]:
    """Extract arXiv ID from Semantic Scholar externalIds dict."""
    if not external_ids:
        return None
    arxiv = external_ids.get("ArXiv")
    if arxiv and isinstance(arxiv, str):
        return arxiv.strip()
    return None


def _extract_doi(external_ids: Optional[Dict[str, Any]]) -> Optional[str]:
    """Extract DOI from Semantic Scholar externalIds dict."""
    if not external_ids:
        return None
    doi = external_ids.get("DOI")
    if doi and isinstance(doi, str):
        return doi.strip()
    return None


def _normalize_paper(raw: Dict[str, Any], rank: int, total: int) -> Dict[str, Any]:
    """
    Normalize a Semantic Scholar paper record into the internal paper dict format.

    The returned dict matches the fields expected by Phase2Processor:
    - title, abstract, authors, year, venue
    - doi, arxiv_id, url, pdf_url, paper_id
    - relevance_score, flags
    """
    external_ids = raw.get("externalIds") or {}
    arxiv_id = _extract_arxiv_id(external_ids)
    doi = _extract_doi(external_ids)

    # Build URL: prefer arXiv, then DOI, then S2 url
    url = raw.get("url") or ""
    if arxiv_id:
        url = url or f"https://arxiv.org/abs/{arxiv_id}"

    # PDF URL: Semantic Scholar provides openAccessPdf when available
    open_pdf = raw.get("openAccessPdf") or {}
    pdf_url = open_pdf.get("url") or ""
    if not pdf_url and arxiv_id:
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

    # Authors: list of name strings
    raw_authors = raw.get("authors") or []
    authors = [a.get("name", "") for a in raw_authors if a.get("name")]

    # Venue: prefer publicationVenue name, fallback to venue string
    pub_venue = raw.get("publicationVenue") or {}
    venue = (pub_venue.get("name") or raw.get("venue") or "").strip()

    # Year
    year = raw.get("year")
    if year is not None:
        try:
            year = int(year)
        except (TypeError, ValueError):
            year = None

    # Relevance score: rank-based (higher rank = lower score; normalized to [0,1])
    # We avoid 0.0 so that all papers pass the score filter.
    if total > 1:
        relevance_score = round(1.0 - (rank - 1) / total, 4)
    else:
        relevance_score = 1.0

    return {
        "paper_id": raw.get("paperId") or "",
        "title": (raw.get("title") or "").strip(),
        "abstract": (raw.get("abstract") or "").strip(),
        "authors": authors,
        "year": year,
        "venue": venue,
        "doi": doi,
        "arxiv_id": arxiv_id,
        "url": url,
        "pdf_url": pdf_url,
        "source_url": url,
        "relevance_score": relevance_score,
        "citations": raw.get("citationCount"),
        # All S2 results are treated as "perfect" because S2 does its own relevance
        # ranking and we rely on the postprocess TopK step for quality selection.
        "flags": {"perfect": True, "partial": False, "no": False},
    }


class SemanticScholarClient:
    """
    Client for the Semantic Scholar Graph API.

    Usage::

        client = SemanticScholarClient(api_key="optional-key")
        papers = client.search("neural machine translation", max_results=20)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 5,
        retry_backoff: float = 2.0,
    ) -> None:
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

        self._session = requests.Session()
        if api_key:
            self._session.headers.update({"x-api-key": api_key})

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        max_results: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Search for academic papers matching *query*.

        Parameters
        ----------
        query:
            Free-text query string.
        max_results:
            Maximum number of results to return (capped at 500 to avoid
            very slow requests; lower is usually faster).

        Returns
        -------
        List of normalized paper dicts ready for Phase2 post-processing.
        Each dict contains: title, abstract, authors, year, venue, doi,
        arxiv_id, url, pdf_url, relevance_score, flags.
        """
        if not query or not query.strip():
            return []

        max_results = max(1, min(max_results, 500))
        raw_papers: List[Dict[str, Any]] = []
        offset = 0

        while len(raw_papers) < max_results:
            batch_limit = min(_S2_MAX_PER_PAGE, max_results - len(raw_papers))
            batch = self._fetch_page(query, offset=offset, limit=batch_limit)
            if not batch:
                break
            raw_papers.extend(batch)
            offset += len(batch)
            if len(batch) < batch_limit:
                break  # No more results from S2
            # Respect rate limits when paginating
            time.sleep(0.5)

        total = len(raw_papers)
        return [
            _normalize_paper(raw, rank=i + 1, total=total)
            for i, raw in enumerate(raw_papers)
        ]

    def health_check(self) -> bool:
        """Return True if the Semantic Scholar API is reachable."""
        try:
            resp = self._session.get(
                _S2_SEARCH_URL,
                params={"query": "deep learning", "limit": 1, "fields": "title"},
                timeout=self.timeout,
            )
            return resp.status_code == 200
        except Exception as exc:
            logger.warning("Semantic Scholar health check failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_page(
        self,
        query: str,
        offset: int,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """
        Fetch one page of search results from Semantic Scholar with retry.

        Returns list of raw paper dicts from the API, or empty list on failure.
        """
        params: Dict[str, Any] = {
            "query": query,
            "offset": offset,
            "limit": limit,
            "fields": _S2_FIELDS,
        }

        delay = self.retry_backoff
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._session.get(
                    _S2_SEARCH_URL,
                    params=params,
                    timeout=self.timeout,
                )

                if resp.status_code == 200:
                    body = resp.json()
                    data = body.get("data") or []
                    logger.debug(
                        "S2 search: query=%r offset=%d limit=%d got=%d total=%s",
                        query,
                        offset,
                        limit,
                        len(data),
                        body.get("total", "?"),
                    )
                    return data

                if resp.status_code == 429:
                    # Rate limited – back off and retry
                    logger.warning(
                        "S2 rate-limited (429). Attempt %d/%d; sleeping %.1fs",
                        attempt,
                        self.max_retries,
                        delay,
                    )
                    time.sleep(delay)
                    delay *= self.retry_backoff
                    continue

                if resp.status_code >= 500:
                    logger.warning(
                        "S2 server error %d. Attempt %d/%d; sleeping %.1fs",
                        resp.status_code,
                        attempt,
                        self.max_retries,
                        delay,
                    )
                    time.sleep(delay)
                    delay *= self.retry_backoff
                    continue

                # Client error (4xx other than 429): log and abort
                logger.error(
                    "S2 client error %d for query %r: %s",
                    resp.status_code,
                    query,
                    resp.text[:200],
                )
                return []

            except requests.exceptions.Timeout:
                logger.warning(
                    "S2 request timed out. Attempt %d/%d; sleeping %.1fs",
                    attempt,
                    self.max_retries,
                    delay,
                )
                time.sleep(delay)
                delay *= self.retry_backoff

            except requests.exceptions.ConnectionError as exc:
                logger.warning(
                    "S2 connection error: %s. Attempt %d/%d; sleeping %.1fs",
                    exc,
                    attempt,
                    self.max_retries,
                    delay,
                )
                time.sleep(delay)
                delay *= self.retry_backoff

            except Exception as exc:
                logger.error("Unexpected error fetching S2 results: %s", exc)
                return []

        logger.error("S2 search failed after %d attempts for query %r", self.max_retries, query)
        return []
