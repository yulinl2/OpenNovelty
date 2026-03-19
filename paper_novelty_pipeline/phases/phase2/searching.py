"""
Phase 2: Paper Searching via Semantic Scholar.

This module handles the academic paper search phase of the novelty analysis pipeline.

Responsibilities:
  1. Prepare search queries from Phase 1 extracted content (core_task + contributions)
  2. Execute concurrent searches via Semantic Scholar Graph API
  3. Save raw API responses to phase2/raw_responses/
  4. Return execution statistics
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from paper_novelty_pipeline.models import ExtractedContent
from paper_novelty_pipeline.services.semantic_scholar_client import SemanticScholarClient

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

# Default number of search results to fetch per query
_DEFAULT_MAX_RESULTS = 100

# Scope identifiers (must match postprocess.py constants)
_SCOPE_CORE_TASK = "core_task"

# Minimum delay between queries (seconds) when no API key is configured
# This keeps the pipeline safely within Semantic Scholar's free-tier rate limits.
_INTER_QUERY_DELAY = 1.0


def _build_queries(extracted: ExtractedContent) -> List[Tuple[str, str, str]]:
    """
    Build a flat list of (scope, query_variant_id, query_text) tuples.

    Scopes:
    - "core_task"        → from ExtractedContent.core_task.query_variants
    - "contribution_N"   → from ExtractedContent.contributions[N].query_variants

    Each query variant is executed as an independent search so that the
    postprocess step can deduplicate across overlapping result sets.

    Returns
    -------
    List of (scope, query_id, query_text) tuples.
    """
    queries: List[Tuple[str, str, str]] = []

    # Core task queries
    if extracted.core_task and extracted.core_task.query_variants:
        for i, qv in enumerate(extracted.core_task.query_variants):
            if qv and qv.strip():
                queries.append((_SCOPE_CORE_TASK, f"v{i}", qv.strip()))
    elif extracted.core_task and extracted.core_task.text:
        # Fallback: use the core_task text itself
        queries.append((_SCOPE_CORE_TASK, "v0", extracted.core_task.text.strip()))

    # Contribution queries
    for contrib in (extracted.contributions or []):
        scope = contrib.id if contrib.id else f"contribution_{len(queries)}"
        # Normalize scope name to "contribution_N" format expected by postprocess
        if not scope.startswith("contribution_"):
            scope = f"contribution_{scope}"

        query_variants = contrib.query_variants or []
        if not query_variants and contrib.prior_work_query:
            query_variants = [contrib.prior_work_query]

        for i, qv in enumerate(query_variants):
            if qv and qv.strip():
                queries.append((scope, f"v{i}", qv.strip()))

    return queries


class PaperSearcher:
    """
    Phase 2: Search for related papers via Semantic Scholar Graph API.

    Parameters
    ----------
    concurrency:
        Maximum number of concurrent search threads.  Defaults to 1 to
        respect Semantic Scholar's free-tier rate limits without an API key.
    max_results:
        Maximum number of results to fetch per query variant.
    api_key:
        Optional Semantic Scholar API key for higher rate limits.
    """

    def __init__(
        self,
        concurrency: Optional[int] = None,
        max_results: int = _DEFAULT_MAX_RESULTS,
        api_key: Optional[str] = None,
    ) -> None:
        self.concurrency = max(1, concurrency or 1)
        self.max_results = max_results

        # Import config here to allow env-based overrides to take effect
        try:
            from paper_novelty_pipeline import config as _cfg

            _api_key = api_key or getattr(_cfg, "SEMANTIC_SCHOLAR_API_KEY", None) or None
            _max_res = getattr(_cfg, "SEMANTIC_SCHOLAR_MAX_RESULTS", None)
            if _max_res:
                try:
                    self.max_results = int(_max_res)
                except (TypeError, ValueError):
                    pass
        except Exception:
            _api_key = api_key

        self._client = SemanticScholarClient(api_key=_api_key)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def search_all(
        self,
        extracted: ExtractedContent,
        out_dir: Path,
    ) -> Dict[str, Any]:
        """
        Execute all search queries derived from *extracted* and persist results.

        Raw responses are written to ``<out_dir>/raw_responses/`` using the
        filename pattern ``raw_<scope>_<query_id>.json``.

        Parameters
        ----------
        extracted:
            ExtractedContent from Phase 1.
        out_dir:
            Phase 2 output directory (``phase2/``).

        Returns
        -------
        Statistics dict::

            {
                "total_queries": int,
                "succeeded": int,
                "failed": int,
                "elapsed_seconds": float,
            }
        """
        out_dir = Path(out_dir)
        raw_dir = out_dir / "raw_responses"
        raw_dir.mkdir(parents=True, exist_ok=True)

        queries = _build_queries(extracted)
        if not queries:
            logger.warning("Phase2: no queries generated from extracted content.")
            return {"total_queries": 0, "succeeded": 0, "failed": 0, "elapsed_seconds": 0.0}

        logger.info(
            "Phase2: executing %d search queries (concurrency=%d, max_results=%d)",
            len(queries),
            self.concurrency,
            self.max_results,
        )

        t0 = time.time()
        succeeded = 0
        failed = 0

        if self.concurrency <= 1:
            # Sequential execution with inter-query delay
            for scope, qid, query_text in queries:
                ok = self._run_single_query(scope, qid, query_text, raw_dir)
                if ok:
                    succeeded += 1
                else:
                    failed += 1
                time.sleep(_INTER_QUERY_DELAY)
        else:
            # Concurrent execution
            with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                futures = {
                    pool.submit(self._run_single_query, scope, qid, query_text, raw_dir): (scope, qid)
                    for scope, qid, query_text in queries
                }
                for future in as_completed(futures):
                    scope, qid = futures[future]
                    try:
                        ok = future.result()
                        if ok:
                            succeeded += 1
                        else:
                            failed += 1
                    except Exception as exc:
                        logger.error("Phase2: unexpected error for %s/%s: %s", scope, qid, exc)
                        failed += 1

        elapsed = time.time() - t0
        stats: Dict[str, Any] = {
            "total_queries": len(queries),
            "succeeded": succeeded,
            "failed": failed,
            "elapsed_seconds": round(elapsed, 2),
        }
        logger.info("Phase2: search complete – %s", stats)
        return stats

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_single_query(
        self,
        scope: str,
        query_id: str,
        query_text: str,
        raw_dir: Path,
    ) -> bool:
        """
        Execute a single search query and persist the raw response.

        Returns True on success, False on failure.
        """
        filename = f"raw_{scope}_{query_id}.json"
        out_path = raw_dir / filename

        logger.info("Phase2: searching scope=%s qid=%s query=%r", scope, query_id, query_text[:80])
        try:
            papers = self._client.search(query_text, max_results=self.max_results)
        except Exception as exc:
            logger.error(
                "Phase2: search failed for scope=%s qid=%s: %s", scope, query_id, exc
            )
            return False

        # Persist in Semantic Scholar raw format
        payload: Dict[str, Any] = {
            "backend": "semantic_scholar",
            "scope": scope,
            "query_id": query_id,
            "query": query_text,
            "paper_count": len(papers),
            "papers": papers,
        }
        try:
            with out_path.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.error("Phase2: failed to write %s: %s", out_path, exc)
            return False

        logger.info(
            "Phase2: scope=%s qid=%s → %d papers → %s",
            scope,
            query_id,
            len(papers),
            out_path.name,
        )
        return True


# ------------------------------------------------------------------
# Module-level convenience function (called by entrypoints.py)
# ------------------------------------------------------------------

def run_phase2_search(
    extracted: ExtractedContent,
    out_dir: Path,
    concurrency: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run Phase 2 search (API calls only) using Semantic Scholar.

    Parameters
    ----------
    extracted:
        ExtractedContent from Phase 1.
    out_dir:
        Phase 2 output directory (``phase2/``).
    concurrency:
        Number of concurrent search threads (default: 1).

    Returns
    -------
    Statistics dict with keys: total_queries, succeeded, failed, elapsed_seconds.
    """
    searcher = PaperSearcher(concurrency=concurrency)
    return searcher.search_all(extracted, Path(out_dir))
