"""
Wispaper API client (legacy – replaced by Semantic Scholar).

This module is kept for backward compatibility only.  The Wispaper API was
never publicly released.  Phase 2 now uses the Semantic Scholar Graph API
via :mod:`paper_novelty_pipeline.services.semantic_scholar_client`.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path


def _run_oauth_flow() -> Optional[str]:
    """Run browser-based OAuth2 flow to obtain access token.

    .. deprecated::
        Wispaper has been replaced by Semantic Scholar.  This function is
        retained for reference and will raise :class:`NotImplementedError`.
    """
    raise NotImplementedError(
        "Wispaper has been replaced by Semantic Scholar for Phase 2 paper search. "
        "See paper_novelty_pipeline.services.semantic_scholar_client."
    )


class WispaperClient:
    """Client for Wispaper academic search API (legacy – no longer used).

    .. deprecated::
        Wispaper has been replaced by Semantic Scholar.  Instantiating this
        class raises :class:`NotImplementedError`.
    """

    def __init__(self):
        raise NotImplementedError(
            "Wispaper has been replaced by Semantic Scholar for Phase 2 paper search. "
            "See paper_novelty_pipeline.services.semantic_scholar_client."
        )

    def search(self, query: str) -> List[Dict[str, Any]]:
        """Search for academic papers."""
        raise NotImplementedError("Wispaper has been replaced by Semantic Scholar.")

    def search_structured(
        self,
        query: str,
        *,
        scope: str = "generic",
        debug_dir: Optional[Path] = None,
        sse_max_seconds: Optional[int] = None,
        sse_max_events: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Perform a structured search with normalized paper metadata."""
        raise NotImplementedError("Wispaper has been replaced by Semantic Scholar.")

    def health_check(self) -> bool:
        """Check if the Wispaper API is accessible."""
        raise NotImplementedError("Wispaper has been replaced by Semantic Scholar.")
