"""
Tests for the Semantic Scholar-based Phase 2 search pipeline.

Covers:
  - SemanticScholarClient: paper normalization, search, retry/pagination, health check
  - _build_queries: query derivation from ExtractedContent
  - PaperSearcher.search_all: query execution + file output format
  - Phase2Processor._extract_papers_from_events: both Semantic Scholar and legacy format
  - Full Phase 2 end-to-end: search → raw_responses → postprocess → citation_index.json

All HTTP calls are mocked via unittest.mock so no real network traffic occurs.

Input format (Phase 2 input – produced by Phase 1)
----------------------------------------------------
ExtractedContent(
    core_task=CoreTask(
        text="graph neural networks for node classification",
        query_variants=[
            "graph neural networks node classification",
            "GNN semi-supervised node labeling",
            "message passing neural network",
        ],
    ),
    contributions=[
        ContributionClaim(
            id="contribution_1",
            name="Adaptive Graph Convolution",
            author_claim_text="We introduce an adaptive convolution layer.",
            description="A convolution that adapts to local graph topology.",
            prior_work_query="adaptive graph convolution",
            query_variants=[
                "adaptive graph convolution layers",
                "dynamic graph convolution",
            ],
        ),
    ],
)

Output format (Phase 2 raw response file)
-----------------------------------------
File: phase2/raw_responses/raw_core_task_v0.json
{
  "backend": "semantic_scholar",
  "scope": "core_task",
  "query_id": "v0",
  "query": "graph neural networks node classification",
  "paper_count": 2,
  "papers": [
    {
      "paper_id": "abc123",
      "title": "Semi-Supervised Classification with Graph Convolutional Networks",
      "abstract": "...",
      "authors": ["Thomas N. Kipf", "Max Welling"],
      "year": 2017,
      "venue": "ICLR",
      "doi": "10.1234/gcn",
      "arxiv_id": "1609.02907",
      "url": "https://arxiv.org/abs/1609.02907",
      "pdf_url": "https://arxiv.org/pdf/1609.02907",
      "source_url": "https://arxiv.org/abs/1609.02907",
      "relevance_score": 1.0,
      "citations": 15000,
      "flags": {"perfect": true, "partial": false, "no": false}
    }
  ]
}

Output format (Phase 2 final – citation_index.json)
-----------------------------------------------------
{
  "generated_at": "2026-01-18T...",
  "count": 3,
  "items": [
    {"index": 0, "title": "My Paper", "roles": [{"type": "original_paper"}], ...},
    {"index": 1, "title": "GCN Paper", "roles": [{"type": "core_task", "rank": 1}], ...},
    {"index": 2, "title": "GAT Paper", "roles": [{"type": "contribution", "scope": "contribution_1", "rank": 1}], ...}
  ]
}
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("LLM_API_KEY", "test-dummy-key")

from paper_novelty_pipeline.models import (
    ContributionClaim,
    CoreTask,
    ExtractedContent,
)
from paper_novelty_pipeline.phases.phase2.postprocess import Phase2Processor
from paper_novelty_pipeline.phases.phase2.searching import (
    PaperSearcher,
    _build_queries,
)
from paper_novelty_pipeline.services.semantic_scholar_client import (
    SemanticScholarClient,
    _extract_arxiv_id,
    _extract_doi,
    _normalize_paper,
)


# ===========================================================================
# Fixtures and helpers
# ===========================================================================

def _make_s2_raw_paper(**overrides) -> Dict[str, Any]:
    """Return a minimal Semantic Scholar API response record."""
    base = {
        "paperId": "abc123",
        "title": "Semi-Supervised Classification with Graph Convolutional Networks",
        "abstract": "We present a scalable approach for semi-supervised classification.",
        "year": 2017,
        "authors": [{"authorId": "1", "name": "Thomas N. Kipf"}, {"authorId": "2", "name": "Max Welling"}],
        "venue": "ICLR",
        "publicationVenue": {"id": "v1", "name": "International Conference on Learning Representations"},
        "externalIds": {"DOI": "10.1234/gcn", "ArXiv": "1609.02907"},
        "url": "https://www.semanticscholar.org/paper/abc123",
        "openAccessPdf": {"url": "https://arxiv.org/pdf/1609.02907"},
        "citationCount": 15000,
    }
    base.update(overrides)
    return base


def _make_extracted_content() -> ExtractedContent:
    """Return a representative ExtractedContent (typical Phase 1 output)."""
    return ExtractedContent(
        core_task=CoreTask(
            text="graph neural networks for node classification",
            query_variants=[
                "graph neural networks node classification",
                "GNN semi-supervised node labeling",
                "message passing neural network",
            ],
        ),
        contributions=[
            ContributionClaim(
                id="contribution_1",
                name="Adaptive Graph Convolution",
                author_claim_text="We introduce an adaptive convolution layer.",
                description="A convolution that adapts to local graph topology.",
                prior_work_query="adaptive graph convolution",
                query_variants=[
                    "adaptive graph convolution layers",
                    "dynamic graph convolution",
                ],
            ),
        ],
    )


def _make_proc(phase2_dir: Path, **kwargs) -> Phase2Processor:
    """Construct a Phase2Processor bypassing __init__ file I/O."""
    proc = Phase2Processor.__new__(Phase2Processor)
    proc.phase2_dir = phase2_dir
    proc.raw_responses_dir = phase2_dir / "raw_responses"
    proc.candidates_dir = phase2_dir / "candidates"
    proc.final_dir = phase2_dir / "final"
    proc.cutoff_year = kwargs.get("cutoff_year")
    proc.self_pdf_url = kwargs.get("self_pdf_url")
    proc.self_title = kwargs.get("self_title")
    proc.topk_core_task = kwargs.get("topk_core_task", 50)
    proc.topk_contribution = kwargs.get("topk_contribution", 10)
    proc.original_paper_canonical_id = kwargs.get("original_paper_canonical_id")
    return proc


def _write_s2_raw(raw_dir: Path, scope: str, qid: str, papers: List[Dict]) -> Path:
    """Write a Semantic Scholar format raw response file."""
    path = raw_dir / f"raw_{scope}_{qid}.json"
    payload = {
        "backend": "semantic_scholar",
        "scope": scope,
        "query_id": qid,
        "query": f"test query for {scope}",
        "paper_count": len(papers),
        "papers": papers,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _sample_s2_paper(
    paper_id: str = "p1",
    title: str = "Graph Attention Networks",
    year: int = 2018,
    arxiv_id: str = "1710.10903",
) -> Dict[str, Any]:
    """Return a normalized paper dict (as stored in raw_responses)."""
    return {
        "paper_id": paper_id,
        "title": title,
        "abstract": "We present graph attention networks.",
        "authors": ["Petar Veličković"],
        "year": year,
        "venue": "ICLR",
        "doi": None,
        "arxiv_id": arxiv_id,
        "url": f"https://arxiv.org/abs/{arxiv_id}",
        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
        "source_url": f"https://arxiv.org/abs/{arxiv_id}",
        "relevance_score": 0.95,
        "citations": 5000,
        "flags": {"perfect": True, "partial": False, "no": False},
    }


# ===========================================================================
# 1. SemanticScholarClient – unit tests (no HTTP)
# ===========================================================================

class TestExtractHelpers:
    def test_extract_arxiv_id_present(self):
        assert _extract_arxiv_id({"ArXiv": "1609.02907"}) == "1609.02907"

    def test_extract_arxiv_id_missing(self):
        assert _extract_arxiv_id({}) is None

    def test_extract_arxiv_id_none(self):
        assert _extract_arxiv_id(None) is None

    def test_extract_doi_present(self):
        assert _extract_doi({"DOI": "10.1234/test"}) == "10.1234/test"

    def test_extract_doi_missing(self):
        assert _extract_doi({"ArXiv": "1234"}) is None


class TestNormalizePaper:
    def test_basic_normalization(self):
        raw = _make_s2_raw_paper()
        result = _normalize_paper(raw, rank=1, total=3)

        assert result["title"] == "Semi-Supervised Classification with Graph Convolutional Networks"
        assert result["abstract"].startswith("We present a scalable")
        assert result["authors"] == ["Thomas N. Kipf", "Max Welling"]
        assert result["year"] == 2017
        assert result["doi"] == "10.1234/gcn"
        assert result["arxiv_id"] == "1609.02907"
        assert result["pdf_url"] == "https://arxiv.org/pdf/1609.02907"
        assert result["citations"] == 15000
        assert result["venue"] == "International Conference on Learning Representations"

    def test_flags_always_perfect(self):
        raw = _make_s2_raw_paper()
        result = _normalize_paper(raw, rank=1, total=1)
        assert result["flags"] == {"perfect": True, "partial": False, "no": False}

    def test_relevance_score_descending(self):
        raw = _make_s2_raw_paper()
        r1 = _normalize_paper(raw, rank=1, total=10)
        r5 = _normalize_paper(raw, rank=5, total=10)
        r10 = _normalize_paper(raw, rank=10, total=10)
        assert r1["relevance_score"] > r5["relevance_score"] > r10["relevance_score"]

    def test_relevance_score_single_paper(self):
        raw = _make_s2_raw_paper()
        result = _normalize_paper(raw, rank=1, total=1)
        assert result["relevance_score"] == 1.0

    def test_arxiv_pdf_fallback(self):
        """When openAccessPdf is missing, arxiv_id should be used for pdf_url."""
        raw = _make_s2_raw_paper(openAccessPdf={})
        result = _normalize_paper(raw, rank=1, total=1)
        assert result["pdf_url"] == "https://arxiv.org/pdf/1609.02907"

    def test_no_arxiv_id(self):
        raw = _make_s2_raw_paper(externalIds={"DOI": "10.1234/test"}, openAccessPdf={})
        result = _normalize_paper(raw, rank=1, total=1)
        assert result["arxiv_id"] is None
        assert result["pdf_url"] == ""

    def test_empty_abstract(self):
        raw = _make_s2_raw_paper(abstract=None)
        result = _normalize_paper(raw, rank=1, total=1)
        assert result["abstract"] == ""

    def test_publication_venue_fallback_to_venue_string(self):
        raw = _make_s2_raw_paper(publicationVenue=None, venue="NeurIPS 2017")
        result = _normalize_paper(raw, rank=1, total=1)
        assert result["venue"] == "NeurIPS 2017"


class TestSemanticScholarClientSearch:
    def _mock_response(self, papers: List[Dict], total: int = None) -> MagicMock:
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "total": total if total is not None else len(papers),
            "data": papers,
        }
        return resp

    def test_search_returns_normalized_papers(self):
        raw_papers = [_make_s2_raw_paper(paperId=f"p{i}") for i in range(3)]
        with patch("requests.Session.get", return_value=self._mock_response(raw_papers)):
            client = SemanticScholarClient()
            results = client.search("graph neural networks", max_results=3)

        assert len(results) == 3
        for r in results:
            assert "title" in r
            assert "flags" in r
            assert r["flags"]["perfect"] is True

    def test_search_empty_query(self):
        client = SemanticScholarClient()
        assert client.search("") == []
        assert client.search("   ") == []

    def test_search_caps_at_500(self):
        """max_results is capped at 500 internally."""
        raw_papers = [_make_s2_raw_paper(paperId=f"p{i}") for i in range(5)]
        with patch("requests.Session.get", return_value=self._mock_response(raw_papers)) as mock_get:
            client = SemanticScholarClient()
            client.search("deep learning", max_results=9999)
            # Verify limit parameter sent to API is <= 100 (_S2_MAX_PER_PAGE)
            call_params = mock_get.call_args[1]["params"]
            assert call_params["limit"] <= 100

    def test_search_handles_rate_limit_retry(self):
        rate_limit_resp = MagicMock()
        rate_limit_resp.status_code = 429

        success_resp = self._mock_response([_make_s2_raw_paper()])

        with patch("requests.Session.get", side_effect=[rate_limit_resp, success_resp]):
            with patch("time.sleep"):  # Skip actual sleep in tests
                client = SemanticScholarClient(max_retries=3)
                results = client.search("test query", max_results=1)

        assert len(results) == 1

    def test_search_returns_empty_on_client_error(self):
        error_resp = MagicMock()
        error_resp.status_code = 400
        error_resp.text = "Bad request"

        with patch("requests.Session.get", return_value=error_resp):
            client = SemanticScholarClient()
            results = client.search("bad query", max_results=10)

        assert results == []

    def test_search_returns_empty_after_all_retries_exhausted(self):
        server_error = MagicMock()
        server_error.status_code = 503
        server_error.text = "Service unavailable"

        with patch("requests.Session.get", return_value=server_error):
            with patch("time.sleep"):
                client = SemanticScholarClient(max_retries=2)
                results = client.search("neural networks", max_results=10)

        assert results == []

    def test_health_check_success(self):
        resp = MagicMock()
        resp.status_code = 200
        with patch("requests.Session.get", return_value=resp):
            client = SemanticScholarClient()
            assert client.health_check() is True

    def test_health_check_failure(self):
        with patch("requests.Session.get", side_effect=Exception("Connection refused")):
            client = SemanticScholarClient()
            assert client.health_check() is False

    def test_api_key_added_to_headers(self):
        client = SemanticScholarClient(api_key="my-secret-key")
        assert client._session.headers.get("x-api-key") == "my-secret-key"

    def test_no_api_key_no_header(self):
        client = SemanticScholarClient(api_key=None)
        assert "x-api-key" not in client._session.headers


# ===========================================================================
# 2. _build_queries – query derivation
# ===========================================================================

class TestBuildQueries:
    def test_core_task_variants_become_queries(self):
        extracted = _make_extracted_content()
        queries = _build_queries(extracted)
        core_queries = [(s, qid, qt) for s, qid, qt in queries if s == "core_task"]
        assert len(core_queries) == 3  # 3 variants
        texts = [qt for _, _, qt in core_queries]
        assert "graph neural networks node classification" in texts
        assert "GNN semi-supervised node labeling" in texts
        assert "message passing neural network" in texts

    def test_contribution_variants_become_queries(self):
        extracted = _make_extracted_content()
        queries = _build_queries(extracted)
        contrib_queries = [(s, qid, qt) for s, qid, qt in queries if s == "contribution_1"]
        assert len(contrib_queries) == 2  # 2 query_variants
        texts = [qt for _, _, qt in contrib_queries]
        assert "adaptive graph convolution layers" in texts
        assert "dynamic graph convolution" in texts

    def test_scope_format_matches_postprocess_regex(self):
        """Scope names must match the regex in Phase2Processor._identify_scope."""
        import re
        regex = re.compile(r"^(core_task|contribution_\d+)$")
        extracted = _make_extracted_content()
        queries = _build_queries(extracted)
        for scope, _, _ in queries:
            assert regex.match(scope), f"Scope {scope!r} doesn't match expected regex"

    def test_fallback_to_prior_work_query_when_no_variants(self):
        extracted = ExtractedContent(
            core_task=CoreTask(text="NLP", query_variants=["NLP tasks"]),
            contributions=[
                ContributionClaim(
                    id="contribution_1",
                    name="Method",
                    author_claim_text="claim",
                    description="desc",
                    prior_work_query="prior work fallback query",
                    query_variants=[],
                )
            ],
        )
        queries = _build_queries(extracted)
        contrib_queries = [qt for s, _, qt in queries if s == "contribution_1"]
        assert len(contrib_queries) == 1
        assert contrib_queries[0] == "prior work fallback query"

    def test_core_task_text_fallback_when_no_variants(self):
        extracted = ExtractedContent(
            core_task=CoreTask(text="graph learning fallback", query_variants=[]),
            contributions=[],
        )
        queries = _build_queries(extracted)
        assert len(queries) == 1
        assert queries[0] == ("core_task", "v0", "graph learning fallback")

    def test_empty_content_returns_empty_list(self):
        extracted = ExtractedContent(
            core_task=CoreTask(text="", query_variants=[]),
            contributions=[],
        )
        queries = _build_queries(extracted)
        assert queries == []

    def test_blank_variants_are_skipped(self):
        extracted = ExtractedContent(
            core_task=CoreTask(
                text="real query",
                query_variants=["  ", "", "valid query", "  "],
            ),
            contributions=[],
        )
        queries = _build_queries(extracted)
        assert len(queries) == 1
        assert queries[0][2] == "valid query"

    def test_non_standard_contribution_id_is_prefixed(self):
        """Contributions with non-standard IDs get contribution_ prefix."""
        extracted = ExtractedContent(
            core_task=CoreTask(text="", query_variants=[]),
            contributions=[
                ContributionClaim(
                    id="myCustomId",
                    name="Custom",
                    author_claim_text="claim",
                    description="desc",
                    prior_work_query="",
                    query_variants=["some query"],
                )
            ],
        )
        queries = _build_queries(extracted)
        scopes = {s for s, _, _ in queries}
        assert "contribution_myCustomId" in scopes


# ===========================================================================
# 3. PaperSearcher – mocked search
# ===========================================================================

class TestPaperSearcher:
    def test_search_all_creates_raw_response_files(self):
        extracted = _make_extracted_content()  # 3 core + 2 contrib = 5 queries

        with tempfile.TemporaryDirectory() as tmpdir:
            phase2_dir = Path(tmpdir) / "phase2"
            searcher = PaperSearcher(concurrency=1)
            # Replace client with mock
            searcher._client = MagicMock()
            searcher._client.search.return_value = [_sample_s2_paper()]

            with patch("time.sleep"):  # Skip delays
                stats = searcher.search_all(extracted, phase2_dir)

            raw_files = list((phase2_dir / "raw_responses").glob("raw_*.json"))
            assert len(raw_files) == 5  # 3 core_task + 2 contribution_1

            assert stats["total_queries"] == 5
            assert stats["succeeded"] == 5
            assert stats["failed"] == 0

    def test_raw_response_file_format(self):
        """Verify the saved raw response file has the correct schema."""
        extracted = ExtractedContent(
            core_task=CoreTask(text="NLP", query_variants=["natural language processing"]),
            contributions=[],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            phase2_dir = Path(tmpdir) / "phase2"
            searcher = PaperSearcher(concurrency=1)
            searcher._client = MagicMock()
            searcher._client.search.return_value = [_sample_s2_paper()]

            with patch("time.sleep"):
                searcher.search_all(extracted, phase2_dir)

            raw_files = list((phase2_dir / "raw_responses").glob("raw_core_task_*.json"))
            assert len(raw_files) == 1

            content = json.loads(raw_files[0].read_text(encoding="utf-8"))
            assert content["backend"] == "semantic_scholar"
            assert content["scope"] == "core_task"
            assert "query" in content
            assert "paper_count" in content
            assert isinstance(content["papers"], list)

            paper = content["papers"][0]
            required_fields = {"paper_id", "title", "abstract", "authors", "year", "flags"}
            assert required_fields.issubset(paper.keys()), (
                f"Missing fields: {required_fields - set(paper.keys())}"
            )
            assert paper["flags"]["perfect"] is True

    def test_failed_query_counted_correctly(self):
        extracted = ExtractedContent(
            core_task=CoreTask(text="test", query_variants=["q1", "q2"]),
            contributions=[],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            phase2_dir = Path(tmpdir) / "phase2"
            searcher = PaperSearcher(concurrency=1)
            searcher._client = MagicMock()
            searcher._client.search.side_effect = [
                [_sample_s2_paper()],   # q1 succeeds
                Exception("API error"),  # q2 fails
            ]

            with patch("time.sleep"):
                stats = searcher.search_all(extracted, phase2_dir)

        assert stats["succeeded"] == 1
        assert stats["failed"] == 1
        assert stats["total_queries"] == 2

    def test_no_queries_returns_zero_stats(self):
        extracted = ExtractedContent(
            core_task=CoreTask(text="", query_variants=[]),
            contributions=[],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            searcher = PaperSearcher()
            stats = searcher.search_all(extracted, Path(tmpdir) / "phase2")

        assert stats == {"total_queries": 0, "succeeded": 0, "failed": 0, "elapsed_seconds": 0.0}


# ===========================================================================
# 4. Phase2Processor._extract_papers_from_events
# ===========================================================================

class TestExtractPapersFromEvents:
    def _proc(self) -> Phase2Processor:
        tmpdir = tempfile.mkdtemp()
        p2 = Path(tmpdir) / "phase2"
        p2.mkdir()
        return _make_proc(p2)

    def test_semantic_scholar_format_basic(self):
        proc = self._proc()
        raw = {
            "backend": "semantic_scholar",
            "papers": [
                _sample_s2_paper(paper_id="p1", title="GCN Paper"),
                _sample_s2_paper(paper_id="p2", title="GAT Paper"),
            ],
        }
        papers = proc._extract_papers_from_events(raw)

        assert len(papers) == 2
        titles = [p["title"] for p in papers]
        assert "GCN Paper" in titles
        assert "GAT Paper" in titles

    def test_semantic_scholar_papers_have_perfect_flag(self):
        proc = self._proc()
        raw = {
            "backend": "semantic_scholar",
            "papers": [_sample_s2_paper()],
        }
        papers = proc._extract_papers_from_events(raw)
        assert papers[0]["flags"]["perfect"] is True

    def test_semantic_scholar_missing_flags_are_filled(self):
        """Papers without flags get perfect=True injected."""
        proc = self._proc()
        paper_no_flags = dict(_sample_s2_paper())
        del paper_no_flags["flags"]

        raw = {"backend": "semantic_scholar", "papers": [paper_no_flags]}
        papers = proc._extract_papers_from_events(raw)
        assert papers[0]["flags"]["perfect"] is True

    def test_semantic_scholar_empty_papers_list(self):
        proc = self._proc()
        raw = {"backend": "semantic_scholar", "papers": []}
        papers = proc._extract_papers_from_events(raw)
        assert papers == []

    def test_unknown_dict_backend_returns_empty(self):
        proc = self._proc()
        papers = proc._extract_papers_from_events({"backend": "unknown_api", "data": []})
        assert papers == []

    def test_legacy_wispaper_format_empty_list(self):
        """Empty WisPaper SSE event list returns empty list."""
        proc = self._proc()
        papers = proc._extract_papers_from_events([])
        assert papers == []

    def test_legacy_wispaper_verification_event_parsed(self):
        """A valid WisPaper SSE verification event is parsed correctly."""
        proc = self._proc()
        verdict = json.dumps({
            "criteria_assessment": [{"assessment": "support", "type": "relevance"}]
        })
        events = [
            {
                "event": "onAgentEnd",
                "name": "verification",
                "data": {
                    "metadata": {"title": "Legacy Paper", "year": 2020},
                    "content": verdict,
                },
            }
        ]
        papers = proc._extract_papers_from_events(events)
        assert len(papers) == 1
        assert papers[0]["title"] == "Legacy Paper"
        assert papers[0]["flags"]["perfect"] is True

    def test_legacy_non_verification_events_ignored(self):
        """WisPaper SSE events that aren't verification events are ignored."""
        proc = self._proc()
        events = [
            {"event": "onAgentStart", "name": "search", "data": {}},
            {"event": "onAgentEnd", "name": "search", "data": {}},
        ]
        papers = proc._extract_papers_from_events(events)
        assert papers == []


# ===========================================================================
# 5. End-to-end: search -> postprocess -> citation_index.json
# ===========================================================================

class TestPhase2EndToEnd:
    """
    Full Phase 2 pipeline test using mocked HTTP and real file I/O.

    Input:   ExtractedContent (Phase 1 output)
    Output:  phase2/final/citation_index.json
    """

    def _run_phase2(self, tmpdir: str) -> Dict[str, Any]:
        """
        Run the full Phase 2 pipeline and return the citation_index.json content.
        """
        base_dir = Path(tmpdir) / "openreview_TestPaper_20260118"
        phase1_dir = base_dir / "phase1"
        phase2_dir = base_dir / "phase2"
        phase1_dir.mkdir(parents=True)

        # ------------------------------------------------------------------
        # Write Phase 1 outputs (paper.json + phase1_extracted.json)
        # ------------------------------------------------------------------
        paper_data = {
            "paper_id": "https://openreview.net/pdf?id=TestPaper",
            "title": "Adaptive Graph Convolution for Node Classification",
            "authors": ["Alice Smith", "Bob Jones"],
            "year": 2024,
            "abstract": "We propose adaptive graph convolution.",
        }
        extracted_data = {
            "core_task": {
                "text": "graph neural networks for node classification",
                "query_variants": [
                    "graph neural networks node classification",
                    "GNN semi-supervised learning",
                ],
            },
            "contributions": [
                {
                    "id": "contribution_1",
                    "name": "Adaptive Graph Convolution",
                    "description": "A convolution that adapts to local graph topology.",
                    "prior_work_query": "adaptive graph convolution",
                    "query_variants": ["adaptive graph convolution layers"],
                    "author_claim_text": "We introduce an adaptive convolution layer.",
                    "source_hint": "Section 3",
                }
            ],
        }
        (phase1_dir / "paper.json").write_text(
            json.dumps(paper_data, indent=2), encoding="utf-8"
        )
        (phase1_dir / "phase1_extracted.json").write_text(
            json.dumps(extracted_data, indent=2), encoding="utf-8"
        )

        # ------------------------------------------------------------------
        # Mock HTTP so PaperSearcher talks to a fake Semantic Scholar API
        # ------------------------------------------------------------------
        gcn_paper = {
            "paperId": "gcn_s2id",
            "title": "Semi-Supervised Classification with Graph Convolutional Networks",
            "abstract": "We propose graph convolutional networks.",
            "year": 2017,
            "authors": [{"name": "Thomas N. Kipf"}, {"name": "Max Welling"}],
            "venue": "ICLR",
            "publicationVenue": {"name": "ICLR"},
            "externalIds": {"ArXiv": "1609.02907"},
            "url": "https://arxiv.org/abs/1609.02907",
            "openAccessPdf": {"url": "https://arxiv.org/pdf/1609.02907"},
            "citationCount": 15000,
        }
        gat_paper = {
            "paperId": "gat_s2id",
            "title": "Graph Attention Networks",
            "abstract": "We propose graph attention networks.",
            "year": 2018,
            "authors": [{"name": "Petar Velickovic"}],
            "venue": "ICLR",
            "publicationVenue": {"name": "ICLR"},
            "externalIds": {"ArXiv": "1710.10903"},
            "url": "https://arxiv.org/abs/1710.10903",
            "openAccessPdf": {"url": "https://arxiv.org/pdf/1710.10903"},
            "citationCount": 8000,
        }

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"total": 2, "data": [gcn_paper, gat_paper]}

        with patch("requests.Session.get", return_value=mock_resp):
            with patch("time.sleep"):  # Skip rate-limit delays
                extracted = ExtractedContent(
                    core_task=CoreTask(
                        text=extracted_data["core_task"]["text"],
                        query_variants=extracted_data["core_task"]["query_variants"],
                    ),
                    contributions=[
                        ContributionClaim(
                            id="contribution_1",
                            name="Adaptive Graph Convolution",
                            author_claim_text="We introduce an adaptive convolution layer.",
                            description="A convolution that adapts to local graph topology.",
                            prior_work_query="adaptive graph convolution",
                            query_variants=["adaptive graph convolution layers"],
                            source_hint="Section 3",
                        )
                    ],
                )

                searcher = PaperSearcher(concurrency=1)
                searcher.search_all(extracted, phase2_dir)

        # ------------------------------------------------------------------
        # Run postprocessing on the saved raw_responses
        # ------------------------------------------------------------------
        with patch.dict(os.environ, {"PHASE2_ALIAS_MODE": "heuristic"}):
            processor = Phase2Processor(
                phase2_dir=phase2_dir,
                cutoff_year=2025,
                self_title="Adaptive Graph Convolution for Node Classification",
                topk_core_task=50,
                topk_contribution=10,
            )
            processor.process()

        citation_index_path = phase2_dir / "final" / "citation_index.json"
        assert citation_index_path.exists(), "citation_index.json was not created"
        return json.loads(citation_index_path.read_text(encoding="utf-8"))

    def test_citation_index_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ci = self._run_phase2(tmpdir)

        assert "items" in ci
        assert "count" in ci
        assert ci["count"] == len(ci["items"])

    def test_citation_index_has_original_paper_at_index_0(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ci = self._run_phase2(tmpdir)

        item0 = ci["items"][0]
        assert item0["index"] == 0
        roles = [r["type"] for r in item0["roles"]]
        assert "original_paper" in roles
        assert item0["title"] == "Adaptive Graph Convolution for Node Classification"

    def test_citation_index_contains_retrieved_papers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ci = self._run_phase2(tmpdir)

        titles = [item["title"] for item in ci["items"]]
        assert any("Graph Convolutional" in t for t in titles)
        assert any("Graph Attention" in t for t in titles)

    def test_core_task_papers_have_correct_role(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ci = self._run_phase2(tmpdir)

        core_task_items = [
            item for item in ci["items"]
            if any(r["type"] == "core_task" for r in item["roles"])
        ]
        assert len(core_task_items) > 0
        for item in core_task_items:
            core_roles = [r for r in item["roles"] if r["type"] == "core_task"]
            assert all("rank" in r for r in core_roles)

    def test_contribution_papers_have_correct_role(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ci = self._run_phase2(tmpdir)

        contrib_items = [
            item for item in ci["items"]
            if any(r["type"] == "contribution" for r in item["roles"])
        ]
        assert len(contrib_items) > 0
        for item in contrib_items:
            contrib_roles = [r for r in item["roles"] if r["type"] == "contribution"]
            for role in contrib_roles:
                assert "scope" in role
                assert role["scope"].startswith("contribution_")

    def test_raw_response_files_created_per_query(self):
        """3 queries total (2 core_task + 1 contribution_1) -> 3 raw files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir) / "paper"
            phase1_dir = base_dir / "phase1"
            phase2_dir = base_dir / "phase2"
            phase1_dir.mkdir(parents=True)

            (phase1_dir / "paper.json").write_text(
                json.dumps({"paper_id": "http://test", "title": "Test Paper", "year": 2024}),
                encoding="utf-8",
            )
            (phase1_dir / "phase1_extracted.json").write_text(
                json.dumps({"core_task": {"text": "test", "query_variants": []}, "contributions": []}),
                encoding="utf-8",
            )

            extracted = ExtractedContent(
                core_task=CoreTask(
                    text="test",
                    query_variants=["query A", "query B"],
                ),
                contributions=[
                    ContributionClaim(
                        id="contribution_1",
                        name="C1",
                        author_claim_text="c",
                        description="",
                        prior_work_query="contrib query",
                        query_variants=[],
                    )
                ],
            )

            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"total": 1, "data": [_make_s2_raw_paper()]}

            with patch("requests.Session.get", return_value=mock_resp):
                with patch("time.sleep"):
                    searcher = PaperSearcher(concurrency=1)
                    stats = searcher.search_all(extracted, phase2_dir)

        raw_files = list((phase2_dir / "raw_responses").glob("raw_*.json"))
        assert len(raw_files) == 3  # raw_core_task_v0, raw_core_task_v1, raw_contribution_1_v0
        assert stats["succeeded"] == 3

    def test_original_paper_excluded_from_candidates(self):
        """The original paper must NOT appear as its own candidate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Add the original paper's exact title to the mock API response
            original_title = "Adaptive Graph Convolution for Node Classification"
            original_paper_s2 = {
                "paperId": "self_ref_id",
                "title": original_title,
                "abstract": "Self reference paper.",
                "year": 2024,
                "authors": [{"name": "Alice Smith"}],
                "venue": "ICLR",
                "publicationVenue": {"name": "ICLR"},
                "externalIds": {},
                "url": "https://openreview.net/pdf?id=TestPaper",
                "openAccessPdf": {},
                "citationCount": 0,
            }
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "total": 1,
                "data": [original_paper_s2],
            }

            base_dir = Path(tmpdir) / "paper"
            phase1_dir = base_dir / "phase1"
            phase2_dir = base_dir / "phase2"
            phase1_dir.mkdir(parents=True)
            (phase1_dir / "paper.json").write_text(
                json.dumps({
                    "paper_id": "https://openreview.net/pdf?id=TestPaper",
                    "title": original_title,
                    "year": 2024,
                }),
                encoding="utf-8",
            )
            (phase1_dir / "phase1_extracted.json").write_text(
                json.dumps({
                    "core_task": {
                        "text": "graph convolution",
                        "query_variants": ["graph convolution"],
                    },
                    "contributions": [],
                }),
                encoding="utf-8",
            )

            extracted = ExtractedContent(
                core_task=CoreTask(text="graph convolution", query_variants=["graph convolution"]),
                contributions=[],
            )

            with patch("requests.Session.get", return_value=mock_resp):
                with patch("time.sleep"):
                    searcher = PaperSearcher(concurrency=1)
                    searcher.search_all(extracted, phase2_dir)

            with patch.dict(os.environ, {"PHASE2_ALIAS_MODE": "heuristic"}):
                processor = Phase2Processor(
                    phase2_dir=phase2_dir,
                    cutoff_year=2025,
                    self_title=original_title,
                )
                processor.process()

            ci = json.loads(
                (phase2_dir / "final" / "citation_index.json").read_text(encoding="utf-8")
            )

        # Original paper appears once (as index 0), not as a candidate too
        titles = [item["title"] for item in ci["items"]]
        assert titles.count(original_title) == 1
        item0 = ci["items"][0]
        assert any(r["type"] == "original_paper" for r in item0["roles"])
