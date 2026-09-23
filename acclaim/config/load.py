"""
Configuration loading for acclaim.

Loads YAML config and returns a typed :class:`EvalConfig` dataclass
that can be passed directly to :func:`~acclaim.evaluate.evaluate`.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TokenOverlapConfig:
    """TokenOverlap metric configuration."""

    log_similarities: bool = False


@dataclass
class JaccardAlignerConfig:
    """Configuration for :class:`~acclaim.alignment.jaccard.JaccardCitationAligner`."""

    weighted: bool = True
    """
    If ``True`` (default), weight tokens by inverse claim-frequency within
    the same answer before computing Jaccard similarity, correcting a bias
    where claims restating a coreference-resolved subject get matched to
    the sentence that introduces that subject rather than the one
    containing the cited fact. Set to ``False`` for plain Jaccard.
    """


@dataclass
class LLMAlignerConfig:
    """
    LiteLLM citation aligner configuration.

    ``model``, ``api_base``, ``api_key``, ``temperature``, ``max_retries``,
    and ``thinking`` default to ``None``, meaning "reuse the top-level
    :attr:`~EvalConfig.judge` config" — so users only need to configure one
    LLM endpoint. ``max_tokens`` and ``system_prompt`` are aligner-specific
    (a batched alignment response needs headroom for many claims' worth of
    JSON), so they carry their own standalone defaults instead of falling
    back to :attr:`~EvalConfig.judge`.
    """

    model: str | None = None
    api_base: str | None = None
    api_key: str | None = None
    temperature: float | None = None
    max_tokens: int = 2048
    max_retries: int | None = None
    thinking: bool | None = None
    system_prompt: str | None = None


@dataclass
class ConcurrencyConfig:
    """Thread-pool concurrency configuration for LLM judge calls."""

    max_workers: int = 1
    """
    Number of worker threads for parallel judge calls within a single
    ``evaluate()`` run (claim-level / citation-level) and across batch
    examples in ``evaluate_batch()``. ``1`` (default) preserves fully
    sequential execution — no thread pool is created. Raise this once
    you've confirmed your LLM endpoint/provider can handle concurrent
    requests.
    """


_DEFAULT_YAML = Path(__file__).parent / "default.yaml"


@dataclass
class JudgeConfig:
    """LiteLLM judge configuration."""

    model: str = "gpt-4o-mini"
    api_base: str | None = None
    api_key: str | None = None
    temperature: float = 0.0
    max_tokens: int = 512
    max_retries: int = 3
    thinking: bool = False
    system_prompt: str | None = None


@dataclass
class AtomicClaimExtractorConfig:
    """
    LiteLLM atomic claim extractor configuration.

    ``model``, ``api_base``, ``api_key``, ``temperature``, ``max_retries``,
    and ``thinking`` default to ``None``, meaning "reuse the top-level
    :attr:`~EvalConfig.judge` config" — so users only need to configure one
    LLM endpoint. ``max_tokens`` and ``system_prompt`` are extractor-specific
    (claim decomposition needs a different prompt and typically more output
    tokens than evidence judgment), so they carry their own standalone
    defaults instead of falling back to :attr:`~EvalConfig.judge`.
    """

    model: str | None = None
    api_base: str | None = None
    api_key: str | None = None
    temperature: float | None = None
    max_tokens: int = 1024
    max_retries: int | None = None
    thinking: bool | None = None
    system_prompt: str | None = None


@dataclass
class EvalConfig:
    """
    Typed configuration for :func:`~acclaim.evaluate.evaluate`.

    Usage::

        config = load_config()                         # defaults
        config = load_config("config/custom.yaml")     # custom file
        result = evaluate(answer, documents, config=config)
    """

    claim_extractor: str = "sentence"
    aligner: str | None = None
    """
    Explicit aligner override: ``"sentence"``, ``"jaccard"``, or ``"llm"``.
    ``None`` (default) picks the aligner implied by :attr:`claim_extractor`
    (``sentence`` → :class:`~acclaim.alignment.sentence.SentenceCitationAligner`,
    ``atomic`` → :class:`~acclaim.alignment.llm.LLMCitationAligner`). Set this
    explicitly to force a specific aligner regardless of extractor — e.g. to
    keep using :class:`~acclaim.alignment.jaccard.JaccardCitationAligner`
    with ``atomic`` claims.
    """
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    atomic_claim_extractor: AtomicClaimExtractorConfig = field(
        default_factory=AtomicClaimExtractorConfig
    )
    """Configuration for :class:`~acclaim.claims.atomic.AtomicClaimExtractor`. Only used when ``claim_extractor: atomic``."""
    jaccard_aligner: JaccardAlignerConfig = field(default_factory=JaccardAlignerConfig)
    """Configuration for :class:`~acclaim.alignment.jaccard.JaccardCitationAligner`. Only used when it is the selected aligner."""
    llm_aligner: LLMAlignerConfig = field(default_factory=LLMAlignerConfig)
    """Configuration for :class:`~acclaim.alignment.llm.LLMCitationAligner`. Only used when it is the selected aligner (the default for ``claim_extractor: atomic``)."""
    token_overlap: TokenOverlapConfig = field(default_factory=TokenOverlapConfig)
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)
    metrics: list[str] = field(
        default_factory=lambda: [
            "citation_precision",
            "citation_recall",
            "citation_f1",
            "citation_correctness",
            "citation_lengths",
            "citation_number",
            "coverage",
            "cvcp",
            "hallucination_rate",
            "supported_claim_rate",
            "token_overlap",
        ]
    )
    citation_recall_judge: JudgeConfig | None = None
    """
    Optional separate LLM config for the CitationRecall metric.
    When ``None`` (default), the metric reuses :attr:`judge`.
    """
    citation_precision_judge: JudgeConfig | None = None
    """
    Optional separate LLM config for the CitationPrecision metric.
    When ``None`` (default), the metric reuses :attr:`judge`.
    """
    relevance_judge: JudgeConfig | None = None
    """
    Optional separate LLM config for the relevance classification judge.
    When ``None`` (default), the judge reuses :attr:`judge`.
    """
    relevance_stratified_metrics: list[str] = field(default_factory=list)
    """
    Base metrics to additionally stratify by claim relevance (CORE /
    COMPLEMENTARY / IRRELEVANT). Opt-in — empty by default, independent of
    :attr:`metrics`. Each name must be present in ``METRIC_REGISTRY``. Only
    used when ``evaluate()`` is called with a ``question``.
    """


def load_config(path: str | Path | None = None) -> EvalConfig:
    """
    Load configuration from a YAML file.

    Args:
        path: Path to a YAML config file.  Defaults to the built-in
              ``config/default.yaml``.

    Returns:
        A fully populated :class:`EvalConfig`.
    """
    raw = _load_yaml(path or _DEFAULT_YAML)
    return _parse_config(raw)


def local_vllm_config(
    model: str = "openai/meta-llama-3-8b-instruct",
    api_base: str = "http://localhost:8000/v1",
    claim_extractor: str = "sentence",
) -> EvalConfig:
    """
    Convenience factory for a local vLLM setup.

    Usage::

        config = local_vllm_config(model="openai/my-model")
        result = evaluate(answer, documents, config=config)
    """
    return EvalConfig(
        claim_extractor=claim_extractor,
        judge=JudgeConfig(
            model=model,
            api_base=api_base,
            api_key="dummy",
        ),
    )


_REDACTED = "***REDACTED***"


def config_to_dict(config: EvalConfig) -> dict[str, Any]:
    """Serialize an EvalConfig to a plain dict with api_key fields redacted."""
    d = asdict(config)
    _redact_api_keys(d)
    return d


def _redact_api_keys(d: dict[str, Any]) -> None:
    for k, v in d.items():
        if k == "api_key" and v is not None:
            d[k] = _REDACTED
        elif isinstance(v, dict):
            _redact_api_keys(v)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _parse_config(raw: dict[str, Any]) -> EvalConfig:
    judge_raw = raw.get("judge", {})
    if not isinstance(judge_raw, dict):
        judge_raw = {}

    token_overlap_raw = raw.get("token_overlap", {})
    if not isinstance(token_overlap_raw, dict):
        token_overlap_raw = {}

    concurrency_raw = raw.get("concurrency", {})
    if not isinstance(concurrency_raw, dict):
        concurrency_raw = {}

    # Resolve env var substitutions like ${OPENAI_API_KEY}
    def _resolve(value: Any) -> Any:
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            return os.environ.get(value[2:-1])
        return value

    judge = JudgeConfig(
        model=_resolve(judge_raw.get("model", "gpt-4o-mini")),
        api_base=_resolve(judge_raw.get("api_base")) or None,
        api_key=_resolve(judge_raw.get("api_key")) or None,
        temperature=judge_raw.get("temperature", 0.0),
        max_tokens=judge_raw.get("max_tokens", 512),
        max_retries=judge_raw.get("max_retries", 3),
        thinking=judge_raw.get("thinking", False),
        system_prompt=judge_raw.get("system_prompt") or None,
    )

    token_overlap = TokenOverlapConfig(
        log_similarities=token_overlap_raw.get("log_similarities", False),
    )

    concurrency = ConcurrencyConfig(
        max_workers=concurrency_raw.get("max_workers", 1),
    )

    def _parse_judge_override(raw_section: dict) -> JudgeConfig:
        return JudgeConfig(
            model=_resolve(raw_section.get("model", "gpt-4o-mini")),
            api_base=_resolve(raw_section.get("api_base")) or None,
            api_key=_resolve(raw_section.get("api_key")) or None,
            temperature=raw_section.get("temperature", 0.0),
            max_tokens=raw_section.get("max_tokens", 512),
            max_retries=raw_section.get("max_retries", 3),
            thinking=raw_section.get("thinking", False),
            system_prompt=raw_section.get("system_prompt") or None,
        )

    # Optional per-metric judge overrides
    recall_judge_raw = raw.get("citation_recall_judge")
    citation_recall_judge: JudgeConfig | None = (
        _parse_judge_override(recall_judge_raw)
        if isinstance(recall_judge_raw, dict)
        else None
    )

    precision_judge_raw = raw.get("citation_precision_judge")
    citation_precision_judge: JudgeConfig | None = (
        _parse_judge_override(precision_judge_raw)
        if isinstance(precision_judge_raw, dict)
        else None
    )

    relevance_judge_raw = raw.get("relevance_judge")
    relevance_judge: JudgeConfig | None = (
        _parse_judge_override(relevance_judge_raw)
        if isinstance(relevance_judge_raw, dict)
        else None
    )

    atomic_claim_extractor_raw = raw.get("atomic_claim_extractor", {})
    if not isinstance(atomic_claim_extractor_raw, dict):
        atomic_claim_extractor_raw = {}
    atomic_claim_extractor = AtomicClaimExtractorConfig(
        model=_resolve(atomic_claim_extractor_raw.get("model")) or None,
        api_base=_resolve(atomic_claim_extractor_raw.get("api_base")) or None,
        api_key=_resolve(atomic_claim_extractor_raw.get("api_key")) or None,
        temperature=atomic_claim_extractor_raw.get("temperature"),
        max_tokens=atomic_claim_extractor_raw.get("max_tokens", 1024),
        max_retries=atomic_claim_extractor_raw.get("max_retries"),
        thinking=atomic_claim_extractor_raw.get("thinking"),
        system_prompt=atomic_claim_extractor_raw.get("system_prompt") or None,
    )

    jaccard_aligner_raw = raw.get("jaccard_aligner", {})
    if not isinstance(jaccard_aligner_raw, dict):
        jaccard_aligner_raw = {}
    jaccard_aligner = JaccardAlignerConfig(
        weighted=jaccard_aligner_raw.get("weighted", True),
    )

    llm_aligner_raw = raw.get("llm_aligner", {})
    if not isinstance(llm_aligner_raw, dict):
        llm_aligner_raw = {}
    llm_aligner = LLMAlignerConfig(
        model=_resolve(llm_aligner_raw.get("model")) or None,
        api_base=_resolve(llm_aligner_raw.get("api_base")) or None,
        api_key=_resolve(llm_aligner_raw.get("api_key")) or None,
        temperature=llm_aligner_raw.get("temperature"),
        max_tokens=llm_aligner_raw.get("max_tokens", 2048),
        max_retries=llm_aligner_raw.get("max_retries"),
        thinking=llm_aligner_raw.get("thinking"),
        system_prompt=llm_aligner_raw.get("system_prompt") or None,
    )

    return EvalConfig(
        claim_extractor=raw.get("claim_extractor", "sentence"),
        aligner=raw.get("aligner") or None,
        judge=judge,
        atomic_claim_extractor=atomic_claim_extractor,
        jaccard_aligner=jaccard_aligner,
        llm_aligner=llm_aligner,
        token_overlap=token_overlap,
        concurrency=concurrency,
        citation_recall_judge=citation_recall_judge,
        citation_precision_judge=citation_precision_judge,
        relevance_judge=relevance_judge,
        relevance_stratified_metrics=raw.get("relevance_stratified_metrics", []),
        metrics=raw.get(
            "metrics",
            [
                "citation_precision",
                "citation_recall",
                "citation_f1",
                "citation_correctness",
                "citation_lengths",
                "citation_number",
                "coverage",
                "cvcp",
                "hallucination_rate",
                "supported_claim_rate",
                "token_overlap",
            ],
        ),
    )
