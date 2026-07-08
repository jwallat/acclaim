"""Acclaim explorer server configuration — self-contained, no private library internals."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from acclaim.config.load import (
    AtomicClaimExtractorConfig,
    ConcurrencyConfig,
    EvalConfig,
    JudgeConfig,
    TokenOverlapConfig,
)

_ACCLAIM_EXPLORER_DIR = Path(__file__).parent.parent
_DEFAULT_EVAL_CONFIG = _ACCLAIM_EXPLORER_DIR / "eval_config.yaml"
_DEFAULT_EXPLORER_CONFIG = _ACCLAIM_EXPLORER_DIR / "acclaim_explorer_config.yaml"


def _resolve(value: object) -> object:
    """Substitute ${ENV_VAR} placeholders with their environment values."""
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        return os.environ.get(value[2:-1])
    return value


def _parse_judge(raw: dict) -> JudgeConfig:
    return JudgeConfig(
        model=str(_resolve(raw.get("model", "gpt-4o-mini"))),
        api_base=_resolve(raw.get("api_base")) or None,  # type: ignore[arg-type]
        api_key=_resolve(raw.get("api_key")) or None,  # type: ignore[arg-type]
        temperature=float(raw.get("temperature", 0.0)),
        max_tokens=int(raw.get("max_tokens", 512)),
        max_retries=int(raw.get("max_retries", 3)),
        thinking=bool(raw.get("thinking", False)),
        system_prompt=raw.get("system_prompt") or None,
    )


def _parse_atomic_claim_extractor(raw: dict) -> AtomicClaimExtractorConfig:
    return AtomicClaimExtractorConfig(
        model=_resolve(raw.get("model")) or None,  # type: ignore[arg-type]
        api_base=_resolve(raw.get("api_base")) or None,  # type: ignore[arg-type]
        api_key=_resolve(raw.get("api_key")) or None,  # type: ignore[arg-type]
        temperature=raw.get("temperature"),
        max_tokens=int(raw.get("max_tokens", 1024)),
        max_retries=raw.get("max_retries"),
        thinking=raw.get("thinking"),
        system_prompt=raw.get("system_prompt") or None,
    )


def _parse_eval(raw: dict) -> EvalConfig:
    judge = _parse_judge(raw.get("judge", {}))

    def _optional_judge(key: str) -> JudgeConfig | None:
        section = raw.get(key)
        return _parse_judge(section) if isinstance(section, dict) else None

    atomic_raw = raw.get("atomic_claim_extractor")
    atomic_claim_extractor = (
        _parse_atomic_claim_extractor(atomic_raw)
        if isinstance(atomic_raw, dict)
        else AtomicClaimExtractorConfig()
    )

    return EvalConfig(
        claim_extractor=raw.get("claim_extractor", "sentence"),
        judge=judge,
        atomic_claim_extractor=atomic_claim_extractor,
        token_overlap=TokenOverlapConfig(
            log_similarities=raw.get("token_overlap", {}).get("log_similarities", False)
        ),
        concurrency=ConcurrencyConfig(
            max_workers=raw.get("concurrency", {}).get("max_workers", 1)
        ),
        metrics=raw.get(
            "metrics",
            ["citation_precision", "citation_recall", "citation_f1",
             "citation_correctness", "citation_lengths", "citation_number",
             "coverage", "cvcp", "hallucination_rate", "supported_claim_rate",
             "token_overlap"],
        ),
        citation_recall_judge=_optional_judge("citation_recall_judge"),
        citation_precision_judge=_optional_judge("citation_precision_judge"),
        relevance_judge=_optional_judge("relevance_judge"),
        relevance_stratified_metrics=raw.get("relevance_stratified_metrics", []),
    )


@dataclass
class GeneratorConfig:
    model: str = "gpt-4o-mini"
    api_base: str | None = None
    api_key: str | None = None
    temperature: float = 0.7
    max_tokens: int = 800
    system_prompt: str | None = None


@dataclass
class RetrieverConfig:
    default: str = "bm25"
    top_k: int = 7
    bm25_index_path: str | None = None       # path to a local Lucene index
    bm25_prebuilt_index: str | None = None   # pyserini prebuilt index name (auto-downloads)


@dataclass
class AcclaimExplorerConfig:
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def load_acclaim_explorer_config(
    eval_path: str | Path | None = None,
    explorer_path: str | Path | None = None,
) -> AcclaimExplorerConfig:
    with open(eval_path or _DEFAULT_EVAL_CONFIG, encoding="utf-8") as fh:
        eval_raw = yaml.safe_load(fh) or {}
    with open(explorer_path or _DEFAULT_EXPLORER_CONFIG, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    retr = raw.get("retriever", {})
    gen = raw.get("generator", {})

    retriever_config = RetrieverConfig(
        default=retr.get("default", "bm25"),
        top_k=int(retr.get("top_k", 7)),
        bm25_index_path=retr.get("bm25", {}).get("index_path"),
        bm25_prebuilt_index=retr.get("bm25", {}).get("prebuilt_index"),
    )

    generator_config = GeneratorConfig(
        model=str(_resolve(gen.get("model", "gpt-4o-mini"))),
        api_base=_resolve(gen.get("api_base")) or None,  # type: ignore[arg-type]
        api_key=_resolve(gen.get("api_key")) or None,  # type: ignore[arg-type]
        temperature=float(gen.get("temperature", 0.7)),
        max_tokens=int(gen.get("max_tokens", 800)),
        system_prompt=gen.get("system_prompt") or None,
    )

    return AcclaimExplorerConfig(
        retriever=retriever_config,
        generator=generator_config,
        eval=_parse_eval(eval_raw),
    )
