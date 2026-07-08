"""
Unit tests for acclaim.config.load: load_config, _parse_config, env var
substitution, and config_to_dict redaction.
"""

import pytest

from acclaim.config.load import (
    EvalConfig,
    JudgeConfig,
    _parse_config,
    config_to_dict,
    load_config,
)


def test_load_config_defaults_reads_bundled_yaml():
    cfg = load_config()
    assert isinstance(cfg, EvalConfig)
    assert cfg.claim_extractor in ("sentence", "atomic")
    assert "coverage" in cfg.metrics


def test_load_config_custom_path(tmp_path):
    custom = tmp_path / "custom.yaml"
    custom.write_text(
        "claim_extractor: sentence\n"
        "metrics:\n"
        "  - coverage\n"
    )
    cfg = load_config(custom)
    assert cfg.claim_extractor == "sentence"
    assert cfg.metrics == ["coverage"]


def test_parse_config_empty_dict_uses_defaults():
    cfg = _parse_config({})
    assert cfg.claim_extractor == "sentence"
    assert cfg.judge.model == "gpt-4o-mini"
    assert cfg.judge.max_retries == 3
    assert cfg.concurrency.max_workers == 1


def test_parse_config_env_var_resolved(monkeypatch):
    monkeypatch.setenv("MY_TEST_API_KEY", "secret-value")
    cfg = _parse_config({"judge": {"api_key": "${MY_TEST_API_KEY}"}})
    assert cfg.judge.api_key == "secret-value"


def test_parse_config_unset_env_var_resolves_to_none(monkeypatch):
    monkeypatch.delenv("MY_UNSET_API_KEY", raising=False)
    cfg = _parse_config({"judge": {"api_key": "${MY_UNSET_API_KEY}"}})
    assert cfg.judge.api_key is None


def test_parse_config_judge_overrides():
    raw = {
        "citation_recall_judge": {"model": "claude-3-5-sonnet-20241022"},
        "citation_precision_judge": {"model": "gpt-4o"},
        "relevance_judge": {"model": "gemini-pro"},
    }
    cfg = _parse_config(raw)
    assert cfg.citation_recall_judge.model == "claude-3-5-sonnet-20241022"
    assert cfg.citation_precision_judge.model == "gpt-4o"
    assert cfg.relevance_judge.model == "gemini-pro"


def test_parse_config_missing_judge_overrides_default_to_none():
    cfg = _parse_config({})
    assert cfg.citation_recall_judge is None
    assert cfg.citation_precision_judge is None
    assert cfg.relevance_judge is None


def test_parse_config_atomic_claim_extractor_falls_back_to_none():
    cfg = _parse_config({"claim_extractor": "atomic"})
    assert cfg.atomic_claim_extractor.model is None
    assert cfg.atomic_claim_extractor.max_tokens == 1024


def test_parse_config_jaccard_aligner_weighted_default_true():
    cfg = _parse_config({})
    assert cfg.jaccard_aligner.weighted is True


def test_parse_config_jaccard_aligner_weighted_override():
    cfg = _parse_config({"jaccard_aligner": {"weighted": False}})
    assert cfg.jaccard_aligner.weighted is False


def test_config_to_dict_redacts_api_key():
    cfg = EvalConfig(judge=JudgeConfig(api_key="super-secret"))
    d = config_to_dict(cfg)
    assert d["judge"]["api_key"] == "***REDACTED***"


def test_config_to_dict_leaves_none_api_key_unredacted():
    cfg = EvalConfig(judge=JudgeConfig(api_key=None))
    d = config_to_dict(cfg)
    assert d["judge"]["api_key"] is None


def test_config_to_dict_redacts_nested_judge_overrides():
    cfg = EvalConfig(citation_recall_judge=JudgeConfig(api_key="another-secret"))
    d = config_to_dict(cfg)
    assert d["citation_recall_judge"]["api_key"] == "***REDACTED***"
