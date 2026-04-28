"""sampling_defaults — recommendation lookup and fill-in semantics."""
from __future__ import annotations

import sampling_defaults as S


def test_lookup_gemma4_variants_all_match():
    rec_a = S.lookup("hf.co/HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive:Q4_K_M")
    rec_b = S.lookup("gemma4:e4b")
    rec_c = S.lookup("mlx:gemma-4-e2b-it-4bit")
    rec_d = S.lookup("llama:hf.co/HauhauCS/Gemma-4-E2B-Uncensored-HauhauCS-Aggressive:latest")
    expected = {"temperature": 1.0, "top_p": 0.95, "top_k": 64}
    for rec in (rec_a, rec_b, rec_c, rec_d):
        assert rec == expected


def test_lookup_qwen3_match():
    assert S.lookup("huihui_ai/qwen3-abliterated:8b") == {"top_p": 0.8, "top_k": 20}
    assert S.lookup("mlx:qwen3.5-9b-4bit") == {"top_p": 0.8, "top_k": 20}


def test_lookup_unknown_returns_empty():
    assert S.lookup("llama3.2:3b") == {}
    assert S.lookup("") == {}


def test_lookup_returns_a_copy_not_a_reference():
    """Mutating the returned dict must not corrupt the table for the
    next caller. Regression guard: easy to forget when refactoring."""
    rec = S.lookup("gemma4:e2b")
    rec["temperature"] = 999
    rec_again = S.lookup("gemma4:e2b")
    assert rec_again["temperature"] == 1.0


def test_fill_defaults_only_fills_unset_fields():
    # User left top_p / top_k unset; recommendations fill them in.
    t, p, k = S.fill_defaults("gemma4:e2b", temperature=0.7, top_p=None, top_k=None)
    assert (t, p, k) == (0.7, 0.95, 64)


def test_fill_defaults_user_values_win():
    t, p, k = S.fill_defaults("gemma4:e2b", temperature=0.3, top_p=0.5, top_k=10)
    assert (t, p, k) == (0.3, 0.5, 10)


def test_fill_defaults_no_recommendations_returns_input_unchanged():
    t, p, k = S.fill_defaults("llama3.2:3b", temperature=0.7, top_p=None, top_k=None)
    assert (t, p, k) == (0.7, None, None)


def test_fill_defaults_partial_recommendation_only_fills_known():
    # qwen3 has no temperature recommendation — only top_p/top_k.
    t, p, k = S.fill_defaults("huihui_ai/qwen3-abliterated:8b", temperature=None, top_p=None, top_k=None)
    assert t is None  # no temp rec, caller's None stays
    assert p == 0.8
    assert k == 20
