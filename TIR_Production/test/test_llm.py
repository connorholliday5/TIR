#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Module: test.test_llm
# The language model is consulted only where the classifiers are weakest, and
# its answer is checked against the closed list of codes before it is used.
# These tests run entirely against FakeBackend: nothing here needs a network,
# and nothing here may send a record anywhere.

import pytest

from src.llm import (
    CodexBackend, FakeBackend, build_prompt, load_definitions, parse_answer,
    suggest,
)


CANDIDATES = ["HAPI Piping", "HABO Bolting", "HA Hanger"]


# -- parsing -----------------------------------------------------------------


# The ordinary case: the model answers with exactly the code asked for.
def test_a_bare_code_is_accepted():
    assert parse_answer("HAPI Piping", CANDIDATES) == "HAPI Piping"


# Models pad answers with punctuation and formatting even when told not to.
def test_surrounding_punctuation_is_ignored():
    assert parse_answer("  **HABO Bolting.**  ", CANDIDATES) == "HABO Bolting"


# Capitalisation is not something to reject an otherwise valid answer over.
def test_matching_is_case_insensitive():
    assert parse_answer("hapi piping", CANDIDATES) == "HAPI Piping"


# Asked for one line, models often give the answer then explain underneath.
def test_the_code_is_found_on_its_own_line():
    reply = "HABO Bolting\n\nThe record describes a loose bolt."
    assert parse_answer(reply, CANDIDATES) == "HABO Bolting"


# A longer code contains a shorter one, and the specific answer is the right
# one: "HAPI Piping" must not be read as the "HA Hanger" it does not contain.
def test_the_longest_matching_code_wins():
    reply = "This is a piping issue, so HAPI Piping is the code."
    assert parse_answer(reply, CANDIDATES) == "HAPI Piping"


# The failure that must never pass through: an invented code would be a
# combination QPS rejects.
def test_a_code_outside_the_list_is_rejected():
    assert parse_answer("HAXX Something Else", CANDIDATES) == ""


# The model is told to answer NONE when nothing fits, and NONE is not a code.
def test_none_is_not_an_answer():
    assert parse_answer("NONE", CANDIDATES) == ""


# An empty reply is a non-answer rather than an error.
def test_an_empty_reply_yields_nothing():
    assert parse_answer("   ", CANDIDATES) == ""


# -- prompting ---------------------------------------------------------------


# The closed list is what the answer is checked against, so it is shown.
def test_the_prompt_lists_every_candidate():
    prompt = build_prompt("cracked weld", "Process Sub", CANDIDATES)

    for name in CANDIDATES:
        assert name in prompt


# The definitions are the reason for consulting a model at all.
def test_definitions_reach_the_prompt():
    prompt = build_prompt(
        "cracked weld", "Process Sub", CANDIDATES,
        {"HAPI Piping": "hangers supporting piping runs"},
    )

    assert "hangers supporting piping runs" in prompt


# -- suggesting --------------------------------------------------------------


# End to end through the fake: a valid answer comes back with its reasoning.
def test_a_valid_answer_is_returned_with_its_rationale():
    backend = FakeBackend({"cracked weld": "HAPI Piping — it names a pipe hanger"})

    code, rationale = suggest("cracked weld on hanger", "Process Sub", CANDIDATES, backend)

    assert code == "HAPI Piping"
    assert "pipe hanger" in rationale


# A rejected answer must not arrive with a rationale that explains it, or the
# coder would be shown reasoning for a code that was never used.
def test_a_rejected_answer_carries_no_rationale():
    backend = FakeBackend({"cracked weld": "HAXX Invented — sounds about right"})

    code, rationale = suggest("cracked weld on hanger", "Process Sub", CANDIDATES, backend)

    assert code == ""
    assert rationale == ""


# Nothing to choose from is not a question worth asking.
def test_no_candidates_means_no_call():
    backend = FakeBackend()

    code, rationale = suggest("cracked weld", "Process Sub", [], backend)

    assert (code, rationale) == ("", "")
    assert backend.prompts == []


# -- the real backend --------------------------------------------------------


# The records carry EB Proprietary markings; sending them anywhere is a
# decision the coding team makes, not a default.
def test_the_codex_backend_refuses_to_send_anything_by_default():
    with pytest.raises(PermissionError, match="allow_calls"):
        CodexBackend(endpoint="https://example.invalid").complete("anything")


# -- definitions -------------------------------------------------------------


# The dictionary is optional, and its absence is not an error.
def test_missing_definitions_are_not_an_error(tmp_path):
    assert load_definitions(tmp_path / "nothing.json") == {}


# The file carries a _README list among the fields, which is not a field.
def test_non_field_entries_are_skipped(tmp_path):
    path = tmp_path / "defs.json"
    path.write_text('{"_README": ["notes"], "process_cat": {"BL Blasting": "abrasive"}}')

    assert load_definitions(path) == {"process_cat": {"BL Blasting": "abrasive"}}
