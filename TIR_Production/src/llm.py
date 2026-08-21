#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Module: src.llm
# Implements:
#   - REQ-014 (Language-model suggestions for the records the models cannot code)
# A language model consulted only where the trained classifiers are weakest.
#
# The classifiers are at the limit of their data: they match how consistently
# people code these fields, and no amount of further modelling moves that.  The
# exception is the rare tail — five Process categories appear so seldom that
# the models have never once got them right — and the tail is exactly where a
# language model has an advantage, because it can read what a code *means*
# rather than infer it from examples it does not have.
#
# So this is a fallback, not a replacement.  A record the classifiers are
# confident about never reaches it.

import json
import os
import re
from typing import Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

from src.config import CODE_DEFINITIONS, LLM


# What any backend must provide.  Keeping this to one method is what lets the
# tests run the whole path against a fake, and what will make swapping in the
# real endpoint a change to one class rather than to the pipeline.
class LLMBackend(Protocol):
    def complete(self, prompt: str, *, max_tokens: int = 256) -> str: ...


# A backend that answers from a lookup table, for the tests.
class FakeBackend:
    """Answers from `replies`, or with the first candidate offered.

    Deterministic on purpose: the suggestion path has routing, parsing and
    validation worth testing, and none of it should need a network to test.
    """

    def __init__(self, replies: Optional[Mapping[str, str]] = None):
        self.replies = dict(replies or {})
        self.prompts: List[str] = []

    def complete(self, prompt: str, *, max_tokens: int = 256) -> str:
        self.prompts.append(prompt)
        for key, reply in self.replies.items():
            if key.lower() in prompt.lower():
                return reply
        return ""


# The airgapped Codex endpoint.
class CodexBackend:
    """Calls the endpoint named by the environment.

    Refuses to send anything unless `llm.allow_calls` is set in config.json.
    The records carry EB Proprietary markings, and whether they may be put
    through a hosted model — even one inside the airgap — is a sign-off the
    coding team has to obtain rather than something a default should decide.
    Bulk processing of the year's 40,000-odd records may also be a separate
    question from a coder consulting it on one TIR.
    """

    def __init__(self, endpoint: str = "", api_key: str = "", model: str = ""):
        self.endpoint = endpoint or os.environ.get("TIR_LLM_ENDPOINT", "")
        self.api_key = api_key or os.environ.get("TIR_LLM_KEY", "")
        self.model = model or LLM.get("model", "")

    def complete(self, prompt: str, *, max_tokens: int = 256) -> str:
        if not LLM.get("allow_calls", False):
            raise PermissionError(
                "Sending TIR text to a language model is switched off. Set "
                "llm.allow_calls in config.json once the coding team has "
                "sign-off for record text leaving the process."
            )
        if not self.endpoint:
            raise RuntimeError(
                "No endpoint configured. Set TIR_LLM_ENDPOINT, or pass one to "
                "CodexBackend."
            )
        raise NotImplementedError(
            "Fill in the request for the airgapped endpoint once its shape is "
            "known — see docs in reports/llm.md. Everything either side of "
            "this call is already tested against FakeBackend."
        )


# Builds the question put to the model for one field.
def build_prompt(
    text: str, field: str, candidates: Sequence[str],
    definitions: Optional[Mapping[str, str]] = None,
) -> str:
    """Return the prompt asking for one code out of `candidates`.

    The candidate list is included in full rather than left implicit.  It is
    what the answer is checked against, and a model shown the closed list is
    answering the question the coders are actually asked.
    """
    definitions = definitions or {}
    lines = [
        "You are helping code a quality record at a shipyard.",
        "",
        f"Choose the one {field} code that best fits the record below.",
        "Answer with the code exactly as written in the list, and nothing else.",
        "If none of them fit, answer NONE.",
        "",
        "Record:",
        text.strip(),
        "",
        f"The {field} codes to choose from:",
    ]
    for name in candidates:
        meaning = definitions.get(name, "")
        lines.append(f"- {name}" + (f" — {meaning}" if meaning else ""))
    lines += ["", "Answer:"]
    return "\n".join(lines)


# Recovers the chosen code from whatever the model replied.
def parse_answer(reply: str, candidates: Sequence[str]) -> str:
    """Return the candidate the reply names, or "" if it names none.

    Matched against the closed list rather than trusted as written, so a model
    that invents a plausible-looking code, pads the answer with a sentence of
    reasoning, or changes the capitalisation is handled the same way.  An
    invented code would be a combination QPS rejects, which is the one failure
    this must not pass through.
    """
    answer = reply.strip()
    if not answer:
        return ""

    lookup = {c.strip().lower(): c for c in candidates}

    # Whole reply, then line by line: models often answer on the first line and
    # explain underneath even when asked not to.
    for line in [answer, *answer.splitlines()]:
        cleaned = line.strip().strip(".:;-*` ").lower()
        if cleaned in lookup:
            return lookup[cleaned]

    # Falls back to naming a candidate anywhere in the reply, longest first so
    # that "HAPI Piping" is preferred over a bare "HA Hanger" it contains.
    for name in sorted(candidates, key=len, reverse=True):
        if re.search(rf"\b{re.escape(name)}\b", answer, flags=re.IGNORECASE):
            return name

    return ""


# Asks the model for one field on one record.
def suggest(
    text: str, field: str, candidates: Sequence[str], backend: LLMBackend,
    definitions: Optional[Mapping[str, str]] = None,
) -> Tuple[str, str]:
    """Return (code, rationale) — both empty when the model does not answer.

    The code is always one of `candidates` or empty; see `parse_answer`.  The
    rationale is the model's own words, and is carried through to the coder
    because being able to say why is the thing the classifiers cannot do.
    """
    if not candidates:
        return "", ""

    prompt = build_prompt(text, field, candidates, definitions)
    reply = backend.complete(prompt, max_tokens=int(LLM.get("max_tokens", 256)))
    code = parse_answer(reply, candidates)
    rationale = reply.strip() if code else ""
    return code, rationale


# Reads the code dictionary, if the team has supplied one.
def load_definitions(path=CODE_DEFINITIONS) -> Dict[str, Dict[str, str]]:
    """Return {field: {code: meaning}}, empty when no dictionary exists.

    Optional by design: without it the model still gets the closed candidate
    list, which is most of the benefit.  With it, the model can reason about a
    code it has almost no examples of, which is the whole reason for consulting
    one.  `reports sufficiency` states how much of the taxonomy is covered.
    """
    if not path.is_file():
        return {}

    raw = json.loads(path.read_text())
    return {
        field: {str(k): str(v) for k, v in codes.items()}
        for field, codes in raw.items()
        if isinstance(codes, dict)
    }
