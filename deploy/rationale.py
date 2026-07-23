"""Retention-brief generation with guardrails (H2-4).

Turns a computed assessment into a short written brief for a retention
specialist. The design point matters more than the code:

**The LLM is a writing tool, not a reasoning tool.**

It receives *only* facts this project has already computed deterministically —
the probability, the risk band, the log-odds contributions, the segment
reference rates. It never sees the raw customer record, is never asked to
predict anything, and is never asked *why* a customer will churn. Asking a
language model to explain a prediction would invent causal claims the model
cannot support, which is precisely the failure this project avoids everywhere
else.

Guardrails, all enforced here rather than assumed:

1. **Grounding** — the prompt carries computed values only.
2. **Structured output** — a fixed schema, validated before display.
3. **Prohibited language** — causal and deterministic phrasing is rejected.
4. **Provenance** — output is always labelled as generated.
5. **Kill switch** — disabled by default; the application is fully functional
   without it and falls back to a deterministic template.
6. **No secrets in the repository** — the token is read from the environment,
   supplied as a Space secret.

The deterministic template is not a degraded mode. It is the default, it is
always available, and the generated variant must clear every guardrail before it
is preferred.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("churn_app.rationale")

#: Environment flag. Absent or false means the layer is off — the shipped default.
ENABLE_ENV_VAR = "ENABLE_LLM_RATIONALE"
TOKEN_ENV_VAR = "HF_TOKEN"

#: Small instruction-tuned model; the task is rendering, not reasoning.
DEFAULT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
MAX_NEW_TOKENS = 320
REQUEST_TIMEOUT_SECONDS = 20

#: Phrasing that asserts causation or certainty. The model must describe what a
#: score reflects, never why a person will act or what will happen to them.
PROHIBITED_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bwill (?:churn|leave|cancel|defect)\b", "asserts a future outcome as certain"),
    (r"\bis going to (?:churn|leave|cancel)\b", "asserts a future outcome as certain"),
    (r"\bbecause (?:they|the customer|he|she)\b", "asserts a cause for an individual"),
    (r"\bdue to the fact that\b", "asserts causation"),
    (r"\bguarantee[sd]?\b", "asserts certainty"),
    (r"\bcertainly\b|\bdefinitely\b|\bundoubtedly\b", "asserts certainty"),
    (r"\bproves?\b|\bproven\b", "overstates evidential strength"),
    # Matches an assertion of cause, not the word "causes" in a disclaimer such
    # as "associations, not causes" — the deterministic template contains
    # exactly that phrase and must pass its own guardrail.
    (r"\bcaused by\b|\bis caused\b|\bcauses (?:them|the customer|churn|this)\b",
     "asserts causation"),
    (r"\bshould be offered\b|\boffer (?:them|the customer) a discount\b",
     "prescribes a commercial action the model must not recommend"),
    (r"\breduce (?:their|the) price\b|\bdiscount\b", "prescribes a pricing action"),
    (r"\bterminate\b|\bcancel (?:their|the) (?:contract|service)\b",
     "prescribes a contract action"),
)

SYSTEM_PROMPT = """You are a careful analyst writing a short internal briefing note for a \
customer-retention specialist at a telecommunications company.

You will be given ONLY pre-computed model outputs. You have no other information about the \
customer, and you must not infer any.

Absolute rules:
- Never state or imply that the customer WILL leave. The model produces a probability, not a
  prediction about a person.
- Never explain WHY the customer might leave. You are given statistical contributions, which
  are associations in a training sample, not causes.
- Never recommend a price change, a discount, a contract change, or any customer-facing offer.
- Never invent any fact not present in the input.
- Write for a professional colleague: plain, measured, no marketing language, no emoji.

Produce exactly four sections with these headings and nothing else:

SUMMARY: two sentences describing what the score is and how it compares to the sample.
FACTORS: three bullet points naming the attributes that contributed most, described as
associated with higher or lower scores. Do not explain them causally.
QUESTIONS: three questions the specialist could ask when reviewing the account.
CAVEATS: one sentence noting this is decision support requiring human review."""


@dataclass(frozen=True)
class Brief:
    """A retention brief, from either source."""

    text: str
    generated: bool
    fallback_reason: str = ""

    @property
    def provenance(self) -> str:
        return (
            "AI-generated from pre-computed model outputs"
            if self.generated
            else "Deterministic template — no AI generation"
        )


def is_enabled() -> bool:
    """Whether generation is switched on and credentials are present."""
    flag = os.environ.get(ENABLE_ENV_VAR, "").strip().lower()
    return flag in {"1", "true", "yes", "on"} and bool(os.environ.get(TOKEN_ENV_VAR))


def check_prohibited(text: str) -> list[str]:
    """Return the reasons *text* fails the language guardrail."""
    lowered = text.lower()
    return [
        reason
        for pattern, reason in PROHIBITED_PATTERNS
        if re.search(pattern, lowered)
    ]


def validate_structure(text: str) -> list[str]:
    """Return the reasons *text* fails the structural contract."""
    problems: list[str] = []
    for heading in ("SUMMARY", "FACTORS", "QUESTIONS", "CAVEATS"):
        if heading not in text.upper():
            problems.append(f"missing the {heading} section")
    if len(text.strip()) < 80:
        problems.append("output too short to be a usable brief")
    if len(text) > 3000:
        problems.append("output far longer than the requested format")
    return problems


def build_facts(
    probability: float,
    tier: str,
    threshold: float,
    contributions: list[tuple[str, str, float]],
    overall_rate: float | None,
) -> str:
    """Assemble the grounded fact block. Only computed values appear here."""
    lines = [
        f"Estimated churn probability: {probability:.1%}",
        f"Risk band: {tier}",
        f"Decision threshold in use: {threshold:.2f}",
        f"Model classification at this threshold: "
        f"{'flagged for review' if probability >= threshold else 'not flagged'}",
    ]
    if overall_rate is not None:
        lines.append(f"Churn rate across the whole training sample: {overall_rate:.1%}")

    lines.append("")
    lines.append("Attributes contributing most to this score (log-odds contribution):")
    for name, value, contribution in contributions:
        direction = "higher" if contribution > 0 else "lower"
        lines.append(f"- {name} = {value}: associated with a {direction} score "
                     f"({contribution:+.3f})")
    return "\n".join(lines)


def deterministic_brief(
    probability: float,
    tier: str,
    threshold: float,
    contributions: list[tuple[str, str, float]],
    overall_rate: float | None,
) -> str:
    """The template brief. Always available, always correct, never invented."""
    comparison = ""
    if overall_rate is not None:
        if probability > overall_rate:
            comparison = (
                f" That is above the {overall_rate:.1%} churn rate observed across the "
                "training sample."
            )
        else:
            comparison = (
                f" That is at or below the {overall_rate:.1%} churn rate observed across "
                "the training sample."
            )

    flagged = "flagged for review" if probability >= threshold else "not flagged"
    raising = [c for c in contributions if c[2] > 0][:3]
    lowering = [c for c in contributions if c[2] < 0][:3]

    parts = [
        "**Summary.** The model gives this account an estimated churn probability of "
        f"{probability:.1%}, placing it in the {tier} band. At the current threshold of "
        f"{threshold:.2f} it is {flagged}.{comparison}",
        "",
        "**Attributes associated with a higher score.**",
    ]
    parts += (
        [f"- {name} = {value} ({contribution:+.3f})" for name, value, contribution in raising]
        or ["- None contributed materially."]
    )
    parts += ["", "**Attributes associated with a lower score.**"]
    parts += (
        [f"- {name} = {value} ({contribution:+.3f})" for name, value, contribution in lowering]
        or ["- None contributed materially."]
    )
    parts += [
        "",
        "**Suggested review questions.**",
        "- Does the account history show recent service issues or complaints?",
        "- Has the customer been contacted recently, and what was the outcome?",
        "- Is the current product mix still appropriate for how the customer uses the service?",
        "",
        "**Caveats.** These are statistical associations learned from a fictional training "
        "sample, not causes and not a prediction about this person. The account requires "
        "human review before any customer is contacted.",
    ]
    return "\n".join(parts)


def _call_provider(facts: str, model: str) -> str:
    """Call the Hugging Face Inference Providers chat endpoint."""
    from huggingface_hub import InferenceClient  # noqa: PLC0415

    client = InferenceClient(api_key=os.environ[TOKEN_ENV_VAR], timeout=REQUEST_TIMEOUT_SECONDS)
    completion = client.chat_completion(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": facts},
        ],
        max_tokens=MAX_NEW_TOKENS,
        temperature=0.2,
    )
    return completion.choices[0].message.content or ""


def generate_brief(
    probability: float,
    tier: str,
    threshold: float,
    contributions: list[tuple[str, str, float]],
    overall_rate: float | None = None,
    model: str = DEFAULT_MODEL,
) -> Brief:
    """Produce a brief, generated if every guardrail passes and template otherwise.

    Failure is never surfaced as an error. The template is returned instead, so
    the specialist always receives a usable brief.
    """
    template = deterministic_brief(probability, tier, threshold, contributions, overall_rate)

    if not is_enabled():
        return Brief(text=template, generated=False,
                     fallback_reason="Generation is disabled for this deployment.")

    facts = build_facts(probability, tier, threshold, contributions, overall_rate)

    # One retry: a single sample can trip a guardrail by chance, but a model that
    # fails twice is not going to be coaxed into compliance by trying again.
    for attempt in (1, 2):
        try:
            candidate = _call_provider(facts, model)
        except Exception:  # noqa: BLE001
            logger.exception("Rationale provider call failed on attempt %d", attempt)
            return Brief(text=template, generated=False,
                         fallback_reason="The generation service was unavailable.")

        structural = validate_structure(candidate)
        prohibited = check_prohibited(candidate)
        if not structural and not prohibited:
            return Brief(text=candidate, generated=True)

        logger.warning(
            "Generated brief rejected on attempt %d — structural=%s prohibited=%s",
            attempt, structural, prohibited,
        )

    return Brief(
        text=template,
        generated=False,
        fallback_reason="The generated text did not meet the safety checks.",
    )


def contributions_from_explanation(explanation: Any, n: int = 5) -> list[tuple[str, str, float]]:
    """Adapt an :class:`explain.Explanation` into the fact-block tuple form."""
    if explanation is None or not getattr(explanation, "supported", False):
        return []
    increases, decreases = explanation.top(n)
    return [
        (c.display_name, c.value, c.contribution) for c in [*increases, *decreases]
    ]
