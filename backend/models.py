from typing import Literal

from pydantic import AliasChoices, BaseModel, Field


class BtwRouteDecision(BaseModel):
    needs_web_search: bool = Field(
        description="Strict JSON boolean. Set to true if a web search is needed, false otherwise. Do NOT use string quotes around it."
    )


class RouterDecision(BaseModel):
    route: Literal["retrieve", "verify_claim", "direct_answer"]


class RelevancyDecision(BaseModel):
    is_relevant: bool = Field(
        description="Strict JSON boolean. Set to true if relevant, false otherwise. Do NOT output as a string."
    )
    reason: str


class SupersedingPaper(BaseModel):
    title: str
    url: str
    summary: str = Field(
        description=(
            "One-sentence explanation of how the paper challenges, updates, or "
            "supersedes the claim. Use the JSON key 'summary'."
        ),
        validation_alias=AliasChoices("summary", "explanation"),
    )


class ClaimVerificationResult(BaseModel):
    is_superseded: bool = Field(
        description="Strict JSON boolean. Set to true if superseded, false otherwise. Do NOT output as a string."
    )
    verdict_summary: str
    superseding_papers: list[SupersedingPaper]
