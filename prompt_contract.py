"""Typed prompt contracts, provenance, authority, and structured issues.

This module is deliberately dependency-light.  Parsers live in the focused
entity/scene modules; the records here are the common language shared by the
GUI, preflight, final validation, and recovery paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, Iterator


SEVERITY_BLOCKER = "blocker"
SEVERITY_HARD = "hard"
SEVERITY_ADVISORY = "advisory"


FIELD_AUTHORITY = {
    "Draft": 10,
    "Story": 10,
    "Concepts": 20,
    "Weighted terms": 20,
    "Goal": 25,
    "Focus": 25,
    "Instructions": 30,
    "Feedback": 40,
    "Visual direction": 50,
    "Camera": 60,
    "Format": 90,
    "Mode": 90,
    "Policy": 100,
    "Candidate": 0,
}


@dataclass(frozen=True)
class SourceRef:
    field: str
    span: tuple[int, int]
    text: str
    authority: int
    current_pass_only: bool = False

    @classmethod
    def create(
        cls,
        field_name: str,
        text: str,
        span: tuple[int, int] | None = None,
        *,
        current_pass_only: bool | None = None,
    ) -> "SourceRef":
        actual_span = span or (0, len(text))
        return cls(
            field=field_name,
            span=actual_span,
            text=text[actual_span[0] : actual_span[1]],
            authority=FIELD_AUTHORITY.get(field_name, 10),
            current_pass_only=(
                field_name == "Feedback"
                if current_pass_only is None
                else current_pass_only
            ),
        )


@dataclass(frozen=True)
class EntityMention:
    mention_id: str
    entity_id: str | None
    kind: str
    canonical_name: str
    number: str
    attributes: tuple[str, ...]
    source: SourceRef
    confidence: float


@dataclass(frozen=True)
class ContractFact:
    code: str
    entity_id: str | None
    value: object
    polarity: str
    scope_id: str
    source: SourceRef
    confidence: float
    fact_id: str = ""
    dimension: str = ""
    context: tuple[str, ...] = ()

    def with_id(self, fact_id: str) -> "ContractFact":
        return replace(self, fact_id=fact_id)


@dataclass(frozen=True)
class RelationFact:
    predicate: str
    actor_id: str | None
    receiver_id: str | None
    object_id: str | None
    body_target_id: str | None
    reaction_owner_id: str | None
    cause_relation_id: str | None
    phase: str | None
    scope_id: str
    source: SourceRef
    confidence: float
    relation_id: str = ""
    polarity: str = "required"

    def with_id(self, relation_id: str) -> "RelationFact":
        return replace(self, relation_id=relation_id)


@dataclass(frozen=True)
class ComplianceIssue:
    code: str
    severity: str
    origin: str
    source_fact_ids: tuple[str, ...]
    candidate_fact_ids: tuple[str, ...]
    message: str
    field: str = ""
    span: tuple[int, int] | None = None
    dimension: str = ""

    def __str__(self) -> str:
        return self.message


class IssueText(str):
    """String-compatible issue carrying its stable structured record."""

    issue: ComplianceIssue

    def __new__(cls, issue: ComplianceIssue) -> "IssueText":
        value = super().__new__(cls, issue.message)
        value.issue = issue
        return value


@dataclass
class PromptContract:
    fields: dict[str, str]
    mentions: list[EntityMention] = field(default_factory=list)
    facts: list[ContractFact] = field(default_factory=list)
    relations: list[RelationFact] = field(default_factory=list)
    issues: list[ComplianceIssue] = field(default_factory=list)

    def active_facts(self, dimension: str | None = None) -> list[ContractFact]:
        values = self.facts
        if dimension is not None:
            values = [fact for fact in values if fact.dimension == dimension]
        return [fact for fact in values if fact.polarity != "superseded"]

    def trace(self) -> dict[str, object]:
        """Return non-private structured diagnostics for tests/debug views."""

        return {
            "fields": tuple(self.fields),
            "mentions": [
                {
                    "id": item.mention_id,
                    "entity": item.entity_id,
                    "kind": item.kind,
                    "name": item.canonical_name,
                    "number": item.number,
                    "field": item.source.field,
                    "confidence": item.confidence,
                }
                for item in self.mentions
            ],
            "facts": [
                {
                    "id": item.fact_id,
                    "code": item.code,
                    "entity": item.entity_id,
                    "value": item.value,
                    "dimension": item.dimension,
                    "polarity": item.polarity,
                    "field": item.source.field,
                    "confidence": item.confidence,
                }
                for item in self.facts
            ],
            "relations": [
                {
                    "id": item.relation_id,
                    "predicate": item.predicate,
                    "actor": item.actor_id,
                    "receiver": item.receiver_id,
                    "object": item.object_id,
                    "body_target": item.body_target_id,
                    "field": item.source.field,
                    "confidence": item.confidence,
                }
                for item in self.relations
            ],
            "issues": [
                {
                    "code": item.code,
                    "severity": item.severity,
                    "origin": item.origin,
                    "field": item.field,
                    "dimension": item.dimension,
                    "message": item.message,
                }
                for item in self.issues
            ],
        }


def assign_fact_ids(facts: Iterable[ContractFact]) -> list[ContractFact]:
    return [
        fact if fact.fact_id else fact.with_id(f"fact_{index}")
        for index, fact in enumerate(facts, 1)
    ]


def assign_relation_ids(relations: Iterable[RelationFact]) -> list[RelationFact]:
    return [
        relation
        if relation.relation_id
        else relation.with_id(f"relation_{index}")
        for index, relation in enumerate(relations, 1)
    ]


def issue_text(issue: ComplianceIssue) -> IssueText:
    return IssueText(issue)


def structured_issue(value: str | ComplianceIssue) -> ComplianceIssue | None:
    if isinstance(value, ComplianceIssue):
        return value
    return getattr(value, "issue", None)


def issue_is_hard(value: str | ComplianceIssue) -> bool | None:
    issue = structured_issue(value)
    if issue is None:
        return None
    return issue.severity in {SEVERITY_BLOCKER, SEVERITY_HARD}


def iter_current_pass_sources(contract: PromptContract) -> Iterator[SourceRef]:
    for fact in contract.facts:
        if fact.source.current_pass_only:
            yield fact.source
    for relation in contract.relations:
        if relation.source.current_pass_only:
            yield relation.source
