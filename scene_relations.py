"""Scoped actor/action/receiver/contact/ownership/reaction relations."""

from __future__ import annotations

import re
from typing import Iterable

from entity_resolution import (
    BODY_PARTS,
    canonical_entity_name,
    extract_entity_mentions,
    resolve_references,
)
from prompt_contract import EntityMention, RelationFact, SourceRef


PREDICATE_FORMS = {
    "embrace": ("embrace", "embraces", "embraced", "embracing"),
    "greet": ("greet", "greets", "greeted", "greeting"),
    "hold": ("hold", "holds", "held", "holding"),
    "kiss": ("kiss", "kisses", "kissed", "kissing"),
    "stroke": ("stroke", "strokes", "stroked", "stroking"),
    "stimulate": ("stimulate", "stimulates", "stimulated", "stimulating"),
    "touch": ("touch", "touches", "touched", "touching"),
    "examine": ("examine", "examines", "examined", "examining"),
    "cuff": ("cuff", "cuffs", "cuffed", "cuffing"),
    "bind": ("bind", "binds", "bound", "binding"),
    "penetrate": ("penetrate", "penetrates", "penetrated", "penetrating"),
    "oral_stimulation": (
        "performs oral stimulation on", "gives oral sex to", "fellates", "sucks",
    ),
    "intercourse": (
        "has sex with", "have sex with", "having sex with", "fucks", "fucking",
    ),
}
FORM_TO_PREDICATE = {
    form: predicate
    for predicate, forms in PREDICATE_FORMS.items()
    for form in forms
}
VERB_PATTERN = "|".join(
    sorted((re.escape(form) for form in FORM_TO_PREDICATE), key=len, reverse=True)
)
ROLE_PHRASE = (
    r"(?:(?:the|an?|first|second|third)\s+)?"
    r"(?:adult\s+)?(?:[a-z][a-z-]*\s+){0,2}"
    r"(?:woman|man|person|partner|doctor|patient|nurse|teacher|guard|queen|king|"
    r"wife|husband|bride|groom|performer|subject|she|he|they|her|him|them)"
)


def _entity_for_phrase(
    phrase: str,
    position: int,
    mentions: Iterable[EntityMention],
) -> str | None:
    canonical = canonical_entity_name(phrase)
    token = canonical.split()[-1] if canonical else ""
    pronoun = token in {"she", "he", "they", "her", "him", "them"}
    possible = [
        item
        for item in mentions
        if item.entity_id
        and item.source.span[0] <= position
        and (
            (pronoun and item.kind == "reference" and item.canonical_name == token)
            or (not pronoun and (
                item.canonical_name == canonical
                or item.canonical_name.endswith(" " + token)
                or canonical.endswith(" " + item.canonical_name)
            ))
        )
    ]
    if possible:
        return possible[-1].entity_id
    return canonical or None


def _body_target(text: str) -> str | None:
    match = re.search(
        r"(?i)\b(?:his|her|their|the\s+[a-z-]+'s)?\s*"
        + "(" + "|".join(sorted(BODY_PARTS, key=len, reverse=True)) + r")(?:s)?\b",
        text,
    )
    return canonical_entity_name(match.group(1)) if match else None


def _relation_scope(text: str, start: int) -> str:
    sentence = len(re.findall(r"[.!?]", text[:start])) + 1
    clause = len(re.findall(r"[,;]", text[text.rfind(".", 0, start) + 1 : start])) + 1
    return f"sentence:{sentence}:clause:{clause}"


def extract_relation_facts(
    text: str,
    field_name: str = "Draft",
    mentions: list[EntityMention] | None = None,
) -> list[RelationFact]:
    source_text = str(text or "")
    resolved_mentions = resolve_references(
        mentions if mentions is not None else extract_entity_mentions(source_text, field_name)
    )
    relations: list[RelationFact] = []

    active = re.compile(
        rf"(?ix)\b(?P<actor>{ROLE_PHRASE})\s+"
        rf"(?P<verb>{VERB_PATTERN})\s+"
        rf"(?P<receiver>{ROLE_PHRASE}|(?:his|her|their|the)\s+[a-z][a-z-]*(?:\s+[a-z][a-z-]*)?)"
    )
    passive = re.compile(
        rf"(?ix)\b(?P<receiver>{ROLE_PHRASE})\s+"
        rf"(?:is|was|are|were|gets?|got)\s+"
        rf"(?P<verb>{VERB_PATTERN})\s+by\s+"
        rf"(?P<actor>{ROLE_PHRASE})"
    )

    occupied: list[tuple[int, int]] = []
    for pattern, is_passive in ((passive, True), (active, False)):
        for match in pattern.finditer(source_text):
            if any(left <= match.start() < right for left, right in occupied):
                continue
            actor_text = match.group("actor")
            receiver_text = match.group("receiver")
            verb = re.sub(r"\s+", " ", match.group("verb").casefold())
            predicate = FORM_TO_PREDICATE.get(verb, verb)
            receiver_entity = _entity_for_phrase(receiver_text, match.start("receiver"), resolved_mentions)
            body_target = _body_target(receiver_text)
            relations.append(
                RelationFact(
                    predicate=predicate,
                    actor_id=_entity_for_phrase(actor_text, match.start("actor"), resolved_mentions),
                    receiver_id=receiver_entity,
                    object_id=None,
                    body_target_id=(
                        f"{receiver_entity}:{body_target}"
                        if receiver_entity and body_target
                        else body_target
                    ),
                    reaction_owner_id=None,
                    cause_relation_id=None,
                    phase="active" if predicate in {"intercourse", "penetrate", "oral_stimulation", "stimulate"} else None,
                    scope_id=_relation_scope(source_text, match.start()),
                    source=SourceRef.create(field_name, source_text, match.span()),
                    confidence=0.96 if is_passive else 0.94,
                )
            )
            occupied.append(match.span())

    # Possessive ownership is a relation in its own right.  It is compared only
    # when both source and candidate state ownership explicitly.
    ownership = re.compile(
        r"(?i)\b(?P<owner>his|her|their|the\s+(?:(?:first|second|third|fourth|adult)\s+)?[a-z-]+'s)\s+"
        r"(?P<body>" + "|".join(sorted(BODY_PARTS, key=len, reverse=True)) + r")(?:s)?\b"
    )
    for match in ownership.finditer(source_text):
        owner_phrase = re.sub(r"(?i)'s$", "", match.group("owner"))
        owner = _entity_for_phrase(owner_phrase, match.start("owner"), resolved_mentions)
        if owner is None:
            continue
        body = canonical_entity_name(match.group("body"))
        relations.append(
            RelationFact(
                predicate="owns_body_part",
                actor_id=owner,
                receiver_id=None,
                object_id=None,
                body_target_id=f"{owner}:{body}",
                reaction_owner_id=None,
                cause_relation_id=None,
                phase=None,
                scope_id=_relation_scope(source_text, match.start()),
                source=SourceRef.create(field_name, source_text, match.span()),
                confidence=0.92,
            )
        )

    relations.extend(_extract_reaction_relations(source_text, field_name, resolved_mentions, relations))
    unique: dict[tuple[object, ...], RelationFact] = {}
    for relation in relations:
        key = (
            relation.predicate, relation.actor_id, relation.receiver_id,
            relation.object_id, relation.body_target_id, relation.reaction_owner_id,
            relation.phase,
        )
        unique.setdefault(key, relation)
    return list(unique.values())


def _extract_reaction_relations(
    text: str,
    field_name: str,
    mentions: list[EntityMention],
    causes: list[RelationFact],
) -> list[RelationFact]:
    reaction_pattern = re.compile(
        rf"(?ix)\b(?P<owner>{ROLE_PHRASE})\s+"
        r"(?P<reaction>trembles?|shudders?|gasps?|moans?|flinches?|smiles?|cries?|flushes?)\b"
    )
    results: list[RelationFact] = []
    for match in reaction_pattern.finditer(text):
        owner = _entity_for_phrase(match.group("owner"), match.start("owner"), mentions)
        preceding = [
            relation for relation in causes
            if relation.source.span[1] <= match.start()
            and len(re.findall(r"[.!?]", text[relation.source.span[1] : match.start()])) <= 1
        ]
        explicit_cause = bool(
            re.search(
                r"(?i)\b(?:because|from|after|in response to|caused by|which makes?)\b",
                text[max(0, match.start() - 90) : match.end() + 40],
            )
        )
        cause = preceding[-1] if preceding and explicit_cause else None
        results.append(
            RelationFact(
                predicate="reaction:" + match.group("reaction").casefold().rstrip("s"),
                actor_id=None,
                receiver_id=None,
                object_id=None,
                body_target_id=None,
                reaction_owner_id=owner,
                cause_relation_id=cause.relation_id or f"source:{cause.source.span[0]}" if cause else None,
                phase=None,
                scope_id=_relation_scope(text, match.start()),
                source=SourceRef.create(field_name, text, match.span()),
                confidence=0.93 if owner and cause else 0.68 if owner else 0.4,
            )
        )
    return results


def relation_signature(relation: RelationFact) -> tuple[object, ...]:
    return (
        relation.predicate,
        relation.actor_id,
        relation.receiver_id,
        relation.object_id,
        relation.body_target_id,
        relation.reaction_owner_id,
        relation.phase,
    )


def relation_reversal(left: RelationFact, right: RelationFact) -> bool:
    def semantic(value: str | None) -> str:
        raw = str(value or "")
        return canonical_entity_name(raw.split(":", 1)[1] if ":" in raw else raw)

    return (
        left.predicate == right.predicate
        and left.actor_id is not None
        and left.receiver_id is not None
        and semantic(left.actor_id) == semantic(right.receiver_id)
        and semantic(left.receiver_id) == semantic(right.actor_id)
    )
