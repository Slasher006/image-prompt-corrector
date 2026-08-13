"""Scoped count, exclusion, camera, spatial, and visual-phase facts."""

from __future__ import annotations

import re
from typing import Iterable

from entity_resolution import canonical_entity_name, canonical_head
from prompt_contract import ContractFact, SourceRef


NUMBER_VALUES = {
    "zero": 0, "no": 0, "one": 1, "single": 1, "two": 2, "three": 3,
    "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10,
}
NUMBER_TOKEN = r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|\d+)"

TRAILING_PREDICATES = {
    "appears", "are", "gleaming", "glowing", "hangs", "is", "lies", "remain",
    "remains", "rests", "shines", "sits", "stand", "stands", "wait", "waits",
}

CAMERA_PATTERNS = {
    "point-of-view": r"\b(?:point[- ]of[- ]view|first[- ]person|pov)\b",
    "over-the-shoulder": r"\b(?:over[- ]the[- ]shoulder|from behind (?:her|his|their|the) shoulder)\b",
    "front view": r"\b(?:front[- ]facing|front view|viewed from the front)\b",
    "rear view": r"\b(?:rear view|back view|viewed from behind)\b",
    "top-down": r"\b(?:top[- ]down|bird['’]?s[- ]eye|directly overhead)\b",
    "low-angle": r"\b(?:low[- ]angle|worm['’]?s[- ]eye|ground[- ]level looking up)\b",
}
CAMERA_CONFLICTS = {
    frozenset(("point-of-view", "over-the-shoulder")),
    frozenset(("point-of-view", "front view")),
    frozenset(("point-of-view", "rear view")),
    frozenset(("front view", "rear view")),
    frozenset(("top-down", "low-angle")),
}

RELATION_CANONICAL = {
    "beneath": "below",
    "under": "below",
    "below": "below",
    "over": "above",
    "above": "above",
    "left of": "left of",
    "to the left of": "left of",
    "right of": "right of",
    "to the right of": "right of",
    "behind": "behind",
    "in front of": "in front of",
    "inside": "inside",
    "within": "inside",
    "outside": "outside",
    "on the left": "left",
    "on the right": "right",
    "image-left": "left",
    "image-right": "right",
    "foreground": "foreground",
    "background": "background",
}

SPATIAL_OPPOSITES = {
    frozenset(("left", "right")),
    frozenset(("left of", "right of")),
    frozenset(("above", "below")),
    frozenset(("behind", "in front of")),
    frozenset(("inside", "outside")),
    frozenset(("foreground", "background")),
}


def _number_value(token: str) -> int:
    lowered = token.casefold()
    return NUMBER_VALUES.get(lowered, int(lowered) if lowered.isdigit() else 0)


def _source(field_name: str, text: str, span: tuple[int, int]) -> SourceRef:
    return SourceRef.create(field_name, text, span)


def extract_count_facts(text: str, field_name: str = "Draft") -> list[ContractFact]:
    """Parse exact counts, including coordinated count clauses."""

    facts: list[ContractFact] = []
    source_text = str(text or "")
    segment_pattern = re.compile(
        rf"(?ix)\b(?P<exact>exactly|only)\s+"
        rf"(?P<count>{NUMBER_TOKEN})\s+"
        r"(?P<entity>[a-z][a-z-]*(?:\s+[a-z][a-z-]*){0,3}?)"
        rf"(?=(?:\s*(?:,|and)\s*(?:(?:exactly|only)\s+)?{NUMBER_TOKEN}\b)|"
        r"\s+(?:appears?|are|gleam(?:s|ing)?|glow(?:s|ing)?|hangs?|is|lies?|remain(?:s)?|rests?|shines?|sits?|stands?|waits?)\b|[.;,:]|$)"
    )
    for match in segment_pattern.finditer(source_text):
        words = match.group("entity").split()
        while words and words[-1].casefold() in TRAILING_PREDICATES:
            words.pop()
        entity = canonical_entity_name(" ".join(words))
        if not entity:
            continue
        facts.append(
            ContractFact(
                code="count.required",
                entity_id=entity,
                value=_number_value(match.group("count")),
                polarity="required",
                scope_id="scene:primary",
                source=_source(field_name, source_text, match.span()),
                confidence=0.98,
                dimension="count",
            )
        )
    # Coordinated continuations may omit a repeated "exactly".
    continuation = re.compile(
        rf"(?ix)(?:,|\band\b)\s*(?P<count>{NUMBER_TOKEN})\s+"
        r"(?P<entity>[a-z][a-z-]*(?:\s+[a-z][a-z-]*){0,3}?)"
        r"(?=\s+(?:appears?|are|gleam(?:s|ing)?|glow(?:s|ing)?|hangs?|is|lies?|remain(?:s)?|rests?|shines?|sits?|stands?|waits?)\b|[.;,:]|$)"
    )
    exact_starts = [match.start() for match in re.finditer(r"(?i)\b(?:exactly|only)\b", source_text)]
    for match in continuation.finditer(source_text):
        nearest = max((start for start in exact_starts if start < match.start()), default=-10_000)
        boundary = max(source_text.rfind(".", nearest, match.start()), source_text.rfind(";", nearest, match.start()))
        if nearest < boundary:
            continue
        entity = canonical_entity_name(match.group("entity"))
        if not entity:
            continue
        key = (entity, _number_value(match.group("count")))
        if any((fact.entity_id, fact.value) == key for fact in facts):
            continue
        facts.append(
            ContractFact(
                code="count.required",
                entity_id=entity,
                value=key[1],
                polarity="required",
                scope_id="scene:primary",
                source=_source(field_name, source_text, match.span()),
                confidence=0.96,
                dimension="count",
            )
        )
    return sorted(facts, key=lambda fact: fact.source.span)


def extract_observed_count_facts(
    text: str,
    field_name: str = "Candidate",
) -> list[ContractFact]:
    """Parse ordinary visible count phrases for candidate comparison."""

    source_text = str(text or "")
    facts = extract_count_facts(source_text, field_name)
    pattern = re.compile(
        rf"(?ix)\b(?P<count>{NUMBER_TOKEN})\s+"
        r"(?P<entity>(?:[a-z][a-z-]*\s+){0,3}?[a-z][a-z-]*)"
        r"(?=\s+(?:appears?|are|gleam(?:s|ing)?|glow(?:s|ing)?|hangs?|is|lies?|remain(?:s)?|rests?|shines?|sits?|stands?|waits?)\b|[.;,:]|$)"
    )
    for match in pattern.finditer(source_text):
        entity = canonical_entity_name(match.group("entity"))
        if not entity:
            continue
        value = _number_value(match.group("count"))
        if any(fact.entity_id == entity and fact.value == value for fact in facts):
            continue
        facts.append(
            ContractFact(
                code="count.observed",
                entity_id=entity,
                value=value,
                polarity="observed",
                scope_id="scene:primary",
                source=_source(field_name, source_text, match.span()),
                confidence=0.91,
                dimension="count",
            )
        )
    return sorted(facts, key=lambda fact: fact.source.span)


def _split_list(body: str) -> list[str]:
    body = re.split(r"(?i)\b(?:but|while|yet|although|except)\b", body, maxsplit=1)[0]
    return [
        re.sub(r"(?i)^(?:and|or)\s+", "", value.strip(" ,.;:")).strip()
        for value in re.split(r"\s*,\s*|\s+and\s+|\s+or\s+", body)
        if value.strip(" ,.;:")
    ]


def extract_exclusion_facts(text: str, field_name: str = "Draft") -> list[ContractFact]:
    source_text = str(text or "")
    pattern = re.compile(
        r"(?ix)(?:"
        r"(?:^|(?<=[.;:\n]))\s*no\s+|\bwith\s+no\s+|\bwithout\s+|"
        r"\b(?:avoid|exclude)\s+(?:(?:add|include|show|depict|use|introduce)\s+)?|"
        r"\b(?:do\s+not|don't|never|remove)\s+(?:(?:add|include|show|depict|use|introduce|all)\s+)?"
        r")(?P<body>[^.!?;\n]+)"
    )
    facts: list[ContractFact] = []
    for match in pattern.finditer(source_text):
        for item in _split_list(match.group("body")):
            # Remove determiner/count wording while keeping descriptors.
            item = re.sub(r"(?i)^(?:any|all|the|a|an)\s+", "", item).strip()
            entity = canonical_entity_name(item)
            if not entity:
                continue
            facts.append(
                ContractFact(
                    code="exclusion.required",
                    entity_id=entity,
                    value="absent",
                    polarity="excluded",
                    scope_id="scene:primary",
                    source=_source(field_name, source_text, match.span()),
                    confidence=0.97,
                    dimension="exclusion",
                )
            )
    deduped: dict[str, ContractFact] = {}
    for fact in facts:
        deduped.setdefault(str(fact.entity_id), fact)
    return list(deduped.values())


def _camera_scope(text: str, start: int) -> str:
    local = text[max(0, start - 55) : start].casefold()
    markers = {
        "camera:reflection": [match.start() for match in re.finditer(r"\b(?:mirror|mirrored|reflection|reflected)\b", local)],
        "camera:inset": [match.start() for match in re.finditer(r"\b(?:inset|panel|monitor|screen|picture-in-picture)\b", local)],
        "camera:reference": [match.start() for match in re.finditer(r"\b(?:reference image|source image)\b", local)],
    }
    nearest = [
        (positions[-1], scope)
        for scope, positions in markers.items()
        if positions
    ]
    if nearest:
        return max(nearest)[1]
    return "camera:primary"


def extract_camera_facts(text: str, field_name: str = "Draft") -> list[ContractFact]:
    source_text = str(text or "")
    facts: list[ContractFact] = []
    negative_ranges = [
        match.span()
        for match in re.finditer(r"(?i)\b(?:no|not|avoid|without|replace|remove)\b[^.!?;]*", source_text)
    ]
    lighting_ranges = [
        match.span()
        for match in re.finditer(r"(?i)\b(?:light|lighting|illumination|shadow)\b[^.!?;]*", source_text)
    ]
    for label, raw_pattern in CAMERA_PATTERNS.items():
        for match in re.finditer(raw_pattern, source_text, flags=re.IGNORECASE):
            if any(left <= match.start() < right for left, right in negative_ranges):
                continue
            local = source_text[max(0, match.start() - 24) : match.end() + 36].casefold()
            if label == "top-down" and re.search(r"\b(?:light|lighting|illumination|shadow)\b", local):
                continue
            if label == "rear view" and re.search(
                r"viewed from behind\s+(?:a|an|the)\s+(?:curtain|door|window|screen|veil|occluder|fence)",
                local,
            ):
                continue
            facts.append(
                ContractFact(
                    code="camera.required",
                    entity_id="camera",
                    value=label,
                    polarity="required",
                    scope_id=_camera_scope(source_text, match.start()),
                    source=_source(field_name, source_text, match.span()),
                    confidence=0.98,
                    dimension="camera",
                )
            )
    return facts


def _clean_spatial_entity(value: str) -> str:
    value = re.sub(r"(?i)\b(?:stands?|sits?|lies?|rests?|is|are|remains?|appears?)\b.*$", "", value)
    value = re.sub(r"(?i)^(?:and|while|with)\s+", "", value)
    value = re.sub(
        rf"(?i)^(?:(?:exactly|only)\s+)?{NUMBER_TOKEN}\s+",
        "",
        value,
    )
    return canonical_entity_name(value)


def extract_spatial_facts(text: str, field_name: str = "Draft") -> list[ContractFact]:
    source_text = str(text or "")
    facts: list[ContractFact] = []
    # subject relation reference (normal order)
    relation_pattern = "|".join(sorted((re.escape(key) for key in RELATION_CANONICAL), key=len, reverse=True))
    normal = re.compile(
        rf"(?ix)\b(?P<subject>(?:(?:the|a|an)\s+)?(?:[a-z][a-z-]*\s+){{0,3}}[a-z][a-z-]*)\s+"
        rf"(?:(?:stands?|sits?|lies?|rests?|is|are|remains?)\s+)?"
        rf"(?P<relation>{relation_pattern})"
        rf"(?:\s+(?P<reference>(?:(?:the|a|an)\s+)?(?:[a-z][a-z-]*\s+){{0,3}}[a-z][a-z-]*))?"
        r"(?=[,.;]|$)"
    )
    for match in normal.finditer(source_text):
        subject = _clean_spatial_entity(match.group("subject"))
        reference = _clean_spatial_entity(match.group("reference") or "")
        relation = RELATION_CANONICAL[match.group("relation").casefold()]
        if relation in {"left", "right", "foreground", "background"}:
            reference = ""
        if not subject:
            continue
        facts.append(
            ContractFact(
                code="position.required",
                entity_id=subject,
                value=relation,
                polarity="required",
                scope_id="scene:primary",
                source=_source(field_name, source_text, match.span()),
                confidence=0.91,
                dimension="position",
                context=(reference,) if reference else (),
            )
        )
    # Inverted: Beneath the tree stands the woman.
    inverted = re.compile(
        rf"(?ix)\b(?P<relation>{relation_pattern})\s+"
        r"(?P<reference>(?:the|a|an)\s+(?:[a-z][a-z-]*\s+){0,2}[a-z][a-z-]*)\s+"
        r"(?:stands?|sits?|lies?|rests?|is|are)\s+"
        r"(?P<subject>(?:the|a|an)\s+(?:[a-z][a-z-]*\s+){0,2}[a-z][a-z-]*)"
    )
    for match in inverted.finditer(source_text):
        fact = ContractFact(
            code="position.required",
            entity_id=_clean_spatial_entity(match.group("subject")),
            value=RELATION_CANONICAL[match.group("relation").casefold()],
            polarity="required",
            scope_id="scene:primary",
            source=_source(field_name, source_text, match.span()),
            confidence=0.97,
            dimension="position",
            context=(_clean_spatial_entity(match.group("reference")),),
        )
        if fact.entity_id:
            facts.append(fact)
    unique: dict[tuple[object, ...], ContractFact] = {}
    for fact in facts:
        unique.setdefault((fact.entity_id, fact.value, fact.context, fact.scope_id), fact)
    return list(unique.values())


def extract_phase_facts(text: str, field_name: str = "Draft") -> list[ContractFact]:
    source_text = str(text or "")
    phases = {
        "setup": r"\b(?:before|preparing|about to|anticipation)\b",
        "active": r"\b(?:during|actively|in progress|having sex|penetrat(?:e|es|ing))\b",
        "climax": r"\b(?:orgasm|climax|ejaculat(?:e|es|ed|ing|ion))\b",
        "aftermath": r"\b(?:afterward|aftermath|aftercare|post[- ]climax)\b",
    }
    facts: list[ContractFact] = []
    for phase, pattern in phases.items():
        for match in re.finditer(pattern, source_text, flags=re.IGNORECASE):
            local = source_text[max(0, match.start() - 25) : match.end() + 25].casefold()
            if phase == "climax" and re.search(r"\b(?:mountain|summit|ridge|career|story)\b", local):
                continue
            facts.append(
                ContractFact(
                    code="phase.required",
                    entity_id="scene",
                    value=phase,
                    polarity="required",
                    scope_id="scene:primary",
                    source=_source(field_name, source_text, match.span()),
                    confidence=0.9,
                    dimension="phase",
                )
            )
    return facts


def compile_dimension_facts(text: str, field_name: str = "Draft") -> list[ContractFact]:
    return (
        extract_count_facts(text, field_name)
        + extract_exclusion_facts(text, field_name)
        + extract_camera_facts(text, field_name)
        + extract_spatial_facts(text, field_name)
        + extract_phase_facts(text, field_name)
    )


def camera_values_conflict(left: ContractFact, right: ContractFact) -> bool:
    return (
        left.scope_id == right.scope_id == "camera:primary"
        and frozenset((str(left.value), str(right.value))) in CAMERA_CONFLICTS
    )


def spatial_values_conflict(left: ContractFact, right: ContractFact) -> bool:
    return (
        left.entity_id == right.entity_id
        and left.context == right.context
        and left.scope_id == right.scope_id
        and frozenset((str(left.value), str(right.value))) in SPATIAL_OPPOSITES
    )


def entity_fact_matches(left: ContractFact, right: ContractFact) -> bool:
    if left.entity_id == right.entity_id:
        return True
    return bool(
        left.entity_id
        and right.entity_id
        and canonical_head(str(left.entity_id)) == canonical_head(str(right.entity_id))
    )
