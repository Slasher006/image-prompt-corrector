"""Shared entity, group, object, body-part, and coreference resolution."""

from __future__ import annotations

from dataclasses import replace
import re
from typing import Iterable

from prompt_contract import EntityMention, SourceRef


IRREGULAR_CANONICAL = {
    "women": "woman",
    "men": "man",
    "people": "person",
    "persons": "person",
    "children": "child",
    "feet": "foot",
    "teeth": "tooth",
    "wives": "wife",
    "husbands": "husband",
    "mice": "mouse",
    "geese": "goose",
}

SINGULAR_S_HEADS = {"glans", "penis"}

SAFE_ENTITY_ALIASES = {
    "human": "person",
    "humans": "person",
    "female": "woman",
    "females": "woman",
    "male": "man",
    "males": "man",
    "lady": "woman",
    "ladies": "woman",
}

PERSON_HEADS = {
    "adult", "actor", "artist", "boy", "bride", "brother", "child",
    "couple", "crowd", "daughter", "doctor", "driver", "father", "female", "girl",
    "groom", "guard", "husband", "king", "lady", "man", "male", "mother",
    "nurse", "partner", "patient", "people", "person", "performer", "queen",
    "sister", "subject", "teacher", "wife", "woman",
}
PLURAL_PERSON_HEADS = {
    "adults", "actors", "artists", "boys", "brides", "brothers", "children",
    "couples", "crowds", "daughters", "doctors", "drivers", "fathers", "females", "girls",
    "grooms", "guards", "husbands", "kings", "ladies", "males", "men", "mothers",
    "nurses", "partners", "patients", "people", "performers", "queens", "sisters",
    "subjects", "teachers", "wives", "women",
}
FEMALE_HEADS = {
    "woman", "women", "female", "females", "lady", "ladies", "girl", "girls",
    "wife", "wives", "bride", "brides", "mother", "mothers", "queen", "queens",
    "sister", "sisters", "daughter", "daughters",
}
MALE_HEADS = {
    "man", "men", "male", "males", "boy", "boys", "husband", "husbands",
    "groom", "grooms", "father", "fathers", "king", "kings", "brother", "brothers",
    "son", "sons",
}
NONBINARY_PATTERN = re.compile(
    r"(?i)\b(?:an?\s+)?(?:non[- ]?binary|agender|genderqueer)\s+(?:adult\s+)?(?:person|subject|adult)\b"
)
BODY_PARTS = {
    "ankle", "arm", "back", "body", "breast", "chest", "elbow", "eye", "face",
    "finger", "foot", "frenulum", "genitals", "glans", "hand", "head", "hip",
    "knee", "leg", "lip", "mouth", "neck", "penis", "shaft", "shoulder", "thigh",
    "tip", "toe", "torso", "vagina", "vulva", "waist", "wrist",
}
NUMBER_WORDS = {
    "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10,
}
NON_ENTITY_HEADS = {
    "big", "small", "large", "thick", "thin", "hot", "wet", "dry", "green",
    "red", "blue", "black", "white", "warm", "cold", "raw", "clean", "open",
    "closed", "visible", "bright", "dark", "soft", "hard", "wrong", "right",
    "dominant", "only", "so",
}

NOUN_PHRASE_BOUNDARIES = {
    "and", "or", "with", "while", "who", "that", "in", "on", "at", "by",
    "from", "to", "over", "under", "beside", "inside", "into", "near", "of",
    "as", "so", "but", "because", "before", "after", "through",
    "enters", "carries", "places", "stands", "sits", "lies", "rests", "is",
    "are", "was", "were", "has", "have", "crosses", "reaches", "keeps",
    "remains", "moves", "glows", "shines", "reflects", "emits", "casts",
    "illuminates", "holds", "wears", "faces", "looks", "watches", "touches",
    "releases", "follows", "leads", "hangs", "floats", "runs", "walks",
    "turns", "opens", "closes", "reveals",
    "embraces", "greets", "holds", "held", "kisses", "kissed", "strokes",
    "stimulates", "touches", "examines", "examined", "cuffs", "cuffed",
    "binds", "bound", "penetrates", "fucks",
}


def canonical_entity_name(value: str) -> str:
    text = re.sub(r"[^a-z0-9 -]+", " ", str(value or "").casefold())
    words = [word for word in text.split() if word not in {"a", "an", "the", "adult"}]
    if not words:
        return ""
    head = words[-1]
    head = SAFE_ENTITY_ALIASES.get(head, IRREGULAR_CANONICAL.get(head, head))
    if head.endswith("ies") and len(head) > 4:
        head = head[:-3] + "y"
    elif (
        head.endswith("s")
        and len(head) > 3
        and not head.endswith("ss")
        and head not in SINGULAR_S_HEADS
    ):
        head = head[:-1]
    words[-1] = head
    return " ".join(words)


def canonical_head(value: str) -> str:
    name = canonical_entity_name(value)
    return name.split()[-1] if name else ""


def _number_for_phrase(phrase: str, head: str) -> str:
    lowered = phrase.casefold()
    if head.casefold() in PLURAL_PERSON_HEADS or head.casefold() in {"couple", "crowd"}:
        return "plural"
    if re.search(r"\b(?:two|three|four|five|six|seven|eight|nine|ten|several|many|multiple)\b", lowered):
        return "plural"
    return "singular"


def _attributes(phrase: str, head: str) -> tuple[str, ...]:
    values: list[str] = []
    lowered = phrase.casefold()
    if head in FEMALE_HEADS:
        values.append("female")
    elif head in MALE_HEADS:
        values.append("male")
    if re.search(r"\b(?:non[- ]?binary|agender|genderqueer)\b", lowered):
        values.append("nonbinary")
    ordinal = re.search(r"\b(first|second|third|fourth)\b", lowered)
    if ordinal:
        values.append(ordinal.group(1))
    return tuple(values)


def extract_entity_mentions(text: str, field_name: str = "Draft") -> list[EntityMention]:
    """Extract conservative explicit noun mentions and reference mentions.

    The graph intentionally prefers missed low-confidence nouns over invented
    identities.  Explicit articles, counts, person roles, possessives, and
    pronouns provide the high-confidence mention set used for hard validation.
    """

    source_text = str(text or "")
    candidates: list[tuple[int, int, str, str, str, tuple[str, ...], float]] = []
    occupied: list[tuple[int, int]] = []

    person_alternation = "|".join(
        sorted(PERSON_HEADS | PLURAL_PERSON_HEADS, key=len, reverse=True)
    )
    person_pattern = re.compile(
        rf"(?ix)\b(?P<phrase>"
        rf"(?:(?:the\s+)?(?:first|second|third|fourth)\s+)?"
        rf"(?:(?:a|an|one|single|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+)?"
        rf"(?:adult\s+)?(?:non[- ]?binary\s+)?"
        rf"(?:(?!(?:a|an|the|and|or|with|while|who|that|in|on|at|by|from|to|over|under|beside|"
        rf"embraces?|greets?|holds?|held|kisses?|kissed|strokes?|stimulates?|touches?|examines?|examined|"
        rf"cuffs?|cuffed|binds?|bound|penetrates?|fucks?|has|have|is|are|was|were)\b)[a-z][a-z-]*\s+){{0,2}}"
        rf"(?P<head>{person_alternation}))\b"
    )
    generic_role_pattern = re.compile(
        r"(?ix)\b(?P<phrase>"
        r"(?:(?:the\s+)?(?:first|second|third|fourth)\s+|(?:a|an|one)\s+)"
        r"(?:(?:adult|lone|solo|young|older?|mature)\s+)?"
        r"(?:(?!(?:and|or|with|while|who|that|in|on|at|by|from|to)\b)[a-z][a-z-]*\s+){0,2}"
        r"(?P<head>[a-z][a-z-]*(?:er|or|ist|ian|ant|ee)))\b"
    )
    noun_boundaries = "|".join(
        sorted((re.escape(word) for word in NOUN_PHRASE_BOUNDARIES), key=len, reverse=True)
    )
    noun_pattern = re.compile(
        rf"(?ix)\b(?P<phrase>"
        rf"(?:(?:the\s+)?(?:first|second|third|fourth)\s+|"
        rf"(?:a|an|the|one|single|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+)"
        rf"(?:(?!(?:{noun_boundaries})\b)[a-z][a-z-]*\s+){{0,3}}"
        rf"(?P<head>(?!(?:{noun_boundaries})\b)[a-z][a-z-]*))\b"
    )
    body_alternation = "|".join(sorted(BODY_PARTS, key=len, reverse=True))
    definite_body_pattern = re.compile(
        rf"(?ix)\b(?P<phrase>the\s+"
        rf"(?:(?!(?:and|or|with|while|who|that|in|on|at|by|from|to)\b)"
        rf"[a-z][a-z-]*\s+){{0,2}}"
        rf"(?P<head>{body_alternation}))\b"
    )
    person_matches = list(person_pattern.finditer(source_text))
    explicit_person_starts = {match.start() for match in person_matches}
    generic_matches = [
        match
        for match in generic_role_pattern.finditer(source_text)
        if match.start() not in explicit_person_starts
    ]
    generic_role_spans = {match.span() for match in generic_matches}
    for match in (
        person_matches
        + generic_matches
        + list(definite_body_pattern.finditer(source_text))
        + list(noun_pattern.finditer(source_text))
    ):
        phrase = re.sub(r"\s+", " ", match.group("phrase")).strip()
        head = match.group("head").casefold()
        match_start, match_end = match.span()
        is_person = (
            head in PERSON_HEADS
            or head in PLURAL_PERSON_HEADS
            or match.span() in generic_role_spans
        )
        is_body = canonical_head(head) in BODY_PARTS
        explicit_noun = not is_person
        if explicit_noun and head in NON_ENTITY_HEADS:
            words = phrase.split()
            while len(words) > 1 and head in NON_ENTITY_HEADS:
                removed = words.pop()
                match_end -= len(removed)
                while match_end > match_start and source_text[match_end - 1].isspace():
                    match_end -= 1
                head = words[-1].casefold()
            phrase = " ".join(words)
            is_body = canonical_head(head) in BODY_PARTS
            if head in NON_ENTITY_HEADS or head in {"a", "an", "the"}:
                continue
        possessive_body = is_body and bool(
            re.search(r"(?i)\b(?:his|her|their|its|the\s+\w+'s)\s+", phrase)
        )
        if not (is_person or explicit_noun or possessive_body):
            continue
        kind = "body_part" if is_body else "person" if is_person else "object"
        if kind == "object" and re.search(
            r"(?i)\b(?:first|last|final)\s+frame\b",
            phrase,
        ):
            # I2V keyframe bindings are transport protocol, not scene entities
            # that should compete with a cat, prop, or body part for it/its.
            continue
        number = _number_for_phrase(phrase, head)
        if kind == "person" and number == "plural":
            kind = "group"
        confidence = 0.98 if is_person else 0.92 if explicit_noun else 0.85
        candidates.append(
            (
                match_start, match_end, phrase, kind, number,
                _attributes(phrase, head), confidence,
            )
        )
        occupied.append((match_start, match_end))

    for match in NONBINARY_PATTERN.finditer(source_text):
        if not any(start <= match.start() and match.end() <= end for start, end in occupied):
            candidates.append(
                (match.start(), match.end(), match.group(0), "person", "singular", ("nonbinary",), 0.99)
            )

    candidates.sort(key=lambda item: (item[0], 0 if item[3] in {"person", "group"} else 1, -(item[1] - item[0])))
    mentions: list[EntityMention] = []
    entity_index = 0
    kept_ranges: list[tuple[int, int]] = []
    for start, end, phrase, kind, number, attributes, confidence in candidates:
        if any(start >= left and end <= right for left, right in kept_ranges):
            continue
        if kind == "object" and any(start < right and end > left for left, right in kept_ranges):
            continue
        entity_index += 1
        canonical = canonical_entity_name(phrase)
        entity_id = f"{kind}_{entity_index}:{canonical or kind}"
        mentions.append(
            EntityMention(
                mention_id=f"mention_{len(mentions) + 1}",
                entity_id=entity_id,
                kind=kind,
                canonical_name=canonical,
                number=number,
                attributes=attributes,
                source=SourceRef.create(field_name, source_text, (start, end)),
                confidence=confidence,
            )
        )
        kept_ranges.append((start, end))

    pronoun_pattern = re.compile(
        r"(?i)\b(it|its|this|that|they|them|their|theirs|he|him|his|she|her|hers)\b"
    )
    for match in pronoun_pattern.finditer(source_text):
        token = match.group(0).casefold()
        if token in {"this", "that"}:
            following = re.match(
                r"\s*([A-Za-z][A-Za-z-]*)",
                source_text[match.end() :],
            )
            if following and following.group(1).casefold() not in {
                "at", "beside", "by", "from", "in", "inside", "into", "near",
                "on", "through", "under", "with",
            }:
                continue
        mentions.append(
            EntityMention(
                mention_id=f"mention_{len(mentions) + 1}",
                entity_id=None,
                kind="reference",
                canonical_name=match.group(0).casefold(),
                number="unknown",
                attributes=(),
                source=SourceRef.create(field_name, source_text, match.span()),
                confidence=0.0,
            )
        )
    ordered = sorted(mentions, key=lambda item: item.source.span)
    linked: list[EntityMention] = []
    for item in ordered:
        if item.kind not in {"person", "group", "object"} or not item.entity_id:
            linked.append(item)
            continue
        definite = (
            item.source.text.casefold().startswith("the ")
            or source_text[
                max(0, item.source.span[0] - 12) : item.source.span[0]
            ].casefold().endswith("the ")
        )
        possessive = source_text[item.source.span[1] : item.source.span[1] + 2].startswith("'s")
        if definite or possessive:
            head = canonical_head(item.canonical_name)
            prior = [
                existing
                for existing in linked
                if existing.entity_id
                and existing.kind == item.kind
                and canonical_head(existing.canonical_name) == head
                and not any(value in existing.attributes for value in ("first", "second", "third", "fourth"))
            ]
            unique_ids = {existing.entity_id for existing in prior}
            if len(unique_ids) == 1:
                item = replace(item, entity_id=prior[-1].entity_id)
        linked.append(item)
    return linked


def _reference_candidates(
    reference: EntityMention,
    mentions: Iterable[EntityMention],
) -> list[EntityMention]:
    prior = [
        item
        for item in mentions
        if item.entity_id
        and item.source.field == reference.source.field
        and item.source.span[1] <= reference.source.span[0]
    ]
    token = reference.canonical_name
    if token in {"it", "its", "this", "that"}:
        # Image prompts regularly use an object-style reference for an animal,
        # prop, or explicitly named body part ("the penis ... its tip"). Keep
        # those candidates in the same graph so a single clear antecedent can
        # be expanded mechanically instead of being rejected or guessed.
        return [item for item in prior if item.kind in {"object", "body_part"}]
    if token in {"she", "her", "hers"}:
        return [item for item in prior if "female" in item.attributes and item.number == "singular"]
    if token in {"he", "him", "his"}:
        return [item for item in prior if "male" in item.attributes and item.number == "singular"]
    if token in {"they", "them", "their", "theirs"}:
        explicit_groups = [item for item in prior if item.kind == "group"]
        if len(explicit_groups) == 1:
            return explicit_groups
        singular_nonbinary = [
            item for item in prior
            if item.kind == "person" and item.number == "singular" and "nonbinary" in item.attributes
        ]
        other_people = [item for item in prior if item.kind in {"person", "group"}]
        if len(singular_nonbinary) == 1 and len(other_people) == 1:
            return singular_nonbinary
        plural_entities = [item for item in prior if item.number == "plural"]
        if len({item.entity_id for item in plural_entities}) == 1:
            return plural_entities
        singular_people = [
            item
            for item in other_people
            if item.kind == "person" and item.number == "singular"
        ]
        if len({item.entity_id for item in singular_people}) >= 2 and not singular_nonbinary:
            return singular_people
    return []


def resolve_references(mentions: list[EntityMention]) -> list[EntityMention]:
    resolved: list[EntityMention] = []
    for item in mentions:
        if item.kind != "reference":
            resolved.append(item)
            continue
        candidates = _reference_candidates(item, mentions)
        unique_ids = {candidate.entity_id for candidate in candidates}
        if len(unique_ids) == 1:
            target = candidates[-1]
            resolved.append(
                replace(
                    item,
                    entity_id=target.entity_id,
                    number=target.number,
                    attributes=target.attributes,
                    confidence=min(0.96, target.confidence),
                )
            )
        elif (
            item.canonical_name in {"they", "them", "their", "theirs"}
            and len(unique_ids) >= 2
            and all(
                candidate.kind == "person" and candidate.number == "singular"
                for candidate in candidates
            )
            and not any("nonbinary" in candidate.attributes for candidate in candidates)
        ):
            # Two or more explicit singular roles can form one clear plural
            # discourse group ("a doctor and a patient ... they"). Keep this
            # distinct from singular-they ambiguity involving a nonbinary role.
            group_id = "group_set:" + "|".join(sorted(str(value) for value in unique_ids))
            resolved.append(
                replace(
                    item,
                    entity_id=group_id,
                    number="plural",
                    confidence=min(0.94, *(candidate.confidence for candidate in candidates)),
                )
            )
        else:
            resolved.append(replace(item, confidence=0.35 if candidates else 0.2))
    return resolved


def reference_ambiguities(mentions: Iterable[EntityMention]) -> list[EntityMention]:
    return [
        item
        for item in mentions
        if item.kind == "reference" and item.entity_id is None
    ]


def _replacement_label(target: EntityMention, token: str) -> str:
    base = re.sub(r"(?i)^(?:exactly|only)\s+", "", target.source.text.strip())
    base = re.sub(r"(?i)^(?:a|an|the)\s+", "", base)
    if not base.casefold().startswith("the "):
        base = "the " + base
    if token in {"its", "their", "theirs", "his", "her", "hers"}:
        return base + ("'" if target.number == "plural" and base.endswith("s") else "'s")
    return base


def rewrite_high_confidence_references(text: str, field_name: str = "Draft") -> str:
    mentions = resolve_references(extract_entity_mentions(text, field_name))
    by_id = {item.entity_id: item for item in mentions if item.entity_id and item.kind != "reference"}
    replacements: list[tuple[int, int, str]] = []
    for mention in mentions:
        if mention.kind != "reference" or mention.entity_id is None or mention.confidence < 0.85:
            continue
        # Natural person/group pronouns are already resolved in the graph and
        # need no visible rewrite. Replacing them makes fluent prose clumsy and
        # can corrupt object-pronoun grammar ("touching him"). Object references
        # are rewritten because generators benefit from the explicit noun.
        if mention.canonical_name not in {"it", "its", "this", "that"}:
            continue
        target = by_id.get(mention.entity_id)
        if target is None:
            continue
        token = mention.canonical_name
        replacement = _replacement_label(target, token)
        original = text[mention.source.span[0] : mention.source.span[1]]
        if original[:1].isupper():
            replacement = replacement[:1].upper() + replacement[1:]
        replacements.append((*mention.source.span, replacement))
    result = text
    for start, end, replacement in reversed(replacements):
        result = result[:start] + replacement + result[end:]
    return result
