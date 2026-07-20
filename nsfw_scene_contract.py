"""Structured fidelity contracts for explicitly requested adult image prompts.

This module contains no GUI or model dependencies.  It turns adult prompt text
and preset entries into compact machine-readable facts that can be shared by
correction, validation, preset compatibility checks, and generated-image review.
"""

from __future__ import annotations

import re
from typing import Iterable


SEXUAL_SIGNAL_PATTERN = re.compile(
    r"\b(?:nsfw|nude|naked|erotic|sexual|sex|intercourse|masturbat\w*|"
    r"oral\s+sex|anal\s+sex|penetrat\w*|orgasm\w*|climax\w*|foreplay|"
    r"seduc\w*|intimate|kiss(?:es|ed|ing)?|dildos?|vibrators?|strap[- ]ons?|"
    r"sex\s+toys?|adult\s+toys?|bondage)\b",
    re.IGNORECASE,
)
ROLE_PATTERN_TEXT = (
    r"(?:(?:adult\s+)?(?:woman|women|man|men|female|females|male|males)"
    r"\s+partners?|"
    r"(?:adult\s+)?(?:woman|women|man|men|female|females|male|males|"
    r"nonbinary\s+(?:person|people)|non-binary\s+(?:person|people)|"
    r"partner|partners|lover|lovers|subject|subjects|performer|performers|"
    r"dominant|dominants|submissive|submissives))"
)
ROLE_PATTERN = re.compile(rf"\b{ROLE_PATTERN_TEXT}\b", re.IGNORECASE)

NEGATIVE_CLAUSE_PATTERN = re.compile(
    r"(?i)\b(?:no|without|avoid|exclude|never|do\s+not|don't)\b[^,.!?;\n]*"
)
TOY_TERM_PATTERN_TEXT = (
    r"(?:dildos?|vibrators?|strap[- ]ons?|sex\s+toys?|adult\s+toys?|"
    r"anal\s+toys?|wand\s+massagers?)"
)
TOY_USE_ACTION_PATTERN_TEXT = (
    r"(?:use\w*|insert\w*|penetrat\w*|thrust\w*|masturbat\w*|"
    r"stimulat\w*|guid\w*|press\w*|fuck\w*)"
)
TOY_USE_PATTERN = re.compile(
    rf"(?:\b{TOY_USE_ACTION_PATTERN_TEXT}\b[^.!?;]{{0,100}}\b{TOY_TERM_PATTERN_TEXT}\b|"
    rf"\b{TOY_TERM_PATTERN_TEXT}\b[^.!?;]{{0,100}}\b"
    r"(?:vaginally|anally|inside|into|penetrat\w*|insert\w*|contact)\b|"
    rf"\b{TOY_TERM_PATTERN_TEXT}\b[^.!?;]{{0,35}}\b(?:in|inside)\s+"
    r"(?:(?:her|his|their|the|a)\s+)?"
    r"(?:vagina|vaginal\s+opening|vulva|pussy|anus|anal\s+opening|rectum)\b)",
    re.IGNORECASE,
)
DILDO_USE_PATTERN = re.compile(
    rf"(?:\b{TOY_USE_ACTION_PATTERN_TEXT}\b[^.!?;]{{0,100}}\bdildos?\b|"
    r"\bdildos?\b[^.!?;]{0,100}\b"
    r"(?:vaginally|anally|inside|into|penetrat\w*|insert\w*|contact)\b|"
    r"\bdildos?\b[^.!?;]{0,35}\b(?:in|inside)\s+"
    r"(?:(?:her|his|their|the|a)\s+)?"
    r"(?:vagina|vaginal\s+opening|vulva|pussy|anus|anal\s+opening|rectum)\b)",
    re.IGNORECASE,
)

ACT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("toy use", TOY_USE_PATTERN),
    (
        "masturbation",
        re.compile(
            r"\b(?:masturbat\w*|solo\s+pleasure|self[- ](?:pleasur\w*|stimulat\w*|"
            r"penetrat\w*)|fuck\w*\s+(?:herself|himself|themself|themselves))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "oral sex",
        re.compile(
            r"\b(?:oral\s+(?:sex|pleasure|intimacy|stimulation)|"
            r"mouth-to-genital\s+contact|blowjobs?|cunnilingus)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "anal sex",
        re.compile(
            r"\b(?:anal\s+(?:sex|intercourse|penetration|intimacy)|"
            r"penetration\s+at\s+(?:the\s+)?(?:anus|anal\s+opening)|"
            r"rear-entry\s+anal)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "vaginal intercourse",
        re.compile(
            r"\b(?:vaginal\s+(?:sex|intercourse|penetration)|"
            r"penetration\s+at\s+(?:the\s+)?vaginal\s+opening|"
            r"(?:missionary|face-to-face|seated|standing|side-by-side|rear-entry)\s+intercourse)\b",
            re.IGNORECASE,
        ),
    ),
    ("intercourse", re.compile(r"\b(?:intercourse|making\s+love|has?\s+sex)\b", re.IGNORECASE)),
    ("manual stimulation", re.compile(r"\b(?:manual\s+stimulation|handjob|fingering)\b", re.IGNORECASE)),
    ("kissing", re.compile(r"\b(?:kiss(?:es|ed|ing)?|making\s+out)\b", re.IGNORECASE)),
    ("undressing", re.compile(r"\b(?:undress\w*|remov\w+\s+(?:clothing|lingerie))\b", re.IGNORECASE)),
    ("bondage", re.compile(r"\b(?:bondage|restraints?|restrain\w*|blindfold|cuffs?|rope)\b", re.IGNORECASE)),
)
BODY_TARGET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "vaginal",
        re.compile(
            r"\b(?:vagina|vaginal|vaginally|vulva|vulval|labia|pussy)\b",
            re.IGNORECASE,
        ),
    ),
    ("anal", re.compile(r"\b(?:anus|anal|anally|rectum|rectal)\b", re.IGNORECASE)),
    ("oral", re.compile(r"\b(?:mouth|oral|lips|tongue)\b", re.IGNORECASE)),
    ("chest", re.compile(r"\b(?:chest|breasts?|nipples?)\b", re.IGNORECASE)),
    ("genital", re.compile(r"\b(?:genitals?|penis|vagina|vulva|testicles?|scrotum)\b", re.IGNORECASE)),
)
OBJECT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("dildo", re.compile(r"\bdildos?\b", re.IGNORECASE)),
    ("vibrator", re.compile(r"\bvibrators?\b", re.IGNORECASE)),
    ("strap-on", re.compile(r"\bstrap[- ]ons?\b", re.IGNORECASE)),
    ("anal toy", re.compile(r"\b(?:anal\s+toy|butt\s+plug|anal\s+beads?)\b", re.IGNORECASE)),
    ("adult toy", re.compile(r"\b(?:adult|sex)\s+toys?\b", re.IGNORECASE)),
)
PHASE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("aftercare", re.compile(r"\b(?:aftercare|afterglow|cleaning\s+up|morning\s+after|post-climax)\b", re.IGNORECASE)),
    ("climax", re.compile(r"\b(?:climax\w*|orgasm\w*|peak|release)\b", re.IGNORECASE)),
    ("active", re.compile(r"\b(?:penetrat\w*|intercourse|masturbat\w*|thrust\w*|oral\s+(?:sex|pleasure)|manual\s+stimulation)\b", re.IGNORECASE)),
    ("foreplay", re.compile(r"\b(?:foreplay|undress\w*|caress\w*|intimate\s+touch|deep\s+kiss)\b", re.IGNORECASE)),
    ("anticipation", re.compile(r"\b(?:anticipat\w*|seduc\w*|teas\w*|almost-touching|inviting\s+closer)\b", re.IGNORECASE)),
)
REACTION_PATTERN = re.compile(
    r"\b(?:gasp\w*|moan\w*|trembl\w*|shudder\w*|flush\w*|"
    r"breathless|quickened\s+breath|parted\s+lips|closed\s+eyes|"
    r"pleasure|ecstasy|arousal|climax\w*|orgasm\w*)\b",
    re.IGNORECASE,
)
ACTION_CAUSE_PATTERN = re.compile(
    r"\b(?:as|while|when|in\s+response\s+to|because\s+of|from|during|"
    r"touch\w*|kiss\w*|penetrat\w*|thrust\w*|stimulat\w*|uses?|"
    r"press\w*|grip\w*|movement|rhythm|contact|pressure)\b",
    re.IGNORECASE,
)
SEQUENCE_PATTERN = re.compile(
    r"\b(?:then|next|afterward|afterwards|before\s+continuing|"
    r"moves?\s+from.+?\s+to|shifts?\s+from.+?\s+into|"
    r"begins?.+?\s+then|progress(?:es|ing)?\s+from)\b",
    re.IGNORECASE,
)
COUNT_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}


def _unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = " ".join(str(value).split()).strip()
        key = cleaned.casefold()
        if cleaned and key not in seen:
            result.append(cleaned)
            seen.add(key)
    return result


def _positive_contract_text(text: str) -> str:
    """Remove clauses that describe absent or prohibited scene content."""

    return " ".join(
        NEGATIVE_CLAUSE_PATTERN.sub(" ", str(text or "")).split()
    )


def _dildo_use_context(text: str) -> str:
    for sentence in re.split(r"(?<=[.!?;])\s+", text):
        if DILDO_USE_PATTERN.search(sentence):
            return sentence
    return ""


def _literal_dildo_contact(text: str) -> str:
    """Return only a source-supplied compact dildo/body relation."""

    match = re.search(
        r"\bdildos?\b\s+(?:in|inside)\s+"
        r"(?:(?:her|his|their|the|a)\s+)?"
        r"(?:vagina|vaginal\s+opening|vulva|pussy|anus|anal\s+opening|rectum)\b",
        text,
        flags=re.IGNORECASE,
    )
    return " ".join(match.group(0).split()) if match else ""


def dildo_direction_instruction(contract: dict[str, object]) -> str:
    """Return one concrete direction only for ordinary active dildo use."""

    target = str(contract.get("dildo_use_target", "")).strip()
    if not target:
        return ""
    return (
        "Keep the dildo visibly separate from the body: its rounded insertion tip "
        f"points toward the {target}, while its base or handle stays outside on the "
        "operator side and points away."
    )


def _participant_count(text: str) -> int | None:
    lowered = text.lower()
    if re.search(
        r"\b(?:solo|single)\s+"
        r"(?:(?:clearly\s+adult|mature\s+adult|middle-aged|older|adult|mature)\s+)?"
        r"(?:woman|man|person|subject)\b",
        lowered,
    ):
        return 1
    matches = re.findall(
        rf"\b(one|two|three|four|five|six|\d+)\s+{ROLE_PATTERN_TEXT}\b",
        lowered,
    )
    if matches:
        value = matches[0]
        return COUNT_WORDS.get(value, int(value) if value.isdigit() else 1)
    if re.search(r"\b(?:couple|two\s+partners|both\s+adults)\b", lowered):
        return 2
    if re.search(r"\b(?:three\s+partners|multiple\s+adults|group\s+sex)\b", lowered):
        return 3
    distinct_roles = {
        _canonical_role(match.group(0))
        for match in ROLE_PATTERN.finditer(lowered)
        if _canonical_role(match.group(0))
        in {"woman", "man", "female", "male", "nonbinary person", "non-binary person"}
    }
    if len(distinct_roles) >= 2:
        return len(distinct_roles)
    return None


RELATION_ACTIONS: tuple[tuple[str, str], ...] = (
    ("kissing", r"kiss(?:es|ed|ing)?"),
    ("penetration", r"penetrat(?:e|es|ed|ing)"),
    ("manual stimulation", r"(?:touch(?:es|ed|ing)?|stimulat(?:e|es|ed|ing))"),
    ("oral sex", r"(?:gives?|performs?)\s+oral\s+(?:sex|pleasure)\s+(?:to|on)"),
)


def _canonical_role(value: str) -> str:
    role = re.sub(r"^adult\s+", "", value.strip().lower())
    role = re.sub(r"\s+partners?$", "", role)
    singular = {
        "women": "woman",
        "men": "man",
        "female": "woman",
        "females": "woman",
        "male": "man",
        "males": "man",
        "partners": "partner",
        "lovers": "lover",
        "subjects": "subject",
        "performers": "performer",
        "dominants": "dominant",
        "submissives": "submissive",
        "nonbinary people": "nonbinary person",
        "non-binary people": "non-binary person",
    }
    return singular.get(role, role)


def _relations(text: str) -> list[dict[str, str]]:
    relations: list[dict[str, str]] = []
    for action, action_pattern in RELATION_ACTIONS:
        verb_pattern = re.compile(rf"\b(?:{action_pattern})\b", re.IGNORECASE)
        for sentence in re.split(r"(?<=[.!?;])\s+", text):
            for verb_match in verb_pattern.finditer(sentence):
                preceding_roles = [
                    match
                    for match in ROLE_PATTERN.finditer(sentence[: verb_match.start()])
                    if verb_match.start() - match.end() <= 90
                ]
                following_roles = [
                    match
                    for match in ROLE_PATTERN.finditer(sentence[verb_match.end():])
                    if match.start() <= 70
                ]
                if not preceding_roles or not following_roles:
                    continue
                actor = preceding_roles[-1].group(0)
                receiver = following_roles[0].group(0)
                relations.append(
                    {
                        "actor": _canonical_role(actor),
                        "action": action,
                        "receiver": _canonical_role(receiver),
                    }
                )
    return relations


def extract_nsfw_scene_contract(
    text: str,
    *,
    content_format: str = "Single Image",
) -> dict[str, object]:
    """Extract participant, act, contact, object, phase, and reaction facts."""

    normalized = _positive_contract_text(text)
    roles = _unique(match.group(0).lower() for match in ROLE_PATTERN.finditer(normalized))
    participant_count = _participant_count(normalized)
    acts = [label for label, pattern in ACT_PATTERNS if pattern.search(normalized)]
    # A solo adult actively using an intimate toy is still masturbating even
    # when a rewrite replaces that label with concrete visual mechanics such
    # as "thrusts a dildo into her vagina." Keep the semantic act stable so the
    # hard fidelity gate does not demand one exact word from the local model.
    if participant_count == 1 and "toy use" in acts and "masturbation" not in acts:
        acts.append("masturbation")
    if "intercourse" in acts and any(
        act in acts for act in ("anal sex", "vaginal intercourse")
    ):
        acts.remove("intercourse")
    targets = (
        [label for label, pattern in BODY_TARGET_PATTERNS if pattern.search(normalized)]
        if acts
        else []
    )
    if "genital" in targets and any(
        value in targets for value in ("vaginal", "anal")
    ):
        targets.remove("genital")
    objects = [label for label, pattern in OBJECT_PATTERNS if pattern.search(normalized)]
    if "adult toy" in objects and any(
        value in objects for value in ("dildo", "vibrator", "strap-on", "anal toy")
    ):
        objects.remove("adult toy")
    phases = [label for label, pattern in PHASE_PATTERNS if pattern.search(normalized)]
    chosen_phase = next(
        (
            phase
            for phase in ("climax", "active", "foreplay", "anticipation", "aftercare")
            if phase in phases
        ),
        "active" if acts else "",
    )
    reaction_terms = _unique(match.group(0).lower() for match in REACTION_PATTERN.finditer(normalized))
    dildo_context = _dildo_use_context(normalized)
    dildo_use_target = ""
    if dildo_context and not re.search(
        r"\b(?:double[- ]ended|product|boxed|unopened|packaged|display)\b",
        dildo_context,
        flags=re.IGNORECASE,
    ):
        dildo_use_target = (
            "vaginal opening"
            if BODY_TARGET_PATTERNS[0][1].search(dildo_context)
            else "anal opening"
            if BODY_TARGET_PATTERNS[1][1].search(dildo_context)
            else "intended body-contact point"
        )
    return {
        "sexual": bool(SEXUAL_SIGNAL_PATTERN.search(normalized)),
        "participant_count": participant_count,
        "participant_roles": roles,
        "acts": acts,
        "body_targets": targets,
        "objects": objects,
        "dildo_use_target": dildo_use_target,
        "literal_core": _literal_dildo_contact(normalized),
        "relations": _relations(normalized),
        "phases": phases,
        "visible_phase": chosen_phase,
        "reactions": reaction_terms,
        "single_phase_required": content_format == "Single Image",
    }


def format_nsfw_scene_contract(
    contract: dict[str, object],
    *,
    risk_level: str = "Balanced improvement",
) -> str:
    """Render a small literal-core contract without teaching scene mechanics."""

    if not contract.get("sexual"):
        return ""
    facts: list[str] = []
    participant_count = contract.get("participant_count")
    if participant_count is not None:
        facts.append(f"adult count={participant_count}")
    acts = ", ".join(contract.get("acts", []))
    if acts:
        facts.append(f"action={acts}")
    targets = ", ".join(contract.get("body_targets", []))
    if targets:
        facts.append(f"contact={targets}")
    objects = ", ".join(contract.get("objects", []))
    if objects:
        facts.append(f"object={objects}")
    literal_core = str(contract.get("literal_core", "")).strip()
    if literal_core:
        facts.append(f"literal wording={literal_core}")
    lines = ["Private literal adult-scene core:"]
    if facts:
        lines.append("- Keep these source facts unchanged: " + "; ".join(facts) + ".")
    relations = contract.get("relations", [])
    if isinstance(relations, list):
        for relation in relations:
            if isinstance(relation, dict):
                lines.append(
                    "- Keep this role direction: "
                    f"{relation.get('actor', '')} -> {relation.get('action', '')} -> "
                    f"{relation.get('receiver', '')}."
                )
    lines.extend(
        (
            "- State the core action once in short, ordinary image-generator wording.",
            "- Do not explain anatomy, toy geometry, insertion mechanics, or chronological phases unless the user requested them.",
        )
    )
    if risk_level == "Strict cleanup":
        lines.append(
            "- Strict cleanup: invent no scene facts; a source containing only the literal core stays concise."
        )
    elif risk_level == "Creative enhancement":
        lines.append(
            "- Creative enhancement: keep the literal core first, then build one coherent visual direction through compatible setting, staging, camera, lighting, style, material, and visible reaction details."
        )
        lines.append(
            "- Creative additions must not introduce another participant, sexual act, fetish, body target, object, identity, power dynamic, fluid, or outcome."
        )
    else:
        lines.append(
            "- Balanced improvement: keep the literal core first, then add one compact compatible visual cluster such as setting and staging, camera and composition, or lighting and style."
        )
        lines.append(
            "- Do not invent nudity, fluids, exclusions, another participant, sexual act, fetish, body target, object, identity, power dynamic, or outcome."
        )
    lines.append("- Do not quote or label this private literal core in the final prompt.")
    return "\n".join(lines)


def reaction_binding_issues(text: str, *, participant_count: int | None) -> list[str]:
    """Find unowned reactions in scenes with more than one adult participant."""

    if not participant_count or participant_count < 2:
        return []
    issues: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        reactions = _unique(match.group(0).lower() for match in REACTION_PATTERN.finditer(sentence))
        if not reactions:
            continue
        if not ROLE_PATTERN.search(sentence):
            issues.append("reaction is not assigned to a named adult role: " + ", ".join(reactions[:3]))
        if not ACTION_CAUSE_PATTERN.search(sentence):
            issues.append("reaction is not tied to its causing action or contact: " + ", ".join(reactions[:3]))
    return _unique(issues)


def single_phase_issues(text: str, *, content_format: str) -> list[str]:
    """Reject visible multi-step progression in a normal still-image prompt."""

    if content_format != "Single Image" or not SEQUENCE_PATTERN.search(text):
        return []
    phases = [label for label, pattern in PHASE_PATTERNS if pattern.search(text)]
    if len(phases) >= 2:
        return [
            "single-image prompt contains a visible multi-phase progression: "
            + ", ".join(phases)
        ]
    return []


def nsfw_scene_contract_issues(
    final_prompt: str,
    original_prompt: str,
    *,
    content_format: str = "Single Image",
) -> list[str]:
    """Compare the final adult scene with explicit source acts and relationships."""

    source = extract_nsfw_scene_contract(original_prompt, content_format=content_format)
    if not source.get("sexual"):
        return []
    candidate = extract_nsfw_scene_contract(final_prompt, content_format=content_format)
    issues: list[str] = []
    candidate_acts = set(candidate.get("acts", []))
    for act in source.get("acts", []):
        if act == "intercourse" and candidate_acts.intersection(
            {"intercourse", "anal sex", "vaginal intercourse"}
        ):
            continue
        if act not in candidate_acts:
            issues.append(f"missing requested sexual act family: {act}")
    source_targets = set(source.get("body_targets", []))
    candidate_targets = set(candidate.get("body_targets", []))
    for target in source_targets - candidate_targets:
        issues.append(f"missing requested body/contact target: {target}")
    if "vaginal" in source_targets and "anal" in candidate_targets - source_targets:
        issues.append("unrequested anal contact added to a vaginal source")
    if "anal" in source_targets and "vaginal" in candidate_targets - source_targets:
        issues.append("unrequested vaginal contact added to an anal source")
    for requested_object in source.get("objects", []):
        candidate_objects = set(candidate.get("objects", []))
        generic_toy_satisfied = (
            requested_object == "adult toy"
            and bool(candidate_objects.intersection(
                {"adult toy", "dildo", "vibrator", "strap-on", "anal toy"}
            ))
        )
        if requested_object not in candidate_objects and not generic_toy_satisfied:
            issues.append(f"missing requested adult object: {requested_object}")

    candidate_relations = {
        (
            str(relation.get("actor", "")),
            str(relation.get("action", "")),
            str(relation.get("receiver", "")),
        )
        for relation in candidate.get("relations", [])
        if isinstance(relation, dict)
    }
    for relation in source.get("relations", []):
        if not isinstance(relation, dict):
            continue
        key = (
            str(relation.get("actor", "")),
            str(relation.get("action", "")),
            str(relation.get("receiver", "")),
        )
        if key not in candidate_relations:
            issues.append(
                "missing or reversed sexual role binding: "
                f"{key[0]} -> {key[1]} -> {key[2]}"
            )
    return _unique(issues)


def infer_nsfw_preset_metadata(
    kind: str,
    category: str,
    value: str,
) -> dict[str, object]:
    """Infer complete compatibility metadata for one adult catalog entry."""

    text = f"{category} {value}".lower()
    contract = extract_nsfw_scene_contract(value)
    participant_modes: list[str] = []
    if re.search(r"\b(?:solo|one adult|single adult|own expression)\b", text):
        participant_modes.append("solo")
    if re.search(r"\b(?:three|multiple|group|polyamorous)\b", text):
        participant_modes.append("group")
    if re.search(r"\b(?:partners?|lovers?|couple|two-person|both adults)\b", text):
        participant_modes.append("couple")
    if not participant_modes:
        participant_modes.append("any")
    return {
        "kind": kind,
        "category": category,
        "value": value,
        "participant_modes": _unique(participant_modes),
        "act_families": list(contract.get("acts", [])),
        "body_targets": list(contract.get("body_targets", [])),
        "objects": list(contract.get("objects", [])),
        "phase": str(contract.get("visible_phase", "")),
        "requires_separate_object": bool(contract.get("objects")),
        "reaction_cues": _unique(
            match.group(0).lower() for match in REACTION_PATTERN.finditer(value)
        ),
    }


def nsfw_preset_compatibility_issues(
    metadata: Iterable[dict[str, object]],
    *,
    content_format: str = "Single Image",
) -> list[str]:
    """Return conflicts across selected adult presets."""

    entries = list(metadata)
    modes = {
        str(mode)
        for entry in entries
        for mode in entry.get("participant_modes", [])
        if str(mode) != "any"
    }
    phases = {
        str(entry.get("phase", ""))
        for entry in entries
        if str(entry.get("phase", ""))
    }
    acts = {
        str(act)
        for entry in entries
        for act in entry.get("act_families", [])
    }
    issues: list[str] = []
    if "solo" in modes and modes.intersection({"couple", "group"}):
        issues.append("solo and multi-participant adult presets are selected together")
    if content_format == "Single Image" and len(phases) > 1:
        issues.append(
            "adult presets span multiple visible phases for one still: "
            + ", ".join(sorted(phases))
        )
    if "anal sex" in acts and "vaginal intercourse" in acts and content_format == "Single Image":
        issues.append("anal and vaginal intercourse presets compete for the same still-image beat")
    return issues


def format_nsfw_preset_contract(
    metadata: Iterable[dict[str, object]],
    *,
    content_format: str = "Single Image",
) -> str:
    """Summarize selected adult preset metadata as private compatibility guidance."""

    entries = list(metadata)
    if not entries:
        return ""
    acts = _unique(
        str(act)
        for entry in entries
        for act in entry.get("act_families", [])
    )
    phases = _unique(str(entry.get("phase", "")) for entry in entries)
    objects = _unique(
        str(obj)
        for entry in entries
        for obj in entry.get("objects", [])
    )
    issues = nsfw_preset_compatibility_issues(entries, content_format=content_format)
    lines = ["Private NSFW preset compatibility:"]
    if acts:
        lines.append("- Selected act families: " + ", ".join(acts))
    if phases:
        lines.append("- Selected phases: " + ", ".join(phases))
    if objects:
        lines.append("- Selected separate objects: " + ", ".join(objects))
    if issues:
        lines.append("- Resolve these conflicts in favor of the user's draft: " + "; ".join(issues))
    lines.append("- Do not quote or label this private preset metadata in the final prompt.")
    return "\n".join(lines)


def strip_nsfw_catalog_labels(text: str) -> str:
    """Remove internal adult-library category names while preserving chosen values."""

    cleaned = re.sub(
        r"(?i)\bNSFW\s*(?:[—–-]|,)\s*[^:.;\n]{1,100}\s*:\s*",
        "",
        str(text or ""),
    )
    cleaned = re.sub(
        r"(?i)\bNSFW\s*[—–-]\s*(?:adult\s+)?(?:erotic|sexual)\s+",
        "",
        cleaned,
    )
    return re.sub(r"\s{2,}", " ", cleaned).strip()


def nsfw_image_audit_contract(
    original_prompt: str,
    corrected_prompt: str,
    *,
    content_format: str = "Single Image",
) -> str:
    """Return exact adult-scene checks for generated-image inspection."""

    source = extract_nsfw_scene_contract(original_prompt, content_format=content_format)
    corrected = extract_nsfw_scene_contract(corrected_prompt, content_format=content_format)
    acts = _unique([*source.get("acts", []), *corrected.get("acts", [])])
    targets = _unique([*source.get("body_targets", []), *corrected.get("body_targets", [])])
    objects = _unique([*source.get("objects", []), *corrected.get("objects", [])])
    direction = dildo_direction_instruction(source) or dildo_direction_instruction(corrected)
    lines = [
        "NSFW visual fidelity audit:",
        f"- Adult participant count and distinct roles: {source.get('participant_count') or 'as requested'}; "
        + (", ".join(source.get("participant_roles", [])) or "preserve the supplied adult roles"),
        "- Required sexual act families: " + (", ".join(acts) or "preserve the requested intimate action"),
        "- Required body/contact targets: " + (", ".join(targets) or "preserve the supplied contact"),
        "- Required separate objects: " + (", ".join(objects) or "none explicitly extracted"),
        *([f"- Required normal-use object direction: {direction}"] if direction else []),
        f"- Required visible phase: {source.get('visible_phase') or corrected.get('visible_phase') or 'one decisive phase'}",
        "- Verify actor and receiver roles, limb ownership, object/body separation, contact direction, "
        "and participant-specific reactions. Report every mismatch in nsfw_fidelity.",
    ]
    return "\n".join(lines)
