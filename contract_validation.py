"""Contract compilation, authority resolution, preflight, and fact deltas."""

from __future__ import annotations

from dataclasses import replace
import re
from typing import Iterable, Mapping

from entity_resolution import (
    canonical_entity_name,
    canonical_head,
    extract_entity_mentions,
    reference_ambiguities,
    resolve_references,
)
from prompt_contract import (
    ComplianceIssue,
    ContractFact,
    IssueText,
    PromptContract,
    RelationFact,
    SEVERITY_ADVISORY,
    SEVERITY_BLOCKER,
    SEVERITY_HARD,
    SourceRef,
    assign_fact_ids,
    assign_relation_ids,
    issue_text,
)
from scene_dimensions import (
    CAMERA_CONFLICTS,
    SPATIAL_OPPOSITES,
    camera_values_conflict,
    compile_dimension_facts,
    entity_fact_matches,
    extract_camera_facts,
    extract_count_facts,
    extract_exclusion_facts,
    extract_observed_count_facts,
    extract_spatial_facts,
    spatial_values_conflict,
)
from scene_relations import extract_relation_facts, relation_reversal


DEFAULT_FIELDS = (
    "Draft", "Story", "Concepts", "Goal", "Focus", "Instructions",
    "Weighted terms", "Feedback", "Visual direction", "Camera", "Format", "Mode",
)

REVISION_MARKER = re.compile(
    r"(?i)\b(?:change|replace|remove|instead|switch|make|turn|delete|move|position|place)\b"
)


def _semantic_entity(value: str | None) -> str:
    raw = str(value or "")
    if ":" in raw and re.match(r"^(?:person|group|object|body_part)_\d+:", raw):
        raw = raw.split(":", 1)[1]
    return canonical_entity_name(raw)


def _fact_key(fact: ContractFact) -> tuple[object, ...]:
    entity = _semantic_entity(fact.entity_id)
    if fact.dimension == "camera":
        return (fact.dimension, "camera", fact.scope_id)
    return (fact.dimension, entity, fact.scope_id, fact.context)


def _revision_count_facts(text: str, field_name: str) -> list[ContractFact]:
    source_text = str(text or "")
    pattern = re.compile(
        r"(?ix)\b(?:change|make|turn|replace)\b[^.!?;]{0,70}?\b(?:to|with|into)\s+"
        r"(?P<count>one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+"
        r"(?P<entity>(?:[a-z][a-z-]*\s+){0,3}[a-z][a-z-]*)"
        r"(?=\s+(?:appears?|are|is|remain(?:s)?|sits?|stands?)\b|[.;,:]|$)"
    )
    values = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
        "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    }
    facts: list[ContractFact] = []
    for match in pattern.finditer(source_text):
        token = match.group("count").casefold()
        facts.append(
            ContractFact(
                code="count.revision",
                entity_id=canonical_entity_name(match.group("entity")),
                value=values.get(token, int(token) if token.isdigit() else 0),
                polarity="replaced",
                scope_id="scene:primary",
                source=SourceRef.create(field_name, source_text, match.span()),
                confidence=0.98,
                dimension="count",
            )
        )
    return facts


def _revision_camera_facts(text: str, field_name: str) -> list[ContractFact]:
    source_text = str(text or "")
    facts: list[ContractFact] = []
    for match in re.finditer(
        r"(?i)\b(?:replace|change|switch)\b[^.!?;]{0,80}?\b(?:with|to|into)\b(?P<new>[^.!?;]+)",
        source_text,
    ):
        for fact in extract_camera_facts(match.group("new"), field_name):
            offset = match.start("new")
            facts.append(
                replace(
                    fact,
                    code="camera.revision",
                    polarity="replaced",
                    source=SourceRef.create(
                        field_name,
                        source_text,
                        (offset + fact.source.span[0], offset + fact.source.span[1]),
                    ),
                )
            )
    return facts


def _revision_position_facts(text: str, field_name: str) -> list[ContractFact]:
    source_text = str(text or "")
    facts: list[ContractFact] = []
    pattern = re.compile(
        r"(?ix)\b(?:move|position|place|change)\s+(?:the\s+)?"
        r"(?P<entity>(?:[a-z][a-z-]*\s+){0,3}?[a-z][a-z-]*)\s+"
        r"(?:(?:from\s+[^.!?;]{1,60}?\s+)?to\s+|on\s+)?(?:the\s+)?"
        r"(?P<relation>far\s+left\s+of|far\s+right\s+of|left\s+of|right\s+of|"
        r"in\s+front\s+of|behind|above|below|inside|outside)\s+(?:the\s+)?"
        r"(?P<target>(?:[a-z][a-z-]*\s+){0,2}[a-z][a-z-]*)"
        r"(?=[,.;!?]|$)"
    )
    for match in pattern.finditer(source_text):
        facts.append(
            ContractFact(
                code="position.revision",
                entity_id=canonical_entity_name(match.group("entity")),
                value=re.sub(r"\s+", " ", match.group("relation").casefold()),
                polarity="replaced",
                scope_id="scene:primary",
                source=SourceRef.create(field_name, source_text, match.span()),
                confidence=0.98,
                dimension="position",
                context=(canonical_entity_name(match.group("target")),),
            )
        )
    return facts


def _revision_facts(text: str, field_name: str) -> list[ContractFact]:
    if not REVISION_MARKER.search(str(text or "")):
        return []
    return (
        _revision_count_facts(text, field_name)
        + _revision_camera_facts(text, field_name)
        + _revision_position_facts(text, field_name)
    )


def _resolve_authority(facts: list[ContractFact]) -> list[ContractFact]:
    """Supersede only matching dimensions/entities targeted by revisions."""

    result = list(facts)
    revisions = [fact for fact in result if fact.polarity == "replaced"]
    for revision in revisions:
        revision_key = _fact_key(revision)
        for index, fact in enumerate(result):
            if fact is revision or fact.polarity == "superseded":
                continue
            if fact.source.authority >= revision.source.authority:
                continue
            if _fact_key(fact) == revision_key:
                result[index] = replace(fact, polarity="superseded")
    # A dedicated control owns its matching dimension even without explicit
    # revision grammar.  It does not erase unrelated dimensions or scopes.
    for dimension, field_name in (("camera", "Camera"), ("camera", "Visual direction")):
        authoritative = [
            fact for fact in result
            if fact.dimension == dimension and fact.source.field == field_name
            and fact.polarity != "superseded"
        ]
        if not authoritative:
            continue
        for index, fact in enumerate(result):
            if (
                fact.dimension == dimension
                and fact.source.field != field_name
                and fact.source.authority < authoritative[0].source.authority
                and fact.scope_id == authoritative[0].scope_id
            ):
                result[index] = replace(fact, polarity="superseded")
    return result


def compile_prompt_contract(
    fields: Mapping[str, str] | Iterable[tuple[str, str]],
) -> PromptContract:
    values = dict(fields)
    values = {field: str(value or "") for field, value in values.items() if str(value or "").strip()}
    mentions = []
    facts: list[ContractFact] = []
    relations: list[RelationFact] = []
    issues: list[ComplianceIssue] = []
    for field_name, text in values.items():
        field_mentions = resolve_references(extract_entity_mentions(text, field_name))
        mentions.extend(field_mentions)
        facts.extend(compile_dimension_facts(text, field_name))
        facts.extend(_revision_facts(text, field_name))
        relations.extend(extract_relation_facts(text, field_name, field_mentions))
        for ambiguous in reference_ambiguities(field_mentions):
            issues.append(
                ComplianceIssue(
                    code="reference.ambiguous",
                    severity=SEVERITY_ADVISORY,
                    origin="source" if field_name != "Candidate" else "candidate",
                    source_fact_ids=(),
                    candidate_fact_ids=(),
                    message=f'Ambiguous reference "{ambiguous.source.text}" in {field_name}',
                    field=field_name,
                    span=ambiguous.source.span,
                    dimension="reference",
                )
            )
    facts = assign_fact_ids(_resolve_authority(facts))
    relations = assign_relation_ids(relations)
    return PromptContract(values, mentions, facts, relations, issues)


def _positive_occurrence(entity: str, text: str) -> bool:
    head = canonical_head(entity)
    if not head:
        return False
    positive = re.sub(
        r"(?i)\b(?:no|without|avoid|exclude|remove|do not|don't|never)\b[^.!?;\n]*",
        " ",
        text,
    )
    aliases = {head}
    if head == "person":
        aliases |= {"people", "human", "humans", "persons"}
    elif head == "woman":
        aliases |= {"women", "female", "females", "lady", "ladies"}
    elif head == "man":
        aliases |= {"men", "male", "males"}
    elif head == "child":
        aliases |= {"children"}
    patterns = set(aliases)
    patterns |= {
        alias + ("es" if alias.endswith(("s", "x", "ch", "sh")) else "s")
        for alias in aliases
        if alias not in {"people", "women", "men", "children"}
    }
    return any(re.search(rf"\b{re.escape(alias)}\b", positive, flags=re.IGNORECASE) for alias in patterns)


def preflight_contract_issues(contract: PromptContract) -> list[ComplianceIssue]:
    active = contract.active_facts()
    issues: list[ComplianceIssue] = []

    exclusions = [fact for fact in active if fact.dimension == "exclusion" and fact.confidence >= 0.85]
    for exclusion in exclusions:
        conflict_fields = [
            field for field, text in contract.fields.items()
            if field != exclusion.source.field
            and _positive_occurrence(str(exclusion.entity_id), text)
            and not (
                exclusion.source.authority > SourceRef.create(field, text).authority
                and REVISION_MARKER.search(exclusion.source.text)
            )
        ]
        if conflict_fields:
            display_entity = str(exclusion.entity_id)
            source_words = re.findall(r"[A-Za-z][A-Za-z-]*", exclusion.source.text)
            head = canonical_head(display_entity)
            authored = next(
                (
                    word for word in source_words
                    if canonical_head(word) == head
                ),
                "",
            )
            if authored:
                display_entity = authored.casefold()
            issues.append(
                ComplianceIssue(
                    code="input.exclusion_conflict",
                    severity=SEVERITY_BLOCKER,
                    origin="input",
                    source_fact_ids=(exclusion.fact_id,),
                    candidate_fact_ids=(),
                    message=(
                        f'Input exclusion conflict for "{display_entity}": excluded in '
                        f'{exclusion.source.field}; requested positively in {", ".join(conflict_fields)}'
                    ),
                    field=exclusion.source.field,
                    span=exclusion.source.span,
                    dimension="exclusion",
                )
            )

    for index, left in enumerate(active):
        for right in active[index + 1 :]:
            if left.confidence < 0.85 or right.confidence < 0.85:
                continue
            conflict = False
            code = ""
            label = ""
            if left.dimension == right.dimension == "count" and _fact_key(left) == _fact_key(right) and left.value != right.value:
                conflict, code, label = True, "input.count_conflict", "count"
            elif left.dimension == right.dimension == "camera" and camera_values_conflict(left, right):
                conflict, code, label = True, "input.camera_conflict", "camera"
            elif left.dimension == right.dimension == "position" and spatial_values_conflict(left, right):
                conflict, code, label = True, "input.position_conflict", "position"
            if not conflict:
                continue
            # Authority resolves only explicit revision/control precedence.  Two
            # ordinary prose fields remain a genuine input contradiction.
            revision_resolved = (
                left.polarity == "replaced" or right.polarity == "replaced"
                or left.source.field in {"Camera", "Visual direction"}
                or right.source.field in {"Camera", "Visual direction"}
            )
            if revision_resolved and left.source.authority != right.source.authority:
                continue
            entity = _semantic_entity(left.entity_id) or label
            issues.append(
                ComplianceIssue(
                    code=code,
                    severity=SEVERITY_BLOCKER,
                    origin="input",
                    source_fact_ids=(left.fact_id, right.fact_id),
                    candidate_fact_ids=(),
                    message=(
                        f'Input {label} conflict for "{entity}": {left.value} in '
                        f'{left.source.field} conflicts with {right.value} in {right.source.field}'
                    ),
                    field=right.source.field,
                    span=right.source.span,
                    dimension=label,
                )
            )
    return _dedupe_issues(issues)


def _matching_fact(source: ContractFact, candidates: Iterable[ContractFact]) -> ContractFact | None:
    for candidate in candidates:
        if source.dimension != candidate.dimension:
            continue
        if source.dimension == "camera":
            if source.scope_id == candidate.scope_id and source.value == candidate.value:
                return candidate
        elif entity_fact_matches(source, candidate) and source.value == candidate.value and source.context == candidate.context:
            return candidate
    return None


def candidate_delta_issues(
    source_contract: PromptContract,
    candidate_text: str,
) -> list[ComplianceIssue]:
    candidate = compile_prompt_contract({"Candidate": candidate_text})
    candidate.facts = assign_fact_ids(
        extract_observed_count_facts(candidate_text)
        + extract_exclusion_facts(candidate_text, "Candidate")
        + extract_camera_facts(candidate_text, "Candidate")
        + extract_spatial_facts(candidate_text, "Candidate")
    )
    issues: list[ComplianceIssue] = []

    source_active = [fact for fact in source_contract.active_facts() if fact.confidence >= 0.85]
    for source in source_active:
        if source.dimension not in {"count", "camera", "position", "exclusion"}:
            continue
        if source.dimension == "exclusion":
            if _positive_occurrence(str(source.entity_id), candidate_text):
                source_label_match = re.search(
                    r"(?i)\b(?:no|without|avoid|exclude|remove|do\s+not|don't|never)\s+"
                    r"(?:any\s+)?([a-z][a-z-]*)",
                    source.source.text,
                )
                excluded_label = (
                    source_label_match.group(1)
                    if source_label_match
                    else str(source.entity_id)
                )
                issues.append(
                    _delta_issue(
                        "candidate.exclusion_added", source, None,
                        f"Excluded content appears positively: {excluded_label}",
                    )
                )
            continue
        matched = _matching_fact(source, candidate.facts)
        if matched is not None:
            continue
        conflicting = next(
            (
                fact for fact in candidate.facts
                if fact.dimension == source.dimension
                and (
                    (source.dimension == "camera" and fact.scope_id == source.scope_id)
                    or entity_fact_matches(source, fact)
                )
            ),
            None,
        )
        if conflicting is not None:
            issues.append(
                _delta_issue(
                    f"candidate.{source.dimension}_changed",
                    source,
                    conflicting,
                    f"Candidate changed {source.dimension} for {_semantic_entity(source.entity_id)} "
                    f"from {source.value} to {conflicting.value}",
                )
            )
        elif source.dimension in {"count", "camera"}:
            issues.append(
                _delta_issue(
                    f"candidate.{source.dimension}_missing",
                    source,
                    None,
                    f"Candidate dropped required {source.dimension} for "
                    f"{_semantic_entity(source.entity_id) or source.scope_id}: {source.value}",
                )
            )

    source_relations = [relation for relation in source_contract.relations if relation.confidence >= 0.85]
    candidate_relations = [relation for relation in candidate.relations if relation.confidence >= 0.85]
    for source in source_relations:
        reversal = next((item for item in candidate_relations if relation_reversal(source, item)), None)
        if reversal is not None:
            issues.append(
                ComplianceIssue(
                    code="candidate.relation_reversed",
                    severity=SEVERITY_HARD,
                    origin="candidate",
                    source_fact_ids=(source.relation_id,),
                    candidate_fact_ids=(reversal.relation_id,),
                    message=f"Candidate reversed actor and receiver for {source.predicate}",
                    field=source.source.field,
                    span=source.source.span,
                    dimension="relation",
                )
            )
        if source.predicate == "owns_body_part" and source.body_target_id:
            source_body = source.body_target_id.rsplit(":", 1)[-1]
            reassigned = next(
                (
                    item for item in candidate_relations
                    if item.predicate == "owns_body_part"
                    and item.body_target_id
                    and item.body_target_id.rsplit(":", 1)[-1] == source_body
                    and _semantic_entity(item.actor_id) != _semantic_entity(source.actor_id)
                ),
                None,
            )
            if reassigned is not None:
                issues.append(
                    ComplianceIssue(
                        code="candidate.ownership_reassigned",
                        severity=SEVERITY_HARD,
                        origin="candidate",
                        source_fact_ids=(source.relation_id,),
                        candidate_fact_ids=(reassigned.relation_id,),
                        message=f"Candidate reassigned ownership of {source_body}",
                        field=source.source.field,
                        span=source.source.span,
                        dimension="ownership",
                    )
                )
    # New unresolved references are hard only when they exceed inherited source
    # uncertainty.  Ambiguity is a structural defect budget, not a pronoun-word
    # identity contract: changing inherited "her" to "them" must not manufacture
    # a new hard failure when the amount of uncertainty did not increase.
    def ambiguity_class(issue: ComplianceIssue) -> str:
        match = re.search(r'(?i)Ambiguous reference "([^"]+)"', issue.message)
        token = match.group(1).casefold() if match else ""
        return "object" if token in {"it", "its", "this", "that"} else "person"

    source_ambiguity_budget: dict[str, int] = {}
    for source_issue in source_contract.issues:
        if source_issue.code != "reference.ambiguous":
            continue
        category = ambiguity_class(source_issue)
        source_ambiguity_budget[category] = source_ambiguity_budget.get(category, 0) + 1
    candidate_ambiguities = [
        issue for issue in candidate.issues
        if issue.code == "reference.ambiguous"
    ]
    candidate_people = {
        item.entity_id
        for item in candidate.mentions
        if item.entity_id and item.kind in {"person", "group"}
    }
    candidate_has_group = any(
        item.entity_id and item.kind == "group"
        for item in candidate.mentions
    )
    consumed_ambiguity: dict[str, int] = {}
    for issue in candidate_ambiguities:
        category = ambiguity_class(issue)
        consumed = consumed_ambiguity.get(category, 0)
        if consumed < source_ambiguity_budget.get(category, 0):
            consumed_ambiguity[category] = consumed + 1
            continue
        # A newly introduced object-style reference is invalid even when the
        # extractor found no antecedent: leading "it" and "its" are just as
        # unusable as a reference shared by two props. Person ambiguity keeps
        # the established conservative two-person hardening threshold.
        if category == "object" or len(candidate_people) >= 2 or candidate_has_group:
            issues.append(replace(issue, severity=SEVERITY_HARD, code="candidate.reference_ambiguous"))
    return _dedupe_issues(issues)


def _delta_issue(
    code: str,
    source: ContractFact,
    candidate: ContractFact | None,
    message: str,
) -> ComplianceIssue:
    return ComplianceIssue(
        code=code,
        severity=SEVERITY_HARD,
        origin="candidate",
        source_fact_ids=(source.fact_id,),
        candidate_fact_ids=((candidate.fact_id,) if candidate else ()),
        message=message,
        field=source.source.field,
        span=source.source.span,
        dimension=source.dimension,
    )


def source_idempotence_issues(contract: PromptContract) -> list[ComplianceIssue]:
    source = apply_contract_revisions(contract.fields.get("Draft", ""), contract)
    return candidate_delta_issues(contract, source)


def apply_contract_revisions(base: str, contract: PromptContract) -> str:
    """Apply high-confidence targeted revisions to an immutable source base."""

    result = str(base or "")
    active = contract.active_facts()
    superseded = [fact for fact in contract.facts if fact.polarity == "superseded"]
    for revision in [fact for fact in active if fact.polarity == "replaced"]:
        prior = next(
            (
                fact for fact in superseded
                if fact.dimension == revision.dimension
                and _fact_key(fact) == _fact_key(revision)
            ),
            None,
        )
        if revision.dimension == "count" and prior is not None:
            words = {
                0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
                6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
            }
            new_count = int(revision.value)
            entity = _semantic_entity(revision.entity_id)
            head = canonical_head(entity)
            plural_head = {
                "woman": "women", "man": "men", "person": "people", "child": "children",
            }.get(head, head + ("es" if head.endswith(("s", "x", "ch", "sh")) else "s"))
            descriptor = entity.rsplit(" ", 1)[0] if " " in entity else ""
            rendered_entity = " ".join(value for value in (descriptor, head if new_count == 1 else plural_head) if value)
            old_entity = _semantic_entity(prior.entity_id)
            old_head = canonical_head(old_entity)
            entity_pattern = re.escape(old_entity).replace(r"\ ", r"\s+")
            result = re.sub(
                rf"(?i)\b(?:exactly\s+|only\s+)?(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+{entity_pattern}(?:s|es)?\b",
                f"{words.get(new_count, str(new_count))} {rendered_entity}",
                result,
                count=1,
            )
            if new_count != 1:
                result = re.sub(r"(?i)\b(stands|sits|rests|lies|appears|remains)\b", lambda match: match.group(1)[:-1], result, count=1)
                result = re.sub(r"(?i)\bis\b", "are", result, count=1)
        elif revision.dimension == "camera" and prior is not None:
            old_text = prior.source.text
            replacement = str(revision.value)
            if replacement in {"front view", "rear view", "top-down", "low-angle"}:
                replacement += " view" if not replacement.endswith("view") else ""
            result = re.sub(re.escape(old_text), replacement, result, count=1, flags=re.IGNORECASE)
        elif revision.dimension == "position" and prior is not None:
            result = re.sub(
                rf"(?i)\b{re.escape(str(prior.value))}\b",
                str(revision.value),
                result,
                count=1,
            )

    for exclusion in [
        fact for fact in active
        if fact.dimension == "exclusion" and fact.source.current_pass_only
    ]:
        head = canonical_head(str(exclusion.entity_id))
        if not head:
            continue
        aliases = {head, head + "s"}
        if head == "woman":
            aliases.add("women")
        elif head == "man":
            aliases.add("men")
        elif head == "person":
            aliases.add("people")
        elif head == "child":
            aliases.add("children")
        noun = "(?:" + "|".join(re.escape(value) for value in sorted(aliases, key=len, reverse=True)) + ")"
        result = re.sub(rf"(?i)\s+with\s+(?:[^,.;]*\s+)?{noun}\b", "", result)
        result = re.sub(rf"(?i)\b{noun}\b[^.!?;]*(?:[.!?;]|$)", "", result)
    result = re.sub(r"\s{2,}", " ", result)
    result = re.sub(r"\s+([,.;])", r"\1", result)
    return result.strip()


def revision_resolution_messages(contract: PromptContract) -> list[str]:
    messages: list[str] = []
    active_revisions = [fact for fact in contract.active_facts() if fact.polarity == "replaced"]
    for revision in active_revisions:
        prior = next(
            (
                fact for fact in contract.facts
                if fact.polarity == "superseded"
                and fact.dimension == revision.dimension
                and _fact_key(fact) == _fact_key(revision)
            ),
            None,
        )
        if prior is not None:
            messages.append(
                f"{revision.source.field} replaced {prior.source.field} "
                f"{revision.dimension} for {_semantic_entity(revision.entity_id) or revision.dimension}"
            )
    for exclusion in [
        fact for fact in contract.active_facts("exclusion")
        if fact.source.current_pass_only
    ]:
        draft = contract.fields.get("Draft", "")
        if _positive_occurrence(str(exclusion.entity_id), draft):
            messages.append(
                f"{exclusion.source.field} removed Draft object "
                f"{_semantic_entity(exclusion.entity_id)}"
            )
    return list(dict.fromkeys(messages))


def validate_continuation_sentences(
    base: str,
    continuation: str,
    source_contract: PromptContract,
) -> tuple[str, list[ComplianceIssue]]:
    """Keep safe sentences while discarding only hard-delta additions."""

    kept: list[str] = []
    rejected: list[ComplianceIssue] = []
    base_contract = compile_prompt_contract({"Candidate": base})
    base_entities = {
        (_semantic_entity(item.entity_id), item.kind)
        for item in base_contract.mentions
        if item.entity_id and item.kind in {"person", "group", "object", "body_part"}
    }
    base_relations = {
        (item.predicate, _semantic_entity(item.actor_id), _semantic_entity(item.receiver_id), item.body_target_id, item.phase)
        for item in base_contract.relations
        if item.confidence >= 0.85
    }
    for sentence in re.findall(r"[^.!?]+[.!?]?", continuation):
        cleaned = sentence.strip()
        if not cleaned:
            continue
        candidate = " ".join(value for value in (base, *kept, cleaned) if value)
        issues = candidate_delta_issues(source_contract, candidate)
        candidate_contract = compile_prompt_contract({"Candidate": candidate})
        base_ambiguity_budget = sum(
            issue.code == "reference.ambiguous"
            for issue in base_contract.issues
        )
        candidate_ambiguities = [
            issue for issue in candidate_contract.issues
            if issue.code == "reference.ambiguous"
        ]
        for index, ambiguity in enumerate(candidate_ambiguities):
            if index >= base_ambiguity_budget:
                issues.append(
                    replace(
                        ambiguity,
                        code="recovery.reference_ambiguous",
                        severity=SEVERITY_HARD,
                        origin="recovery",
                    )
                )
        for mention in candidate_contract.mentions:
            signature = (_semantic_entity(mention.entity_id), mention.kind)
            if (
                mention.entity_id
                and mention.kind in {"person", "group", "body_part"}
                and signature not in base_entities
                and mention.confidence >= 0.85
            ):
                issues.append(
                    ComplianceIssue(
                        code="recovery.entity_added",
                        severity=SEVERITY_HARD,
                        origin="recovery",
                        source_fact_ids=(),
                        candidate_fact_ids=(mention.mention_id,),
                        message=f"Creative continuation added {mention.kind}: {mention.canonical_name}",
                        field="Candidate",
                        span=mention.source.span,
                        dimension="entity",
                    )
                )
        for relation in candidate_contract.relations:
            signature = (
                relation.predicate,
                _semantic_entity(relation.actor_id),
                _semantic_entity(relation.receiver_id),
                relation.body_target_id,
                relation.phase,
            )
            if relation.confidence >= 0.85 and signature not in base_relations:
                issues.append(
                    ComplianceIssue(
                        code="recovery.relation_added",
                        severity=SEVERITY_HARD,
                        origin="recovery",
                        source_fact_ids=(),
                        candidate_fact_ids=(relation.relation_id,),
                        message=f"Creative continuation added relation: {relation.predicate}",
                        field="Candidate",
                        span=relation.source.span,
                        dimension="relation",
                    )
                )
        new_hard = [issue for issue in issues if issue.severity in {SEVERITY_HARD, SEVERITY_BLOCKER}]
        if new_hard:
            rejected.extend(new_hard)
        else:
            kept.append(cleaned)
    return " ".join(kept), _dedupe_issues(rejected)


LEGACY_HARD_RULES: tuple[tuple[str, str], ...] = tuple(
    (prefix, "legacy." + re.sub(r"[^a-z0-9]+", "_", prefix.casefold()).strip("_"))
    for prefix in (
        "Final prompt is empty", "Final prompt contains multiple lines", "Visible prompt format",
        "Camera viewpoint conflict", "Internal prompt guidance leaked", "Forbidden syntax matched",
        "FLUX.2 Klein positive-prompt contract", "Contradictory terms", "Hand-use contradiction",
        "Selected visual mode missing or changed", "Requested medium missing or changed",
        "Intent drift risk", "Explicit user directives missing", "Count contract", "Spatial contract",
        "Excluded content appears positively", "Unexpected output language/script",
        "Multi-panel story structure", "Missing required concepts", "Missing weighted visual emphasis",
        "Missing quoted rendered text", "Requested focus not represented", "Story element contract",
        "Creative development contract", "Variation structure", "Krea settings", "Krea controls",
        "Generator controls", "Content format", "Safe-for-work contract violated",
        "Multi-person role ambiguity", "Gender identity contract", "Person/body ownership contract",
        "Adult toy object contract", "Inserted object/body contact contract",
        "FLUX cross-subject anatomy binding contract", "Penile ventral orientation contract",
        "Semen origin contract",
        "Unrequested gender/anatomy traits", "Explicit support participant contract",
        "Untranslated explicit adult slang", "Explicit adult grammar contract",
        "NSFW scene fidelity contract", "Sexual content involving an underage or ambiguous-age subject",
        "Krea Official unsupported main addition", "Krea Official detailed-input contract",
    )
)


def legacy_compliance_issue(message: str) -> ComplianceIssue:
    normalized = re.sub(r"^Variation\s+\d+\s*:\s*", "", str(message or ""))
    for prefix, code in LEGACY_HARD_RULES:
        if normalized.startswith(prefix):
            return ComplianceIssue(code, SEVERITY_HARD, "candidate", (), (), str(message))
    return ComplianceIssue("legacy.advisory", SEVERITY_ADVISORY, "candidate", (), (), str(message))


def structure_legacy_issues(values: Iterable[str | ComplianceIssue]) -> list[IssueText]:
    result: list[IssueText] = []
    for value in values:
        if isinstance(value, ComplianceIssue):
            result.append(issue_text(value))
        elif isinstance(value, IssueText):
            result.append(value)
        else:
            result.append(issue_text(legacy_compliance_issue(str(value))))
    return result


def _dedupe_issues(values: Iterable[ComplianceIssue]) -> list[ComplianceIssue]:
    result: list[ComplianceIssue] = []
    seen: set[tuple[object, ...]] = set()
    for value in values:
        key = (value.code, value.message, value.field, value.span)
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result
