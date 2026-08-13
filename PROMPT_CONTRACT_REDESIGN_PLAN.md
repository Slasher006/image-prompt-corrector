# Prompt Contract and Validator Redesign Plan

Status: Implemented locally; configured-model live replay blocked by LM Studio model-load failure
Created: 2026-08-02
Scope: Prompt Corrector, FLUX Image Edit, Comic Story, Meme Creator, shared validation, and GUI diagnostics

Implementation verification (2026-08-02):

- Added the five proposed shared modules and routed backend, adult-scene, recovery, and GUI preflight paths through typed compatibility adapters.
- Added structured provenance/severity, targeted field revisions, entity/coreference graphs, scoped scene dimensions and relations, source-relative deltas, idempotent revision rendering, and sentence-level immutable recovery.
- Added regression coverage for the matrix families in this plan; the complete offscreen suite passes 667 tests.
- Python compilation, Python 3.10 AST parsing, and `git diff --check` pass.
- The isolated saved-path replay reached LM Studio, but the configured `huihui-qwen3-vl-4b-instruct-abliterated@q8_0` engine exited before becoming healthy and LM Studio returned HTTP 400. No settings, results, or ComfyUI state were written by the replay.

## 1. Goal

Replace the current collection of partially overlapping regex validators with one provenance-aware prompt contract system that can:

- distinguish people, groups, objects, body parts, cameras, and scene scopes;
- resolve only high-confidence references such as `it`, `its`, `they`, `them`, `their`, `this`, and `that`;
- preserve explicit identities, actors, receivers, ownership, contacts, counts, exclusions, positions, reactions, and outcomes;
- understand current-pass revisions instead of treating them as contradictions;
- reject only genuine policy violations or high-confidence candidate changes;
- treat inherited parser uncertainty as advisory rather than terminal;
- return a valid source-preserving result without unnecessary full-prompt rewrite loops;
- keep all private contract data out of the visible image prompt.

This is an architectural repair. It must not be implemented as another growing list of word-specific exceptions.

## 2. Current Verified Baseline

The implementation should begin from the current local worktree without resetting or discarding existing changes.

Current verified behavior:

- 634 offscreen tests pass.
- Python compilation and Python 3.10 AST parsing pass.
- The current saved-prompt LM Studio replay completes in two model calls.
- That replay has zero preflight conflicts, zero final hard issues, and zero advisories.
- Source-authored person ambiguity is advisory; a genuinely new ambiguity remains hard.
- Explicit high-confidence exclusions and exact cross-field contradictions remain blocking.
- A validated deterministic fidelity fallback can be selected before another full rewrite when Audit is enabled.

The redesign must preserve these gains while covering the remaining gaps below.

## 3. Confirmed Remaining Gaps

### 3.1 Entity and coreference gaps

- Object references `it`, `its`, `this`, and `that` are not resolved or validated.
- `resolve_unambiguous_multi_person_pronouns()` currently returns the prompt unchanged.
- Clear generic groups such as `Two people ... they ... them` can be falsely flagged.
- Same-gender ambiguity can be missed, especially after a collective-scene shortcut.
- Repeated same-gender roles can be mistaken for one unique gender referent.
- `they`, `them`, and `their` can be accepted when three or more antecedents are possible.
- Singular `they` is not distinguished from plural-group `they` when several people are present.
- Common plural roles such as doctors and patients are not consistently represented.
- Reducing an existing ambiguity can be misclassified as introducing a new hard ambiguity because diagnostics are compared as whole strings.

### 3.2 Field authority and revision gaps

- Draft, Story, Goal, Focus, Instructions, Feedback, Concepts, Weighted terms, and Visual direction do not yet have dimension-aware authority rules.
- `Change one red car to two red cars` is treated as a count contradiction.
- `Replace POV with over-the-shoulder` is treated as a camera contradiction.
- `Remove all flowers` is not compiled into an enforceable current-pass removal.
- Generation Feedback is passed to the model but is not part of the final hard contract comparison.
- A current revision cannot supersede only the matching older fact while retaining unrelated source facts.

### 3.3 Count, exclusion, camera, and spatial gaps

- Irregular equivalents such as `woman/women` and `person/people` do not share one canonical entity ID.
- Coordinated counts can be partially parsed.
- A trailing visual verb can become part of the counted entity, such as `pale eye gleaming`.
- Coordinated exclusion lists can lose later members.
- Exclusions do not consistently canonicalize `child/children`, `human/person`, and similar aliases.
- A same-field high-confidence position contradiction can be skipped.
- Inverted word order can lose the spatial subject.
- Spatial comparisons do not always preserve the reference object.
- Relation synonyms such as `under/beneath` and `over/above` are incomplete.
- Primary camera, reflected camera, inset camera, lighting direction, and occluder wording share one lexical camera scope.
- Negative camera guidance can be mistaken for a positive camera request.
- Mirror wording can create a false primary-camera conflict.

### 3.4 Actor, receiver, ownership, and adult-scene relation gaps

- Passive voice can reverse actor and receiver.
- A role may not be detected when another clause appears before its action.
- Body ownership is stored by gender category rather than by individual entity.
- Two same-gender people can exchange body-part ownership without detection.
- Mere omission of redundant ownership wording can be treated the same as an explicit ownership swap.
- Body and contact targets are detected scene-wide instead of being bound to an action.
- Environmental `rope`, clothing cuffs, or a blindfold prop can be mistaken for a sexual act.
- Camera framing of a chest can be mistaken for sexual contact.
- A mountain `peak` or ordinary `release` can be mistaken for a sexual outcome.
- Natural act wording is unevenly recognized because act detection is lexical rather than relation-based.
- Reaction validation uses sentence-local token co-occurrence rather than an owner and causal relation.
- A valid cross-sentence cause can be rejected.
- An unrelated role and cause word in the same sentence can incorrectly satisfy reaction binding.
- Source-existing multi-phase wording can make the unchanged source reject itself in Single Image mode.

### 3.5 Validation and recovery gaps

- Final issue severity is still derived from human-readable string prefixes.
- Final validators reconstruct source meaning from concatenated prose instead of consuming the compiled contract facts.
- Source idempotence is not universal.
- Candidate changes are not always compared as fact-level deltas.
- Early source-preserving fallback selection currently depends on Audit being enabled.
- With Audit disabled, an avoidable full-prompt repair can still run before fallback.
- Immutable continuation sanitation checks only a subset of possible new violations.
- New participants, acts, objects, contacts, outcomes, or ownership changes can survive continuation sanitation until the final gate.
- Repair diagnostics expose repeated intermediate failures even when recovery succeeds.

## 4. Required Design Principles

### 4.1 Source idempotence

Once genuine input conflicts and policy violations have passed preflight, a mechanically normalized source-preserving candidate must not fail a terminal fidelity check for preserving that same source.

Invariant:

```text
accepted source -> normalized fidelity base -> zero source-derived terminal deltas
```

### 4.2 Candidate-delta validation

Hard fidelity validation must answer one question:

> What high-confidence fact did the candidate add, remove, reverse, or reassign relative to the active source contract?

It must not reject a candidate merely because a heuristic parser is uncertain in both source and candidate.

### 4.3 Provenance and field authority

Every fact must retain its source field, text span, current-pass lifetime, confidence, polarity, and scope. Revision directives must supersede only the matching lower-authority facts.

### 4.4 High-confidence deterministic repair only

Automatic pronoun or entity rewriting is allowed only when there is one high-confidence antecedent. Ties remain advisory in the source and hard only when introduced by a candidate.

### 4.5 Relation scope instead of keyword bags

Actions, body targets, objects, reactions, outcomes, cameras, and positions must be represented as scoped relations. A noun elsewhere in the scene must not become an action target merely because a sexual or physical action exists somewhere in the prompt.

### 4.6 Stable issue codes

Severity must come from structured issue codes and provenance, not message text. User-facing wording may change without changing validation behavior.

### 4.7 Safety remains independent and hard

The redesign must not weaken:

- the underage or ambiguous-age sexual-content boundary;
- mutual exclusion between Safe for work and Explicit adult;
- source-aware script handling;
- private-guidance and internal-schema leakage checks;
- explicit format, quoted rendered text, and non-overridable generator constraints.

## 5. Target Architecture

```text
Input fields and dedicated controls
                |
                v
      Field and revision compiler
                |
                v
     Entity and mention resolution
                |
                v
   Scoped fact and relation contract
                |
       +--------+---------+
       |                  |
       v                  v
 Genuine preflight     Model guidance
 blockers/advisories       |
       |                  v
       |             Candidate output
       |                  |
       +--------+---------+
                v
     Candidate fact-delta validation
                |
       +--------+---------+
       |                  |
       v                  v
 Valid candidate    Immutable safe recovery
       |                  |
       +--------+---------+
                v
      One natural visible prompt
```

### 5.1 Proposed modules

Create small focused modules instead of extending the monolith indefinitely:

- `prompt_contract.py`
  - contract data classes;
  - field authority and revision application;
  - stable issue codes and severity.
- `entity_resolution.py`
  - person, group, object, body-part, and pronoun mentions;
  - canonical entity IDs;
  - high-confidence antecedent resolution.
- `scene_relations.py`
  - actor/action/receiver/object/contact/ownership/reaction/outcome relations;
  - active/passive normalization;
  - clause and sentence scope.
- `scene_dimensions.py`
  - counts, exclusions, camera scopes, positions, spatial references, and visual phases.
- `contract_validation.py`
  - preflight conflict detection;
  - source-idempotence checks;
  - candidate-delta validation;
  - immutable continuation validation.

Existing entry points in `krea_prompt_corrector.py`, `nsfw_scene_contract.py`, and `krea_prompt_gui.py` should call these modules through compatibility adapters during migration.

### 5.2 Core data model

The exact names may change, but the contract needs equivalent information:

```python
@dataclass(frozen=True)
class SourceRef:
    field: str
    span: tuple[int, int]
    text: str
    authority: int
    current_pass_only: bool = False


@dataclass(frozen=True)
class EntityMention:
    mention_id: str
    entity_id: str | None
    kind: str              # person, group, object, body_part, camera, setting
    canonical_name: str
    number: str            # singular, plural, unknown
    attributes: tuple[str, ...]
    source: SourceRef
    confidence: float


@dataclass(frozen=True)
class ContractFact:
    code: str
    entity_id: str | None
    value: object
    polarity: str          # required, excluded, replaced
    scope_id: str
    source: SourceRef
    confidence: float


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


@dataclass(frozen=True)
class ComplianceIssue:
    code: str
    severity: str          # blocker, hard, advisory
    origin: str            # policy, input, source, candidate, recovery
    source_fact_ids: tuple[str, ...]
    candidate_fact_ids: tuple[str, ...]
    message: str
```

### 5.3 Field authority model

Implement a dimension-aware authority resolver rather than one universal numeric overwrite.

Suggested order:

1. Non-overridable policy and content-format constraints.
2. Explicit current-pass revision directives that identify a target fact with `change`, `replace`, `remove`, `instead`, or equivalent grammar.
3. Dedicated UI controls for their own dimensions, such as Camera, Format, Mode, and Visual direction.
4. Explicit Model Instructions.
5. Goal, Focus, Concepts, and Weighted terms for their declared dimensions.
6. Draft and Story as the source baseline.

Rules:

- A revision replaces only the identified dimension and entity.
- Unrelated source facts remain active.
- Generation Feedback is active for the current correction pass only.
- If a revision target cannot be identified confidently, log an advisory and let the model repair it without preflight blocking.
- Two unresolved facts at the same active authority level block only when they have the same entity, dimension, scope, and high confidence.

## 6. Phased Implementation

### Phase 0: Freeze behavior and build the audit corpus

Purpose: prevent the redesign from losing existing correct behavior.

Tasks:

- Record the current 634-test baseline.
- Add synthetic fixtures for every confirmed gap in Section 3.
- Add anonymized structural fixtures derived from saved Activity failures; never commit private prompt history or settings.
- Add a helper that prints structured fact and delta traces only in tests or debug diagnostics.
- Add invariant tests before replacing legacy logic.
- Record current full-pipeline model-call counts with mocked and live LM Studio paths.

Acceptance criteria:

- Every confirmed gap has at least one failing regression test.
- Existing tests remain unchanged unless they encode behavior explicitly replaced by this plan.
- No private prompt, token, local path, or internal contract text enters committed fixtures.

### Phase 1: Introduce typed contracts and field authority

Purpose: create one source of truth for all later validators.

Tasks:

- Add `SourceRef`, `ContractFact`, `RelationFact`, and `ComplianceIssue`.
- Compile every input field separately with provenance.
- Define dimension-aware authority and current-pass revision behavior.
- Compile `remove`, `replace`, `change`, and `instead` directives.
- Make Feedback override matching Draft facts for one pass.
- Make dedicated controls authoritative within their dimensions.
- Keep a compatibility adapter that emits current Activity message text.
- Run the typed compiler in shadow mode beside current preflight.

Acceptance criteria:

- `Exactly one red car` plus Feedback `change it to two red cars` yields one active count fact: two.
- `POV` plus `replace POV with over-the-shoulder` yields one active primary-camera fact.
- `Remove flowers` creates an active exclusion for the current pass.
- Unrelated Draft facts survive every targeted revision.
- Policy and format constraints cannot be overridden.

### Phase 2: Build the shared entity and mention graph

Purpose: resolve people, groups, objects, body parts, and references consistently.

Tasks:

- Create entity IDs from noun mentions rather than gender categories or raw noun heads.
- Canonicalize irregular number forms and safe aliases.
- Represent explicit groups and group membership.
- Track repeated same-gender and same-role entities separately.
- Add common plural person-role morphology without maintaining an exhaustive role list.
- Add mentions for `it`, `its`, `this`, `that`, `they`, `them`, `their`, `he`, `him`, `his`, `she`, `her`, and `hers`.
- Distinguish singular `they` from plural `they` using entity number and candidate antecedents.
- Remove the collective-scene early return that hides individual ambiguity.
- Replace the no-op pronoun resolver with a high-confidence resolver.
- Emit an advisory instead of rewriting when more than one antecedent remains plausible.

Acceptance criteria:

- `A cup and a book. She places it...` is advisory because two object antecedents exist.
- `A cup. She places it...` resolves to the cup.
- `Two women... she...` remains ambiguous unless one woman is explicitly identified.
- `The first woman... the second woman...` preserves two distinct entities and ownership.
- `A nonbinary person stands alone. They...` resolves as singular.
- `A doctor, patient, and nurse... they...` does not pass unless the referenced group is explicit.
- Reducing ambiguity cannot increase severity.

### Phase 3: Add scoped actor, receiver, ownership, contact, and reaction relations

Purpose: replace role and adult-scene keyword bags with relation facts.

Tasks:

- Normalize active and passive voice to the same actor/action/receiver relation.
- Bind actions across subordinate clauses and pronoun references.
- Attach body parts to entity IDs, including same-gender individuals.
- Separate missing redundant ownership wording from an explicit ownership swap.
- Bind action objects and body targets only inside the action relation.
- Bind reactions to an owner and a cause relation.
- Allow explicit adjacent-sentence causal links.
- Reject coincidental role/cause token co-occurrence without a relation.
- Scope adult acts, objects, contacts, fluids, and outcomes to their relations.
- Treat environmental props and camera framing as non-contact unless relation evidence says otherwise.
- Normalize visual phases while preserving source-idempotence.

Acceptance criteria:

- Active and passive paraphrases produce the same relation.
- Reversing actor and receiver produces a hard candidate delta.
- Swapping body ownership between two women is detected.
- Omitting a redundant repeated owner label is advisory unless it creates real ambiguity.
- Climbing rope does not become bondage.
- Camera-framed chest does not become a contact target.
- Mountain peak does not become climax.
- A cross-sentence explicit cause binds the reaction.
- An unchanged accepted source never rejects itself for its original relations.

### Phase 4: Replace count, exclusion, camera, and spatial heuristics with scoped facts

Purpose: complete the non-person scene dimensions using the same contract system.

Tasks:

- Canonicalize irregular count entities and aliases.
- Parse coordinated counts and exclusion lists at clause level.
- Separate entity descriptors from trailing predicates.
- Preserve distinct descriptors such as red car and blue car without collapsing them.
- Parse exclusions with polarity and revision scope.
- Parse primary, reflection, inset, background, and reference-image camera scopes.
- Distinguish camera direction from lighting direction and occlusion wording.
- Store spatial subject, relation, reference object, and scope independent of word order.
- Canonicalize relation synonyms.
- Compare same-field facts only when entity/reference identity is high-confidence.

Acceptance criteria:

- `woman/women` and `person/people` share canonical count identities.
- Coordinated exact counts produce all intended facts.
- Coordinated exclusions preserve every list member.
- A reflected rear view does not conflict with a primary front view.
- Top-down lighting does not become a camera fact.
- Inverted spatial wording produces the same subject/relation/reference triple.
- Left-of-tree and right-of-house are allowed simultaneously.
- A true same-entity, same-reference position reversal remains blocking.

### Phase 5: Convert final validation to source-relative fact deltas

Purpose: eliminate prefix-based severity and source self-rejection.

Tasks:

- Validate candidates against the compiled active contract, not concatenated source prose.
- Compute added, removed, reassigned, reversed, and unresolved fact deltas.
- Assign severity from issue code, origin, confidence, and user authority.
- Remove `HARD_COMPLIANCE_PREFIXES` after compatibility parity is proven.
- Make inherited uncertainty advisory by fact identity, not diagnostic string equality.
- Enforce the source-idempotence invariant across all format and content modes.
- Route Advisory versus Hard versus Blocker consistently to GUI and CLI.

Suggested severity rules:

- Blocker:
  - non-overridable policy violation;
  - genuine same-layer input contradiction that cannot be resolved by precedence.
- Hard:
  - candidate-introduced or candidate-reversed high-confidence identity, actor, receiver, count, exclusion, ownership, contact, participant, act, fluid, outcome, quoted text, format, or camera fact.
- Advisory:
  - inherited source uncertainty;
  - low-confidence antecedent or parse;
  - style, grammar, length, creative-depth, or redundant ownership quality concern.

Acceptance criteria:

- Human-readable message wording can change without changing severity.
- A candidate that resolves only part of a source ambiguity does not become harder.
- A candidate that introduces a different ambiguity remains hard.
- Accepted sources produce zero source-derived terminal deltas.
- True actor, receiver, ownership, count, exclusion, and contact changes remain hard.

### Phase 6: Unify repair and fallback around an immutable validated base

Purpose: stop whole-prompt rewrite loops from creating new failures.

Tasks:

- Build the deterministic fidelity base immediately after contract compilation.
- Validate the base before model inference or repair selection.
- Make the valid base available regardless of Audit setting.
- Keep correction and optional audit as candidate improvements over the base.
- Remove additional full-prompt repair when a valid base already exists.
- Replace broad repair with fact-targeted edits or sentence-level immutable additions.
- Validate every proposed continuation sentence against the complete candidate-delta validator.
- Discard only the sentence that introduces a hard delta.
- Aggregate recovered intermediate issues into one concise Activity event.

Acceptance criteria:

- Default paths use at most correction plus the requested audit when a valid base exists.
- Audit disabled still receives the same early fallback protection.
- A valid fallback is never rejected for inherited source uncertainty.
- Creative continuation cannot add a participant, act, object, contact, ownership swap, fluid, outcome, phase, or ambiguous reference.
- Successful recovery does not instruct the user to edit an already recovered prompt.

### Phase 7: Update GUI highlighting and Activity diagnostics

Purpose: make contract feedback understandable without recreating user-edit loops.

Tasks:

- Highlight blockers in red and advisories in amber.
- Highlight the exact source field and span that produced each fact.
- Show revision resolution as `Feedback replaced Draft camera`, not as a conflict.
- Separate `Input blocker`, `Source advisory`, `Candidate hard delta`, and `Automatic recovery` events.
- Keep Activity workspace-scoped.
- Do not expose private contract structures, IDs, or policy scaffolding in the result prompt.
- Keep the user-edit request only for unresolved high-confidence blockers.

Acceptance criteria:

- Clear source ambiguity continues automatically with an optional amber highlight.
- True conflicting active facts stop once before any model call.
- Activity does not repeat the same issue at correction, audit, repair, and fallback stages.
- Error guidance names the correct field and semantic dimension.

### Phase 8: Remove shadow logic and complete cross-mode verification

Purpose: retire legacy paths only after behavior parity and improved precision are proven.

Tasks:

- Compare typed and legacy validator outputs over the synthetic corpus.
- Review every mismatch; classify it as intended improvement, regression, or uncertain advisory.
- Switch Single Image and FLUX Image Edit first.
- Apply shared entity and relation logic to Comic Story and Meme Creator while preserving their format-specific contracts.
- Remove duplicated GUI/backend preflight extraction.
- Remove dead regex tables and compatibility adapters only after full coverage.
- Run compile, Python 3.10 AST, diff, full offscreen, and live LM Studio verification.
- Do not commit, push, publish, or deploy without explicit authorization.

Acceptance criteria:

- All modes use one contract compiler and one severity system.
- No visible prompt contains internal schemas, IDs, JSON, or repair labels.
- Full tests and selected live saved-prompt replays pass.
- The current saved prompt completes with no false blocker and no avoidable repair cycle.

## 7. Regression Matrix

The implementation is incomplete until the following cases are represented in tests.

### 7.1 Coreference

- one object plus `it`;
- two objects plus ambiguous `it`;
- possessive `its`;
- demonstratives `this` and `that`;
- explicit two-person group plus `they/them/their`;
- generic two-person group plus collective action;
- three-person scene with an underspecified `they`;
- singular nonbinary `they` alone;
- singular nonbinary `they` with another person present;
- two same-gender people with `she/her` or `he/his`;
- first/second role labels;
- repeated identical roles;
- plural occupation and relationship roles;
- ambiguity reduced, preserved, resolved, and newly introduced.

### 7.2 Revisions and provenance

- Feedback changes exact count;
- Feedback removes an object;
- Instructions replace camera;
- Goal changes only style while retaining scene facts;
- dedicated Camera overrides an older Draft camera;
- uncertain revision target becomes advisory;
- revision does not erase unrelated source constraints.

### 7.3 Counts and exclusions

- irregular singular/plural pairs;
- numeric and word-number equivalence;
- coordinated exact counts;
- count phrase followed by a visual predicate;
- red and blue same-type entities;
- coordinated exclusion lists;
- exclusion aliases and irregular morphology;
- genuine exclusion reintroduction remains hard.

### 7.4 Camera and space

- primary versus reflected view;
- primary versus inset view;
- camera versus lighting direction;
- camera versus occluder wording;
- negated camera instruction;
- inverted spatial word order;
- same subject with different reference objects;
- same subject/reference with reversed relation;
- same-field contradiction;
- relation synonyms.

### 7.5 Actors, ownership, contacts, and reactions

- active and passive voice parity;
- subordinate-clause action;
- same-gender actor/receiver reversal;
- same-gender body ownership swap;
- redundant owner omission;
- explicit ownership reassignment;
- environmental rope and clothing cuffs;
- camera-framed anatomy without contact;
- prop/body noun outside the action clause;
- mountain peak and ordinary release;
- same-sentence cause;
- adjacent-sentence cause;
- unrelated role and cause words;
- source-existing and candidate-added outcomes;
- source-existing multi-phase wording.

### 7.6 Recovery

- Audit on and off;
- small and non-small model paths;
- initial candidate invalid, audit valid;
- both model candidates invalid, base valid;
- base valid with advisories;
- continuation adds safe environment detail;
- continuation adds one forbidden participant or contact;
- continuation contains both safe and unsafe sentences;
- recovery diagnostics are aggregated.

## 8. Test and Verification Commands

Run focused tests after each phase, followed by the full baseline:

```bash
python3 -m py_compile krea_prompt_corrector.py krea_prompt_gui.py nsfw_scene_contract.py
python3 -m unittest -q tests.test_krea_prompt_corrector.PromptCorrectorTests
python3 -m unittest -q tests.test_nsfw_scene_contract
QT_QPA_PLATFORM=offscreen python3 -m unittest -q tests.test_krea_prompt_gui.PromptCorrectorGuiTests
QT_QPA_PLATFORM=offscreen python3 -m unittest discover -q
git diff --check
```

Also parse changed Python with Python 3.10 grammar before declaring completion.

Live verification must:

- use the currently configured local LM Studio provider and model;
- reproduce the saved GUI path, including camera injection and support fields;
- count model calls;
- capture only diagnostics and result metadata unless prompt text is explicitly needed;
- avoid ComfyUI execution and settings/result overwrite during isolated tests;
- distinguish live model proof from mocked or static validation.

## 9. Quality Gates and Metrics

The redesign is ready only when all gates pass.

### Precision gates

- Zero false blockers across the confirmed regression matrix.
- Every true high-confidence contradiction fixture blocks before model inference.
- Every candidate-introduced actor, receiver, ownership, participant, act, contact, count, exclusion, fluid, or outcome change remains hard.
- Low-confidence parser uncertainty is advisory.

### Idempotence gates

- Every accepted synthetic source passes normalized fallback validation with zero terminal source-derived deltas.
- Reducing or resolving ambiguity never increases severity.
- Active/passive and safe semantic paraphrases produce equivalent relation facts.

### Recovery gates

- A valid deterministic base is available with Audit on or off.
- No third full-prompt rewrite occurs when correction plus audit fail but the base is valid.
- Unsafe continuation sentences are discarded without discarding safe sentences.
- Successful recovery returns a usable prompt instead of a user-edit loop.

### UX gates

- Activity reports one semantic issue once.
- Advisory highlights do not block correction.
- Blockers name their field, entity, dimension, and conflicting value.
- Visible prompts remain natural language with no internal contract leakage.

## 10. Migration and Rollback Strategy

- Implement on a dedicated local branch when authorized.
- Keep legacy validators callable during shadow mode.
- Add a compatibility adapter from typed issues to current Activity messages.
- Switch one validation family at a time.
- Do not delete legacy code until typed behavior passes the full regression matrix and live replay.
- Keep every phase independently revertible.
- Preserve the current deterministic fallback until the new immutable-base path has parity.
- Preserve all unrelated user work in the dirty worktree.
- Do not alter saved settings or prompt history during tests unless a test uses an isolated temporary file.

## 11. Definition of Done

The full project is complete when:

- all confirmed gaps in Section 3 have regression coverage and pass;
- the shared entity graph handles people, groups, objects, body parts, and pronouns;
- targeted revisions override only matching older facts;
- camera, count, exclusion, and spatial facts carry identity and scope;
- actor, receiver, ownership, contact, reaction, and outcome relations are entity-bound;
- validation severity uses structured codes and provenance;
- accepted sources are idempotent;
- recovery uses an immutable validated base regardless of Audit setting;
- Single Image, FLUX Image Edit, Comic Story, and Meme Creator use the shared core without cross-workspace state leakage;
- the complete offscreen suite, compile checks, Python 3.10 AST checks, diff checks, and isolated live LM Studio replays pass;
- no internal contract data appears in visible prompts;
- no commit, push, publish, ComfyUI run, or settings overwrite occurs without explicit authorization.

## 12. Recommended Execution Order

Do not implement the gaps independently. Use this dependency order:

1. Phase 0: tests and audit corpus.
2. Phase 1: typed contract and field authority.
3. Phase 2: entity and mention graph.
4. Phase 3: actor/action/receiver/ownership/contact/reaction relations.
5. Phase 4: scoped counts, exclusions, cameras, and positions.
6. Phase 5: source-relative typed validation.
7. Phase 6: immutable-base recovery.
8. Phase 7: GUI and Activity behavior.
9. Phase 8: shadow comparison, legacy removal, and cross-mode verification.

This order prevents the project from returning to phrase-by-phrase patches and ensures each later validator consumes the same structured facts.
