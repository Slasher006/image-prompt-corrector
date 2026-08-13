import unittest

from contract_validation import (
    apply_contract_revisions,
    candidate_delta_issues,
    compile_prompt_contract,
    preflight_contract_issues,
    revision_resolution_messages,
    source_idempotence_issues,
    validate_continuation_sentences,
)
from entity_resolution import (
    canonical_entity_name,
    extract_entity_mentions,
    reference_ambiguities,
    resolve_references,
    rewrite_high_confidence_references,
)
from prompt_contract import (
    ComplianceIssue,
    SEVERITY_HARD,
    issue_is_hard,
    issue_text,
)
from scene_dimensions import (
    extract_camera_facts,
    extract_count_facts,
    extract_exclusion_facts,
    extract_spatial_facts,
)
from scene_relations import extract_relation_facts


class PromptContractCoreTests(unittest.TestCase):
    def test_structured_severity_does_not_depend_on_message_wording(self):
        first = ComplianceIssue(
            "candidate.count_changed", SEVERITY_HARD, "candidate",
            (), (), "Count changed",
        )
        second = ComplianceIssue(
            "candidate.count_changed", SEVERITY_HARD, "candidate",
            (), (), "Completely different user-facing wording",
        )
        self.assertTrue(issue_is_hard(issue_text(first)))
        self.assertTrue(issue_is_hard(issue_text(second)))

    def test_contract_trace_uses_typed_records_without_source_prose_dump(self):
        contract = compile_prompt_contract({"Draft": "Exactly two red cars stand together."})
        trace = contract.trace()
        self.assertEqual(trace["fields"], ("Draft",))
        self.assertEqual(trace["facts"][0]["dimension"], "count")
        self.assertNotIn("Exactly two red cars", str(trace["facts"]))

    def test_feedback_count_revision_supersedes_only_matching_fact(self):
        contract = compile_prompt_contract(
            {
                "Draft": "Exactly one red car stands beside exactly one blue bicycle.",
                "Feedback": "Change it to two red cars.",
            }
        )
        active = [(fact.entity_id, fact.value) for fact in contract.active_facts("count")]
        self.assertIn(("red car", 2), active)
        self.assertIn(("blue bicycle", 1), active)
        self.assertNotIn(("red car", 1), active)
        self.assertEqual(preflight_contract_issues(contract), [])

    def test_camera_revision_supersedes_draft_camera(self):
        contract = compile_prompt_contract(
            {"Draft": "POV shot of a woman.", "Instructions": "Replace POV with over-the-shoulder."}
        )
        active = [fact.value for fact in contract.active_facts("camera")]
        self.assertEqual(active, ["over-the-shoulder"])
        self.assertEqual(
            revision_resolution_messages(contract),
            ["Instructions replaced Draft camera for camera"],
        )

    def test_position_revision_supersedes_draft_position(self):
        contract = compile_prompt_contract(
            {
                "Draft": "One woman stands left of the tree.",
                "Feedback": "Move the woman to the right of the tree.",
            }
        )
        active = [
            (fact.entity_id, fact.value, fact.context)
            for fact in contract.active_facts("position")
            if fact.polarity == "replaced"
        ]
        self.assertEqual(active, [("woman", "right of", ("tree",))])
        revised = apply_contract_revisions(contract.fields["Draft"], contract)
        self.assertEqual(revised, "One woman stands right of the tree.")
        self.assertEqual(candidate_delta_issues(contract, revised), [])

    def test_feedback_removal_creates_current_pass_exclusion(self):
        contract = compile_prompt_contract(
            {"Draft": "A vase with pink flowers on a table.", "Feedback": "Remove all flowers."}
        )
        exclusion = contract.active_facts("exclusion")[0]
        self.assertEqual(exclusion.entity_id, "flower")
        self.assertTrue(exclusion.source.current_pass_only)
        self.assertEqual(apply_contract_revisions(contract.fields["Draft"], contract), "A vase on a table.")

    def test_accepted_revised_sources_are_idempotent(self):
        cases = (
            {"Draft": "Exactly one red car stands.", "Feedback": "Change it to two red cars."},
            {"Draft": "POV shot of a woman.", "Instructions": "Replace POV with over-the-shoulder."},
            {"Draft": "A vase with flowers. A lamp glows.", "Feedback": "Remove all flowers."},
        )
        for fields in cases:
            with self.subTest(fields=fields):
                self.assertEqual(source_idempotence_issues(compile_prompt_contract(fields)), [])

    def test_same_layer_true_count_conflict_blocks(self):
        contract = compile_prompt_contract(
            {"Draft": "Exactly one red car stands.", "Goal": "Exactly two red cars stand."}
        )
        issues = preflight_contract_issues(contract)
        self.assertEqual(issues[0].code, "input.count_conflict")
        self.assertEqual(issues[0].severity, "blocker")

    def test_dedicated_visual_direction_overrides_old_primary_camera(self):
        contract = compile_prompt_contract(
            {"Draft": "POV shot.", "Visual direction": "Over-the-shoulder view."}
        )
        self.assertEqual(preflight_contract_issues(contract), [])
        self.assertEqual(
            [fact.value for fact in contract.active_facts("camera")],
            ["over-the-shoulder"],
        )

    def test_dedicated_camera_field_overrides_injected_draft_camera_words(self):
        contract = compile_prompt_contract(
            {
                "Draft": "Point-of-view shot. Over-the-shoulder source wording.",
                "Camera": "Point-of-view shot.",
            }
        )
        self.assertEqual(preflight_contract_issues(contract), [])
        self.assertEqual(
            [fact.value for fact in contract.active_facts("camera")],
            ["point-of-view"],
        )

    def test_one_object_it_resolves_and_two_objects_stay_advisory(self):
        one = resolve_references(extract_entity_mentions("A cup. She places it on a table."))
        two = resolve_references(extract_entity_mentions("A cup and a book. She places it on a table."))
        one_it = next(item for item in one if item.canonical_name == "it")
        two_it = next(item for item in two if item.canonical_name == "it")
        self.assertIsNotNone(one_it.entity_id)
        self.assertIsNone(two_it.entity_id)
        self.assertIn(two_it, reference_ambiguities(two))

    def test_object_resolver_rewrites_only_high_confidence_object_reference(self):
        self.assertEqual(
            rewrite_high_confidence_references("A cup. She places it on a table."),
            "A cup. She places the cup on a table.",
        )
        ambiguous = "A cup and a book. She places it on a table."
        self.assertEqual(rewrite_high_confidence_references(ambiguous), ambiguous)

    def test_explicit_group_and_singular_nonbinary_they_resolve(self):
        group = compile_prompt_contract({"Draft": "Two people enter. They carry a torch."})
        singular = compile_prompt_contract({"Draft": "A nonbinary person stands alone. They smile."})
        self.assertFalse(any(issue.code == "reference.ambiguous" and 'They' in issue.message for issue in group.issues))
        self.assertFalse(any(issue.code == "reference.ambiguous" and 'They' in issue.message for issue in singular.issues))

    def test_three_explicit_people_resolve_as_plural_group(self):
        contract = compile_prompt_contract(
            {"Draft": "A doctor, a patient, and a nurse enter. They wait."}
        )
        self.assertFalse(any(issue.code == "reference.ambiguous" for issue in contract.issues))

    def test_same_gender_singular_reference_remains_ambiguous(self):
        contract = compile_prompt_contract({"Draft": "Two women enter. She carries a torch."})
        self.assertTrue(any('"She"' in issue.message for issue in contract.issues))

    def test_irregular_entity_forms_share_canonical_head(self):
        self.assertEqual(canonical_entity_name("women"), "woman")
        self.assertEqual(canonical_entity_name("people"), "person")
        self.assertEqual(canonical_entity_name("children"), "child")

    def test_coordinated_counts_and_predicate_boundary(self):
        facts = extract_count_facts(
            "Exactly two women and three men stand together. Exactly one pale eye gleaming."
        )
        self.assertEqual(
            [(fact.entity_id, fact.value) for fact in facts],
            [("woman", 2), ("man", 3), ("pale eye", 1)],
        )

    def test_coordinated_exclusions_keep_every_member_and_alias(self):
        facts = extract_exclusion_facts("No flowers, children, or human figures.")
        self.assertEqual(
            [fact.entity_id for fact in facts],
            ["flower", "child", "human figure"],
        )

    def test_camera_scopes_separate_primary_reflection_and_inset(self):
        facts = extract_camera_facts(
            "Primary front view. A mirror shows a rear view. An inset shows a top-down view."
        )
        self.assertEqual(
            [(fact.value, fact.scope_id) for fact in facts],
            [
                ("front view", "camera:primary"),
                ("rear view", "camera:reflection"),
                ("top-down", "camera:inset"),
            ],
        )

    def test_lighting_occluder_and_negated_camera_are_not_positive_camera_facts(self):
        self.assertEqual(extract_camera_facts("Top-down lighting shapes the room."), [])
        self.assertEqual(extract_camera_facts("The subject is viewed from behind a curtain."), [])
        self.assertEqual(extract_camera_facts("Avoid a rear view; use neutral framing."), [])

    def test_inverted_and_normal_spatial_order_match(self):
        normal = extract_spatial_facts("The woman is beneath the tree.")[0]
        inverted = extract_spatial_facts("Beneath the tree stands the woman.")[0]
        self.assertEqual(
            (normal.entity_id, normal.value, normal.context),
            (inverted.entity_id, inverted.value, inverted.context),
        )

    def test_same_subject_different_spatial_references_do_not_conflict(self):
        contract = compile_prompt_contract(
            {"Draft": "The woman is left of the tree and right of the house."}
        )
        self.assertEqual(preflight_contract_issues(contract), [])

    def test_active_and_passive_voice_produce_same_semantic_relation(self):
        active = extract_relation_facts("A doctor examines a patient.")[0]
        passive = extract_relation_facts("A patient is examined by a doctor.")[0]
        self.assertEqual(active.predicate, passive.predicate)
        self.assertTrue(active.actor_id.endswith(":doctor"))
        self.assertTrue(passive.actor_id.endswith(":doctor"))
        self.assertTrue(active.receiver_id.endswith(":patient"))
        self.assertTrue(passive.receiver_id.endswith(":patient"))

    def test_actor_receiver_reversal_is_hard_candidate_delta(self):
        source = compile_prompt_contract(
            {"Draft": "The first woman kisses the second woman."}
        )
        issues = candidate_delta_issues(
            source,
            "The second woman kisses the first woman.",
        )
        self.assertTrue(any(issue.code == "candidate.relation_reversed" for issue in issues))

    def test_same_gender_body_ownership_swap_is_hard(self):
        source = compile_prompt_contract(
            {
                "Draft": (
                    "The first woman raises the first woman's hand. "
                    "The second woman holds the second woman's wrist."
                )
            }
        )
        issues = candidate_delta_issues(
            source,
            (
                "The first woman raises the second woman's hand. "
                "The second woman holds the first woman's wrist."
            ),
        )
        self.assertEqual(
            {issue.code for issue in issues},
            {"candidate.ownership_reassigned"},
        )

    def test_environmental_props_and_camera_anatomy_are_not_contacts(self):
        for text in (
            "A woman climbs a rope beside clothing cuffs.",
            "A camera frames a woman's chest.",
        ):
            relations = extract_relation_facts(text)
            self.assertFalse(any(relation.predicate in {"bind", "cuff", "touch"} for relation in relations))

    def test_mountain_peak_and_ordinary_release_are_not_climax(self):
        contract = compile_prompt_contract(
            {"Draft": "A climber reaches a mountain peak before releasing a rope."}
        )
        self.assertNotIn("climax", [fact.value for fact in contract.active_facts("phase")])

    def test_cross_sentence_explicit_reaction_cause_is_bound(self):
        relations = extract_relation_facts(
            "A man holds a woman. She trembles because of his touch."
        )
        reaction = next(item for item in relations if item.predicate.startswith("reaction:"))
        self.assertIsNotNone(reaction.reaction_owner_id)
        self.assertIsNotNone(reaction.cause_relation_id)

    def test_candidate_count_and_exclusion_changes_are_hard_deltas(self):
        source = compile_prompt_contract(
            {"Draft": "Exactly two red cars stand in a clean studio. No flowers."}
        )
        issues = candidate_delta_issues(
            source,
            "Three red cars stand in a clean studio beside flowers.",
        )
        self.assertTrue(any(issue.code == "candidate.count_changed" for issue in issues))
        self.assertTrue(any(issue.code == "candidate.exclusion_added" for issue in issues))

    def test_reducing_source_ambiguity_never_increases_severity(self):
        source = compile_prompt_contract(
            {"Draft": "A cup and a book. She places it on a table."}
        )
        issues = candidate_delta_issues(
            source,
            "A cup and a book. The woman places the cup on a table.",
        )
        self.assertFalse(any(issue.code == "candidate.reference_ambiguous" for issue in issues))

    def test_new_multi_person_ambiguity_is_hard(self):
        source = compile_prompt_contract(
            {"Draft": "A doctor greets a patient in a clinic."}
        )
        issues = candidate_delta_issues(
            source,
            "A doctor greets a patient in a clinic. She waves.",
        )
        self.assertTrue(any(issue.code == "candidate.reference_ambiguous" for issue in issues))

    def test_new_plural_group_ambiguity_is_hard(self):
        source = compile_prompt_contract(
            {"Draft": "A woman waits in a studio."}
        )
        issues = candidate_delta_issues(
            source,
            "Two women stand together. Her coat moves.",
        )
        self.assertTrue(any(issue.code == "candidate.reference_ambiguous" for issue in issues))

    def test_explicit_collective_group_resolves_they(self):
        contract = compile_prompt_contract(
            {"Candidate": "A couple enters. They smile."}
        )
        self.assertFalse(any(issue.code == "reference.ambiguous" for issue in contract.issues))

    def test_two_explicit_roles_resolve_plural_group_references(self):
        for token, tail in (
            ("They", "smile"),
            ("them", "A camera follows them"),
            ("Their", "Their coats move"),
            ("theirs", "The coats are theirs"),
        ):
            sentence = tail if token.casefold() in tail.casefold() else f"{token} {tail}"
            contract = compile_prompt_contract(
                {"Candidate": f"A doctor enters with a patient. {sentence}."}
            )
            self.assertFalse(
                any(issue.code == "reference.ambiguous" for issue in contract.issues),
                token,
            )

    def test_singular_they_stays_ambiguous_with_two_possible_scopes(self):
        contract = compile_prompt_contract(
            {
                "Candidate": (
                    "A nonbinary person greets a woman. They smile."
                )
            }
        )
        self.assertTrue(any(issue.code == "reference.ambiguous" for issue in contract.issues))

    def test_new_it_ambiguity_between_objects_is_hard(self):
        source = compile_prompt_contract(
            {"Draft": "A cat sits beside a lamp."}
        )
        issues = candidate_delta_issues(
            source,
            "A cat sits beside a lamp. It glows.",
        )
        self.assertTrue(any(issue.code == "candidate.reference_ambiguous" for issue in issues))

    def test_inherited_it_ambiguity_remains_advisory(self):
        source_text = "A cat sits beside a lamp. It glows."
        source = compile_prompt_contract({"Draft": source_text})
        issues = candidate_delta_issues(source, source_text)
        self.assertFalse(any(issue.code == "candidate.reference_ambiguous" for issue in issues))

    def test_continuation_keeps_safe_sentence_and_discards_added_participant(self):
        base = "A woman waits beneath amber light."
        contract = compile_prompt_contract({"Draft": base})
        kept, rejected = validate_continuation_sentences(
            base,
            "Mist softens the distant archway. A second man enters the room.",
            contract,
        )
        self.assertIn("Mist softens", kept)
        self.assertNotIn("second man", kept)
        self.assertTrue(any(issue.code == "recovery.entity_added" for issue in rejected))


if __name__ == "__main__":
    unittest.main()
