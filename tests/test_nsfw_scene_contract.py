import unittest

from action_emotion_presets import (
    EXPLICIT_ADULT_ACTION_PRESET_KEYS,
    EXPLICIT_ADULT_ACTION_PRESET_METADATA,
    EXPLICIT_ADULT_EMOTION_PRESET_KEYS,
    EXPLICIT_ADULT_EMOTION_PRESET_METADATA,
)
from concept_presets import (
    EXPLICIT_ADULT_CONCEPT_PRESET_KEYS,
    EXPLICIT_ADULT_CONCEPT_PRESET_METADATA,
)
from mix_ingredient_presets import (
    EXPLICIT_ADULT_MIX_INGREDIENT_PRESET_METADATA,
    mix_ingredient_key,
    mix_ingredient_preset_catalog,
)
from nsfw_scene_contract import (
    dildo_direction_instruction,
    extract_nsfw_scene_contract,
    format_nsfw_preset_contract,
    format_nsfw_scene_contract,
    nsfw_image_audit_contract,
    nsfw_preset_compatibility_issues,
    nsfw_scene_contract_issues,
    reaction_binding_issues,
    single_phase_issues,
    strip_nsfw_catalog_labels,
)
from visual_direction_presets import (
    EXPLICIT_ADULT_VISUAL_DIRECTION_PRESET_KEYS,
    EXPLICIT_ADULT_VISUAL_DIRECTION_PRESET_METADATA,
)


class NsfwSceneContractTests(unittest.TestCase):
    def test_generic_adult_toy_is_satisfied_by_specific_toy(self):
        self.assertEqual(
            nsfw_scene_contract_issues(
                "A solo adult woman uses a dildo vaginally.",
                "A solo adult woman uses an adult toy vaginally.",
            ),
            [],
        )

    def test_gender_role_aliases_preserve_relation_direction(self):
        self.assertEqual(
            nsfw_scene_contract_issues(
                "An adult female partner kisses an adult male partner.",
                "An adult woman kisses an adult man.",
            ),
            [],
        )

    def test_malformed_possessive_role_still_preserves_second_adult(self):
        source = "A mature adult woman performs a blowjob on a mans penis."
        candidate = (
            "A mature adult woman performs oral stimulation on the penis of an "
            "adult man with visible mouth-to-penis contact."
        )
        missing_man = (
            "A mature adult woman performs oral stimulation with visible "
            "mouth-to-penis contact."
        )

        source_contract = extract_nsfw_scene_contract(source)
        self.assertEqual(source_contract["participant_count"], 2)
        self.assertEqual(
            set(source_contract["participant_roles"]),
            {"woman", "man"},
        )
        self.assertEqual(nsfw_scene_contract_issues(candidate, source), [])
        self.assertTrue(
            any(
                "missing requested adult participant role: man" in issue
                for issue in nsfw_scene_contract_issues(missing_man, source)
            )
        )

    def test_extracts_participants_act_target_object_phase_and_direction(self):
        contract = extract_nsfw_scene_contract(
            "Two adult partners: the adult woman kisses the adult man while using "
            "a dildo for vaginal contact during active intimacy."
        )

        self.assertTrue(contract["sexual"])
        self.assertEqual(contract["participant_count"], 2)
        self.assertIn("kissing", contract["acts"])
        self.assertIn("toy use", contract["acts"])
        self.assertIn("vaginal", contract["body_targets"])
        self.assertIn("dildo", contract["objects"])
        self.assertEqual(contract["dildo_use_target"], "vaginal opening")
        self.assertIn(
            "rounded insertion tip points toward the vaginal opening",
            dildo_direction_instruction(contract),
        )
        self.assertEqual(contract["visible_phase"], "active")
        self.assertIn(
            {"actor": "woman", "action": "kissing", "receiver": "man"},
            contract["relations"],
        )

    def test_compact_private_contract_protects_the_literal_core_without_geometry(self):
        text = format_nsfw_scene_contract(
            extract_nsfw_scene_contract(
                "A solo adult woman uses a dildo vaginally."
            )
        )

        self.assertIn("Private literal adult-scene core", text)
        self.assertIn("adult count=1", text)
        self.assertIn("action=toy use", text)
        self.assertIn("contact=vaginal", text)
        self.assertIn("object=dildo", text)
        self.assertIn("State the core action once", text)
        self.assertIn("Do not explain anatomy, toy geometry", text)
        self.assertIn("Balanced improvement", text)
        self.assertIn("one compact compatible visual cluster", text)
        self.assertNotIn("rounded insertion tip", text)
        self.assertNotIn("visible phase", text)

    def test_literal_core_expansion_tracks_rewrite_risk(self):
        contract = extract_nsfw_scene_contract("dildo in vagina")

        strict = format_nsfw_scene_contract(
            contract,
            risk_level="Strict cleanup",
        )
        balanced = format_nsfw_scene_contract(
            contract,
            risk_level="Balanced improvement",
        )
        creative = format_nsfw_scene_contract(
            contract,
            risk_level="Creative enhancement",
        )

        self.assertIn("invent no scene facts", strict)
        self.assertIn("one compact compatible visual cluster", balanced)
        self.assertIn("build one coherent visual direction", creative)

    def test_plain_dildo_in_vagina_is_recognized_as_the_complete_visual_core(self):
        contract = extract_nsfw_scene_contract("dildo in vagina")

        self.assertIn("toy use", contract["acts"])
        self.assertEqual(contract["body_targets"], ["vaginal"])
        self.assertEqual(contract["objects"], ["dildo"])
        self.assertEqual(contract["dildo_use_target"], "vaginal opening")
        self.assertEqual(contract["literal_core"], "dildo in vagina")
        self.assertIn(
            "literal wording=dildo in vagina",
            format_nsfw_scene_contract(contract),
        )
        self.assertEqual(
            nsfw_scene_contract_issues("dildo in vagina", "dildo in vagina"),
            [],
        )

    def test_dildo_direction_resolves_anally_to_the_anal_opening(self):
        contract = extract_nsfw_scene_contract(
            "A solo adult man uses a dildo anally."
        )

        self.assertIn("anal", contract["body_targets"])
        self.assertEqual(contract["dildo_use_target"], "anal opening")
        self.assertIn(
            "rounded insertion tip points toward the anal opening",
            dildo_direction_instruction(contract),
        )

    def test_negative_and_non_use_mentions_do_not_become_active_toy_use(self):
        cases = (
            "A clearly adult woman poses with no dildo or other sex toy.",
            "A boxed dildo lies unopened beside a vaginal anatomy textbook.",
            "A product photograph of a double-ended dildo on a white background.",
        )

        for source in cases:
            with self.subTest(source=source):
                contract = extract_nsfw_scene_contract(source)
                self.assertNotIn("toy use", contract["acts"])
                self.assertEqual(contract["body_targets"], [])
                self.assertEqual(contract["dildo_use_target"], "")
                self.assertEqual(dildo_direction_instruction(contract), "")

        negated = extract_nsfw_scene_contract(cases[0])
        self.assertEqual(negated["objects"], [])
        self.assertFalse(negated["sexual"])

    def test_scene_fidelity_detects_changed_act_target_object_and_role_direction(self):
        source = (
            "The adult woman kisses the adult man and uses a dildo for vaginal contact."
        )
        candidate = (
            "The adult man kisses the adult woman during anal intercourse without the toy."
        )

        issues = nsfw_scene_contract_issues(candidate, source)
        joined = "\n".join(issues)
        self.assertIn("missing requested sexual act family: toy use", joined)
        self.assertIn("missing requested body/contact target: vaginal", joined)
        self.assertIn("unrequested anal contact", joined)
        self.assertIn("missing requested adult object: dildo", joined)
        self.assertIn("missing or reversed sexual role binding", joined)

    def test_solo_toy_mechanics_satisfy_masturbation_act_family(self):
        source = "A solo adult woman masturbates with a dildo vaginally."
        candidate = (
            "A solo adult woman thrusts a separate dildo into her vagina, "
            "with its base remaining visibly outside."
        )

        candidate_contract = extract_nsfw_scene_contract(candidate)
        self.assertIn("toy use", candidate_contract["acts"])
        self.assertIn("masturbation", candidate_contract["acts"])
        self.assertFalse(
            any(
                issue == "missing requested sexual act family: masturbation"
                for issue in nsfw_scene_contract_issues(candidate, source)
            )
        )

    def test_translated_mature_adult_role_remains_a_solo_masturbation_scene(self):
        contract = extract_nsfw_scene_contract(
            "A solo mature adult woman repeatedly thrusts a dildo into her vulva."
        )

        self.assertEqual(contract["participant_count"], 1)
        self.assertIn("toy use", contract["acts"])
        self.assertIn("masturbation", contract["acts"])

    def test_reported_self_directed_wording_is_canonicalized(self):
        source = (
            "A solo adult woman fucking herself with a big thick dildo sex toy, "
            "hammering it inside her wet pussy with her hands."
        )
        contract = extract_nsfw_scene_contract(source)

        self.assertIn("toy use", contract["acts"])
        self.assertIn("masturbation", contract["acts"])
        self.assertIn("vaginal", contract["body_targets"])
        self.assertEqual(contract["dildo_use_target"], "vaginal opening")

    def test_partnered_toy_use_does_not_imply_solo_masturbation(self):
        contract = extract_nsfw_scene_contract(
            "Two adult partners use a dildo for vaginal contact."
        )

        self.assertIn("toy use", contract["acts"])
        self.assertNotIn("masturbation", contract["acts"])

    def test_reactions_must_name_the_adult_and_visible_cause(self):
        weak = reaction_binding_issues(
            "Two adult partners touch. Breathless pleasure and trembling follow.",
            participant_count=2,
        )
        strong = reaction_binding_issues(
            "The adult woman trembles in response to the adult partner's touch.",
            participant_count=2,
        )

        self.assertTrue(any("named adult role" in issue for issue in weak))
        self.assertTrue(any("causing action" in issue for issue in weak))
        self.assertEqual(strong, [])

    def test_single_image_rejects_visible_multi_phase_progression(self):
        prompt = (
            "The adult partners begin with teasing, then move into intercourse and "
            "afterward relax in afterglow."
        )

        self.assertTrue(single_phase_issues(prompt, content_format="Single Image"))
        self.assertEqual(single_phase_issues(prompt, content_format="Comic Story"), [])

    def test_every_adult_catalog_entry_has_complete_metadata(self):
        expected_maps = (
            (EXPLICIT_ADULT_ACTION_PRESET_KEYS, EXPLICIT_ADULT_ACTION_PRESET_METADATA),
            (EXPLICIT_ADULT_EMOTION_PRESET_KEYS, EXPLICIT_ADULT_EMOTION_PRESET_METADATA),
            (EXPLICIT_ADULT_CONCEPT_PRESET_KEYS, EXPLICIT_ADULT_CONCEPT_PRESET_METADATA),
            (
                EXPLICIT_ADULT_VISUAL_DIRECTION_PRESET_KEYS,
                EXPLICIT_ADULT_VISUAL_DIRECTION_PRESET_METADATA,
            ),
        )
        required_fields = {
            "kind",
            "category",
            "value",
            "participant_modes",
            "act_families",
            "body_targets",
            "objects",
            "phase",
            "requires_separate_object",
            "reaction_cues",
        }
        for keys, metadata in expected_maps:
            self.assertEqual(set(keys), set(metadata))
            for value in metadata.values():
                self.assertTrue(required_fields.issubset(value))

        adult_catalog = mix_ingredient_preset_catalog(explicit_nsfw=True)
        expected_mixer_keys = {
            mix_ingredient_key(category, value)
            for category, values in adult_catalog.items()
            if "NSFW" in category
            for value in values
        }
        self.assertEqual(
            expected_mixer_keys,
            set(EXPLICIT_ADULT_MIX_INGREDIENT_PRESET_METADATA),
        )

    def test_preset_metadata_detects_participant_and_phase_conflicts(self):
        metadata = [
            {
                "participant_modes": ["solo"],
                "phase": "anticipation",
                "act_families": ["masturbation"],
                "objects": [],
            },
            {
                "participant_modes": ["couple"],
                "phase": "aftercare",
                "act_families": ["intercourse"],
                "objects": [],
            },
        ]

        issues = nsfw_preset_compatibility_issues(metadata)
        self.assertTrue(any("solo and multi-participant" in issue for issue in issues))
        self.assertTrue(any("multiple visible phases" in issue for issue in issues))
        contract = format_nsfw_preset_contract(metadata)
        self.assertIn("Resolve these conflicts in favor of the user's draft", contract)

    def test_catalog_labels_are_removed_without_losing_direction(self):
        cleaned = strip_nsfw_catalog_labels(
            "Visual direction: NSFW — Adult erotic tone: candid spontaneous adult intimacy."
        )

        self.assertEqual(
            cleaned,
            "Visual direction: candid spontaneous adult intimacy.",
        )

    def test_generated_image_audit_uses_the_same_scene_contract(self):
        audit = nsfw_image_audit_contract(
            "A solo adult woman uses a dildo vaginally.",
            "A solo adult woman uses a separate dildo with a visible contact boundary.",
        )

        self.assertIn("Adult participant count", audit)
        self.assertIn("toy use", audit)
        self.assertIn("vaginal", audit)
        self.assertIn("dildo", audit)
        self.assertIn("rounded insertion tip points toward the vaginal opening", audit)
        self.assertIn("base or handle stays outside", audit)
        self.assertIn("actor and receiver roles", audit)


if __name__ == "__main__":
    unittest.main()
