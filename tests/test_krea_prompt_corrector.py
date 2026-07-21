import json
import random
import re
import ssl
import threading
import urllib.error
import urllib.parse
import unittest
from unittest.mock import patch

import krea_prompt_corrector as corrector


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload
        self.headers = {"Content-Type": "application/json"}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class FakeBinaryResponse:
    def __init__(self, body, content_type="image/jpeg"):
        self.body = body
        self.headers = {"Content-Type": content_type}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self, size=-1):
        if size is None or size < 0:
            return self.body
        return self.body[:size]

    def close(self):
        return None


class FakeStreamingResponse:
    def __init__(self, lines):
        self.lines = iter(lines)
        self.closed = False
        self.headers = {"Content-Type": "text/event-stream"}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.closed = True
        return False

    def readline(self):
        return next(self.lines, b"")

    def close(self):
        self.closed = True


def random_bad_prompt():
    subjects = [
        "a dancer",
        "a knight",
        "a chef",
        "a skateboarder",
    ]
    actions = [
        "jumping while standing still",
        "swinging a sword with wrong hand hand",
        "running but sitting in a chair",
        "throwing a ball with no arm motion",
    ]
    styles = [
        "photoreal flat vector icon",
        "cinematic anime product photo",
        "macro wide full body closeup",
        "minimal chaotic clutter everywhere",
    ]
    random.seed(7)
    return (
        f"{random.choice(subjects)},,, {random.choice(actions)}   "
        f"{random.choice(styles)} noon midnight, ultra detialed detialed, "
        "bad anatomy, watermark, sign saying hello world"
    )


class PromptCorrectorTests(unittest.TestCase):
    def test_meme_response_normalizes_small_model_caption_schemas(self):
        response = (
            "SCENE: A square reaction meme showing an airport departure board replaced "
            "by a giant red consequence meter while shocked travelers step away from one "
            "smug passenger under stark terminal lighting.\n\n"
            "TOP TEXT: THAT ESCALATED FAST\n"
            "**Bottom Caption:** `NOW ENJOY THE NO-FLY LIST`"
        )

        normalized = corrector.normalize_meme_response_text(response)

        self.assertNotIn("SCENE:", normalized)
        self.assertIn('top caption "THAT ESCALATED FAST"', normalized)
        self.assertIn('bottom caption "NOW ENJOY THE NO-FLY LIST"', normalized)
        self.assertEqual(
            corrector.meme_prompt_issues(
                normalized,
                original_prompt=(
                    "Create an original meme response tailored specifically to this "
                    "background situation, which is context and not text to render: "
                    "an airport passenger behaved badly."
                ),
            ),
            [],
        )

    def test_meme_response_assigns_placement_to_unlabelled_invented_quotes(self):
        one_caption = corrector.normalize_meme_response_text(
            'A smug passenger discovers the exit door has become a consequence meter. '
            '"WELCOME TO THE NO-FLY LIST"'
        )
        two_captions = corrector.normalize_meme_response_text(
            'A manager opens a meeting inside another meeting. '
            '"WE SCHEDULED A MEETING" "ABOUT TOO MANY MEETINGS"'
        )

        self.assertIn(
            'Place the top caption "WELCOME TO THE NO-FLY LIST"',
            one_caption,
        )
        self.assertIn(
            'Place the top caption "WE SCHEDULED A MEETING"',
            two_captions,
        )
        self.assertIn(
            'Place the bottom caption "ABOUT TOO MANY MEETINGS"',
            two_captions,
        )

    def test_meme_response_accepts_unlabelled_model_invention_without_retry(self):
        brief = (
            "Create an original meme response tailored specifically to this background "
            "situation, which is context and not text to render: a manager scheduled "
            "another meeting about meetings."
        )
        response = (
            'A square office meme showing an exhausted employee trapped inside an absurd '
            'nesting-doll stack of conference rooms under flat fluorescent light. '
            '"WE FORMED A COMMITTEE" "TO REDUCE COMMITTEES"'
        )

        with patch(
            "krea_prompt_corrector.chat_completion",
            return_value=response,
        ) as completion:
            result = corrector.post_chat_completion(
                base_url="http://127.0.0.1:1234/v1",
                model="test-4b",
                prompt=brief,
                content_format="Meme",
                temperature=0.2,
                max_tokens=800,
                timeout=30,
                api_key="test",
            )

        self.assertEqual(completion.call_count, 1)
        self.assertIn('top caption "WE FORMED A COMMITTEE"', result)
        self.assertIn('bottom caption "TO REDUCE COMMITTEES"', result)

    def test_meme_generation_uses_reference_analysis_as_glossary_only(self):
        messages = corrector.build_meme_generation_messages(
            prompt="A courier reacts to a late delivery.",
            generator_target="Krea 2",
            variation_count=1,
            reference_context=(
                "Allowed concept facts: red waxed-canvas courier bag.\n"
                "Rejected scene details: rooftop pose and blue-hour camera."
            ),
        )

        user = str(messages[1]["content"])
        self.assertIn("User-selected reference analysis", user)
        self.assertIn("red waxed-canvas courier bag", user)
        self.assertIn("Use only its allowed facts", user)
        self.assertIn("never copy", user)

    def test_meme_caption_suggestion_uses_context_and_normalizes_one_line(self):
        messages = corrector.build_meme_caption_suggestion_messages(
            position="bottom",
            response_context="A manager scheduled another meeting about reducing meetings.",
            response_goal="Friendly disbelief",
            scene="An exhausted employee trapped inside nested conference rooms",
            tone="Dry observational",
            caption_style="Classic bold white with black outline",
            current_caption="REDUCE THE MEETINGS",
            other_caption="WE FORMED A COMMITTEE",
        )

        self.assertIn("bottom caption under 8 words", messages[0]["content"])
        self.assertIn("mandatory creative seed", messages[0]["content"])
        self.assertIn("Return only the caption words", messages[0]["content"])
        self.assertIn("manager scheduled another meeting", messages[1]["content"])
        self.assertIn("Current bottom caption: REDUCE THE MEETINGS", messages[1]["content"])
        self.assertIn("WE FORMED A COMMITTEE", messages[1]["content"])
        self.assertEqual(
            corrector.normalize_meme_caption_suggestion(
                '**BOTTOM TEXT:** "TO REDUCE COMMITTEES"'
            ),
            "TO REDUCE COMMITTEES",
        )
        self.assertEqual(
            corrector.normalize_meme_caption_suggestion("CAPTION WORDS"),
            "",
        )
        self.assertEqual(
            corrector.normalize_meme_caption_suggestion(
                "Turkish men: safety = slap & smile. Blood on the floor, grin on his"
            ),
            "Turkish men: safety means slap and smile.",
        )

    def test_meme_field_suggestion_uses_all_context_and_normalizes_one_value(self):
        messages = corrector.build_meme_field_suggestion_messages(
            field="scene",
            response_context="A manager scheduled another meeting.",
            response_goal="Dry disbelief",
            scene="A tired employee",
            focus="The employee's direct stare",
            tone="Ironic",
            caption_style="Classic outlined text",
            aspect_ratio="1:1 square",
            visual_direction="Deadpan flash photography",
            camera_direction="Low-angle medium-wide shot, 35mm lens",
            top_caption="THIS COULD BE AN EMAIL",
            bottom_caption="SO WE BOOKED A MEETING",
        )

        self.assertIn("filling exactly one form field", messages[0]["content"])
        self.assertIn("mandatory creative seed", messages[0]["content"])
        self.assertIn("Do not include panels", messages[0]["content"])
        self.assertIn("manager scheduled", messages[1]["content"])
        self.assertIn("Current field value: A tired employee", messages[1]["content"])
        self.assertIn("employee's direct stare", messages[1]["content"])
        self.assertIn("THIS COULD BE AN EMAIL", messages[1]["content"])
        self.assertIn("Low-angle medium-wide shot", messages[1]["content"])
        self.assertEqual(
            corrector.normalize_meme_field_suggestion(
                "Scene: An employee opens one door\u2014revealing another conference room.",
                "scene",
            ),
            "An employee opens one door, revealing another conference room.",
        )
        self.assertEqual(
            corrector.normalize_meme_field_suggestion("not supplied", "scene"),
            "",
        )
        self.assertEqual(
            corrector.normalize_meme_field_suggestion(
                "Primary focus: The employee's direct stare",
                "focus",
            ),
            "The employee's direct stare",
        )

    def test_single_image_field_suggestion_uses_context_and_normalizes_one_value(self):
        messages = corrector.build_single_image_field_suggestion_messages(
            field="story_elements",
            draft="A courier reaches a flooded city gate.",
            concepts="solarpunk, storm photography",
            concept_mix="watercolor:60%, ink:40%",
            concept_mix_guidance=(
                'Private scoped concept mix groups: keep each blend bound only '
                'to its named visual target. Do not average percentages between '
                'groups, move attributes between targets, or print group names, '
                'targets, percentages, or numeric weights in the final image '
                'prompt.\n- Group "Courier clothing" targets Alice: linen 70%, '
                "chrome 30%."
            ),
            visual_direction="Ominous mood, cool moonlight, and low valley mist.",
            goal_headline="A hopeful arrival during a dangerous flood",
            focus="the courier's medicine satchel",
            story_elements="The courier keeps the medicine dry.",
            weighted_terms="red satchel:1.5",
            model_instructions="Keep the action readable.",
            generation_feedback="Make the flood feel more dangerous.",
            mode="Cinematic",
            generator_target="Krea 2",
            camera_direction="Eye-level medium shot, 50mm lens",
        )

        self.assertIn("filling exactly one form field", messages[0]["content"])
        self.assertIn("single still image", messages[0]["content"])
        self.assertIn("mandatory creative seed", messages[0]["content"])
        self.assertIn("courier reaches", messages[1]["content"])
        self.assertIn("medicine satchel", messages[1]["content"])
        self.assertIn("Ominous mood, cool moonlight", messages[1]["content"])
        self.assertIn(
            "Keep the selected camera direction coherent",
            messages[0]["content"],
        )
        self.assertIn("Eye-level medium shot", messages[1]["content"])
        self.assertIn(
            'Group "Courier clothing" targets Alice',
            messages[1]["content"],
        )
        self.assertIn("private control context", messages[0]["content"])
        self.assertIn("never quote or expose", messages[0]["content"])
        self.assertIn(
            "Current field value: The courier keeps the medicine dry.",
            messages[1]["content"],
        )
        self.assertEqual(
            corrector.normalize_creative_field_suggestion(
                "Story beat: The courier raises the dry medicine satchel\u2014while floodwater strikes the gate.",
                "story_elements",
            ),
            "The courier raises the dry medicine satchel, while floodwater strikes the gate.",
        )

    def test_blank_invent_field_explicitly_requests_invention_from_scratch(self):
        messages = corrector.build_single_image_field_suggestion_messages(
            field="focus",
            draft="A courier reaches a flooded city gate.",
            focus="",
        )

        self.assertIn(
            "The current focus is blank. Invent it from scratch",
            messages[0]["content"],
        )
        self.assertIn("Current field value: blank", messages[1]["content"])

    def test_invent_prompt_research_is_glossary_only_inside_the_invent_pass(self):
        messages = corrector.build_single_image_field_suggestion_messages(
            field="draft",
            draft="A courier reaches a flooded city gate.",
            concepts="Art Nouveau",
            research_context="Grounded fact: whiplash curves are characteristic.",
            concept_context="Allowed concept facts: organic linework.",
        )

        self.assertIn("Never copy a research source's subject", messages[0]["content"])
        self.assertIn(
            "Grounded research for this Invent pass",
            messages[1]["content"],
        )
        self.assertIn("whiplash curves", messages[1]["content"])
        self.assertIn("organic linework", messages[1]["content"])

    def test_invented_form_fields_have_explicit_and_enforced_length_limits(self):
        focus_messages = corrector.build_single_image_field_suggestion_messages(
            field="focus",
            draft="A courier reaches a flooded city gate.",
        )
        self.assertIn("Use at most 14 words", focus_messages[0]["content"])
        long_focus = " ".join(f"word{index}" for index in range(1, 26))
        normalized_focus = corrector.normalize_creative_field_suggestion(
            f"Focus: {long_focus}",
            "focus",
        )
        self.assertEqual(len(normalized_focus.split()), 14)
        self.assertTrue(normalized_focus.endswith("word14"))

        long_concepts = ", ".join(
            f"concept {index} with several unnecessary descriptive extra words"
            for index in range(1, 8)
        )
        normalized_concepts = corrector.normalize_creative_field_suggestion(
            f"Concepts: {long_concepts}",
            "concepts",
        )
        concept_items = normalized_concepts.split(", ")
        self.assertGreaterEqual(len(concept_items), 3)
        self.assertLessEqual(len(concept_items), 6)
        self.assertTrue(all(len(item.split()) <= 4 for item in concept_items))
        self.assertLessEqual(len(normalized_concepts.split()), 24)
        self.assertNotIn("concept 7", normalized_concepts)
        concept_messages = corrector.build_single_image_field_suggestion_messages(
            field="concepts",
        )
        self.assertIn(
            "one-to-four-word noun phrase",
            concept_messages[0]["content"],
        )

        comic_messages = corrector.build_comic_field_suggestion_messages(
            field="concepts",
            premise="A courier crosses a flooded city with medicine.",
            panel_count=2,
        )
        self.assertIn("Use at most 24 words", comic_messages[0]["content"])
        long_panel = " ".join(f"beat{index}" for index in range(1, 60))
        normalized_panel = corrector.normalize_creative_field_suggestion(
            f"Panel 1 beat: {long_panel}",
            "panel_1",
        )
        self.assertEqual(len(normalized_panel.split()), 48)

        meme_messages = corrector.build_meme_field_suggestion_messages(
            field="focus",
            scene="A cat watches a broken printer.",
        )
        self.assertIn("Use at most 14 words", meme_messages[0]["content"])
        normalized_meme_focus = corrector.normalize_meme_field_suggestion(
            f"Primary focus: {long_focus}",
            "focus",
        )
        self.assertEqual(len(normalized_meme_focus.split()), 14)

    def test_invented_weighted_words_repair_priority_prose_and_spaced_decimals(self):
        leaked = (
            "Prominent visual elements: cooking "
            "(clear visual priority, 1. 55), performing "
            "(strong visual priority, 1. 95)."
        )

        normalized = corrector.normalize_creative_field_suggestion(
            leaked,
            "weighted_terms",
        )

        self.assertEqual(normalized, "cooking:1.55, performing:1.8")
        self.assertEqual(
            corrector.parse_weighted_terms(leaked),
            [("cooking", 1.55), ("performing", 1.95)],
        )
        visible = corrector.strip_weighted_term_syntax(
            (
                "Prominent visual elements: cooking "
                "(clear visual priority, 1. 55), performing "
                "(strong visual priority, 1. 95)."
            ),
            leaked,
        )
        self.assertEqual(
            visible,
            "Prominent visual elements: cooking, performing.",
        )

    def test_central_invent_boundary_is_canonical_idempotent_and_rejects_bad_shapes(self):
        cases = (
            (
                "single",
                "concepts",
                "Concepts: wet concrete, red raincoat, luminous algae",
                "wet concrete, red raincoat, luminous algae",
            ),
            (
                "single",
                "concept_mix",
                "watercolor: 70%, ink: 30%",
                "watercolor:70%, ink:30%",
            ),
            (
                "single",
                "weighted_terms",
                (
                    "Prominent visual elements: cooking "
                    "(clear visual priority, 1. 55), performing "
                    "(strong visual priority, 1. 95)"
                ),
                "cooking:1.55, performing:1.8",
            ),
            (
                "single",
                "focus",
                "Focus: the courier's dry medicine satchel",
                "the courier's dry medicine satchel",
            ),
            (
                "comic",
                "title",
                "Working title: The Last Delivery",
                "The Last Delivery",
            ),
            (
                "meme",
                "focus",
                "Primary focus: the cat's completely calm expression",
                "the cat's completely calm expression",
            ),
            (
                "meme",
                "top",
                "Top caption: THIS COULD BE AN EMAIL",
                "THIS COULD BE AN EMAIL",
            ),
        )
        for workspace, field, raw, expected in cases:
            with self.subTest(workspace=workspace, field=field):
                normalized = corrector.normalize_and_validate_invent(
                    workspace,
                    field,
                    raw,
                )
                self.assertEqual(normalized, expected)
                self.assertEqual(
                    corrector.normalize_and_validate_invent(
                        workspace,
                        field,
                        normalized,
                    ),
                    normalized,
                )

        rejected = (
            ("single", "concepts", "only one concept"),
            ("single", "concept_mix", "watercolor:100%"),
            ("single", "focus", "Prominent visual elements: the courier"),
            (
                "single",
                "focus",
                "one two three four five six seven eight nine ten eleven twelve "
                "thirteen fourteen fifteen",
            ),
            (
                "meme",
                "scene",
                'A tired employee stares at a glowing sign reading "MEETING".',
            ),
            (
                "meme",
                "top",
                "THIS CAPTION CONTAINS FAR TOO MANY WORDS FOR ONE MEME CAPTION",
            ),
            ("unknown", "focus", "the courier"),
        )
        for workspace, field, raw in rejected:
            with self.subTest(workspace=workspace, field=field, raw=raw):
                self.assertEqual(
                    corrector.normalize_and_validate_invent(
                        workspace,
                        field,
                        raw,
                    ),
                    "",
                )
        oversized_focus = (
            "one two three four five six seven eight nine ten eleven twelve "
            "thirteen fourteen fifteen"
        )
        normalized_oversized = corrector.normalize_invent_candidate(
            "single",
            "focus",
            oversized_focus,
        )
        self.assertEqual(normalized_oversized, oversized_focus)
        self.assertIn(
            "Invented field exceeds 14 words",
            corrector.invent_field_issues(
                "single",
                "focus",
                normalized_oversized,
            ),
        )
        oversized_draft = " ".join(f"detail{index}" for index in range(1, 176))
        recovered_draft = corrector.recover_invent_length_overflow(
            "single",
            "draft",
            oversized_draft,
            seed_value="detail1 detail2 courier",
        )
        self.assertEqual(corrector.CREATIVE_FIELD_WORD_LIMITS["draft"], 160)
        self.assertLessEqual(len(recovered_draft.split()), 160)
        self.assertEqual(
            corrector.invent_field_issues(
                "single",
                "draft",
                recovered_draft,
                seed_value="detail1 detail2 courier",
            ),
            [],
        )

        repair_messages = corrector.build_invent_field_repair_messages(
            workspace="single",
            field="focus",
            candidate=(
                "The courier's brass medicine satchel remains dry while the "
                "flood wave crashes across the clinic entrance behind it."
            ),
            issues=["Invented field exceeds 14 words"],
            seed_value="the courier's brass medicine satchel",
        )
        self.assertIn("Hard maximum: 14 words", repair_messages[0]["content"])
        self.assertIn("mandatory original field value", repair_messages[0]["content"])
        self.assertIn("exceeds 14 words", repair_messages[1]["content"])
        preserved_concepts = corrector.preserve_invent_seed_value(
            "single",
            "concepts",
            "exploration, wet reflections, neon machinery, cinematic realism",
            seed_value="discovering",
        )
        self.assertEqual(
            preserved_concepts,
            (
                "discovering, exploration, wet reflections, neon machinery, "
                "cinematic realism"
            ),
        )
        self.assertEqual(
            corrector.normalize_and_validate_invent(
                "single",
                "concepts",
                "exploration, wet reflections, neon machinery",
                seed_value="discovering",
            ),
            "discovering, exploration, wet reflections, neon machinery",
        )
        concept_repair = corrector.build_invent_field_repair_messages(
            workspace="single",
            field="concepts",
            candidate="exploration, wet reflections, neon machinery",
            issues=["Invented field weakened or replaced its mandatory seed"],
            seed_value="discovering",
        )
        self.assertIn(
            "Keep every existing concept verbatim",
            concept_repair[0]["content"],
        )
        concept_messages = corrector.build_single_image_field_suggestion_messages(
            field="concepts",
            concepts="discovering",
        )
        self.assertIn(
            "Keep every existing concept verbatim",
            concept_messages[0]["content"],
        )
        expanded_story = (
            "The pair exchange a quiet glance beneath cascading water while reflected "
            "neon ripples across chrome walls and drifting steam softens the surrounding "
            "machinery as their synchronized posture establishes a calm emotional center "
            "within the otherwise chaotic industrial setting."
        )
        preserved_story = corrector.preserve_invent_seed_value(
            "single",
            "story_elements",
            expanded_story,
            seed_value="mutual trust shown through relaxed proximity",
        )
        self.assertTrue(
            preserved_story.startswith(
                "mutual trust shown through relaxed proximity."
            )
        )
        self.assertLessEqual(len(preserved_story.split()), 30)
        self.assertEqual(
            corrector.invent_field_issues(
                "single",
                "story_elements",
                preserved_story,
                seed_value="mutual trust shown through relaxed proximity",
            ),
            [],
        )
        story_messages = corrector.build_single_image_field_suggestion_messages(
            field="story_elements",
            story_elements="mutual trust shown through relaxed proximity",
        )
        self.assertIn(
            "Keep the existing story beat verbatim",
            story_messages[0]["content"],
        )
        self.assertEqual(
            corrector.normalize_and_validate_invent(
                "single",
                "goal_headline",
                (
                    "Courier lifts medicine bag as flood surges, hope saved "
                    "and city submerged."
                ),
                seed_value=(
                    "A tense but hopeful medicine delivery through a flooded city."
                ),
            ),
            (
                "Courier lifts medicine bag as flood surges, hope saved "
                "and city submerged."
            ),
        )
        self.assertEqual(
            corrector.normalize_and_validate_invent(
                "single",
                "focus",
                "a completely unrelated mountain observatory",
                seed_value="the courier's brass medicine satchel",
            ),
            "",
        )

    def test_saved_typed_invent_values_migrate_without_rewriting_manual_prose(self):
        self.assertEqual(
            corrector.canonicalize_saved_invent_value(
                "single",
                "weighted_terms",
                (
                    "Prominent visual elements: cooking "
                    "(clear visual priority, 1. 55), performing "
                    "(strong visual priority, 1. 95)"
                ),
            ),
            "cooking:1.55, performing:1.95",
        )
        self.assertEqual(
            corrector.canonicalize_saved_invent_value(
                "single",
                "concept_mix",
                "watercolor=7, ink=3",
            ),
            "watercolor:70%, ink:30%",
        )
        manual_focus = (
            "Keep this intentionally long manually authored focus exactly as written "
            "because it is an explicit user instruction."
        )
        self.assertEqual(
            corrector.canonicalize_saved_invent_value(
                "single",
                "focus",
                manual_focus,
            ),
            manual_focus,
        )

    def test_weighted_invent_boundary_handles_adversarial_model_formats(self):
        variants = (
            "cooking:1.55, performing:1.8",
            "Weighted words: cooking = 1.55, performing * 1.8",
            "Weighted visual priorities: cooking:1. 55, performing:1. 8",
            (
                "cooking (clear visual priority, 1.55), "
                "performing (strong visual priority, 1.8)"
            ),
            (
                "Prominent visual elements: cooking "
                "(clear visual priority, 1. 55), performing "
                "(strong visual priority, 1. 8)"
            ),
            "```text\nWeighted words: cooking:1.55, performing:1.8\n```",
        )
        for raw in variants:
            with self.subTest(raw=raw):
                self.assertEqual(
                    corrector.normalize_and_validate_invent(
                        "single",
                        "weighted_terms",
                        raw,
                    ),
                    "cooking:1.55, performing:1.8",
                )

    def test_comic_panel_suggestion_uses_neighboring_panels_and_normalizes_one_value(self):
        messages = corrector.build_comic_field_suggestion_messages(
            field="panel_2",
            title="The Last Delivery",
            premise="A courier crosses a flooded city with medicine.",
            continuity="Same red coat and brass satchel in every panel.",
            concepts="bioluminescent algae, Art Nouveau ironwork",
            visual_direction="Muted watercolor with clean ink lines.",
            dialogue_direction="Use short caveman grammar and no modern slang.",
            panels=[
                "The courier reaches the flooded gate.",
                "The courier forces the rusted gate open.",
                "The courier hands the medicine to a doctor.",
                "The clinic lights come back on.",
            ],
            panel_count=4,
            layout="2 x 2 grid",
            reading_order="Left to right, top to bottom",
            aspect_ratio="4:5 portrait",
            camera_direction="Overhead top-down shot, 35mm lens",
            speech_bubbles=True,
            artistic_detail_freedom=True,
            concept_research=(
                "Grounded concept glossary and factual verification only:\n"
                "Bioluminescent algae emits visible light through a chemical reaction."
            ),
        )

        self.assertIn("only panel 2", messages[0]["content"])
        self.assertIn("Selected shared camera direction", messages[0]["content"])
        self.assertIn("Overhead top-down shot", messages[1]["content"])
        self.assertIn("mandatory creative seed", messages[0]["content"])
        self.assertIn("Speech bubbles are allowed", messages[0]["content"])
        self.assertIn(
            "Mandatory dialogue wording contract for invented speech",
            messages[0]["content"],
        )
        self.assertIn("short caveman grammar", messages[0]["content"])
        self.assertIn("Required concept integration contract", messages[0]["content"])
        self.assertIn("bioluminescent algae", messages[0]["content"])
        self.assertIn("Grounded concept research contract", messages[0]["content"])
        self.assertIn("emits visible light", messages[0]["content"])
        self.assertIn(
            "Do not copy or infer another source image's subject",
            messages[0]["content"],
        )
        self.assertIn("Mandatory shared comic style direction", messages[0]["content"])
        self.assertIn("Muted watercolor", messages[0]["content"])
        self.assertIn("Artistic detail freedom is enabled", messages[0]["content"])
        self.assertIn("Panel 1: The courier reaches", messages[1]["content"])
        self.assertIn(
            "Current field value: The courier forces the rusted gate open.",
            messages[1]["content"],
        )
        self.assertIn("Panel 3: The courier hands", messages[1]["content"])
        self.assertEqual(
            corrector.normalize_creative_field_suggestion(
                "Panel 2 beat: The courier braces against a wave and shouts \u201cHold the gate!\u201d",
                "panel_2",
            ),
            'The courier braces against a wave and shouts "Hold the gate!"',
        )
        enforced = corrector.enforce_comic_speech_bubble_contract(
            "Panel 2 beat: The courier braces against a wave and shouts \u201cHold the gate!\u201d",
            speech_bubbles=True,
        )
        self.assertIn('"Hold the gate!"', enforced)
        self.assertIn("clearly readable speech bubble", enforced)
        self.assertIn("tail pointing unambiguously", enforced)
        self.assertEqual(
            corrector.enforce_comic_speech_bubble_contract(
                'The courier shouts "Hold the gate!"',
                speech_bubbles=False,
            ),
            'The courier shouts "Hold the gate!"',
        )

    def test_auto_comic_layout_matches_panel_count_and_all_panel_invention(self):
        self.assertEqual(
            corrector.resolve_comic_layout("Auto grid", 3),
            (
                "three-panel page with two equal panels across the top and one "
                "full-width panel across the bottom"
            ),
        )
        self.assertNotIn(
            "3 x 2",
            corrector.resolve_comic_layout("3 x 2 grid", 3),
        )
        self.assertIn(
            "three columns and two rows",
            corrector.resolve_comic_layout("3 x 2 grid", 6),
        )
        horizontal = corrector.resolve_comic_layout("Horizontal strip", 3)
        self.assertEqual(
            corrector.resolve_comic_layout(horizontal, 3),
            horizontal,
        )

        messages = corrector.build_all_comic_panels_suggestion_messages(
            title="The Last Delivery",
            premise="A courier crosses a flooded city with medicine.",
            continuity="Same red coat and brass satchel in every panel.",
            concepts="bioluminescent algae, Art Nouveau ironwork",
            visual_direction="Muted watercolor with clean ink lines.",
            dialogue_direction=(
                "Characters speak like cavemen with short grammar, simple words, "
                "and no modern slang."
            ),
            panels=[
                "The courier reaches the flooded clinic gate.",
                "",
                "The courier hands the medicine to a doctor.",
            ],
            panel_count=3,
            layout="Auto grid",
            speech_bubbles=True,
            artistic_detail_freedom=True,
            concept_research=(
                "Grounded concept glossary and factual verification only:\n"
                "Art Nouveau ironwork commonly uses flowing organic curves."
            ),
        )

        self.assertIn("exactly 3 chronological panel beats", messages[0]["content"])
        self.assertIn("mandatory creative seed", messages[0]["content"])
        self.assertIn("Invent from scratch only for blank beats", messages[0]["content"])
        self.assertIn("Panel 1: through Panel 3:", messages[0]["content"])
        self.assertIn("Characters speak like cavemen", messages[0]["content"])
        self.assertIn("Required concept integration contract", messages[0]["content"])
        self.assertIn("Grounded concept research contract", messages[0]["content"])
        self.assertIn("flowing organic curves", messages[0]["content"])
        self.assertIn("bioluminescent algae", messages[0]["content"])
        self.assertIn("Mandatory shared comic style direction", messages[0]["content"])
        self.assertIn("Muted watercolor", messages[0]["content"])
        self.assertIn(
            "Dialogue writing direction: Characters speak like cavemen",
            messages[1]["content"],
        )
        self.assertIn(
            "Mandatory invented-dialogue contract: Characters speak like cavemen",
            messages[1]["content"],
        )
        self.assertIn("full-width panel across the bottom", messages[1]["content"])
        self.assertNotIn("2 x 3", messages[1]["content"])
        panels = corrector.normalize_all_comic_panel_suggestions(
            "\n".join(
                (
                    "Panel 1: The red-coated courier reaches the flooded clinic gate.",
                    (
                        "Panel 2: The courier lifts the brass satchel and says "
                        "\u201cOpen the gate!\u201d"
                    ),
                    "Panel 3: The courier hands the dry medicine to the doctor.",
                )
            ),
            panel_count=3,
            speech_bubbles=True,
        )
        self.assertEqual(len(panels), 3)
        self.assertIn('"Open the gate!"', panels[1])
        self.assertIn("clearly readable speech bubble", panels[1])
        self.assertEqual(
            corrector.normalize_all_comic_panel_suggestions(
                "Panel 1: Setup.\nPanel 3: Payoff.",
                panel_count=3,
                speech_bubbles=False,
            ),
            [],
        )

        direction_messages = corrector.build_comic_field_suggestion_messages(
            field="dialogue_direction",
            premise="Two cave people discover a strange metal door.",
            dialogue_direction="",
            panels=["They find the door.", "They open it."],
            panel_count=2,
            speech_bubbles=True,
        )
        self.assertIn(
            "Do not write an actual dialogue line",
            direction_messages[0]["content"],
        )
        self.assertIn(
            "Return only a reusable speech-writing instruction",
            direction_messages[0]["content"],
        )

        concept_messages = corrector.build_comic_field_suggestion_messages(
            field="concepts",
            premise="A courier crosses a flooded city with medicine.",
            concepts="",
            visual_direction="Muted watercolor with clean ink lines.",
            panels=["The courier reaches the gate.", "The clinic opens."],
            panel_count=2,
        )
        self.assertIn(
            "comma-separated list of compatible visual concepts",
            concept_messages[0]["content"],
        )
        self.assertIn("Mandatory shared comic style direction", concept_messages[0]["content"])
        self.assertNotIn("Required concept integration contract", concept_messages[0]["content"])

    def test_artistic_detail_freedom_applies_to_single_comic_and_meme_prompts(self):
        single_system = corrector.build_system_prompt(
            content_format="Single Image",
            risk_level="Strict cleanup",
            preserve_strictly=True,
            develop_story=False,
            artistic_detail_freedom=True,
        )
        comic_system = corrector.build_system_prompt(
            content_format="Comic Story",
            artistic_detail_freedom=True,
        )
        meme_messages = corrector.build_meme_generation_messages(
            prompt="A cat watches a broken printer.",
            generator_target="Krea 2",
            variation_count=1,
            artistic_detail_freedom=True,
        )

        for prompt in (
            single_system,
            comic_system,
            meme_messages[0]["content"],
        ):
            self.assertIn("Artistic detail freedom is enabled", prompt)
            self.assertIn("Preserve the requested main subject", prompt)
            self.assertIn("Do not replace what the image is about", prompt)

    def test_comic_panel_invention_can_disable_speech_bubbles(self):
        messages = corrector.build_comic_field_suggestion_messages(
            field="panel_1",
            premise="A robot repairs a lunar greenhouse.",
            panels=["", ""],
            panel_count=2,
            speech_bubbles=False,
        )

        self.assertIn("Do not invent speech bubbles", messages[0]["content"])
        self.assertNotIn("Speech bubbles are allowed", messages[0]["content"])

    def test_manual_meme_captions_are_restored_when_model_drops_or_rewrites_them(self):
        brief = (
            "A single image-macro meme. "
            'Place one centered top caption reading exactly "EXACT TOP WORDS". '
            'Place one centered bottom caption reading exactly "EXACT BOTTOM WORDS".'
        )
        wrong = (
            'A square reaction image showing a tired employee beside a broken server. '
            'Place the top caption "PARAPHRASED TOP". '
            'Place the bottom caption "PARAPHRASED BOTTOM".'
        )
        dropped = (
            "A square reaction image showing a tired employee beside a broken server "
            "under harsh office lighting."
        )

        self.assertEqual(
            corrector.meme_caption_requirements(brief),
            {
                "top": "EXACT TOP WORDS",
                "bottom": "EXACT BOTTOM WORDS",
            },
        )
        for response in (wrong, dropped):
            with self.subTest(response=response):
                with patch(
                    "krea_prompt_corrector.chat_completion",
                    return_value=response,
                ) as completion:
                    result = corrector.post_chat_completion(
                        base_url="http://127.0.0.1:1234/v1",
                        model="test-4b",
                        prompt=brief,
                        content_format="Meme",
                        temperature=0.2,
                        max_tokens=800,
                        timeout=30,
                        api_key="test",
                    )

                self.assertEqual(completion.call_count, 1)
                self.assertIn('top caption "EXACT TOP WORDS"', result)
                self.assertIn('bottom caption "EXACT BOTTOM WORDS"', result)
                self.assertNotIn("PARAPHRASED", result)
                self.assertIn("separate flat graphic overlay", result)
                self.assertIn("normal spelling and letter order", result)
                self.assertIn("Show no other text", result)
                self.assertEqual(
                    result.casefold().count("separate flat graphic overlay"),
                    1,
                )
                self.assertEqual(
                    corrector.meme_prompt_issues(
                        result,
                        original_prompt=brief,
                    ),
                    [],
                )

    def test_meme_typography_removes_competing_environmental_text(self):
        candidate = (
            'A passenger stands beneath a smudged "Safety First" sticker with only '
            "an 'S' still visible. "
            'Place the top caption "EXACT TOP". '
            'Place the bottom caption "EXACT BOTTOM".'
        )

        result = corrector.enforce_meme_typography_contract(candidate)

        self.assertNotIn('"Safety First"', result)
        self.assertNotIn("'S'", result)
        self.assertIn('top caption "EXACT TOP"', result)
        self.assertIn('bottom caption "EXACT BOTTOM"', result)
        self.assertIn("no more than two balanced lines", result)

    def test_caption_removes_unicode_dashes_inside_quotes(self):
        caption = "She’s bleeding\u2014he’s still smiling."
        prompt = f'Airport reaction scene. Place the top caption "{caption}".'

        normalized = corrector.normalize_final_prompt_text(prompt)

        expected = "She’s bleeding, he’s still smiling."
        self.assertNotIn("\u2014", normalized)
        self.assertIn(f'"{expected}"', normalized)
        self.assertEqual(corrector.quoted_phrases(normalized), [expected])

    def test_meme_response_retries_echoed_brief_and_returns_invented_caption(self):
        brief = (
            "A single image-macro meme in 1:1 square format. "
            "Create an original meme response tailored specifically to this background "
            "situation, which is context and not text to render: a manager scheduled "
            "another meeting about reducing meetings. Intended response or stance: "
            "friendly disbelief. Find a concise visual analogy, reaction, reversal, or contrast."
        )
        invented = (
            'A square deadpan office-photo meme showing a conference-room table buried '
            'under an absurd nesting-doll stack of smaller conference tables, while one '
            'exhausted employee stares directly at the camera under flat fluorescent light. '
            'Place the top caption "WE FORMED A COMMITTEE" and the bottom caption '
            '"TO REDUCE COMMITTEES" in bold white text with a black outline.'
        )

        with patch(
            "krea_prompt_corrector.chat_completion",
            side_effect=[brief, invented],
        ) as completion:
            result = corrector.post_chat_completion(
                base_url="http://127.0.0.1:1234/v1",
                model="test-4b",
                prompt=brief,
                content_format="Meme",
                temperature=0.2,
                max_tokens=800,
                timeout=30,
                api_key="test",
                seed=100,
            )

        self.assertTrue(result.startswith(invented))
        self.assertIn("separate flat graphic overlay", result)
        self.assertIn("Show no other text", result)
        self.assertEqual(completion.call_count, 2)
        self.assertEqual(completion.call_args_list[0].kwargs["seed"], 100)
        self.assertEqual(completion.call_args_list[1].kwargs["seed"], 101)
        self.assertEqual(completion.call_args_list[0].kwargs["temperature"], 0.2)
        self.assertEqual(completion.call_args_list[1].kwargs["temperature"], 0.2)
        self.assertFalse(
            corrector.meme_prompt_issues(result, original_prompt=brief)
        )

    def test_meme_response_repairs_only_caption_after_two_scene_candidates(self):
        brief = (
            "Create an original meme response tailored specifically to this background "
            "situation, which is context and not text to render: a passenger behaved "
            "aggressively on a flight. Intended response or stance: disgust."
        )
        scene = (
            "A square deadpan airport meme showing a smug passenger standing beneath an "
            "enormous departures board that has transformed into a bright red consequence "
            "meter, while nearby travelers recoil and an unimpressed gate agent points "
            "toward a tiny exit sign under stark terminal lighting."
        )
        repaired = (
            "SCENE: " + scene + "\n\n"
            "TOP TEXT: CONGRATULATIONS\n"
            "BOTTOM TEXT: YOU UNLOCKED THE NO-FLY LIST"
        )
        diagnostics = []

        with patch(
            "krea_prompt_corrector.chat_completion",
            side_effect=[scene, scene, repaired],
        ) as completion:
            result = corrector.post_chat_completion(
                base_url="http://127.0.0.1:1234/v1",
                model="huihui-qwen3-vl-2b-instruct-abliterated",
                prompt=brief,
                content_format="Meme",
                temperature=0.2,
                max_tokens=800,
                timeout=30,
                api_key="test",
                seed=8,
                diagnostic_callback=diagnostics.append,
            )

        self.assertEqual(completion.call_count, 3)
        self.assertEqual(
            [call.kwargs["seed"] for call in completion.call_args_list],
            [8, 9, 10],
        )
        self.assertIn('top caption "CONGRATULATIONS"', result)
        self.assertIn('bottom caption "YOU UNLOCKED THE NO-FLY LIST"', result)
        self.assertTrue(
            any("caption-only repair" in message for message in diagnostics)
        )
        self.assertIn(
            "Return only the final prompt in one line",
            completion.call_args_list[0].kwargs["messages"][0]["content"],
        )

    def test_meme_validator_rejects_unfinished_instruction_echo(self):
        brief = (
            "Create an original meme response tailored specifically to this background "
            "situation, which is context and not text to render: too many meetings. "
            "Invent the clearest, funniest underlying visual scene."
        )

        issues = corrector.meme_prompt_issues(brief, original_prompt=brief)

        self.assertTrue(any("unfinished meme brief" in issue for issue in issues))
        self.assertTrue(any("no invented quoted caption" in issue for issue in issues))

    def test_chat_completion_sends_fixed_seed_and_omits_random_seed(self):
        response = FakeResponse(
            {"choices": [{"message": {"content": "finished prompt"}}]}
        )
        with patch("urllib.request.urlopen", return_value=response) as urlopen:
            corrector.chat_completion(
                base_url="http://127.0.0.1:1234/v1",
                model="test-model",
                messages=[{"role": "user", "content": "test"}],
                temperature=0.2,
                max_tokens=100,
                timeout=10,
                api_key="test",
                seed=314,
                ttl=86_400,
            )
        fixed_payload = json.loads(urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertEqual(fixed_payload["seed"], 314)
        self.assertEqual(fixed_payload["ttl"], 86_400)

        with patch("urllib.request.urlopen", return_value=response) as urlopen:
            corrector.chat_completion(
                base_url="http://127.0.0.1:1234/v1",
                model="test-model",
                messages=[{"role": "user", "content": "test"}],
                temperature=0.2,
                max_tokens=100,
                timeout=10,
                api_key="test",
            )
        random_payload = json.loads(urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertNotIn("seed", random_payload)
        self.assertNotIn("ttl", random_payload)

    def test_safe_for_work_cleanup_removes_explicit_content_and_gore(self):
        unsafe = "A naked warrior in a seductive pose, bloody and surrounded by graphic gore"
        cleaned = corrector.make_prompt_safe_for_work(unsafe)
        self.assertFalse(corrector.safe_for_work_issues(cleaned))
        self.assertIn("fully clothed", cleaned)
        self.assertNotIn("safe-for-work", cleaned.lower())

    def test_safe_for_work_cleanup_avoids_double_negative_safety_language(self):
        source = (
            "A studio portrait with no explicit nudity or erotic framing, "
            "no graphic gore and no non-sexual framing."
        )

        cleaned = corrector.make_prompt_safe_for_work(source)

        self.assertFalse(corrector.safe_for_work_issues(cleaned))
        self.assertNotIn("no non-sexual", cleaned.lower())
        self.assertNotIn("no non-graphic", cleaned.lower())
        self.assertNotIn("nudity", cleaned.lower())
        self.assertNotIn("erotic", cleaned.lower())
        self.assertNotIn("gore", cleaned.lower())
        self.assertIn("complete opaque clothing", cleaned.lower())
        self.assertIn("neutral non-sexual framing", cleaned.lower())

    def test_safe_for_work_cleanup_upgrades_old_positive_safety_contract(self):
        source = (
            "A portrait. Preserve the core subject. "
            "Safe-for-work presentation with complete non-transparent clothing, "
            "non-sexual framing, and no visible graphic injury detail."
        )

        cleaned = corrector.make_prompt_safe_for_work(source)

        self.assertFalse(corrector.safe_for_work_issues(cleaned))
        self.assertNotIn("non-opaque", cleaned.lower())
        self.assertNotIn("no visible graphic injury", cleaned.lower())
        self.assertNotIn("safe-for-work", cleaned.lower())
        self.assertNotIn("general audience", cleaned.lower())
        self.assertNotIn("preserve the core subject", cleaned.lower())
        self.assertEqual(cleaned, "A portrait.")

    def test_safe_for_work_cleanup_catches_suggestive_terms_and_framing(self):
        unsafe = (
            "A sexy woman in a sheer dress with cleavage and bedroom eyes, "
            "a suggestive pose and a close-up shot emphasizing her body."
        )

        cleaned = corrector.make_prompt_safe_for_work(unsafe)

        self.assertFalse(corrector.safe_for_work_issues(cleaned))
        for term in ("sexy", "sheer dress", "cleavage", "bedroom eyes", "suggestive pose"):
            self.assertNotIn(term, cleaned.lower())
        self.assertIn("opaque modest clothing", cleaned.lower())
        self.assertIn("friendly expression", cleaned.lower())

    def test_safe_for_work_neutralizes_suggestive_food_closeup_without_dropping_action(self):
        source = (
            "A woman with a visible tongue actively licking a giant ice cream cone, "
            "close-up shot emphasizing the tongue's motion and contact with the dessert."
        )

        cleaned = corrector.make_prompt_safe_for_work(source)

        self.assertFalse(corrector.safe_for_work_issues(cleaned))
        self.assertIn("licking", cleaned.lower())
        self.assertNotIn("visible tongue", cleaned.lower())
        self.assertNotIn("emphasizing the tongue", cleaned.lower())
        self.assertIn("food-advertising composition", cleaned.lower())

    def test_safe_for_work_is_a_hard_final_contract(self):
        issues = corrector.final_compliance_issues(
            "an erotic nude portrait",
            original_prompt="a portrait",
            safe_for_work=True,
        )
        hard, _soft = corrector.split_compliance_issues(issues)
        self.assertTrue(any("Safe-for-work contract violated" in issue for issue in hard))

    @patch("krea_prompt_corrector.chat_completion", return_value="a nude figure with graphic gore")
    def test_post_completion_deterministically_enforces_safe_for_work(self, _chat):
        result = corrector.post_chat_completion(
            base_url="http://127.0.0.1:1234/v1",
            model="test-7b",
            prompt="an adult figure after a battle",
            temperature=0.1,
            max_tokens=512,
            timeout=30,
            api_key="test",
            audit_repair=False,
            safe_for_work=True,
        )
        self.assertFalse(corrector.safe_for_work_issues(result))
        self.assertNotIn("safe-for-work", result.lower())
        self.assertNotIn("general audience", result.lower())
        self.assertNotIn("preserve the core subject", result.lower())
        self.assertIn("fully clothed", result.lower())

    def test_explicit_adult_mode_is_internal_and_preserves_requested_adult_content(self):
        completed_prompt = (
            "Two clearly adult women pose nude in a private bedroom portrait with direct "
            "anatomical detail, warm window light, and an intimate close composition."
        )
        with patch(
            "krea_prompt_corrector.chat_completion",
            return_value=completed_prompt,
        ) as completion:
            result = corrector.post_chat_completion(
                base_url="http://127.0.0.1:1234/v1",
                model="qwen3-vl-4b-instruct",
                prompt="Two adult women in an explicit nude bedroom portrait.",
                temperature=0.2,
                max_tokens=400,
                timeout=30,
                api_key="test",
                explicit_nsfw=True,
                audit_repair=False,
                final_gate_repair=False,
            )

        sent_message = completion.call_args.kwargs["messages"][1]["content"]
        self.assertIn("Explicit adult mode is enabled", sent_message)
        self.assertIn("unambiguously adult", sent_message)
        self.assertIn("short, literal image-generator wording", sent_message)
        self.assertIn("clear direct phrase", sent_message)
        self.assertNotIn("dildo in vagina", sent_message)
        self.assertIn("State the core action once", sent_message)
        self.assertIn("Balanced improvement", sent_message)
        self.assertIn("one compact compatible visual cluster", sent_message)
        self.assertIn("must not dilute the core action", sent_message)
        self.assertEqual(result, completed_prompt)
        self.assertNotIn("NSFW mode", result)
        self.assertNotIn("age policy", result)

    def test_explicit_adult_mode_sends_a_small_literal_core_privately(self):
        source = "A solo adult woman uses a dildo vaginally during active intimacy."
        completed_prompt = (
            "A solo adult woman uses a dildo vaginally during active intimacy. "
            "The dildo is a separate manufactured sex toy with a visible base, outer "
            "contour, orientation, and contact boundary against the woman's body."
        )
        with patch(
            "krea_prompt_corrector.chat_completion",
            return_value=completed_prompt,
        ) as completion:
            result = corrector.post_chat_completion(
                base_url="http://127.0.0.1:1234/v1",
                model="qwen3-vl-4b-instruct",
                prompt=source,
                temperature=0.2,
                max_tokens=500,
                timeout=30,
                api_key="test",
                explicit_nsfw=True,
                audit_repair=False,
                final_gate_repair=False,
            )

        sent_message = completion.call_args.kwargs["messages"][1]["content"]
        self.assertIn("Private literal adult-scene core", sent_message)
        self.assertIn("adult count=1", sent_message)
        self.assertIn("action=toy use", sent_message)
        self.assertIn("contact=vaginal", sent_message)
        self.assertIn("object=dildo", sent_message)
        self.assertNotIn("rounded insertion tip", sent_message)
        self.assertNotIn("base or handle stays outside", sent_message)
        self.assertNotIn("Private literal adult-scene core", result)

    def test_nsfw_scene_fidelity_is_a_hard_final_contract(self):
        source = "The adult woman kisses the adult man."
        candidate = (
            "The adult man on image-left kisses the adult woman on image-right."
        )
        issues = corrector.final_compliance_issues(
            candidate,
            original_prompt=source,
            output_length="Concise",
            explicit_nsfw=True,
        )

        contract_issue = next(
            issue
            for issue in issues
            if issue.startswith("NSFW scene fidelity contract")
        )
        self.assertIn("missing or reversed sexual role binding", contract_issue)
        self.assertTrue(corrector.is_hard_compliance_issue(contract_issue))

    def test_post_completion_strips_nsfw_catalog_labels_for_krea(self):
        source = "Two adult partners share an intimate kiss."
        candidate = (
            "Two adult partners share an intimate kiss. Visual direction: "
            "NSFW — Adult erotic tone: candid spontaneous adult intimacy."
        )
        with patch(
            "krea_prompt_corrector.chat_completion",
            return_value=candidate,
        ):
            result = corrector.post_chat_completion(
                base_url="http://127.0.0.1:1234/v1",
                model="test-4b",
                prompt=source,
                temperature=0.2,
                max_tokens=400,
                timeout=30,
                api_key="test",
                explicit_nsfw=True,
                audit_repair=False,
                final_gate_repair=False,
            )

        self.assertNotIn("NSFW", result)
        self.assertIn("candid spontaneous adult intimacy", result)
        self.assertNotIn("Visual direction:", result)

    def test_explicit_adult_mode_rejects_underage_and_ambiguous_age_terms(self):
        for prompt in (
            "an explicit portrait of a teen",
            "a nude schoolgirl",
            "a sexualized young-looking character",
            "an explicit portrait of a young woman",
        ):
            with self.subTest(prompt=prompt):
                with self.assertRaisesRegex(RuntimeError, "unambiguously adult"):
                    corrector.post_chat_completion(
                        base_url="http://127.0.0.1:1234/v1",
                        model="test-model",
                        prompt=prompt,
                        temperature=0.1,
                        max_tokens=300,
                        timeout=30,
                        api_key="test",
                        explicit_nsfw=True,
                    )

        self.assertFalse(
            corrector.explicit_adult_mode_issues(
                "an explicit portrait of a young adult woman age 24"
            )
        )

    def test_minor_sexual_content_is_rejected_in_every_mode_before_model_call(self):
        invalid_prompt = "A nude 17-year-old character in an erotic scene."
        diagnostics = []

        self.assertTrue(corrector.minor_sexual_content_issues(invalid_prompt))
        with patch("krea_prompt_corrector.chat_completion") as completion:
            with self.assertRaisesRegex(RuntimeError, "underage or ambiguous-age"):
                corrector.post_chat_completion(
                    base_url="http://127.0.0.1:1234/v1",
                    model="test-model",
                    prompt=invalid_prompt,
                    temperature=0.1,
                    max_tokens=300,
                    timeout=30,
                    api_key="test",
                    explicit_nsfw=False,
                    diagnostic_callback=diagnostics.append,
                )
        completion.assert_not_called()
        self.assertEqual(diagnostics, [])

    def test_minor_subjects_remain_allowed_in_nonsexual_scenes(self):
        prompt = "Two children in winter coats build a snowman in a public park."

        self.assertEqual(corrector.minor_sexual_content_issues(prompt), [])

    def test_safe_for_work_and_explicit_adult_modes_are_mutually_exclusive(self):
        with self.assertRaisesRegex(RuntimeError, "cannot both be enabled"):
            corrector.post_chat_completion(
                base_url="http://127.0.0.1:1234/v1",
                model="test-model",
                prompt="an adult portrait",
                temperature=0.1,
                max_tokens=300,
                timeout=30,
                api_key="test",
                safe_for_work=True,
                explicit_nsfw=True,
            )

    def test_flux2_klein_target_uses_model_specific_prompting(self):
        exact = corrector.build_system_prompt(
            generator_target="FLUX.2 Klein 9B",
            risk_level="Strict cleanup",
            preserve_strictly=True,
            enhance_actions=False,
            develop_story=False,
        )
        improve = corrector.build_system_prompt(
            generator_target="FLUX.2 Klein 9B",
            risk_level="Balanced improvement",
            develop_story=False,
        )
        user = corrector.build_user_message(
            "a red car on a wet road",
            generator_target="FLUX.2 Klein 9B",
            risk_level="Strict cleanup",
            develop_story=False,
        )

        self.assertIn("precision editor for FLUX.2 Klein 9B", exact)
        self.assertIn("no prompt upsampling", exact)
        self.assertIn("main subject, key action, critical style", exact)
        self.assertIn("no prompt upsampling", improve)
        self.assertIn("Medium-length natural-language prompts", improve)
        self.assertIn("FLUX.2 Klein 9B-ready prompt", user)
        self.assertNotIn("Krea creativity", exact)

    def test_flux2_klein_setup_is_external_and_fixed(self):
        recommendation = corrector.format_generator_recommendation(
            "FLUX.2 Klein 9B"
        )
        self.assertIn("4 inference steps", recommendation)
        self.assertIn("guidance 1.0", recommendation)
        self.assertIn("Prompt upsampling is unavailable", recommendation)
        self.assertIn("Non-Commercial License", recommendation)

        prompt = corrector.enforce_generator_settings_contract(
            "A red car on wet asphalt. FLUX.2 setup: steps=4, guidance=1.0"
        )
        self.assertEqual(prompt, "A red car on wet asphalt.")
        hard, _soft = corrector.split_compliance_issues(
            corrector.krea_settings_issues(
                "A red car, steps=4, guidance=1.0",
                include_krea_settings=False,
                creativity="raw",
                intensity=0,
                complexity=0,
                movement=0,
            )
        )
        self.assertTrue(hard)

    def test_flux2_audit_and_repair_prompts_keep_flux_target(self):
        audit = corrector.build_audit_system_prompt(
            generator_target="FLUX.2 Klein 9B"
        )
        repair = corrector.build_final_repair_system_prompt(
            generator_target="FLUX.2 Klein 9B"
        )

        self.assertIn("strict FLUX.2 Klein 9B prompt compliance auditor", audit)
        self.assertIn("Repaired FLUX.2 Klein prompt:", audit)
        self.assertIn("You repair a FLUX.2 Klein 9B prompt", repair)
        self.assertEqual(
            corrector.extract_repaired_prompt(
                "Audit score: 90/100\nRepaired FLUX.2 Klein prompt:\nA red car."
            ),
            "A red car.",
        )

    def test_post_completion_threads_flux_target_and_strips_flux_setup(self):
        captured = {}
        response = {
            "choices": [{"message": {"content": (
                "A red coupe on wet asphalt. FLUX.2 setup: steps=4, guidance=1.0"
            )}}]
        }

        def fake_urlopen(request, timeout):
            captured.update(json.loads(request.data.decode("utf-8")))
            return FakeResponse(response)

        with patch("urllib.request.urlopen", fake_urlopen):
            result = corrector.post_chat_completion(
                base_url="http://127.0.0.1:1234/v1",
                model="qwen3-vl-4b-instruct",
                prompt="A red coupe on wet asphalt.",
                generator_target="FLUX.2 Klein 9B",
                temperature=0.1,
                max_tokens=200,
                timeout=5,
                api_key="test-key",
                risk_level="Strict cleanup",
                preserve_strictly=True,
                enhance_actions=False,
                develop_story=False,
            )

        self.assertIn("FLUX.2 Klein 9B image prompts", captured["messages"][0]["content"])
        self.assertEqual(result, "A red coupe on wet asphalt.")
        self.assertNotIn("steps=4", result)

    def test_cli_defaults_to_exact_fidelity(self):
        with patch("sys.argv", ["krea_prompt_corrector.py", "--prompt", "a cup"]):
            args = corrector.parse_args()

        self.assertEqual(args.risk_level, "Strict cleanup")
        self.assertTrue(args.preserve_strictly)
        self.assertTrue(args.no_story_development)
        self.assertEqual(args.temperature, 0.1)
        self.assertEqual(args.detail, "Balanced")
        self.assertEqual(args.context_tokens, corrector.CONTEXT_TOKEN_AUTO)
        self.assertEqual(args.creativity, "raw")
        self.assertEqual(args.content_format, "Single Image")

        with patch(
            "sys.argv",
            [
                "krea_prompt_corrector.py",
                "--prompt",
                "a cup",
                "--target",
                "FLUX.2 Klein 9B",
            ],
        ):
            flux_args = corrector.parse_args()
        self.assertEqual(flux_args.target, "FLUX.2 Klein 9B")

    def test_single_image_and_comic_story_prompts_work_for_both_generators(self):
        for target in corrector.GENERATOR_TARGETS:
            single = corrector.build_system_prompt(
                generator_target=target,
                content_format="Single Image",
                risk_level="Balanced improvement",
            )
            comic = corrector.build_system_prompt(
                generator_target=target,
                content_format="Comic Story",
                risk_level="Balanced improvement",
            )
            self.assertIn("exactly one still image", single)
            self.assertIn("multi-panel comic or story page", comic)
            self.assertIn(target, single)
            self.assertIn(target, comic)

    def test_content_format_validator_separates_single_images_and_comics(self):
        single_issues = corrector.final_compliance_issues(
            "A comic strip with Panel 1: a fox wakes. Panel 2: the fox runs.",
            original_prompt="A fox adventure",
            content_format="Single Image",
            altered_text_encoder=False,
        )
        self.assertTrue(any(issue.startswith("Content format:") for issue in single_issues))

        comic_issues = corrector.final_compliance_issues(
            "A fox running through a forest.",
            original_prompt="A fox adventure",
            content_format="Comic Story",
            altered_text_encoder=False,
        )
        self.assertTrue(any("expected 4 explicitly identified panels" in issue for issue in comic_issues))

    def test_single_image_fails_fast_when_source_requests_panels(self):
        with self.assertRaisesRegex(RuntimeError, "Switch Format to Comic Story"):
            corrector.post_chat_completion(
                base_url="http://127.0.0.1:1234/v1",
                model="qwen3-vl-4b-instruct",
                prompt="A three-panel comic strip about a fox.",
                generator_target="Krea 2",
                content_format="Single Image",
                temperature=0.1,
                max_tokens=200,
                timeout=5,
                api_key="test-key",
            )

    def test_comic_story_user_message_defaults_to_four_panels(self):
        message = corrector.build_user_message(
            "A fox finds its way home",
            content_format="Comic Story",
            generator_target="FLUX.2 Klein 9B",
        )
        self.assertIn("Required output format: Comic Story", message)
        self.assertIn("Requested panel count: 4", message)
        self.assertIn("continuity", message)

    def test_slider_value_clamps_to_krea_range(self):
        self.assertEqual(corrector.slider_value(-150), -100)
        self.assertEqual(corrector.slider_value(42), 42)
        self.assertEqual(corrector.slider_value(150), 100)

    def test_rule_strength_clamps_and_relaxes_only_advisory_issues(self):
        self.assertEqual(corrector.rule_strength_value(-5), 0)
        self.assertEqual(corrector.rule_strength_value(65), 65)
        self.assertEqual(corrector.rule_strength_value(120), 100)
        issues = [
            "Missing quoted rendered text: KEEP OUT",
            "Weak or non-visual phrasing: beautiful",
            "Plausibility risk: accidental artifact",
        ]

        self.assertEqual(
            corrector.rule_strength_compliance_issues(issues, 20),
            ["Missing quoted rendered text: KEEP OUT"],
        )
        self.assertIn(
            "explicit user requirements",
            corrector.rule_strength_instruction(65).casefold(),
        )

    def test_normalize_lm_studio_base_url_accepts_remote_hosts(self):
        self.assertEqual(
            corrector.normalize_lm_studio_base_url("192.168.1.50:1234"),
            "http://192.168.1.50:1234/v1",
        )
        self.assertEqual(
            corrector.normalize_lm_studio_base_url("http://studio-box.local:1234"),
            "http://studio-box.local:1234/v1",
        )
        self.assertEqual(
            corrector.normalize_lm_studio_base_url(
                "http://studio-box.local:1234/v1/chat/completions"
            ),
            "http://studio-box.local:1234/v1",
        )

    def test_list_lm_studio_models_reads_native_inventory_and_filters_embeddings(self):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["timeout"] = timeout
            captured["auth"] = request.headers.get("Authorization")
            return FakeResponse(
                {
                    "models": [
                        {"type": "llm", "key": "qwen3-vl-4b-instruct"},
                        {"type": "embedding", "key": "nomic-embed-text-v1.5"},
                    ]
                }
            )

        with patch("urllib.request.urlopen", fake_urlopen):
            models = corrector.list_lm_studio_models(
                base_url="studio-box.local:1234",
                timeout=8.0,
                api_key="test-token",
            )

        self.assertEqual(captured["url"], "http://studio-box.local:1234/api/v1/models")
        self.assertEqual(captured["timeout"], 8.0)
        self.assertEqual(captured["auth"], "Bearer test-token")
        self.assertEqual(models, ["qwen3-vl-4b-instruct"])

    def test_list_ollama_models_reads_native_tags(self):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["timeout"] = timeout
            return FakeResponse(
                {
                    "models": [
                        {"name": "qwen3:4b"},
                        {"name": "gemma3:12b"},
                    ]
                }
            )

        with patch("urllib.request.urlopen", fake_urlopen):
            models = corrector.list_local_models(
                provider="Ollama",
                base_url="http://ollama-box:11434/v1",
                timeout=8.0,
                api_key="",
            )

        self.assertEqual(captured["url"], "http://ollama-box:11434/api/tags")
        self.assertEqual(captured["timeout"], 8.0)
        self.assertEqual(models, ["qwen3:4b", "gemma3:12b"])

    def test_ollama_base_url_and_provider_detection(self):
        self.assertEqual(
            corrector.normalize_model_base_url("localhost:11434"),
            "http://localhost:11434/v1",
        )
        self.assertEqual(
            corrector.model_provider_from_base_url("http://localhost:11434/v1"),
            "Ollama",
        )

    def test_estimate_max_tokens_scales_by_detail_and_variations(self):
        self.assertEqual(corrector.estimate_max_tokens("Short", 1), 220)
        self.assertEqual(corrector.estimate_max_tokens("Detailed", 3, "Detailed"), 1680)
        self.assertEqual(corrector.estimate_max_tokens("Unknown", 0, "Detailed"), 560)
        self.assertEqual(corrector.estimate_audit_max_tokens("Short", 1), 920)
        self.assertEqual(corrector.estimate_max_tokens("Rich caption", 1, "Concise"), 220)
        self.assertEqual(corrector.estimate_max_tokens("Rich caption", 1, "Expanded"), 760)
        self.assertEqual(corrector.estimate_max_tokens("Detailed", 3, "Balanced"), 1080)
        self.assertEqual(corrector.estimate_max_tokens("Short", 1, "Concise", 180), 552)

    def test_small_model_detection_reads_parameter_count_from_model_name(self):
        self.assertTrue(corrector.is_small_model("qwen3-vl-4b-instruct"))
        self.assertTrue(corrector.is_small_model("model-3.8B-Q4"))
        self.assertFalse(corrector.is_small_model("qwen2.5-vl-7b-instruct"))
        self.assertFalse(corrector.is_small_model("small-local-model"))

    def test_custom_output_word_bounds_drive_guidance_and_validation(self):
        system = corrector.build_system_prompt(
            output_length="Balanced",
            output_min_words=60,
            output_max_words=90,
        )
        user = corrector.build_user_message(
            "a knight in a castle",
            output_length="Balanced",
            output_min_words=60,
            output_max_words=90,
        )
        repair = corrector.build_final_repair_user_message(
            original_prompt="a knight in a castle",
            current_prompt="A knight.",
            issues=["Prompt too short"],
            output_length="Balanced",
            output_min_words=60,
            output_max_words=90,
        )

        self.assertIn("between 60 and 90 words", system)
        self.assertIn("Output length guidance:", user)
        self.assertIn("between 60 and 90 words", user)
        self.assertIn("between 60 and 90 words", repair)
        issues = corrector.final_compliance_issues(
            "A knight stands in a castle.",
            output_length="Balanced",
            output_min_words=60,
            output_max_words=90,
        )
        self.assertTrue(any("expected at least 60" in issue for issue in issues))

    def test_expanded_output_enforces_its_existing_word_range(self):
        guidance = corrector.length_guidance_text("Expanded")
        issues = corrector.final_compliance_issues(
            "A knight stands in a castle.",
            output_length="Expanded",
        )

        self.assertIn("Develop the scene generously", guidance)
        self.assertIn("between 140 and 280 words", guidance)
        self.assertTrue(any("Prompt too short" in issue for issue in issues))

    def test_weighted_explicit_slang_matches_its_canonical_visible_language(self):
        self.assertEqual(
            corrector.missing_weighted_terms(
                "A mature adult performs clear oral stimulation of the penis.",
                "blowjob:1.7",
            ),
            [],
        )
        self.assertEqual(
            corrector.missing_weighted_terms(
                "A mature adult performs clear manual stimulation of the penis.",
                "handjob:1.7",
            ),
            [],
        )
        self.assertIn(
            "blowjob (strong visual priority, 1.7)",
            corrector.missing_weighted_terms(
                "A mature adult stands in a rain-darkened alley.",
                "blowjob:1.7",
            ),
        )

    def test_required_explicit_concepts_match_canonical_visible_actions(self):
        candidate = (
            "A mature adult woman performs oral stimulation of the penis while "
            "straddling a dildo with rhythmic penetrative motion."
        )

        self.assertEqual(
            corrector.missing_required_concepts(
                candidate,
                "blowjob, dildo fucking",
            ),
            [],
        )
        self.assertEqual(
            corrector.missing_required_concepts(
                "A mature adult woman poses beside a boxed dildo on a shelf.",
                "blowjob, dildo fucking",
            ),
            ["blowjob", "dildo fucking"],
        )
        self.assertEqual(
            corrector.translate_explicit_adult_language("dildo fucking"),
            "rhythmic penetrative use of a dildo",
        )

    def test_compact_model_gets_one_targeted_expanded_length_repair(self):
        source = "A red car crosses a stone bridge at dawn."
        short = "A red car crosses a stone bridge at dawn."
        expanded = (
            "A red car crosses a weathered stone bridge at dawn as the first warm sunlight "
            "reaches the road. The low front three-quarter viewpoint keeps the red car "
            "dominant while revealing its wheels following the damp curve of the bridge. "
            "Fine mist hangs above the river below and catches pale gold light between the "
            "arches. Moisture darkens the old masonry and creates broken reflections beneath "
            "the moving car. The driver remains only a subtle silhouette so the vehicle and "
            "its journey stay central. Long shadows from the parapet repeat across the road "
            "and lead toward distant hills emerging from blue morning haze. A few loose "
            "leaves lift behind the rear tires to make the forward motion visible. Natural "
            "paint reflections follow the car's curved panels without becoming glossy or "
            "artificial. The composition balances crisp foreground stone texture against a "
            "soft atmospheric background and preserves a calm purposeful sense of arrival."
            " Subtle tire spray catches the backlight near the bridge joints, while weathered "
            "iron rail details and moss-filled mortar lines make the crossing feel specific, "
            "inhabited, and grounded in the cool river valley."
        )
        self.assertGreaterEqual(corrector.word_count(expanded), 140)
        with patch(
            "krea_prompt_corrector.chat_completion",
            side_effect=[short, expanded],
        ) as completion:
            result = corrector.post_chat_completion(
                base_url="http://127.0.0.1:1234/v1",
                model="test-4b",
                prompt=source,
                temperature=0.2,
                max_tokens=760,
                timeout=30,
                api_key="test",
                output_length="Expanded",
                audit_repair=False,
            )

        self.assertEqual(completion.call_count, 2)
        self.assertEqual(result, expanded)

    def test_compact_maximum_development_repairs_fidelity_then_reexpands(self):
        source = "Exactly two red lanterns hang beside a sealed stone gate at dawn."
        continuation = (
            "Their warm light defines the threshold against rain-darkened masonry. A lone "
            "courier in a weathered blue coat pauses in the foreground, one gloved hand "
            "hovering over the bronze latch while fresh muddy footprints continue beneath "
            "the door. Her guarded posture shifts into resolve as she notices a torn warning "
            "ribbon trapped under the lower hinge. Wind presses the coat against her "
            "forward-leaning stance and drives fine rain across the lantern glass, scattering "
            "copper reflections over the carved stone. A low three-quarter camera angle makes "
            "the paired lanterns the dominant frame around her decision, with the nearest "
            "footprint sharply focused and the empty road dissolving into mist behind her. "
            "One lantern swings toward the gate while the other remains still, creating a "
            "subtle imbalance that suggests recent passage. Cold blue dawn fills the distant "
            "archway, contrasting with the lantern glow on her face, wet leather satchel, and "
            "tense fingertips. Displaced gravel, dripping ivy, and a thin line of light under "
            "the door imply movement inside and turn the scene into the instant before she enters."
        )
        developed = corrector.append_creative_continuation(
            source,
            continuation,
            max_added_words=190,
        )
        count_broken = developed.replace(
            "Exactly two red lanterns",
            "Exactly three red lanterns",
            1,
        )
        shallow_fidelity_repair = source
        responses = [count_broken, shallow_fidelity_repair, continuation]
        temperatures = []
        diagnostics = []

        def fake_completion(**kwargs):
            temperatures.append(kwargs["temperature"])
            return responses.pop(0)

        with patch(
            "krea_prompt_corrector.chat_completion",
            side_effect=fake_completion,
        ) as completion:
            result = corrector.post_chat_completion(
                base_url="http://127.0.0.1:1234/v1",
                model="test-4b",
                prompt=source,
                temperature=0.25,
                max_tokens=760,
                timeout=30,
                api_key="test",
                content_format="Single Image",
                detail_level="Rich caption",
                output_length="Expanded",
                risk_level="Creative enhancement",
                enhance_actions=True,
                develop_story=True,
                artistic_detail_freedom=True,
                altered_text_encoder=False,
                audit_repair=False,
                diagnostic_callback=diagnostics.append,
            )

        self.assertEqual(completion.call_count, 3)
        self.assertEqual(temperatures, [0.25, 0.1, 0.3])
        self.assertEqual(result, developed)
        self.assertFalse(
            any("Final repair attempt 2/" in message for message in diagnostics)
        )
        self.assertTrue(
            any(
                "immutable-base creative expansion passed validation"
                in message.casefold()
                for message in diagnostics
            )
        )
        self.assertEqual(
            corrector.creative_development_issues(
                result,
                source,
                output_length="Expanded",
                risk_level="Creative enhancement",
                develop_story=True,
            ),
            [],
        )

    def test_compact_candidate_mechanically_restores_visible_story_before_audit(self):
        source = (
            "A mature adult woman and an adult man stand in a sunny public park "
            "with a large green sculpture between them."
        )
        story = "unconditional acceptance with open body language"
        candidate = (
            "A mature adult woman on image-left and an adult man on image-right "
            "stand in a sunny public park with a large green sculpture between them."
        )
        diagnostics: list[str] = []

        with patch(
            "krea_prompt_corrector.chat_completion",
            side_effect=[candidate, candidate],
        ) as completion:
            result = corrector.post_chat_completion(
                base_url="http://127.0.0.1:1234/v1",
                model="test-4b",
                prompt=source,
                temperature=0.2,
                max_tokens=500,
                timeout=30,
                api_key="test",
                content_format="Single Image",
                story_elements=story,
                audit_repair=True,
                diagnostic_callback=diagnostics.append,
            )

        self.assertEqual(completion.call_count, 2)
        self.assertIn(story, result)
        self.assertIn("relaxed shoulders", result)
        self.assertIn("open palms", result)
        self.assertFalse(
            any(
                "Story element contract" in message
                for message in diagnostics
            )
        )
        self.assertFalse(
            any("Final repair attempt" in message for message in diagnostics)
        )

    def test_compact_maximum_development_legacy_full_rewrite_is_not_retried(self):
        source = "Exactly two red lanterns hang beside a sealed stone gate at dawn."
        count_broken = (
            "Exactly three red lanterns hang beside a sealed stone gate at dawn "
            "under rain-darkened clouds."
        )
        shallow = source
        continuation = (
            "Warm reflections cross wet masonry while mist separates the gate from distant "
            "hills. A low camera emphasizes the paired lanterns, rain beads along ironwork, "
            "and wind moves torn banners above moss-filled joints. Copper light catches "
            "puddles, carved stone, and drifting moisture, giving the threshold layered "
            "depth and a clear visual path toward the sealed entrance."
        )
        diagnostics: list[str] = []

        with patch(
            "krea_prompt_corrector.chat_completion",
            side_effect=[count_broken, shallow, continuation],
        ) as completion:
            corrector.post_chat_completion(
                base_url="http://127.0.0.1:1234/v1",
                model="test-4b",
                prompt=source,
                temperature=0.25,
                max_tokens=760,
                timeout=30,
                api_key="test",
                content_format="Single Image",
                output_length="Expanded",
                risk_level="Creative enhancement",
                enhance_actions=True,
                develop_story=True,
                artistic_detail_freedom=True,
                altered_text_encoder=False,
                audit_repair=False,
                diagnostic_callback=diagnostics.append,
            )

        self.assertEqual(completion.call_count, 3)
        self.assertTrue(
            any("Final repair attempt 1/1" in message for message in diagnostics)
        )
        self.assertFalse(
            any("Final repair attempt 2/" in message for message in diagnostics)
        )


    def test_compact_maximum_development_reexpands_fidelity_fallback(self):
        source = "Exactly two red lanterns hang beside a sealed stone gate at dawn."
        continuation = (
            "Their warm light defines the threshold against rain-darkened masonry. A lone courier "
            "in a weathered blue coat pauses in the foreground, one gloved hand hovering over "
            "the bronze latch while fresh muddy footprints continue beneath the door. Her "
            "guarded posture shifts into resolve as she notices a torn warning ribbon trapped "
            "under the lower hinge. Wind presses the coat against her forward-leaning stance "
            "and drives fine rain across the lantern glass, scattering copper reflections over "
            "the carved stone. A low three-quarter camera angle makes the paired lanterns the "
            "dominant frame around her decision, with the nearest footprint sharply focused "
            "and the empty road dissolving into mist behind her. One lantern swings toward the "
            "gate while the other remains still, creating a subtle imbalance that suggests "
            "recent passage. Cold blue dawn fills the distant archway, contrasting with the "
            "lantern glow on her face, wet leather satchel, and tense fingertips. Displaced "
            "gravel, dripping ivy, and a thin line of light under the door imply movement "
            "inside and turn the scene into the instant before she enters."
        )
        developed = corrector.append_creative_continuation(
            source,
            continuation,
            max_added_words=190,
        )
        count_broken = developed.replace(
            "Exactly two red lanterns",
            "Exactly three red lanterns",
            1,
        )
        responses = [count_broken, count_broken, continuation]
        temperatures = []
        diagnostics = []

        def fake_completion(**kwargs):
            temperatures.append(kwargs["temperature"])
            return responses.pop(0)

        with patch(
            "krea_prompt_corrector.chat_completion",
            side_effect=fake_completion,
        ) as completion:
            result = corrector.post_chat_completion(
                base_url="http://127.0.0.1:1234/v1",
                model="test-4b",
                prompt=source,
                temperature=0.25,
                max_tokens=760,
                timeout=30,
                api_key="test",
                content_format="Single Image",
                detail_level="Rich caption",
                output_length="Expanded",
                risk_level="Creative enhancement",
                enhance_actions=True,
                develop_story=True,
                artistic_detail_freedom=True,
                altered_text_encoder=False,
                audit_repair=False,
                diagnostic_callback=diagnostics.append,
            )

        self.assertEqual(completion.call_count, 3)
        self.assertEqual(temperatures, [0.25, 0.1, 0.3])
        self.assertEqual(result, developed)
        expansion_messages = completion.call_args_list[2].kwargs["messages"]
        self.assertIn(
            "Return continuation prose only",
            expansion_messages[0]["content"],
        )
        self.assertNotIn(
            "Return the repaired full final prompt",
            "\n".join(message["content"] for message in expansion_messages),
        )
        self.assertTrue(
            any(
                "immutable-base creative expansion" in message.casefold()
                for message in diagnostics
            )
        )

    def test_long_fidelity_fallback_restores_story_then_reexpands(self):
        source = " ".join(
            [
                "A weathered knight waits beside an abandoned gate at pale dawn.",
                "Rain darkens the carved stone and gathers along the iron threshold.",
                "A low camera keeps the silent archway behind her guarded stance.",
            ]
            * 6
        )
        story = "unconditional acceptance with open body language"
        continuation = (
            "Copper reflections travel through shallow runoff, linking the foreground stones "
            "to distant windows veiled by pearl fog. Torn banners breathe with each crosswind, "
            "their softened shadows sweeping over eroded carvings in a slow visual rhythm. "
            "Moss brightens inside mortar seams, droplets bead along hammered edges, and thin "
            "sun shafts reveal suspended moisture above the road. The framing leaves generous "
            "negative space beyond the threshold while layered haze separates nearby masonry "
            "from muted hills, giving the quiet arrival depth and a restrained hopeful release."
        )
        self.assertGreaterEqual(
            corrector.word_count(source),
            corrector.OUTPUT_WORD_RANGES["Expanded"][0],
        )
        diagnostics = []

        with patch(
            "krea_prompt_corrector.chat_completion",
            side_effect=[source, source, continuation],
        ) as completion:
            result = corrector.post_chat_completion(
                base_url="http://127.0.0.1:1234/v1",
                model="test-4b",
                prompt=source,
                temperature=0.25,
                max_tokens=760,
                timeout=30,
                api_key="test",
                content_format="Single Image",
                detail_level="Rich caption",
                output_length="Expanded",
                risk_level="Creative enhancement",
                develop_story=True,
                artistic_detail_freedom=True,
                story_elements=story,
                audit_repair=False,
                diagnostic_callback=diagnostics.append,
            )

        self.assertEqual(completion.call_count, 3)
        self.assertIn(story, result)
        self.assertEqual(
            corrector.final_compliance_issues(
                result,
                original_prompt=source,
                story_elements=story,
                content_format="Single Image",
                output_length="Expanded",
                risk_level="Creative enhancement",
                develop_story=True,
            ),
            [],
        )
        self.assertTrue(
            any(
                "immutable-base creative expansion passed validation"
                in message.casefold()
                for message in diagnostics
            )
        )

    def test_user_message_preserves_messy_draft_for_model_inspection(self):
        draft = random_bad_prompt()
        message = corrector.build_user_message(draft)

        self.assertIn("Inspect and correct this draft Krea 2 prompt.", message)
        self.assertIn("noon midnight", message)
        self.assertIn("ultra detialed detialed", message)
        self.assertIn("sign saying hello world", message)

    def test_user_message_flags_visual_slang_for_translation(self):
        message = corrector.build_user_message("a baddie with drip, lit city vibes")

        self.assertIn("Slang and shorthand detected:", message)
        self.assertIn("baddie", message)
        self.assertIn("drip", message)
        self.assertIn("vibes", message)
        self.assertIn("concrete renderable visual language", message)

    def test_user_message_includes_vague_prompt_analysis(self):
        message = corrector.build_user_message("cool aesthetic scene")

        self.assertIn("Vague prompt request analysis:", message)
        self.assertIn("generic praise adjective", message)
        self.assertIn("undefined mood or aesthetic", message)
        self.assertIn("choose concrete visual specifics", message)

    def test_user_message_includes_visual_feeling_interpretation(self):
        message = corrector.build_user_message("a lonely sad woman in a room")

        self.assertIn("Visual feeling interpretation:", message)
        self.assertIn("loneliness: isolated framing", message)
        self.assertIn("sadness: downcast eyes", message)
        self.assertIn("visible image evidence", message)

    def test_visual_feeling_issues_require_visible_cues(self):
        unresolved = corrector.visual_feeling_issues("A sad portrait of a woman.")
        resolved = corrector.visual_feeling_issues(
            "A sad portrait of a woman with downcast eyes, slumped shoulders, and subdued lighting."
        )

        self.assertTrue(unresolved)
        self.assertEqual(resolved, [])

    def test_final_compliance_issues_reports_abstract_feelings(self):
        issues = corrector.final_compliance_issues(
            "A lonely angry man in a room.",
            output_length="Concise",
        )

        self.assertTrue(any("Unresolved abstract feeling" in issue for issue in issues))

    def test_vague_prompt_issues_find_underspecified_requests(self):
        issues = corrector.vague_prompt_issues("make something cool")

        self.assertIn("undefined subject placeholder", issues)
        self.assertIn("generic praise adjective", issues)
        self.assertIn("missing concrete subject, action, setting, camera, or lighting detail", issues)

    def test_multi_person_role_issues_find_ambiguous_bindings(self):
        issues = corrector.multi_person_role_issues(
            "two people in a cave, he holding a torch while she helps them"
        )

        joined = "\n".join(issues)
        self.assertIn("ambiguous person references", joined)
        self.assertIn("action verbs are not clearly bound", joined)

    def test_multi_person_role_issues_accept_clear_role_positions(self):
        issues = corrector.multi_person_role_issues(
            "the woman on the left holds a torch while the man on the right kneels beside a stone altar"
        )

        self.assertEqual(issues, [])
        self.assertEqual(
            corrector.multi_person_role_issues(
                "The man behind the woman faces the camera in the carwash tunnel."
            ),
            [],
        )

    def test_both_hands_is_not_treated_as_a_person_reference(self):
        prompt = (
            "A woman and a doctor stand in a park while the woman grips "
            "a green dildo with both hands."
        )

        issues = corrector.multi_person_role_issues(prompt)

        self.assertEqual(issues, [])
        self.assertFalse(
            any("ambiguous person references: both" in issue for issue in issues)
        )

    def test_single_person_gender_aliases_do_not_become_fake_multi_person_scenes(self):
        for prompt in (
            "A woman stands alone while the female subject raises her hands.",
            "A man stands alone while the male subject raises his hands.",
            "A solo woman poses while the adult female subject looks into a mirror.",
        ):
            with self.subTest(prompt=prompt):
                self.assertFalse(corrector.appears_multi_person_scene(prompt))
                self.assertEqual(corrector.multi_person_role_issues(prompt), [])

    def test_adult_is_a_descriptor_not_a_second_person_role(self):
        prompt = "An adult woman uses a dildo vaginally."

        self.assertEqual(corrector.person_role_mentions(prompt), ["woman"])
        self.assertFalse(corrector.appears_multi_person_scene(prompt))
        self.assertEqual(corrector.multi_person_role_issues(prompt), [])

    def test_adult_magazine_style_is_not_a_second_person_role(self):
        prompt = (
            "Photoreal, glossy adult magazine finish. A mature adult woman "
            "kneels in a sunny park and grips a green dildo with both hands."
        )

        self.assertEqual(corrector.person_role_mentions(prompt), ["woman"])
        self.assertFalse(corrector.appears_multi_person_scene(prompt))
        self.assertEqual(corrector.multi_person_role_issues(prompt), [])

        self.assertTrue(
            corrector.appears_multi_person_scene(
                "An adult stands beside a woman in a studio."
            )
        )

    def test_same_gender_aliases_still_detect_explicitly_distinct_people(self):
        for prompt in (
            "A woman kisses another female beside a window.",
            "A woman greets another woman beside a window.",
            "A man stands opposite another male in a boxing ring.",
            "A woman greets a female doctor in a clinic.",
            "Two women stand together beneath warm light.",
            "A woman confronts a queen in the throne room.",
            "A woman stands beside another female subject.",
        ):
            with self.subTest(prompt=prompt):
                self.assertTrue(corrector.appears_multi_person_scene(prompt))

    def test_post_completion_accepts_one_person_rephrased_with_gender_aliases(self):
        source = "A solo woman raises both hands in warm window light."
        candidate = (
            "A solo woman stands in warm window light while the female subject "
            "raises her hands with an expression of relief."
        )
        with patch(
            "krea_prompt_corrector.chat_completion",
            return_value=candidate,
        ) as completion:
            result = corrector.post_chat_completion(
                base_url="http://127.0.0.1:1234/v1",
                model="test-4b",
                prompt=source,
                temperature=0.2,
                max_tokens=400,
                timeout=30,
                api_key="test",
                audit_repair=False,
                altered_text_encoder=False,
            )

        self.assertEqual(completion.call_count, 1)
        self.assertEqual(result, candidate)
        self.assertEqual(corrector.multi_person_role_issues(result), [])

    def test_unpositioned_unique_woman_and_man_get_stable_frame_labels(self):
        prompt = (
            "Woman and man, nude in a neon-lit carwash tunnel, locked together "
            "beneath cascading water jets; reflections distort their forms while "
            "steam swirls around them."
        )

        bound = corrector.bind_unpositioned_mixed_gender_pair(prompt)
        resolved = corrector.resolve_unambiguous_multi_person_pronouns(bound)

        self.assertIn("Woman on image-left", resolved)
        self.assertIn("man on image-right", resolved)
        self.assertIn("their forms", resolved)
        self.assertEqual(corrector.multi_person_role_issues(resolved), [])

    def test_mixed_gender_position_binding_leaves_ambiguous_groups_unchanged(self):
        same_gender = "Two women stand together while one woman holds a lantern."
        already_bound = (
            "The woman on image-left faces the man on image-right beneath neon light."
        )

        self.assertEqual(
            corrector.bind_unpositioned_mixed_gender_pair(same_gender),
            same_gender,
        )
        self.assertEqual(
            corrector.bind_unpositioned_mixed_gender_pair(already_bound),
            already_bound,
        )

    def test_post_completion_repairs_unpositioned_unique_adult_pair(self):
        prompt = (
            "Woman and man, nude in a neon carwash tunnel, locked together beneath "
            "cascading water jets; reflections distort their forms while steam "
            "swirls around them."
        )
        with patch(
            "krea_prompt_corrector.chat_completion",
            return_value=prompt,
        ) as completion:
            result = corrector.post_chat_completion(
                base_url="http://127.0.0.1:1234/v1",
                model="test-4b",
                prompt=prompt,
                temperature=0.2,
                max_tokens=500,
                timeout=30,
                api_key="test",
                audit_repair=False,
                altered_text_encoder=False,
                explicit_nsfw=True,
            )

        self.assertEqual(completion.call_count, 1)
        self.assertIn("Woman on image-left", result)
        self.assertIn("man on image-right", result)
        self.assertEqual(corrector.multi_person_role_issues(result), [])

    def test_collective_people_do_not_require_individual_position_labels(self):
        collective_prompts = (
            "A couple embraces beneath warm window light.",
            "A crowd watches fireworks from a rooftop.",
            "Two women dance together under stage lights.",
            "Women and men march together through the plaza.",
        )

        for prompt in collective_prompts:
            with self.subTest(prompt=prompt):
                self.assertEqual(corrector.multi_person_role_issues(prompt), [])
                self.assertEqual(
                    corrector.bind_unpositioned_distinct_people(prompt),
                    prompt,
                )

    def test_distinct_roles_receive_natural_local_position_bindings(self):
        cases = (
            (
                "A doctor examines a patient in a bright clinic.",
                "A doctor on image-left examines a patient on image-right",
            ),
            (
                "The red-haired woman greets the bearded man at a doorway.",
                "The red-haired woman on image-left greets the bearded man on image-right",
            ),
            (
                "A woman and a nonbinary person face each other in a studio.",
                "A woman on image-left and a nonbinary person on image-right",
            ),
            (
                "A queen addresses a guard in the throne room.",
                "A queen on image-left addresses a guard on image-right",
            ),
        )

        for prompt, expected in cases:
            with self.subTest(prompt=prompt):
                bound = corrector.bind_unpositioned_distinct_people(prompt)
                self.assertIn(expected, bound)
                self.assertNotIn("A the ", bound)
                self.assertNotIn("red-haired the ", bound)
                self.assertEqual(corrector.multi_person_role_issues(bound), [])

    def test_three_unique_roles_receive_three_distinct_positions(self):
        prompt = "A doctor, nurse, and patient wait in a clinic."

        bound = corrector.bind_unpositioned_distinct_people(prompt)

        self.assertIn("doctor on image-left", bound)
        self.assertIn("nurse at image-center", bound)
        self.assertIn("patient on image-right", bound)
        self.assertEqual(corrector.multi_person_role_issues(bound), [])

    def test_environment_positions_do_not_count_as_person_bindings(self):
        prompt = (
            "A woman and man in a room, with a lamp on the left and a window "
            "on the right."
        )

        issues = corrector.multi_person_role_issues(prompt)
        bound = corrector.bind_unpositioned_distinct_people(prompt)

        self.assertEqual(issues, [])
        self.assertIn("woman on image-left", bound)
        self.assertIn("man on image-right", bound)
        self.assertEqual(corrector.multi_person_role_issues(bound), [])

    def test_repeated_clear_roles_do_not_require_invented_frame_positions(self):
        prompt = (
            "A doctor holds a patient's arm during an examination. "
            "The doctor checks the bandage while the patient sits calmly."
        )

        self.assertTrue(corrector.appears_multi_person_scene(prompt))
        self.assertEqual(corrector._unique_person_descriptors(prompt), [])
        self.assertEqual(corrector.multi_person_role_issues(prompt), [])

    def test_clear_pair_group_pronouns_do_not_require_frame_positions(self):
        prompt = (
            "A doctor greets a patient at the clinic entrance. "
            "They walk toward the examination room together."
        )

        self.assertEqual(corrector.multi_person_role_issues(prompt), [])

    def test_post_completion_accepts_clear_repeated_roles_without_positions(self):
        prompt = (
            "A doctor holds a patient's arm during an examination. "
            "The doctor checks the bandage while the patient sits calmly."
        )
        with patch(
            "krea_prompt_corrector.chat_completion",
            return_value=prompt,
        ) as completion:
            result = corrector.post_chat_completion(
                base_url="http://127.0.0.1:1234/v1",
                model="test-4b",
                prompt=prompt,
                temperature=0.2,
                max_tokens=400,
                timeout=30,
                api_key="test",
                output_length="Concise",
                audit_repair=False,
                altered_text_encoder=False,
            )

        self.assertEqual(completion.call_count, 1)
        self.assertEqual(result, prompt)
        self.assertEqual(corrector.multi_person_role_issues(result), [])

    def test_extended_relational_bindings_are_accepted(self):
        for prompt in (
            "The woman leans above the man on a sofa.",
            "The guard stands between the doctor and the patient.",
            "The dancer straddles the other dancer.",
        ):
            with self.subTest(prompt=prompt):
                issues = corrector.multi_person_role_issues(prompt)
                self.assertEqual(issues, [])

    def test_spouse_pronoun_becomes_the_positioned_spouse_label(self):
        prompt = "A wife hugs her husband in a kitchen."

        bound = corrector.bind_unpositioned_distinct_people(prompt)
        resolved = corrector.resolve_unambiguous_multi_person_pronouns(bound)

        self.assertIn("wife on image-left hugs her husband on image-right", resolved)
        self.assertEqual(corrector.multi_person_role_issues(resolved), [])

    def test_nonbinary_identity_is_a_hard_source_fidelity_label(self):
        issues = corrector.gender_identity_contract_issues(
            "A woman on image-left faces a person on image-right.",
            "A woman faces a nonbinary person.",
        )

        self.assertIn(
            "missing explicit nonbinary identity label from the source",
            issues,
        )

    def test_each_other_is_not_reported_twice_for_two_distinct_roles(self):
        issues = corrector.multi_person_role_issues(
            "A doctor on image-left and patient on image-right face each other."
        )

        self.assertFalse(any("each other" in issue for issue in issues))
        self.assertFalse(any("other" in issue for issue in issues))

    def test_two_positioned_gendered_roles_allow_clear_natural_pronouns(self):
        prompt = (
            "The female ghost in the foreground lifts a blanket while the male sleeper "
            "in the background rests. Her hand reaches toward him, his posture remains "
            "relaxed, and they keep their established positions."
        )

        issues = corrector.multi_person_role_issues(prompt)
        self.assertEqual(issues, [])
        self.assertEqual(
            corrector.multi_person_role_issues(
                "The female ghost in the foreground lifts a blanket with the female ghost's "
                "right hand while the male sleeper in the background remains relaxed."
            ),
            [],
        )

        ambiguous = corrector.multi_person_role_issues(
            "Two people are in a room, they are running and grabbing without named roles."
        )
        self.assertTrue(ambiguous)

    def test_resolve_unambiguous_multi_person_pronouns_keeps_natural_language(self):
        prompt = (
            "The woman on the left reaches toward the man on the right, touching him "
            "while his blue coat catches the light."
        )

        resolved = corrector.resolve_unambiguous_multi_person_pronouns(prompt)

        self.assertEqual(resolved, prompt)
        self.assertEqual(corrector.multi_person_role_issues(resolved), [])

    def test_resolve_multi_person_pronouns_keeps_same_gender_ambiguity(self):
        prompt = (
            "Two men stand on the left and right, one reaches toward him while his "
            "blue coat catches the light."
        )

        self.assertEqual(
            corrector.resolve_unambiguous_multi_person_pronouns(prompt),
            prompt,
        )

    def test_resolve_unambiguous_group_pronouns_keeps_clear_group_reference(self):
        prompt = (
            "The woman on the left and the man on the right raise their lanterns "
            "while they face the camera, both smiling."
        )

        resolved = corrector.resolve_unambiguous_multi_person_pronouns(prompt)

        self.assertEqual(resolved, prompt)
        self.assertEqual(corrector.multi_person_role_issues(resolved), [])

    def test_gender_identity_contract_preserves_both_source_identities(self):
        source = "A woman in a red coat stands left of a man in a blue shirt."

        self.assertEqual(
            corrector.gender_identity_contract_issues(
                "The woman in a red coat on image-left faces the man in a blue shirt on image-right.",
                source,
            ),
            [],
        )
        issues = corrector.gender_identity_contract_issues(
            "Two people in coats face each other.",
            source,
        )
        self.assertIn("missing explicit female identity label from the source", issues)
        self.assertIn("missing explicit male identity label from the source", issues)

    def test_adult_toy_contract_does_not_reject_omitted_direction(self):
        source = "A solo adult woman thrusts rhythmically with a large dildo on a bed."

        weak = corrector.adult_toy_object_contract_issues(
            "A solo adult woman thrusts rhythmically with a large dildo on a bed.",
            source,
        )
        strong = corrector.adult_toy_object_contract_issues(
            "A solo adult woman uses a large dildo. The dildo is a separate manufactured "
            "sex toy with a visible base and exposed outer contour. Its rounded insertion "
            "tip points toward the intended body-contact point; the base or handle remains "
            "outside on the operator side and points away.",
            source,
        )

        self.assertEqual(weak, [])
        self.assertEqual(strong, [])

    def test_adult_toy_contract_rejects_a_dildo_facing_the_wrong_way(self):
        source = "A solo adult woman uses a dildo vaginally."
        candidate = (
            "The dildo is a separate manufactured sex toy with a visible contour. "
            "Its wider base faces toward the vaginal opening while its rounded insertion "
            "tip points away from the woman's body."
        )

        issues = corrector.adult_toy_object_contract_issues(candidate, source)

        self.assertTrue(any("dildo is reversed" in issue for issue in issues))

    def test_adult_toy_contract_leaves_direct_use_wording_unchanged(self):
        for source in (
            "A solo adult woman uses a dildo vaginally.",
            "A solo adult man uses a dildo anally.",
            "dildo in vagina",
        ):
            with self.subTest(source=source):
                repaired = corrector.enforce_adult_toy_object_contract(source, source)

                self.assertEqual(repaired, source)
                self.assertNotIn("rounded insertion tip", repaired)
                self.assertNotIn("base or handle", repaired)
                self.assertEqual(
                    corrector.adult_toy_object_contract_issues(repaired, source),
                    [],
                )

    def test_adult_toy_contract_ignores_excluded_objects(self):
        source = "A solo adult woman poses on a bed with no dildo or other sex toy."

        self.assertEqual(corrector.requested_adult_toy_objects(source), [])
        self.assertEqual(
            corrector.enforce_adult_toy_object_contract(
                "A solo adult woman poses on a bed.",
                source,
            ),
            "A solo adult woman poses on a bed.",
        )

    def test_adult_toy_direction_ignores_products_props_and_double_ended_designs(self):
        cases = (
            "A boxed dildo lies unopened beside a vaginal anatomy textbook.",
            "A product photograph of a double-ended dildo on a white background.",
            "A dildo sex toy lies unused on a bedside table.",
        )

        for source in cases:
            with self.subTest(source=source):
                self.assertEqual(
                    corrector.enforce_adult_toy_object_contract(source, source),
                    source,
                )
                self.assertEqual(
                    corrector.adult_toy_object_contract_issues(source, source),
                    [],
                )

        self.assertEqual(
            [
                label
                for label, _pattern in corrector.requested_adult_toy_objects(
                    cases[-1]
                )
            ],
            ["dildo"],
        )

    def test_safe_for_work_product_prompt_is_not_blocked_by_adult_toy_contract(self):
        source = "A product photograph of a dildo on a white background."
        candidate = "A clean product photograph on a white background."

        with patch(
            "krea_prompt_corrector.chat_completion",
            return_value=candidate,
        ):
            result = corrector.post_chat_completion(
                base_url="http://127.0.0.1:1234/v1",
                model="test-4b",
                prompt=source,
                temperature=0.1,
                max_tokens=300,
                timeout=30,
                api_key="test",
                safe_for_work=True,
                audit_repair=False,
            )

        self.assertEqual(result, candidate)

    def test_adult_toy_contract_does_not_append_object_geometry(self):
        source = "A solo adult woman thrusts rhythmically with a large dildo on a bed."
        repaired = corrector.enforce_adult_toy_object_contract(
            "A solo adult woman thrusts rhythmically with a large dildo on a bed.",
            source,
        )

        self.assertEqual(repaired, source)
        self.assertNotIn("rounded insertion tip", repaired)
        self.assertNotIn("base or handle", repaired)
        self.assertEqual(
            corrector.adult_toy_object_contract_issues(repaired, source),
            [],
        )

    def test_adult_toy_contract_validates_later_clarifying_mention_in_expanded_prompt(self):
        source = "A solo adult woman uses a large dildo on a bed."
        filler = " ".join(["warm light shapes the room"] * 60)
        candidate = (
            f"A solo adult woman uses a large dildo on a bed. {filler}. "
            "The dildo is a separate manufactured sex toy with a visible base and "
            "exposed outer contour. Its rounded insertion tip points toward the intended "
            "body-contact point; the base or handle remains outside on the operator side "
            "and points away."
        )

        self.assertGreater(
            candidate.lower().rfind("dildo") - candidate.lower().find("dildo"),
            220,
        )
        self.assertEqual(
            corrector.adult_toy_object_contract_issues(candidate, source),
            [],
        )

    def test_adult_toy_enforcement_does_not_interrupt_the_first_object_sentence(self):
        source = "A solo adult woman uses a large dildo on a bed."
        candidate = (
            "A solo adult woman uses a large dildo on a bed. "
            + " ".join(["Warm light shapes the surrounding room."] * 30)
        )

        repaired = corrector.enforce_adult_toy_object_contract(candidate, source)
        self.assertEqual(repaired, candidate)
        self.assertNotIn("Keep the dildo visibly separate from the body", repaired)
        self.assertEqual(
            corrector.adult_toy_object_contract_issues(repaired, source),
            [],
        )

    def test_inserted_object_contract_leaves_clear_direct_contact_wording_alone(self):
        for source in (
            "A solo adult woman inserts a glass bottle into her vagina.",
            "A solo adult woman slides a peeled cucumber inside her vagina.",
            "A glass bottle is inserted into the adult woman's vagina.",
            "A solo adult man puts a smooth object in his anus.",
        ):
            with self.subTest(source=source):
                repaired = corrector.enforce_inserted_object_contract(source, source)
                self.assertEqual(repaired, source)
                self.assertNotIn("continuous outer contour", repaired)
                self.assertEqual(
                    corrector.inserted_object_contract_issues(repaired, source),
                    [],
                )

    def test_inserted_object_contract_does_not_reclassify_body_parts_as_objects(self):
        for source in (
            "A solo adult woman inserts two fingers into her vagina.",
            "A solo adult man slides his tongue inside the other adult's anus.",
        ):
            with self.subTest(source=source):
                self.assertEqual(
                    corrector.requested_inserted_object_targets(source),
                    [],
                )
                self.assertEqual(
                    corrector.enforce_inserted_object_contract(source, source),
                    source,
                )

    def test_inserted_object_contract_ignores_negative_constraints(self):
        source = "A solo adult woman poses with no bottle inserted into her vagina."

        self.assertEqual(corrector.requested_inserted_object_targets(source), [])
        self.assertEqual(
            corrector.inserted_object_contract_issues(source, source),
            [],
        )

    def test_missing_inserted_object_contact_target_is_a_hard_contract(self):
        source = "A solo adult woman inserts a glass bottle into her vagina."
        issues = corrector.final_compliance_issues(
            "A solo adult woman holds a glass bottle.",
            original_prompt=source,
            output_length="Concise",
            explicit_nsfw=True,
        )

        contract_issue = next(
            issue
            for issue in issues
            if issue.startswith("Inserted object/body contact contract")
        )
        self.assertIn("missing requested contact at the vaginal opening", contract_issue)
        self.assertTrue(corrector.is_hard_compliance_issue(contract_issue))

    def test_unrequested_gender_and_cross_gender_anatomy_are_hard_failures(self):
        source = "A solo adult woman uses a large dildo on a bed."
        issues = corrector.final_compliance_issues(
            "A solo adult transgender futanari woman with an erect penis uses a large "
            "dildo. The dildo is a separate manufactured sex toy with a visible base "
            "and contact boundary.",
            original_prompt=source,
            output_length="Concise",
            explicit_nsfw=True,
        )

        trait_issue = next(
            issue
            for issue in issues
            if issue.startswith("Unrequested gender/anatomy traits")
        )
        self.assertIn("unrequested futanari", trait_issue)
        self.assertIn("unrequested transgender", trait_issue)
        self.assertIn("male genital anatomy added to a female-only source", trait_issue)
        self.assertTrue(corrector.is_hard_compliance_issue(trait_issue))

    def test_explicitly_requested_gender_traits_remain_allowed(self):
        source = "A solo adult transgender futanari woman with a penis uses a dildo."
        candidate = (
            "A solo adult transgender futanari woman with a penis uses a dildo. "
            "The dildo is a separate manufactured sex toy with a visible base and "
            "contact boundary."
        )

        self.assertEqual(
            corrector.unrequested_gender_trait_issues(candidate, source),
            [],
        )

    def test_support_fields_authorize_the_anatomy_they_explicitly_require(self):
        candidate = (
            "A mature adult woman on image-left performs oral stimulation on "
            "an adult partner on image-right's penis."
        )
        authorized_source = (
            "A mature adult woman masturbates with a dildo.\n"
            "Required concept: blowjob\nWeighted priority: penis in mouth:1.2"
        )
        self.assertEqual(
            corrector.unrequested_gender_trait_issues(
                candidate,
                authorized_source,
            ),
            [],
        )

    def test_support_implied_partner_gets_adult_role_and_position_binding(self):
        source = (
            "A mature adult woman masturbates with a large green dildo in a "
            "sunny public park."
        )
        support = "blowjob:1.3, penis in mouth:1.2"
        instruction = corrector.explicit_support_participant_contract(
            source,
            support,
        )
        self.assertIn("exactly one adult partner", instruction)
        self.assertIn("image-left", instruction)
        self.assertIn("image-right", instruction)

        fallback = corrector.apply_explicit_support_participant_contract(
            source,
            source,
            support,
        )
        self.assertEqual(corrector.minor_sexual_content_issues(fallback), [])
        self.assertEqual(corrector.multi_person_role_issues(fallback), [])
        self.assertIn("visible mouth-to-penis contact", fallback)

    def test_support_implied_oral_act_rejects_actor_receiver_reversal(self):
        source = "A mature adult woman masturbates with a green dildo."
        support = "blowjob:1.3, penis in mouth:1.2"
        reversed_roles = (
            "A mature adult woman on image-left uses a green dildo while an "
            "adult man on image-right leans forward, his tongue stimulating his "
            "own penis inside her mouth."
        )
        issues = corrector.explicit_support_participant_issues(
            reversed_roles,
            source,
            support,
        )
        self.assertEqual(len(issues), 1)
        self.assertTrue(corrector.is_hard_compliance_issue(issues[0]))

        correctly_bound = (
            "A mature adult woman on image-left performs oral stimulation on "
            "an adult partner on image-right's penis with visible mouth-to-penis contact."
        )
        self.assertEqual(
            corrector.explicit_support_participant_issues(
                correctly_bound,
                source,
                support,
            ),
            [],
        )

    def test_explicit_translation_repairs_singular_self_reference_and_genital_grammar(self):
        translated = corrector.translate_explicit_adult_language(
            "A mature adult woman masturbating while pushing it inside her hot and pussy."
        )
        self.assertIn("her own genitals", translated)
        self.assertNotIn("their own genitals", translated)
        self.assertIn("her hot vulva", translated)
        self.assertEqual(corrector.explicit_adult_grammar_issues(translated), [])

    def test_explicit_translation_repairs_clear_pronoun_self_reference(self):
        translated = corrector.translate_explicit_adult_language(
            "A mature adult woman kneels in a park. "
            "During self-stimulation of their own genitals, she grips a dildo. "
            "She performs self-stimulation of their own genitals."
        )
        self.assertIn(
            "She performs self-stimulation of her own genitals",
            translated,
        )
        self.assertIn(
            "During self-stimulation of her own genitals",
            translated,
        )
        self.assertNotIn("their own genitals", translated)

    def test_deterministic_fallback_preserves_blank_line_separated_source(self):
        source = (
            "A mature adult woman masturbates with a green dildo.\n\n"
            "She pushes it inside her wet vulva with both hands.\n\n"
            "She kneels in a sunny public park."
        )
        fallback = corrector.deterministic_fidelity_fallback(
            source,
            concept_keywords="blowjob",
            weighted_terms="penis in mouth:1.2",
        )
        self.assertIn("wet vulva", fallback)
        self.assertIn("both hands", fallback)
        self.assertIn("sunny public park", fallback)
        self.assertIn("adult partner", fallback)

    def test_explicit_solo_source_does_not_gain_support_implied_partner(self):
        self.assertEqual(
            corrector.explicit_support_participant_contract(
                "A solo mature adult woman masturbates with a dildo.",
                "blowjob:1.3, penis in mouth:1.2",
            ),
            "",
        )

    def test_transgender_source_does_not_apply_binary_anatomy_assumption(self):
        self.assertEqual(
            corrector.unrequested_gender_trait_issues(
                "A solo adult transgender woman with a penis poses on a bed.",
                "A solo adult transgender woman poses on a bed.",
            ),
            [],
        )

    def test_post_completion_preserves_simple_adult_toy_wording(self):
        source = "A solo adult woman thrusts rhythmically with a large dildo on a bed."
        candidate = "A solo adult woman thrusts rhythmically with a large dildo on a bed."

        with patch(
            "krea_prompt_corrector.chat_completion",
            return_value=candidate,
        ):
            result = corrector.post_chat_completion(
                base_url="http://127.0.0.1:1234/v1",
                model="test-4b",
                prompt=source,
                temperature=0.2,
                max_tokens=400,
                timeout=30,
                api_key="test",
                explicit_nsfw=True,
            )

        self.assertEqual(result, candidate)
        self.assertNotIn("rounded insertion tip", result)
        self.assertNotIn("base or handle", result)
        self.assertNotRegex(result.lower(), r"\b(?:futanari|futa|transgender|shemale)\b")

    def test_post_completion_keeps_plain_dildo_in_vagina_without_geometry(self):
        source = "dildo in vagina"

        with patch(
            "krea_prompt_corrector.chat_completion",
            return_value=source,
        ) as completion:
            result = corrector.post_chat_completion(
                base_url="http://127.0.0.1:1234/v1",
                model="test-4b",
                prompt=source,
                temperature=0.2,
                max_tokens=200,
                timeout=30,
                api_key="test",
                explicit_nsfw=True,
                audit_repair=False,
                final_gate_repair=False,
        )

        self.assertEqual(completion.call_count, 1)
        self.assertEqual(result, source)
        self.assertNotIn("insertion tip", result)
        self.assertNotIn("base or handle", result)
        sent_message = completion.call_args.kwargs["messages"][1]["content"]
        self.assertIn("literal wording=dildo in vagina", sent_message)
        self.assertNotIn("rounded insertion tip", sent_message)
        issues = corrector.final_compliance_issues(
            result,
            original_prompt=source,
            output_length="Concise",
            explicit_nsfw=True,
        )
        hard_issues, _soft_issues = corrector.split_compliance_issues(issues)
        self.assertEqual(hard_issues, [])

    def test_post_completion_translates_explicit_slang_before_final_validation(self):
        source = (
            "A solo MILF woman fucking herself with a thick sextoy, hammering "
            "a dildo inside her wet pussy."
        )
        with patch(
            "krea_prompt_corrector.chat_completion",
            return_value=source,
        ):
            result = corrector.post_chat_completion(
                base_url="http://127.0.0.1:1234/v1",
                model="test-4b",
                prompt=source,
                temperature=0.2,
                max_tokens=400,
                timeout=30,
                api_key="test",
                explicit_nsfw=True,
                audit_repair=False,
                final_gate_repair=False,
            )

        self.assertIn("mature adult woman", result)
        self.assertIn("performing self-penetration for genital stimulation", result)
        self.assertIn("repeatedly thrusting a dildo", result)
        self.assertIn("wet vulva", result)
        self.assertEqual(corrector.explicit_adult_language_terms(result), [])

    def test_post_completion_accepts_canonical_selected_adult_concepts(self):
        source = (
            "A mature adult woman uses a large green dildo for vaginal penetration "
            "while performing a blowjob on an adult man in an alley."
        )
        candidate = (
            "On the left, a mature adult woman straddles a large green dildo "
            "with rhythmic penetrative use for vaginal penetration while "
            "performing oral stimulation of the penis of an adult man standing "
            "on the right in a rain-darkened alley."
        )
        diagnostics: list[str] = []

        with patch(
            "krea_prompt_corrector.chat_completion",
            return_value=candidate,
        ) as completion:
            result = corrector.post_chat_completion(
                base_url="http://127.0.0.1:1234/v1",
                model="test-4b",
                prompt=source,
                temperature=0.2,
                max_tokens=400,
                timeout=30,
                api_key="test",
                concept_keywords="blowjob, dildo fucking",
                weighted_terms="blowjob:1.3, dildo fucking:1.75",
                explicit_nsfw=True,
                audit_repair=True,
                diagnostic_callback=diagnostics.append,
            )

        self.assertEqual(completion.call_count, 2)
        self.assertEqual(
            corrector.missing_required_concepts(
                result,
                "blowjob, dildo fucking",
            ),
            [],
        )
        self.assertEqual(
            corrector.missing_weighted_terms(
                result,
                "blowjob:1.3, dildo fucking:1.75",
            ),
            [],
        )
        self.assertFalse(
            any("Missing required concepts" in message for message in diagnostics)
        )

    def test_untranslated_explicit_slang_is_a_hard_contract_issue(self):
        issues = corrector.final_compliance_issues(
            "A solo adult woman is fucking herself.",
            original_prompt="A solo adult woman masturbates.",
            output_length="Concise",
            explicit_nsfw=True,
        )

        slang_issue = next(
            issue
            for issue in issues
            if issue.startswith("Untranslated explicit adult slang")
        )
        self.assertTrue(corrector.is_hard_compliance_issue(slang_issue))

    def test_explicit_meme_completion_translates_scene_but_preserves_caption(self):
        brief = (
            "Create an original single-image meme showing a solo adult woman "
            "masturbating with a dildo. Keep quoted caption text exact."
        )
        response = (
            "A square reaction meme showing a solo MILF woman fucking herself "
            "with a thick sextoy while hammering a dildo inside her wet pussy. "
            'Place the top caption "MILF HAMMER TIME" in bold white text.'
        )
        with patch(
            "krea_prompt_corrector.chat_completion",
            return_value=response,
        ) as completion:
            result = corrector.post_meme_completion(
                base_url="http://127.0.0.1:1234/v1",
                model="test-4b",
                prompt=brief,
                generator_target="Krea 2",
                temperature=0.2,
                max_tokens=800,
                timeout=30,
                api_key="test",
                explicit_nsfw=True,
            )

        self.assertEqual(completion.call_count, 1)
        self.assertIn("solo mature adult woman", result)
        self.assertIn("performing self-penetration for genital stimulation", result)
        self.assertIn("wet vulva", result)
        self.assertIn('"MILF HAMMER TIME"', result)
        self.assertEqual(corrector.explicit_adult_language_terms(result), [])

    def test_post_completion_reports_candidate_rejection_and_repair_difficulty(self):
        source = "A solo adult woman uses a dildo vaginally."
        reversed_candidate = (
            "A solo adult woman uses a dildo vaginally. The dildo is a separate "
            "manufactured object with a visible contour. Its wider base faces toward "
            "the vaginal opening while its rounded insertion tip points away from the body."
        )
        repaired_candidate = (
            "A solo adult woman uses a dildo vaginally. The dildo is a separate "
            "manufactured sex toy with a visible outer contour. Its rounded insertion "
            "tip points toward the vaginal opening while its wider base or handle "
            "remains outside on the operator side and points away from the contact point."
        )
        diagnostics = []

        with patch(
            "krea_prompt_corrector.chat_completion",
            side_effect=[reversed_candidate, repaired_candidate],
        ) as completion:
            result = corrector.post_chat_completion(
                base_url="http://127.0.0.1:1234/v1",
                model="test-4b",
                prompt=source,
                temperature=0.2,
                max_tokens=500,
                timeout=30,
                api_key="test",
                explicit_nsfw=True,
                diagnostic_callback=diagnostics.append,
            )

        self.assertEqual(completion.call_count, 2)
        self.assertIn("tip points toward the vaginal opening", result)
        self.assertTrue(
            any(
                "Initial model candidate rejected by validation" in message
                and "dildo is reversed" in message
                for message in diagnostics
            )
        )
        self.assertTrue(
            any(
                "Final repair attempt 1/1 is addressing" in message
                for message in diagnostics
            )
        )

    def test_reported_explicit_failure_uses_clean_validated_fallback_and_logs_selection(self):
        source = (
            "a mature adult woman genital stimulation, kneeing, with a big, thick, "
            "knotty, green dildo sex toy. she forces the dildo inside her hot and wet "
            "vulva, with painful force, with both hands holding the base riding it. "
            "she have trouble pushing it in because its so big. she looks at the camera, "
            "smiling with lust. set in public park, mid day sunny. while she is riding "
            "the dildo she blowjob's a mans. penis in mouth."
        )
        story = "unconditional acceptance with open body language"
        rejected = (
            "A mature adult woman kneels in a sunny public park while posing beside "
            "a green object."
        )
        diagnostics: list[str] = []

        with patch(
            "krea_prompt_corrector.chat_completion",
            side_effect=[rejected, rejected, rejected],
        ) as completion:
            result = corrector.post_chat_completion(
                base_url="http://127.0.0.1:1234/v1",
                model="test-4b",
                prompt=source,
                temperature=0.2,
                max_tokens=500,
                timeout=30,
                api_key="test",
                concept_keywords="blowjob, dildo fucking",
                weighted_terms="blowjob:1.3, dildo fucking:1.75",
                story_elements=story,
                explicit_nsfw=True,
                audit_repair=True,
                content_format="Single Image",
                diagnostic_callback=diagnostics.append,
            )

        self.assertEqual(completion.call_count, 3)
        self.assertIn("kneeling", result)
        self.assertIn("she has trouble", result)
        self.assertIn("because it is so big", result)
        self.assertIn("the penis of an adult man", result)
        self.assertIn("visible mouth-to-penis contact", result)
        self.assertIn("relaxed shoulders", result)
        self.assertIn("open posture", result)
        self.assertNotIn("open palms", result)
        self.assertEqual(corrector.explicit_adult_grammar_issues(result), [])
        self.assertEqual(
            corrector.final_compliance_issues(
                result,
                original_prompt=source,
                concept_keywords="blowjob, dildo fucking",
                weighted_terms="blowjob:1.3, dildo fucking:1.75",
                story_elements=story,
                explicit_nsfw=True,
                content_format="Single Image",
            ),
            [],
        )
        self.assertIn(
            "Deterministic fidelity fallback passed validation and was selected.",
            diagnostics,
        )

    def test_post_completion_reports_optional_audit_failure_and_continues(self):
        source = (
            "A red sports car parked beneath warm streetlights on a rainy city street."
        )
        diagnostics = []

        with patch(
            "krea_prompt_corrector.chat_completion",
            side_effect=[source, RuntimeError("audit response was empty")],
        ):
            result = corrector.post_chat_completion(
                base_url="http://127.0.0.1:1234/v1",
                model="test-4b",
                prompt=source,
                temperature=0.2,
                max_tokens=500,
                timeout=30,
                api_key="test",
                audit_repair=True,
                final_gate_repair=False,
                diagnostic_callback=diagnostics.append,
            )

        self.assertEqual(result, source)
        self.assertTrue(
            any(
                "Optional audit model call failed" in message
                and "audit response was empty" in message
                for message in diagnostics
            )
        )

    def test_post_completion_does_not_repeat_advisory_activity_messages(self):
        source = "A red car."
        diagnostics = []

        with patch(
            "krea_prompt_corrector.chat_completion",
            return_value=source,
        ):
            result = corrector.post_chat_completion(
                base_url="http://127.0.0.1:1234/v1",
                model="test-4b",
                prompt=source,
                temperature=0.1,
                max_tokens=100,
                timeout=30,
                api_key="test",
                diagnostic_callback=diagnostics.append,
            )

        self.assertEqual(result, source)
        self.assertEqual(diagnostics, [])

    def test_expanded_post_completion_does_not_append_toy_geometry(self):
        source = "A solo adult woman uses a large dildo while posing on a bed."
        candidate = (
            "A solo adult woman uses a large dildo while posing on a bed. "
            + " ".join(
                [
                    "Warm window light reveals textured linen, relaxed posture, "
                    "natural skin detail, and a quiet bedroom atmosphere."
                ]
                * 12
            )
        )

        with patch(
            "krea_prompt_corrector.chat_completion",
            return_value=candidate,
        ) as completion:
            result = corrector.post_chat_completion(
                base_url="http://127.0.0.1:1234/v1",
                model="test-4b",
                prompt=source,
                temperature=0.2,
                max_tokens=800,
                timeout=30,
                api_key="test",
                explicit_nsfw=True,
                output_length="Expanded",
                audit_repair=False,
            )

        self.assertEqual(completion.call_count, 1)
        self.assertNotIn("Keep the dildo visibly separate from the body", result)
        self.assertNotIn("rounded insertion tip", result)
        self.assertEqual(
            corrector.adult_toy_object_contract_issues(result, source),
            [],
        )

    def test_post_completion_preserves_clear_inserted_object_wording(self):
        source = "A solo adult woman inserts a glass bottle into her vagina."

        with patch(
            "krea_prompt_corrector.chat_completion",
            return_value=source,
        ):
            result = corrector.post_chat_completion(
                base_url="http://127.0.0.1:1234/v1",
                model="test-4b",
                prompt=source,
                temperature=0.2,
                max_tokens=400,
                timeout=30,
                api_key="test",
                explicit_nsfw=True,
            )

        self.assertEqual(result, source)
        self.assertNotIn("inserted non-anatomical item", result)
        self.assertNotIn("contact boundary", result)
        self.assertEqual(
            corrector.inserted_object_contract_issues(result, source),
            [],
        )

    def test_user_message_includes_multi_person_role_analysis(self):
        message = corrector.build_user_message(
            "a man and woman in a cave, they fight while another watches"
        )

        self.assertIn("Multi-person role binding analysis:", message)
        self.assertIn("Make it obvious who does what to whom", message)
        self.assertIn("distinct identity label", message)

    def test_final_compliance_issues_reports_multi_person_role_ambiguity(self):
        issues = corrector.final_compliance_issues(
            "Two people in a cave, they hold a torch and another person stands nearby.",
            output_length="Concise",
        )

        self.assertTrue(any("Multi-person role ambiguity" in issue for issue in issues))
        hard, _soft = corrector.split_compliance_issues(issues)
        self.assertTrue(any("Multi-person role ambiguity" in issue for issue in hard))

    def test_source_ambiguity_is_advisory_instead_of_a_terminal_contract(self):
        source = (
            "Two people stand in a cave; he holds a torch while she helps them."
        )
        issues = corrector.final_compliance_issues(
            source,
            original_prompt=source,
            output_length="Concise",
        )

        hard, soft = corrector.split_compliance_issues(issues)
        self.assertFalse(any("Multi-person role ambiguity" in issue for issue in hard))
        self.assertTrue(
            any("Source multi-person ambiguity preserved" in issue for issue in soft)
        )

    def test_post_completion_returns_preserved_source_ambiguity_without_failing(self):
        source = (
            "Two people stand in a cave; he holds a torch while she helps them."
        )
        with patch(
            "krea_prompt_corrector.chat_completion",
            return_value=source,
        ) as completion:
            result = corrector.post_chat_completion(
                base_url="http://127.0.0.1:1234/v1",
                model="test-4b",
                prompt=source,
                temperature=0.2,
                max_tokens=400,
                timeout=30,
                api_key="test",
                output_length="Concise",
                audit_repair=False,
                altered_text_encoder=False,
            )

        self.assertEqual(completion.call_count, 1)
        self.assertEqual(result, corrector.normalize_final_prompt_text(source))

    def test_final_compliance_issues_reports_dropped_gender_identity(self):
        issues = corrector.final_compliance_issues(
            "Two adults stand on opposite sides of a doorway.",
            original_prompt="A woman stands left of a man at a doorway.",
            output_length="Concise",
        )

        self.assertTrue(any("Gender identity contract" in issue for issue in issues))
        hard, _soft = corrector.split_compliance_issues(issues)
        self.assertTrue(any("Gender identity contract" in issue for issue in hard))

    def test_negative_constraints_do_not_create_text_or_people_false_positives(self):
        prompt = "A matte black bottle centered on white, no people, no visible text."

        self.assertEqual(corrector.rendered_text_issues(prompt, prompt), [])
        self.assertEqual(corrector.multi_person_role_issues(prompt), [])

    def test_count_contract_detects_changed_object_count(self):
        original = "Exactly two red cups beside one blue plate."

        self.assertEqual(
            corrector.count_contract_issues(
                "Exactly two red cups beside one blue plate.", original
            ),
            [],
        )
        self.assertTrue(
            corrector.count_contract_issues(
                "Three red cups beside one blue plate.", original
            )
        )
        self.assertTrue(
            corrector.count_contract_issues(
                "Exactly two red cups beside one blue plate, three cups visible in a reflection.",
                original,
            )
        )

    def test_count_contract_accepts_pair_and_dozen_equivalents(self):
        self.assertEqual(
            corrector.count_contract_issues(
                "A pair of red roses rests in a vase.",
                "Exactly two red roses rest in a vase.",
            ),
            [],
        )
        self.assertEqual(
            corrector.count_contract_issues(
                "A dozen candles stand on the cake.",
                "Exactly twelve candles stand on the cake.",
            ),
            [],
        )

    def test_spatial_contract_detects_changed_side(self):
        original = "A red cup on the left and a blue vase on the right."

        self.assertEqual(corrector.spatial_contract_issues(original, original), [])
        issues = corrector.spatial_contract_issues(
            "A red cup on the right and a blue vase on the left.", original
        )
        self.assertTrue(issues)

    def test_spatial_contract_accepts_equivalent_relation_wording(self):
        cases = (
            ("A cup on the left of a vase.", "A cup on image-left of a vase."),
            ("A cup sits inside a cabinet.", "A cup sits within a cabinet."),
            ("A chair stands behind a desk.", "A chair stands at the rear of a desk."),
        )
        for source, candidate in cases:
            with self.subTest(source=source):
                self.assertEqual(
                    corrector.spatial_contract_issues(candidate, source),
                    [],
                )

    def test_sexual_inside_contact_is_not_a_generic_spatial_contract(self):
        for source in (
            "she forces the dildo inside her hot",
            "A solo adult woman thrusts a dildo inside her wet vagina.",
            "An adult man finishes inside her.",
        ):
            with self.subTest(source=source):
                self.assertEqual(corrector.extract_spatial_contracts(source), [])
                self.assertFalse(
                    any(
                        issue.startswith("Spatial contract")
                        for issue in corrector.final_compliance_issues(
                            "A solo adult woman uses a dildo vaginally.",
                            original_prompt=source,
                            output_length="Concise",
                            explicit_nsfw=True,
                        )
                    )
                )

    def test_ordinary_inside_relation_remains_a_spatial_contract(self):
        source = "A red cup sits inside a wooden cabinet."

        self.assertTrue(corrector.extract_spatial_contracts(source))
        self.assertTrue(
            corrector.spatial_contract_issues(
                "A red cup sits on a wooden cabinet.",
                source,
            )
        )

    def test_exclusions_and_unexpected_scripts_are_hard_contracts(self):
        original = "A clean product photo without flowers or people."
        clean = "A clean product photo without flowers or people."
        broken = "A clean product photo with pink flowers. 产品摄影"

        self.assertEqual(corrector.exclusion_contract_issues(clean, original), [])
        issues = corrector.final_compliance_issues(
            broken,
            original_prompt=original,
            output_length="Concise",
        )
        hard, _soft = corrector.split_compliance_issues(issues)
        self.assertTrue(any("Excluded content appears positively" in item for item in hard))
        self.assertTrue(any("Unexpected output language/script" in item for item in hard))

        comma_prompt = "A centered black bottle, no people, clean white background."
        self.assertEqual(corrector.extract_excluded_terms(comma_prompt), ["people"])
        self.assertTrue(
            corrector.exclusion_contract_issues(
                "A centered black bottle beside one smiling woman.", comma_prompt
            )
        )

    def test_exclusion_contract_understands_absence_forms_and_people_proxies(self):
        source = "A clean product photo without flowers or people."
        self.assertEqual(
            corrector.exclusion_contract_issues(
                "A flower-free vase in an empty studio.",
                source,
            ),
            [],
        )
        self.assertTrue(
            corrector.exclusion_contract_issues(
                "A vase displayed in a crowded plaza.",
                source,
            )
        )

    def test_unexpected_script_stripper_removes_leaks_but_preserves_user_text(self):
        english_source = "A weathered portrait on a gallery wall."

        self.assertEqual(
            corrector.strip_unexpected_scripts(
                "A weathered portrait 产品摄影。 on a gallery wall.",
                english_source,
            ),
            "A weathered portrait on a gallery wall.",
        )
        self.assertEqual(
            corrector.strip_unexpected_scripts(
                "A weathered portrait with a \U00020000 artifact.",
                english_source,
            ),
            "A weathered portrait with a artifact.",
        )
        self.assertEqual(
            corrector.strip_unexpected_scripts(
                "A sign reading 你好 remains visible.",
                "Keep the sign text 你好 exactly as written.",
            ),
            "A sign reading 你好 remains visible.",
        )

    def test_unexpected_script_checks_cover_kana_and_hangul(self):
        original = "A clean English-language product photograph."
        issues = corrector.unexpected_script_issues(
            "A clean product photograph かな 한글.",
            original,
        )

        self.assertTrue(any("Hiragana/Katakana" in item for item in issues))
        self.assertTrue(any("Hangul" in item for item in issues))

    def test_exact_prompt_path_is_compact_and_contains_fidelity_contract(self):
        system = corrector.build_system_prompt(
            risk_level="Strict cleanup",
            preserve_strictly=True,
            enhance_actions=False,
            develop_story=False,
        )
        user = corrector.build_user_message(
            "Exactly two cups, red on the left, no people.",
            risk_level="Strict cleanup",
            develop_story=False,
        )

        self.assertIn("Fidelity is the highest priority", system)
        self.assertLess(len(system), 3000)
        self.assertIn("Hard fidelity contract", user)
        self.assertIn("Exactly two cups", user)
        self.assertIn("red on the left", user)
        self.assertIn("people", user)

    def test_exact_mode_rejects_model_drift_and_returns_faithful_fallback(self):
        response = {
            "choices": [{"message": {"content": (
                "Three blue cups on the right beside flowers and one smiling woman. 产品摄影"
            )}}]
        }
        with patch("urllib.request.urlopen", return_value=FakeResponse(response)) as request:
            result = corrector.post_chat_completion(
                base_url="http://127.0.0.1:1234/v1",
                model="qwen3-vl-4b-instruct",
                prompt=(
                    "Exactly two red cups on the left beside one blue plate, "
                    "no flowers, no people."
                ),
                temperature=0.1,
                max_tokens=300,
                timeout=5,
                api_key="test-key",
                risk_level="Strict cleanup",
                preserve_strictly=True,
                enhance_actions=False,
                develop_story=False,
            )

        self.assertEqual(
            result,
            "Exactly two red cups on the left beside one blue plate, no flowers, no people.",
        )
        self.assertEqual(request.call_count, 1)

    def test_exact_fallback_preserves_required_support_fields(self):
        response = {
            "choices": [{"message": {"content": "A generic blue car in a parking lot."}}]
        }
        with patch("urllib.request.urlopen", return_value=FakeResponse(response)):
            result = corrector.post_chat_completion(
                base_url="http://127.0.0.1:1234/v1",
                model="qwen3-vl-4b-instruct",
                prompt="A woman walks through a rainy city street.",
                temperature=0.1,
                max_tokens=300,
                timeout=5,
                api_key="test-key",
                risk_level="Strict cleanup",
                preserve_strictly=True,
                enhance_actions=False,
                develop_story=False,
                concept_keywords="red umbrella",
                focus="the red umbrella",
                weighted_terms="red umbrella:1.6",
            )

        self.assertIn("A woman walks through a rainy city street", result)
        self.assertIn(
            "The red umbrella receives strong visual prominence in the scene",
            result,
        )
        self.assertNotIn("The composition prominently features", result)
        self.assertNotIn("prominently integrate", result)
        self.assertNotIn("existing subject and object design", result)
        self.assertNotIn("Required visual elements", result)
        self.assertNotIn("Prominent visual elements", result)
        self.assertFalse(
            corrector.missing_required_concepts(result, "red umbrella")
        )
        hard, _soft = corrector.split_compliance_issues(
            corrector.final_compliance_issues(
                result,
                original_prompt="A woman walks through a rainy city street.",
                concept_keywords="red umbrella",
                focus="the red umbrella",
                weighted_terms="red umbrella:1.6",
            )
        )
        self.assertEqual(hard, [])

    def test_exact_fallback_binds_missing_mix_influence_to_existing_concept(self):
        result = corrector.deterministic_fidelity_fallback(
            "A large green dildo rests at the center of a sunlit park scene.",
            concept_keywords="dildo, xenomorph",
            weighted_terms="dildo:1.4, xenomorph:2.1",
        )

        self.assertIn(
            "The dildo's design prominently incorporates xenomorph-inspired visual traits",
            result,
        )
        self.assertNotIn("The composition prominently features xenomorph", result)
        self.assertFalse(
            corrector.missing_required_concepts(result, "dildo, xenomorph")
        )
        self.assertFalse(
            corrector.missing_weighted_terms(result, "dildo:1.4, xenomorph:2.1")
        )

    def test_model_prominence_boilerplate_binds_mix_to_existing_subject(self):
        response = {
            "choices": [
                {
                    "message": {
                        "content": (
                            "A large green dildo rests at the center of a sunlit park. "
                            "The composition prominently features xenomorph."
                        )
                    }
                }
            ]
        }
        with patch("urllib.request.urlopen", return_value=FakeResponse(response)):
            result = corrector.post_chat_completion(
                base_url="http://127.0.0.1:1234/v1",
                model="qwen3-vl-4b-instruct",
                prompt="A large green dildo rests at the center of a sunlit park.",
                temperature=0.1,
                max_tokens=300,
                timeout=5,
                api_key="test-key",
                concept_keywords="dildo, xenomorph",
                weighted_terms="dildo:1.4, xenomorph:2.1",
                final_gate_repair=False,
            )

        self.assertIn(
            "The dildo's design prominently incorporates xenomorph-inspired visual traits",
            result,
        )
        self.assertNotIn("The composition prominently features xenomorph", result)
        self.assertFalse(
            corrector.missing_required_concepts(result, "dildo, xenomorph")
        )
        self.assertFalse(
            corrector.missing_weighted_terms(result, "dildo:1.4, xenomorph:2.1")
        )

    def test_exact_mode_uses_fallback_when_model_returns_empty_content(self):
        response = {"choices": [{"message": {"content": ""}}]}
        with patch("urllib.request.urlopen", return_value=FakeResponse(response)):
            result = corrector.post_chat_completion(
                base_url="http://127.0.0.1:1234/v1",
                model="qwen3-vl-4b-instruct",
                prompt="A black bottle centered on white.",
                temperature=0.1,
                max_tokens=200,
                timeout=5,
                api_key="test-key",
                risk_level="Strict cleanup",
                preserve_strictly=True,
                enhance_actions=False,
                develop_story=False,
                concept_keywords="gold cap",
            )

        self.assertIn("A black bottle centered on white", result)
        self.assertIn("gold cap", result)

    def test_normalize_final_prompt_text_translates_visual_slang(self):
        prompt = 'a baddie with drip and rizz, neon vibes, sign says "lit"'

        normalized = corrector.normalize_final_prompt_text(prompt)

        self.assertIn("confident glamorous stylish person", normalized)
        self.assertIn("fashionable streetwear and polished accessories", normalized)
        self.assertIn("charismatic confident expression", normalized)
        self.assertIn("atmospheric mood", normalized)
        self.assertIn('"lit"', normalized)
        self.assertNotIn(" baddie ", f" {normalized.lower()} ")
        self.assertNotIn(" drip ", f" {normalized.lower()} ")

    def test_visual_slang_controls_match_their_normalized_visible_meaning(self):
        for _pattern, replacement, label in corrector.VISUAL_SLANG_TRANSLATIONS:
            with self.subTest(label=label):
                canonical = corrector.translate_visual_slang(label)
                self.assertNotEqual(canonical, label)
                self.assertEqual(
                    corrector.missing_required_concepts(canonical, label),
                    [],
                )
                self.assertEqual(
                    corrector.missing_weighted_terms(
                        canonical,
                        f"{label}:1.7",
                    ),
                    [],
                )
                self.assertIsNone(corrector.focus_issue(canonical, label))
                self.assertEqual(
                    corrector.single_image_story_element_issues(canonical, label),
                    [],
                )
                self.assertEqual(
                    corrector.intent_lock_issues(
                        label,
                        canonical,
                        goal_headline=label,
                    ),
                    [],
                )
                self.assertEqual(
                    corrector.translate_visual_slang(replacement),
                    replacement,
                )

    def test_explicit_adult_language_translates_slang_to_renderable_terms(self):
        prompt = (
            "A solo MILF woman fucking herself with a big thick sextoy while "
            "hammering a dildo inside her wet pussy, with visible pre-cum."
        )

        translated = corrector.translate_explicit_adult_language(prompt)

        self.assertIn("mature adult woman", translated)
        self.assertIn("performing self-penetration for genital stimulation", translated)
        self.assertIn("sex toy", translated)
        self.assertIn("repeatedly thrusting a dildo", translated)
        self.assertIn("wet vulva", translated)
        self.assertIn("pre-ejaculate fluid", translated)
        self.assertNotIn("woman woman", translated)
        self.assertEqual(corrector.explicit_adult_language_terms(translated), [])

    def test_explicit_adult_pipeline_repairs_malformed_possessive_action_grammar(self):
        source = (
            "A mature adult woman, kneeing beside a large dildo. "
            "She have trouble because its so large while she blowjob's a mans. "
            "penis in mouth."
        )

        translated = corrector.translate_explicit_adult_language(
            corrector.normalize_concept_text(source)
        )

        self.assertIn("kneeling", translated)
        self.assertIn("She has trouble because it is so large", translated)
        self.assertIn(
            "she performs oral stimulation on the penis of an adult man "
            "with visible mouth-to-penis contact",
            translated,
        )
        self.assertNotIn("penis's", translated)
        self.assertNotIn("a mans", translated)
        self.assertEqual(corrector.explicit_adult_grammar_issues(translated), [])
        self.assertEqual(
            corrector.translate_explicit_adult_language(translated),
            translated,
        )

    def test_explicit_adult_pipeline_repairs_similar_manual_action_grammar(self):
        source = "She handjob's a mans penis while they has a steady pose."

        translated = corrector.translate_explicit_adult_language(source)

        self.assertEqual(
            translated,
            (
                "She manually stimulates the penis of an adult man with visible "
                "hand-to-penis contact while they have a steady pose."
            ),
        )
        self.assertEqual(corrector.explicit_adult_grammar_issues(translated), [])

    def test_explicit_adult_pipeline_repairs_actor_to_recipient_variants(self):
        cases = {
            "She blowjobs a man.": (
                "She performs oral stimulation on the penis of an adult man "
                "with visible mouth-to-penis contact."
            ),
            "She gives a blowjob to an adult man.": (
                "She performs oral stimulation on the penis of an adult man "
                "with visible mouth-to-penis contact."
            ),
            "An adult woman blowjob's an adult man.": (
                "An adult woman performs oral stimulation on the penis of an adult man "
                "with visible mouth-to-penis contact."
            ),
            "He gives a handjob to an adult man.": (
                "He manually stimulates the penis of an adult man with visible "
                "hand-to-penis contact."
            ),
        }

        for source, expected in cases.items():
            with self.subTest(source=source):
                translated = corrector.translate_explicit_adult_language(source)
                self.assertEqual(translated, expected)
                self.assertEqual(
                    corrector.explicit_adult_grammar_issues(translated),
                    [],
                )
                self.assertEqual(
                    corrector.translate_explicit_adult_language(translated),
                    translated,
                )

    def test_explicit_adult_pipeline_keeps_possessive_quality_on_the_action(self):
        cases = {
            "The blowjob's rhythm is steady.": (
                "the rhythm of the oral stimulation of the penis is steady."
            ),
            "Her handjob's pace is slow.": (
                "the pace of her manual stimulation of the penis is slow."
            ),
            "The fellatio's framing is clear.": (
                "the framing of the oral stimulation of the penis is clear."
            ),
            "The masturbation's intensity rises.": (
                "the intensity of the self-stimulation of the genitals rises."
            ),
        }

        for source, expected in cases.items():
            with self.subTest(source=source):
                translated = corrector.translate_explicit_adult_language(source)
                self.assertEqual(translated.casefold(), expected.casefold())
                self.assertNotRegex(
                    translated,
                    r"(?i)\b(?:penis|vulva|anus|genitals)['’]s\b",
                )
                self.assertEqual(
                    corrector.explicit_adult_grammar_issues(translated),
                    [],
                )
                self.assertEqual(
                    corrector.translate_explicit_adult_language(translated),
                    translated,
                )

    def test_explicit_adult_malformed_grammar_is_a_hard_contract(self):
        issues = corrector.final_compliance_issues(
            "An adult woman she oral stimulation of the penis's a man.",
            original_prompt="An adult woman performs oral sex on an adult man.",
            explicit_nsfw=True,
            output_length="Concise",
        )

        grammar_issue = next(
            issue
            for issue in issues
            if issue.startswith("Explicit adult grammar contract")
        )
        self.assertTrue(corrector.is_hard_compliance_issue(grammar_issue))

    def test_explicit_adult_language_preserves_quoted_rendered_text(self):
        prompt = 'A mature adult woman beside a sign reading "MILF pussy hammer".'

        translated = corrector.translate_explicit_adult_language(prompt)

        self.assertIn('"MILF pussy hammer"', translated)

    def test_explicit_adult_language_translates_multiword_idioms_first(self):
        cases = (
            (
                "She is getting railed in doggy style.",
                (
                    "being penetrated with forceful repeated thrusts",
                    "a rear-entry penetrative sex position",
                ),
            ),
            (
                "She is going down on him while he is jerking himself off.",
                (
                    "performing oral genital stimulation on him",
                    "performing manual self-stimulation of the genitals",
                ),
            ),
            (
                "She is riding his cock in reverse cowgirl.",
                (
                    "straddling his penis with rhythmic penetrative motion",
                    "facing away from the partner",
                ),
            ),
            (
                "He cums on her face after a money shot.",
                (
                    "ejaculates semen onto her face",
                    "visible ejaculation as the central focal action",
                ),
            ),
            (
                "A horny adult threesome includes double penetration and a creampie.",
                (
                    "sexual scene among three sexually aroused adults",
                    "simultaneous vaginal and anal penetration",
                    "internal ejaculation with semen visible at the body opening",
                ),
            ),
            (
                "A dom/sub scene shows her clit, his balls, her booty, and his taint.",
                (
                    "dominant/submissive adult power-exchange",
                    "her clitoris",
                    "his testicles",
                    "her buttocks",
                    "his perineum",
                ),
            ),
            (
                "A quickie includes going raw, facesitting, a footjob, and squirting.",
                (
                    "brief sexual encounter",
                    "visible penetration without a condom",
                    "oral genital stimulation",
                    "penis stimulated with the feet",
                    "visible fluid release at peak sexual response",
                ),
            ),
        )

        for source, expected_phrases in cases:
            with self.subTest(source=source):
                translated = corrector.translate_explicit_adult_language(source)
                for phrase in expected_phrases:
                    self.assertIn(phrase.casefold(), translated.casefold())
                self.assertEqual(
                    corrector.explicit_adult_language_terms(translated),
                    [],
                )
                self.assertEqual(
                    corrector.translate_explicit_adult_language(translated),
                    translated,
                )

    def test_explicit_adult_language_catalog_covers_every_supported_category(self):
        categories = (
            corrector.EXPLICIT_ADULT_PHRASE_TRANSLATIONS,
            corrector.EXPLICIT_ADULT_STANDARD_ACT_TRANSLATIONS,
            corrector.EXPLICIT_ADULT_ANATOMY_AND_FLUID_TRANSLATIONS,
            corrector.EXPLICIT_ADULT_POSITION_TRANSLATIONS,
            corrector.EXPLICIT_ADULT_GROUP_AND_RELATIONSHIP_TRANSLATIONS,
            corrector.EXPLICIT_ADULT_BDSM_AND_FETISH_TRANSLATIONS,
            corrector.EXPLICIT_ADULT_PORN_AND_CAMERA_TRANSLATIONS,
        )

        self.assertGreaterEqual(len(corrector.EXPLICIT_ADULT_LANGUAGE_TRANSLATIONS), 230)
        for rules in categories:
            self.assertTrue(rules)
            for pattern, replacement, label in rules:
                re.compile(pattern, flags=re.IGNORECASE)
                self.assertTrue(replacement.strip())
                self.assertTrue(label.strip())
                self.assertEqual(
                    corrector.translate_explicit_adult_language(replacement),
                    replacement,
                )
                self.assertEqual(
                    corrector.explicit_adult_language_terms(replacement),
                    [],
                )

    def test_translatable_adult_labels_match_all_semantic_control_validators(self):
        checked = 0
        for _pattern, _replacement, label in (
            corrector.EXPLICIT_ADULT_LANGUAGE_TRANSLATIONS
        ):
            canonical = corrector.translate_explicit_adult_language(label)
            if canonical == label:
                continue
            checked += 1
            with self.subTest(label=label):
                self.assertEqual(
                    corrector.missing_required_concepts(canonical, label),
                    [],
                )
                if "," not in label:
                    self.assertEqual(
                        corrector.missing_weighted_terms(
                            canonical,
                            f"{label}:1.7",
                        ),
                        [],
                    )
                self.assertIsNone(corrector.focus_issue(canonical, label))
                self.assertEqual(
                    corrector.single_image_story_element_issues(canonical, label),
                    [],
                )
                self.assertEqual(
                    corrector.intent_lock_issues(
                        label,
                        canonical,
                        goal_headline=label,
                    ),
                    [],
                )
        self.assertGreaterEqual(checked, 200)

    def test_canonical_adult_actions_match_scene_and_identity_contracts(self):
        for source in ("vaginal sex", "anal sex", "female ejaculation"):
            canonical = corrector.translate_explicit_adult_language(source)
            with self.subTest(source=source):
                self.assertEqual(
                    corrector.nsfw_scene_contract_issues(canonical, source),
                    [],
                )
                self.assertEqual(
                    corrector.gender_identity_contract_issues(canonical, source),
                    [],
                )

    def test_canonical_semantics_cover_instruction_panels_and_fidelity_ranking(self):
        source = (
            "Panel 1: a baddie with main character energy. "
            "Panel 2: the baddie shows drip and rizz."
        )
        canonical = corrector.canonical_validation_text(source)

        self.assertEqual(
            corrector.panel_description_issues(canonical, source),
            [],
        )
        self.assertEqual(
            corrector.missing_explicit_instructions(
                canonical,
                "",
                "Keep the main character energy and drip",
            ),
            [],
        )
        self.assertEqual(
            corrector.prompt_fidelity_penalty(source, canonical),
            0,
        )

    def test_canonical_semantics_cover_count_spatial_exclusion_and_invent_seed(self):
        count_source = "Exactly one baddie stands beneath the light."
        count_final = corrector.canonical_validation_text(count_source)
        spatial_source = "A baddie on the left, a red car on the right."
        spatial_final = corrector.canonical_validation_text(spatial_source)
        excluded_source = "No baddie. A product photograph."
        excluded_final = (
            "A confident glamorous stylish person stands in the product photograph."
        )

        self.assertEqual(
            corrector.count_contract_issues(count_final, count_source),
            [],
        )
        self.assertEqual(
            corrector.spatial_contract_issues(spatial_final, spatial_source),
            [],
        )
        self.assertIn(
            "Excluded content appears positively: stylish person",
            corrector.exclusion_contract_issues(
                excluded_final,
                excluded_source,
            ),
        )
        self.assertEqual(
            corrector.invent_field_issues(
                "single",
                "focus",
                "confident glamorous stylish person",
                seed_value="baddie",
            ),
            [],
        )

    def test_weighted_active_adult_concept_rejects_passive_prop_only(self):
        self.assertIn(
            "dildo fucking (strong visual priority, 1.7)",
            corrector.missing_weighted_terms(
                "A boxed dildo sits unopened on a product shelf.",
                "dildo fucking:1.7",
            ),
        )

    def test_explicit_adult_standard_language_is_made_visually_concrete(self):
        cases = (
            (
                "A woman masturbates and then has an orgasm.",
                (
                    "woman performs self-stimulation of her own genitals",
                    "shows a visible peak sexual response",
                    "muscle tension, altered breathing, and facial reaction",
                ),
            ),
            (
                "Mutual masturbation leads to simultaneous orgasm after foreplay and oral sex.",
                (
                    "each performing visible self-stimulation of their own genitals",
                    "visible orgasm reactions",
                    "pre-intercourse intimate touching",
                    "mouth-to-genital contact",
                ),
            ),
            (
                "Fellatio, cunnilingus, anal sex, vaginal intercourse, tribadism, "
                "pegging, and urethral sounding.",
                (
                    "oral stimulation of the penis",
                    "oral stimulation of the vulva and clitoris",
                    "visible penetration at the anus",
                    "visible penetration at the vaginal opening",
                    "vulva-to-vulva rubbing",
                    "strap-on anal penetration",
                    "sounding rod into the urethral opening",
                ),
            ),
            (
                "Her coochie and clit are visible with his schlong, nuts, sack, "
                "her pussy juice, and jizz.",
                (
                    "Her vulva and clitoris",
                    "his penis, testicles, and scrotum",
                    "visible vaginal lubrication",
                    "semen",
                ),
            ),
            (
                "Mating press, prone bone, lotus sex position, spooning sex, "
                "and wheelbarrow position.",
                (
                    "knees pressed toward the chest",
                    "receiving adult lying face-down",
                    "face-to-face seated straddling",
                    "side-lying rear-entry penetration",
                    "supported on hands while hips are held",
                ),
            ),
            (
                "An MFM scene includes swingers, a hotwife, cuckold, lesbian sex, "
                "and a one-night stand.",
                (
                    "two adult men and one adult woman",
                    "exchanging sexual partners",
                    "married adult woman having consensual sex",
                    "partner watching or reacting",
                    "sexual contact between adult women",
                    "single casual sexual encounter",
                ),
            ),
            (
                "BDSM with shibari, impact play, spanking, wax play, pet play, and CNC.",
                (
                    "consensual adult restraint",
                    "decorative rope restraint",
                    "visible implement contact",
                    "open-hand impact against the buttocks",
                    "warm candle wax visibly dripping",
                    "adult human role-playing a pet",
                    "pre-consented adult force-roleplay",
                ),
            ),
            (
                "POV sex, amateur porn, hentai, glory hole, spit roast, "
                "double anal, and a cumshot.",
                (
                    "first-person participant camera view",
                    "candid homemade explicit-adult recording aesthetic",
                    "explicit adult anime-style illustration",
                    "penis extending through a small wall opening",
                    "simultaneously receiving oral and rear penetration",
                    "simultaneous anal penetration",
                    "visible semen release captured as the focal action",
                ),
            ),
        )

        for source, expected_phrases in cases:
            with self.subTest(source=source):
                translated = corrector.translate_explicit_adult_language(source)
                for phrase in expected_phrases:
                    self.assertIn(phrase.casefold(), translated.casefold())
                self.assertEqual(
                    corrector.explicit_adult_language_terms(translated),
                    [],
                )
                self.assertEqual(
                    corrector.translate_explicit_adult_language(translated),
                    translated,
                )

    def test_standard_language_translation_preserves_extracted_act_families(self):
        source = (
            "Two adult partners engage in mutual masturbation and oral sex "
            "before making love."
        )
        translated = corrector.translate_explicit_adult_language(source)

        self.assertEqual(
            corrector.nsfw_scene_contract_issues(translated, source),
            [],
        )
        self.assertEqual(corrector.explicit_adult_language_terms(translated), [])

    def test_explicit_adult_language_keeps_nonsexual_came_and_hammering(self):
        prompt = (
            "An adult woman came into the workshop while hammering a nail beside "
            "an unopened boxed dildo. The missionary visits a rail yard carrying "
            "five pounds of flour while a cowgirl rides her horse, turns on a light, "
            "and leads an ass."
        )

        translated = corrector.translate_explicit_adult_language(prompt)

        self.assertIn("came into the workshop", translated)
        self.assertIn("hammering a nail", translated)
        self.assertIn("The missionary visits a rail yard", translated)
        self.assertIn("a cowgirl rides her horse", translated)
        self.assertIn("turns on a light", translated)
        self.assertIn("leads an ass", translated)

    def test_normalize_final_prompt_text_polishes_request_phrasing(self):
        prompt = 'please generate an image of a knight with some stuff, "make me a sign"'

        normalized = corrector.normalize_final_prompt_text(prompt)

        self.assertIn("a knight with clear supporting visual details", normalized)
        self.assertIn('"make me a sign"', normalized)
        self.assertNotIn("please", normalized.lower())

    def test_explicit_direct_phrasing_is_allowed_for_altered_models(self):
        issues = corrector.final_compliance_issues(
            "Generate an image of a knight in polished steel armor standing in a torchlit stone hall.",
            output_length="Concise",
            altered_text_encoder=True,
        )

        self.assertFalse(any("Weak or non-visual phrasing" in issue for issue in issues))

    def test_final_compliance_issues_reports_weak_phrasing(self):
        issues = corrector.final_compliance_issues(
            "please make me an image of a knight with some things",
            output_length="Concise",
        )

        self.assertTrue(any("Weak or non-visual phrasing" in issue for issue in issues))

    def test_final_compliance_issues_reports_unresolved_vagueness(self):
        issues = corrector.final_compliance_issues(
            "A cool aesthetic scene with something interesting.",
            output_length="Concise",
        )

        self.assertTrue(any("Vague prompt request unresolved" in issue for issue in issues))

    def test_collect_vague_prompt_research_uses_visual_queries(self):
        queries = []

        def fake_collect(query, max_results=3, timeout=10.0, search_engine="Auto (all engines)"):
            queries.append(query)
            return f"Research for {query}"

        with patch("krea_prompt_corrector.collect_concept_research", fake_collect):
            context = corrector.collect_vague_prompt_research("cool aesthetic scene")

        self.assertIn("Vague prompt clarification research targets:", context)
        self.assertTrue(any("clarify ambiguous image prompt meaning" in query for query in queries))
        self.assertTrue(any("subject setting lighting composition" in query for query in queries))
        self.assertTrue(any("visual reference concrete depiction cool" in query for query in queries))
        self.assertTrue(any("visual reference concrete depiction aesthetic" in query for query in queries))

    def test_vague_prompt_research_queries_use_concrete_anchors(self):
        queries = corrector.vague_prompt_research_queries(
            "cool aesthetic scene with a samurai in neon rain"
        )

        self.assertTrue(any("samurai" in query and "neon" in query for query in queries))
        self.assertTrue(any("clarify ambiguous image prompt meaning" in query for query in queries))

    def test_vague_prompt_needs_clarification_research_detects_too_vague(self):
        self.assertTrue(corrector.vague_prompt_needs_clarification_research("cool aesthetic scene"))
        self.assertFalse(
            corrector.vague_prompt_needs_clarification_research(
                "A knight holding a torch in a stone castle hallway, low angle, warm firelight."
            )
        )

    def test_action_pose_terms_extracts_action_phrases(self):
        terms = corrector.action_pose_terms(
            "a skateboarder jumping over a rail while holding a camera"
        )

        joined = " ".join(terms)
        self.assertIn("skateboarder jumping over", joined)
        self.assertIn("holding camera", joined)

    def test_collect_action_pose_research_uses_body_mechanics_queries(self):
        queries = []

        def fake_collect(query, max_results=3, timeout=10.0, search_engine="Auto (all engines)"):
            queries.append((query, max_results, timeout, search_engine))
            return f"Research for {query}"

        with patch("krea_prompt_corrector.collect_concept_research", fake_collect):
            context = corrector.collect_action_pose_research(
                "a dancer kneeling and reaching toward a falling scarf",
                timeout=5,
            )

        self.assertIn("Action and pose research targets:", context)
        self.assertTrue(any("body mechanics pose contact points balance weight shift" in query for query, _, _, _ in queries))
        self.assertTrue(any("kneeling" in query for query, _, _, _ in queries))
        self.assertTrue(any("reaching" in query for query, _, _, _ in queries))
        self.assertTrue(all(timeout == 5 for _, _, timeout, _ in queries))

    def test_final_compliance_issues_reports_plausibility_risks(self):
        issues = corrector.final_compliance_issues(
            "A portrait with extra fingers, distorted face, and random objects.",
            output_length="Concise",
        )

        self.assertTrue(any("Plausibility risk" in issue for issue in issues))

    def test_plausibility_allows_intentional_surreal_scenes(self):
        issues = corrector.final_compliance_issues(
            "A surreal dreamlike figure walking while lying down inside an abstract room.",
            original_prompt="surreal dreamlike impossible pose",
            output_length="Concise",
        )

        self.assertFalse(any("Plausibility risk" in issue for issue in issues))

    def test_final_compliance_issues_reports_untranslated_slang(self):
        issues = corrector.final_compliance_issues(
            "a dope portrait with main character energy",
            output_length="Concise",
        )

        self.assertTrue(any("Untranslated slang" in issue for issue in issues))

    def test_user_message_can_include_live_research_context(self):
        message = corrector.build_user_message(
            "samurai using katana wrong",
            "Search query: katana stance\nFindings:\n1. Katana basics - grip and stance matter.",
        )

        self.assertIn("Grounded research context:", message)
        self.assertIn("katana stance", message)
        self.assertIn("Do not cite sources in the final prompt", message)

    def test_user_message_can_include_concept_integration_context(self):
        message = corrector.build_user_message(
            "a warrior in a city",
            concept_context=(
                "User-requested concepts to integrate:\n"
                "1. Concept: brutalist architecture\n"
                "Findings:\n1. Brutalism uses massive concrete forms."
            ),
        )

        self.assertIn("Concept integration context:", message)
        self.assertIn("brutalist architecture", message)
        self.assertIn("Integrate the requested concepts naturally", message)
        self.assertIn("do not dump raw keywords", message)

    def test_user_message_requires_concept_keywords(self):
        message = corrector.build_user_message(
            "a warrior in a city",
            concept_keywords=" medivial armor, brutalist architecture ",
        )

        self.assertIn("Required concept integration:", message)
        self.assertIn("medieval armor, brutalist architecture", message)
        self.assertIn("must visibly include every required concept", message)
        self.assertIn("Do not ignore a required concept", message)

    def test_user_message_can_include_result_focus(self):
        message = corrector.build_user_message(
            "a knight entering a castle",
            focus="armor accuracy and dramatic face lighting",
        )

        self.assertIn("User-requested result focus:", message)
        self.assertIn("armor accuracy and dramatic face lighting", message)
        self.assertIn("Use it as emphasis, not as a replacement", message)

    def test_user_message_can_include_story_elements(self):
        message = corrector.build_user_message(
            "a knight entering a castle",
            story_elements="the knight bursts through the gate, arrows hit the shield, villagers react",
        )

        self.assertIn("Visual storytelling elements:", message)
        self.assertIn("the knight bursts through the gate", message)
        self.assertIn("action phase", message)
        self.assertIn("cause and effect", message)
        self.assertIn("one image", message)

    def test_multi_panel_story_detection_and_panel_count(self):
        self.assertTrue(corrector.appears_multi_panel_story("three-panel comic strip"))
        self.assertTrue(corrector.appears_multi_panel_story("a story told across 4 panels"))
        self.assertTrue(
            corrector.appears_multi_panel_story("Panel 1: the gate opens. Panel 2: the knight enters.")
        )
        self.assertEqual(corrector.requested_panel_count("a triptych story"), 3)
        self.assertEqual(corrector.requested_panel_count("a story told across 4 panels"), 4)
        self.assertEqual(
            corrector.requested_panel_count("Panel 1: arrival, Panel 2: attack, Panel 3: escape"),
            3,
        )

    def test_unlabelled_lines_and_pipes_become_automatic_panels(self):
        lines = "She finds the key\nShe opens the cellar door\nThe creature emerges"
        self.assertEqual(
            corrector.implicit_panel_beats(lines),
            ["She finds the key", "She opens the cellar door", "The creature emerges"],
        )
        self.assertEqual(
            corrector.extract_panel_descriptions(lines),
            [
                (1, "She finds the key"),
                (2, "She opens the cellar door"),
                (3, "The creature emerges"),
            ],
        )
        self.assertTrue(corrector.appears_multi_panel_story("comic story", lines))
        self.assertEqual(corrector.requested_panel_count("comic story", lines), 3)
        self.assertEqual(
            corrector.extract_panel_descriptions("Finds the key | Opens the door"),
            [(1, "Finds the key"), (2, "Opens the door")],
        )
        self.assertFalse(corrector.appears_multi_panel_story("a knight", "She finds the key"))

        message = corrector.build_user_message("comic story", story_elements=lines)
        self.assertIn("Requested panel count: 3", message)
        self.assertIn("Panel 2 must preserve: She opens the cellar door", message)

    def test_user_message_preserves_multi_panel_story_structure(self):
        message = corrector.build_user_message(
            "three-panel comic strip about a red-cloaked knight",
            story_elements=(
                'panel 1 she finds a broken gate, panel 2 arrows hit her shield, '
                'panel 3 she shouts "Close the gate!"'
            ),
        )

        self.assertIn("Multi-panel storytelling request detected", message)
        self.assertIn("Requested panel count: 3", message)
        self.assertIn("Explicitly label every panel", message)
        self.assertIn("wardrobe and environment continuity", message)
        self.assertIn("correct panel and speaker", message)
        self.assertIn("Required panel-by-panel content contract", message)
        self.assertIn("Panel 2 must preserve: arrows hit her shield", message)
        self.assertIn("Each line is mandatory in its matching panel", message)
        self.assertNotIn("not a written plot summary or multi-panel sequence", message)

    def test_panel_descriptions_are_extracted_and_enforced_per_panel(self):
        story = (
            'Panel 1: the knight finds a broken gate. '
            'Panel 2: arrows strike her steel shield. '
            'Panel 3: she points to the tower and shouts "Close the gate!"'
        )
        self.assertEqual(
            corrector.extract_panel_descriptions(story),
            [
                (1, "the knight finds a broken gate"),
                (2, "arrows strike her steel shield"),
                (3, 'she points to the tower and shouts "Close the gate!"'),
            ],
        )

        wrong = (
            'A three-panel comic in a clearly divided horizontal layout, read left to right. '
            'Panel 1 shows the knight walking through a forest and saying "Close the gate!". '
            'Panel 2 shows the knight looking at a distant river. '
            'Panel 3 shows the knight resting beside a campfire.'
        )
        issues = corrector.final_compliance_issues(
            wrong,
            original_prompt="three-panel comic about a knight",
            story_elements=story,
            output_length="Expanded",
        )
        joined = " ".join(issues)
        self.assertIn("Panel 1 does not preserve enough requested beat content", joined)
        self.assertIn("Panel 2 does not preserve enough requested beat content", joined)
        self.assertIn("Panel 3 is missing its assigned quoted text", joined)

        correct = (
            'A three-panel comic in a clearly divided horizontal layout, read left to right. '
            'Panel 1 shows the knight finding a broken castle gate. '
            'Panel 2 shows arrows striking the knight\'s raised steel shield. '
            'Panel 3 shows the knight pointing to the tower and shouting "Close the gate!".'
        )
        panel_issues = corrector.multi_panel_story_issues(
            correct,
            "three-panel comic about a knight",
            story,
        )
        self.assertEqual(panel_issues, [])

    def test_failed_model_repair_gets_deterministic_three_panel_contract(self):
        story = "She finds the key\nShe opens the cellar door\nThe creature emerges"
        collapsed = "A knight confronts a creature in one cinematic cellar scene."

        repaired = corrector.enforce_multi_panel_contract(
            collapsed,
            "three-panel comic with two panels on top and one big panel at the bottom",
            story,
        )

        self.assertIn("two clearly separated panels across the top", repaired)
        self.assertIn("one large panel across the bottom", repaired)
        self.assertIn("Panel 1: She finds the key", repaired)
        self.assertIn("Panel 2: She opens the cellar door", repaired)
        self.assertIn("Panel 3: The creature emerges", repaired)
        self.assertEqual(
            corrector.multi_panel_story_issues(
                repaired,
                "three-panel comic with two panels on top and one big panel at the bottom",
                story,
            ),
            [],
        )

    def test_post_completion_enforces_panels_when_final_model_repair_still_collapses_them(self):
        responses = [
            {"choices": [{"message": {"content": "One collapsed castle scene."}}]},
            {"choices": [{"message": {"content": (
                "A red-cloaked knight crosses a storm-dark castle approach in sequential "
                "wide and medium camera views. Directional torchlight reveals wet stone, "
                "the raised steel shield, flying arrows, and the final courtyard entrance "
                "while maintaining consistent armor, weather, color, and identity."
            )}}]},
        ]

        def fake_urlopen(request, timeout):
            return FakeResponse(responses.pop(0))

        story = "Knight reaches gate\nArrows strike shield\nKnight enters courtyard"
        with patch("urllib.request.urlopen", fake_urlopen):
            result = corrector.post_chat_completion(
                base_url="http://127.0.0.1:1234/v1",
                model="small-local-model",
                prompt="three-panel comic, two panels on top and one big panel at the bottom",
                temperature=0.2,
                max_tokens=300,
                timeout=5,
                api_key="test-key",
                story_elements=story,
                audit_repair=False,
            )

        self.assertIn("Panel 1: Knight reaches gate", result)
        self.assertIn("Panel 2: Arrows strike shield", result)
        self.assertIn("Panel 3: Knight enters courtyard", result)
        self.assertFalse(
            corrector.multi_panel_story_issues(
                result,
                "three-panel comic, two panels on top and one big panel at the bottom",
                story,
            )
        )

    def test_variation_and_krea_settings_contracts_are_validated_and_enforced(self):
        variations = (
            "Variation 1: A red coupe crosses a rain-dark bridge under blue city light. "
            "Variation 2: A blue coupe waits beneath warm garage lights at dawn."
        )
        self.assertEqual(corrector.variation_issues(variations, 2), [])
        self.assertTrue(corrector.variation_issues("One red car. Another blue car.", 2))

        with_settings = corrector.enforce_krea_settings_contract(
            variations + " Krea settings: creativity=raw, intensity=99",
            include_krea_settings=True,
            creativity="high",
            intensity=25,
            complexity=-10,
            movement=70,
        )
        self.assertEqual(with_settings, variations)
        self.assertNotIn("Krea settings:", with_settings)
        self.assertEqual(
            corrector.enforce_krea_settings_contract(
                variations + " Set separately in Krea: creativity=raw, intensity=0.",
                include_krea_settings=True,
                creativity="raw",
                intensity=0,
                complexity=0,
                movement=0,
            ),
            variations,
        )
        hard, _soft = corrector.split_compliance_issues(
            corrector.krea_settings_issues(
                "A red car. creativity=raw, intensity=0",
                include_krea_settings=True,
                creativity="raw",
                intensity=0,
                complexity=0,
                movement=0,
            )
        )
        self.assertTrue(hard)
        self.assertIn(
            "creativity=high, intensity=25, complexity=-10, movement=70",
            corrector.format_krea_recommendation(
                creativity="high",
                intensity=25,
                complexity=-10,
                movement=70,
            ),
        )
        self.assertEqual(
            corrector.krea_settings_issues(
                with_settings,
                include_krea_settings=True,
                creativity="high",
                intensity=25,
                complexity=-10,
                movement=70,
            ),
            [],
        )

    def test_repair_pass_carries_all_generation_inputs_and_is_revalidated(self):
        captured_payloads = []
        responses = [
            {"choices": [{"message": {"content": "A generic creature scene."}}]},
            {"choices": [{"message": {"content": (
                'A solarpunk dragon guards a glass greenhouse beside a sign reading "OPEN". '
                "Its glowing wings dominate the foreground beneath warm sunrise lighting, "
                "while dense vines, polished brass irrigation pipes, reflective glass, and a "
                "low cinematic camera angle establish a coherent botanical environment."
            )}}]},
        ]

        def fake_urlopen(request, timeout):
            captured_payloads.append(json.loads(request.data.decode("utf-8")))
            return FakeResponse(responses.pop(0))

        with patch("urllib.request.urlopen", fake_urlopen):
            result = corrector.post_chat_completion(
                base_url="http://127.0.0.1:1234/v1",
                model="small-local-model",
                prompt='A solarpunk dragon beside a sign reading "OPEN"',
                temperature=0.2,
                max_tokens=400,
                timeout=5,
                api_key="test-key",
                mode="Cinematic",
                detail_level="Rich caption",
                output_length="Concise",
                output_min_words=35,
                output_max_words=75,
                risk_level="Strict cleanup",
                prompt_preset="Cinematic action",
                preserve_strictly=True,
                optimize_quoted_text=True,
                fix_logic=True,
                enhance_actions=True,
                develop_story=False,
                clean_constraints=True,
                altered_text_encoder=True,
                thinking_mode=True,
                concept_keywords="solarpunk",
                goal_headline="Solarpunk dragon guards greenhouse",
                focus="glowing wings",
                weighted_terms="dragon:2.0",
                model_instructions="Keep the greenhouse architecture prominent.",
                research_context="Solarpunk uses greenery and renewable technology.",
                image_context="Reference has brass pipes and reflective glass.",
                concept_context="Dragon silhouette must remain readable.",
            )

        self.assertIn("solarpunk dragon", result.lower())
        self.assertIn('"OPEN"', result)
        self.assertEqual(len(captured_payloads), 2)
        repair_system = captured_payloads[1]["messages"][0]["content"]
        repair_user = captured_payloads[1]["messages"][1]["content"]
        for expected in (
            "Shape the prompt toward this visual direction: Cinematic",
            "Preserve the user's wording and visual intent very strictly",
            "Enhance described actions",
            "Story development is disabled",
        ):
            self.assertIn(expected, repair_system)
        for expected in (
            "Mode: Cinematic",
            "Detail level: Rich caption",
            "Rewrite risk: Strict cleanup",
            "Prompt preset: Cinematic action",
            "Preserve wording strictly: True",
            "Keep the greenhouse architecture prominent",
            "Grounded research context",
            "Reference image findings",
            "Concept integration context",
        ):
            self.assertIn(expected, repair_user)

    def test_noncompliant_repairs_return_best_usable_candidate(self):
        responses = [
            {"choices": [{"message": {"content": "Too short."}}]},
            {"choices": [{"message": {"content": "Still too short."}}]},
            {"choices": [{"message": {"content": "Again too short."}}]},
            {"choices": [{"message": {"content": "Persistently too short."}}]},
            {"choices": [{"message": {"content": "Still persistently too short."}}]},
        ]

        call_count = 0

        def fake_urlopen(request, timeout):
            nonlocal call_count
            call_count += 1
            return FakeResponse(responses.pop(0))

        with patch("urllib.request.urlopen", fake_urlopen):
            result = corrector.post_chat_completion(
                base_url="http://127.0.0.1:1234/v1",
                model="small-local-model",
                prompt="a red car on a bridge",
                temperature=0.2,
                max_tokens=200,
                timeout=5,
                api_key="test-key",
                output_length="Expanded",
                output_min_words=100,
                output_max_words=120,
            )

        self.assertTrue(result)
        self.assertEqual(call_count, 3)

    def test_short_fidelity_fallback_uses_supplied_story_beats(self):
        fallback = " ".join(["source-grounded"] * 117)
        story = (
            "the adult subject increases the manual rhythm while reacting visibly; "
            "the adult subject changes position to intensify solo pleasure"
        )

        expanded = corrector.extend_short_fidelity_fallback(
            fallback,
            story,
            output_length="Expanded",
        )

        self.assertNotIn("Visible action details:", expanded)
        self.assertIn(
            "the adult subject increases the manual rhythm while reacting visibly",
            expanded,
        )
        self.assertIn(
            "the adult subject changes position to intensify solo pleasure",
            expanded,
        )
        self.assertIsNone(corrector.length_issue(expanded, "Expanded"))
        self.assertEqual(
            corrector.extend_short_fidelity_fallback(
                expanded,
                story,
                output_length="Expanded",
            ),
            expanded,
        )

    def test_4b_model_uses_compact_audit_without_extra_soft_repair(self):
        captured_payloads = []
        responses = [
            {"choices": [{"message": {"content": "A red car crosses a bridge."}}]},
            {"choices": [{"message": {"content": (
                "A solarpunk red car crosses a glass-and-brass bridge surrounded by greenery."
            )}}]},
        ]

        def fake_urlopen(request, timeout):
            captured_payloads.append(json.loads(request.data.decode("utf-8")))
            return FakeResponse(responses.pop(0))

        with patch("urllib.request.urlopen", fake_urlopen):
            result = corrector.post_chat_completion(
                base_url="http://127.0.0.1:1234/v1",
                model="qwen3-vl-4b-instruct",
                prompt="a red car on a bridge",
                temperature=0.2,
                max_tokens=200,
                timeout=5,
                api_key="test-key",
                audit_repair=True,
                concept_keywords="solarpunk",
            )

        self.assertEqual(len(captured_payloads), 2)
        self.assertIn("solarpunk", result.lower())
        self.assertIn(
            "prompt compliance auditor",
            captured_payloads[1]["messages"][0]["content"],
        )
        self.assertLess(len(captured_payloads[0]["messages"][0]["content"].split()), 220)

    def test_pose_contract_requires_viewpoint_and_action_critical_limb_chain(self):
        source = "A woman in side view reaches with her anatomical right hand for a railing."
        weak = corrector.pose_contract_issues("A woman reaches for a railing.", source)
        strong = corrector.pose_contract_issues(
            "Side view of a woman reaching with her anatomical right hand, her right shoulder, elbow, wrist, and palm aligned toward the railing contact point.",
            source,
        )

        self.assertTrue(any("limb chain" in issue for issue in weak))
        self.assertTrue(any("viewpoint" in issue for issue in weak))
        self.assertEqual(strong, [])

    def test_comic_contract_preserves_geometry_continuity_and_avoids_panel_duplication(self):
        source = (
            "A 2-panel comic story page. Page layout: Vertical strip. "
            "Reading order: Top to bottom. Aspect ratio: 4:5 portrait. "
            "Shared continuity anchors: same red coat and brass key."
        )
        beats = "Panel 1: a courier finds the key\nPanel 2: the same courier opens the gate"
        repaired = corrector.enforce_multi_panel_contract("A collapsed scene.", source, beats)

        self.assertIn("Vertical strip", repaired)
        self.assertIn("Top to bottom", repaired)
        self.assertRegex(repaired, r"4:\s*5 portrait")
        self.assertIn("same red coat and brass key", repaired)
        self.assertNotIn("Shared visual direction for every panel: A collapsed scene", repaired)
        self.assertEqual(corrector.comic_metadata_issues(repaired, source), [])

    def test_comic_continuity_does_not_absorb_following_concepts_or_style(self):
        source = (
            "A 2-panel comic story page. "
            "Shared continuity anchors: same red coat and brass key. "
            "Required concepts to integrate across the comic page: dieselpunk, chrome. "
            "Mandatory shared comic style direction: inked graphic novel."
        )
        final_prompt = (
            "A clearly divided 2-panel comic page with visible gutters. "
            "The courier wears the same red coat and carries the same brass key in both panels."
        )

        self.assertEqual(
            corrector._comic_metadata_value(source, "Shared continuity anchors"),
            "same red coat and brass key",
        )
        self.assertEqual(
            corrector._comic_metadata_value(source, "Shared visual direction"),
            "inked graphic novel",
        )
        self.assertEqual(corrector.comic_metadata_issues(final_prompt, source), [])

    def test_long_comic_validates_late_shared_continuity_section(self):
        continuity = (
            "Character identity: Liora has glowing skin and liquid-mercury nails. "
            "Wardrobe and props: a woven-light gown and crescent vessel held at her chest. "
            "Environment: suspended golden ash in a wall-less cathedral. "
            "Palette: cold violet and turquoise glow against warm gold haze. "
            "Screen direction: the camera orbits without moving forward or backward."
        )
        source = (
            "A 2-panel comic story page. Page layout: horizontal strip. "
            "Reading order: left to right. Aspect ratio: 4:5 portrait. "
            f"Shared continuity anchors: {continuity}. "
            "Mandatory shared comic style direction: luminous graphic novel."
        )
        long_panels = " ".join(
            f"Panel {number}: " + "detailed visual action " * 100
            for number in (1, 2)
        )
        final_prompt = (
            "A clearly divided two-panel horizontal strip with visible gutters, "
            "reading order left to right, aspect ratio 4:5 portrait. "
            f"{long_panels} Shared continuity anchors: {continuity}. "
            "Shared visual direction: luminous graphic novel."
        )

        self.assertEqual(corrector.comic_metadata_issues(final_prompt, source), [])

    def test_fidelity_ranking_does_not_penalize_useful_added_visual_detail(self):
        source = "A knight holds a shield."
        expanded = "A knight holds a shield with a firm hand grip and visible forearm alignment."

        self.assertEqual(
            corrector.prompt_fidelity_penalty(source, expanded),
            corrector.prompt_fidelity_penalty(source, source),
        )

    def test_multiple_variations_keep_exact_labels_and_settings(self):
        response = {
            "choices": [{"message": {"content": (
                "Variation 1: A red coupe crosses a rain-dark bridge under blue city light. "
                "Variation 2: A blue coupe waits beneath warm garage lights at dawn."
            )}}]
        }
        with patch("urllib.request.urlopen", return_value=FakeResponse(response)):
            result = corrector.post_chat_completion(
                base_url="http://127.0.0.1:1234/v1",
                model="small-local-model",
                prompt="a car",
                temperature=0.2,
                max_tokens=300,
                timeout=5,
                api_key="test-key",
                output_length="Concise",
                output_min_words=5,
                output_max_words=30,
                variation_count=2,
                include_krea_settings=True,
                creativity="high",
                intensity=25,
                complexity=-10,
                movement=70,
            )

        self.assertIn("Variation 1:", result)
        self.assertIn("Variation 2:", result)
        self.assertNotIn("Krea settings:", result)

    def test_story_development_can_invent_and_extend_without_changing_core_intent(self):
        system = corrector.build_system_prompt(
            risk_level="Creative enhancement",
            develop_story=True,
        )
        message = corrector.build_user_message(
            "a knight discovers an abandoned gate",
            story_elements="the knight notices fresh footprints",
            develop_story=True,
        )

        self.assertIn("Story development is enabled", system)
        self.assertIn("establish the situation", system)
        self.assertIn("escalation or obstacle", system)
        self.assertIn("For a fixed panel count", system)
        self.assertIn("do not introduce unrelated main characters", system)
        self.assertIn("Story invention and extension", message)
        self.assertIn("motivation, escalation, reaction, transition, consequence, or payoff", message)
        self.assertIn("Do not force a story onto a static portrait", message)

        disabled_system = corrector.build_system_prompt(develop_story=False)
        disabled_message = corrector.build_user_message(
            "a knight at a gate",
            develop_story=False,
        )
        self.assertIn("Story development is disabled", disabled_system)
        self.assertIn("Story invention and extension is disabled", disabled_message)

    def test_maximum_development_is_a_substantial_visual_and_story_contract(self):
        full = corrector.build_system_prompt(
            detail_level="Rich caption",
            output_length="Expanded",
            risk_level="Creative enhancement",
            develop_story=True,
            artistic_detail_freedom=True,
        )
        compact = corrector.build_small_model_system_prompt(
            generator_target="Krea 2",
            content_format="Single Image",
            output_length="Expanded",
            output_min_words=None,
            output_max_words=None,
            risk_level="Creative enhancement",
            prompt_preset="Cinematic keyframe",
            variation_count=1,
            enhance_actions=True,
            develop_story=True,
            detail_level="Rich caption",
            artistic_detail_freedom=True,
        )
        compact_user = corrector.build_small_model_user_message(
            "A knight discovers an abandoned gate.",
            generator_target="Krea 2",
            content_format="Single Image",
            story_elements="Fresh footprints lead through the opening.",
            output_length="Expanded",
            risk_level="Creative enhancement",
            develop_story=True,
            artistic_detail_freedom=True,
        )

        for prompt in (full, compact):
            self.assertIn("Substantial expansion", prompt)
            self.assertIn("cause-effect", prompt)
            self.assertIn("environmental response", prompt)
            self.assertIn("Rephrasing", prompt)
        self.assertIn("maximum depth", full)
        self.assertIn("maximum depth", compact)
        self.assertIn("Maximum development contract", compact_user)
        self.assertIn("situation, motivation, pressure or change, reaction", compact_user)

    def test_single_image_story_elements_are_a_hard_visible_contract(self):
        missing = corrector.final_compliance_issues(
            (
                "A knight stands at a weathered gate under pale dawn light, framed from a "
                "low three-quarter angle with damp stone and distant hills."
            ),
            original_prompt="A knight stands at a weathered gate.",
            story_elements="unconditional acceptance with open body language",
            content_format="Single Image",
        )
        preserved = corrector.final_compliance_issues(
            (
                "A knight stands at a weathered gate with unconditional acceptance visible "
                "in open body language, relaxed shoulders, welcoming palms, and a steady gaze."
            ),
            original_prompt="A knight stands at a weathered gate.",
            story_elements="unconditional acceptance with open body language",
            content_format="Single Image",
        )

        self.assertTrue(any(issue.startswith("Story element contract") for issue in missing))
        self.assertFalse(any(issue.startswith("Story element contract") for issue in preserved))
        hard, _soft = corrector.split_compliance_issues(missing)
        self.assertTrue(any(issue.startswith("Story element contract") for issue in hard))

    def test_long_fidelity_fallback_still_restores_required_story_direction(self):
        source = " ".join(
            [
                "A weathered knight waits beside an abandoned gate at pale dawn.",
                "Rain darkens the carved stone and gathers along the iron threshold.",
                "A low camera keeps the silent archway behind her guarded stance.",
            ]
            * 6
        )
        story = "unconditional acceptance with open body language"
        self.assertGreaterEqual(
            corrector.word_count(source),
            corrector.OUTPUT_WORD_RANGES["Expanded"][0],
        )

        restored = corrector.extend_short_fidelity_fallback(
            source,
            story,
            output_length="Expanded",
        )

        self.assertIn(story, restored)
        self.assertFalse(
            corrector.single_image_story_element_issues(restored, story)
        )

    def test_maximum_expansion_rejects_padding_and_accepts_authored_development(self):
        source = "A knight discovers an abandoned gate."
        story = "Fresh footprints lead through the opening."
        padded = " ".join([source, story] * 10)
        developed = (
            "A rain-darkened knight discovers the abandoned gate at first light, stopping as "
            "fresh footprints cross the flooded threshold and lead into the silent courtyard. "
            "Her guarded stance softens into wary resolve, one gauntlet lifting a broken ivy "
            "strand while the other steadies a chipped lantern. Wind pushes loose ash outward "
            "from the opening, revealing that something inside has disturbed the long-sealed "
            "hall. A low over-the-shoulder composition makes the boot prints the visual path "
            "from foreground mud to the deep archway. Cold blue dawn fills the ruined masonry, "
            "while the lantern lays warm copper reflections across wet armor and carved stone. "
            "A snapped warning cord, trembling weeds, and fresh water rings around the newest "
            "print imply recent passage and turn her discovery into the instant before pursuit."
        )

        self.assertTrue(
            corrector.creative_development_issues(
                padded,
                source,
                story,
                output_length="Expanded",
                risk_level="Creative enhancement",
                develop_story=True,
            )
        )
        self.assertEqual(
            corrector.creative_development_issues(
                developed,
                source,
                story,
                output_length="Expanded",
                risk_level="Creative enhancement",
                develop_story=True,
            ),
            [],
        )

    def test_multi_panel_compliance_requires_ordered_distinct_panels(self):
        original = "three-panel comic strip about a knight reaching a castle"
        bad_issues = corrector.final_compliance_issues(
            "A knight reaches a castle gate in one dramatic scene.",
            original_prompt=original,
            output_length="Expanded",
        )
        self.assertTrue(any("Multi-panel story structure" in issue for issue in bad_issues))

        good_prompt = (
            "A three-panel comic strip in a clearly divided horizontal layout, read left to right. "
            "Panel 1 shows the red-cloaked knight approaching the broken castle gate in a wide shot. "
            "Panel 2 shows the same red-cloaked knight raising her steel shield as arrows strike it. "
            "Panel 3 shows the same knight inside the courtyard, pointing back toward the gate and shouting."
        )
        good_issues = corrector.final_compliance_issues(
            good_prompt,
            original_prompt=original,
            output_length="Expanded",
        )
        self.assertFalse(any("Multi-panel story structure" in issue for issue in good_issues))

    def test_user_message_can_include_goal_headline(self):
        message = corrector.build_user_message(
            "a knight entering a castle",
            goal_headline="A wounded knight reaches the last safe gate",
        )

        self.assertIn("Prompt goal headline:", message)
        self.assertIn("A wounded knight reaches the last safe gate", message)
        self.assertIn("north-star for relevance", message)

    def test_system_prompt_contains_multi_person_role_binding_rules(self):
        prompt = corrector.build_system_prompt()

        self.assertIn("For individually tracked people", prompt)
        self.assertIn("bind every person explicitly", prompt)
        self.assertIn("short stable identity-or-role plus position label", prompt)
        self.assertIn("Repeat that label only when needed", prompt)
        self.assertIn("Use natural pronouns when the referent is unambiguous", prompt)
        self.assertIn("female, male, and nonbinary identities", prompt)
        self.assertIn("acting only collectively may keep one collective label", prompt)

    def test_krea_system_prompts_require_visual_thesis_order_without_ui_labels(self):
        full = corrector.build_system_prompt(generator_target="Krea 2")
        compact = corrector.build_small_model_system_prompt(
            generator_target="Krea 2",
            content_format="Single Image",
            output_length="Balanced",
            output_min_words=None,
            output_max_words=None,
            risk_level="Balanced improvement",
            prompt_preset="Auto",
            variation_count=1,
            enhance_actions=False,
            develop_story=False,
        )

        for prompt in (full, compact):
            self.assertIn("compact visual thesis", prompt)
            self.assertIn("core action or state", prompt)
            self.assertIn('"Camera framing and viewpoint:"', prompt)
            self.assertIn('"Visual direction:"', prompt)
        self.assertIn("Never emit workflow labels", full)
        self.assertIn("Never emit control labels", compact)
        self.assertIn(
            "defining medium or composition-critical shot when supplied",
            full,
        )

    def test_compact_system_prompt_receives_selected_visual_mode(self):
        compact = corrector.build_small_model_system_prompt(
            generator_target="Krea 2",
            content_format="Single Image",
            output_length="Balanced",
            output_min_words=None,
            output_max_words=None,
            risk_level="Balanced improvement",
            prompt_preset="Auto",
            variation_count=1,
            enhance_actions=False,
            develop_story=False,
            mode="Watercolor",
        )

        self.assertIn("Required visual mode: Watercolor.", compact)

    def test_krea_workflow_labels_are_naturalized_outside_quoted_text(self):
        prompt = (
            "A courier reaches the gate. Camera framing and viewpoint: low-angle "
            "wide shot. Visual direction: magical realism. Visible action details: "
            'rain strikes her coat. A sign reads "Visual direction: KEEP ME".'
        )

        cleaned = corrector.naturalize_krea_workflow_labels(prompt)

        self.assertNotIn("Camera framing and viewpoint:", corrector.unquoted_text(cleaned))
        self.assertNotIn("Visible action details:", corrector.unquoted_text(cleaned))
        self.assertIn("low-angle wide shot", cleaned)
        self.assertIn("magical realism", cleaned)
        self.assertIn('"Visual direction: KEEP ME"', cleaned)

    def test_user_message_can_include_non_visual_model_instructions(self):
        message = corrector.build_user_message(
            "a knight entering a castle",
            model_instructions="Prefer a grounded historical rewrite and avoid comedy.",
        )

        self.assertIn("Model instructions from user:", message)
        self.assertIn("Prefer a grounded historical rewrite", message)
        self.assertIn("Do not copy these instruction sentences into the final prompt", message)

    def test_parse_concepts_splits_deduplicates_and_limits(self):
        concepts = corrector.parse_concepts(
            " samurai armor, brutalist architecture, samurai armor, neon rain, x, y ",
            max_concepts=3,
        )

        self.assertEqual(
            concepts,
            ["samurai armor", "brutalist architecture", "neon rain"],
        )

    def test_parse_concepts_normalizes_common_typos(self):
        self.assertEqual(corrector.parse_concepts("medivial castle"), ["medieval castle"])

    def test_common_spelling_checks_cover_prompt_terms_but_preserve_quoted_text(self):
        self.assertEqual(
            corrector.common_spelling_issues(
                'a detialed lightbulb with ligthing, sign reads "DETIALED"'
            ),
            ["detialed -> detailed", "ligthing -> lighting"],
        )
        issues = corrector.final_compliance_issues(
            "A detialed lamp lights a quiet room with soft shadows and warm reflections.",
            output_length="Concise",
        )
        self.assertTrue(any("Possible spelling errors" in issue for issue in issues))
        self.assertIn("medieval", corrector.concept_search_query("medivial knight"))

    def test_parse_weighted_terms_reads_priority_values(self):
        terms = corrector.parse_weighted_terms(
            "face:1.6, red cloak=1.3, torchlight*1.15, face:1.1, background:4"
        )

        self.assertEqual(
            terms,
            [
                ("face", 1.6),
                ("red cloak", 1.3),
                ("torchlight", 1.15),
                ("background", 3.0),
            ],
        )

    def test_concept_mix_normalizes_percentages_and_fills_missing_shares(self):
        self.assertEqual(
            corrector.parse_concept_mix("watercolor:60%, cyberpunk:30, botanical"),
            [("watercolor", 60), ("cyberpunk", 30), ("botanical", 10)],
        )
        self.assertEqual(
            corrector.parse_concept_mix("Art Nouveau, brutalism"),
            [("Art Nouveau", 50), ("brutalism", 50)],
        )

    def test_concept_mix_reuses_concept_and_weight_contracts(self):
        concepts = corrector.concept_mix_to_concepts(
            "portrait photography", "watercolor:70%, cyberpunk:30%"
        )
        weighted = corrector.concept_mix_to_weighted_terms(
            "face:2.4", "watercolor:70%, cyberpunk:30%"
        )
        instruction = corrector.concept_mix_instruction(
            "watercolor:70%, cyberpunk:30%"
        )

        self.assertEqual(concepts, "portrait photography, watercolor, cyberpunk")
        self.assertEqual(
            corrector.parse_weighted_terms(weighted),
            [("face", 2.4), ("watercolor", 2.25), ("cyberpunk", 1.25)],
        )
        self.assertIn("watercolor 70%", instruction)
        self.assertIn("cyberpunk 30%", instruction)
        self.assertIn("relative creative influence", instruction)
        self.assertIn("not a literal pixel measurement", instruction)
        self.assertIn("do not create an additional person", instruction)

    def test_scoped_concept_mix_groups_remain_independent(self):
        groups = corrector.normalize_concept_mix_groups(
            [
                {
                    "name": "Subject surface",
                    "target": "Main subject",
                    "mix": "xenomorph:70%, porcelain:30%",
                },
                {
                    "name": "Room treatment",
                    "target": "Alice",
                    "mix": "greenhouse:25%, brutalism:75%",
                },
            ]
        )

        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0]["mix"], "xenomorph:70%, porcelain:30%")
        self.assertEqual(groups[1]["mix"], "greenhouse:25%, brutalism:75%")
        self.assertEqual(
            corrector.concept_mix_groups_to_concepts("", "", groups),
            "xenomorph, porcelain, greenhouse, brutalism",
        )
        instruction = corrector.concept_mix_groups_instruction("", groups)
        self.assertIn(
            'Group "Subject surface" targets Main subject: '
            "xenomorph 70%, porcelain 30%",
            instruction,
        )
        self.assertIn(
            'Group "Room treatment" targets Alice: '
            "greenhouse 25%, brutalism 75%",
            instruction,
        )
        self.assertIn("Do not average percentages between groups", instruction)

    def test_scoped_concept_mix_groups_are_bounded_and_canonical(self):
        raw = [
            {
                "name": f"  Mix   {index}  ",
                "target": "unsupported" if index == 0 else "Rendering style",
                "mix": "ink, pastel",
            }
            for index in range(8)
        ]
        raw.extend([{"name": "bad", "target": "Environment", "mix": ""}, "bad"])

        groups = corrector.normalize_concept_mix_groups(raw)

        self.assertEqual(len(groups), corrector.CONCEPT_MIX_GROUP_LIMIT)
        self.assertEqual(groups[0]["name"], "Mix 0")
        self.assertEqual(groups[0]["target"], "unsupported")
        self.assertEqual(groups[0]["mix"], "ink:50%, pastel:50%")

    def test_adjust_weighted_terms_text_changes_term_at_cursor(self):
        text, cursor = corrector.adjust_weighted_terms_text(
            "face:1.5, red cloak, torchlight:2.0",
            12,
            0.1,
        )

        self.assertEqual(text, "face:1.5, red cloak:1.1, torchlight:2.0")
        self.assertEqual(cursor, len("face:1.5, red cloak:1.1"))

    def test_adjust_weighted_terms_text_supports_five_hundredths(self):
        text, _cursor = corrector.adjust_weighted_terms_text(
            "face:1.1",
            2,
            0.05,
        )

        self.assertEqual(text, "face:1.15")

    def test_adjust_weighted_terms_text_clamps_weight_range(self):
        high, _ = corrector.adjust_weighted_terms_text("face:3.0", 2, 0.1)
        low, _ = corrector.adjust_weighted_terms_text("face:0.1", 2, -0.1)

        self.assertEqual(high, "face:3.0")
        self.assertEqual(low, "face:0.1")

    def test_adjust_named_weighted_term_adds_or_updates_term(self):
        added = corrector.adjust_named_weighted_term("face:1.2", "red cloak", 0.1)
        updated = corrector.adjust_named_weighted_term(added, "face", 0.1)

        self.assertEqual(added, "face:1.2, red cloak:1.1")
        self.assertEqual(updated, "face:1.3, red cloak:1.1")

    def test_adjust_named_weighted_term_normalizes_spacing_and_clamps(self):
        updated = corrector.adjust_named_weighted_term(
            "red cloak:1.9",
            "  red   cloak  ",
            0.3,
        )

        self.assertEqual(updated, "red cloak:2.2")

    def test_user_message_can_include_weighted_visual_emphasis(self):
        message = corrector.build_user_message(
            "a knight in a dark hall",
            weighted_terms="face:2.2, red cloak:1.6",
        )

        self.assertIn("Weighted visual emphasis:", message)
        self.assertIn("face (dominant visual priority, 2.2)", message)
        self.assertIn("red cloak (strong visual priority, 1.6)", message)
        self.assertIn("hard composition priority", message)
        self.assertIn("Terms at 2.0 or above must become dominant focal elements", message)
        self.assertIn("Do not output numeric weights", message)

    def test_strip_weighted_term_syntax_keeps_terms_without_numeric_weights(self):
        cleaned = corrector.strip_weighted_term_syntax(
            "Fish:1.4 in the foreground, RED   CLOAK: 2.0 behind it, aspect ratio 3:2.",
            "fish:1.4, red cloak:2.0",
        )

        self.assertEqual(
            cleaned,
            "Fish in the foreground, RED   CLOAK behind it, aspect ratio 3:2.",
        )

    def test_strip_weighted_term_syntax_removes_model_invented_decimal_weight(self):
        cleaned = corrector.strip_weighted_term_syntax(
            'A Car : 1 . 3 beside a Bus=1.5 and Bike*2.0, aspect ratio 3:2, '
            'clock at 10:30, sign reading "CAR:1.3".',
            "",
        )

        self.assertEqual(
            cleaned,
            'A Car beside a Bus and Bike, aspect ratio 3:2, clock at 10:30, '
            'sign reading "CAR:1.3".',
        )

    def test_strip_weighted_term_syntax_removes_priority_prose_globally(self):
        leaked = (
            'A fox (clear visual priority, 1.3) beside a lantern '
            '(strong visual priority, 1. 65), sign reading '
            '"priority (clear visual priority, 1.3)".'
        )
        cleaned = corrector.strip_weighted_term_syntax(leaked, "")

        self.assertEqual(
            cleaned,
            'A fox beside a lantern, sign reading '
            '"priority (clear visual priority, 1.3)".',
        )
        self.assertEqual(
            corrector.strip_private_prompt_guidance(leaked),
            cleaned,
        )

    def test_post_completion_removes_leaked_weight_syntax_from_final_prompt(self):
        with patch(
            "krea_prompt_corrector.chat_completion",
            return_value="A silver Fish:1.4 swimming in a clear aquarium, soft daylight.",
        ):
            result = corrector.post_chat_completion(
                base_url="http://127.0.0.1:1234/v1",
                model="test-4b",
                prompt="A silver fish swimming in a clear aquarium.",
                temperature=0.2,
                max_tokens=300,
                timeout=5,
                api_key="test",
                weighted_terms="fish:1.4",
                context_token_budget=1_000,
                final_gate_repair=False,
            )

        self.assertIn("Fish swimming", result)
        self.assertNotIn("Fish:1.4", result)

    def test_post_completion_removes_invented_weight_without_weighted_words(self):
        with patch(
            "krea_prompt_corrector.chat_completion",
            return_value="A silver Car:1.3 parked on wet asphalt under soft daylight.",
        ):
            result = corrector.post_chat_completion(
                base_url="http://127.0.0.1:1234/v1",
                model="test-4b",
                prompt="silver car",
                temperature=0.2,
                max_tokens=300,
                timeout=5,
                api_key="test",
                context_token_budget=1_000,
                final_gate_repair=False,
            )

        self.assertIn("silver Car parked", result)
        self.assertNotIn("Car:1.3", result)

    def test_post_completion_keeps_clear_him_and_his_role_references(self):
        response = (
            "The woman on the left reaches toward the man on the right, touching him "
            "while his blue coat reflects the lantern's warm light."
        )
        with patch("krea_prompt_corrector.chat_completion", return_value=response):
            result = corrector.post_chat_completion(
                base_url="http://127.0.0.1:1234/v1",
                model="test-4b",
                prompt=response,
                temperature=0.2,
                max_tokens=300,
                timeout=5,
                api_key="test",
                context_token_budget=1_000,
            )

        self.assertEqual(result, response)
        self.assertEqual(corrector.multi_person_role_issues(result), [])

    def test_final_compliance_issues_reports_missing_weighted_emphasis(self):
        issues = corrector.final_compliance_issues(
            "A knight stands in a dark hall.",
            weighted_terms="red cloak:2.4, torchlight:1.5",
            output_length="Concise",
        )

        joined = "\n".join(issues)
        self.assertIn("Missing weighted visual emphasis", joined)
        self.assertIn("red cloak", joined)
        self.assertIn("torchlight", joined)

    def test_normalize_final_prompt_text_removes_unicode_dashes(self):
        prompt = 'Final prompt: A knight \u2014 focused pose; high detail \u2013 "SALE; TODAY"'

        self.assertEqual(
            corrector.normalize_final_prompt_text(prompt),
            'A knight, focused pose, high detail, "SALE; TODAY"',
        )

    def test_normalize_final_prompt_text_removes_generation_syntax(self):
        prompt = (
            "```text\n"
            "Prompt: warrior; <lora:bad:1>; CFG: 7.5; steps: 30; "
            "(cinematic lighting:1.2); [[sharp armor]]!!!\n"
            "```"
        )

        self.assertEqual(
            corrector.normalize_final_prompt_text(prompt),
            "warrior, cinematic lighting, sharp armor!",
        )

    def test_normalize_final_prompt_text_removes_think_blocks(self):
        prompt = "<think>hidden analysis</think>\nFinal prompt: A focused knight in torchlight."

        self.assertEqual(
            corrector.normalize_final_prompt_text(prompt),
            "A focused knight in torchlight.",
        )

    def test_normalize_final_prompt_text_extracts_prompt_from_model_chatter(self):
        prompt = (
            "Sure, I fixed the prompt.\n\n"
            "Final prompt:\n"
            "A focused knight in torchlit chainmail, low angle, rain on stone walls.\n\n"
            "Notes:\n"
            "- Removed duplicate wording."
        )

        self.assertEqual(
            corrector.normalize_final_prompt_text(prompt),
            "A focused knight in torchlit chainmail, low angle, rain on stone walls.",
        )

    def test_normalize_final_prompt_text_removes_trailing_audit_sections(self):
        prompt = (
            "Repaired Krea prompt: A cinematic portrait of a lonely queen with downcast eyes, "
            "soft window light, and empty throne room space.\n"
            "Audit score: 94/100\n"
            "Breakage points:\n"
            "- None"
        )

        self.assertEqual(
            corrector.normalize_final_prompt_text(prompt),
            "A cinematic portrait of a lonely queen with downcast eyes, soft window light, and empty throne room space.",
        )

    def test_extract_prompt_from_model_text_filters_explanatory_lines(self):
        prompt = (
            "Here is the corrected version:\n"
            "A red cloak in sharp foreground focus, medieval market background, warm lantern light.\n"
            "Explanation: I made the cloak more prominent."
        )

        self.assertEqual(
            corrector.extract_prompt_from_model_text(prompt),
            "A red cloak in sharp foreground focus, medieval market background, warm lantern light.",
        )

    def test_final_compliance_issues_find_prompt_breakage(self):
        issues = corrector.final_compliance_issues(
            "A photoreal vector warrior at noon midnight, CFG: 7",
            original_prompt='a warrior holding a sign "NO ENTRY"',
            concept_keywords="medieval armor",
            focus="dramatic face lighting",
            output_length="Balanced",
        )

        joined = "\n".join(issues)
        self.assertIn("Contradictory terms", joined)
        self.assertIn("Missing required concepts: medieval armor", joined)
        self.assertIn("Missing quoted rendered text: NO ENTRY", joined)
        self.assertIn("Requested focus not represented", joined)

    def test_contradiction_check_does_not_treat_nightstand_as_nighttime(self):
        prompt = (
            "A sunny mid day comic bedroom scene with a wooden nightstand "
            "beside the bed."
        )

        self.assertEqual(corrector.contradiction_issues(prompt), [])

    def test_contradiction_check_still_finds_complete_day_and_night_terms(self):
        self.assertIn(
            "Contradictory terms: day / night",
            corrector.contradiction_issues(
                "The same scene is described as bright day and dark night."
            ),
        )
        self.assertTrue(
            corrector.contradiction_issues(
                "The lighting is both daytime sunlight and nighttime moonlight."
            )
        )

    def test_contradiction_check_preserves_intentional_source_hybrids(self):
        for prompt in (
            "A sunny rainy afternoon with a rainbow.",
            "A doorway connects the indoors and outdoors.",
            "A photoreal vector illustration.",
        ):
            with self.subTest(prompt=prompt):
                self.assertEqual(
                    corrector.contradiction_issues(prompt, prompt),
                    [],
                )

    def test_intent_lock_ignores_prepended_style_and_camera_controls(self):
        source = (
            "Magical realism. Wide establishing shot, 24mm lens. "
            "A red sports car crosses a bridge."
        )
        self.assertTrue(
            corrector.intent_lock_issues(
                source,
                "Magical realism, wide establishing shot with a 24mm lens, "
                "showing a river crossing a valley.",
            )
        )
        self.assertEqual(
            corrector.intent_lock_issues(
                source,
                "A scarlet sports automobile traverses a bridge, rendered with "
                "dreamlike realism from a broad distant viewpoint.",
            ),
            [],
        )

    def test_boundary_safe_semantic_matching_avoids_substring_false_passes(self):
        self.assertEqual(
            corrector.missing_required_concepts("A red carpet fills the hall.", "car"),
            ["car"],
        )
        self.assertEqual(
            corrector.focus_issue("A cathedral at sunset.", "cat"),
            "Requested focus not represented: cat",
        )
        self.assertIsNotNone(
            corrector.focus_issue("A red carpet fills the hall.", "red sports car")
        )

    def test_semantic_matching_accepts_faithful_concept_and_weight_paraphrases(self):
        self.assertEqual(
            corrector.missing_required_concepts(
                "A monumental raw-concrete library.",
                "brutalist architecture",
            ),
            [],
        )
        self.assertEqual(
            corrector.missing_weighted_terms(
                "A scarlet automobile crosses the bridge.",
                "red car:2.0",
            ),
            [],
        )

    def test_explicit_style_directive_accepts_visual_paraphrase(self):
        self.assertEqual(
            corrector.explicit_instruction_issues(
                "A lighthouse painted with translucent washes and visible paper grain.",
                "Please make it look like watercolor. A lighthouse on a cliff.",
            ),
            [],
        )

    def test_style_mode_is_a_hard_contract_for_distinctive_modes(self):
        issues = corrector.final_compliance_issues(
            "A hyperrealistic photograph of a lighthouse.",
            original_prompt="A lighthouse.",
            mode="Watercolor",
            output_length="Concise",
        )
        self.assertIn(
            "Selected visual mode missing or changed: Watercolor",
            issues,
        )
        hard, _soft = corrector.split_compliance_issues(issues)
        self.assertIn(
            "Selected visual mode missing or changed: Watercolor",
            hard,
        )
        self.assertEqual(
            corrector.style_mode_issues(
                "A lighthouse painted with translucent washes on visible paper grain.",
                "Watercolor",
            ),
            [],
        )

    def test_selected_style_mode_is_enforced_without_preserving_model_added_conflict(self):
        result = corrector.enforce_style_mode_contract(
            "A hyperrealistic photograph of a lighthouse in hard studio light.",
            "Watercolor",
            "A lighthouse in hard studio light.",
        )

        self.assertEqual(corrector.style_mode_issues(result, "Watercolor"), [])
        self.assertIn("Watercolor", result)
        self.assertNotIn("hyperrealistic photograph", result.lower())

    def test_selected_style_mode_preserves_user_requested_hybrid(self):
        result = corrector.enforce_style_mode_contract(
            "A photoreal vector illustration of a lighthouse.",
            "Photoreal",
            "A photoreal vector illustration of a lighthouse.",
        )

        self.assertIn("photoreal", result.lower())
        self.assertIn("vector illustration", result.lower())

    def test_all_selected_modes_and_visual_directions_are_kept_compactly(self):
        self.assertEqual(
            corrector.enforce_style_mode_contract(
                "A lighthouse in hard studio light.",
                "Cinematic",
            ),
            "Cinematic. A lighthouse in hard studio light.",
        )
        self.assertEqual(
            corrector.enforce_visual_direction_contract(
                "A lighthouse in hard studio light.",
                "Dark maritime atmosphere",
            ),
            "Dark maritime atmosphere. A lighthouse in hard studio light.",
        )
        self.assertEqual(
            corrector.enforce_visual_direction_contract(
                "A lighthouse in a dark maritime atmosphere.",
                "Dark maritime atmosphere",
            ),
            "A lighthouse in a dark maritime atmosphere.",
        )
        self.assertEqual(
            corrector.enforce_visual_direction_contract(
                "joyful and celebratory. A family raises a toast.",
                "Mood and emotional tone: joyful and celebratory.",
            ),
            "joyful and celebratory. A family raises a toast.",
        )
        self.assertEqual(
            corrector.enforce_visual_direction_contract(
                "Mood and emotional tone: joyful and celebratory. A family raises a toast.",
                "Mood and emotional tone: joyful and celebratory.",
            ),
            "joyful and celebratory. A family raises a toast.",
        )

    @patch(
        "krea_prompt_corrector.chat_completion",
        return_value="joyful and celebratory. A family raises a toast.",
    )
    def test_post_completion_does_not_repeat_labeled_visual_direction(self, _chat):
        result = corrector.post_chat_completion(
            base_url="http://127.0.0.1:1234/v1",
            model="qwen3-vl-4b-instruct",
            prompt="A family raises a toast.",
            temperature=0.2,
            max_tokens=300,
            timeout=30,
            api_key="test",
            visual_direction="Mood and emotional tone: joyful and celebratory.",
            output_length="Concise",
            audit_repair=False,
        )

        self.assertEqual(
            result,
            "joyful and celebratory. A family raises a toast.",
        )
        self.assertEqual(result.casefold().count("joyful and celebratory"), 1)
        self.assertNotIn("Mood and emotional tone:", result)

    @patch(
        "krea_prompt_corrector.chat_completion",
        return_value="A hyperrealistic photograph of a lighthouse in hard studio light.",
    )
    def test_post_completion_enforces_selected_mode_on_compact_model(self, _chat):
        result = corrector.post_chat_completion(
            base_url="http://127.0.0.1:1234/v1",
            model="qwen3-vl-4b-instruct",
            prompt="A lighthouse in hard studio light.",
            temperature=0.2,
            max_tokens=300,
            timeout=30,
            api_key="test",
            mode="Watercolor",
            output_length="Concise",
            audit_repair=False,
        )

        self.assertEqual(corrector.style_mode_issues(result, "Watercolor"), [])
        self.assertNotIn("hyperrealistic photograph", result.lower())

    def test_concept_verifier_accepts_aliases(self):
        self.assertFalse(
            corrector.missing_required_concepts(
                "A knight wearing chainmail and gauntlets.",
                "medivial armor",
            )
        )
        self.assertEqual(
            corrector.missing_required_concepts(
                "A knight in a plain cloak.",
                "medivial armor",
            ),
            ["medieval armor"],
        )

    def test_context_token_budget_normalizes_and_truncates_by_tokens(self):
        long_line = "Finding: " + "visual detail " * 80
        normalized = corrector.normalize_research_context(f"  {long_line}\n\n second   line ")
        self.assertIn("second line", normalized)
        self.assertGreater(len(normalized.splitlines()[0]), 260)

        compressed = corrector.compress_context_to_token_budget(normalized, max_tokens=40)
        self.assertLessEqual(corrector.estimate_context_tokens(compressed), 41)
        self.assertTrue(compressed.endswith("."))

    def test_context_token_budget_defaults_and_lower_limit(self):
        self.assertEqual(corrector.CONTEXT_TOKEN_MIN, 512)
        self.assertEqual(corrector.CONTEXT_TOKEN_DEFAULT, 32_000)
        sections = corrector.fit_context_sections_to_token_budget(
            "R" * 1600,
            "I" * 1600,
            "C" * 1600,
            token_budget=1,
        )
        self.assertLessEqual(
            sum(corrector.estimate_context_tokens(section) for section in sections),
            515,
        )

    def test_automatic_context_budget_reserves_most_of_loaded_window(self):
        self.assertEqual(
            corrector.automatic_context_token_budget(
                32_000,
                max_tokens=760,
                core_tokens=2_000,
            ),
            8_000,
        )
        self.assertEqual(
            corrector.automatic_context_token_budget(
                8_192,
                max_tokens=760,
                core_tokens=2_000,
            ),
            2_048,
        )
        self.assertEqual(
            corrector.automatic_context_token_budget(
                None,
                max_tokens=760,
                core_tokens=2_000,
            ),
            4_096,
        )

    def test_lm_studio_model_context_length_uses_loaded_instance_config(self):
        def fake_urlopen(request, timeout):
            self.assertEqual(
                request.full_url,
                "http://127.0.0.1:1234/api/v1/models",
            )
            return FakeResponse(
                {
                    "models": [
                        {
                            "key": "qwen3-vl-4b-instruct",
                            "loaded_instances": [
                                {
                                    "id": "qwen3-vl-4b-instruct",
                                    "config": {"context_length": 32_000},
                                }
                            ],
                        }
                    ]
                }
            )

        with patch("urllib.request.urlopen", fake_urlopen):
            context_length = corrector.lm_studio_model_context_length(
                base_url="http://127.0.0.1:1234/v1",
                model="qwen3-vl-4b-instruct",
                timeout=3,
                api_key="test-key",
            )

        self.assertEqual(context_length, 32_000)

    def test_final_repair_user_message_lists_issues_and_requirements(self):
        message = corrector.build_final_repair_user_message(
            original_prompt='a sign saying "NO ENTRY"',
            current_prompt="a sign",
            issues=["Missing quoted rendered text: NO ENTRY", "Missing required concepts: medieval armor"],
            concept_keywords="medivial armor",
            focus="readable sign text",
            output_length="Balanced",
        )

        self.assertIn("Validation issues:", message)
        self.assertIn("- Missing quoted rendered text: NO ENTRY", message)
        self.assertIn("Required concepts: medieval armor", message)
        self.assertIn('Quoted rendered text that must be preserved exactly: NO ENTRY', message)

    def test_collect_integrated_concept_research_researches_each_concept(self):
        calls = []

        def fake_collect(concept, max_results, timeout, search_engine="Auto (all engines)"):
            calls.append((concept, max_results, timeout, search_engine))
            return f"Findings for {concept}"

        with patch("krea_prompt_corrector.collect_concept_research", fake_collect):
            context = corrector.collect_integrated_concept_research(
                "samurai armor, brutalist architecture",
                max_results=2,
                timeout=4,
                search_engine="Bing",
            )

        self.assertCountEqual(
            calls,
            [
                ("samurai armor", 2, 4, "Bing"),
                ("brutalist architecture", 2, 4, "Bing"),
            ],
        )
        self.assertIn("Concept: samurai armor", context)
        self.assertIn("Findings for brutalist architecture", context)

    def test_collect_integrated_concept_research_can_skip_text_research_for_image_only(self):
        with patch("krea_prompt_corrector.collect_concept_research") as text_research:
            with patch(
                "krea_prompt_corrector.collect_reference_image_candidates",
                return_value=[{"title": "Armor image", "url": "https://example.com/armor.jpg", "summary": ""}],
            ) as image_search:
                with patch(
                    "krea_prompt_corrector.analyze_reference_images",
                    return_value="Reference image analysis for samurai armor:\nVisual findings.",
                ) as image_analysis:
                    context = corrector.collect_integrated_concept_research(
                        "samurai armor",
                        timeout=4,
                        text_research=False,
                        image_analysis=True,
                        image_source="Gelbooru",
                        image_timeout=900,
                    )

        text_research.assert_not_called()
        image_search.assert_called_once()
        self.assertEqual(image_search.call_args.kwargs["source"], "Gelbooru")
        self.assertEqual(image_analysis.call_args.kwargs["timeout"], 900)
        self.assertIn("Live text research disabled", context)
        self.assertIn("Reference image analysis", context)

    def test_concept_search_query_extracts_visual_terms(self):
        query = corrector.concept_search_query(
            "A photoreal samurai swinging a katana with wrong hand in a dojo"
        )

        self.assertIn("visual reference accurate meaning depiction", query)
        self.assertIn("samurai", query)
        self.assertIn("katana", query)
        self.assertNotIn(" the ", query)

    def test_parse_search_results_extracts_titles_snippets_and_urls(self):
        html = """
        <a class="result__a" href="/l/?uddg=https%3A%2F%2Fexample.com%2Fkatana">Katana stance</a>
        <a class="result__snippet">A katana is commonly gripped with two hands and a stable stance.</a>
        <a class="result__a" href="https://example.com/dojo">Dojo etiquette</a>
        <div class="result__snippet">Dojo scenes usually include mats and formal posture.</div>
        """
        results = corrector.parse_search_results(html, max_results=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["title"], "Katana stance")
        self.assertEqual(results[0]["url"], "https://example.com/katana")
        self.assertIn("two hands", results[0]["snippet"])

    def test_parse_bing_search_results_extracts_titles_snippets_and_urls(self):
        html = """
        <li class="b_algo">
          <h2><a href="https://example.com/castles">Medieval castles</a></h2>
          <p>Castles often include stone walls, towers, gates, and defensive details.</p>
        </li>
        <li class="b_algo">
          <h2><a href="https://example.com/armor">Medieval armor</a></h2>
          <p>Armor references include mail, plate, helmets, and leather straps.</p>
        </li>
        """
        results = corrector.parse_bing_search_results(html, max_results=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["title"], "Medieval castles")
        self.assertEqual(results[0]["url"], "https://example.com/castles")
        self.assertIn("stone walls", results[0]["snippet"])

    def test_strip_html_cleans_wikipedia_snippets(self):
        self.assertEqual(
            corrector.strip_html("A <span>katana</span> is a sword."),
            "A katana is a sword.",
        )

    def test_collect_wikipedia_research_formats_api_results(self):
        payload = {
            "query": {
                "search": [
                    {
                        "title": "Katana",
                        "snippet": "A <span class='searchmatch'>katana</span> is a Japanese sword.",
                    }
                ]
            }
        }

        with patch("urllib.request.urlopen", return_value=FakeResponse(payload)):
            context = corrector.collect_wikipedia_research("samurai katana", timeout=1)

        self.assertIn("Wikipedia query:", context)
        self.assertIn("Katana - A katana is a Japanese sword.", context)

    def test_collect_wikipedia_page_summary_formats_extract(self):
        payload = {
            "title": "Medieval architecture",
            "extract": "Medieval architecture includes religious, military, and civic buildings.",
        }

        with patch("urllib.request.urlopen", return_value=FakeResponse(payload)):
            summary = corrector.collect_wikipedia_page_summary(
                "Medieval architecture",
                timeout=1,
            )

        self.assertEqual(
            summary,
            "Medieval architecture includes religious, military, and civic buildings.",
        )

    def test_collect_wikipedia_image_candidates_uses_page_thumbnails(self):
        search_payload = {
            "query": {
                "search": [
                    {"title": "Medieval architecture", "snippet": "Buildings."}
                ]
            }
        }
        page_payload = {
            "extract": "Medieval architecture includes castles and cathedrals.",
            "thumbnail": {"source": "https://upload.wikimedia.org/thumb.jpg"},
        }

        with patch(
            "urllib.request.urlopen",
            side_effect=[FakeResponse(search_payload), FakeResponse(page_payload)],
        ):
            candidates = corrector.collect_wikipedia_image_candidates(
                "medivial",
                max_images=1,
                timeout=1,
            )

        self.assertEqual(
            candidates,
            [
                {
                    "title": "Medieval architecture",
                    "url": "https://upload.wikimedia.org/thumb.jpg",
                    "summary": "Medieval architecture includes castles and cathedrals.",
                }
            ],
        )

    def test_collect_wikimedia_commons_image_candidates_uses_imageinfo_urls(self):
        payload = {
            "query": {
                "pages": {
                    "1": {
                        "title": "File:Castle armor.jpg",
                        "imageinfo": [
                            {
                                "url": "https://upload.wikimedia.org/castle-armor.jpg",
                                "mime": "image/jpeg",
                            }
                        ],
                    }
                }
            }
        }

        with patch("urllib.request.urlopen", return_value=FakeResponse(payload)):
            candidates = corrector.collect_wikimedia_commons_image_candidates(
                "medieval castle armor",
                max_images=1,
                timeout=1,
            )

        self.assertEqual(candidates[0]["title"], "Castle armor.jpg")
        self.assertEqual(candidates[0]["url"], "https://upload.wikimedia.org/castle-armor.jpg")

    def test_extract_duckduckgo_vqd_reads_image_search_token(self):
        self.assertEqual(
            corrector.extract_duckduckgo_vqd("some script vqd='4-123456789' more"),
            "4-123456789",
        )

    def test_collect_duckduckgo_image_candidates_uses_image_search_results(self):
        search_html = "some script vqd='4-123456789' more"
        image_payload = {
            "results": [
                {
                    "title": "Medieval armor reference",
                    "image": "https://images.example.com/armor.jpg",
                    "source": "example.com",
                },
                {
                    "title": "Castle gate",
                    "image": "https://images.example.com/castle.jpg",
                    "url": "https://example.com/castle",
                },
            ]
        }
        urls = []

        def fake_urlopen(request, timeout):
            urls.append(request.full_url)
            if "duckduckgo.com/?" in request.full_url:
                return FakeBinaryResponse(search_html.encode("utf-8"), "text/html")
            return FakeResponse(image_payload)

        with patch("urllib.request.urlopen", fake_urlopen):
            candidates = corrector.collect_duckduckgo_image_candidates(
                "medivial armor",
                max_images=2,
                timeout=1,
            )

        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[0]["title"], "Medieval armor reference")
        self.assertEqual(candidates[0]["url"], "https://images.example.com/armor.jpg")
        self.assertIn("DuckDuckGo Images result", candidates[0]["summary"])
        self.assertTrue(any("duckduckgo.com/i.js" in url for url in urls))

    def test_parse_yandex_image_candidates_reads_serp_item_metadata(self):
        html = """
        <div class="serp-item" data-bem='{&quot;serp-item&quot;:{&quot;img_href&quot;:&quot;https://images.example.com/yandex-armor?id=123&quot;,&quot;snippet&quot;:&quot;Medieval armor reference&quot;,&quot;domain&quot;:&quot;example.com&quot;}}'></div>
        """

        candidates = corrector.parse_yandex_image_candidates(html, max_images=1)

        self.assertEqual(
            candidates,
            [
                {
                    "title": "Medieval armor reference",
                    "url": "https://images.example.com/yandex-armor?id=123",
                    "summary": "Yandex Images result. Source: example.com",
                }
            ],
        )

    def test_collect_yandex_image_candidates_uses_yandex_images_search(self):
        html = """
        <div class="serp-item" data-bem='{&quot;serp-item&quot;:{&quot;img_href&quot;:&quot;https://images.example.com/yandex-castle.png&quot;,&quot;snippet&quot;:&quot;Castle gate reference&quot;}}'></div>
        """
        captured = {}

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            return FakeBinaryResponse(html.encode("utf-8"), "text/html")

        with patch("urllib.request.urlopen", fake_urlopen):
            candidates = corrector.collect_yandex_image_candidates(
                "medivial castle",
                max_images=1,
                timeout=1,
            )

        self.assertIn("https://yandex.com/images/search?", captured["url"])
        self.assertEqual(candidates[0]["title"], "Castle gate reference")
        self.assertEqual(candidates[0]["url"], "https://images.example.com/yandex-castle.png")

    def test_booru_search_tags_extracts_provider_query_tags(self):
        tags = corrector.booru_search_tags(
            "a cave woman holding a torch in a stone cave",
            max_tags=5,
        )

        self.assertEqual(tags, "cave woman holding torch stone")

    def test_parse_booru_image_candidates_reads_common_json_shapes(self):
        payload = {
            "post": [
                {
                    "id": 123,
                    "file_url": "//images.example.com/booru.jpg",
                    "tags": "cavewoman cave torch",
                    "rating": "explicit",
                    "source": "artist page",
                }
            ]
        }

        candidates = corrector.parse_booru_image_candidates(
            payload,
            provider_name="Gelbooru",
            max_images=1,
        )

        self.assertEqual(candidates[0]["title"], "Gelbooru post 123")
        self.assertEqual(candidates[0]["url"], "https://images.example.com/booru.jpg")
        self.assertIn("Rating: explicit", candidates[0]["summary"])
        self.assertIn("cavewoman cave torch", candidates[0]["summary"])

    def test_collect_gelbooru_image_candidates_uses_dapi_json(self):
        captured = {}
        payload = [
            {
                "id": 44,
                "file_url": "https://images.example.com/gelbooru.png",
                "tags": "castle armor",
            }
        ]

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            return FakeResponse(payload)

        with patch("urllib.request.urlopen", fake_urlopen):
            candidates = corrector.collect_gelbooru_image_candidates(
                "medieval castle armor",
                max_images=1,
                timeout=1,
            )

        query = urllib.parse.parse_qs(urllib.parse.urlsplit(captured["url"]).query)
        self.assertIn("https://gelbooru.com/index.php", captured["url"])
        self.assertEqual(query["page"], ["dapi"])
        self.assertEqual(query["s"], ["post"])
        self.assertEqual(query["q"], ["index"])
        self.assertEqual(query["json"], ["1"])
        self.assertIn("medieval", query["tags"][0])
        self.assertEqual(candidates[0]["title"], "Gelbooru post 44")

    def test_collect_booru_image_candidates_retries_first_tag_when_full_query_is_empty(self):
        captured_tags = []

        def fake_urlopen(request, timeout):
            query = urllib.parse.parse_qs(urllib.parse.urlsplit(request.full_url).query)
            captured_tags.append(query["tags"][0])
            if len(captured_tags) == 1:
                return FakeResponse([])
            return FakeResponse(
                [
                    {
                        "id": 45,
                        "file_url": "https://images.example.com/broader.png",
                        "tags": "medieval",
                    }
                ]
            )

        with patch("urllib.request.urlopen", fake_urlopen):
            candidates = corrector.collect_gelbooru_image_candidates(
                "medieval castle armor",
                max_images=1,
                timeout=1,
            )

        self.assertEqual(captured_tags, ["medieval castle armor", "medieval"])
        self.assertEqual(candidates[0]["url"], "https://images.example.com/broader.png")

    def test_collect_rule34_image_candidates_uses_rule34_dapi_json(self):
        captured = {}
        payload = {
            "post": {
                "id": "77",
                "sample_url": "https://images.example.com/rule34.jpg",
                "tags": "pose anatomy",
            }
        }

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            return FakeResponse(payload)

        with patch("urllib.request.urlopen", fake_urlopen):
            candidates = corrector.collect_rule34_image_candidates(
                "dynamic pose anatomy",
                max_images=1,
                timeout=1,
            )

        query = urllib.parse.parse_qs(urllib.parse.urlsplit(captured["url"]).query)
        self.assertIn("https://api.rule34.xxx/index.php", captured["url"])
        self.assertEqual(query["page"], ["dapi"])
        self.assertEqual(query["json"], ["1"])
        self.assertIn("dynamic", query["tags"][0])
        self.assertEqual(candidates[0]["title"], "Rule34 post 77")

    def test_collect_reference_image_candidates_auto_uses_only_safe_sources(self):
        with patch(
            "krea_prompt_corrector.collect_yandex_image_candidates",
            return_value=[
                {
                    "title": "Yandex armor",
                    "url": "https://images.example.com/yandex-armor.jpg",
                    "summary": "Yandex Images result.",
                }
            ],
        ):
            with patch(
                "krea_prompt_corrector.collect_gelbooru_image_candidates",
                return_value=[
                    {
                        "title": "Gelbooru armor",
                        "url": "https://images.example.com/gelbooru-armor.jpg",
                        "summary": "Gelbooru imageboard result.",
                    }
                ],
            ):
                with patch(
                    "krea_prompt_corrector.collect_rule34_image_candidates",
                    return_value=[
                        {
                            "title": "Rule34 armor",
                            "url": "https://images.example.com/rule34-armor.jpg",
                            "summary": "Rule34 imageboard result.",
                        }
                    ],
                ):
                    with patch(
                        "krea_prompt_corrector.collect_duckduckgo_image_candidates",
                        return_value=[
                            {
                                "title": "DuckDuckGo armor",
                                "url": "https://images.example.com/ddg-armor.jpg",
                                "summary": "DuckDuckGo Images result.",
                            }
                        ],
                    ):
                        with patch(
                            "krea_prompt_corrector.collect_wikipedia_image_candidates",
                            return_value=[
                                {
                                    "title": "Wikipedia armor",
                                    "url": "https://upload.wikimedia.org/armor.jpg",
                                    "summary": "Wikipedia image.",
                                }
                            ],
                        ):
                            candidates = corrector.collect_reference_image_candidates(
                                "medivial armor",
                                max_images=5,
                                timeout=1,
                            )

        self.assertEqual(
            [candidate["title"] for candidate in candidates],
            ["Yandex armor", "DuckDuckGo armor", "Wikipedia armor"],
        )

    def test_collect_reference_image_candidates_can_use_single_selected_source(self):
        with patch(
            "krea_prompt_corrector.collect_yandex_image_candidates",
            return_value=[
                {
                    "title": "Yandex armor",
                    "url": "https://images.example.com/yandex-armor.jpg",
                    "summary": "Yandex Images result.",
                }
            ],
        ) as yandex:
            with patch(
                "krea_prompt_corrector.collect_gelbooru_image_candidates",
                return_value=[
                    {
                        "title": "Gelbooru armor",
                        "url": "https://images.example.com/gelbooru-armor.jpg",
                        "summary": "Gelbooru imageboard result.",
                    }
                ],
            ) as gelbooru:
                with patch(
                    "krea_prompt_corrector.collect_rule34_image_candidates",
                    return_value=[],
                ) as rule34:
                    candidates = corrector.collect_reference_image_candidates(
                        "medivial armor",
                        max_images=3,
                        timeout=1,
                        source="Gelbooru",
                    )

        yandex.assert_not_called()
        rule34.assert_not_called()
        gelbooru.assert_called_once()
        self.assertEqual([candidate["title"] for candidate in candidates], ["Gelbooru armor"])

    def test_collect_reference_image_candidates_deduplicates_provider_results(self):
        with patch(
            "krea_prompt_corrector.collect_yandex_image_candidates",
            return_value=[
                {
                    "title": "Yandex armor",
                    "url": "https://images.example.com/armor.jpg",
                    "summary": "Yandex Images result.",
                }
            ],
        ):
            with patch(
                "krea_prompt_corrector.collect_gelbooru_image_candidates",
                return_value=[
                    {
                        "title": "Duplicate armor",
                        "url": "https://images.example.com/armor.jpg",
                        "summary": "Gelbooru imageboard result.",
                    }
                ],
            ):
                with patch(
                    "krea_prompt_corrector.collect_rule34_image_candidates",
                    return_value=[],
                ):
                    with patch(
                        "krea_prompt_corrector.collect_duckduckgo_image_candidates",
                        return_value=[],
                    ):
                        with patch(
                            "krea_prompt_corrector.collect_wikipedia_image_candidates",
                            return_value=[],
                        ):
                            candidates = corrector.collect_reference_image_candidates(
                                "medivial armor",
                                max_images=2,
                                timeout=1,
                            )

        self.assertEqual(
            [candidate["title"] for candidate in candidates],
            ["Yandex armor"],
        )

    def test_fetch_image_data_url_builds_base64_data_url(self):
        with patch(
            "urllib.request.urlopen",
            return_value=FakeBinaryResponse(b"\x89PNG\r\n\x1a\nimage-bytes", "image/png"),
        ):
            data_url = corrector.fetch_image_data_url(
                "https://upload.wikimedia.org/test.png",
                timeout=1,
            )

        self.assertTrue(data_url.startswith("data:image/png;base64,"))

    def test_fetch_image_data_url_rejects_html_or_placeholder_results(self):
        with patch(
            "urllib.request.urlopen",
            return_value=FakeBinaryResponse(b"<html>not an image</html>", "text/html"),
        ):
            with self.assertRaisesRegex(RuntimeError, "supported direct image"):
                corrector.fetch_image_data_url(
                    "https://images.example.com/not-direct",
                    timeout=1,
                )

    def test_fetch_image_data_url_follows_html_og_image(self):
        calls = []
        html = b'<html><meta property="og:image" content="https://images.example.com/direct.png"></html>'

        def fake_urlopen(request, timeout):
            calls.append(request.full_url)
            if request.full_url.endswith("/page"):
                return FakeBinaryResponse(html, "text/html")
            return FakeBinaryResponse(b"\x89PNG\r\n\x1a\nimage-bytes", "image/png")

        with patch("urllib.request.urlopen", fake_urlopen):
            data_url = corrector.fetch_image_data_url(
                "https://example.com/page",
                timeout=1,
            )

        self.assertEqual(calls, ["https://example.com/page", "https://images.example.com/direct.png"])
        self.assertTrue(data_url.startswith("data:image/png;base64,"))

    def test_analyze_reference_images_sends_images_to_lm_studio(self):
        captured = {}

        def fake_urlopen(request, timeout):
            if request.full_url == "https://upload.wikimedia.org/thumb.jpg":
                return FakeBinaryResponse(b"\xff\xd8\xffimage-bytes", "image/jpeg")
            captured["url"] = request.full_url
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    "Allowed concept facts:\n"
                                    "- Pointed arches, load-bearing stone, narrow lancet windows.\n"
                                    "Rejected scene details:\n"
                                    "- A lone knight, low-angle camera, red sunset, centered composition."
                                )
                            }
                        }
                    ]
                }
            )

        with patch("urllib.request.urlopen", fake_urlopen):
            analysis = corrector.analyze_reference_images(
                base_url="http://127.0.0.1:1234/v1",
                model="qwen3-vl-4b-instruct",
                concept="medivial",
                image_candidates=[
                    {
                        "title": "Medieval architecture",
                        "url": "https://upload.wikimedia.org/thumb.jpg",
                        "summary": "Stone religious and military buildings.",
                    }
                ],
                timeout=1,
                api_key="test-key",
            )

        self.assertIn("Web reference concept glossary for medieval", analysis)
        self.assertIn("Pointed arches", analysis)
        self.assertNotIn("lone knight", analysis)
        self.assertNotIn("centered composition", analysis)
        self.assertEqual(captured["url"], "http://127.0.0.1:1234/v1/chat/completions")
        messages = captured["payload"]["messages"]
        self.assertEqual(messages[1]["role"], "user")
        content = messages[1]["content"]
        self.assertIn("Target prompt: medieval", content[0]["text"])
        self.assertIn("untrusted visual glossary entry", content[0]["text"])
        self.assertIn("Never transfer its subject identity", content[0]["text"])
        self.assertIn("pose, action, expression, camera angle", content[0]["text"])
        self.assertIn("Put all scene-specific observations in the rejected section", content[0]["text"])
        self.assertIn("Automatically found web reference image", content[1]["text"])
        self.assertEqual(content[2]["type"], "image_url")
        self.assertTrue(
            content[2]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        )

    def test_web_reference_analysis_drops_unstructured_scene_description(self):
        def fake_urlopen(request, timeout):
            if request.full_url == "https://images.example.com/reference.jpg":
                return FakeBinaryResponse(b"\xff\xd8\xffimage-bytes", "image/jpeg")
            return FakeResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    "A woman stands under blue neon in a centered portrait, "
                                    "shot from below with a rainy city behind her."
                                )
                            }
                        }
                    ]
                }
            )

        with patch("urllib.request.urlopen", fake_urlopen):
            analysis = corrector.analyze_reference_images(
                base_url="http://127.0.0.1:1234/v1",
                model="qwen3-vl-4b-instruct",
                concept="cyberpunk",
                image_candidates=[
                    {
                        "title": "Cyberpunk reference",
                        "url": "https://images.example.com/reference.jpg",
                        "summary": "Web image result.",
                    }
                ],
                timeout=1,
                api_key="test-key",
            )

        self.assertIn("No safe concept-only facts were retained", analysis)
        self.assertNotIn("woman stands", analysis)
        self.assertNotIn("centered portrait", analysis)

    def test_analyze_reference_images_reports_lm_studio_image_rejection(self):
        def fake_urlopen(request, timeout):
            if request.full_url == "https://images.example.com/bad.jpg":
                return FakeBinaryResponse(b"\xff\xd8\xffimage-bytes", "image/jpeg")
            raise urllib.error.HTTPError(
                request.full_url,
                400,
                "Bad Request",
                {},
                FakeBinaryResponse(b'{"error":"url field must be a base64 encoded image"}', "application/json"),
            )

        with patch("urllib.request.urlopen", fake_urlopen):
            analysis = corrector.analyze_reference_images(
                base_url="http://127.0.0.1:1234/v1",
                model="qwen3-vl-4b-instruct",
                concept="cave scene",
                image_candidates=[
                    {
                        "title": "Blocked search result",
                        "url": "https://images.example.com/bad.jpg",
                        "summary": "Search result.",
                    }
                ],
                timeout=1,
                api_key="test-key",
            )

        self.assertIn("Image analysis unavailable", analysis)
        self.assertIn("LM Studio rejected", analysis)

    def test_audit_messages_request_score_breakage_points_and_repair(self):
        system_prompt = corrector.build_audit_system_prompt(include_krea_settings=True)
        user_message = corrector.build_audit_user_message(
            "bad draft",
            "corrected prompt",
            focus="clear readable hand pose",
            concept_keywords="medivial armor, brutalist architecture",
        )

        self.assertIn("strict Krea 2 prompt compliance auditor", system_prompt)
        self.assertIn("Audit score: <0-100>/100", system_prompt)
        self.assertIn("Breakage points:", system_prompt)
        self.assertIn("Repaired Krea prompt:", system_prompt)
        self.assertIn("Krea generation controls are external parameters", system_prompt)
        self.assertIn("Remove creativity, intensity, complexity, movement", system_prompt)
        self.assertIn("Every breakage point you list must be fixed", system_prompt)
        self.assertIn("Do not preserve contradictions", system_prompt)
        self.assertIn("subject's anatomical left and right", system_prompt)
        self.assertIn("palm or paw direction", system_prompt)
        self.assertIn("running while seated", system_prompt)
        self.assertIn("macro wide full-body close-up", system_prompt)
        self.assertIn("photoreal vector icon", system_prompt)
        self.assertIn("Do not use em dashes, en dashes, semicolons", system_prompt)
        self.assertIn("Do not add notes", system_prompt)
        self.assertIn("Original draft prompt:", user_message)
        self.assertIn("Corrected prompt to audit:", user_message)
        self.assertIn("User-requested result focus:", user_message)
        self.assertIn("clear readable hand pose", user_message)
        self.assertIn("Required concepts that must be represented", user_message)
        self.assertIn("medieval armor, brutalist architecture", user_message)
        self.assertIn("If any are missing, add them", user_message)

    def test_compact_audit_receives_normalized_candidate_and_complete_contract(self):
        source = "A courier reaches a flooded city gate."
        initial = "A courier reaches a flooded city gate\u2014holding a brass satchel."
        calls = []

        def complete(**kwargs):
            calls.append(kwargs["messages"])
            return initial

        with patch("krea_prompt_corrector.chat_completion", side_effect=complete):
            corrector.post_chat_completion(
                base_url="http://127.0.0.1:1234/v1",
                model="test-4b",
                prompt=source,
                temperature=0.2,
                max_tokens=500,
                timeout=30,
                api_key="test",
                visual_direction="Muted watercolor and cool moonlight",
                goal_headline="Deliver medicine through the flood",
                focus="the brass satchel",
                concept_keywords="Art Nouveau ironwork",
                weighted_terms="brass satchel:1.5",
                story_elements="The courier keeps the medicine dry.",
                private_model_instructions="Blend watercolor at 70 percent.",
                generation_feedback="Make the flood visibly more dangerous.",
                research_context="Verified ironwork uses flowing organic curves.",
                image_context="Allowed concept facts: oxidized brass fittings.",
                concept_context="Concept glossary: botanical linework.",
                mode="Surrealist",
                detail_level="Rich caption",
                output_length="Expanded",
                risk_level="Creative enhancement",
                prompt_preset="Cinematic action",
                audit_repair=True,
                final_gate_repair=False,
            )

        self.assertEqual(len(calls), 2)
        self.assertIn(
            "Muted watercolor and cool moonlight",
            calls[0][1]["content"],
        )
        audit_user = calls[1][1]["content"]
        self.assertIn("Current mechanically normalized candidate", audit_user)
        self.assertIn("city gate, holding", audit_user)
        self.assertNotIn("city gate\u2014holding", audit_user)
        for expected in (
            "Muted watercolor and cool moonlight",
            "Deliver medicine through the flood",
            "the brass satchel",
            "Art Nouveau ironwork",
            "brass satchel:1.5",
            "The courier keeps the medicine dry.",
            "Blend watercolor at 70 percent.",
            "Make the flood visibly more dangerous.",
            "flowing organic curves",
            "oxidized brass fittings",
            "botanical linework",
            "mode=Surrealist",
            "detail=Rich caption",
            "risk=Creative enhancement",
            "preset=Cinematic action",
        ):
            self.assertIn(expected, audit_user)

    def test_full_audit_receives_support_context_and_correction_controls(self):
        source = "A red car beneath rainy streetlights."
        with patch(
            "krea_prompt_corrector.chat_completion",
            side_effect=[source, source],
        ) as completion:
            corrector.post_chat_completion(
                base_url="http://127.0.0.1:1234/v1",
                model="test-8b",
                prompt=source,
                temperature=0.2,
                max_tokens=500,
                timeout=30,
                api_key="test",
                visual_direction="Deadpan flash photography",
                goal_headline="A lonely late-night arrival",
                focus="wet red paint",
                model_instructions="Keep the camera low.",
                research_context="Verified wet asphalt creates broad reflections.",
                image_context="Allowed concept facts: hard frontal flash.",
                concept_context="Concept glossary: sodium-vapor amber.",
                mode="Cinematic",
                risk_level="Strict cleanup",
                audit_repair=True,
                final_gate_repair=False,
            )

        audit_user = completion.call_args_list[1].kwargs["messages"][1]["content"]
        for expected in (
            "Deadpan flash photography",
            "A lonely late-night arrival",
            "wet red paint",
            "Keep the camera low.",
            "broad reflections",
            "hard frontal flash",
            "sodium-vapor amber",
            "Mode: Cinematic",
            "Rewrite risk: Strict cleanup",
        ):
            self.assertIn(expected, audit_user)

    def test_meme_model_contract_receives_shared_controls_and_safety(self):
        messages = corrector.build_meme_generation_messages(
            prompt="A cat stares at a broken printer.",
            generator_target="Krea 2",
            variation_count=2,
            safe_for_work=True,
            mode="Surrealist",
            visual_direction="Deadpan flash photography",
            detail_level="Rich caption",
            output_length="Expanded",
            output_min_words=40,
            output_max_words=90,
            risk_level="Creative enhancement",
            prompt_preset="Cinematic action",
            preserve_strictly=True,
            fix_logic=False,
            enhance_actions=True,
            develop_story=False,
            clean_constraints=False,
            altered_text_encoder=False,
            thinking_mode=True,
        )

        system = messages[0]["content"]
        for expected in (
            "Safe-for-work output is mandatory",
            "Required visual mode: Surrealist",
            "Required visual direction: Deadpan flash photography",
            "detail=Rich caption",
            "output length=Expanded",
            "between 40 and 90 words",
            "rewrite risk=Creative enhancement",
            "prompt preset=Cinematic action",
            "Preserve source wording strictly=True",
            "fix logic=False",
            "enhance actions=True",
            "develop supporting story context=False",
            "clean generator constraints=False",
            "altered encoder safe=False",
            "reason internally",
            "Variation 1: through Variation 2:",
        ):
            self.assertIn(expected, system)

    def test_main_completion_forwards_shared_controls_to_meme_pipeline(self):
        with patch(
            "krea_prompt_corrector.post_meme_completion",
            return_value="finished meme",
        ) as meme_completion:
            result = corrector.post_chat_completion(
                base_url="http://127.0.0.1:1234/v1",
                model="test-4b",
                prompt='A cat meme with top caption "MONDAY".',
                generator_target="Krea 2",
                content_format="Meme",
                temperature=0.35,
                max_tokens=500,
                timeout=30,
                api_key="test",
                mode="Surrealist",
                visual_direction="Deadpan flash photography",
                detail_level="Rich caption",
                output_length="Expanded",
                output_min_words=40,
                output_max_words=90,
                risk_level="Creative enhancement",
                prompt_preset="Cinematic action",
                preserve_strictly=True,
                fix_logic=False,
                enhance_actions=True,
                develop_story=False,
                clean_constraints=False,
                altered_text_encoder=False,
                thinking_mode=True,
                safe_for_work=True,
            )

        self.assertEqual(result, "finished meme")
        forwarded = meme_completion.call_args.kwargs
        for key, expected in (
            ("mode", "Surrealist"),
            ("visual_direction", "Deadpan flash photography"),
            ("detail_level", "Rich caption"),
            ("output_length", "Expanded"),
            ("output_min_words", 40),
            ("output_max_words", 90),
            ("risk_level", "Creative enhancement"),
            ("prompt_preset", "Cinematic action"),
            ("preserve_strictly", True),
            ("fix_logic", False),
            ("enhance_actions", True),
            ("develop_story", False),
            ("clean_constraints", False),
            ("altered_text_encoder", False),
            ("thinking_mode", True),
            ("safe_for_work", True),
        ):
            self.assertEqual(forwarded[key], expected)

    def test_knowledge_probe_receives_all_visual_support_fields(self):
        messages = corrector.build_model_knowledge_probe_messages(
            "A courier reaches a flooded gate.",
            concept_keywords="Art Nouveau ironwork",
            story_elements="The courier keeps medicine dry.",
            weighted_terms="brass satchel:1.5",
            goal_headline="A hopeful delivery",
            focus="the medicine satchel",
            model_instructions="Use historically accurate gate hardware.",
        )

        user = messages[1]["content"]
        for expected in (
            "Art Nouveau ironwork",
            "The courier keeps medicine dry.",
            "brass satchel:1.5",
            "A hopeful delivery",
            "the medicine satchel",
            "historically accurate gate hardware",
        ):
            self.assertIn(expected, user)

    def test_support_fields_authorize_their_writing_system_during_validation(self):
        issues = corrector.final_compliance_issues(
            "A storefront sign displays the exact Japanese word \u5e0c\u671b.",
            original_prompt="A storefront sign.",
            concept_keywords="\u5e0c\u671b",
            output_length="Concise",
        )

        self.assertFalse(
            any("Unexpected writing system" in issue for issue in issues)
        )

    def test_extract_repaired_prompt_returns_only_final_prompt(self):
        audit_response = (
            "Audit score: 88/100\n"
            "Breakage points:\n"
            "- Issue\n"
            "Repaired Krea prompt:\n"
            "A clean final prompt.\n"
            "Note: hidden explanation"
        )

        self.assertEqual(
            corrector.extract_repaired_prompt(audit_response),
            "A clean final prompt.",
        )

    def test_system_prompt_contains_expected_krea_and_quality_rules(self):
        prompt = corrector.build_system_prompt(
            mode="Cinematic",
            detail_level="Rich caption",
            output_length="Expanded",
            variation_count=3,
            preserve_strictly=True,
            optimize_quoted_text=True,
            fix_logic=True,
            enhance_actions=True,
            clean_constraints=True,
            include_krea_settings=True,
            creativity="high",
            intensity=200,
            complexity=-200,
            movement=25,
        )

        self.assertIn("Use natural language visual descriptions", prompt)
        self.assertIn("integrate every one into the final prompt", prompt)
        self.assertIn("Shape the prompt toward this visual direction: Cinematic.", prompt)
        self.assertIn("Requested output length: Expanded", prompt)
        self.assertIn("Develop the scene generously", prompt)
        self.assertIn("dense visual detail and coherent caption-like structure", prompt)
        self.assertIn("Return exactly 3 clearly different prompt variations", prompt)
        self.assertIn("Detect and fix prompt logic failures", prompt)
        self.assertIn("Enhance described actions", prompt)
        self.assertIn("Krea 2 does not use a separate negative prompt field", prompt)
        self.assertIn("Interpret feelings visually", prompt)
        self.assertIn("facial expression, gaze, posture", prompt)
        self.assertIn("Build visual storytelling", prompt)
        self.assertIn("comic page, comic strip, storyboard", prompt)
        self.assertIn("Do not blend different moments", prompt)
        self.assertIn("Do not use em dashes, en dashes, semicolons", prompt)
        self.assertIn("put the exact rendered words in double quotes", prompt)
        self.assertNotIn("creativity=high, intensity=100, complexity=-100, movement=25", prompt)
        self.assertIn("Krea generation controls are external parameters", prompt)
        self.assertIn("altered text encoder", prompt)
        self.assertIn("Explicit direct phrasing is allowed", prompt)
        self.assertIn("bind each modifier directly", prompt)
        self.assertIn("Thinking mode is disabled", prompt)

    def test_creativity_spelling_and_semantic_grouping_are_explicit(self):
        system = corrector.build_system_prompt(risk_level="Creative enhancement")
        user = corrector.build_user_message(
            "a lighbulb on a desk, light across the wall",
            concept_keywords="vintage ligthing",
            goal_headline="a suprising discovery",
            focus="the reflecion on glass",
            weighted_terms="lighbulb:1.5",
            story_elements="the charachter switches it on",
            model_instructions="make it more detialed",
            risk_level="Creative enhancement",
        )
        audit = corrector.build_audit_system_prompt()
        repair = corrector.build_final_repair_system_prompt()

        self.assertIn("creative director and precision prompt editor", system)
        self.assertIn("internally explore at least three", system)
        self.assertIn("semantic entity clusters", system)
        self.assertIn("light bulb together with its fixture", system)
        self.assertIn("every user-authored text input", system)
        self.assertIn("Input-wide spelling and organization pass", user)
        self.assertIn("draft prompt, concepts, goal headline, focus", user)
        self.assertIn("keep a light bulb, its fixture and glass", user)
        self.assertIn("semantic entity clusters", audit)
        self.assertIn("potentially misspelled", audit)
        self.assertIn("Regroup related visual details", repair)

    def test_disabled_options_change_prompt_rules(self):
        prompt = corrector.build_system_prompt(
            optimize_quoted_text=False,
            fix_logic=False,
            enhance_actions=False,
            clean_constraints=False,
            altered_text_encoder=False,
            include_krea_settings=False,
        )

        self.assertIn("Do not add extra quote handling", prompt)
        self.assertIn("Do not resolve logic conflicts", prompt)
        self.assertIn("Do not add extra action mechanics", prompt)
        self.assertIn("Leave avoidance wording mostly as provided", prompt)
        self.assertIn("expected Krea 2 text interpretation", prompt)
        self.assertIn("Krea generation controls are external parameters", prompt)

    def test_user_message_can_request_altered_encoder_compatibility(self):
        message = corrector.build_user_message(
            "cinematic cyberpunk portrait",
            altered_text_encoder=True,
        )

        self.assertIn("Altered text encoder compatibility", message)
        self.assertIn("common descriptive phrases", message)

    def test_user_message_sets_thinking_control(self):
        thinking = corrector.build_user_message("a knight", thinking_mode=True)
        no_thinking = corrector.build_user_message("a knight", thinking_mode=False)

        self.assertIn("Thinking control: /think", thinking)
        self.assertIn("Thinking control: /no_think", no_thinking)

    def test_instruction_classifier_splits_prompt_parts(self):
        classified = corrector.classify_prompt_parts(
            'a medieval knight, use cinematic lighting, avoid watermark, poster style, sign says "GATE"'
        )

        self.assertIn("a medieval knight", classified["visual_content"])
        self.assertTrue(any("use cinematic lighting" in item for item in classified["model_instructions"]))
        self.assertTrue(any("avoid watermark" in item for item in classified["avoidances"]))
        self.assertTrue(any("poster style" in item for item in classified["style_references"]))
        self.assertEqual(classified["rendered_text"], ["GATE"])

    def test_instruction_classifier_recognizes_natural_request_instructions(self):
        classified = corrector.classify_prompt_parts(
            "a wounded knight at a castle gate. keep it historically grounded. "
            "make sure the armor is accurate. prioritize readable sword pose. "
            "do not turn it into fantasy. no watermark."
        )

        instructions = " ".join(classified["model_instructions"]).lower()
        avoidances = " ".join(classified["avoidances"]).lower()
        self.assertIn("keep it historically grounded", instructions)
        self.assertIn("make sure the armor is accurate", instructions)
        self.assertIn("prioritize readable sword pose", instructions)
        self.assertIn("do not turn it into fantasy", instructions)
        self.assertIn("no watermark", avoidances)
        self.assertIn("a wounded knight at a castle gate", classified["visual_content"])

    def test_explicit_instruction_contract_reports_dropped_orders(self):
        original = (
            "A knight waits at a castle gate. Keep the gate open. "
            "Make sure the knight wears a red cloak. Do not turn it into fantasy."
        )
        issues = corrector.explicit_instruction_issues(
            "A knight in steel armor waits before a closed stone arch.",
            original,
        )

        joined = "\n".join(issues).lower()
        self.assertIn("explicit user directives missing", joined)
        self.assertIn("gate open", joined)
        self.assertIn("red cloak", joined)
        self.assertIn("fantasy", joined)

    def test_explicit_instruction_contract_restores_missing_orders(self):
        original = "A knight at a gate. Keep the gate open. Use cold moonlight."

        repaired = corrector.enforce_explicit_instruction_contract(
            "A knight stands at a stone gate.",
            original,
        )

        self.assertIn("Keep the gate open", repaired)
        self.assertIn("Use cold moonlight", repaired)
        self.assertNotIn("Mandatory user constraints:", repaired)
        self.assertFalse(corrector.explicit_instruction_issues(repaired, original))

    def test_private_concept_mix_guidance_is_removed_from_visible_output(self):
        private_guidance = corrector.concept_mix_instruction(
            "watercolor:70%, cyberpunk:30%"
        )
        leaked = (
            "A rain-soaked cyberpunk city rendered with translucent watercolor washes. "
            + private_guidance
        )

        cleaned = corrector.strip_private_prompt_guidance(leaked)

        self.assertIn("rain-soaked cyberpunk city", cleaned)
        self.assertNotIn("Deliberate concept and style blend", cleaned)
        self.assertNotIn("relative creative influence", cleaned)
        self.assertNotIn("every nonzero ingredient", cleaned)
        self.assertNotIn("additional person", cleaned)
        self.assertNotIn("numeric weights", cleaned)
        self.assertFalse(corrector.internal_prompt_guidance_issues(cleaned))

    def test_adult_fidelity_audit_summary_is_removed_from_visible_output(self):
        visible = "An adult uses a dildo in a direct vaginal self-stimulation scene."
        leaked = (
            visible
            + " All facts preserved: action=toy use, self-stimulation of their "
            "own genitals, contact=vaginal, object=dildo. No euphemism, no added "
            "identity or dynamic."
        )

        self.assertTrue(corrector.internal_prompt_guidance_issues(leaked))
        cleaned = corrector.strip_private_prompt_guidance(leaked)

        self.assertEqual(cleaned, visible)
        self.assertNotIn("All facts preserved", cleaned)
        self.assertNotIn("action=", cleaned)
        self.assertFalse(corrector.internal_prompt_guidance_issues(cleaned))

    def test_scoped_mix_control_text_is_removed_from_visible_output(self):
        guidance = corrector.concept_mix_groups_instruction(
            "",
            [
                {
                    "name": "Subject",
                    "target": "Main subject",
                    "mix": "ink:60%, pastel:40%",
                }
            ],
        )
        visible = "An ink-lined subject with restrained pastel shading."

        cleaned = corrector.strip_private_prompt_guidance(
            visible + " " + guidance
        )

        self.assertEqual(cleaned, visible)
        self.assertFalse(corrector.internal_prompt_guidance_issues(cleaned))

    def test_visual_element_control_labels_are_removed_from_visible_output(self):
        leaked = (
            "A green alien-textured toy in a sunny park. "
            "Required visual elements: xenomorph. "
            "Prominent visual elements: xenomorph."
        )

        self.assertTrue(corrector.internal_prompt_guidance_issues(leaked))
        cleaned = corrector.naturalize_krea_workflow_labels(
            corrector.strip_private_prompt_guidance(leaked)
        )

        self.assertNotIn("Required visual elements", cleaned)
        self.assertNotIn("Prominent visual elements", cleaned)
        self.assertIn("xenomorph", cleaned)
        self.assertFalse(corrector.internal_prompt_guidance_issues(cleaned))

    def test_private_revision_block_is_removed_from_visible_output(self):
        leaked = (
            "A red-cloaked knight at a castle gate under warm torchlight. "
            "Private revision guidance for this correction pass only: "
            "Use warmer torchlight and a lower camera.\n"
            "Apply its meaning through concrete visual changes. Do not quote, label, mention, "
            "or append this guidance in the final prompt."
        )

        cleaned = corrector.strip_private_prompt_guidance(leaked)

        self.assertEqual(
            cleaned,
            "A red-cloaked knight at a castle gate under warm torchlight.",
        )

    def test_private_rule_strength_instruction_is_removed_from_visible_output(self):
        leaked = (
            "A red-cloaked knight at a castle gate. "
            + corrector.rule_strength_instruction(65)
        )

        cleaned = corrector.strip_private_prompt_guidance(leaked)

        self.assertEqual(cleaned, "A red-cloaked knight at a castle gate.")
        self.assertFalse(corrector.internal_prompt_guidance_issues(cleaned))

    def test_post_completion_keeps_mix_guidance_private(self):
        private_guidance = corrector.concept_mix_instruction(
            "watercolor:70%, cyberpunk:30%"
        )
        leaked = (
            "A rain-soaked cyberpunk city rendered with translucent watercolor washes. "
            + private_guidance
        )
        with patch(
            "krea_prompt_corrector.chat_completion",
            return_value=leaked,
        ) as completion:
            result = corrector.post_chat_completion(
                base_url="http://127.0.0.1:1234/v1",
                model="qwen3-vl-4b-instruct",
                prompt="A rain-soaked city.",
                temperature=0.2,
                max_tokens=300,
                timeout=12,
                api_key="test-key",
                concept_keywords="watercolor, cyberpunk",
                private_model_instructions=private_guidance,
                context_token_budget=4096,
                audit_repair=False,
                final_gate_repair=False,
            )

        sent_message = completion.call_args.kwargs["messages"][1]["content"]
        self.assertIn("watercolor 70%", sent_message)
        self.assertIn("rain-soaked cyberpunk city", result)
        self.assertNotIn("Mandatory user constraints", result)
        self.assertNotIn("every nonzero ingredient", result)
        self.assertNotIn("relative creative influence", result)

    def test_post_completion_removes_adult_fidelity_audit_summary(self):
        visible = "An adult woman vaginally stimulates herself with a dildo."
        leaked = (
            visible
            + " All facts preserved: action=toy use, self-stimulation of their "
            "own genitals, contact=vaginal, object=dildo. No euphemism, no added "
            "identity or dynamic."
        )
        with patch(
            "krea_prompt_corrector.chat_completion",
            return_value=leaked,
        ):
            result = corrector.post_chat_completion(
                base_url="http://127.0.0.1:1234/v1",
                model="qwen3-vl-4b-instruct",
                prompt=visible,
                temperature=0.2,
                max_tokens=300,
                timeout=12,
                api_key="test-key",
                explicit_nsfw=True,
                context_token_budget=4096,
                audit_repair=False,
                final_gate_repair=False,
            )

        self.assertEqual(result, visible)
        self.assertFalse(corrector.internal_prompt_guidance_issues(result))

    def test_internal_prompt_guidance_is_a_hard_compliance_issue(self):
        issues = corrector.final_compliance_issues(
            "A watercolor city. Validation issues: missing required concepts.",
            original_prompt="A watercolor city.",
        )

        leak_issues = [
            issue
            for issue in issues
            if issue.startswith("Internal prompt guidance leaked")
        ]
        self.assertTrue(leak_issues)
        self.assertTrue(corrector.is_hard_compliance_issue(leak_issues[0]))

    def test_panel_description_contract_requires_more_than_one_surviving_word(self):
        source = (
            "Panel 1: a knight opens the iron gate with a torch. "
            "Panel 2: the knight crosses the flooded courtyard carrying the torch."
        )
        final = (
            "Panel 1: a knight waits. "
            "Panel 2: the knight crosses the courtyard carrying a torch."
        )

        issues = corrector.panel_description_issues(final, source)

        self.assertTrue(any("Panel 1" in issue for issue in issues))
        self.assertFalse(any("Panel 2" in issue for issue in issues))

    def test_local_audit_helpers_report_diff_confidence_and_score(self):
        diff = corrector.prompt_diff_summary(
            "a knight in a castle with a torch",
            "a knight in polished armor inside a stone castle gate",
        )
        confidence = corrector.research_confidence_report(
            "Findings:\n1. Medieval armor uses mail.\n2. Castles use stone gates."
        )
        score = corrector.final_score_report(
            "A knight in polished medieval armor inside a stone castle gate, torchlit and coherent.",
            original_prompt="a knight in a castle",
            concept_keywords="medieval armor",
            output_length="Concise",
        )

        self.assertIn("Prompt diff review", diff)
        self.assertIn("Added terms", diff)
        self.assertIn("Research confidence: strong", confidence)
        self.assertIn("Final score panel", score)
        self.assertIn("Overall:", score)

    def test_sanitation_helpers_find_breakage_points(self):
        self.assertTrue(
            corrector.rendered_text_issues(
                "A poster on a wall",
                original_prompt="poster says welcome home",
            )
        )
        self.assertTrue(corrector.style_conflict_issues("photoreal flat vector icon"))
        self.assertTrue(
            corrector.entity_consistency_issues(
                "A character wearing steel iron gold wooden leather armor."
            )
        )
        self.assertTrue(
            corrector.intent_lock_issues(
                "a knight carrying a torch in a castle",
                "a futuristic car on a beach",
            )
        )

    def test_final_compliance_issues_reports_altered_encoder_risks(self):
        issues = corrector.final_compliance_issues(
            "masterpiece cyberpunk portrait, cinematic, anime",
            output_length="Concise",
            altered_text_encoder=True,
        )

        self.assertTrue(any("Risky phrasing for altered text encoder" in issue for issue in issues))

    def test_post_chat_completion_sends_krea_options_to_lm_studio(self):
        captured = {}
        responses = [
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                "A cinematic skateboarder captured mid-kickflip, knees bent, "
                                "weight centered over the board \u2014 clean urban background."
                            )
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                "A cinematic skateboarder lands a clean trick by catching the board "
                                "mid-kickflip above rain slick concrete, knees bent and weight centered "
                                "for clear contact and motion. brutalist architecture frames the grounded "
                                "sports photograph, while directional city light outlines the athlete and "
                                'a background sign reads "hello world".'
                            )
                        }
                    }
                ]
            },
        ]

        def fake_urlopen(request, timeout):
            captured["timeout"] = timeout
            captured["url"] = request.full_url
            captured["headers"] = dict(request.header_items())
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            captured.setdefault("payloads", []).append(captured["payload"])
            return FakeResponse(responses.pop(0))

        draft = random_bad_prompt()
        with patch("urllib.request.urlopen", fake_urlopen):
            result = corrector.post_chat_completion(
                base_url="http://127.0.0.1:1234/v1",
                model="qwen3-vl-4b-instruct",
                prompt=draft,
                temperature=0.2,
                max_tokens=600,
                timeout=12,
                api_key="test-key",
                mode="Cinematic",
                detail_level="Detailed",
                output_length="Concise",
                output_min_words=35,
                output_max_words=75,
                risk_level="Strict cleanup",
                prompt_preset="Cinematic action",
                variation_count=1,
                preserve_strictly=False,
                optimize_quoted_text=True,
                fix_logic=True,
                enhance_actions=True,
                clean_constraints=True,
                include_krea_settings=False,
                creativity="medium",
                intensity=0,
                complexity=0,
                movement=50,
                research_context="Search query: skateboard kickflip\nFindings:\n1. Kickflip - feet flick the board.",
                concept_context=(
                    "User-requested concepts to integrate:\n"
                    "1. Concept: brutalist architecture\n"
                    "Findings:\n1. Massive raw concrete forms."
                ),
                goal_headline="Skateboarder lands a clean trick in a concrete city",
                focus="board contact and motion clarity",
                concept_keywords="brutalist architecture, rain slick concrete",
                model_instructions="Prefer a grounded sports photography rewrite.",
                story_elements="the skateboarder catches the board as rain splashes from the concrete",
            )

        self.assertIn("mid-kickflip", result)
        self.assertNotIn("\u2014", result)
        self.assertIn("brutalist architecture", result)
        self.assertIn("rain slick concrete", result)
        self.assertEqual(captured["timeout"], 12)
        self.assertEqual(captured["url"], "http://127.0.0.1:1234/v1/chat/completions")
        self.assertEqual(captured["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(captured["payloads"][0]["model"], "qwen3-vl-4b-instruct")
        self.assertEqual(captured["payloads"][0]["temperature"], 0.2)
        self.assertEqual(captured["payloads"][0]["max_tokens"], 600)

        messages = captured["payloads"][0]["messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("Enhance described actions", messages[0]["content"])
        self.assertIn("Anatomical left and right", messages[0]["content"])
        self.assertIn("weight-bearing limb", messages[0]["content"])
        self.assertIn("Story development is enabled", messages[0]["content"])
        self.assertIn("Requested output length: Concise", messages[0]["content"])
        self.assertIn("between 35 and 75 words", messages[0]["content"])
        self.assertIn("Rewrite risk level: Strict cleanup", messages[0]["content"])
        self.assertIn("Prompt preset: Cinematic action", messages[0]["content"])
        self.assertLess(len(messages[0]["content"].split()), 220)
        self.assertIn("Grounded research", messages[1]["content"])
        self.assertIn("feet flick the board", messages[1]["content"])
        self.assertIn("Concept context", messages[1]["content"])
        self.assertIn("Massive raw concrete forms", messages[1]["content"])
        self.assertIn("Required concepts", messages[1]["content"])
        self.assertIn("brutalist architecture, rain slick concrete", messages[1]["content"])
        self.assertIn("Goal", messages[1]["content"])
        self.assertIn("Skateboarder lands a clean trick", messages[1]["content"])
        self.assertIn("Primary focus", messages[1]["content"])
        self.assertIn("board contact and motion clarity", messages[1]["content"])
        self.assertIn("Story or panel beats", messages[1]["content"])
        self.assertIn("rain splashes from the concrete", messages[1]["content"])
        self.assertIn("Output length: Concise", messages[1]["content"])
        self.assertIn("User transformation instructions", messages[1]["content"])
        self.assertIn("grounded sports photography", messages[1]["content"])
        self.assertIn(draft, messages[1]["content"])
        self.assertEqual(captured["payloads"][1]["messages"][0]["role"], "system")
        self.assertIn("deterministic validation", captured["payloads"][1]["messages"][0]["content"])
        self.assertIn("Coherent story invention and extension is authorized", captured["payloads"][1]["messages"][0]["content"])
        self.assertIn("Validation issues", captured["payloads"][1]["messages"][1]["content"])
        self.assertIn("between 35 and 75 words", captured["payloads"][1]["messages"][1]["content"])

    def test_generation_feedback_guides_revision_without_becoming_a_final_constraint(self):
        completed_prompt = (
            "A red-cloaked knight stands at an ancient castle gate, seen from a low camera "
            "angle as warm torchlight shapes the stone arch and polished armor."
        )
        with patch(
            "krea_prompt_corrector.chat_completion",
            return_value=completed_prompt,
        ) as completion:
            result = corrector.post_chat_completion(
                base_url="http://127.0.0.1:1234/v1",
                model="qwen3-vl-4b-instruct",
                prompt="A red-cloaked knight stands at an ancient castle gate.",
                temperature=0.2,
                max_tokens=300,
                timeout=12,
                api_key="test-key",
                generation_feedback=(
                    "The previous iteration ignored my revision request. In the next version, "
                    "use warmer torchlight and a lower camera."
                ),
                audit_repair=False,
                final_gate_repair=False,
            )

        sent_message = completion.call_args.kwargs["messages"][1]["content"]
        self.assertIn("Private revision guidance for this correction pass only", sent_message)
        self.assertIn("previous iteration ignored my revision request", sent_message)
        self.assertIn("Do not quote, label, mention, or append this guidance", sent_message)
        self.assertEqual(result, completed_prompt)
        self.assertNotIn("Mandatory user constraints", result)
        self.assertNotIn("revision guidance", result.lower())

    def test_post_chat_completion_respects_context_token_budget(self):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse(
                {
                    "choices": [
                        {"message": {"content": "A concise corrected Krea prompt."}}
                    ]
                }
            )

        research_context = "RESEARCH_START " + "R" * 1600 + " RESEARCH_END"
        concept_context = "CONCEPT_START " + "C" * 1600 + " CONCEPT_END"
        with patch("urllib.request.urlopen", fake_urlopen):
            corrector.post_chat_completion(
                base_url="http://127.0.0.1:1234/v1",
                model="qwen3-vl-4b-instruct",
                prompt="a knight in a castle",
                temperature=0.2,
                max_tokens=300,
                timeout=12,
                api_key="test-key",
                research_context=research_context,
                concept_context=concept_context,
                context_token_budget=512,
                audit_repair=False,
                final_gate_repair=False,
            )

        message = captured["payload"]["messages"][1]["content"]
        self.assertIn("RESEARCH_START", message)
        self.assertNotIn("RESEARCH_END", message)
        self.assertIn("CONCEPT_START", message)
        self.assertNotIn("CONCEPT_END", message)

    def test_reference_image_context_has_its_own_preserved_section(self):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse(
                {"choices": [{"message": {"content": "A corrected prompt."}}]}
            )

        with patch("urllib.request.urlopen", fake_urlopen):
            corrector.post_chat_completion(
                base_url="http://127.0.0.1:1234/v1",
                model="qwen3-vl-4b-instruct",
                prompt="a knight",
                temperature=0.2,
                max_tokens=200,
                timeout=3,
                api_key="test-key",
                research_context="\n".join(f"Research {index}" for index in range(20)),
                image_context="Relevant visual traits: red cloak and steel armor.",
                context_token_budget=512,
                final_gate_repair=False,
            )

        message = captured["payload"]["messages"][1]["content"]
        self.assertIn("Reference image findings", message)
        self.assertIn("red cloak and steel armor", message)
        self.assertIn("every image is a glossary, never a scene template", message)
        self.assertIn("Never copy a source scene, subject, pose, action, camera, composition", message)

    def test_chat_completion_stream_can_be_cancelled_and_closes_response(self):
        class Cancelled(Exception):
            pass

        response = FakeStreamingResponse(
            [
                b'data: {"choices":[{"delta":{"content":"partial"}}]}\n',
                b'data: {"choices":[{"delta":{"content":" result"}}]}\n',
                b"data: [DONE]\n",
            ]
        )
        calls = 0
        chunks = []

        def cancel_check():
            nonlocal calls
            calls += 1
            if calls >= 3:
                raise Cancelled()

        with patch("urllib.request.urlopen", return_value=response) as urlopen:
            with self.assertRaises(Cancelled):
                corrector.chat_completion(
                    base_url="http://127.0.0.1:1234/v1",
                    model="qwen3-vl-4b-instruct",
                    messages=[{"role": "user", "content": "test"}],
                    temperature=0.1,
                    max_tokens=100,
                    timeout=10,
                    api_key="test-key",
                    cancel_check=cancel_check,
                    chunk_callback=chunks.append,
                )

        payload = json.loads(urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertTrue(payload["stream"])
        self.assertTrue(response.closed)
        self.assertEqual(chunks, ["partial"])

    def test_chat_completion_cancel_closes_a_blocked_stream(self):
        class Cancelled(Exception):
            pass

        class BlockingResponse(FakeStreamingResponse):
            def __init__(self):
                super().__init__([])
                self.read_started = threading.Event()
                self.closed_event = threading.Event()

            def readline(self):
                self.read_started.set()
                self.closed_event.wait(2)
                return b""

            def close(self):
                self.closed = True
                self.closed_event.set()

        response = BlockingResponse()
        cancelled = threading.Event()
        caught = []

        def cancel_check():
            if cancelled.is_set():
                raise Cancelled()

        def run_completion():
            try:
                corrector.chat_completion(
                    base_url="http://127.0.0.1:1234/v1",
                    model="qwen3-vl-4b-instruct",
                    messages=[{"role": "user", "content": "test"}],
                    temperature=0.1,
                    max_tokens=100,
                    timeout=10,
                    api_key="test-key",
                    cancel_check=cancel_check,
                )
            except Exception as exc:
                caught.append(exc)

        with patch("urllib.request.urlopen", return_value=response):
            worker = threading.Thread(target=run_completion)
            worker.start()
            self.assertTrue(response.read_started.wait(1))
            cancelled.set()
            worker.join(1)

        self.assertFalse(worker.is_alive())
        self.assertTrue(response.closed)
        self.assertEqual(len(caught), 1)
        self.assertIsInstance(caught[0], Cancelled)

    def test_audit_repair_runs_second_lm_studio_pass(self):
        captured_payloads = []
        responses = [
            {
                "choices": [
                    {
                        "message": {
                            "content": "A corrected but still imperfect Krea prompt."
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                "Audit score: 92/100\n"
                                "Breakage points:\n"
                                "- Minor wording issue\n"
                                "Repaired Krea prompt:\n"
                                "A repaired Krea-ready prompt."
                            )
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                "A chef in articulated medieval armor swings a sword with the right hand "
                                "in a balanced forward stance inside a stone kitchen courtyard. Warm "
                                "directional light reveals chainmail, plate details, cooking tools, and "
                                'coherent shadows, while a wooden sign clearly reads "hello world" behind '
                                "the focused subject."
                            )
                        }
                    }
                ]
            },
        ]

        def fake_urlopen(request, timeout):
            captured_payloads.append(json.loads(request.data.decode("utf-8")))
            return FakeResponse(responses.pop(0))

        with patch("urllib.request.urlopen", fake_urlopen):
            result = corrector.post_chat_completion(
                base_url="http://127.0.0.1:1234/v1",
                model="qwen3-vl-8b-instruct",
                prompt=random_bad_prompt(),
                temperature=0.2,
                max_tokens=900,
                timeout=12,
                api_key="test-key",
                audit_repair=True,
                include_krea_settings=True,
                concept_keywords="medivial armor",
            )

        self.assertEqual(len(captured_payloads), 3)
        self.assertIn("medieval armor", result)
        self.assertIn("creative director and precision prompt editor", captured_payloads[0]["messages"][0]["content"])
        self.assertIn(
            "prompt compliance auditor",
            captured_payloads[1]["messages"][0]["content"],
        )
        self.assertIn(
            "A corrected but still imperfect Krea prompt.",
            captured_payloads[1]["messages"][1]["content"],
        )
        self.assertIn(
            "Required concepts that must be represented",
            captured_payloads[1]["messages"][1]["content"],
        )
        self.assertIn("medieval armor", captured_payloads[1]["messages"][1]["content"])
        self.assertIn(
            "deterministic validation",
            captured_payloads[2]["messages"][0]["content"],
        )

    def test_unexpected_lm_studio_response_raises_clear_error(self):
        with patch("urllib.request.urlopen", return_value=FakeResponse({"bad": "shape"})):
            with self.assertRaisesRegex(RuntimeError, "Unexpected LM Studio response"):
                corrector.post_chat_completion(
                    base_url="http://127.0.0.1:1234/v1",
                    model="qwen3-vl-4b-instruct",
                    prompt=random_bad_prompt(),
                    temperature=0.2,
                    max_tokens=200,
                    timeout=3,
                    api_key="test-key",
                )

    def test_reasoning_only_response_reports_token_budget_problem(self):
        response = {
            "choices": [{
                "message": {"content": "", "reasoning_content": "Still thinking..."},
                "finish_reason": "length",
            }]
        }
        with patch("urllib.request.urlopen", return_value=FakeResponse(response)):
            with self.assertRaisesRegex(RuntimeError, "output budget for hidden reasoning"):
                corrector.chat_completion(
                    base_url="http://127.0.0.1:1234/v1",
                    model="qwen3.5-4b-uncensored-hauhaucs-aggressive",
                    messages=[{"role": "user", "content": "Correct this prompt."}],
                    temperature=0.1,
                    max_tokens=128,
                    timeout=3,
                    api_key="test-key",
                )

    def test_ollama_chat_disables_reasoning_and_recognizes_reasoning_field(self):
        captured = {}
        response = {
            "choices": [{
                "message": {"content": "", "reasoning": "Still thinking..."},
                "finish_reason": "length",
            }]
        }

        def fake_urlopen(request, timeout):
            captured.update(json.loads(request.data.decode("utf-8")))
            return FakeResponse(response)

        with patch("urllib.request.urlopen", fake_urlopen):
            with self.assertRaisesRegex(RuntimeError, "Ollama used the output budget"):
                corrector.chat_completion(
                    base_url="http://127.0.0.1:11434/v1",
                    model="qwen3.5:0.8b",
                    messages=[{"role": "user", "content": "Correct this prompt."}],
                    temperature=0.1,
                    max_tokens=128,
                    timeout=3,
                    api_key="",
                )

        self.assertEqual(captured["reasoning_effort"], "none")

    def test_lm_studio_timeout_raises_clear_error(self):
        with patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
            with self.assertRaisesRegex(RuntimeError, "LM Studio timed out"):
                corrector.post_chat_completion(
                    base_url="http://127.0.0.1:1234/v1",
                    model="qwen3-vl-4b-instruct",
                    prompt=random_bad_prompt(),
                    temperature=0.2,
                    max_tokens=200,
                    timeout=3,
                    api_key="test-key",
                )

    def test_lm_studio_rest_api_base_url_removes_openai_v1_suffix(self):
        self.assertEqual(
            corrector.lm_studio_rest_api_base_url("http://remote-pc:1234/v1"),
            "http://remote-pc:1234",
        )

    def test_unload_lm_studio_model_uses_loaded_instance_id(self):
        captured = []
        models_payload = {
            "models": [
                {
                    "key": "qwen3-vl-4b-instruct",
                    "display_name": "Qwen VL",
                    "loaded_instances": [{"id": "qwen3-vl-4b-instruct"}],
                }
            ]
        }
        unload_payload = {"instance_id": "qwen3-vl-4b-instruct"}

        def fake_urlopen(request, timeout):
            captured.append(
                {
                    "url": request.full_url,
                    "method": request.get_method(),
                    "body": json.loads(request.data.decode("utf-8")) if request.data else None,
                }
            )
            if request.full_url.endswith("/api/v1/models"):
                return FakeResponse(models_payload)
            return FakeResponse(unload_payload)

        with patch("urllib.request.urlopen", fake_urlopen):
            unloaded = corrector.unload_lm_studio_model(
                base_url="http://127.0.0.1:1234/v1",
                model="qwen3-vl-4b-instruct",
                timeout=3,
                api_key="test-key",
            )

        self.assertEqual(unloaded, ["qwen3-vl-4b-instruct"])
        self.assertEqual(captured[0]["url"], "http://127.0.0.1:1234/api/v1/models")
        self.assertEqual(captured[0]["method"], "GET")
        self.assertEqual(captured[1]["url"], "http://127.0.0.1:1234/api/v1/models/unload")
        self.assertEqual(captured[1]["method"], "POST")
        self.assertEqual(captured[1]["body"], {"instance_id": "qwen3-vl-4b-instruct"})

    def test_ollama_context_and_unload_use_native_api(self):
        captured = []

        def fake_urlopen(request, timeout):
            payload = json.loads(request.data.decode("utf-8")) if request.data else None
            captured.append((request.full_url, payload, timeout))
            if request.full_url.endswith("/api/show"):
                return FakeResponse(
                    {"model_info": {"qwen3.context_length": 32768}}
                )
            return FakeResponse({"done": True})

        with patch("urllib.request.urlopen", fake_urlopen):
            context_length = corrector.model_context_length(
                base_url="http://localhost:11434/v1",
                model="qwen3:4b",
                timeout=3,
                api_key="",
            )
            unloaded = corrector.unload_local_model(
                provider="Ollama",
                base_url="http://localhost:11434/v1",
                model="qwen3:4b",
                timeout=3,
                api_key="",
            )

        self.assertEqual(context_length, 32768)
        self.assertEqual(unloaded, ["qwen3:4b"])
        self.assertEqual(captured[0][0], "http://localhost:11434/api/show")
        self.assertEqual(captured[0][1], {"model": "qwen3:4b"})
        self.assertEqual(captured[1][0], "http://localhost:11434/api/generate")
        self.assertEqual(captured[1][1], {"model": "qwen3:4b", "keep_alive": 0})

    def test_collect_duckduckgo_research_handles_search_failure(self):
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("network blocked"),
        ):
            context = corrector.collect_duckduckgo_research("samurai katana", timeout=1)

        self.assertIn("DuckDuckGo supplemental search unavailable", context)

    def test_collect_bing_research_formats_results(self):
        html = """
        <li class="b_algo">
          <h2><a href="https://example.com/castles">Medieval castles</a></h2>
          <p>Castles often include stone walls, towers, gates, and defensive details.</p>
        </li>
        """

        with patch(
            "urllib.request.urlopen",
            return_value=FakeBinaryResponse(html.encode("utf-8"), "text/html"),
        ):
            context = corrector.collect_bing_research("medivial", timeout=1)

        self.assertIn("Bing supplemental query", context)
        self.assertIn("Medieval castles - Castles often include stone walls", context)

    def test_research_worker_merges_provider_results_in_stable_order(self):
        def fake_wikipedia(prompt, max_results, timeout):
            return "Wikipedia facts"

        def fake_bing(prompt, max_results, timeout):
            return "Bing facts"

        def fake_duckduckgo(prompt, max_results, timeout):
            return "DuckDuckGo facts"

        with patch("krea_prompt_corrector.collect_wikipedia_research", fake_wikipedia):
            with patch("krea_prompt_corrector.collect_bing_research", fake_bing):
                with patch("krea_prompt_corrector.collect_duckduckgo_research", fake_duckduckgo):
                    context = corrector.collect_research_worker_context(
                        "medivial",
                        timeout=1,
                    )

        self.assertLess(
            context.index("Baseline source - Wikipedia"),
            context.index("Supplemental source - Bing"),
        )
        self.assertLess(
            context.index("Supplemental source - Bing"),
            context.index("Supplemental source - DuckDuckGo"),
        )
        self.assertIn("Wikipedia facts", context)
        self.assertIn("Bing facts", context)
        self.assertIn("DuckDuckGo facts", context)

    def test_collect_concept_research_uses_worker_pipeline(self):
        with patch(
            "krea_prompt_corrector.collect_research_worker_context",
            return_value="Deep research worker results:\nBaseline source - Wikipedia:\nWikipedia facts",
        ) as worker:
            context = corrector.collect_concept_research("medivial", timeout=1)

        worker.assert_called_once()
        self.assertIn("Deep research worker results", context)
        self.assertIn("Wikipedia facts", context)

    def test_collect_concept_research_can_use_single_search_engine(self):
        with patch(
            "krea_prompt_corrector.collect_research_worker_context",
            return_value="Deep research worker results:\nSupplemental source - Bing:\nBing facts",
        ) as worker:
            context = corrector.collect_concept_research(
                "medivial",
                timeout=1,
                search_engine="Bing",
            )

        self.assertEqual(worker.call_args.kwargs["providers"], ("bing",))
        self.assertIn("Bing facts", context)

    def test_model_knowledge_probe_covers_visual_target_categories(self):
        messages = corrector.build_model_knowledge_probe_messages(
            "a samurai draws a katana beside a torii gate",
            concept_keywords="Edo armor",
            story_elements="the blade catches on the scabbard",
            weighted_terms="katana:2.0",
        )

        self.assertIn("Before web research", messages[0]["content"])
        self.assertIn("actions or pose mechanics", messages[0]["content"])
        self.assertIn("objects, materials, places", messages[0]["content"])
        self.assertIn("TARGET | category | term", messages[0]["content"])
        self.assertIn("Edo armor", messages[1]["content"])
        self.assertIn("blade catches on the scabbard", messages[1]["content"])

    def test_prompt_research_targets_merge_model_concepts_actions_and_objects(self):
        probe = "\n".join(
            (
                "TARGET | object | katana | known | curved Japanese sword with a guard and scabbard",
                "TARGET | action | drawing a katana | uncertain | hand placement and blade path need checking",
                "TARGET | material | tamahagane steel | known | layered blade material",
            )
        )
        targets = corrector.prompt_research_targets(
            "a samurai drawing a katana",
            probe,
            concept_keywords="Edo armor",
            weighted_terms="lacquered scabbard:1.8",
        )

        pairs = {(target["category"], target["term"]) for target in targets}
        self.assertIn(("concept", "Edo armor"), pairs)
        self.assertIn(("object", "katana"), pairs)
        self.assertIn(("action", "drawing a katana"), pairs)
        self.assertIn(("material", "tamahagane steel"), pairs)
        self.assertIn(("visual term", "lacquered scabbard"), pairs)

    def test_targeted_research_checks_each_term_and_reconciliation_compares_sources(self):
        targets = [
            {"category": "object", "term": "astrolabe", "confidence": "known", "knowledge": ""},
            {"category": "action", "term": "drawing a longbow", "confidence": "uncertain", "knowledge": ""},
        ]

        with patch(
            "krea_prompt_corrector.collect_concept_research",
            side_effect=lambda query, **_kwargs: f"Web evidence for {query}",
        ) as collect:
            web_context = corrector.collect_targeted_prompt_research(
                targets,
                search_engine="Wikipedia",
            )

        self.assertEqual(collect.call_count, 2)
        self.assertIn("object: astrolabe", web_context)
        self.assertIn("action: drawing a longbow", web_context)
        self.assertIn("parts use scale", web_context)
        self.assertIn("body mechanics", web_context)

        messages = corrector.build_knowledge_reconciliation_messages(
            "an archer with an astrolabe",
            "model prior knowledge",
            web_context,
        )
        self.assertIn("Compare every target", messages[0]["content"])
        self.assertIn("Corrections to model knowledge", messages[0]["content"])
        self.assertIn("objects to their parts/materials/use", messages[0]["content"])
        self.assertIn("model prior knowledge", messages[1]["content"])
        self.assertIn("Targeted web verification results", messages[1]["content"])

    def test_research_urlopen_preserves_ssl_certificate_failure(self):
        calls = 0
        ssl_error = ssl.SSLCertVerificationError("self-signed certificate")

        def fake_urlopen(request, timeout):
            nonlocal calls
            calls += 1
            raise urllib.error.URLError(ssl_error)

        with patch("urllib.request.urlopen", fake_urlopen):
            with self.assertRaises(urllib.error.URLError) as raised:
                corrector.research_urlopen("request", timeout=1)

        self.assertIs(raised.exception.reason, ssl_error)
        self.assertEqual(calls, 1)

    def test_visual_modes_and_prompt_presets_cover_major_workflows(self):
        self.assertGreaterEqual(len(corrector.PROMPT_MODES), 80)
        self.assertEqual(len(corrector.PROMPT_MODES), len(set(corrector.PROMPT_MODES)))
        for mode in (
            "Photoreal",
            "Underwater photography",
            "Manga",
            "Photoreal 3D render",
            "Watercolor",
            "Ukiyo-e",
            "Cyberpunk",
        ):
            self.assertIn(mode, corrector.PROMPT_MODES)

        self.assertGreaterEqual(len(corrector.PROMPT_PRESETS), 60)
        self.assertEqual(
            set(corrector.PROMPT_PRESETS),
            set(corrector.PROMPT_PRESET_GUIDANCE),
        )
        self.assertTrue(
            all(corrector.PROMPT_PRESET_GUIDANCE[preset].strip() for preset in corrector.PROMPT_PRESETS)
        )
        for preset in (
            "Group portrait",
            "Jewelry and watch",
            "Architecture exterior",
            "Creature design",
            "Comic page",
            "Scientific illustration",
            "Game asset",
        ):
            self.assertIn(preset, corrector.PROMPT_PRESETS)

        system_prompt = corrector.build_system_prompt(
            prompt_preset="Jewelry and watch",
        )
        self.assertIn("gemstone or metal behavior", system_prompt)

        for preset, guidance in corrector.PROMPT_PRESET_GUIDANCE.items():
            compact_prompt = corrector.build_small_model_system_prompt(
                generator_target="Krea 2",
                content_format="Single Image",
                output_length="Balanced",
                output_min_words=None,
                output_max_words=None,
                risk_level="Balanced improvement",
                prompt_preset=preset,
                variation_count=1,
                enhance_actions=False,
                develop_story=False,
            )
            self.assertIn(f"Prompt preset: {preset}.", compact_prompt)
            self.assertIn(f"Preset guidance: {guidance}", compact_prompt)

        unknown_compact_prompt = corrector.build_small_model_system_prompt(
            generator_target="Krea 2",
            content_format="Single Image",
            output_length="Balanced",
            output_min_words=None,
            output_max_words=None,
            risk_level="Balanced improvement",
            prompt_preset="Unknown preset",
            variation_count=1,
            enhance_actions=False,
            develop_story=False,
        )
        self.assertIn(
            f"Preset guidance: {corrector.PROMPT_PRESET_GUIDANCE['Auto']}",
            unknown_compact_prompt,
        )

    def test_expanded_fixed_comic_layouts_enforce_compatible_panel_counts(self):
        self.assertEqual(
            corrector.resolve_comic_layout("3 x 3 grid", 9),
            "3 x 3 grid with three columns and three rows",
        )
        self.assertEqual(
            corrector.resolve_comic_layout("3 x 3 grid", 8),
            corrector.automatic_comic_layout(8),
        )
        self.assertIn(
            "exactly 6 clearly separated panel regions",
            corrector.resolve_comic_layout("Asymmetric cinematic panels", 6),
        )

    def test_requested_medium_is_a_hard_contract_in_every_profile(self):
        source = "A photograph of a red ceramic cup on a white table."
        issues = corrector.requested_medium_issues(
            "A digital painting of a red ceramic cup on a white table.",
            source,
        )
        self.assertTrue(
            any(issue.startswith("Requested medium missing or changed") for issue in issues)
        )
        self.assertTrue(all(corrector.is_hard_compliance_issue(issue) for issue in issues))
        self.assertEqual(
            corrector.requested_medium_issues(
                "A close-up photograph of a red ceramic cup on a white table.",
                source,
            ),
            [],
        )

    def test_open_palms_conflict_with_two_occupied_hands(self):
        conflict = (
            "A woman grips the base with both hands while keeping open palms "
            "and relaxed shoulders."
        )
        issues = corrector.hand_use_contradiction_issues(conflict)
        self.assertEqual(len(issues), 1)
        self.assertTrue(corrector.is_hard_compliance_issue(issues[0]))
        self.assertEqual(
            corrector.hand_use_contradiction_issues(
                "A woman grips the base with both hands, with relaxed shoulders "
                "and an open posture."
            ),
            [],
        )

    def test_medium_detection_does_not_treat_an_action_as_an_art_medium(self):
        self.assertEqual(
            corrector.requested_medium_families(
                "A samurai drawing a sword beneath a cedar tree."
            ),
            set(),
        )

    def test_krea_official_rejects_only_high_confidence_main_additions(self):
        source = "A rainy alley at night with wet pavement and blue reflections."
        hard_additions = corrector.krea_official_addition_issues(
            "A rainy alley at night where a dog stands beside a motorcycle.",
            source,
        )
        self.assertTrue(any("animal subject" in issue for issue in hard_additions))
        self.assertTrue(any("vehicle" in issue for issue in hard_additions))
        self.assertTrue(
            all(
                corrector.is_hard_compliance_issue(issue)
                for issue in hard_additions
            )
        )

        decorative = corrector.krea_official_addition_issues(
            "A rainy alley at night with wet pavement, blue reflections, and lanterns.",
            source,
        )
        self.assertEqual(len(decorative), 1)
        self.assertIn("advisory", decorative[0])
        self.assertFalse(corrector.is_hard_compliance_issue(decorative[0]))

    def test_krea_official_allows_implied_or_explicitly_authorized_objects(self):
        self.assertEqual(
            corrector.krea_official_addition_issues(
                "A samurai grips a sword beneath a cedar tree.",
                "A samurai beneath a cedar tree.",
            ),
            [],
        )
        self.assertEqual(
            corrector.krea_official_addition_issues(
                "A rainy alley with a dog under blue reflections.",
                "A rainy alley. Required concept: dog.",
            ),
            [],
        )

    def test_krea_official_lightly_polishes_already_detailed_prompts(self):
        source = (
            "A cinematic photograph of a matte black designer toy standing at the "
            "center of a brushed steel table, seen in a low-angle medium shot. Soft "
            "window light enters from the left, forming a narrow rim light around its "
            "rounded silhouette. The background remains charcoal gray with shallow "
            "depth of field, subtle reflections, restrained contrast, and a precise "
            "monochrome palette. Fine vinyl texture is visible across the face and arms."
        )
        self.assertTrue(corrector.prompt_is_already_detailed(source))
        expanded = source + " " + " ".join(
            [
                "Additional",
                "atmospheric",
                "cinematic",
                "luxurious",
                "dramatic",
                "polished",
                "editorial",
                "gallery",
                "studio",
                "premium",
                "expressive",
                "layered",
                "intricate",
                "ornamental",
                "glowing",
                "volumetric",
                "ethereal",
                "majestic",
                "surreal",
                "dynamic",
                "immersive",
                "spectacular",
                "elaborate",
                "vivid",
                "textured",
            ]
        ) + "."
        issues = corrector.krea_official_compliance_issues(
            expanded,
            original_prompt=source,
            source_context=source,
        )
        self.assertTrue(
            any(issue.startswith("Krea Official detailed-input contract") for issue in issues)
        )

    def test_krea_guideline_status_labels_compliance_extensions_and_exception(self):
        common = {
            "generator_target": "Krea 2",
            "content_format": "Single Image",
            "variation_count": 1,
            "risk_level": "Strict cleanup",
            "preserve_strictly": True,
            "enhance_actions": False,
            "develop_story": False,
            "artistic_detail_freedom": False,
            "safe_for_work": True,
            "explicit_nsfw": False,
        }
        self.assertIn(
            "Krea Official compliant",
            corrector.krea_guideline_status(
                workflow_profile="Krea Official", **common
            ),
        )
        self.assertIn(
            "creative extension",
            corrector.krea_guideline_status(
                workflow_profile="Explore", **common
            ),
        )
        explicit = dict(common)
        explicit["safe_for_work"] = False
        explicit["explicit_nsfw"] = True
        self.assertIn(
            "Explicit-mode exception",
            corrector.krea_guideline_status(
                workflow_profile="Krea Official", **explicit
            ),
        )

    def test_krea_official_contract_reaches_the_model(self):
        source = (
            "A watercolor painting of a red lighthouse above a gray sea, with "
            "soft morning light and a centered composition."
        )
        with patch.object(
            corrector,
            "chat_completion",
            return_value=source,
        ) as mocked_completion:
            result = corrector.post_chat_completion(
                base_url="http://127.0.0.1:1234/v1",
                model="large-test-model",
                prompt=source,
                generator_target="Krea 2",
                content_format="Single Image",
                temperature=0.1,
                max_tokens=512,
                timeout=10,
                api_key="lm-studio",
                risk_level="Strict cleanup",
                preserve_strictly=True,
                develop_story=False,
                safe_for_work=True,
                audit_repair=False,
                final_gate_repair=False,
                krea_official=True,
            )

        sent_messages = mocked_completion.call_args.kwargs["messages"]
        sent_text = "\n".join(message["content"] for message in sent_messages)
        self.assertIn("Krea Official expansion contract", sent_text)
        self.assertIn("Do not add a new subject, object, prop", sent_text)
        self.assertIn("watercolor painting", result)


if __name__ == "__main__":
    unittest.main()
