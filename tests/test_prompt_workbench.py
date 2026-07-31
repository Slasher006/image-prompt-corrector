import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import prompt_workbench as workbench


class PromptWorkbenchTests(unittest.TestCase):
    def test_new_project_has_portable_workflow_collections(self):
        project = workbench.new_project("Demo")
        self.assertEqual(project["name"], "Demo")
        for key in ("references", "results", "reviews", "versions", "characters", "composition", "text_layout"):
            self.assertEqual(project[key], [])

    def test_reference_roles_include_crop_mask_panel_and_subject(self):
        project = workbench.new_project()
        reference = workbench.add_reference(
            project,
            "/tmp/face.png",
            role="Face",
            panel=2,
            subject="Mara",
            crop=(0.1, 0.2, 0.3, 0.4),
            mask_path="/tmp/mask.png",
            notes="Ignore the background",
        )
        self.assertEqual(reference["role"], "Face")
        self.assertEqual(reference["panel"], 2)
        self.assertEqual(reference["crop"], [0.1, 0.2, 0.3, 0.4])
        self.assertIn("Panel 2", workbench.reference_instruction(project["references"]))

    def test_project_bundle_round_trip_copies_assets(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / "result.png"
            image.write_bytes(b"fake image")
            project = workbench.new_project("Round trip")
            workbench.add_reference(project, str(image), role="Style")
            project["results"].append({"path": str(image), "id": "r1"})
            bundle = workbench.save_project_bundle(project, root / "demo")
            loaded = workbench.load_project_bundle(bundle, root / "extracted")
            self.assertEqual(loaded["name"], "Round trip")
            self.assertTrue(Path(loaded["references"][0]["path"]).is_file())
            self.assertTrue(Path(loaded["results"][0]["path"]).is_file())

    def test_project_bundle_rejects_missing_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "broken.ipcp"
            import zipfile
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("other.txt", "x")
            with self.assertRaisesRegex(ValueError, "project.json"):
                workbench.load_project_bundle(path)

    def test_contract_dashboard_groups_failures(self):
        rows = workbench.contract_dashboard("", original_prompt="two red cubes")
        self.assertTrue(any(row["status"] == "fail" for row in rows))
        self.assertTrue(any(row["category"] == "Structure" for row in rows))

    def test_contract_dashboard_can_enforce_safe_for_work(self):
        rows = workbench.contract_dashboard(
            "an erotic nude portrait",
            original_prompt="a portrait",
            safe_for_work=True,
        )
        self.assertTrue(any("Safe-for-work" in row["detail"] for row in rows))

    def test_clarification_questions_only_raise_material_gaps(self):
        questions = workbench.clarification_questions('two people holding a sign')
        self.assertTrue(any("character" in question.lower() for question in questions))
        self.assertTrue(any("exact text" in question.lower() for question in questions))
        self.assertEqual(workbench.clarification_questions("front-facing knight at center, subject-left hand holding a shield, cinematic dusk"), [])

    def test_result_review_messages_include_images_and_reference_roles(self):
        messages = workbench.build_result_review_messages(
            original_prompt="cat",
            corrected_prompt="a black cat",
            reference_instruction_text="Use as identity reference.",
            image_data_urls=["data:image/png;base64,AAAA"],
        )
        content = messages[1]["content"]
        self.assertEqual(content[-1]["type"], "image_url")
        self.assertIn("Reference roles", content[0]["text"])

    def test_result_review_can_evaluate_explicit_adult_mode_without_censoring(self):
        messages = workbench.build_result_review_messages(
            original_prompt="Two adult women in an explicit nude portrait.",
            corrected_prompt="Two clearly adult women pose nude in a private studio.",
            image_data_urls=["data:image/png;base64,AAAA"],
            explicit_nsfw=True,
        )
        review_text = messages[1]["content"][0]["text"]
        self.assertIn("evaluate requested adult content directly", review_text)
        self.assertIn("age 18 or older", review_text)
        self.assertIn("NSFW visual fidelity audit", review_text)
        self.assertIn("act families", review_text)
        self.assertIn("contact targets", review_text)
        self.assertIn("object/body separation", review_text)
        self.assertIn("body_ownership", messages[0]["content"])
        self.assertIn("anatomical_attachment", messages[0]["content"])
        self.assertIn("anatomical_orientation", messages[0]["content"])

    def test_review_response_parses_json_and_targeted_repair(self):
        parsed = workbench.parse_review_response(
            'Result: {"score": 73, "summary":"Mostly right", "passed":["coat"], "failed":["hand"], "warnings":[], "repair_prompt":"Fix only the hand."}'
        )
        self.assertEqual(parsed["score"], 73)
        self.assertEqual(workbench.targeted_repair_prompt(parsed, "prompt"), "Fix only the hand.")

    def test_review_parses_nsfw_fidelity_and_builds_targeted_feedback(self):
        parsed = workbench.parse_review_response(
            '{"score":61,"summary":"Wrong contact","passed":[],"failed":[],'
            '"warnings":[],"repair_prompt":"","nsfw_fidelity":{'
            '"participant_count":"pass","action_roles":"fail","body_ownership":"fail",'
            '"anatomical_attachment":"fail","anatomical_orientation":"fail",'
            '"contact_targets":"fail",'
            '"object_separation":"fail","visible_phase":"pass","reactions":"fail",'
            '"discrepancies":["receiver role is reversed","toy merged with anatomy"]}}'
        )

        self.assertEqual(parsed["nsfw_fidelity"]["action_roles"], "fail")
        self.assertEqual(parsed["nsfw_fidelity"]["body_ownership"], "fail")
        self.assertEqual(parsed["nsfw_fidelity"]["anatomical_attachment"], "fail")
        self.assertEqual(parsed["nsfw_fidelity"]["anatomical_orientation"], "fail")
        self.assertEqual(parsed["nsfw_fidelity"]["object_separation"], "fail")
        repair = workbench.targeted_repair_prompt(parsed, "Current prompt.")
        self.assertIn("NSFW fidelity: failed body ownership", repair)
        self.assertIn("NSFW fidelity: failed anatomical attachment", repair)
        self.assertIn("NSFW fidelity: failed anatomical orientation", repair)
        self.assertIn("NSFW fidelity: receiver role is reversed", repair)
        self.assertIn("NSFW fidelity: toy merged with anatomy", repair)
        self.assertIn("Current prompt.", repair)

    def test_invalid_review_response_fails_visibly(self):
        parsed = workbench.parse_review_response("not json")
        self.assertEqual(parsed["score"], 0)
        self.assertIn("not valid JSON", parsed["failed"][0])

    def test_character_continuity_reports_missing_anchor_per_panel(self):
        characters = [{"name": "Mara", "anchors": ["red coat", "brass satchel"]}]
        issues = workbench.character_continuity_issues(characters, ["Mara wears a red coat", "Mara with red coat and brass satchel"])
        self.assertEqual(len(issues), 1)
        self.assertIn("Panel 1", issues[0])
        self.assertIn("brass satchel", issues[0])

    def test_composition_and_text_layout_become_explicit_contracts(self):
        composition = workbench.composition_instruction([
            {"kind": "subject", "label": "knight", "x": 0.1, "y": 0.2, "width": 0.3, "height": 0.6, "panel": 1}
        ])
        lettering = workbench.text_layout_instruction([
            {"kind": "speech bubble", "speaker": "Mara", "text": "RUN!", "panel": 2, "placement": "top right"}
        ])
        self.assertIn("normalized box", composition)
        self.assertIn("Panel 1", composition)
        self.assertIn('exactly "RUN!"', lettering)
        self.assertIn("Panel 2", lettering)

    def test_batch_csv_skips_empty_prompts_and_is_resumable(self):
        rows = workbench.parse_batch_csv("id,prompt,goal,focus\n1,red cube,test,cube\n2,,,\n")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "pending")

    def test_ab_variants_lock_hard_contracts(self):
        variants = workbench.prompt_variant_instructions("two cats, no text")
        self.assertEqual(len(variants), 4)
        self.assertTrue(all("counts" in item["instruction"] for item in variants))
        self.assertEqual({item["label"] for item in variants}, {"Faithful", "Composition", "Camera", "Atmosphere"})

    def test_diagnostic_trace_records_stage_duration(self):
        trace = workbench.DiagnosticTrace()
        trace.start("vision")
        trace.finish("ok", "two images")
        self.assertEqual(trace.stages[0]["name"], "vision")
        self.assertGreaterEqual(trace.total_seconds(), 0)

    def test_generator_profiles_merge_custom_profiles(self):
        profiles = workbench.load_generator_profiles({"My model": {"negative_prompt": True}})
        self.assertIn("Krea 2", profiles)
        self.assertTrue(profiles["My model"]["negative_prompt"])

    def test_flux_fixed_seed_benchmark_crosses_variant_and_encoder(self):
        manifest = workbench.build_flux_fixed_seed_benchmark(
            '{"scene":"two people","subjects":[]}',
            seed=31415,
        )

        self.assertEqual(len(manifest["cases"]), 4)
        self.assertEqual(
            {case["seed"] for case in manifest["cases"]},
            {31415},
        )
        self.assertEqual(
            {
                (
                    case["setup"]["variant"],
                    case["setup"]["text_encoder"],
                )
                for case in manifest["cases"]
            },
            {
                ("Distilled (4-step)", "Official Qwen3 8B"),
                ("Distilled (4-step)", "Abliterated Qwen3 8B"),
                ("Base (50-step)", "Official Qwen3 8B"),
                ("Base (50-step)", "Abliterated Qwen3 8B"),
            },
        )
        self.assertTrue(
            all(case["prompt"] == manifest["cases"][0]["prompt"] for case in manifest["cases"])
        )

    @mock.patch("prompt_workbench.urllib.request.urlopen")
    def test_comfyui_handoff_injects_prompt_into_selected_node(self, urlopen):
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = b'{"prompt_id":"abc"}'
        urlopen.return_value = response
        workflow = {"6": {"inputs": {"text": "old"}, "class_type": "CLIPTextEncode"}}
        result = workbench.enqueue_comfyui(
            server_url="http://127.0.0.1:8188",
            workflow=workflow,
            prompt="new prompt",
            positive_node_id="6",
        )
        self.assertEqual(result["prompt_id"], "abc")
        request = urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["prompt"]["6"]["inputs"]["text"], "new prompt")

    def test_image_edit_prompt_separates_change_preservation_and_target(self):
        prompt = workbench.build_image_edit_prompt(
            instruction="change the coat to yellow",
            preserve="face, pose, and background",
            target_prompt="A woman in a yellow coat under soft daylight.",
            image_analysis="Image 2 visibly has a wool coat texture.",
            reference_roles=[
                "base composition",
                "use this person's face",
                "take the coat texture from this image",
            ],
            masked_reference_indices=[1, 3],
        )
        self.assertIn("Requested change: change the coat to yellow", prompt)
        self.assertIn("Preserve unchanged: face, pose, and background", prompt)
        self.assertIn("Target result: A woman in a yellow coat", prompt)
        self.assertIn("Image 1: base composition.", prompt)
        self.assertIn("Image 2: use this person's face.", prompt)
        self.assertIn("Image 3: take the coat texture from this image.", prompt)
        self.assertIn("Verified image observations", prompt)
        self.assertIn(
            "Edit only inside the supplied mask for Image 1, Image 3",
            prompt,
        )

    def test_image_edit_prompt_adds_requested_ventral_frenulum_orientation(self):
        prompt = workbench.build_image_edit_prompt(
            instruction=(
                "Correct the penis underside so the frenulum is visible to the camera."
            ),
            preserve="identity, framing, lighting, and everything outside the mask",
            reference_roles=["base composition", "anatomical orientation"],
            masked_reference_indices=[1],
        )

        self.assertIn("ventral underside facing the camera", prompt)
        self.assertIn("frenulum visibly centered", prompt)
        self.assertIn("dorsal surface facing away", prompt)

    def test_image_edit_prompt_does_not_add_anatomy_to_unrelated_edit(self):
        prompt = workbench.build_image_edit_prompt(
            instruction="Change the coat to yellow.",
            preserve="face, pose, and background",
        )

        self.assertNotIn("frenulum", prompt.lower())

    def test_image_edit_prompt_infers_underside_from_verified_viewpoint(self):
        prompt = workbench.build_image_edit_prompt(
            instruction="Correct the visible anatomical orientation.",
            preserve="identity and lighting",
            image_analysis=(
                "The clearly adult subject's visible penis is photographed in a "
                "low-angle view from below."
            ),
        )

        self.assertIn("ventral underside facing the camera", prompt)
        self.assertIn("frenulum visibly centered", prompt)

    @mock.patch("prompt_workbench.urllib.request.urlopen")
    def test_upload_comfyui_image_posts_multipart_and_returns_subfolder_name(self, urlopen):
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = (
            b'{"name":"source.png","subfolder":"promptcorrector","type":"input"}'
        )
        urlopen.return_value = response
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source.png"
            source.write_bytes(b"png-data")
            name = workbench.upload_comfyui_image(
                server_url="http://127.0.0.1:8188",
                path=source,
            )
        self.assertEqual(name, "promptcorrector/source.png")
        request = urlopen.call_args.args[0]
        self.assertTrue(request.full_url.endswith("/upload/image"))
        self.assertIn(b'filename="source.png"', request.data)

    def test_benchmark_scores_contract_responses(self):
        responses = iter([
            "red cube left of two blue spheres; no text.",
            'a sign reading "NORTH GATE"; no other lettering.',
            "front-facing knight, shield in subject-left hand, sword in subject-right hand.",
            "Panel 1: cat finds key. Panel 2: same cat opens door.",
        ])
        with mock.patch("prompt_workbench.chat_completion", side_effect=lambda **_kwargs: next(responses)):
            result = workbench.benchmark_model(base_url="http://localhost:1234/v1", model="test")
        self.assertEqual(result["passed"], 4)
        self.assertEqual(result["score"], 100)


if __name__ == "__main__":
    unittest.main()
