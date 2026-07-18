import os
import inspect
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import krea_prompt_gui as gui


class PromptCorrectorGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = gui.QApplication.instance() or gui.QApplication([])

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_settings_path = gui.SETTINGS_PATH
        gui.SETTINGS_PATH = Path(self.temp_dir.name) / "settings.json"
        self.root = gui.PromptCorrectorWindow()
        self.controller = gui.PromptCorrectorApp(self.root)
        self.root.controller = self.controller
        self.root.show()
        self.application.processEvents()

    def tearDown(self):
        self.root.close()
        gui.SETTINGS_PATH = self.original_settings_path
        self.temp_dir.cleanup()

    def test_qol_menus_and_shortcuts_are_available(self):
        menu_bar = self.root.menuBar()
        menu_actions = menu_bar.actions()
        menus = {action.text(): action.menu() for action in menu_actions}
        self.assertIn("File", menus)
        self.assertIn("Edit", menus)
        self.assertIn("Prompt", menus)
        self.assertIn("Chat", menus)

        shortcuts = {
            action.text(): action.shortcut().toString()
            for menu in menus.values()
            for action in menu.actions()
            if not action.isSeparator()
        }
        self.assertEqual(shortcuts["Exit"], "Ctrl+Q")
        self.assertEqual(shortcuts["Correct prompt"], "Ctrl+Return")
        self.assertEqual(shortcuts["Copy corrected prompt"], "Ctrl+Shift+C")
        self.assertEqual(shortcuts["Iterate corrected prompt"], "Ctrl+Shift+R")

    def test_iterate_result_promotes_output_and_applies_optional_feedback(self):
        result = "A red knight stands at an ancient castle gate under cold moonlight."
        self.controller.corrected_text.setPlainText(result)
        self.application.processEvents()
        self.assertTrue(self.controller.iterate_button.isEnabled())

        with mock.patch.object(
            gui.QInputDialog,
            "getMultiLineText",
            return_value=("Use warmer torchlight and a lower camera.", True),
        ):
            with mock.patch.object(self.controller, "correct_prompt") as correct_prompt:
                self.controller.iterate_corrected_prompt()

        self.assertEqual(self.controller.draft_text.toPlainText(), result)
        self.assertEqual(
            self.controller.generation_feedback_var.get(),
            "Use warmer torchlight and a lower camera.",
        )
        correct_prompt.assert_called_once_with()

    def test_iteration_feedback_stays_separate_from_persistent_model_instructions(self):
        self.controller.draft_text.setPlainText("A red knight at a castle gate.")
        self.controller.model_instructions_var.set("Keep the knight's red cloak.")
        self.controller.generation_feedback_var.set(
            "Use warmer torchlight and a lower camera."
        )
        self.controller.reference_images_var.set(False)
        self.controller.live_research_var.set(False)
        self.controller.audit_repair_var.set(False)

        with mock.patch("krea_prompt_gui.threading.Thread") as thread_class:
            self.controller.correct_prompt()
        worker_args = thread_class.call_args.kwargs["args"]

        with mock.patch(
            "krea_prompt_gui.post_chat_completion",
            return_value="A corrected knight prompt.",
        ) as completion:
            self.controller._correct_prompt_worker(*worker_args)

        sent = completion.call_args.kwargs
        self.assertEqual(sent["model_instructions"], "Keep the knight's red cloak.")
        self.assertEqual(sent["private_model_instructions"], "")
        self.assertEqual(
            sent["generation_feedback"],
            "Use warmer torchlight and a lower camera.",
        )
        self.assertNotIn("Generation feedback", sent["model_instructions"])

    def test_context_tokens_default_dropdown_and_legacy_migration(self):
        self.assertEqual(
            self.controller.context_token_budget_var.get(),
            gui.CONTEXT_TOKEN_AUTO_LABEL,
        )
        self.assertEqual(self.controller._context_token_budget(), gui.CONTEXT_TOKEN_AUTO)
        self.controller.context_token_budget_var.set("4K")
        self.assertEqual(self.controller._context_token_budget(), 4096)
        self.assertEqual(
            self.controller._context_token_setting(
                {"context_line_budget": 120},
                gui.CONTEXT_TOKEN_AUTO_LABEL,
            ),
            gui.CONTEXT_TOKEN_AUTO_LABEL,
        )
        self.assertEqual(
            self.controller._context_token_setting(
                {"context_token_budget": 64_000},
                gui.CONTEXT_TOKEN_AUTO_LABEL,
            ),
            "64K",
        )
        self.controller.context_token_budget_var.set(gui.CONTEXT_TOKEN_AUTO_LABEL)
        snapshot = self.controller._settings_snapshot()
        self.assertEqual(snapshot["context_token_budget"], "auto")
        self.assertNotIn("context_line_budget", snapshot)

    def test_model_refresh_replaces_cached_entries_and_drops_missing_selection(self):
        self.controller.available_models = ["deleted-model", "old-embedding"]
        self.controller.model_var.set("deleted-model")

        with mock.patch.object(self.controller, "_save_settings") as save_settings:
            self.controller._update_available_models(
                ["qwen3-vl-4b-instruct", "qwen3-vl-4b-instruct"]
            )

        self.assertEqual(
            self.controller.available_models,
            ["qwen3-vl-4b-instruct"],
        )
        self.assertEqual(self.controller.model_var.get(), "qwen3-vl-4b-instruct")
        save_settings.assert_called_once_with()

    def test_counters_diff_and_draft_recovery(self):
        self.controller.draft_text.setPlainText("red knight at a castle gate")
        self.controller.corrected_text.setPlainText(
            "A red knight stands at an ancient castle gate."
        )
        self.application.processEvents()

        self.assertIn("6 words", self.controller.draft_counter_label.text())
        self.assertIn("9 words", self.controller.corrected_counter_label.text())
        self.assertNotIn("target", self.controller.corrected_counter_label.text())
        self.assertIn("background", self.controller.diff_text.toHtml())

        snapshot = self.controller._settings_snapshot()
        self.assertNotIn("output_min_words", snapshot)
        self.assertNotIn("output_max_words", snapshot)

        self.controller._save_settings()
        second_root = gui.PromptCorrectorWindow()
        second_controller = gui.PromptCorrectorApp(second_root)
        second_root.controller = second_controller
        self.assertEqual(
            second_controller.draft_text.toPlainText(),
            "red knight at a castle gate",
        )
        self.assertEqual(
            second_controller.corrected_text.toPlainText(),
            "A red knight stands at an ancient castle gate.",
        )
        second_root.close()

    def test_custom_presets_and_searchable_pinned_history(self):
        self.controller.focus_var.set("sharp armor")
        self.controller.concept_mix_var.set("watercolor:65%, cyberpunk:35%")
        concept_key = gui.concept_preset_key(
            "Character archetypes",
            "reluctant hero",
        )
        emotion_key = gui.narrative_preset_key(
            "emotion",
            "Confidence courage and pride",
            "quiet confidence with steady eye contact",
        )
        self.controller.concept_preset_selections["prompt"] = [concept_key]
        self.controller.narrative_preset_selections["prompt"]["emotion"] = [
            emotion_key
        ]
        self.assertTrue(self.controller._store_custom_preset("Armor"))
        self.controller.focus_var.set("changed")
        self.controller.concept_mix_var.set("")
        self.controller.concept_preset_selections["prompt"] = []
        self.controller.narrative_preset_selections["prompt"]["emotion"] = []
        self.assertTrue(self.controller._apply_custom_preset("Armor"))
        self.assertEqual(self.controller.focus_var.get(), "sharp armor")
        self.assertEqual(self.controller.concept_mix_var.get(), "watercolor:65%, cyberpunk:35%")
        self.assertEqual(
            self.controller.concept_preset_selections["prompt"],
            [concept_key],
        )
        self.assertEqual(
            self.controller.narrative_preset_selections["prompt"]["emotion"],
            [emotion_key],
        )

        self.controller.prompt_history = [
            {
                "title": "Castle",
                "requested_prompt": "knight",
                "corrected_prompt": "knight at gate",
                "pinned": False,
            },
            {
                "title": "Forest",
                "requested_prompt": "elf",
                "corrected_prompt": "elf in woods",
                "pinned": True,
            },
        ]
        self.controller._refresh_history_listbox()
        self.assertTrue(self.controller.history_listbox.item(0).text().startswith("★"))

        self.controller.history_search_entry.setText("castle")
        self.application.processEvents()
        self.assertEqual(self.controller.history_listbox.count(), 1)
        self.assertEqual(self.controller._selected_history_entry(), None)
        self.controller.history_listbox.setCurrentRow(0)
        self.assertEqual(self.controller._selected_history_entry()["title"], "Castle")

    def test_mix_editor_has_exhaustive_searchable_ingredient_library(self):
        chosen_suffixes = {
            "Watercolor",
            "courier",
            "floating village",
            "nostalgic and bittersweet",
            "soft window light",
            "brushed metal texture",
        }

        def choose_ingredients():
            dialog = next(
                widget
                for widget in self.application.topLevelWidgets()
                if (
                    isinstance(widget, gui.QDialog)
                    and widget.windowTitle() == "Mixer ingredient library"
                    and widget.parent() is self.root
                )
            )
            preset_list = dialog.findChild(gui.QListWidget)
            for index in range(preset_list.count()):
                item = preset_list.item(index)
                if any(item.text().endswith(value) for value in chosen_suffixes):
                    item.setCheckState(gui.Qt.CheckState.Checked)
            return gui.QDialog.DialogCode.Accepted

        with mock.patch.object(
            gui.QDialog,
            "exec",
            side_effect=choose_ingredients,
        ):
            result = self.controller._open_mix_ingredient_picker([])

        self.assertIsNotNone(result)
        names, replace = result
        self.assertEqual(set(names), chosen_suffixes)
        self.assertEqual(len(names), gui.MIX_INGREDIENT_LIMIT)
        self.assertFalse(replace)

    def test_mix_library_adds_rows_and_balances_percentages(self):
        self.controller.concept_mix_var.set("custom user style:70%, Watercolor:30%")
        library_names = ["Watercolor", "courier", "soft window light"]

        def apply_library_and_save():
            dialog = next(
                widget
                for widget in self.application.topLevelWidgets()
                if (
                    isinstance(widget, gui.QDialog)
                    and widget.windowTitle() == "Concept and style mix"
                    and widget.parent() is self.root
                )
            )
            library_button = dialog.findChild(
                gui.QPushButton,
                "mixLibraryButton",
            )
            library_button.click()
            names = [
                dialog.findChild(gui.QLineEdit, f"mixIngredient{index}").text()
                for index in range(1, 7)
            ]
            shares = [
                dialog.findChild(gui.QSpinBox, f"mixShare{index}").value()
                for index in range(1, 7)
            ]
            self.assertEqual(
                names[:4],
                ["custom user style", "Watercolor", "courier", "soft window light"],
            )
            self.assertEqual(sum(shares[:4]), 100)
            return gui.QDialog.DialogCode.Accepted

        with mock.patch.object(
            self.controller,
            "_open_mix_ingredient_picker",
            return_value=(library_names, False),
        ):
            with mock.patch.object(
                gui.QDialog,
                "exec",
                side_effect=apply_library_and_save,
            ):
                self.controller._open_concept_mix_editor()

        parsed = gui.parse_concept_mix(self.controller.concept_mix_var.get())
        self.assertEqual(
            [name for name, _share in parsed],
            ["custom user style", "Watercolor", "courier", "soft window light"],
        )
        self.assertEqual(sum(share for _name, share in parsed), 100)

    def test_local_reference_image_gets_thumbnail_and_candidate(self):
        image_path = Path(self.temp_dir.name) / "reference.png"
        pixmap = gui.QPixmap(12, 12)
        pixmap.fill(gui.QColor("#ff0000"))
        self.assertTrue(pixmap.save(str(image_path)))

        self.controller.add_local_reference_paths([str(image_path)])

        self.assertTrue(self.controller.reference_images_var.get())
        self.assertEqual(self.controller.reference_preview_list.count(), 1)
        candidate = self.controller._local_reference_candidates()[0]
        self.assertEqual(candidate["title"], "reference.png")
        self.assertTrue(candidate["url"].startswith("file:"))

    def test_local_references_skip_automatic_web_image_lookup(self):
        local = [
            {
                "title": "chosen.png",
                "url": "file:///tmp/chosen.png",
                "summary": "User-provided local reference",
            }
        ]
        with mock.patch("krea_prompt_gui.collect_reference_image_diagnostics") as collect:
            candidates, diagnostics = self.controller._collect_reference_images_for_prompt(
                "a knight",
                "Auto (safe sources)",
                local,
            )

        collect.assert_not_called()
        self.assertEqual(candidates, local)
        self.assertIn("Automatic web image lookup skipped", diagnostics[1])

    def test_stop_immediately_invalidates_worker_and_allows_restart(self):
        self.controller.active_request_id = 7
        self.controller.request_in_progress = True
        self.controller.cancel_event.clear()

        self.controller.stop_current_request()

        self.assertEqual(self.controller.active_request_id, 8)
        self.assertTrue(self.controller.cancel_event.is_set())
        self.assertFalse(self.controller.request_in_progress)
        self.assertEqual(
            self.controller.status_var.get(),
            "Stopped - ready for a new request",
        )

        # A replacement correction can start immediately.
        self.controller.draft_text.setPlainText("a knight at a gate")
        with mock.patch("krea_prompt_gui.threading.Thread") as thread_class:
            self.controller.correct_prompt()
        thread_class.return_value.start.assert_called_once()
        self.assertEqual(self.controller.active_request_id, 9)
        self.assertFalse(self.controller.cancel_event.is_set())
        self.assertTrue(self.controller.request_in_progress)

        # The cancelled worker may report back later, but it must not change the
        # state of the replacement request.
        self.controller._show_cancelled(7)
        self.assertTrue(self.controller.request_in_progress)
        self.assertEqual(self.controller.active_request_id, 9)

    def test_ui_groups_prompt_workflow_and_direct_model_chat(self):
        self.assertEqual(self.controller.mode_tabs.count(), 5)
        self.assertEqual(self.controller.mode_tabs.tabText(0), "Prompt Corrector")
        self.assertEqual(self.controller.mode_tabs.tabText(1), "Comic Story")
        self.assertEqual(self.controller.mode_tabs.tabText(2), "Meme Creator")
        self.assertEqual(self.controller.mode_tabs.tabText(3), "Model Chat")
        self.assertEqual(self.controller.mode_tabs.tabText(4), "Workbench")
        self.assertIsNotNone(self.controller.workbench_widget)
        self.assertEqual(self.controller.setup_tabs.count(), 4)
        self.assertEqual(self.controller.setup_tabs.tabText(0), "Generation")
        self.assertEqual(self.controller.setup_tabs.tabText(2), "Processing")
        self.assertEqual(self.controller.setup_tabs.tabText(3), "Connection")
        self.assertIsNotNone(self.controller.chat_transcript)
        self.assertIsNotNone(self.controller.chat_input)
        self.assertEqual(self.controller.workflow_profile_var.get(), "Exact")
        self.assertFalse(self.controller.setup_tabs.isVisible())
        self.assertFalse(self.controller.prompt_guidance_page.isVisible())
        self.controller.prompt_options_button.setChecked(True)
        self.application.processEvents()
        self.assertTrue(self.controller.prompt_guidance_page.isVisible())
        self.controller.mode_tabs.setCurrentIndex(2)
        self.application.processEvents()
        self.assertFalse(self.controller.prompt_guidance_page.isVisible())
        settings_button = next(
            button
            for button in self.root.findChildren(gui.QPushButton)
            if button.text() == "Settings"
        )
        settings_button.setChecked(True)
        self.application.processEvents()
        self.assertTrue(self.controller.setup_tabs.isVisible())
        self.assertFalse(self.controller.develop_story_var.get())
        self.assertFalse(self.controller._settings_snapshot()["develop_story"])

    def test_workflow_profiles_apply_safe_coherent_defaults(self):
        self.assertEqual(self.controller.risk_level_var.get(), "Strict cleanup")
        self.assertTrue(self.controller.preserve_var.get())
        self.assertEqual(self.controller.creativity_var.get(), "raw")
        self.assertFalse(self.controller.live_research_var.get())
        self.assertEqual(
            self.controller.reference_image_source_var.get(), "Auto (safe sources)"
        )

        self.controller.workflow_profile_var.set("Improve")
        self.assertEqual(self.controller.risk_level_var.get(), "Balanced improvement")
        self.assertFalse(self.controller.preserve_var.get())
        self.assertEqual(self.controller.creativity_var.get(), "low")
        self.assertFalse(self.controller.develop_story_var.get())
        self.assertFalse(self.controller.artistic_detail_freedom_var.get())

        self.controller.workflow_profile_var.set("Explore")
        self.assertEqual(self.controller.risk_level_var.get(), "Creative enhancement")
        self.assertTrue(self.controller.develop_story_var.get())
        self.assertTrue(self.controller.artistic_detail_freedom_var.get())
        self.assertEqual(self.controller.creativity_var.get(), "medium")

    def test_fixed_seed_is_optional_persisted_and_bound_to_meme_worker(self):
        self.assertFalse(self.controller.fixed_seed_var.get())
        self.assertEqual(self.controller._sampling_seed(), None)
        self.assertFalse(self.controller.seed_spin.isEnabled())

        self.controller.fixed_seed_var.set(True)
        self.controller.seed_var.set(31415)
        self.application.processEvents()

        self.assertTrue(self.controller.seed_spin.isEnabled())
        self.assertEqual(self.controller._sampling_seed(), 31415)
        snapshot = self.controller._settings_snapshot()
        self.assertTrue(snapshot["fixed_seed"])
        self.assertEqual(snapshot["seed"], 31415)

        self.controller.mode_tabs.setCurrentIndex(2)
        self.controller.meme_scene_var.set("a cat staring at an overflowing inbox")
        self.controller.meme_top_text_var.set("ONE QUICK EMAIL")
        with mock.patch("krea_prompt_gui.threading.Thread") as thread_class:
            self.controller.correct_meme()

        worker_args = thread_class.call_args.kwargs["args"]
        bound = inspect.signature(self.controller._correct_prompt_worker).bind(*worker_args)
        self.assertEqual(bound.arguments["content_format"], "Meme")
        self.assertEqual(bound.arguments["seed"], 31415)

    def test_safe_for_work_option_is_visible_persistent_and_worker_bound(self):
        self.controller.safe_for_work_var.set(True)
        snapshot = self.controller._settings_snapshot()
        self.assertTrue(snapshot["safe_for_work"])
        self.assertTrue(self.controller._prompt_option_snapshot()["safe_for_work"])
        labels = [checkbox.text() for checkbox in self.root.findChildren(gui.QCheckBox)]
        self.assertIn("Safe for work", labels)

    def test_explicit_adult_mode_is_persistent_mutually_exclusive_and_worker_bound(self):
        self.controller.explicit_nsfw_var.set(True)
        self.assertFalse(self.controller.safe_for_work_var.get())
        snapshot = self.controller._settings_snapshot()
        self.assertTrue(snapshot["explicit_nsfw"])
        self.assertTrue(self.controller._prompt_option_snapshot()["explicit_nsfw"])
        labels = [checkbox.text() for checkbox in self.root.findChildren(gui.QCheckBox)]
        self.assertIn("Explicit adult (NSFW)", labels)

        self.controller.safe_for_work_var.set(True)
        self.assertFalse(self.controller.explicit_nsfw_var.get())
        self.controller.explicit_nsfw_var.set(True)
        self.assertFalse(self.controller.safe_for_work_var.get())

        self.controller.draft_text.setPlainText("Two adults in a private portrait.")
        with mock.patch("krea_prompt_gui.threading.Thread") as thread_class:
            self.controller.correct_prompt()
        worker_args = thread_class.call_args.kwargs["args"]
        bound = inspect.signature(self.controller._correct_prompt_worker).bind(*worker_args)
        self.assertTrue(bound.arguments["explicit_nsfw"])
        self.assertFalse(bound.arguments["safe_for_work"])

    def test_generator_target_switches_krea_and_flux_guidance(self):
        self.assertEqual(self.controller.generator_target_var.get(), "Krea 2")
        self.assertTrue(self.controller.generator_controls_page.isEnabled())
        self.assertIn("creativity=raw", self.controller._krea_recommendation_text())

        self.controller.generator_target_var.set("FLUX.2 Klein 9B")
        self.application.processEvents()

        self.assertFalse(self.controller.generator_controls_page.isEnabled())
        self.assertEqual(
            self.controller.setup_tabs.tabText(
                self.controller.generator_controls_tab_index
            ),
            "FLUX setup (fixed)",
        )
        recommendation = self.controller._krea_recommendation_text()
        self.assertIn("4 inference steps", recommendation)
        self.assertIn("guidance 1.0", recommendation)
        self.assertEqual(
            self.controller._settings_snapshot()["generator_target"],
            "FLUX.2 Klein 9B",
        )

    def test_format_switch_is_independent_and_persisted(self):
        self.assertEqual(self.controller.content_format_var.get(), "Single Image")
        self.controller.generator_target_var.set("FLUX.2 Klein 9B")
        self.controller.mode_tabs.setCurrentIndex(1)
        self.application.processEvents()

        snapshot = self.controller._settings_snapshot()
        options = self.controller._prompt_option_snapshot()
        self.assertEqual(snapshot["content_format"], "Comic Story")
        self.assertEqual(options["content_format"], "Comic Story")
        self.assertEqual(snapshot["generator_target"], "FLUX.2 Klein 9B")
        self.assertIn("four-panel default", self.controller.profile_summary_label.text())

        self.controller.mode_tabs.setCurrentIndex(0)
        self.assertEqual(self.controller.content_format_var.get(), "Single Image")

    def test_camera_control_applies_to_all_three_image_workspaces(self):
        direction = "Low-angle medium-wide shot, 35mm lens"
        self.controller.camera_control_var.set(direction)

        self.assertEqual(
            self.controller._settings_snapshot()["camera_control"],
            direction,
        )
        self.assertEqual(
            self.controller._prompt_option_snapshot()["camera_control"],
            direction,
        )
        for destination, expected_scope in (
            ("prompt", "Camera framing and viewpoint"),
            ("comic", "across the comic panels"),
            ("meme", "underlying meme image"),
        ):
            effective = self.controller._apply_camera_direction(
                "A red-coated courier reaches a flooded gate",
                destination,
            )
            self.assertIn(expected_scope, effective)
            self.assertIn(direction, effective)
            self.assertNotIn("Mandatory user constraints", effective)

        self.assertEqual(
            self.controller._single_image_field_context()["camera_direction"],
            direction,
        )
        self.assertEqual(
            self.controller._comic_field_context()["camera_direction"],
            direction,
        )
        self.assertEqual(
            self.controller._meme_field_context()["camera_direction"],
            direction,
        )
        for index in (0, 1, 2):
            self.controller.mode_tabs.setCurrentIndex(index)
            self.application.processEvents()
            self.assertTrue(self.controller.camera_combo.isVisible())
        for index in (3, 4):
            self.controller.mode_tabs.setCurrentIndex(index)
            self.application.processEvents()
            self.assertFalse(self.controller.camera_combo.isVisible())
        self.controller.mode_tabs.setCurrentIndex(0)

        original = "A red-coated courier reaches a flooded gate."
        self.controller.draft_text.setPlainText(original)
        with mock.patch("krea_prompt_gui.threading.Thread") as thread_class:
            self.controller.correct_prompt()
        worker_args = thread_class.call_args.kwargs["args"]
        bound = inspect.signature(self.controller._correct_prompt_worker).bind(
            *worker_args
        )
        self.assertIn(direction, bound.arguments["draft"])
        self.assertEqual(bound.arguments["requested_prompt"], original)

        self.controller.camera_control_var.set(gui.CAMERA_CONTROL_AUTO)
        self.assertEqual(
            self.controller._apply_camera_direction(original, "prompt"),
            original,
        )

    def test_visual_direction_picker_is_available_in_each_image_workspace(self):
        self.assertEqual(
            set(self.controller.visual_preset_buttons),
            {"prompt", "comic", "meme"},
        )
        self.assertTrue(
            all(
                "mood, lighting, palette" in button.toolTip()
                for button in self.controller.visual_preset_buttons.values()
            )
        )

    def test_content_concept_library_is_available_in_prompt_and_comic_only(self):
        self.assertEqual(
            set(self.controller.concept_preset_buttons),
            {"prompt", "comic"},
        )
        self.assertTrue(
            all(
                "characters, roles, relationships" in button.toolTip()
                for button in self.controller.concept_preset_buttons.values()
            )
        )

    def test_action_and_emotion_libraries_are_available_in_all_image_workspaces(self):
        self.assertEqual(
            set(self.controller.narrative_preset_buttons),
            {
                "prompt:action",
                "prompt:emotion",
                "comic:action",
                "comic:emotion",
                "meme:action",
                "meme:emotion",
            },
        )
        self.assertTrue(
            all(
                "Example:" in button.toolTip()
                for button in self.controller.narrative_preset_buttons.values()
            )
        )

    def test_preset_picker_clear_selection_really_clears_every_visible_choice(self):
        action_category, action_values = next(iter(gui.ACTION_PRESETS.items()))
        action_key = gui.narrative_preset_key(
            "action",
            action_category,
            action_values[0],
        )
        concept_category, concept_values = next(iter(gui.CONCEPT_PRESETS.items()))
        concept_key = gui.concept_preset_key(
            concept_category,
            concept_values[0],
        )
        visual_category, visual_values = next(
            iter(gui.VISUAL_DIRECTION_PRESETS.items())
        )
        visual_key = gui.visual_preset_key(
            visual_category,
            visual_values[0],
        )

        self.controller.narrative_preset_selections["prompt"]["action"] = [
            action_key
        ]
        original_story = gui.format_narrative_presets("action", [action_key])
        self.controller.story_elements_var.set(original_story)
        self.controller.concept_preset_selections["prompt"] = [concept_key]
        self.controller.concepts_var.set(
            gui.format_concept_presets([concept_key])
        )
        self.controller.visual_preset_selections["prompt"] = [visual_key]
        self.controller.visual_direction_var.set(
            gui.format_visual_direction_presets([visual_key])
        )

        def clear_current_dialog(title):
            dialog = next(
                widget
                for widget in self.application.topLevelWidgets()
                if (
                    isinstance(widget, gui.QDialog)
                    and widget.parent() is self.root
                    and widget.windowTitle() == title
                )
            )
            clear_button = next(
                button
                for button in dialog.findChildren(gui.QtButton)
                if button.text() == "Clear selection"
            )
            clear_button.click()
            preset_list = dialog.findChild(gui.QListWidget)
            self.assertTrue(
                all(
                    preset_list.item(index).checkState()
                    == gui.Qt.CheckState.Unchecked
                    for index in range(preset_list.count())
                )
            )
            # QDialog.exec is mocked, so detach the dialog to let it be
            # destroyed when the picker returns instead of leaking a hidden
            # top-level widget into later GUI tests.
            dialog.setParent(None)
            return gui.QDialog.DialogCode.Accepted

        for title, open_picker in (
            (
                "Action preset library",
                lambda: self.controller._open_narrative_preset_picker(
                    "prompt",
                    "action",
                ),
            ),
            (
                "Content concept library",
                lambda: self.controller._open_concept_preset_picker("prompt"),
            ),
            (
                "Creative direction presets",
                lambda: self.controller._open_visual_preset_picker("prompt"),
            ),
        ):
            with mock.patch.object(
                gui.QDialog,
                "exec",
                side_effect=lambda current_title=title: clear_current_dialog(
                    current_title
                ),
            ):
                open_picker()

        self.assertEqual(
            self.controller.narrative_preset_selections["prompt"]["action"],
            [],
        )
        self.assertEqual(
            self.controller.story_elements_var.get(),
            original_story,
        )
        self.assertEqual(self.controller.concept_preset_selections["prompt"], [])
        self.assertEqual(self.controller.concepts_var.get(), "")
        self.assertEqual(self.controller.visual_preset_selections["prompt"], [])
        self.assertEqual(self.controller.visual_direction_var.get(), "")

    def test_single_image_clear_all_clears_its_workspace_and_preserves_other_modes(self):
        prompt_concept_key = next(iter(gui.CONCEPT_PRESET_KEYS))
        prompt_visual_key = next(iter(gui.VISUAL_DIRECTION_PRESET_KEYS))
        prompt_action_key = next(iter(gui.ACTION_PRESET_KEYS))
        comic_concept_key = next(iter(gui.CONCEPT_PRESET_KEYS))
        meme_visual_key = next(iter(gui.VISUAL_DIRECTION_PRESET_KEYS))

        self.controller.draft_text.setPlainText("A courier reaches a flooded gate")
        self.controller.corrected_text.setPlainText("Corrected courier prompt")
        for variable, value in (
            (self.controller.concepts_var, "courier, flooded gate"),
            (self.controller.concept_mix_var, "watercolor:60%, noir:40%"),
            (self.controller.visual_direction_var, "cold moonlight"),
            (self.controller.goal_headline_var, "tense arrival"),
            (self.controller.focus_var, "the courier"),
            (self.controller.weighted_terms_var, "courier:2.0"),
            (self.controller.story_elements_var, "the gate opens"),
            (self.controller.model_instructions_var, "Keep the blue coat"),
            (self.controller.generation_feedback_var, "Use a lower camera"),
        ):
            variable.set(value)
        self.controller.concept_preset_selections["prompt"] = [
            prompt_concept_key
        ]
        self.controller.visual_preset_selections["prompt"] = [prompt_visual_key]
        self.controller.narrative_preset_selections["prompt"]["action"] = [
            prompt_action_key
        ]
        self.controller.local_reference_paths = ["/tmp/reference.png"]
        self.controller._refresh_local_reference_previews()

        self.controller.comic_premise_var.set("Keep this comic")
        self.controller.concept_preset_selections["comic"] = [comic_concept_key]
        self.controller.meme_scene_var.set("Keep this meme")
        self.controller.visual_preset_selections["meme"] = [meme_visual_key]

        self.assertEqual(self.controller.single_clear_button.text(), "Clear all")
        self.assertIn("Single Image", self.controller.single_clear_button.toolTip())
        self.controller.single_clear_button.click()

        self.assertEqual(self.controller.draft_text.toPlainText(), "")
        self.assertEqual(self.controller.corrected_text.toPlainText(), "")
        for variable in (
            self.controller.concepts_var,
            self.controller.concept_mix_var,
            self.controller.visual_direction_var,
            self.controller.goal_headline_var,
            self.controller.focus_var,
            self.controller.weighted_terms_var,
            self.controller.story_elements_var,
            self.controller.model_instructions_var,
            self.controller.generation_feedback_var,
        ):
            self.assertEqual(variable.get(), "")
        self.assertEqual(self.controller.concept_preset_selections["prompt"], [])
        self.assertEqual(
            self.controller.narrative_preset_selections["prompt"],
            {"action": [], "emotion": []},
        )
        self.assertEqual(self.controller.visual_preset_selections["prompt"], [])
        self.assertEqual(self.controller.local_reference_paths, [])
        self.assertEqual(self.controller.reference_preview_list.count(), 0)
        self.assertEqual(self.controller.comic_premise_var.get(), "Keep this comic")
        self.assertEqual(
            self.controller.concept_preset_selections["comic"],
            [comic_concept_key],
        )
        self.assertEqual(self.controller.meme_scene_var.get(), "Keep this meme")
        self.assertEqual(
            self.controller.visual_preset_selections["meme"],
            [meme_visual_key],
        )
        self.assertEqual(self.controller.status_var.get(), "Single Image cleared")

    def test_action_picker_applies_to_prompt_story_beat_and_persists(self):
        chosen_suffixes = {
            "running uphill while glancing back",
            "opening a sealed container",
            "reacting to a sudden revelation",
        }

        def choose_actions():
            dialog = next(
                widget
                for widget in self.application.topLevelWidgets()
                if (
                    isinstance(widget, gui.QDialog)
                    and widget.windowTitle() == "Action preset library"
                    and widget.parent() is self.root
                )
            )
            preset_list = dialog.findChild(gui.QListWidget)
            for index in range(preset_list.count()):
                item = preset_list.item(index)
                if any(item.text().endswith(value) for value in chosen_suffixes):
                    item.setCheckState(gui.Qt.CheckState.Checked)
            return gui.QDialog.DialogCode.Accepted

        with mock.patch.object(
            gui.QDialog,
            "exec",
            side_effect=choose_actions,
        ):
            self.controller._open_narrative_preset_picker("prompt", "action")

        story = self.controller.story_elements_var.get()
        for value in chosen_suffixes:
            self.assertIn(value, story)
        self.assertEqual(
            len(self.controller.narrative_preset_selections["prompt"]["action"]),
            3,
        )
        snapshot = self.controller._settings_snapshot()
        self.assertEqual(
            snapshot["narrative_preset_selections"]["prompt"]["action"],
            self.controller.narrative_preset_selections["prompt"]["action"],
        )
        second_root = gui.PromptCorrectorWindow()
        second_controller = gui.PromptCorrectorApp(second_root)
        second_root.controller = second_controller
        self.assertEqual(second_controller.story_elements_var.get(), story)
        self.assertEqual(
            second_controller.narrative_preset_selections["prompt"]["action"],
            self.controller.narrative_preset_selections["prompt"]["action"],
        )
        second_root.close()

    def test_emotion_picker_appends_to_comic_premise_without_erasing_action(self):
        self.controller.comic_premise_var.set(
            "A courier opens a sealed door"
        )
        chosen_suffixes = {
            "curiosity mixed with dread",
            "fear turning into determination",
        }

        def choose_emotions():
            dialog = next(
                widget
                for widget in self.application.topLevelWidgets()
                if (
                    isinstance(widget, gui.QDialog)
                    and widget.windowTitle() == "Emotion preset library"
                    and widget.parent() is self.root
                )
            )
            preset_list = dialog.findChild(gui.QListWidget)
            for index in range(preset_list.count()):
                item = preset_list.item(index)
                if any(item.text().endswith(value) for value in chosen_suffixes):
                    item.setCheckState(gui.Qt.CheckState.Checked)
            return gui.QDialog.DialogCode.Accepted

        with mock.patch.object(
            gui.QDialog,
            "exec",
            side_effect=choose_emotions,
        ):
            self.controller._open_narrative_preset_picker("comic", "emotion")

        premise = self.controller.comic_premise_var.get()
        self.assertIn("A courier opens a sealed door", premise)
        for value in chosen_suffixes:
            self.assertIn(value, premise)
        self.assertEqual(
            len(self.controller.narrative_preset_selections["comic"]["emotion"]),
            2,
        )

    def test_action_picker_enforces_six_presets_and_meme_clear_is_isolated(self):
        def choose_too_many():
            dialog = next(
                widget
                for widget in self.application.topLevelWidgets()
                if (
                    isinstance(widget, gui.QDialog)
                    and widget.windowTitle() == "Action preset library"
                    and widget.parent() is self.root
                )
            )
            preset_list = dialog.findChild(gui.QListWidget)
            for index in range(7):
                preset_list.item(index).setCheckState(gui.Qt.CheckState.Checked)
            checked = [
                preset_list.item(index)
                for index in range(preset_list.count())
                if preset_list.item(index).checkState() == gui.Qt.CheckState.Checked
            ]
            self.assertEqual(len(checked), gui.NARRATIVE_PRESET_LIMIT)
            return gui.QDialog.DialogCode.Accepted

        with mock.patch.object(
            gui.QDialog,
            "exec",
            side_effect=choose_too_many,
        ):
            self.controller._open_narrative_preset_picker("meme", "action")

        self.assertEqual(
            len(self.controller.narrative_preset_selections["meme"]["action"]),
            gui.NARRATIVE_PRESET_LIMIT,
        )
        prompt_key = gui.narrative_preset_key(
            "emotion",
            "Joy delight and amusement",
            "quiet joy with a relaxed smile and bright eyes",
        )
        self.controller.narrative_preset_selections["prompt"]["emotion"] = [
            prompt_key
        ]
        self.controller.clear_meme()
        self.assertEqual(
            self.controller.narrative_preset_selections["meme"],
            {"action": [], "emotion": []},
        )
        self.assertEqual(
            self.controller.narrative_preset_selections["prompt"]["emotion"],
            [prompt_key],
        )

    def test_content_concept_picker_combines_and_persists_prompt_selection(self):
        chosen_suffixes = {
            "courier",
            "floating village",
            "antique compass",
            "impossible delivery",
        }

        def choose_concepts():
            dialog = next(
                widget
                for widget in self.application.topLevelWidgets()
                if (
                    isinstance(widget, gui.QDialog)
                    and widget.windowTitle() == "Content concept library"
                    and widget.parent() is self.root
                )
            )
            preset_list = dialog.findChild(gui.QListWidget)
            self.assertIsNotNone(preset_list)
            for index in range(preset_list.count()):
                item = preset_list.item(index)
                if any(item.text().endswith(value) for value in chosen_suffixes):
                    item.setCheckState(gui.Qt.CheckState.Checked)
            return gui.QDialog.DialogCode.Accepted

        with mock.patch.object(
            gui.QDialog,
            "exec",
            side_effect=choose_concepts,
        ):
            self.controller._open_concept_preset_picker("prompt")

        concepts = self.controller.concepts_var.get()
        for value in chosen_suffixes:
            self.assertIn(value, concepts)
        self.assertNotIn("Professions and social roles", concepts)
        self.assertEqual(
            len(self.controller.concept_preset_selections["prompt"]),
            4,
        )
        snapshot = self.controller._settings_snapshot()
        self.assertEqual(snapshot["concepts"], concepts)
        self.assertEqual(
            snapshot["concept_preset_selections"]["prompt"],
            self.controller.concept_preset_selections["prompt"],
        )
        second_root = gui.PromptCorrectorWindow()
        second_controller = gui.PromptCorrectorApp(second_root)
        second_root.controller = second_controller
        self.assertEqual(second_controller.concepts_var.get(), concepts)
        self.assertEqual(
            second_controller.concept_preset_selections["prompt"],
            self.controller.concept_preset_selections["prompt"],
        )
        second_root.close()

    def test_content_concept_picker_enforces_eight_active_concepts(self):
        def choose_too_many():
            dialog = next(
                widget
                for widget in self.application.topLevelWidgets()
                if (
                    isinstance(widget, gui.QDialog)
                    and widget.windowTitle() == "Content concept library"
                    and widget.parent() is self.root
                )
            )
            preset_list = dialog.findChild(gui.QListWidget)
            for index in range(9):
                preset_list.item(index).setCheckState(gui.Qt.CheckState.Checked)
            checked = [
                preset_list.item(index)
                for index in range(preset_list.count())
                if (
                    preset_list.item(index).checkState()
                    == gui.Qt.CheckState.Checked
                )
            ]
            self.assertEqual(len(checked), gui.CONCEPT_SELECTION_LIMIT)
            return gui.QDialog.DialogCode.Accepted

        with mock.patch.object(
            gui.QDialog,
            "exec",
            side_effect=choose_too_many,
        ):
            self.controller._open_concept_preset_picker("comic")

        self.assertEqual(
            len(self.controller.concept_preset_selections["comic"]),
            gui.CONCEPT_SELECTION_LIMIT,
        )
        self.assertEqual(
            len(gui.parse_concepts(self.controller.comic_concepts_var.get())),
            gui.CONCEPT_SELECTION_LIMIT,
        )

    def test_prompt_and_comic_concept_library_selections_stay_independent(self):
        prompt_key = gui.concept_preset_key(
            "Professions and social roles",
            "courier",
        )
        comic_key = gui.concept_preset_key(
            "Mythical beings and folklore",
            "benevolent dragon",
        )
        self.controller.concepts_var.set("courier")
        self.controller.comic_concepts_var.set("benevolent dragon")
        self.controller.concept_preset_selections["prompt"] = [prompt_key]
        self.controller.concept_preset_selections["comic"] = [comic_key]

        self.controller.clear_comic_story()

        self.assertEqual(self.controller.concepts_var.get(), "courier")
        self.assertEqual(
            self.controller.concept_preset_selections["prompt"],
            [prompt_key],
        )
        self.assertEqual(self.controller.comic_concepts_var.get(), "")
        self.assertEqual(
            self.controller.concept_preset_selections["comic"],
            [],
        )

    def test_visual_direction_picker_combines_presets_and_persists_selection(self):
        chosen_suffixes = {
            "nostalgic and bittersweet",
            "warm golden-hour sunlight",
            "low valley mist",
        }

        def choose_presets():
            dialog = next(
                widget
                for widget in self.application.topLevelWidgets()
                if (
                    isinstance(widget, gui.QDialog)
                    and widget.windowTitle() == "Creative direction presets"
                )
            )
            preset_list = dialog.findChild(gui.QListWidget)
            self.assertIsNotNone(preset_list)
            for index in range(preset_list.count()):
                item = preset_list.item(index)
                if any(item.text().endswith(value) for value in chosen_suffixes):
                    item.setCheckState(gui.Qt.CheckState.Checked)
            return gui.QDialog.DialogCode.Accepted

        with mock.patch.object(
            gui.QDialog,
            "exec",
            side_effect=choose_presets,
        ):
            self.controller._open_visual_preset_picker("prompt")

        direction = self.controller.visual_direction_var.get()
        self.assertIn(
            "Mood and emotional tone: nostalgic and bittersweet",
            direction,
        )
        self.assertIn("Lighting: warm golden-hour sunlight", direction)
        self.assertIn("Weather and atmosphere: low valley mist", direction)
        self.assertEqual(
            len(self.controller.visual_preset_selections["prompt"]),
            3,
        )
        snapshot = self.controller._settings_snapshot()
        self.assertEqual(snapshot["visual_direction"], direction)
        self.assertEqual(
            snapshot["visual_preset_selections"]["prompt"],
            self.controller.visual_preset_selections["prompt"],
        )
        second_root = gui.PromptCorrectorWindow()
        second_controller = gui.PromptCorrectorApp(second_root)
        second_root.controller = second_controller
        self.assertEqual(second_controller.visual_direction_var.get(), direction)
        self.assertEqual(
            second_controller.visual_preset_selections["prompt"],
            self.controller.visual_preset_selections["prompt"],
        )
        second_root.close()

    def test_single_visual_direction_reaches_prompt_without_leaking_to_other_modes(self):
        self.controller.visual_direction_var.set(
            "Mood: ominous; Lighting: moonlit; Weather: low fog."
        )
        prompt = self.controller._apply_visual_direction(
            "A courier reaches a flooded gate",
            "prompt",
        )
        comic = self.controller._apply_visual_direction(
            "A four-panel courier story",
            "comic",
        )
        meme = self.controller._apply_visual_direction(
            "A courier reaction meme",
            "meme",
        )

        self.assertIn("Visual direction: Mood: ominous", prompt)
        self.assertNotIn("ominous", comic)
        self.assertNotIn("ominous", meme)
        self.assertEqual(
            self.controller._single_image_field_context()["visual_direction"],
            "Mood: ominous; Lighting: moonlit; Weather: low fog.",
        )

    def test_single_image_story_editor_is_not_a_panel_editor(self):
        self.assertIsInstance(self.controller.story_elements_entry, gui.QTextEdit)
        self.controller.story_elements_entry.setPlainText("She reaches the door as it opens")
        self.application.processEvents()

        self.assertEqual(
            self.controller.story_elements_var.get(),
            "She reaches the door as it opens",
        )
        self.assertIn("one still image", self.controller.story_elements_entry.placeholderText())

    def test_comic_panel_count_adjusts_dedicated_editors_and_builds_contract(self):
        self.controller.mode_tabs.setCurrentIndex(1)
        self.controller.comic_panel_count_var.set(3)
        self.controller.comic_title_var.set("The Key")
        self.controller.comic_premise_var.set("A courier opens a sealed door")
        self.controller.comic_continuity_var.set("same blue coat and brass key")
        self.controller.comic_aspect_ratio_var.set("3:4 portrait")
        self.controller.comic_panel_vars[0].set("the courier finds the brass key")
        self.controller.comic_panel_vars[1].set("the same courier reaches the sealed door")
        self.controller.comic_panel_vars[2].set('the door opens and she says "Finally"')
        self.application.processEvents()

        self.assertTrue(all(group.isVisible() for group in self.controller.comic_panel_groups[:3]))
        self.assertFalse(self.controller.comic_panel_groups[3].isVisible())
        draft, story = self.controller._comic_story_inputs()
        self.assertIn("A 3-panel comic story page", draft)
        self.assertIn("same blue coat and brass key", draft)
        self.assertIn("Aspect ratio: 3:4 portrait", draft)
        self.assertIn("full-width panel across the bottom", draft)
        self.assertNotIn("2 x 3", draft)
        self.assertIn(
            "full-width panel across the bottom",
            self.controller.comic_layout_preview_label.text(),
        )
        self.assertNotIn('Working title: "The Key"', draft)
        self.assertIn("Panel 1: the courier finds the brass key", story)
        self.assertIn('Panel 3: the door opens and she says "Finally"', story)

        snapshot = self.controller._settings_snapshot()
        self.assertEqual(snapshot["comic_panel_count"], 3)
        self.assertEqual(snapshot["comic_aspect_ratio"], "3:4 portrait")
        self.assertEqual(snapshot["comic_panels"][1], "the same courier reaches the sealed door")
        self.controller.comic_layout_var.set("3 x 2 grid")
        self.assertEqual(self.controller.comic_layout_var.get(), "Auto grid")

    def test_all_visible_controls_have_descriptive_tooltips_with_examples(self):
        widget_types = (
            gui.QPushButton,
            gui.QCheckBox,
            gui.QComboBox,
            gui.QSpinBox,
            gui.QDoubleSpinBox,
            gui.QLineEdit,
            gui.QTextEdit,
            gui.QListWidget,
            gui.QSlider,
            gui.QProgressBar,
            gui.QGroupBox,
            gui.QLabel,
        )
        for widget_type in widget_types:
            for widget in self.root.findChildren(widget_type):
                with self.subTest(widget_type=widget_type.__name__, object_name=widget.objectName()):
                    self.assertTrue(widget.toolTip().strip())
                    self.assertIn("Example:", widget.toolTip())

        for tabs in self.root.findChildren(gui.QTabWidget):
            for index in range(tabs.count()):
                with self.subTest(tab=tabs.tabText(index)):
                    self.assertIn("Example:", tabs.tabToolTip(index))

        for action in self.root.findChildren(gui.QAction):
            if not action.isSeparator() and action.text().strip():
                with self.subTest(action=action.text()):
                    self.assertIn("Example:", action.toolTip())
        authored_menus = [menu for menu in self.root.findChildren(gui.QMenu) if menu.title().strip()]
        self.assertTrue(all(menu.toolTipsVisible() for menu in authored_menus))

    def test_chat_sends_system_instruction_and_full_conversation(self):
        self.controller.chat_system_prompt_var.set("Answer precisely.")
        self.controller.chat_temperature_var.set(0.4)
        self.controller.chat_max_tokens_var.set(4096)
        self.controller.chat_messages = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
        ]
        self.controller.chat_input.setPlainText("Follow-up question")

        with mock.patch("krea_prompt_gui.threading.Thread") as thread_class:
            self.controller.send_chat_message()

        worker_args = thread_class.call_args.kwargs["args"]
        inspect.signature(self.controller._chat_worker).bind(*worker_args)
        messages = worker_args[3]
        self.assertEqual(messages[0], {"role": "system", "content": "Answer precisely."})
        self.assertEqual(messages[-1], {"role": "user", "content": "Follow-up question"})
        self.assertEqual(len(messages), 4)
        self.assertTrue(self.controller.request_in_progress)
        self.assertFalse(self.controller.chat_send_button.isEnabled())
        self.assertTrue(self.controller.chat_stop_button.isEnabled())

        request_id, base_url, model, messages = worker_args
        with mock.patch("krea_prompt_gui.chat_completion", return_value="Direct answer") as completion:
            self.controller._chat_worker(request_id, base_url, model, messages)
        self.application.processEvents()

        sent = completion.call_args.kwargs
        self.assertEqual(sent["messages"], messages)
        self.assertEqual(sent["temperature"], 0.4)
        self.assertEqual(sent["max_tokens"], 4096)
        self.assertTrue(callable(sent["chunk_callback"]))
        self.assertEqual(self.controller.chat_messages[-1]["content"], "Direct answer")
        self.assertFalse(self.controller.request_in_progress)
        self.assertTrue(self.controller.chat_send_button.isEnabled())

    def test_correction_worker_arguments_stay_in_sync(self):
        self.controller.draft_text.setPlainText("knight at a castle gate")
        self.controller.concepts_var.set("portrait")
        self.controller.weighted_terms_var.set("face:2.4")
        self.controller.concept_mix_var.set("watercolor:70%, cyberpunk:30%")
        with mock.patch("krea_prompt_gui.threading.Thread") as thread_class:
            self.controller.correct_prompt()
        worker_args = thread_class.call_args.kwargs["args"]
        bound = inspect.signature(self.controller._correct_prompt_worker).bind(*worker_args)
        self.assertEqual(bound.arguments["concepts"], "portrait, watercolor, cyberpunk")
        self.assertIn("watercolor:2.25", bound.arguments["weighted_terms"])
        self.assertIn("cyberpunk:1.25", bound.arguments["weighted_terms"])
        self.assertEqual(bound.arguments["model_instructions"], "")
        self.assertIn(
            "watercolor 70%",
            bound.arguments["private_model_instructions"],
        )

    def test_saved_results_and_history_drop_legacy_private_guidance(self):
        history = self.controller._history_setting(
            [
                {
                    "requested_prompt": "A watercolor city.",
                    "corrected_prompt": (
                        "A translucent watercolor city at dusk. "
                        "Mandatory user constraints: "
                        "keep every nonzero ingredient visibly recognizable."
                    ),
                }
            ]
        )

        self.assertEqual(
            history[0]["corrected_prompt"],
            "A translucent watercolor city at dusk.",
        )

    def test_automatic_image_research_uses_concepts_not_the_whole_draft(self):
        self.controller.draft_text.setPlainText(
            "A courier runs through a rainy station, close camera, red coat."
        )
        self.controller.concepts_var.set("Art Nouveau")
        self.controller.reference_images_var.set(True)
        self.controller.audit_repair_var.set(False)
        with mock.patch("krea_prompt_gui.threading.Thread") as thread_class:
            self.controller.correct_prompt()
        worker_args = thread_class.call_args.kwargs["args"]

        with mock.patch.object(
            self.controller, "_collect_reference_images_for_prompt"
        ) as whole_prompt_lookup:
            with mock.patch(
                "krea_prompt_gui.collect_integrated_concept_research",
                return_value="Web reference concept glossary for Art Nouveau: curved organic lines.",
            ) as concept_lookup:
                with mock.patch(
                    "krea_prompt_gui.post_chat_completion",
                    return_value="A corrected courier prompt.",
                ) as completion:
                    self.controller._correct_prompt_worker(*worker_args)

        whole_prompt_lookup.assert_not_called()
        concept_lookup.assert_called_once()
        self.assertEqual(concept_lookup.call_args.args[0], "Art Nouveau")
        self.assertEqual(completion.call_args.kwargs["image_context"], "")
        self.assertIn(
            "Web reference concept glossary",
            completion.call_args.kwargs["concept_context"],
        )
        self.assertTrue(callable(completion.call_args.kwargs["diagnostic_callback"]))

    def test_comic_workspace_starts_comic_worker_with_numbered_panels(self):
        self.controller.mode_tabs.setCurrentIndex(1)
        self.controller.comic_panel_count_var.set(2)
        self.controller.comic_premise_var.set("A pilot repairs a stranded airship")
        self.controller.comic_concepts_var.set(
            "dieselpunk engineering, storm electricity"
        )
        self.controller.comic_visual_direction_var.set(
            "Ink wash with copper highlights and angular panel compositions."
        )
        self.controller.comic_panel_vars[0].set("the pilot discovers the broken engine")
        self.controller.comic_panel_vars[1].set("the same pilot restarts it at sunrise")
        self.controller.comic_speech_bubbles_var.set(True)
        self.controller.comic_dialogue_direction_var.set(
            "Characters speak like tired airship mechanics using clipped workshop slang."
        )
        self.controller.artistic_detail_freedom_var.set(True)

        with mock.patch("krea_prompt_gui.threading.Thread") as thread_class:
            self.controller.correct_comic_story()

        worker_args = thread_class.call_args.kwargs["args"]
        bound = inspect.signature(self.controller._correct_prompt_worker).bind(*worker_args)
        self.assertEqual(bound.arguments["content_format"], "Comic Story")
        self.assertEqual(bound.arguments["destination"], "comic")
        self.assertIn("A 2-panel comic story page", bound.arguments["draft"])
        self.assertIn("Speech bubbles are allowed", bound.arguments["draft"])
        self.assertIn("Mandatory dialogue wording contract", bound.arguments["draft"])
        self.assertIn("clipped workshop slang", bound.arguments["draft"])
        self.assertIn("Required concepts to integrate", bound.arguments["draft"])
        self.assertIn("dieselpunk engineering", bound.arguments["draft"])
        self.assertIn("Mandatory shared comic style direction", bound.arguments["draft"])
        self.assertEqual(
            bound.arguments["concepts"],
            "dieselpunk engineering, storm electricity",
        )
        self.assertIn(
            "Ink wash with copper highlights",
            bound.arguments["model_instructions"],
        )
        self.assertTrue(bound.arguments["artistic_detail_freedom"])
        self.assertIn("Panel 1: the pilot discovers", bound.arguments["story_elements"])
        self.assertIn("Panel 2: the same pilot restarts", bound.arguments["story_elements"])

    def test_comic_speech_bubbles_can_be_disabled(self):
        self.controller.comic_panel_count_var.set(2)
        self.controller.comic_premise_var.set("Two robots repair a lunar greenhouse")
        self.controller.comic_panel_vars[0].set("one robot finds the broken window")
        self.controller.comic_panel_vars[1].set("both robots seal it before sunrise")
        self.controller.comic_speech_bubbles_var.set(False)
        self.controller.comic_dialogue_direction_var.set(
            "Use caveman grammar."
        )

        draft, _story_elements = self.controller._comic_story_inputs()

        self.assertIn("Do not invent speech bubbles", draft)
        self.assertNotIn("Use caveman grammar", draft)
        self.assertFalse(
            self.controller._settings_snapshot()["comic_speech_bubbles"]
        )

    def test_comic_workspace_does_not_inherit_prompt_corrector_content(self):
        self.controller.mode_tabs.setCurrentIndex(1)
        self.controller.comic_panel_count_var.set(2)
        self.controller.comic_premise_var.set("A pilot repairs a stranded airship")
        self.controller.comic_panel_vars[0].set("the pilot discovers the broken engine")
        self.controller.comic_panel_vars[1].set("the pilot restarts it at sunrise")
        self.controller.concepts_var.set("unrelated ice cream")
        self.controller.concept_mix_var.set("unrelated watercolor:100%")
        self.controller.goal_headline_var.set("unrelated beach scene")
        self.controller.focus_var.set("unrelated cone")
        self.controller.weighted_terms_var.set("unrelated storm:1.8")
        self.controller.model_instructions_var.set("Add unrelated rain.")
        self.controller.generation_feedback_var.set("Use the previous unrelated image.")
        self.controller.reference_images_var.set(True)

        with (
            mock.patch.object(
                self.controller,
                "_local_reference_candidates",
                return_value=[{"title": "unrelated.png", "url": "file:///tmp/unrelated.png"}],
            ) as local_references,
            mock.patch("krea_prompt_gui.threading.Thread") as thread_class,
        ):
            self.controller.correct_comic_story()

        local_references.assert_not_called()
        worker_args = thread_class.call_args.kwargs["args"]
        bound = inspect.signature(self.controller._correct_prompt_worker).bind(*worker_args)
        self.assertEqual(bound.arguments["concepts"], "")
        self.assertEqual(bound.arguments["goal_headline"], "")
        self.assertEqual(bound.arguments["focus"], "")
        self.assertEqual(bound.arguments["weighted_terms"], "")
        self.assertEqual(bound.arguments["model_instructions"], "")
        self.assertEqual(bound.arguments["generation_feedback"], "")
        self.assertFalse(bound.arguments["reference_image_analysis"])
        self.assertEqual(bound.arguments["local_reference_candidates"], [])

    def test_meme_creator_builds_exact_top_and_bottom_text_contract(self):
        self.controller.mode_tabs.setCurrentIndex(2)
        self.controller.meme_scene_var.set(
            "an orange cat calmly sitting in front of a burning spreadsheet"
        )
        self.controller.meme_top_text_var.set("I SAID I FIXED IT")
        self.controller.meme_bottom_text_var.set("I DIDN'T SAY WHAT")
        self.controller.meme_aspect_ratio_var.set("4:5 portrait")
        self.controller.meme_caption_style_var.set("Clean bold sans-serif")
        self.controller.meme_visual_direction_var.set("deadpan office photography")
        self.controller.meme_focus_var.set("the cat's completely calm expression")
        self.controller.meme_temperature_var.set(0.35)

        draft = self.controller._meme_inputs()

        self.assertIn("single image-macro meme in 4:5 portrait format", draft)
        self.assertIn('upper edge reading exactly "I SAID I FIXED IT"', draft)
        self.assertIn('lower edge reading exactly "I DIDN\'T SAY WHAT"', draft)
        self.assertIn("Caption treatment: Clean bold sans-serif", draft)
        self.assertIn("deadpan office photography", draft)
        self.assertIn("Primary visual focus: the cat's completely calm expression", draft)
        self.assertEqual(self.controller.content_format_var.get(), "Meme")

        snapshot = self.controller._settings_snapshot()
        self.assertEqual(snapshot["meme_top_text"], "I SAID I FIXED IT")
        self.assertEqual(snapshot["meme_bottom_text"], "I DIDN'T SAY WHAT")
        self.assertEqual(snapshot["meme_aspect_ratio"], "4:5 portrait")
        self.assertEqual(snapshot["meme_focus"], "the cat's completely calm expression")
        self.assertEqual(snapshot["meme_temperature"], 0.35)
        self.assertIsNotNone(self.controller.meme_temperature_spin)
        self.assertEqual(self.controller.meme_temperature_spin.value(), 0.35)

    def test_single_image_invent_buttons_fill_one_field_and_keep_model_loaded(self):
        self.controller.mode_tabs.setCurrentIndex(0)
        self.assertEqual(len(self.controller.single_image_invent_buttons), 10)
        self.controller.concepts_var.set("solarpunk, flooded city")
        self.controller.focus_var.set("the courier's brass medicine satchel")
        self.controller.draft_text.setPlainText(
            "A courier reaches a flooded city gate."
        )
        self.controller.temperature_var.set(0.32)
        self.controller.artistic_detail_freedom_var.set(True)

        with mock.patch("krea_prompt_gui.threading.Thread") as thread_class:
            self.controller.invent_single_image_field("draft")

        worker_args = thread_class.call_args.kwargs["args"]
        bound = inspect.signature(
            self.controller._invent_single_image_field_worker
        ).bind(*worker_args)
        self.assertEqual(bound.arguments["field"], "draft")
        self.assertIn(
            "courier reaches a flooded city gate",
            bound.arguments["context"]["draft"],
        )
        self.assertIn("solarpunk", bound.arguments["context"]["concepts"])
        self.assertTrue(bound.arguments["context"]["artistic_detail_freedom"])
        self.assertTrue(
            all(
                not button.isEnabled()
                for button in self.controller.single_image_invent_buttons
            )
        )

        with mock.patch(
            "krea_prompt_gui.chat_completion",
            return_value=(
                "Image prompt: A red-coated courier raises a brass medicine satchel "
                "above floodwater at a luminous solarpunk city gate. 产品摄影。"
            ),
        ) as completion:
            with mock.patch("krea_prompt_gui.unload_lm_studio_model") as unload:
                self.controller._invent_single_image_field_worker(*worker_args)
        self.application.processEvents()

        unload.assert_not_called()
        self.assertEqual(completion.call_args.kwargs["temperature"], 0.32)
        self.assertIn(
            "Artistic detail freedom is enabled",
            completion.call_args.kwargs["messages"][0]["content"],
        )
        self.assertIn(
            "mandatory creative seed",
            completion.call_args.kwargs["messages"][0]["content"],
        )
        self.assertEqual(
            completion.call_args.kwargs["ttl"],
            gui.CREATIVE_SESSION_TTL_SECONDS,
        )
        self.assertIn("red-coated courier", self.controller.draft_text.toPlainText())
        self.assertNotIn("产品摄影", self.controller.draft_text.toPlainText())
        self.assertTrue(
            all(
                button.isEnabled()
                for button in self.controller.single_image_invent_buttons
            )
        )

    def test_every_inventable_input_has_a_disabled_recall_button_initially(self):
        expected = {
            "single:draft",
            "single:concepts",
            "single:concept_mix",
            "single:visual_direction",
            "single:goal_headline",
            "single:focus",
            "single:story_elements",
            "single:weighted_terms",
            "single:model_instructions",
            "single:generation_feedback",
            "comic:title",
            "comic:premise",
            "comic:continuity",
            "comic:concepts",
            "comic:visual_direction",
            "comic:dialogue_direction",
            "comic:all_panels",
            *(f"comic:panel_{index}" for index in range(1, 13)),
            "meme:response_context",
            "meme:response_goal",
            "meme:scene",
            "meme:focus",
            "meme:top",
            "meme:bottom",
            "meme:visual_direction",
        }

        self.assertEqual(set(self.controller.invent_recall_buttons), expected)
        self.assertTrue(
            all(
                not button.isEnabled()
                for button in self.controller.invent_recall_buttons.values()
            )
        )
        self.assertTrue(
            all(
                "immediately before" in button.toolTip()
                for button in self.controller.invent_recall_buttons.values()
            )
        )

    def test_single_image_recall_restores_exact_pre_invent_input_once(self):
        original = "  A courier reaches a flooded city gate.\nKeep the brass satchel.  "
        self.controller.draft_text.setPlainText(original)
        recall_button = self.controller.invent_recall_buttons["single:draft"]

        with mock.patch("krea_prompt_gui.threading.Thread") as thread_class:
            self.controller.invent_single_image_field("draft")
        worker_args = thread_class.call_args.kwargs["args"]
        request_id = worker_args[0]
        self.assertFalse(recall_button.isEnabled())

        self.controller._show_invented_single_image_field(
            request_id,
            "draft",
            "A generated courier prompt.",
        )

        self.assertEqual(
            self.controller.draft_text.toPlainText(),
            "A generated courier prompt.",
        )
        self.assertTrue(recall_button.isEnabled())
        recall_button.click()
        self.application.processEvents()

        self.assertEqual(self.controller.draft_text.toPlainText(), original)
        self.assertFalse(recall_button.isEnabled())
        self.assertEqual(
            self.controller.status_var.get(),
            "Recalled input from before Invent",
        )

    def test_failed_invent_keeps_the_previous_successful_recall_value(self):
        self.controller.focus_var.set("original brass medicine satchel")
        with mock.patch("krea_prompt_gui.threading.Thread") as thread_class:
            self.controller.invent_single_image_field("focus")
        first_request_id = thread_class.call_args.kwargs["args"][0]
        self.controller._show_invented_single_image_field(
            first_request_id,
            "focus",
            "generated luminous satchel",
        )

        with mock.patch("krea_prompt_gui.threading.Thread") as thread_class:
            self.controller.invent_single_image_field("focus")
        second_request_id = thread_class.call_args.kwargs["args"][0]
        with mock.patch.object(gui.messagebox, "showerror"):
            self.controller._show_invent_error(
                second_request_id,
                "The model request failed.",
            )

        self.controller.invent_recall_buttons["single:focus"].click()
        self.assertEqual(
            self.controller.focus_var.get(),
            "original brass medicine satchel",
        )

    def test_comic_invent_buttons_fill_one_panel_and_keep_model_loaded(self):
        self.controller.mode_tabs.setCurrentIndex(1)
        self.assertEqual(len(self.controller.comic_invent_buttons), 19)
        self.assertEqual(len(self.controller.comic_panel_invent_buttons), 12)
        self.controller.comic_panel_count_var.set(4)
        self.controller.comic_title_var.set("The Last Delivery")
        self.controller.comic_premise_var.set(
            "A courier crosses a flooded city with medicine."
        )
        self.controller.comic_panel_vars[0].set(
            "The courier reaches the flooded clinic gate."
        )
        self.controller.comic_panel_vars[1].set(
            "The courier starts forcing the rusted gate open."
        )
        self.controller.comic_panel_vars[2].set(
            "The courier hands the medicine to a doctor."
        )
        self.controller.temperature_var.set(0.38)
        self.controller.artistic_detail_freedom_var.set(True)
        self.controller.comic_speech_bubbles_var.set(True)
        self.controller.comic_concepts_var.set(
            "bioluminescent algae, Art Nouveau ironwork"
        )
        self.controller.comic_visual_direction_var.set(
            "Muted watercolor with clean ink lines."
        )
        self.controller.comic_dialogue_direction_var.set(
            "Use short caveman grammar, simple words, and no modern slang."
        )

        with mock.patch("krea_prompt_gui.threading.Thread") as thread_class:
            self.controller.invent_comic_field("panel_2")

        worker_args = thread_class.call_args.kwargs["args"]
        bound = inspect.signature(
            self.controller._invent_comic_field_worker
        ).bind(*worker_args)
        self.assertEqual(bound.arguments["field"], "panel_2")
        self.assertIn(
            "flooded clinic gate",
            bound.arguments["context"]["panels"][0],
        )
        self.assertIn(
            "starts forcing the rusted gate open",
            bound.arguments["context"]["panels"][1],
        )
        self.assertTrue(bound.arguments["context"]["speech_bubbles"])
        self.assertIn(
            "bioluminescent algae",
            bound.arguments["context"]["concepts"],
        )
        self.assertIn(
            "Muted watercolor",
            bound.arguments["context"]["visual_direction"],
        )
        self.assertIn(
            "short caveman grammar",
            bound.arguments["context"]["dialogue_direction"],
        )
        self.assertTrue(bound.arguments["context"]["artistic_detail_freedom"])
        self.assertTrue(
            all(not button.isEnabled() for button in self.controller.comic_invent_buttons)
        )

        with mock.patch(
            "krea_prompt_gui.chat_completion",
            return_value=(
                'Panel 2 beat: The courier braces against a wave and shouts "Open the gate!"'
            ),
        ) as completion:
            with mock.patch("krea_prompt_gui.unload_lm_studio_model") as unload:
                self.controller._invent_comic_field_worker(*worker_args)
        self.application.processEvents()

        unload.assert_not_called()
        self.assertEqual(completion.call_args.kwargs["temperature"], 0.38)
        self.assertIn(
            "Speech bubbles are allowed",
            completion.call_args.kwargs["messages"][0]["content"],
        )
        self.assertIn(
            "short caveman grammar",
            completion.call_args.kwargs["messages"][0]["content"],
        )
        self.assertIn(
            "Required concept integration contract",
            completion.call_args.kwargs["messages"][0]["content"],
        )
        self.assertIn(
            "Mandatory shared comic style direction",
            completion.call_args.kwargs["messages"][0]["content"],
        )
        self.assertIn(
            "Artistic detail freedom is enabled",
            completion.call_args.kwargs["messages"][0]["content"],
        )
        self.assertIn(
            "mandatory creative seed",
            completion.call_args.kwargs["messages"][0]["content"],
        )
        self.assertIn(
            "Current field value: The courier starts forcing the rusted gate open.",
            completion.call_args.kwargs["messages"][1]["content"],
        )
        self.assertEqual(
            completion.call_args.kwargs["ttl"],
            gui.CREATIVE_SESSION_TTL_SECONDS,
        )
        invented_panel = self.controller.comic_panel_vars[1].get()
        self.assertIn(
            'The courier braces against a wave and shouts "Open the gate!"',
            invented_panel,
        )
        self.assertIn("clearly readable speech bubble", invented_panel)
        self.assertIn("tail pointing unambiguously", invented_panel)
        self.assertTrue(
            all(button.isEnabled() for button in self.controller.comic_invent_buttons)
        )

    def test_invent_all_comic_panels_uses_one_request_and_fills_visible_panels(self):
        self.controller.mode_tabs.setCurrentIndex(1)
        self.controller.comic_panel_count_var.set(3)
        self.controller.comic_premise_var.set(
            "A courier crosses a flooded city with medicine."
        )
        self.controller.comic_continuity_var.set(
            "Same red coat and brass satchel in every panel."
        )
        self.controller.temperature_var.set(0.44)
        self.controller.comic_speech_bubbles_var.set(True)
        self.controller.artistic_detail_freedom_var.set(True)
        self.controller.comic_concepts_var.set(
            "bioluminescent algae, Art Nouveau ironwork"
        )
        self.controller.comic_visual_direction_var.set(
            "Muted watercolor with clean ink lines."
        )
        self.controller.comic_dialogue_direction_var.set(
            "Characters speak like cavemen using short grammar and simple words."
        )
        self.controller.comic_panel_vars[0].set(
            "The courier reaches the flooded clinic gate."
        )

        with mock.patch("krea_prompt_gui.threading.Thread") as thread_class:
            self.controller.invent_all_comic_panels()

        worker_args = thread_class.call_args.kwargs["args"]
        bound = inspect.signature(
            self.controller._invent_all_comic_panels_worker
        ).bind(*worker_args)
        self.assertEqual(bound.arguments["context"]["panel_count"], 3)
        self.assertIn(
            "full-width panel across the bottom",
            bound.arguments["context"]["layout"],
        )
        self.assertIn(
            "speak like cavemen",
            bound.arguments["context"]["dialogue_direction"],
        )
        self.assertIn(
            "bioluminescent algae",
            bound.arguments["context"]["concepts"],
        )
        self.assertIn(
            "Muted watercolor",
            bound.arguments["context"]["visual_direction"],
        )
        self.assertIn(
            "courier reaches the flooded clinic gate",
            bound.arguments["context"]["panels"][0].lower(),
        )
        self.assertTrue(
            all(not button.isEnabled() for button in self.controller.comic_invent_buttons)
        )

        response = "\n".join(
            (
                "Panel 1: The red-coated courier reaches the flooded clinic gate.",
                (
                    'Panel 2: The courier raises the brass satchel and says '
                    '"Open the gate!" 产品摄影。'
                ),
                "Panel 3: The courier hands the dry medicine to the doctor.",
            )
        )
        with mock.patch(
            "krea_prompt_gui.chat_completion",
            return_value=response,
        ) as completion:
            with mock.patch("krea_prompt_gui.unload_lm_studio_model") as unload:
                self.controller._invent_all_comic_panels_worker(*worker_args)
        self.application.processEvents()

        completion.assert_called_once()
        unload.assert_not_called()
        self.assertEqual(completion.call_args.kwargs["temperature"], 0.44)
        self.assertEqual(
            completion.call_args.kwargs["ttl"],
            gui.CREATIVE_SESSION_TTL_SECONDS,
        )
        self.assertIn(
            "full-width panel across the bottom",
            completion.call_args.kwargs["messages"][1]["content"],
        )
        self.assertIn(
            "mandatory creative seed",
            completion.call_args.kwargs["messages"][0]["content"],
        )
        self.assertIn(
            "Invent from scratch only for blank beats",
            completion.call_args.kwargs["messages"][0]["content"],
        )
        self.assertIn(
            "speak like cavemen",
            completion.call_args.kwargs["messages"][0]["content"],
        )
        self.assertIn(
            "Required concept integration contract",
            completion.call_args.kwargs["messages"][0]["content"],
        )
        self.assertIn(
            "Mandatory shared comic style direction",
            completion.call_args.kwargs["messages"][0]["content"],
        )
        self.assertIn("reaches the flooded clinic gate", self.controller.comic_panel_vars[0].get())
        self.assertIn("clearly readable speech bubble", self.controller.comic_panel_vars[1].get())
        self.assertNotIn("产品摄影", self.controller.comic_panel_vars[1].get())
        self.assertIn("hands the dry medicine", self.controller.comic_panel_vars[2].get())
        self.assertTrue(
            all(button.isEnabled() for button in self.controller.comic_invent_buttons)
        )

    def test_recall_all_panels_restores_every_visible_pre_invent_beat(self):
        original_panels = [
            "The courier reaches the clinic gate.",
            "",
            "The doctor receives the medicine.",
        ]
        self.controller.comic_panel_count_var.set(3)
        for index, value in enumerate(original_panels):
            self.controller.comic_panel_vars[index].set(value)

        with mock.patch("krea_prompt_gui.threading.Thread") as thread_class:
            self.controller.invent_all_comic_panels()
        request_id = thread_class.call_args.kwargs["args"][0]
        self.controller._show_invented_all_comic_panels(
            request_id,
            [
                "Generated panel one.",
                "Generated panel two.",
                "Generated panel three.",
            ],
        )

        recall_all = self.controller.invent_recall_buttons["comic:all_panels"]
        self.assertTrue(recall_all.isEnabled())
        self.assertTrue(
            all(
                self.controller.invent_recall_buttons[
                    f"comic:panel_{index}"
                ].isEnabled()
                for index in range(1, 4)
            )
        )
        recall_all.click()
        self.application.processEvents()

        self.assertEqual(
            [
                self.controller.comic_panel_vars[index].get()
                for index in range(3)
            ],
            original_panels,
        )
        self.assertFalse(recall_all.isEnabled())
        self.assertTrue(
            all(
                not self.controller.invent_recall_buttons[
                    f"comic:panel_{index}"
                ].isEnabled()
                for index in range(1, 4)
            )
        )

    def test_comic_panel_invention_runs_enabled_grounded_concept_research(self):
        self.controller.mode_tabs.setCurrentIndex(1)
        self.controller.comic_panel_count_var.set(2)
        self.controller.comic_premise_var.set(
            "A pilot repairs a stranded airship."
        )
        self.controller.comic_concepts_var.set(
            "dieselpunk engineering, storm electricity"
        )
        self.controller.comic_panel_vars[0].set(
            "The pilot opens the damaged engine housing."
        )
        self.controller.live_research_var.set(True)

        with mock.patch("krea_prompt_gui.threading.Thread") as thread_class:
            self.controller.invent_comic_field("panel_2")
        worker_args = thread_class.call_args.kwargs["args"]

        with mock.patch.object(
            self.controller,
            "_collect_grounded_comic_concept_research",
            return_value=(
                "Grounded concept glossary and factual verification only:\n"
                "Dieselpunk machinery uses visibly mechanical industrial components."
            ),
        ) as research:
            with mock.patch(
                "krea_prompt_gui.chat_completion",
                return_value="Panel 2: The pilot reconnects a copper busbar.",
            ) as completion:
                self.controller._invent_comic_field_worker(*worker_args)
        self.application.processEvents()

        research.assert_called_once()
        self.assertEqual(
            research.call_args.kwargs["concepts"],
            "dieselpunk engineering, storm electricity",
        )
        system_message = completion.call_args.kwargs["messages"][0]["content"]
        self.assertIn("Grounded concept research contract", system_message)
        self.assertIn("visibly mechanical industrial components", system_message)
        self.assertIn(
            "Do not copy or infer another source image's subject",
            system_message,
        )

    def test_grounded_comic_research_probes_web_and_reconciles_concepts_only(self):
        self.controller.active_request_id = 7
        self.controller.lm_timeout_var.set("120")
        order = []

        def probe(**kwargs):
            order.append("probe")
            self.assertIn("Art Nouveau ironwork", kwargs["concept_keywords"])
            self.assertEqual(kwargs["story_elements"], "")
            return "TARGET | style | Art Nouveau ironwork | known | organic curves"

        def targets(*args, **kwargs):
            order.append("targets")
            self.assertIn("comic concept keywords", args[0])
            return [
                {
                    "category": "style",
                    "term": "Art Nouveau ironwork",
                    "confidence": "known",
                    "knowledge": "organic curves",
                }
            ]

        def web(*_args, **_kwargs):
            order.append("web")
            return "Targeted verification evidence"

        def reconcile(**kwargs):
            order.append("reconcile")
            self.assertIn("comic concept keywords", kwargs["prompt"])
            return "Verified visual facts: flowing botanical iron curves."

        with mock.patch(
            "krea_prompt_gui.probe_model_visual_knowledge", side_effect=probe
        ):
            with mock.patch(
                "krea_prompt_gui.prompt_research_targets", side_effect=targets
            ):
                with mock.patch(
                    "krea_prompt_gui.collect_targeted_prompt_research",
                    side_effect=web,
                ):
                    with mock.patch(
                        "krea_prompt_gui.reconcile_model_knowledge_with_web",
                        side_effect=reconcile,
                    ):
                        result = (
                            self.controller._collect_grounded_comic_concept_research(
                                request_id=7,
                                base_url="http://127.0.0.1:1234",
                                model="test-model",
                                concepts="Art Nouveau ironwork",
                                search_engine="DuckDuckGo HTML",
                            )
                        )

        self.assertEqual(order, ["probe", "targets", "web", "reconcile"])
        self.assertIn("Grounded concept glossary", result)
        self.assertIn("flowing botanical iron curves", result)

    def test_comic_dialogue_direction_is_inventable_persisted_and_clearable(self):
        self.controller.mode_tabs.setCurrentIndex(1)
        self.controller.comic_premise_var.set(
            "Two cave people discover a strange metal door."
        )
        self.controller.comic_dialogue_direction_var.set(
            "Use short caveman grammar."
        )
        self.controller.comic_concepts_var.set(
            "Stone Age astronomy, bioluminescent fungi"
        )
        self.controller.comic_visual_direction_var.set(
            "Charcoal linework with ochre washes."
        )
        self.assertEqual(
            self.controller._settings_snapshot()["comic_dialogue_direction"],
            "Use short caveman grammar.",
        )
        snapshot = self.controller._settings_snapshot()
        self.assertIn("Stone Age astronomy", snapshot["comic_concepts"])
        self.assertIn("Charcoal linework", snapshot["comic_visual_direction"])

        with mock.patch("krea_prompt_gui.threading.Thread") as thread_class:
            self.controller.invent_comic_field("dialogue_direction")

        worker_args = thread_class.call_args.kwargs["args"]
        bound = inspect.signature(
            self.controller._invent_comic_field_worker
        ).bind(*worker_args)
        self.assertEqual(bound.arguments["field"], "dialogue_direction")
        self.assertIn(
            "short caveman grammar",
            bound.arguments["context"]["dialogue_direction"],
        )

        with mock.patch(
            "krea_prompt_gui.chat_completion",
            return_value=(
                "Use blunt one-clause speech, basic words, grunted rhythm, "
                "and no modern slang."
            ),
        ):
            self.controller._invent_comic_field_worker(*worker_args)
        self.application.processEvents()

        self.assertIn(
            "grunted rhythm",
            self.controller.comic_dialogue_direction_var.get(),
        )
        self.controller.active_request_id += 1
        self.controller._show_invented_comic_field(
            self.controller.active_request_id,
            "concepts",
            "megalithic geometry, cave pigments",
        )
        self.assertIn("megalithic geometry", self.controller.comic_concepts_var.get())
        self.controller.clear_comic_story()
        self.assertEqual(self.controller.comic_dialogue_direction_var.get(), "")
        self.assertEqual(self.controller.comic_concepts_var.get(), "")
        self.assertEqual(self.controller.comic_visual_direction_var.get(), "")

    def test_meme_creator_accepts_either_caption_position_on_its_own(self):
        self.controller.meme_scene_var.set("a sleepy cat beside an alarm clock")
        self.controller.meme_top_text_var.set("MONDAYS")

        top_only = self.controller._meme_inputs()

        self.assertIn('top caption at the upper edge reading exactly "MONDAYS"', top_only)
        self.assertNotIn("bottom caption", top_only)
        self.assertIn("the exact quoted caption", top_only)

        self.controller.meme_top_text_var.set("")
        self.controller.meme_bottom_text_var.set("FIVE MORE MINUTES")

        bottom_only = self.controller._meme_inputs()

        self.assertNotIn("top caption", bottom_only)
        self.assertIn(
            'bottom caption at the lower edge reading exactly "FIVE MORE MINUTES"',
            bottom_only,
        )
        self.assertIn("the exact quoted caption", bottom_only)

        self.controller.meme_bottom_text_var.set("")
        with self.assertRaisesRegex(ValueError, "top text, bottom text, or a response"):
            self.controller._meme_inputs()

    def test_meme_response_brief_can_invent_scene_and_captions(self):
        self.controller.mode_tabs.setCurrentIndex(2)
        self.controller.meme_preset_var.set("Deadpan Irony")
        self.controller.meme_response_context_var.set(
            'The manager promised "no more meetings" and then scheduled three meetings about it.'
        )
        self.controller.meme_response_goal_var.set(
            "Respond with friendly disbelief about the contradiction"
        )
        self.controller.meme_temperature_var.set(0.42)

        draft = self.controller._meme_inputs()

        self.assertIn("Create an original meme response tailored specifically", draft)
        self.assertIn("manager promised ”no more meetings”", draft)
        self.assertNotIn('"no more meetings"', draft)
        self.assertIn("Intended response or stance: Respond with friendly disbelief", draft)
        self.assertIn("Invent the clearest, funniest underlying visual scene", draft)
        self.assertIn("Invent either one concise top caption", draft)
        self.assertIn("exact wording in straight double quotes", draft)
        self.assertIn("Humor tone: Ironic", draft)

        snapshot = self.controller._settings_snapshot()
        self.assertIn("scheduled three meetings", snapshot["meme_response_context"])
        self.assertIn("friendly disbelief", snapshot["meme_response_goal"])

        with mock.patch("krea_prompt_gui.threading.Thread") as thread_class:
            self.controller.correct_meme()

        worker_args = thread_class.call_args.kwargs["args"]
        bound = inspect.signature(self.controller._correct_prompt_worker).bind(*worker_args)
        self.assertEqual(bound.arguments["destination"], "meme")
        self.assertEqual(bound.arguments["temperature"], 0.42)
        self.assertEqual(bound.arguments["story_elements"], "")
        self.assertIn("original meme response", bound.arguments["draft"])

    def test_meme_caption_buttons_generate_one_field_from_current_context(self):
        self.controller.mode_tabs.setCurrentIndex(2)
        self.controller.meme_response_context_var.set(
            "A manager scheduled another meeting about reducing meetings."
        )
        self.controller.meme_response_goal_var.set("Friendly disbelief")
        self.controller.meme_scene_var.set(
            "An exhausted employee trapped inside nested conference rooms"
        )
        self.controller.meme_tone_var.set("Dry observational")
        self.controller.meme_focus_var.set("the employee's exhausted stare")
        self.controller.meme_temperature_var.set(0.28)
        self.controller.meme_top_text_var.set("WE FORMED A COMMITTEE")
        self.controller.meme_bottom_text_var.set("REDUCE THE MEETINGS")

        with mock.patch("krea_prompt_gui.threading.Thread") as thread_class:
            self.controller.invent_meme_caption("bottom")

        worker_args = thread_class.call_args.kwargs["args"]
        bound = inspect.signature(
            self.controller._invent_meme_caption_worker
        ).bind(*worker_args)
        self.assertEqual(bound.arguments["position"], "bottom")
        self.assertIn("manager scheduled", bound.arguments["response_context"])
        self.assertEqual(
            bound.arguments["other_caption"],
            "WE FORMED A COMMITTEE",
        )
        self.assertEqual(
            bound.arguments["current_caption"],
            "REDUCE THE MEETINGS",
        )
        self.assertEqual(
            bound.arguments["focus"],
            "the employee's exhausted stare",
        )
        self.assertTrue(self.controller.request_in_progress)
        self.assertFalse(self.controller.meme_top_caption_button.isEnabled())
        self.assertFalse(self.controller.meme_bottom_caption_button.isEnabled())
        self.assertTrue(self.controller.meme_stop_button.isEnabled())

        with mock.patch(
            "krea_prompt_gui.chat_completion",
            return_value='BOTTOM TEXT: "TO REDUCE COMMITTEES 产品摄影"',
        ) as completion:
            self.controller._invent_meme_caption_worker(*worker_args)
        self.application.processEvents()

        messages = completion.call_args.kwargs["messages"]
        self.assertEqual(completion.call_args.kwargs["temperature"], 0.28)
        self.assertEqual(
            completion.call_args.kwargs["ttl"],
            gui.CREATIVE_SESSION_TTL_SECONDS,
        )
        self.assertIn("manager scheduled", messages[1]["content"])
        self.assertIn("employee's exhausted stare", messages[1]["content"])
        self.assertIn("WE FORMED A COMMITTEE", messages[1]["content"])
        self.assertIn("Current bottom caption: REDUCE THE MEETINGS", messages[1]["content"])
        self.assertIn("mandatory creative seed", messages[0]["content"])
        self.assertEqual(
            self.controller.meme_bottom_text_var.get(),
            "TO REDUCE COMMITTEES",
        )
        self.assertFalse(self.controller.request_in_progress)
        self.assertTrue(self.controller.meme_top_caption_button.isEnabled())
        self.assertTrue(self.controller.meme_bottom_caption_button.isEnabled())

    def test_meme_caption_recall_restores_caption_from_before_invent(self):
        self.controller.meme_bottom_text_var.set("MY ORIGINAL CAPTION")
        with mock.patch("krea_prompt_gui.threading.Thread") as thread_class:
            self.controller.invent_meme_caption("bottom")
        request_id = thread_class.call_args.kwargs["args"][0]

        self.controller._show_invented_meme_caption(
            request_id,
            "bottom",
            "THE GENERATED CAPTION",
        )
        recall_button = self.controller.invent_recall_buttons["meme:bottom"]
        self.assertTrue(recall_button.isEnabled())

        recall_button.click()
        self.application.processEvents()

        self.assertEqual(
            self.controller.meme_bottom_text_var.get(),
            "MY ORIGINAL CAPTION",
        )
        self.assertFalse(recall_button.isEnabled())

    def test_every_meme_text_field_has_an_invent_button(self):
        self.controller.mode_tabs.setCurrentIndex(2)
        self.assertEqual(len(self.controller.meme_invent_buttons), 7)
        self.controller.meme_response_context_var.set(
            "A manager scheduled another meeting about reducing meetings."
        )
        self.controller.meme_response_goal_var.set("Dry disbelief")
        self.controller.meme_focus_var.set("the impossible doorway")
        self.controller.meme_scene_var.set(
            "A tired employee opens a suspicious conference-room door."
        )
        self.controller.meme_temperature_var.set(0.46)
        self.controller.meme_top_text_var.set("THIS COULD BE AN EMAIL")

        with mock.patch("krea_prompt_gui.threading.Thread") as thread_class:
            self.controller.invent_meme_field("scene")

        worker_args = thread_class.call_args.kwargs["args"]
        bound = inspect.signature(
            self.controller._invent_meme_field_worker
        ).bind(*worker_args)
        self.assertEqual(bound.arguments["field"], "scene")
        self.assertIn(
            "manager scheduled",
            bound.arguments["context"]["response_context"],
        )
        self.assertEqual(
            bound.arguments["context"]["top_caption"],
            "THIS COULD BE AN EMAIL",
        )
        self.assertEqual(
            bound.arguments["context"]["focus"],
            "the impossible doorway",
        )
        self.assertIn(
            "tired employee opens",
            bound.arguments["context"]["scene"],
        )
        self.assertTrue(all(not button.isEnabled() for button in self.controller.meme_invent_buttons))

        with mock.patch(
            "krea_prompt_gui.chat_completion",
            return_value=(
                "Scene: An employee opens one conference-room door\u2014revealing "
                "another conference room. 产品摄影。"
            ),
        ) as completion:
            self.controller._invent_meme_field_worker(*worker_args)
        self.application.processEvents()

        self.assertEqual(completion.call_args.kwargs["temperature"], 0.46)
        self.assertEqual(
            completion.call_args.kwargs["ttl"],
            gui.CREATIVE_SESSION_TTL_SECONDS,
        )
        self.assertIn("Field to invent: underlying visual scene", completion.call_args.kwargs["messages"][1]["content"])
        self.assertIn(
            "mandatory creative seed",
            completion.call_args.kwargs["messages"][0]["content"],
        )
        self.assertIn(
            "Current field value: A tired employee opens a suspicious conference-room door.",
            completion.call_args.kwargs["messages"][1]["content"],
        )
        self.assertEqual(
            self.controller.meme_scene_var.get(),
            "An employee opens one conference-room door, revealing another conference room.",
        )
        self.assertTrue(all(button.isEnabled() for button in self.controller.meme_invent_buttons))

    def test_meme_presets_include_sarcasm_and_irony_and_apply_full_style(self):
        self.assertIn("Sarcastic", gui.MEME_TONES)
        self.assertIn("Ironic", gui.MEME_TONES)
        self.assertIn("Classic Sarcasm", gui.MEME_PRESETS)
        self.assertIn("Deadpan Irony", gui.MEME_PRESETS)

        self.controller.meme_preset_var.set("Classic Sarcasm")

        self.assertEqual(self.controller.meme_tone_var.get(), "Sarcastic")
        self.assertEqual(
            self.controller.meme_caption_style_var.get(),
            "Classic bold white with black outline",
        )
        self.assertEqual(self.controller.meme_aspect_ratio_var.get(), "1:1 square")
        self.assertIn("reaction image", self.controller.meme_visual_direction_var.get())

        self.controller.meme_scene_var.set("a manager celebrating beside a broken server")
        self.controller.meme_bottom_text_var.set("ANOTHER PERFECT DEPLOYMENT")
        sarcastic_draft = self.controller._meme_inputs()
        self.assertIn("Humor tone: Sarcastic", sarcastic_draft)

        self.controller.meme_preset_var.set("Deadpan Irony")
        self.assertEqual(self.controller.meme_tone_var.get(), "Ironic")
        self.assertEqual(
            self.controller.meme_caption_style_var.get(),
            "Clean bold sans-serif",
        )
        ironic_draft = self.controller._meme_inputs()
        self.assertIn("Humor tone: Ironic", ironic_draft)
        self.assertIn("quietly contradicts", ironic_draft)

        snapshot = self.controller._settings_snapshot()
        self.assertEqual(snapshot["meme_preset"], "Deadpan Irony")
        self.assertEqual(snapshot["meme_tone"], "Ironic")

    def test_meme_workspace_starts_worker_and_routes_result_separately(self):
        self.controller.mode_tabs.setCurrentIndex(2)
        self.controller.meme_scene_var.set("a tired programmer staring at one red error")
        self.controller.meme_top_text_var.set("ONE LAST CHANGE")
        self.controller.meme_bottom_text_var.set("FAMOUS LAST WORDS")

        with mock.patch("krea_prompt_gui.threading.Thread") as thread_class:
            self.controller.correct_meme()

        worker_args = thread_class.call_args.kwargs["args"]
        bound = inspect.signature(self.controller._correct_prompt_worker).bind(*worker_args)
        self.assertEqual(bound.arguments["content_format"], "Meme")
        self.assertEqual(bound.arguments["destination"], "meme")
        self.assertIn('"ONE LAST CHANGE"', bound.arguments["draft"])
        self.assertIn('"FAMOUS LAST WORDS"', bound.arguments["draft"])
        self.assertEqual(bound.arguments["story_elements"], "")

        request_id = self.controller.active_request_id
        self.controller._show_corrected(
            request_id,
            bound.arguments["draft"],
            "A finished meme prompt.",
            "meme",
            "",
        )
        self.assertEqual(
            self.controller.meme_result_text.toPlainText(),
            "A finished meme prompt.",
        )
        self.assertEqual(self.controller.corrected_text.toPlainText(), "")
        self.assertEqual(self.controller.prompt_history[0]["content_format"], "Meme")

    def test_meme_workspace_does_not_inherit_prompt_corrector_content(self):
        self.controller.mode_tabs.setCurrentIndex(2)
        self.controller.meme_response_context_var.set(
            "A passenger behaved badly on a flight"
        )
        self.controller.meme_response_goal_var.set("Respond with disgust")
        self.controller.concepts_var.set("meme, comedy, unrelated ice cream")
        self.controller.concept_mix_var.set("unrelated watercolor:100%")
        self.controller.goal_headline_var.set("ice cream in harsh weather")
        self.controller.focus_var.set("licking an ice cream cone")
        self.controller.weighted_terms_var.set("storm:1.8")
        self.controller.model_instructions_var.set("Add an unrelated beach.")
        self.controller.generation_feedback_var.set("Reuse the previous ice cream image.")
        self.controller.reference_images_var.set(True)

        with (
            mock.patch.object(
                self.controller,
                "_local_reference_candidates",
                return_value=[{"title": "ice-cream.png", "url": "file:///tmp/ice-cream.png"}],
            ) as local_references,
            mock.patch("krea_prompt_gui.threading.Thread") as thread_class,
        ):
            self.controller.correct_meme()

        local_references.assert_not_called()
        worker_args = thread_class.call_args.kwargs["args"]
        bound = inspect.signature(self.controller._correct_prompt_worker).bind(*worker_args)
        self.assertEqual(bound.arguments["destination"], "meme")
        self.assertEqual(bound.arguments["concepts"], "")
        self.assertEqual(bound.arguments["goal_headline"], "")
        self.assertEqual(bound.arguments["focus"], "")
        self.assertEqual(bound.arguments["weighted_terms"], "")
        self.assertEqual(bound.arguments["model_instructions"], "")
        self.assertEqual(bound.arguments["generation_feedback"], "")
        self.assertFalse(bound.arguments["reference_image_analysis"])
        self.assertEqual(bound.arguments["local_reference_candidates"], [])
        activity = self.controller.activity_text.toPlainText()
        self.assertNotIn("ice cream in harsh weather", activity)
        self.assertNotIn("unrelated watercolor", activity)

    def test_grounded_research_asks_model_then_web_then_reconciles_before_correction(self):
        self.controller.draft_text.setPlainText(
            "A samurai draws a katana beside a torii gate, low-angle camera, warm sunset lighting."
        )
        self.controller.concepts_var.set("Edo armor")
        self.controller.live_research_var.set(True)
        self.controller.audit_repair_var.set(False)
        with mock.patch("krea_prompt_gui.threading.Thread") as thread_class:
            self.controller.correct_prompt()
        worker_args = thread_class.call_args.kwargs["args"]

        order = []

        def probe(**_kwargs):
            order.append("probe")
            return "TARGET | object | katana | known | curved sword"

        def targets(*_args, **_kwargs):
            order.append("targets")
            return [{"category": "object", "term": "katana", "confidence": "known", "knowledge": "curved sword"}]

        def web(*_args, **_kwargs):
            order.append("web")
            return "Targeted web verification results: katana evidence"

        def reconcile(**_kwargs):
            order.append("reconcile")
            return "Verified visual facts: katana construction and drawing action"

        def correct(**kwargs):
            order.append("correct")
            self.assertIn("Grounded concept glossary and factual verification only", kwargs["research_context"])
            self.assertIn("Verified visual facts", kwargs["research_context"])
            self.assertNotIn("Raw targeted web evidence", kwargs["research_context"])
            return "A corrected samurai prompt."

        with mock.patch("krea_prompt_gui.probe_model_visual_knowledge", side_effect=probe):
            with mock.patch("krea_prompt_gui.prompt_research_targets", side_effect=targets):
                with mock.patch("krea_prompt_gui.collect_targeted_prompt_research", side_effect=web):
                    with mock.patch("krea_prompt_gui.reconcile_model_knowledge_with_web", side_effect=reconcile):
                        with mock.patch("krea_prompt_gui.post_chat_completion", side_effect=correct):
                            self.controller._correct_prompt_worker(*worker_args)
        self.application.processEvents()

        self.assertEqual(order, ["probe", "targets", "web", "reconcile", "correct"])

    def test_visual_preset_catalogs_are_comprehensive_and_unique(self):
        catalogs = (
            gui.CAMERA_CONTROL_PRESETS,
            gui.MEME_CAPTION_STYLES,
            gui.MEME_ASPECT_RATIOS,
            gui.MEME_TONES,
            gui.COMIC_LAYOUT_PRESETS,
            gui.COMIC_READING_ORDER_PRESETS,
            gui.COMIC_ASPECT_RATIO_PRESETS,
        )
        for catalog in catalogs:
            self.assertEqual(len(catalog), len(set(catalog)))

        self.assertGreaterEqual(len(gui.CAMERA_CONTROL_PRESETS), 100)
        for camera in (
            "Eye-level medium shot, 50mm lens",
            "Extreme low worm's-eye view, 14mm lens",
            "Isometric orthographic view, no lens distortion",
            "Anamorphic wide shot, 40mm lens, 2.39:1 framing",
            "Security-camera high corner view, ultra-wide lens",
            "Underwater wide shot, dome-port 16mm lens",
        ):
            self.assertIn(camera, gui.CAMERA_CONTROL_PRESETS)

        self.assertGreaterEqual(len(gui.MEME_PRESETS), 20)
        self.assertGreaterEqual(len(gui.MEME_CAPTION_STYLES), 20)
        self.assertGreaterEqual(len(gui.MEME_TONES), 20)
        self.assertGreaterEqual(len(gui.COMIC_LAYOUT_PRESETS), 20)
        self.assertGreaterEqual(len(gui.COMIC_ASPECT_RATIO_PRESETS), 12)
        for name, preset in gui.MEME_PRESETS.items():
            if name == "Custom":
                self.assertEqual(preset, {})
                continue
            self.assertIn(preset["tone"], gui.MEME_TONES)
            self.assertIn(preset["caption_style"], gui.MEME_CAPTION_STYLES)
            self.assertIn(preset["aspect_ratio"], gui.MEME_ASPECT_RATIOS)
            self.assertTrue(preset["visual_direction"].strip())

        self.controller.meme_preset_var.set("Mock News")
        self.assertEqual(self.controller.meme_tone_var.get(), "Satirical")
        self.assertEqual(
            self.controller.meme_caption_style_var.get(),
            "News chyron lower third",
        )
        self.assertEqual(self.controller.meme_aspect_ratio_var.get(), "16:9 landscape")


if __name__ == "__main__":
    unittest.main()
