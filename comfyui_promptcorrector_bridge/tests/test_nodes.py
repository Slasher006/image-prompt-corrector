import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    from comfyui_promptcorrector_bridge import nodes
except ModuleNotFoundError:
    node_path = Path(__file__).resolve().parents[1] / "nodes.py"
    spec = importlib.util.spec_from_file_location(
        "promptcorrector_bridge_nodes",
        node_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load bridge nodes from {node_path}")
    nodes = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nodes)


class PromptCorrectorBridgeTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.settings_path = Path(self.temp_dir.name) / "settings.json"

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_settings(self, payload):
        self.settings_path.write_text(json.dumps(payload), encoding="utf-8")

    def test_latest_result_uses_first_valid_history_entry(self):
        self.write_settings(
            {
                "prompt_history": [
                    {
                        "workspace": "meme",
                        "corrected_prompt": "Newest meme prompt",
                        "created_at": "2026-07-20 12:00",
                    },
                    {
                        "workspace": "prompt",
                        "corrected_prompt": "Older image prompt",
                    },
                ]
            }
        )

        result = nodes.read_promptcorrector_result(
            settings_path=self.settings_path
        )

        self.assertEqual(result["prompt"], "Newest meme prompt")
        self.assertEqual(result["source"], "Meme Creator")

    def test_workspace_selection_does_not_cross_into_another_workspace(self):
        self.write_settings(
            {
                "prompt_history": [
                    {"workspace": "meme", "corrected_prompt": "Meme"},
                    {"workspace": "prompt", "corrected_prompt": "Image"},
                ],
                "comic_result": "Saved comic",
            }
        )

        comic = nodes.read_promptcorrector_result(
            "Comic Story",
            settings_path=self.settings_path,
        )
        prompt = nodes.read_promptcorrector_result(
            "Prompt Corrector",
            settings_path=self.settings_path,
        )

        self.assertEqual(comic["prompt"], "Saved comic")
        self.assertEqual(prompt["prompt"], "Image")

    def test_environment_override_selects_settings_file(self):
        self.write_settings({"corrected_prompt": "Configured prompt"})

        with patch.dict(
            os.environ,
            {"PROMPTCORRECTOR_SETTINGS_PATH": str(self.settings_path)},
        ):
            result = nodes.read_promptcorrector_result()

        self.assertEqual(result["prompt"], "Configured prompt")

    def test_node_metadata_does_not_expose_saved_prompt(self):
        prompt_options = nodes.PromptCorrectorBridge.INPUT_TYPES()[
            "required"
        ]["prompt"][1]

        self.assertEqual(prompt_options["default"], "")
        self.assertIn(
            "prompt corrector",
            nodes.PromptCorrectorBridge.SEARCH_ALIASES,
        )

    def test_invalid_settings_raise_actionable_error(self):
        self.settings_path.write_text("{bad json", encoding="utf-8")

        with self.assertRaisesRegex(
            nodes.PromptCorrectorBridgeError,
            "could not be read",
        ):
            nodes.read_promptcorrector_result(
                settings_path=self.settings_path
            )

    def test_transfer_refreshes_or_preserves_displayed_text(self):
        node = nodes.PromptCorrectorBridge()
        refreshed = {
            "prompt": "Fresh corrected result",
            "source": "Prompt Corrector",
            "created_at": "",
            "settings_updated_ns": "1",
        }
        with patch.object(
            nodes,
            "read_promptcorrector_result",
            return_value=refreshed,
        ):
            self.assertEqual(
                node.transfer("Stale text", "Refresh on queue"),
                ("Fresh corrected result", "Prompt Corrector"),
            )
        self.assertEqual(
            node.transfer("Edited in ComfyUI", "Use displayed text"),
            ("Edited in ComfyUI", "Displayed text"),
        )

    def test_refresh_falls_back_to_visible_text_when_settings_are_unavailable(self):
        node = nodes.PromptCorrectorBridge()
        with patch.object(
            nodes,
            "read_promptcorrector_result",
            side_effect=nodes.PromptCorrectorBridgeError("offline"),
        ):
            self.assertEqual(
                node.transfer("Visible fallback", "Refresh on queue"),
                ("Visible fallback", "Displayed text fallback"),
            )


if __name__ == "__main__":
    unittest.main()
