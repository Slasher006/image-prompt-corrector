import asyncio
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import comfyui_promptcorrector_bridge as bridge_package
    from comfyui_promptcorrector_bridge import nodes
except ModuleNotFoundError:
    bridge_package = None
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

    def test_browser_waits_for_node_update_before_queueing_workflow(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "web"
            / "promptcorrector_bridge.js"
        ).read_text(encoding="utf-8")

        self.assertIn("const QUEUE_AFTER_UPDATE_DELAY_MS = 500;", script)
        wait_index = script.index("await waitForBridgeUpdate();")
        queue_index = script.index("await app.queuePrompt();")
        self.assertLess(wait_index, queue_index)
        self.assertIn("await promptWidget.callback?.(payload.prompt);", script)

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

    def test_push_payload_is_validated_without_unrelated_state(self):
        result = nodes.validate_bridge_push_payload(
            {
                "prompt": "  Visible corrected prompt  ",
                "workspace": "Prompt Corrector",
                "ignored": {"private": "setting"},
            }
        )

        self.assertEqual(
            result,
            {
                "prompt": "Visible corrected prompt",
                "workspace": "Prompt Corrector",
                "source": "Prompt Corrector",
            },
        )

    def test_push_payload_rejects_empty_or_unknown_workspace(self):
        with self.assertRaisesRegex(
            nodes.PromptCorrectorBridgeError,
            "empty",
        ):
            nodes.validate_bridge_push_payload(
                {"prompt": "", "workspace": "Prompt Corrector"}
            )
        with self.assertRaisesRegex(
            nodes.PromptCorrectorBridgeError,
            "Unsupported",
        ):
            nodes.validate_bridge_push_payload(
                {"prompt": "Result", "workspace": "Latest result"}
            )

    def test_push_payload_accepts_only_boolean_queue_request(self):
        result = nodes.validate_bridge_push_payload(
            {
                "prompt": "Queued result",
                "workspace": "Prompt Corrector",
                "queue_after_send": True,
            }
        )
        self.assertTrue(result["queue_after_send"])
        with self.assertRaisesRegex(
            nodes.PromptCorrectorBridgeError,
            "true or false",
        ):
            nodes.validate_bridge_push_payload(
                {
                    "prompt": "Queued result",
                    "workspace": "Prompt Corrector",
                    "queue_after_send": "yes",
                }
            )

    @unittest.skipIf(
        bridge_package is None,
        "Route registration is tested from the repository package layout.",
    )
    def test_push_route_broadcasts_validated_prompt_event(self):
        registered = {}

        class Routes:
            @staticmethod
            def get(path):
                return lambda handler: registered.setdefault(("GET", path), handler)

            @staticmethod
            def post(path):
                return lambda handler: registered.setdefault(("POST", path), handler)

        instance = types.SimpleNamespace(
            routes=Routes(),
            send_sync=lambda event, payload: registered.setdefault(
                ("EVENT", event),
                payload,
            ),
        )
        server_module = types.ModuleType("server")
        server_module.PromptServer = types.SimpleNamespace(instance=instance)
        aiohttp_module = types.ModuleType("aiohttp")
        aiohttp_module.web = types.SimpleNamespace(
            json_response=lambda payload, status=200: (payload, status)
        )

        with patch.dict(
            sys.modules,
            {
                "server": server_module,
                "aiohttp": aiohttp_module,
            },
        ):
            bridge_package._register_latest_prompt_route()

        handler = registered[
            ("POST", "/promptcorrector_bridge/push")
        ]

        class Request:
            async def json(self):
                return {
                    "prompt": "Pushed result",
                    "workspace": "Meme Creator",
                    "queue_after_send": True,
                }

        response, status = asyncio.run(handler(Request()))

        self.assertEqual(status, 200)
        self.assertEqual(response["characters"], len("Pushed result"))
        self.assertTrue(response["queue_requested"])
        self.assertEqual(
            registered[("EVENT", "promptcorrector_bridge_prompt")],
            {
                "prompt": "Pushed result",
                "workspace": "Meme Creator",
                "source": "Meme Creator",
                "queue_after_send": True,
            },
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
