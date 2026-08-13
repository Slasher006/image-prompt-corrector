import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSize
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication

from flux_image_edit_gui import (
    FLUX_REFERENCE_ROLE_PRESETS,
    FluxImageEditWindow,
    MAX_FLUX_EDIT_REFERENCES,
    encode_mask_png,
    masked_reference_transport_path,
    normalize_flux_image_edit_state,
)


class FluxImageEditWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.paths = []
        for index in range(1, 4):
            path = Path(self.temp_dir.name) / f"reference-{index}.png"
            path.write_bytes(b"not-a-real-png")
            self.paths.append(str(path))
        self.sent = []
        self.llm_calls = []
        self.stopped = []
        self.window = FluxImageEditWindow(
            state=None,
            current_prompt=lambda: "Current corrected prompt",
            send_to_comfyui=lambda url, prompt, paths: self.sent.append(
                (url, prompt, paths)
            ),
            run_llm=lambda action, state: self.llm_calls.append((action, state)),
            stop_llm=lambda: self.stopped.append(True),
            save_state=lambda: None,
        )

    def tearDown(self):
        self.window.close()
        self.temp_dir.cleanup()

    def test_window_supports_at_least_three_and_up_to_eight_references(self):
        self.assertEqual(MAX_FLUX_EDIT_REFERENCES, 8)
        for index, path in enumerate(self.paths, start=1):
            self.window._append_reference(path, f"role {index}")
        self.assertEqual(self.window.reference_paths(), self.paths)
        self.assertEqual(
            self.window.reference_roles(),
            ["role 1", "role 2", "role 3"],
        )
        self.assertGreaterEqual(len(FLUX_REFERENCE_ROLE_PRESETS), 10)

    def test_prepare_and_send_carries_three_references(self):
        for index, path in enumerate(self.paths, start=1):
            self.window._append_reference(path, f"role {index}")
        self.window.edit_instruction.setPlainText("Combine the three references.")
        prompt = self.window.prepare_prompt()
        self.assertIn("Image 3: role 3.", prompt)
        self.window.send()
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(self.sent[0][2], self.paths)

    def test_state_normalization_keeps_existing_reference_files_only(self):
        state = normalize_flux_image_edit_state(
            {
                "references": [
                    {"path": self.paths[0], "role": "base"},
                    {"path": "/missing.png", "role": "missing"},
                ]
            }
        )
        self.assertEqual(
            state["references"],
            [{"path": self.paths[0], "role": "base", "mask_png": ""}],
        )

    def test_reload_latest_output_replaces_base_but_keeps_other_references(self):
        output = Path(self.temp_dir.name) / "output"
        output.mkdir()
        older = output / "older.png"
        latest = output / "latest.webp"
        older.write_bytes(b"older")
        latest.write_bytes(b"latest")
        os.utime(older, (1, 1))
        os.utime(latest, (2, 2))
        self.window._append_reference(self.paths[0], "base")
        self.window._append_reference(self.paths[1], "identity")
        self.window.output_folder.setText(str(output))

        self.window.reload_latest_output()

        self.assertEqual(self.window.reference_paths()[0], str(latest))
        self.assertEqual(self.window.reference_paths()[1], self.paths[1])
        self.assertEqual(
            self.window.reference_roles()[0],
            "Base image / preserve overall composition",
        )

    def test_mask_transport_copy_exposes_painted_area_through_alpha(self):
        source_path = Path(self.temp_dir.name) / "mask-source.png"
        source = QImage(QSize(12, 8), QImage.Format.Format_RGB32)
        source.fill(QColor("#5b84d7"))
        self.assertTrue(source.save(str(source_path), "PNG"))
        mask = QImage(QSize(12, 8), QImage.Format.Format_Grayscale8)
        mask.fill(255)

        transport = masked_reference_transport_path(
            str(source_path),
            encode_mask_png(mask),
            index=1,
        )

        self.assertNotEqual(transport, str(source_path))
        loaded = QImage(transport)
        self.assertFalse(loaded.isNull())
        self.assertEqual(loaded.pixelColor(4, 4).alpha(), 0)

    def test_output_folder_and_mask_persist_in_edit_state(self):
        source_path = Path(self.temp_dir.name) / "persistent-source.png"
        source = QImage(QSize(10, 6), QImage.Format.Format_RGB32)
        source.fill(QColor("#d7855b"))
        self.assertTrue(source.save(str(source_path), "PNG"))
        self.window._append_reference(str(source_path), "base")
        mask = QImage(QSize(10, 6), QImage.Format.Format_Grayscale8)
        mask.fill(255)
        self.window._reference_masks[str(source_path)] = mask
        self.window.output_folder.setText(str(Path(self.temp_dir.name) / "output"))

        snapshot = self.window.snapshot()
        restored = normalize_flux_image_edit_state(snapshot)

        self.assertEqual(
            restored["output_folder"],
            str(Path(self.temp_dir.name) / "output"),
        )
        self.assertTrue(restored["references"][0]["mask_png"])

    def test_analyze_images_calls_llm_with_paths_and_role_presets(self):
        for path in self.paths:
            self.window._append_reference(path)
        self.window.request_llm("analyze")
        self.assertEqual(len(self.llm_calls), 1)
        action, state = self.llm_calls[0]
        self.assertEqual(action, "analyze")
        self.assertEqual(len(state["references"]), 3)
        self.assertEqual(
            state["references"][0]["role"],
            "Base image / preserve overall composition",
        )

    def test_llm_result_can_be_recalled_and_cleared(self):
        self.window.edit_instruction.setPlainText("My original edit")
        self.window.apply_llm_result("invent_instruction", "Invented edit")
        self.assertEqual(
            self.window.edit_instruction.toPlainText(),
            "Invented edit",
        )
        self.assertTrue(self.window._recall_buttons["instruction"].isEnabled())
        self.window.recall_field("instruction")
        self.assertEqual(
            self.window.edit_instruction.toPlainText(),
            "My original edit",
        )
        self.window.clear_field("instruction")
        self.assertEqual(self.window.edit_instruction.toPlainText(), "")


if __name__ == "__main__":
    unittest.main()
