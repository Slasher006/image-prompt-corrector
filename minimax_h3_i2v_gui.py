#!/usr/bin/env python3
"""MiniMax H3 image-to-video workspace and pure prompt/state helpers."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import re

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from entity_resolution import (
    extract_entity_mentions,
    reference_ambiguities,
    resolve_references,
    rewrite_high_confidence_references,
)


H3_RESOLUTION_PRESETS = {
    "480p Landscape (864 x 480)": (864, 480),
    "480p Portrait (480 x 864)": (480, 864),
    "480p Square (480 x 480)": (480, 480),
    "Native Landscape (1344 x 768)": (1344, 768),
    "Native Portrait (768 x 1344)": (768, 1344),
    "Native Square (768 x 768)": (768, 768),
}
DEFAULT_H3_RESOLUTION = "480p Landscape (864 x 480)"
H3_IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.webp *.bmp)"


def h3_resolution_dimensions(value: object) -> tuple[int, int]:
    return H3_RESOLUTION_PRESETS.get(str(value or ""), H3_RESOLUTION_PRESETS[DEFAULT_H3_RESOLUTION])


def normalize_minimax_h3_i2v_state(value: object) -> dict[str, object]:
    source = value if isinstance(value, dict) else {}
    resolution = str(source.get("resolution", DEFAULT_H3_RESOLUTION))
    if resolution not in H3_RESOLUTION_PRESETS:
        resolution = DEFAULT_H3_RESOLUTION
    try:
        duration = max(1.0, min(15.0, float(source.get("duration", 5.0))))
    except (TypeError, ValueError):
        duration = 5.0
    try:
        seed = max(0, min(9_223_372_036_854_775_807, int(source.get("seed", 42))))
    except (TypeError, ValueError):
        seed = 42
    return {
        "server_url": str(source.get("server_url", "http://127.0.0.1:8188")).strip()
        or "http://127.0.0.1:8188",
        "first_frame": str(source.get("first_frame", "")).strip(),
        "last_frame": str(source.get("last_frame", "")).strip(),
        "scene": str(source.get("scene", "")).strip(),
        "motion": str(source.get("motion", "")).strip(),
        "audio": str(source.get("audio", "")).strip(),
        "prepared_prompt": str(source.get("prepared_prompt", "")).strip(),
        "duration": duration,
        "resolution": resolution,
        "seed": seed,
    }


def build_minimax_h3_i2v_prompt(
    *,
    scene: str,
    motion: str,
    audio: str = "",
    duration: float = 5.0,
    has_last_frame: bool = False,
) -> str:
    """Build H3's natural multimodal prompt without sampler-control leakage."""

    scene = re.sub(r"\s+", " ", str(scene or "")).strip(" .")
    motion = re.sub(r"\s+", " ", str(motion or "")).strip(" .")
    audio = re.sub(r"\s+", " ", str(audio or "")).strip(" .")
    seconds = max(1.0, min(15.0, float(duration)))
    duration_text = f"{seconds:g} seconds"
    parts = ["Use <Picture 1> as the exact first frame."]
    if scene:
        parts.append(
            "Preserve Picture 1's subject identity, geometry, wardrobe, objects, and setting. "
            + scene
            + "."
        )
    if motion:
        parts.append(f"Over {duration_text}, {motion[0].lower() + motion[1:] if len(motion) > 1 else motion.lower()}.")
    else:
        parts.append(f"Over {duration_text}, animate only coherent natural motion while preserving continuity.")
    if has_last_frame:
        parts.append("Transition coherently and end exactly on <Picture 2> as the final frame.")
    if audio:
        parts.append("Audio: " + audio + ".")
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def minimax_h3_i2v_prompt_issues(
    prompt: str,
    *,
    has_last_frame: bool = False,
    check_references: bool = True,
) -> list[str]:
    cleaned = str(prompt or "").strip()
    issues: list[str] = []
    if not cleaned:
        return ["MiniMax H3 prompt is empty"]
    if not re.search(r"(?i)(?:<Picture\s*1>|image\s*1|first\s+frame)", cleaned):
        issues.append("prompt does not bind the exact first frame")
    if has_last_frame and not re.search(
        r"(?i)(?:<Picture\s*2>|image\s*2|last\s+frame|final\s+frame)", cleaned
    ):
        issues.append("prompt does not bind the supplied last frame")
    if re.search(r"(?i)\b(?:negative\s+prompt|cfg|sampler|steps|denoise)\s*[:=]", cleaned):
        issues.append("sampler controls must stay outside the H3 natural-language prompt")
    if check_references:
        ambiguous = reference_ambiguities(
            resolve_references(extract_entity_mentions(cleaned, "Candidate"))
        )
        if ambiguous:
            issues.append(
                "unresolved entity references: "
                + ", ".join(
                    f'"{item.source.text}"' for item in ambiguous[:4]
                )
            )
    if len(cleaned) > 6000:
        issues.append("prompt exceeds the 6000-character workspace limit")
    return issues


def enforce_minimax_h3_i2v_prompt_contract(
    prompt: str,
    *,
    has_last_frame: bool = False,
) -> str:
    """Add only missing keyframe bindings; reject leaked technical controls."""

    cleaned = re.sub(r"\s+", " ", str(prompt or "")).strip()
    if not cleaned:
        return ""
    if re.search(r"(?i)\b(?:negative\s+prompt|cfg|sampler|steps|denoise)\s*[:=]", cleaned):
        return cleaned
    if not re.search(r"(?i)(?:<Picture\s*1>|image\s*1|first\s+frame)", cleaned):
        cleaned = "Use <Picture 1> as the exact first frame. " + cleaned
    if has_last_frame and not re.search(
        r"(?i)(?:<Picture\s*2>|image\s*2|last\s+frame|final\s+frame)", cleaned
    ):
        cleaned = cleaned.rstrip(" .") + ". End exactly on <Picture 2> as the final frame."
    return rewrite_high_confidence_references(cleaned, "Candidate")


class MiniMaxH3I2VWidget(QWidget):
    """Persistent local-ComfyUI MiniMax H3 first/last-frame workspace."""

    def __init__(
        self,
        *,
        state: object = None,
        current_prompt: Callable[[], str],
        send_to_comfyui: Callable[[str, str, list[str], dict[str, object]], None],
        run_llm: Callable[[str, dict[str, object]], None],
        stop_llm: Callable[[], None],
        save_state: Callable[[], None],
    ) -> None:
        super().__init__()
        self._state = normalize_minimax_h3_i2v_state(state)
        self._current_prompt = current_prompt
        self._send_to_comfyui = send_to_comfyui
        self._run_llm = run_llm
        self._stop_llm = stop_llm
        self._save_state = save_state
        self._model_buttons: list[QPushButton] = []
        self._build_ui()
        self._restore_state()

    def _button(self, label: str, callback: Callable, *, primary: bool = False) -> QPushButton:
        button = QPushButton(label)
        if primary:
            button.setObjectName("primaryButton")
        button.clicked.connect(callback)
        return button

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 10, 8, 8)
        intro = QLabel(
            "Animate an exact first frame with local MiniMax H3 in ComfyUI. "
            "An optional last frame constrains the ending; H3 generates video and native stereo audio together."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color:#9aa6bd")
        outer.addWidget(intro)

        connection = QHBoxLayout()
        connection.addWidget(QLabel("ComfyUI"))
        self.server_url = QLineEdit()
        self.server_url.setPlaceholderText("http://127.0.0.1:8188")
        connection.addWidget(self.server_url, 1)
        outer.addLayout(connection)

        frames = QGroupBox("I2V keyframes")
        frame_layout = QHBoxLayout(frames)
        self.first_preview, first_column = self._frame_column("First frame (required)", True)
        self.last_preview, last_column = self._frame_column("Last frame (optional)", False)
        frame_layout.addWidget(first_column, 1)
        frame_layout.addWidget(last_column, 1)
        outer.addWidget(frames)

        direction = QGroupBox("Video direction")
        form = QFormLayout(direction)
        self.scene = QTextEdit()
        self.scene.setMaximumHeight(85)
        self.scene.setPlaceholderText("Stable scene, subject, material, lighting, and continuity facts")
        self.motion = QTextEdit()
        self.motion.setMaximumHeight(85)
        self.motion.setPlaceholderText("Subject motion, camera movement, shot changes, timing, and final action")
        self.audio = QLineEdit()
        self.audio.setPlaceholderText("Dialogue, ambience, sound effects, or music; leave blank for no instruction")
        form.addRow("Scene continuity", self.scene)
        form.addRow("Motion and camera", self.motion)
        form.addRow("Native stereo audio", self.audio)
        outer.addWidget(direction)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Duration"))
        self.duration = QDoubleSpinBox()
        self.duration.setRange(1.0, 15.0)
        self.duration.setDecimals(1)
        self.duration.setSuffix(" s")
        controls.addWidget(self.duration)
        controls.addWidget(QLabel("Resolution"))
        self.resolution = QComboBox()
        self.resolution.addItems(tuple(H3_RESOLUTION_PRESETS))
        controls.addWidget(self.resolution, 1)
        controls.addWidget(QLabel("Seed"))
        self.seed = QSpinBox()
        self.seed.setRange(0, 2_147_483_647)
        controls.addWidget(self.seed)
        outer.addLayout(controls)

        prompt_group = QGroupBox("MiniMax H3 I2V prompt")
        prompt_layout = QVBoxLayout(prompt_group)
        buttons = QHBoxLayout()
        buttons.addWidget(self._button("Use main result", self.use_main_result))
        buttons.addWidget(self._button("Prepare", self.prepare_prompt))
        invent = self._button("Invent motion with images", lambda: self.request_llm("invent"))
        correct = self._button("Correct H3 prompt", lambda: self.request_llm("correct"), primary=True)
        self._model_buttons.extend((invent, correct))
        buttons.addWidget(invent)
        buttons.addWidget(correct)
        self.stop_button = self._button("Stop", self._stop_llm)
        self.stop_button.setEnabled(False)
        buttons.addWidget(self.stop_button)
        buttons.addStretch()
        prompt_layout.addLayout(buttons)
        self.prepared_prompt = QTextEdit()
        self.prepared_prompt.setPlaceholderText(
            "One natural-language H3 block describing first-frame continuity, motion, shots, and audio."
        )
        prompt_layout.addWidget(self.prepared_prompt, 1)
        outer.addWidget(prompt_group, 1)

        send_row = QHBoxLayout()
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color:#9aa6bd")
        send_row.addWidget(self.status_label, 1)
        send_row.addWidget(self._button("Send H3 I2V to ComfyUI", self.send, primary=True))
        outer.addLayout(send_row)

    def _frame_column(self, title: str, first: bool) -> tuple[QLabel, QWidget]:
        column = QWidget()
        layout = QVBoxLayout(column)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel(title))
        preview = QLabel("No image selected")
        preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview.setMinimumHeight(150)
        preview.setStyleSheet("border:1px solid #3a4354;color:#7f8ba3")
        layout.addWidget(preview, 1)
        row = QHBoxLayout()
        row.addWidget(self._button("Choose image...", lambda: self.choose_frame(first)))
        row.addWidget(self._button("Clear", lambda: self.clear_frame(first)))
        row.addStretch()
        layout.addLayout(row)
        return preview, column

    def _restore_state(self) -> None:
        self.server_url.setText(str(self._state["server_url"]))
        self.first_frame = str(self._state["first_frame"])
        self.last_frame = str(self._state["last_frame"])
        self.scene.setPlainText(str(self._state["scene"]))
        self.motion.setPlainText(str(self._state["motion"]))
        self.audio.setText(str(self._state["audio"]))
        self.prepared_prompt.setPlainText(str(self._state["prepared_prompt"]))
        self.duration.setValue(float(self._state["duration"]))
        self.resolution.setCurrentText(str(self._state["resolution"]))
        self.seed.setValue(min(2_147_483_647, int(self._state["seed"])))
        self._refresh_frame_preview(True)
        self._refresh_frame_preview(False)

    def snapshot(self) -> dict[str, object]:
        return normalize_minimax_h3_i2v_state(
            {
                "server_url": self.server_url.text(),
                "first_frame": self.first_frame,
                "last_frame": self.last_frame,
                "scene": self.scene.toPlainText(),
                "motion": self.motion.toPlainText(),
                "audio": self.audio.text(),
                "prepared_prompt": self.prepared_prompt.toPlainText(),
                "duration": self.duration.value(),
                "resolution": self.resolution.currentText(),
                "seed": self.seed.value(),
            }
        )

    def choose_frame(self, first: bool) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose MiniMax H3 keyframe", "", H3_IMAGE_FILTER)
        if not path:
            return
        if first:
            self.first_frame = path
        else:
            self.last_frame = path
        self._refresh_frame_preview(first)
        self._save_state()

    def clear_frame(self, first: bool) -> None:
        if first:
            self.first_frame = ""
        else:
            self.last_frame = ""
        self._refresh_frame_preview(first)
        self._save_state()

    def _refresh_frame_preview(self, first: bool) -> None:
        path = self.first_frame if first else self.last_frame
        label = self.first_preview if first else self.last_preview
        pixmap = QPixmap(path) if path and Path(path).is_file() else QPixmap()
        if pixmap.isNull():
            label.setPixmap(QPixmap())
            label.setText(Path(path).name if path else "No image selected")
            return
        label.setText("")
        label.setPixmap(
            pixmap.scaled(520, 210, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        )
        label.setToolTip(path)

    def use_main_result(self) -> None:
        value = str(self._current_prompt() or "").strip()
        if not value:
            QMessageBox.warning(self, "No main result", "Create a corrected still-image prompt first.")
            return
        self.scene.setPlainText(value)
        self.prepare_prompt()

    def prepare_prompt(self) -> None:
        value = build_minimax_h3_i2v_prompt(
            scene=self.scene.toPlainText(),
            motion=self.motion.toPlainText(),
            audio=self.audio.text(),
            duration=self.duration.value(),
            has_last_frame=bool(self.last_frame),
        )
        self.prepared_prompt.setPlainText(value)
        self.status_label.setText("Prepared H3 I2V prompt")
        self._save_state()

    def request_llm(self, action: str) -> None:
        if not self.first_frame or not Path(self.first_frame).is_file():
            self.set_model_error("Choose an available first-frame image first.")
            return
        if action == "correct" and not self.prepared_prompt.toPlainText().strip():
            self.prepare_prompt()
        self._run_llm(action, self.snapshot())

    def apply_llm_result(self, result: str) -> None:
        self.prepared_prompt.setPlainText(str(result or "").strip())
        self.status_label.setText("H3 prompt ready")
        self._save_state()

    def set_model_running(self, running: bool) -> None:
        for button in self._model_buttons:
            button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self.status_label.setText("Running selected vision model..." if running else "Ready")

    def set_model_error(self, message: str) -> None:
        self.status_label.setText(str(message or "MiniMax H3 model request failed"))

    def set_send_result(self, message: str, *, error: bool = False) -> None:
        self.status_label.setText(message)
        self.status_label.setStyleSheet("color:#e06c75" if error else "color:#9aa6bd")

    def send(self) -> None:
        prompt = self.prepared_prompt.toPlainText().strip()
        issues = minimax_h3_i2v_prompt_issues(prompt, has_last_frame=bool(self.last_frame))
        if not self.first_frame or not Path(self.first_frame).is_file():
            issues.insert(0, "first-frame image is missing or unavailable")
        if self.last_frame and not Path(self.last_frame).is_file():
            issues.append("last-frame image is unavailable")
        if issues:
            QMessageBox.warning(self, "MiniMax H3 I2V is not ready", "\n".join(f"- {issue}" for issue in issues))
            return
        width, height = h3_resolution_dimensions(self.resolution.currentText())
        paths = [self.first_frame] + ([self.last_frame] if self.last_frame else [])
        self._send_to_comfyui(
            self.server_url.text().strip(),
            prompt,
            paths,
            {
                "duration": self.duration.value(),
                "width": width,
                "height": height,
                "seed": self.seed.value(),
            },
        )
