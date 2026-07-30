#!/usr/bin/env python3
"""Separate FLUX.2 Klein multi-reference image-edit window."""

from __future__ import annotations

import base64
import hashlib
import tempfile
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QBuffer, QIODevice, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from prompt_workbench import build_image_edit_prompt


MAX_FLUX_EDIT_REFERENCES = 8
FLUX_REFERENCE_ROLE_PRESETS = (
    "Base image / preserve overall composition",
    "Subject identity and face",
    "Body, pose, and expression",
    "Outfit and accessories",
    "Object or prop",
    "Background and environment",
    "Visual style and medium",
    "Lighting and color palette",
    "Texture and material",
    "Text, logo, or layout",
)
FLUX_LLM_ACTIONS = (
    "analyze",
    "correct",
    "invent_instruction",
    "invent_preserve",
)
FLUX_RECALL_FIELDS = ("instruction", "preserve", "analysis", "prepared_prompt")
FLUX_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
MAX_SAVED_MASK_CHARACTERS = 16_000_000


def encode_mask_png(mask: QImage) -> str:
    if mask.isNull():
        return ""
    grayscale = mask.convertToFormat(QImage.Format.Format_Grayscale8)
    if not any(bytes(grayscale.constBits())[:grayscale.sizeInBytes()]):
        return ""
    buffer = QBuffer()
    if not buffer.open(QIODevice.OpenModeFlag.WriteOnly):
        return ""
    try:
        if not mask.save(buffer, "PNG"):
            return ""
        raw = bytes(buffer.data())
    finally:
        buffer.close()
    return base64.b64encode(raw).decode("ascii")


def decode_mask_png(value: str, size) -> QImage:
    mask = QImage()
    if value:
        try:
            mask.loadFromData(base64.b64decode(value, validate=True), "PNG")
        except (ValueError, TypeError):
            mask = QImage()
    if mask.isNull():
        mask = QImage(size, QImage.Format.Format_Grayscale8)
        mask.fill(0)
    elif mask.size() != size:
        mask = mask.scaled(
            size,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
    return mask.convertToFormat(QImage.Format.Format_Grayscale8)


def latest_flux_output_image(folder: str) -> str:
    root = Path(str(folder or "")).expanduser()
    if not root.is_dir():
        return ""
    candidates: list[tuple[float, Path]] = []
    try:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.casefold() in FLUX_IMAGE_SUFFIXES:
                try:
                    candidates.append((path.stat().st_mtime, path))
                except OSError:
                    continue
    except OSError:
        return ""
    return str(max(candidates, default=(0.0, Path()))[1]) if candidates else ""


def masked_reference_transport_path(
    path: str,
    mask_png: str,
    *,
    index: int,
) -> str:
    """Create an RGBA transport copy whose alpha yields the ComfyUI edit mask."""

    if not mask_png:
        return path
    source = QImage(path)
    if source.isNull():
        raise ValueError(f"Could not load reference Image {index}: {path}")
    mask = decode_mask_png(mask_png, source.size())
    alpha = mask.copy()
    alpha.invertPixels()
    rgba = source.convertToFormat(QImage.Format.Format_ARGB32)
    rgba.setAlphaChannel(alpha)
    digest = hashlib.sha256(
        (str(Path(path).resolve()) + mask_png).encode("utf-8")
    ).hexdigest()[:16]
    cache = Path(tempfile.gettempdir()) / "promptcorrector-flux-masks"
    cache.mkdir(parents=True, exist_ok=True)
    transport = cache / f"reference_{index}_{digest}.png"
    if not transport.is_file() and not rgba.save(str(transport), "PNG"):
        raise ValueError(
            f"Could not prepare the mask for reference Image {index}."
        )
    return str(transport)


class FluxMaskCanvas(QWidget):
    """Small paint/erase canvas for a single reference-image edit mask."""

    maskChanged = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._source = QImage()
        self._mask = QImage()
        self._painting = False
        self._erase = False
        self._brush_size = 64
        self._last_source_point: tuple[float, float] | None = None
        self.setMinimumSize(360, 260)
        self.setMouseTracking(True)

    def set_reference(self, path: str, mask_png: str = "") -> None:
        self._source = QImage(path)
        if self._source.isNull():
            self._mask = QImage()
        else:
            self._mask = decode_mask_png(mask_png, self._source.size())
        self.update()

    def set_brush_size(self, size: int) -> None:
        self._brush_size = max(4, min(512, int(size)))

    def set_erase(self, erase: bool) -> None:
        self._erase = bool(erase)

    def clear_mask(self) -> None:
        if self._mask.isNull():
            return
        self._mask.fill(0)
        self.maskChanged.emit()
        self.update()

    def invert_mask(self) -> None:
        if self._mask.isNull():
            return
        self._mask.invertPixels()
        self.maskChanged.emit()
        self.update()

    def mask_image(self) -> QImage:
        return self._mask.copy()

    def _target_rect(self) -> QRectF:
        if self._source.isNull():
            return QRectF()
        scaled = self._source.size()
        scaled.scale(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        left = (self.width() - scaled.width()) / 2
        top = (self.height() - scaled.height()) / 2
        return QRectF(left, top, scaled.width(), scaled.height())

    def _source_point(self, position) -> tuple[float, float] | None:
        target = self._target_rect()
        if target.isEmpty() or not target.contains(position):
            return None
        return (
            (position.x() - target.left()) * self._source.width() / target.width(),
            (position.y() - target.top()) * self._source.height() / target.height(),
        )

    def _paint_at(self, position) -> None:
        point = self._source_point(position)
        if point is None or self._mask.isNull():
            return
        painter = QPainter(self._mask)
        pen = QPen(
            QColor(0 if self._erase else 255, 0 if self._erase else 255, 0 if self._erase else 255),
            self._brush_size,
            Qt.PenStyle.SolidLine,
            Qt.PenCapStyle.RoundCap,
            Qt.PenJoinStyle.RoundJoin,
        )
        painter.setPen(pen)
        if self._last_source_point is None:
            painter.drawPoint(round(point[0]), round(point[1]))
        else:
            painter.drawLine(
                round(self._last_source_point[0]),
                round(self._last_source_point[1]),
                round(point[0]),
                round(point[1]),
            )
        painter.end()
        self._last_source_point = point
        self.maskChanged.emit()
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._painting = True
            self._last_source_point = None
            self._paint_at(event.position())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._painting:
            self._paint_at(event.position())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._painting:
            self._painting = False
            self._last_source_point = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#090c12"))
        if self._source.isNull():
            painter.setPen(QColor("#7f8ba3"))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Select a reference image",
            )
            return
        target = self._target_rect()
        painter.drawImage(target, self._source)
        overlay = QImage(self._source.size(), QImage.Format.Format_ARGB32)
        overlay.fill(QColor("#ff4d67"))
        overlay.setAlphaChannel(self._mask)
        painter.setOpacity(0.55)
        painter.drawImage(target, overlay)
        painter.setOpacity(1.0)
        painter.setPen(QColor("#30384a"))
        painter.drawRect(target)


def normalize_flux_image_edit_state(value: object) -> dict[str, object]:
    state: dict[str, object] = {
        "server_url": "http://127.0.0.1:8188",
        "instruction": "",
        "preserve": "",
        "analysis": "",
        "prepared_prompt": "",
        "output_folder": "",
        "references": [],
        "recall": {},
    }
    if not isinstance(value, dict):
        return state
    state["server_url"] = (
        str(value.get("server_url", "")).strip() or str(state["server_url"])
    )
    for key in (
        "instruction",
        "preserve",
        "analysis",
        "prepared_prompt",
        "output_folder",
    ):
        state[key] = str(value.get(key, ""))
    recall = value.get("recall", {})
    if isinstance(recall, dict):
        state["recall"] = {
            key: str(recall[key])
            for key in FLUX_RECALL_FIELDS
            if key in recall
        }
    references: list[dict[str, str]] = []
    stored_references = value.get("references", [])
    if isinstance(stored_references, list):
        for entry in stored_references:
            if not isinstance(entry, dict):
                continue
            path = str(entry.get("path", "")).strip()
            if not path or not Path(path).is_file():
                continue
            mask_png = str(entry.get("mask_png", ""))
            references.append(
                {
                    "path": path,
                    "role": str(entry.get("role", "")).strip(),
                    "mask_png": (
                        mask_png
                        if len(mask_png) <= MAX_SAVED_MASK_CHARACTERS
                        else ""
                    ),
                }
            )
            if len(references) >= MAX_FLUX_EDIT_REFERENCES:
                break
    state["references"] = references
    return state


class FluxImageEditWidget(QWidget):
    """Embedded FLUX.2 Klein multi-reference image-edit workspace."""

    def __init__(
        self,
        *,
        state: object = None,
        current_prompt: Callable[[], str],
        send_to_comfyui: Callable[[str, str, list[str]], None],
        run_llm: Callable[[str, dict[str, object]], None],
        stop_llm: Callable[[], None],
        save_state: Callable[[], None],
    ) -> None:
        super().__init__()
        self._current_prompt = current_prompt
        self._send_to_comfyui = send_to_comfyui
        self._run_llm = run_llm
        self._stop_llm = stop_llm
        self._save_state = save_state
        self._state = normalize_flux_image_edit_state(state)
        self._recall_values = dict(self._state["recall"])
        self._reference_masks: dict[str, QImage] = {}
        self._model_buttons: list[QPushButton] = []
        self._recall_buttons: dict[str, QPushButton] = {}
        self._build_ui()
        self._restore_state()

    def _button(
        self,
        text: str,
        callback: Callable,
        *,
        primary: bool = False,
    ) -> QPushButton:
        button = QPushButton(text)
        if primary:
            button.setObjectName("primaryButton")
        button.clicked.connect(callback)
        return button

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 8, 0, 0)
        intro = QLabel(
            "Edit with up to eight role-aware references. Analyze or correct "
            "with the selected vision model, then send the prompt, images, and "
            "painted masks to ComfyUI."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color:#9aa6bd")
        outer.addWidget(intro)

        connection_row = QHBoxLayout()
        connection_row.addWidget(QLabel("ComfyUI"))
        self.server_url = QLineEdit()
        self.server_url.setPlaceholderText("http://127.0.0.1:8188")
        connection_row.addWidget(self.server_url, 1)
        outer.addLayout(connection_row)

        self.workspace_tabs = QTabWidget()
        edit_page = QWidget()
        edit_outer = QVBoxLayout(edit_page)
        edit_outer.setContentsMargins(8, 8, 8, 8)

        edit_group = QGroupBox("Edit request")
        edit_layout = QFormLayout(edit_group)
        self.edit_instruction = QTextEdit()
        self.edit_instruction.setPlaceholderText(
            "Describe the desired final result and exact changes."
        )
        self.edit_instruction.setMaximumHeight(95)
        self.preserve = QLineEdit()
        self.preserve.setPlaceholderText(
            "Identity, framing, geometry, text, lighting, or details to keep"
        )
        edit_layout.addRow(
            "Requested edit",
            self._field_controls(
                self.edit_instruction,
                field="instruction",
                invent_action="invent_instruction",
            ),
        )
        edit_layout.addRow(
            "Preserve",
            self._field_controls(
                self.preserve,
                field="preserve",
                invent_action="invent_preserve",
            ),
        )
        edit_outer.addWidget(edit_group)

        analysis_group = QGroupBox("Image understanding")
        analysis_layout = QVBoxLayout(analysis_group)
        analysis_buttons = QHBoxLayout()
        analyze_button = self._button(
            "Analyze references",
            lambda: self.request_llm("analyze"),
            primary=True,
        )
        self._model_buttons.append(analyze_button)
        analysis_buttons.addWidget(analyze_button)
        analysis_buttons.addWidget(self._make_recall_button("analysis"))
        analysis_buttons.addWidget(
            self._button("Clear", lambda: self.clear_field("analysis"))
        )
        analysis_buttons.addStretch()
        analysis_layout.addLayout(analysis_buttons)
        self.reference_analysis = QTextEdit()
        self.reference_analysis.setPlaceholderText(
            "Optional observations from the selected LM Studio vision model."
        )
        self.reference_analysis.setMaximumHeight(100)
        analysis_layout.addWidget(self.reference_analysis)
        edit_outer.addWidget(analysis_group)

        prompt_group = QGroupBox("FLUX edit prompt")
        prompt_layout = QVBoxLayout(prompt_group)
        prompt_buttons = QHBoxLayout()
        prompt_buttons.addWidget(
            self._button("Prepare", self.prepare_prompt)
        )
        correct_button = self._button(
            "Correct with images",
            lambda: self.request_llm("correct"),
            primary=True,
        )
        self._model_buttons.append(correct_button)
        prompt_buttons.addWidget(correct_button)
        prompt_buttons.addWidget(
            self._button(
                "Use main result",
                self.use_current_corrected_prompt,
            )
        )
        prompt_buttons.addWidget(self._make_recall_button("prepared_prompt"))
        prompt_buttons.addWidget(
            self._button("Clear", lambda: self.clear_field("prepared_prompt"))
        )
        self.stop_button = self._button("Stop", self._stop_llm)
        self.stop_button.setEnabled(False)
        prompt_buttons.addWidget(self.stop_button)
        prompt_buttons.addStretch()
        prompt_layout.addLayout(prompt_buttons)
        self.prepared_prompt = QTextEdit()
        self.prepared_prompt.setPlaceholderText(
            "The editable multi-reference FLUX prompt appears here."
        )
        prompt_layout.addWidget(self.prepared_prompt, 1)
        edit_outer.addWidget(prompt_group, 1)
        self.workspace_tabs.addTab(edit_page, "Edit")

        references_page = QWidget()
        references_outer = QVBoxLayout(references_page)
        references_outer.setContentsMargins(8, 8, 8, 8)

        output_row = QHBoxLayout()
        output_row.addWidget(QLabel("ComfyUI output folder"))
        self.output_folder = QLineEdit()
        self.output_folder.setPlaceholderText(
            "Choose the ComfyUI output folder for iterative edits"
        )
        output_row.addWidget(self.output_folder, 1)
        output_row.addWidget(
            self._button("Choose…", self.choose_output_folder)
        )
        output_row.addWidget(
            self._button(
                "Reload latest output as base",
                self.reload_latest_output,
                primary=True,
            )
        )
        references_outer.addLayout(output_row)

        split = QSplitter(Qt.Orientation.Horizontal)
        references = QGroupBox("Reference images")
        references_layout = QVBoxLayout(references)
        reference_buttons = QHBoxLayout()
        reference_buttons.addWidget(
            self._button("Add image(s)", self.add_reference_images, primary=True)
        )
        reference_buttons.addWidget(
            self._button("Remove selected", self.remove_selected_reference)
        )
        reference_buttons.addWidget(
            self._button("Clear all", self.clear_references)
        )
        references_layout.addLayout(reference_buttons)
        self.reference_table = QTableWidget(0, 3)
        self.reference_table.setHorizontalHeaderLabels(["#", "Image", "Role in edit"])
        self.reference_table.verticalHeader().setVisible(False)
        self.reference_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self.reference_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Stretch
        )
        self.reference_table.setColumnWidth(0, 42)
        self.reference_table.itemSelectionChanged.connect(
            self._refresh_reference_preview
        )
        references_layout.addWidget(self.reference_table, 1)
        references_layout.addWidget(
            QLabel(
                "Choose a role preset or type a custom role. Image 1 is the "
                "base replaced by Reload latest output."
            )
        )
        split.addWidget(references)

        mask_group = QGroupBox("Mask selected reference")
        mask_layout = QVBoxLayout(mask_group)
        self.mask_canvas = FluxMaskCanvas()
        self.mask_canvas.maskChanged.connect(self._store_current_mask)
        mask_layout.addWidget(self.mask_canvas, 1)
        mask_tools = QHBoxLayout()
        self.paint_mask_button = self._button("Paint", self._use_paint_tool)
        self.paint_mask_button.setCheckable(True)
        self.paint_mask_button.setChecked(True)
        self.erase_mask_button = self._button("Erase", self._use_erase_tool)
        self.erase_mask_button.setCheckable(True)
        mask_tools.addWidget(self.paint_mask_button)
        mask_tools.addWidget(self.erase_mask_button)
        mask_tools.addWidget(QLabel("Brush"))
        self.mask_brush_size = QSlider(Qt.Orientation.Horizontal)
        self.mask_brush_size.setRange(4, 512)
        self.mask_brush_size.setValue(64)
        self.mask_brush_size.setMinimumWidth(110)
        self.mask_brush_size.valueChanged.connect(
            self.mask_canvas.set_brush_size
        )
        mask_tools.addWidget(self.mask_brush_size, 1)
        mask_tools.addWidget(
            self._button("Invert", self.mask_canvas.invert_mask)
        )
        mask_tools.addWidget(
            self._button("Clear mask", self.mask_canvas.clear_mask)
        )
        mask_layout.addLayout(mask_tools)
        mask_hint = QLabel(
            "Paint the area FLUX may change. Unpainted pixels remain protected."
        )
        mask_hint.setWordWrap(True)
        mask_hint.setStyleSheet("color:#9aa6bd")
        mask_layout.addWidget(mask_hint)
        split.addWidget(mask_group)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 4)
        references_outer.addWidget(split, 1)
        self.workspace_tabs.addTab(references_page, "References && Mask")
        outer.addWidget(self.workspace_tabs, 1)

        send_row = QHBoxLayout()
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color:#9aa6bd")
        send_row.addWidget(self.status_label, 1)
        send_row.addWidget(
            self._button(
                "Send prompt + references to ComfyUI",
                self.send,
                primary=True,
            )
        )
        outer.addLayout(send_row)

    def _restore_state(self) -> None:
        self.server_url.setText(str(self._state["server_url"]))
        self.output_folder.setText(str(self._state["output_folder"]))
        self.edit_instruction.setPlainText(str(self._state["instruction"]))
        self.preserve.setText(str(self._state["preserve"]))
        self.reference_analysis.setPlainText(str(self._state["analysis"]))
        self.prepared_prompt.setPlainText(str(self._state["prepared_prompt"]))
        for reference in self._state["references"]:
            self._append_reference(
                str(reference["path"]),
                str(reference.get("role", "")),
                str(reference.get("mask_png", "")),
            )
        if self.reference_table.rowCount():
            self.reference_table.selectRow(0)
        self._refresh_recall_buttons()

    def _field_controls(
        self,
        editor: QWidget,
        *,
        field: str,
        invent_action: str,
    ) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(editor, 1)
        invent = self._button(
            "Invent",
            lambda: self.request_llm(invent_action),
        )
        self._model_buttons.append(invent)
        layout.addWidget(invent)
        layout.addWidget(self._make_recall_button(field))
        layout.addWidget(self._button("Clear", lambda: self.clear_field(field)))
        return container

    def _make_recall_button(self, field: str) -> QPushButton:
        button = self._button("Recall", lambda: self.recall_field(field))
        button.setEnabled(False)
        self._recall_buttons[field] = button
        return button

    def _append_reference(
        self,
        path: str,
        role: str = "",
        mask_png: str = "",
    ) -> None:
        row = self.reference_table.rowCount()
        self.reference_table.insertRow(row)
        number = QTableWidgetItem(str(row + 1))
        number.setFlags(number.flags() & ~Qt.ItemFlag.ItemIsEditable)
        path_item = QTableWidgetItem(path)
        path_item.setFlags(path_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.reference_table.setItem(row, 0, number)
        self.reference_table.setItem(row, 1, path_item)
        role_combo = QComboBox()
        role_combo.setEditable(True)
        role_combo.addItems(FLUX_REFERENCE_ROLE_PRESETS)
        if role:
            index = role_combo.findText(role)
            if index >= 0:
                role_combo.setCurrentIndex(index)
            else:
                role_combo.setEditText(role)
        else:
            role_combo.setCurrentIndex(min(row, len(FLUX_REFERENCE_ROLE_PRESETS) - 1))
        self.reference_table.setCellWidget(row, 2, role_combo)
        if mask_png:
            source = QImage(path)
            if not source.isNull():
                self._reference_masks[path] = decode_mask_png(
                    mask_png,
                    source.size(),
                )

    def add_reference_images(self) -> None:
        remaining = MAX_FLUX_EDIT_REFERENCES - self.reference_table.rowCount()
        if remaining <= 0:
            QMessageBox.information(
                self,
                "Reference limit reached",
                f"This window supports up to {MAX_FLUX_EDIT_REFERENCES} references.",
            )
            return
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Add FLUX edit reference images",
            "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp *.gif)",
        )
        existing = set(self.reference_paths())
        for path in paths:
            if path in existing:
                continue
            self._append_reference(path)
            existing.add(path)
            remaining -= 1
            if remaining <= 0:
                break
        if self.reference_table.rowCount():
            self.reference_table.selectRow(self.reference_table.rowCount() - 1)

    def remove_selected_reference(self) -> None:
        rows = sorted(
            {index.row() for index in self.reference_table.selectedIndexes()},
            reverse=True,
        )
        for row in rows:
            path_item = self.reference_table.item(row, 1)
            if path_item is not None:
                self._reference_masks.pop(path_item.text(), None)
            self.reference_table.removeRow(row)
        self._renumber_references()
        self._refresh_reference_preview()

    def clear_references(self) -> None:
        self.reference_table.setRowCount(0)
        self._reference_masks.clear()
        self._refresh_reference_preview()

    def _renumber_references(self) -> None:
        for row in range(self.reference_table.rowCount()):
            self.reference_table.item(row, 0).setText(str(row + 1))

    def _refresh_reference_preview(self) -> None:
        row = self.reference_table.currentRow()
        if row < 0:
            self.mask_canvas.set_reference("")
            return
        path_item = self.reference_table.item(row, 1)
        path = path_item.text() if path_item else ""
        mask = self._reference_masks.get(path, QImage())
        self.mask_canvas.set_reference(
            path,
            encode_mask_png(mask) if not mask.isNull() else "",
        )

    def _store_current_mask(self) -> None:
        row = self.reference_table.currentRow()
        if row < 0:
            return
        path_item = self.reference_table.item(row, 1)
        if path_item is None:
            return
        self._reference_masks[path_item.text()] = self.mask_canvas.mask_image()
        self.status_label.setText(
            f"Mask updated for Image {row + 1}"
        )
        self.status_label.setStyleSheet("color:#9aa6bd")

    def _use_paint_tool(self) -> None:
        self.paint_mask_button.setChecked(True)
        self.erase_mask_button.setChecked(False)
        self.mask_canvas.set_erase(False)

    def _use_erase_tool(self) -> None:
        self.paint_mask_button.setChecked(False)
        self.erase_mask_button.setChecked(True)
        self.mask_canvas.set_erase(True)

    def choose_output_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            "Choose ComfyUI output folder",
            self.output_folder.text().strip(),
        )
        if folder:
            self.output_folder.setText(folder)
            self._save_state()

    def reload_latest_output(self) -> None:
        folder = self.output_folder.text().strip()
        if not folder:
            self.choose_output_folder()
            folder = self.output_folder.text().strip()
        latest = latest_flux_output_image(folder)
        if not latest:
            QMessageBox.warning(
                self,
                "No output image found",
                "No PNG, JPEG, WebP, or BMP image was found in that output folder.",
            )
            return
        if self.reference_table.rowCount():
            path_item = self.reference_table.item(0, 1)
            old_path = path_item.text() if path_item is not None else ""
            self._reference_masks.pop(old_path, None)
            if path_item is not None:
                path_item.setText(latest)
            role_combo = self.reference_table.cellWidget(0, 2)
            if isinstance(role_combo, QComboBox):
                role_combo.setCurrentText(
                    FLUX_REFERENCE_ROLE_PRESETS[0]
                )
        else:
            self._append_reference(
                latest,
                FLUX_REFERENCE_ROLE_PRESETS[0],
            )
        self.reference_table.selectRow(0)
        self._refresh_reference_preview()
        self.workspace_tabs.setCurrentIndex(1)
        self.status_label.setText(
            f"Reloaded latest output as Image 1: {Path(latest).name}"
        )
        self.status_label.setStyleSheet("color:#8bd5a5")
        self._save_state()

    def reference_paths(self) -> list[str]:
        return [
            self.reference_table.item(row, 1).text()
            for row in range(self.reference_table.rowCount())
            if self.reference_table.item(row, 1) is not None
        ]

    def reference_roles(self) -> list[str]:
        roles: list[str] = []
        for row in range(self.reference_table.rowCount()):
            combo = self.reference_table.cellWidget(row, 2)
            roles.append(
                combo.currentText().strip()
                if isinstance(combo, QComboBox)
                else ""
            )
        return roles

    def prepare_prompt(self) -> str:
        masked_indices = [
            index
            for index, path in enumerate(self.reference_paths(), start=1)
            if encode_mask_png(self._reference_masks.get(path, QImage()))
        ]
        try:
            prompt = build_image_edit_prompt(
                instruction=self.edit_instruction.toPlainText(),
                preserve=self.preserve.text(),
                image_analysis=self.reference_analysis.toPlainText(),
                reference_roles=self.reference_roles(),
                masked_reference_indices=masked_indices,
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Incomplete image edit", str(exc))
            return ""
        self.prepared_prompt.setPlainText(prompt)
        self.status_label.setText("FLUX edit prompt prepared")
        self.status_label.setStyleSheet("color:#8bd5a5")
        return prompt

    def use_current_corrected_prompt(self) -> None:
        prompt = self._current_prompt().strip()
        if not prompt:
            QMessageBox.warning(
                self,
                "No corrected prompt",
                "Create or enter a corrected prompt in the main window first.",
            )
            return
        self.edit_instruction.setPlainText(prompt)
        self.prepare_prompt()

    def request_llm(self, action: str) -> None:
        if action not in FLUX_LLM_ACTIONS:
            return
        if not self.reference_paths():
            QMessageBox.warning(
                self,
                "No reference images",
                "Add at least one reference image before using the vision model.",
            )
            return
        if action == "correct" and not self.prepared_prompt.toPlainText().strip():
            if not self.prepare_prompt():
                return
        self._state = self.snapshot()
        self._save_state()
        self.set_model_running(True)
        self.status_label.setStyleSheet("color:#9aa6bd")
        labels = {
            "analyze": "Analyzing all reference images with the selected model...",
            "correct": "Correcting the FLUX edit prompt with image understanding...",
            "invent_instruction": "Inventing an edit from the reference images...",
            "invent_preserve": "Inspecting what the edit should preserve...",
        }
        self.status_label.setText(labels[action])
        self._run_llm(action, self.snapshot())

    def _field_value(self, field: str) -> str:
        return {
            "instruction": self.edit_instruction.toPlainText(),
            "preserve": self.preserve.text(),
            "analysis": self.reference_analysis.toPlainText(),
            "prepared_prompt": self.prepared_prompt.toPlainText(),
        }.get(field, "")

    def _set_field_value(self, field: str, value: str) -> None:
        if field == "instruction":
            self.edit_instruction.setPlainText(value)
        elif field == "preserve":
            self.preserve.setText(value)
        elif field == "analysis":
            self.reference_analysis.setPlainText(value)
        elif field == "prepared_prompt":
            self.prepared_prompt.setPlainText(value)

    def clear_field(self, field: str) -> None:
        self._set_field_value(field, "")
        self._save_state()
        self.status_label.setText(f"Cleared {field.replace('_', ' ')}")
        self.status_label.setStyleSheet("color:#9aa6bd")

    def recall_field(self, field: str) -> None:
        if field not in self._recall_values:
            return
        value = self._recall_values.pop(field)
        self._set_field_value(field, value)
        self._refresh_recall_buttons()
        self._save_state()
        self.status_label.setText(
            f"Recalled {field.replace('_', ' ')} from before the last LLM change"
        )
        self.status_label.setStyleSheet("color:#8bd5a5")

    def _refresh_recall_buttons(self) -> None:
        for field, button in self._recall_buttons.items():
            button.setEnabled(field in self._recall_values)

    def apply_llm_result(self, action: str, text: str) -> None:
        target = {
            "analyze": "analysis",
            "correct": "prepared_prompt",
            "invent_instruction": "instruction",
            "invent_preserve": "preserve",
        }.get(action)
        if target is None:
            return
        self._recall_values[target] = self._field_value(target)
        self._set_field_value(target, text.strip())
        if action in {"invent_instruction", "invent_preserve"}:
            self.prepare_prompt()
        self._refresh_recall_buttons()
        self.set_model_running(False)
        self.status_label.setText(
            {
                "analyze": "Reference-image analysis complete",
                "correct": "FLUX edit prompt corrected with image understanding",
                "invent_instruction": "Edit request invented from the images",
                "invent_preserve": "Preservation constraints invented from the images",
            }[action]
        )
        self.status_label.setStyleSheet("color:#8bd5a5")
        self._save_state()

    def set_model_running(self, running: bool) -> None:
        for button in self._model_buttons:
            button.setEnabled(not running)
        self.stop_button.setEnabled(running)

    def set_model_error(self, message: str) -> None:
        self.set_model_running(False)
        self.status_label.setText(message)
        self.status_label.setStyleSheet("color:#ff8b8b")

    def snapshot(self) -> dict[str, object]:
        references = [
            {
                "path": path,
                "role": role,
                "mask_png": encode_mask_png(
                    self._reference_masks.get(path, QImage())
                ),
            }
            for path, role in zip(self.reference_paths(), self.reference_roles())
        ]
        return {
            "server_url": self.server_url.text().strip()
            or "http://127.0.0.1:8188",
            "output_folder": self.output_folder.text().strip(),
            "instruction": self.edit_instruction.toPlainText(),
            "preserve": self.preserve.text(),
            "analysis": self.reference_analysis.toPlainText(),
            "prepared_prompt": self.prepared_prompt.toPlainText(),
            "references": references,
            "recall": dict(self._recall_values),
        }

    def send(self) -> None:
        paths = self.reference_paths()
        if not paths:
            QMessageBox.warning(
                self,
                "No reference images",
                "Add at least one reference image. You can use up to eight.",
            )
            return
        prompt = self.prepared_prompt.toPlainText().strip() or self.prepare_prompt()
        if not prompt:
            return
        state = self.snapshot()
        try:
            transport_paths = [
                masked_reference_transport_path(
                    str(reference["path"]),
                    str(reference.get("mask_png", "")),
                    index=index,
                )
                for index, reference in enumerate(
                    state["references"],
                    start=1,
                )
            ]
        except ValueError as exc:
            QMessageBox.warning(self, "Could not prepare mask", str(exc))
            return
        self._state = state
        self._save_state()
        self.status_label.setText(
            f"Sending prompt and {len(paths)} reference image(s)..."
        )
        self._send_to_comfyui(
            self.server_url.text().strip(),
            prompt,
            transport_paths,
        )

    def set_send_result(self, message: str, *, error: bool = False) -> None:
        self.status_label.setText(message)
        self.status_label.setStyleSheet(
            "color:#ff8b8b" if error else "color:#8bd5a5"
        )

    def closeEvent(self, event) -> None:
        self._state = self.snapshot()
        self._save_state()
        event.accept()


# Compatibility import for older callers while the editor now lives in a tab.
FluxImageEditWindow = FluxImageEditWidget
