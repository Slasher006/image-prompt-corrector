#!/usr/bin/env python3
"""Qt workbench pages for project-based prompt iteration."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from krea_prompt_corrector import (
    EXPLICIT_ADULT_MODE_INSTRUCTION,
    chat_completion,
    make_prompt_safe_for_work,
    post_chat_completion,
    validate_explicit_adult_mode,
)
from prompt_workbench import (
    GENERATOR_PROFILE_DEFAULTS,
    REFERENCE_ROLES,
    add_reference,
    add_version,
    benchmark_model,
    character_continuity_issues,
    clarification_questions,
    composition_instruction,
    contract_dashboard,
    enqueue_comfyui,
    load_generator_profiles,
    load_project_bundle,
    new_project,
    normalize_project,
    parse_batch_csv,
    prompt_variant_instructions,
    review_generated_images,
    save_project_bundle,
    targeted_repair_prompt,
    text_layout_instruction,
    utc_now,
)


class CompositionCanvas(QWidget):
    """Small normalized-box editor for composition and crop planning."""

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(360)
        self.items: list[dict[str, object]] = []
        self.kind = "subject"
        self.label = "subject"
        self.panel: int | None = None
        self._start: QPointF | None = None
        self._preview: QRectF | None = None

    def set_tool(self, kind: str, label: str, panel: int) -> None:
        self.kind = kind.strip() or "subject"
        self.label = label.strip() or self.kind
        self.panel = panel if panel > 0 else None

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._start = event.position()
            self._preview = QRectF(self._start, self._start)
            self.update()

    def mouseMoveEvent(self, event) -> None:
        if self._start is not None:
            self._preview = QRectF(self._start, event.position()).normalized()
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if self._start is None or event.button() != Qt.MouseButton.LeftButton:
            return
        rect = QRectF(self._start, event.position()).normalized()
        self._start = None
        self._preview = None
        if rect.width() < 8 or rect.height() < 8 or self.width() <= 0 or self.height() <= 0:
            self.update()
            return
        self.items.append({
            "kind": self.kind,
            "label": self.label,
            "panel": self.panel,
            "x": round(rect.x() / self.width(), 4),
            "y": round(rect.y() / self.height(), 4),
            "width": round(rect.width() / self.width(), 4),
            "height": round(rect.height() / self.height(), 4),
        })
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#151a22"))
        painter.setPen(QPen(QColor("#30394a"), 1))
        for fraction in (0.333, 0.5, 0.666):
            painter.drawLine(int(self.width() * fraction), 0, int(self.width() * fraction), self.height())
            painter.drawLine(0, int(self.height() * fraction), self.width(), int(self.height() * fraction))
        for index, item in enumerate(self.items, start=1):
            rect = QRectF(
                float(item["x"]) * self.width(),
                float(item["y"]) * self.height(),
                float(item["width"]) * self.width(),
                float(item["height"]) * self.height(),
            )
            painter.fillRect(rect, QColor(109, 93, 252, 45))
            painter.setPen(QPen(QColor("#8174ff"), 2))
            painter.drawRect(rect)
            painter.drawText(rect.adjusted(6, 4, -4, -4), f"{index}. {item['label']}")
        if self._preview is not None:
            painter.setPen(QPen(QColor("#a99fff"), 2, Qt.PenStyle.DashLine))
            painter.drawRect(self._preview)

    def remove_last(self) -> None:
        if self.items:
            self.items.pop()
            self.update()

    def clear_items(self) -> None:
        self.items = []
        self.update()


class PromptWorkbench(QWidget):
    def __init__(self, controller: Any) -> None:
        super().__init__()
        self.controller = controller
        self.project = normalize_project(getattr(controller, "recovered_workbench_project", None))
        self.project_path = ""
        self.generator_profiles = load_generator_profiles(
            getattr(controller, "recovered_generator_profiles", None)
        )
        self.batch_rows: list[dict[str, str]] = []
        self.review_paths: list[str] = []
        self.mask_path = ""
        self.workflow_path = ""
        self.variant_results: list[dict[str, str]] = []
        self._build_ui()
        self.refresh_all()

    def snapshot(self) -> dict[str, object]:
        self.project["name"] = self.project_name.text().strip() or "Untitled project"
        self.project["updated_at"] = utc_now()
        return {
            "project": self.project,
            "project_path": self.project_path,
            "generator_profiles": self.generator_profiles,
        }

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        intro = QLabel(
            "Close the loop from prompt to generated image: keep projects, audit results, repair failures, "
            "manage continuity, compare variants, batch work, and hand prompts to external workflows."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color:#9aa6bd")
        outer.addWidget(intro)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_project_review_page(), "Project & Review")
        self.tabs.addTab(self._build_reference_page(), "References & Characters")
        self.tabs.addTab(self._build_contract_page(), "Contracts & A/B")
        self.tabs.addTab(self._build_composition_page(), "Composition & Text")
        self.tabs.addTab(self._build_batch_page(), "Batch")
        self.tabs.addTab(self._build_tools_page(), "Models & Integrations")
        outer.addWidget(self.tabs, 1)

    def _button(self, text: str, callback: Callable, primary: bool = False) -> QPushButton:
        button = QPushButton(text)
        if primary:
            button.setObjectName("primaryButton")
        button.clicked.connect(callback)
        return button

    def _build_project_review_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        project = QGroupBox("Portable project")
        row = QHBoxLayout(project)
        self.project_name = QLineEdit()
        self.project_name.setPlaceholderText("Project name")
        row.addWidget(self.project_name, 1)
        row.addWidget(self._button("New", self.new_project))
        row.addWidget(self._button("Sync current prompt", self.sync_current_prompt))
        row.addWidget(self._button("Open .ipcp", self.open_project))
        row.addWidget(self._button("Save .ipcp", self.save_project, True))
        layout.addWidget(project)

        split = QSplitter(Qt.Orientation.Horizontal)
        inputs = QWidget()
        left = QVBoxLayout(inputs)
        result_group = QGroupBox("Generated results")
        result_layout = QVBoxLayout(result_group)
        self.result_list = QListWidget()
        result_layout.addWidget(self.result_list)
        result_buttons = QHBoxLayout()
        result_buttons.addWidget(self._button("Add generated images", self.add_result_images))
        result_buttons.addWidget(self._button("Remove selected", self.remove_result_image))
        result_layout.addLayout(result_buttons)
        left.addWidget(result_group, 1)
        review_actions = QHBoxLayout()
        self.review_button = self._button("Audit selected/all results", self.start_result_review, True)
        review_actions.addWidget(self.review_button)
        review_actions.addWidget(self._button("Stop", self.controller.stop_current_request))
        review_actions.addWidget(self._button("Use repair as feedback", self.apply_review_repair))
        left.addLayout(review_actions)
        split.addWidget(inputs)

        review_group = QGroupBox("Visual audit and targeted repair")
        review_layout = QVBoxLayout(review_group)
        self.review_output = QTextEdit()
        self.review_output.setReadOnly(True)
        self.review_output.setPlaceholderText("Audit score, passed requirements, failures, warnings, and a minimal repair prompt appear here.")
        review_layout.addWidget(self.review_output)
        split.addWidget(review_group)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 3)
        layout.addWidget(split, 1)
        return page

    def _build_reference_page(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)
        references = QGroupBox("Role-aware references")
        ref_layout = QVBoxLayout(references)
        form = QFormLayout()
        self.reference_role = QComboBox()
        self.reference_role.addItems(REFERENCE_ROLES)
        self.reference_subject = QLineEdit()
        self.reference_subject.setPlaceholderText("Character or element this reference controls")
        self.reference_panel = QSpinBox()
        self.reference_panel.setRange(0, 12)
        self.reference_panel.setSpecialValueText("All panels")
        self.reference_crop = QLineEdit()
        self.reference_crop.setPlaceholderText("Optional normalized x,y,width,height, e.g. 0.1,0.2,0.5,0.6")
        self.reference_notes = QLineEdit()
        self.reference_notes.setPlaceholderText("What to preserve or ignore")
        form.addRow("Role", self.reference_role)
        form.addRow("Subject", self.reference_subject)
        form.addRow("Panel", self.reference_panel)
        form.addRow("Crop", self.reference_crop)
        form.addRow("Notes", self.reference_notes)
        ref_layout.addLayout(form)
        controls = QHBoxLayout()
        controls.addWidget(self._button("Add image(s)", self.add_role_references, True))
        controls.addWidget(self._button("Choose mask", self.choose_reference_mask))
        controls.addWidget(self._button("Remove", self.remove_reference))
        ref_layout.addLayout(controls)
        self.reference_list = QListWidget()
        ref_layout.addWidget(self.reference_list, 1)
        layout.addWidget(references, 3)

        characters = QGroupBox("Character bible and continuity")
        char_layout = QVBoxLayout(characters)
        char_form = QFormLayout()
        self.character_name = QLineEdit()
        self.character_anchors = QLineEdit()
        self.character_anchors.setPlaceholderText("red coat, brass satchel, green eyes")
        self.character_forbid = QLineEdit()
        self.character_forbid.setPlaceholderText("traits that must not drift")
        char_form.addRow("Name", self.character_name)
        char_form.addRow("Identity anchors", self.character_anchors)
        char_form.addRow("Forbidden drift", self.character_forbid)
        char_layout.addLayout(char_form)
        char_controls = QHBoxLayout()
        char_controls.addWidget(self._button("Add character", self.add_character, True))
        char_controls.addWidget(self._button("Remove", self.remove_character))
        char_layout.addLayout(char_controls)
        self.character_list = QListWidget()
        char_layout.addWidget(self.character_list, 1)
        char_layout.addWidget(self._button("Inspect current comic continuity", self.inspect_continuity))
        self.continuity_output = QTextEdit()
        self.continuity_output.setReadOnly(True)
        char_layout.addWidget(self.continuity_output, 1)
        layout.addWidget(characters, 2)
        return page

    def _build_contract_page(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)
        audit = QWidget()
        audit_layout = QVBoxLayout(audit)
        controls = QHBoxLayout()
        controls.addWidget(self._button("Inspect current prompt", self.inspect_contracts, True))
        controls.addWidget(self._button("Find essential questions", self.inspect_clarifications))
        audit_layout.addLayout(controls)
        self.contract_table = QTableWidget(0, 3)
        self.contract_table.setHorizontalHeaderLabels(["Status", "Contract", "Detail"])
        self.contract_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        audit_layout.addWidget(self.contract_table, 2)
        self.clarification_output = QTextEdit()
        self.clarification_output.setReadOnly(True)
        self.clarification_output.setPlaceholderText("Only ambiguities that materially change the composition are raised.")
        audit_layout.addWidget(self.clarification_output, 1)
        layout.addWidget(audit, 3)

        variants = QGroupBox("A/B prompt lab")
        variant_layout = QVBoxLayout(variants)
        variant_layout.addWidget(QLabel("Hard contracts remain locked while one visual dimension changes."))
        self.variant_list = QListWidget()
        variant_layout.addWidget(self.variant_list, 1)
        variant_controls = QHBoxLayout()
        variant_controls.addWidget(self._button("Prepare variants", self.prepare_variants))
        variant_controls.addWidget(self._button("Generate selected", self.generate_selected_variant, True))
        variant_controls.addWidget(self._button("Generate all", self.generate_all_variants))
        variant_controls.addWidget(self._button("Use selected winner", self.use_selected_variant))
        variant_layout.addLayout(variant_controls)
        self.variant_output = QTextEdit()
        self.variant_output.setReadOnly(True)
        variant_layout.addWidget(self.variant_output, 2)
        layout.addWidget(variants, 2)
        return page

    def _build_composition_page(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)
        composition = QGroupBox("Composition canvas")
        comp_layout = QVBoxLayout(composition)
        tool_row = QHBoxLayout()
        self.composition_kind = QComboBox()
        self.composition_kind.addItems(("subject", "prop", "text region", "environment", "focus", "exclusion"))
        self.composition_label = QLineEdit("subject")
        self.composition_panel = QSpinBox()
        self.composition_panel.setRange(0, 12)
        self.composition_panel.setSpecialValueText("Whole image")
        tool_row.addWidget(self.composition_kind)
        tool_row.addWidget(self.composition_label, 1)
        tool_row.addWidget(self.composition_panel)
        comp_layout.addLayout(tool_row)
        self.composition_canvas = CompositionCanvas()
        comp_layout.addWidget(self.composition_canvas, 1)
        buttons = QHBoxLayout()
        buttons.addWidget(self._button("Draw box", self.arm_composition_tool))
        buttons.addWidget(self._button("Undo box", self.composition_canvas.remove_last))
        buttons.addWidget(self._button("Clear", self.composition_canvas.clear_items))
        buttons.addWidget(self._button("Apply to model instructions", self.apply_composition, True))
        comp_layout.addLayout(buttons)
        self.composition_output = QTextEdit()
        self.composition_output.setReadOnly(True)
        self.composition_output.setMaximumHeight(130)
        comp_layout.addWidget(self.composition_output)
        layout.addWidget(composition, 3)

        text_group = QGroupBox("Exact text and speech bubbles")
        text_layout = QVBoxLayout(text_group)
        form = QFormLayout()
        self.text_exact = QLineEdit()
        self.text_speaker = QLineEdit()
        self.text_kind = QComboBox()
        self.text_kind.addItems(("speech bubble", "caption", "sign", "title", "sound effect"))
        self.text_panel = QSpinBox()
        self.text_panel.setRange(0, 12)
        self.text_panel.setSpecialValueText("Whole image")
        self.text_placement = QLineEdit("auto, without covering faces")
        form.addRow("Exact text", self.text_exact)
        form.addRow("Speaker", self.text_speaker)
        form.addRow("Kind", self.text_kind)
        form.addRow("Panel", self.text_panel)
        form.addRow("Placement", self.text_placement)
        text_layout.addLayout(form)
        text_controls = QHBoxLayout()
        text_controls.addWidget(self._button("Add", self.add_text_layout, True))
        text_controls.addWidget(self._button("Remove", self.remove_text_layout))
        text_layout.addLayout(text_controls)
        self.text_layout_list = QListWidget()
        text_layout.addWidget(self.text_layout_list, 1)
        text_layout.addWidget(self._button("Apply exact text contract", self.apply_text_layout))
        layout.addWidget(text_group, 2)
        return page

    def _build_batch_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        hint = QLabel("Paste CSV with a required prompt column and optional id, goal, and focus columns, or import a .csv file.")
        layout.addWidget(hint)
        self.batch_input = QTextEdit()
        self.batch_input.setPlaceholderText("id,prompt,goal,focus\n1,red robot in rain,cinematic portrait,robot face")
        self.batch_input.setMaximumHeight(130)
        layout.addWidget(self.batch_input)
        controls = QHBoxLayout()
        controls.addWidget(self._button("Import CSV", self.import_batch_csv))
        controls.addWidget(self._button("Load pasted queue", self.load_batch_queue))
        controls.addWidget(self._button("Run/resume", self.run_batch, True))
        controls.addWidget(self._button("Stop", self.controller.stop_current_request))
        controls.addWidget(self._button("Export results", self.export_batch_results))
        layout.addLayout(controls)
        self.batch_progress = QProgressBar()
        layout.addWidget(self.batch_progress)
        self.batch_table = QTableWidget(0, 4)
        self.batch_table.setHorizontalHeaderLabels(["ID", "Status", "Prompt", "Result / error"])
        self.batch_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.batch_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.batch_table, 1)
        return page

    def _build_tools_page(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)
        models = QGroupBox("LM Studio model benchmark")
        model_layout = QVBoxLayout(models)
        model_layout.addWidget(QLabel("Runs compact instruction, exact-text, spatial, and panel-fidelity probes."))
        self.benchmark_button = self._button("Benchmark loaded model", self.run_benchmark, True)
        model_layout.addWidget(self.benchmark_button)
        self.benchmark_output = QTextEdit()
        self.benchmark_output.setReadOnly(True)
        model_layout.addWidget(self.benchmark_output, 1)
        layout.addWidget(models, 1)

        integration = QGroupBox("Generator profiles and ComfyUI")
        integration_layout = QVBoxLayout(integration)
        profile_row = QHBoxLayout()
        self.profile_name = QComboBox()
        self.profile_name.setEditable(True)
        profile_row.addWidget(self.profile_name, 1)
        profile_row.addWidget(self._button("Load", self.load_profile_editor))
        profile_row.addWidget(self._button("Save profile", self.save_profile_editor))
        profile_row.addWidget(self._button("Apply", self.apply_profile_editor))
        integration_layout.addLayout(profile_row)
        self.profile_editor = QTextEdit()
        self.profile_editor.setPlaceholderText("Generator profile JSON")
        integration_layout.addWidget(self.profile_editor, 1)
        comfy_form = QFormLayout()
        self.comfy_url = QLineEdit("http://127.0.0.1:8188")
        self.comfy_node = QLineEdit()
        self.comfy_node.setPlaceholderText("Positive CLIP text node id")
        self.comfy_workflow_label = QLineEdit()
        self.comfy_workflow_label.setReadOnly(True)
        comfy_form.addRow("ComfyUI URL", self.comfy_url)
        comfy_form.addRow("Positive node", self.comfy_node)
        comfy_form.addRow("Workflow", self.comfy_workflow_label)
        integration_layout.addLayout(comfy_form)
        comfy_controls = QHBoxLayout()
        comfy_controls.addWidget(self._button("Choose workflow JSON", self.choose_comfy_workflow))
        comfy_controls.addWidget(self._button("Enqueue current prompt", self.send_to_comfyui, True))
        integration_layout.addLayout(comfy_controls)
        bridge_automation = QHBoxLayout()
        bridge_automation.addWidget(
            self.controller._bind_check(
                "Auto-send completed results",
                self.controller.comfyui_auto_send_var,
            )
        )
        bridge_automation.addWidget(
            self.controller._bind_check(
                "Queue workflow after sending",
                self.controller.comfyui_queue_after_send_var,
            )
        )
        bridge_automation.addStretch()
        integration_layout.addLayout(bridge_automation)
        self.integration_output = QTextEdit()
        self.integration_output.setReadOnly(True)
        self.integration_output.setMaximumHeight(120)
        integration_layout.addWidget(self.integration_output)
        layout.addWidget(integration, 1)
        return page

    def _current_texts(self) -> tuple[str, str]:
        if self.controller.content_format_var.get() == "Comic Story" and self.controller.comic_result_text is not None:
            try:
                draft, _story = self.controller._comic_story_inputs()
            except ValueError:
                draft = self.controller.comic_premise_var.get().strip()
            return draft, self.controller.comic_result_text.toPlainText().strip()
        return self.controller.draft_text.toPlainText().strip(), self.controller.corrected_text.toPlainText().strip()

    def refresh_all(self) -> None:
        self.project_name.setText(str(self.project.get("name", "Untitled project")))
        self.composition_canvas.items = list(self.project.get("composition", []))
        self.composition_canvas.update()
        self.refresh_results()
        self.refresh_references()
        self.refresh_characters()
        self.refresh_text_layout()
        self.profile_name.clear()
        self.profile_name.addItems(sorted(self.generator_profiles))
        self.load_profile_editor()

    def new_project(self) -> None:
        self.project = new_project()
        self.project_path = ""
        self.review_paths = []
        self.refresh_all()

    def sync_current_prompt(self) -> None:
        draft, corrected = self._current_texts()
        self.project["name"] = self.project_name.text().strip() or "Untitled project"
        self.project["draft_prompt"] = draft
        self.project["corrected_prompt"] = corrected
        self.project["generator_target"] = self.controller.generator_target_var.get()
        if corrected:
            versions = self.project.setdefault("versions", [])
            parent = str(versions[-1].get("id", "")) if versions else ""
            add_version(self.project, prompt=corrected, label=f"Revision {len(versions) + 1}", parent_id=parent)
        self.controller._save_settings()
        self.controller.status_var.set("Current prompt synced to project")

    def save_project(self) -> None:
        self.sync_current_prompt()
        path = self.project_path
        if not path:
            path, _ = QFileDialog.getSaveFileName(self, "Save PromptCorrector project", f"{self.project_name.text() or 'project'}.ipcp", "PromptCorrector Project (*.ipcp)")
        if not path:
            return
        try:
            saved = save_project_bundle(self.project, path)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Project save failed", str(exc))
            return
        self.project_path = str(saved)
        self.controller.status_var.set(f"Saved project {saved.name}")

    def open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open PromptCorrector project", "", "PromptCorrector Project (*.ipcp)")
        if not path:
            return
        try:
            extracted = Path(path).with_suffix("").with_name(Path(path).stem + "_assets")
            self.project = load_project_bundle(path, extracted)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Project open failed", str(exc))
            return
        self.project_path = path
        self.refresh_all()
        self.controller.status_var.set(f"Opened project {Path(path).name}")

    def add_result_images(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Add generated images", "", "Images (*.png *.jpg *.jpeg *.webp *.bmp)")
        for path in paths:
            self.project.setdefault("results", []).append({"id": Path(path).stem, "path": path, "created_at": utc_now()})
        self.refresh_results()

    def remove_result_image(self) -> None:
        row = self.result_list.currentRow()
        results = self.project.get("results", [])
        if isinstance(results, list) and 0 <= row < len(results):
            results.pop(row)
            self.refresh_results()

    def refresh_results(self) -> None:
        self.result_list.clear()
        for result in self.project.get("results", []):
            self.result_list.addItem(Path(str(result.get("path", ""))).name)

    def start_result_review(self) -> None:
        if self.controller.request_in_progress:
            self.controller.status_var.set("Another model request is already running")
            return
        results = self.project.get("results", [])
        if not results:
            QMessageBox.warning(self, "No generated images", "Add at least one generated result first.")
            return
        selected = self.result_list.currentRow()
        chosen = [results[selected]] if 0 <= selected < len(results) else list(results)
        paths = [str(item.get("path", "")) for item in chosen if Path(str(item.get("path", ""))).is_file()]
        if not paths:
            QMessageBox.warning(self, "Missing images", "The selected generated image files are unavailable.")
            return
        self.sync_current_prompt()
        self.controller.active_request_id += 1
        request_id = self.controller.active_request_id
        self.controller.cancel_event.clear()
        self.controller.request_in_progress = True
        self.controller._set_request_controls(True)
        self.review_button.setEnabled(False)
        self.controller.status_var.set("Reviewing generated image(s)...")
        thread = threading.Thread(target=self._review_worker, args=(request_id, paths), daemon=True)
        thread.start()

    def _review_worker(self, request_id: int, paths: list[str]) -> None:
        try:
            review = review_generated_images(
                base_url=self.controller._current_base_url(),
                model=self.controller.model_var.get(),
                image_paths=paths,
                original_prompt=str(self.project.get("draft_prompt", "")),
                corrected_prompt=str(self.project.get("corrected_prompt", "")),
                references=self.project.get("references", []),
                content_format=self.controller.content_format_var.get(),
                timeout=self.controller._lm_timeout_seconds(),
                api_key=os.getenv("LM_STUDIO_API_KEY", "lm-studio"),
                cancel_check=lambda: self.controller._raise_if_cancelled(request_id),
                safe_for_work=self.controller.safe_for_work_var.get(),
                explicit_nsfw=self.controller.explicit_nsfw_var.get(),
            )
        except Exception as exc:
            self.controller._after_threadsafe(0, self._finish_workbench_error, request_id, str(exc), self.review_button)
            return
        self.controller._after_threadsafe(0, self._finish_review, request_id, review)

    def _finish_review(self, request_id: int, review: dict[str, object]) -> None:
        if request_id != self.controller.active_request_id or self.controller.cancel_event.is_set():
            return
        self.project.setdefault("reviews", []).append(review)
        self.review_paths = list(review.get("image_paths", []))
        blocks = [f"Score: {review.get('score', 0)}/100", str(review.get("summary", ""))]
        for key, title in (("passed", "Passed"), ("failed", "Failed"), ("warnings", "Warnings")):
            values = review.get(key, [])
            blocks.append(f"\n{title}:\n" + ("\n".join(f"• {item}" for item in values) if values else "• None"))
        nsfw_fidelity = review.get("nsfw_fidelity", {})
        if isinstance(nsfw_fidelity, dict) and nsfw_fidelity:
            checks = (
                ("participant_count", "Participant count"),
                ("action_roles", "Action roles"),
                ("contact_targets", "Contact targets"),
                ("object_separation", "Object separation"),
                ("visible_phase", "Visible phase"),
                ("reactions", "Participant reactions"),
            )
            blocks.append(
                "\nNSFW fidelity:\n"
                + "\n".join(
                    f"• {label}: {nsfw_fidelity.get(key, 'not_applicable')}"
                    for key, label in checks
                )
            )
            discrepancies = nsfw_fidelity.get("discrepancies", [])
            if isinstance(discrepancies, list) and discrepancies:
                blocks.append(
                    "\nNSFW discrepancies:\n"
                    + "\n".join(f"• {item}" for item in discrepancies)
                )
        blocks.append("\nTargeted repair prompt:\n" + targeted_repair_prompt(review, str(self.project.get("corrected_prompt", ""))))
        diagnostics = review.get("diagnostics", {})
        if isinstance(diagnostics, dict):
            blocks.append(
                "\nDiagnostics:\n"
                f"• Images: {diagnostics.get('image_count', 0)}\n"
                f"• Image loading: {diagnostics.get('image_load_seconds', 0)}s\n"
                f"• Model review: {diagnostics.get('model_review_seconds', 0)}s\n"
                f"• Total: {diagnostics.get('total_seconds', 0)}s"
            )
        self.review_output.setPlainText("\n".join(blocks))
        self.review_button.setEnabled(True)
        self.controller.request_in_progress = False
        self.controller._set_request_controls(False)
        self.controller.status_var.set("Generated image review complete")
        self.controller._save_settings()

    def _finish_workbench_error(self, request_id: int, error: str, button: QPushButton | None = None) -> None:
        if request_id != self.controller.active_request_id:
            return
        if button is not None:
            button.setEnabled(True)
        self.controller.request_in_progress = False
        self.controller._set_request_controls(False)
        self.controller.status_var.set("Error")
        QMessageBox.critical(self, "Workbench error", error)

    def on_request_stopped(self) -> None:
        """Restore workbench-local state after the shared Stop action."""
        self.review_button.setEnabled(True)
        self.benchmark_button.setEnabled(True)
        for row in self.batch_rows:
            if row.get("status") == "running":
                row["status"] = "pending"
        self.refresh_batch_table()

    def apply_review_repair(self) -> None:
        reviews = self.project.get("reviews", [])
        if not reviews:
            return
        repair = targeted_repair_prompt(reviews[-1], str(self.project.get("corrected_prompt", "")))
        self.controller.generation_feedback_var.set(repair)
        self.controller.status_var.set("Targeted repair copied to Generation feedback")

    def _parse_crop(self) -> tuple[float, float, float, float] | None:
        text = self.reference_crop.text().strip()
        if not text:
            return None
        try:
            values = tuple(float(value.strip()) for value in text.split(","))
        except ValueError as exc:
            raise ValueError("Crop must contain four decimal values: x,y,width,height") from exc
        if len(values) != 4 or any(value < 0 or value > 1 for value in values):
            raise ValueError("Crop must contain four normalized values between 0 and 1")
        return values  # type: ignore[return-value]

    def choose_reference_mask(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose reference mask", "", "Images (*.png *.jpg *.jpeg *.webp *.bmp)")
        if path:
            self.mask_path = path
            self.controller.status_var.set(f"Reference mask selected: {Path(path).name}")

    def add_role_references(self) -> None:
        try:
            crop = self._parse_crop()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid crop", str(exc))
            return
        paths, _ = QFileDialog.getOpenFileNames(self, "Add role-aware reference images", "", "Images (*.png *.jpg *.jpeg *.webp *.bmp)")
        for path in paths:
            add_reference(
                self.project,
                path,
                role=self.reference_role.currentText(),
                panel=self.reference_panel.value() or None,
                subject=self.reference_subject.text(),
                crop=crop,
                mask_path=self.mask_path,
                notes=self.reference_notes.text(),
            )
            if path not in self.controller.local_reference_paths:
                self.controller.local_reference_paths.append(path)
        self.mask_path = ""
        self.refresh_references()
        self.controller._save_settings()

    def remove_reference(self) -> None:
        row = self.reference_list.currentRow()
        references = self.project.get("references", [])
        if isinstance(references, list) and 0 <= row < len(references):
            references.pop(row)
            self.refresh_references()

    def refresh_references(self) -> None:
        self.reference_list.clear()
        for reference in self.project.get("references", []):
            panel = f" · Panel {reference.get('panel')}" if reference.get("panel") else ""
            crop = " · crop" if reference.get("crop") else ""
            mask = " · mask" if reference.get("mask_path") else ""
            self.reference_list.addItem(f"{reference.get('role')} · {reference.get('subject') or Path(str(reference.get('path'))).name}{panel}{crop}{mask}")

    def add_character(self) -> None:
        name = self.character_name.text().strip()
        if not name:
            return
        character = {
            "id": name.lower().replace(" ", "_"),
            "name": name,
            "anchors": [item.strip() for item in self.character_anchors.text().split(",") if item.strip()],
            "forbidden": [item.strip() for item in self.character_forbid.text().split(",") if item.strip()],
        }
        self.project.setdefault("characters", []).append(character)
        self.character_name.clear()
        self.character_anchors.clear()
        self.character_forbid.clear()
        self.refresh_characters()

    def remove_character(self) -> None:
        row = self.character_list.currentRow()
        characters = self.project.get("characters", [])
        if isinstance(characters, list) and 0 <= row < len(characters):
            characters.pop(row)
            self.refresh_characters()

    def refresh_characters(self) -> None:
        self.character_list.clear()
        for character in self.project.get("characters", []):
            self.character_list.addItem(f"{character.get('name')}: {', '.join(character.get('anchors', []))}")

    def inspect_continuity(self) -> None:
        count = int(self.controller.comic_panel_count_var.get())
        panels = [self.controller.comic_panel_vars[index].get() for index in range(count)]
        issues = character_continuity_issues(self.project.get("characters", []), panels)
        self.continuity_output.setPlainText("\n".join(f"• {issue}" for issue in issues) if issues else "All named-character anchors are present in their panel descriptions.")

    def inspect_contracts(self) -> None:
        draft, corrected = self._current_texts()
        story = self.controller.story_elements_var.get()
        if self.controller.content_format_var.get() == "Comic Story":
            try:
                _draft, story = self.controller._comic_story_inputs()
            except ValueError:
                pass
        rows = contract_dashboard(
            corrected,
            original_prompt=draft,
            story_elements=story,
            content_format=self.controller.content_format_var.get(),
            weighted_terms=self.controller.weighted_terms_var.get(),
            model_instructions=self.controller.model_instructions_var.get(),
            safe_for_work=self.controller.safe_for_work_var.get(),
        )
        self.contract_table.setRowCount(len(rows))
        colors = {"pass": "#62d49b", "warning": "#e8bf66", "fail": "#ff7676"}
        for row_index, row in enumerate(rows):
            values = (str(row["status"]).upper(), row["category"], row["detail"])
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 0:
                    item.setForeground(QColor(colors.get(row["status"], "#ffffff")))
                self.contract_table.setItem(row_index, column, item)

    def inspect_clarifications(self) -> None:
        draft, _corrected = self._current_texts()
        questions = clarification_questions(draft)
        self.clarification_output.setPlainText("\n".join(f"• {question}" for question in questions) if questions else "No essential clarification is needed.")

    def prepare_variants(self) -> None:
        draft, corrected = self._current_texts()
        source = corrected or draft
        self.variant_list.clear()
        for variant in prompt_variant_instructions(source):
            item = QListWidgetItem(variant["label"])
            item.setData(Qt.ItemDataRole.UserRole, variant)
            self.variant_list.addItem(item)
        if self.variant_list.count():
            self.variant_list.setCurrentRow(0)

    def generate_selected_variant(self) -> None:
        item = self.variant_list.currentItem()
        if item is None:
            self.prepare_variants()
            item = self.variant_list.currentItem()
        if item is not None:
            self._start_variant_generation([item.data(Qt.ItemDataRole.UserRole)])

    def generate_all_variants(self) -> None:
        if not self.variant_list.count():
            self.prepare_variants()
        variants = [self.variant_list.item(index).data(Qt.ItemDataRole.UserRole) for index in range(self.variant_list.count())]
        self._start_variant_generation(variants)

    def _start_variant_generation(self, variants: list[dict[str, str]]) -> None:
        if self.controller.request_in_progress or not variants:
            return
        self.controller.active_request_id += 1
        request_id = self.controller.active_request_id
        self.controller.cancel_event.clear()
        self.controller.request_in_progress = True
        self.controller._set_request_controls(True)
        self.controller.status_var.set("Generating controlled prompt variants...")
        threading.Thread(target=self._variant_worker, args=(request_id, variants), daemon=True).start()

    def _variant_worker(self, request_id: int, variants: list[dict[str, str]]) -> None:
        results: list[dict[str, str]] = []
        try:
            for variant_index, variant in enumerate(variants):
                self.controller._raise_if_cancelled(request_id)
                variant_system = "Rewrite one image prompt. Preserve every hard contract. Return only the prompt."
                if self.controller.safe_for_work_var.get():
                    variant_system += " Return safe-for-work content only: complete opaque clothing, non-sexual framing, and no graphic gore."
                elif self.controller.explicit_nsfw_var.get():
                    validate_explicit_adult_mode(variant["instruction"])
                    variant_system += " " + EXPLICIT_ADULT_MODE_INSTRUCTION
                response = chat_completion(
                    base_url=self.controller._current_base_url(),
                    model=self.controller.model_var.get(),
                    messages=[
                        {"role": "system", "content": variant_system},
                        {"role": "user", "content": variant["instruction"]},
                    ],
                    temperature=0.2,
                    max_tokens=1200,
                    timeout=self.controller._lm_timeout_seconds(),
                    api_key=os.getenv("LM_STUDIO_API_KEY", "lm-studio"),
                    seed=(
                        (self.controller._sampling_seed() + variant_index) % 2_147_483_648
                        if self.controller._sampling_seed() is not None
                        else None
                    ),
                    cancel_check=lambda: self.controller._raise_if_cancelled(request_id),
                )
                if self.controller.safe_for_work_var.get():
                    response = make_prompt_safe_for_work(response)
                results.append({"label": variant["label"], "prompt": response.strip()})
        except Exception as exc:
            self.controller._after_threadsafe(0, self._finish_workbench_error, request_id, str(exc), None)
            return
        self.controller._after_threadsafe(0, self._finish_variants, request_id, results)

    def _finish_variants(self, request_id: int, results: list[dict[str, str]]) -> None:
        if request_id != self.controller.active_request_id:
            return
        self.variant_results.extend(results)
        for result in results:
            add_version(self.project, prompt=result["prompt"], label=result["label"])
        self.variant_output.setPlainText("\n\n".join(f"{item['label']}:\n{item['prompt']}" for item in self.variant_results))
        self.controller.request_in_progress = False
        self.controller._set_request_controls(False)
        self.controller.status_var.set("Prompt variants ready")

    def use_selected_variant(self) -> None:
        item = self.variant_list.currentItem()
        if item is None:
            return
        label = str(item.text())
        result = next((entry for entry in reversed(self.variant_results) if entry.get("label") == label), None)
        if result is None:
            QMessageBox.warning(self, "No generated variant", f"Generate the {label} variant first.")
            return
        prompt = result["prompt"]
        if self.controller.content_format_var.get() == "Comic Story" and self.controller.comic_result_text is not None:
            self.controller.comic_result_text.setPlainText(prompt)
        else:
            self.controller.corrected_text.setPlainText(prompt)
        self.project["corrected_prompt"] = prompt
        self.project["winning_variant"] = label
        add_version(self.project, prompt=prompt, label=f"Winner · {label}")
        self.controller.status_var.set(f"Selected {label} as the project winner")

    def arm_composition_tool(self) -> None:
        self.composition_canvas.set_tool(self.composition_kind.currentText(), self.composition_label.text(), self.composition_panel.value())
        self.controller.status_var.set("Drag on the composition canvas to place the box")

    def apply_composition(self) -> None:
        self.project["composition"] = list(self.composition_canvas.items)
        instruction = composition_instruction(self.composition_canvas.items)
        self.composition_output.setPlainText(instruction)
        self._append_model_instruction("Composition contract:\n" + instruction)

    def add_text_layout(self) -> None:
        exact = self.text_exact.text().strip()
        if not exact:
            return
        self.project.setdefault("text_layout", []).append({
            "text": exact,
            "speaker": self.text_speaker.text().strip(),
            "kind": self.text_kind.currentText(),
            "panel": self.text_panel.value() or None,
            "placement": self.text_placement.text().strip(),
        })
        self.text_exact.clear()
        self.refresh_text_layout()

    def remove_text_layout(self) -> None:
        row = self.text_layout_list.currentRow()
        items = self.project.get("text_layout", [])
        if isinstance(items, list) and 0 <= row < len(items):
            items.pop(row)
            self.refresh_text_layout()

    def refresh_text_layout(self) -> None:
        self.text_layout_list.clear()
        for item in self.project.get("text_layout", []):
            panel = f"Panel {item.get('panel')} · " if item.get("panel") else ""
            self.text_layout_list.addItem(f"{panel}{item.get('kind')}: \"{item.get('text')}\"")

    def apply_text_layout(self) -> None:
        instruction = text_layout_instruction(self.project.get("text_layout", []))
        self._append_model_instruction("Exact text layout contract:\n" + instruction)

    def _append_model_instruction(self, instruction: str) -> None:
        existing = self.controller.model_instructions_var.get().strip()
        if instruction.strip() and instruction.strip() not in existing:
            self.controller.model_instructions_var.set((existing + "\n" + instruction.strip()).strip())
        self.controller.status_var.set("Applied workbench contract to Model instructions")

    def import_batch_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import prompt batch", "", "CSV files (*.csv);;All files (*)")
        if path:
            self.batch_input.setPlainText(Path(path).read_text(encoding="utf-8"))
            self.load_batch_queue()

    def load_batch_queue(self) -> None:
        self.batch_rows = parse_batch_csv(self.batch_input.toPlainText())
        self.refresh_batch_table()

    def refresh_batch_table(self) -> None:
        self.batch_table.setRowCount(len(self.batch_rows))
        for row, item in enumerate(self.batch_rows):
            values = (item["id"], item["status"], item["prompt"], item["result"] or item["error"])
            for column, value in enumerate(values):
                self.batch_table.setItem(row, column, QTableWidgetItem(value))
        completed = sum(item["status"] in ("done", "error") for item in self.batch_rows)
        self.batch_progress.setMaximum(max(1, len(self.batch_rows)))
        self.batch_progress.setValue(completed)

    def run_batch(self) -> None:
        if not self.batch_rows:
            self.load_batch_queue()
        pending = [index for index, row in enumerate(self.batch_rows) if row["status"] in ("pending", "error")]
        if not pending or self.controller.request_in_progress:
            return
        self.controller.active_request_id += 1
        request_id = self.controller.active_request_id
        self.controller.cancel_event.clear()
        self.controller.request_in_progress = True
        self.controller._set_request_controls(True)
        self.controller.status_var.set("Running prompt batch...")
        threading.Thread(target=self._batch_worker, args=(request_id, pending), daemon=True).start()

    def _batch_worker(self, request_id: int, pending: list[int]) -> None:
        for index in pending:
            try:
                self.controller._raise_if_cancelled(request_id)
                row = self.batch_rows[index]
                row["status"] = "running"
                self.controller._after_threadsafe(0, self.refresh_batch_table)
                row["result"] = post_chat_completion(
                    base_url=self.controller._current_base_url(),
                    model=self.controller.model_var.get(),
                    prompt=row["prompt"],
                    generator_target=self.controller.generator_target_var.get(),
                    content_format="Single Image",
                    temperature=self.controller._temperature_value(),
                    max_tokens=1400,
                    timeout=self.controller._lm_timeout_seconds(),
                    api_key=os.getenv("LM_STUDIO_API_KEY", "lm-studio"),
                    seed=(
                        (self.controller._sampling_seed() + index) % 2_147_483_648
                        if self.controller._sampling_seed() is not None
                        else None
                    ),
                    risk_level="Strict cleanup",
                    preserve_strictly=True,
                    goal_headline=row["goal"],
                    focus=row["focus"],
                    audit_repair=True,
                    safe_for_work=self.controller.safe_for_work_var.get(),
                    explicit_nsfw=self.controller.explicit_nsfw_var.get(),
                    cancel_check=lambda: self.controller._raise_if_cancelled(request_id),
                )
                row["status"] = "done"
                row["error"] = ""
            except Exception as exc:
                if self.controller._request_cancelled(request_id):
                    break
                self.batch_rows[index]["status"] = "error"
                self.batch_rows[index]["error"] = str(exc)
            self.controller._after_threadsafe(0, self.refresh_batch_table)
        self.controller._after_threadsafe(0, self._finish_batch, request_id)

    def _finish_batch(self, request_id: int) -> None:
        if request_id != self.controller.active_request_id:
            return
        self.controller.request_in_progress = False
        self.controller._set_request_controls(False)
        self.refresh_batch_table()
        self.controller.status_var.set("Prompt batch finished")

    def export_batch_results(self) -> None:
        if not self.batch_rows:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export prompt batch results", "prompt_results.json", "JSON files (*.json)")
        if path:
            Path(path).write_text(json.dumps(self.batch_rows, indent=2), encoding="utf-8")

    def run_benchmark(self) -> None:
        if self.controller.request_in_progress:
            return
        self.controller.active_request_id += 1
        request_id = self.controller.active_request_id
        self.controller.cancel_event.clear()
        self.controller.request_in_progress = True
        self.controller._set_request_controls(True)
        self.benchmark_button.setEnabled(False)
        self.controller.status_var.set("Benchmarking loaded model...")
        threading.Thread(target=self._benchmark_worker, args=(request_id,), daemon=True).start()

    def _benchmark_worker(self, request_id: int) -> None:
        try:
            result = benchmark_model(
                base_url=self.controller._current_base_url(),
                model=self.controller.model_var.get(),
                timeout=min(180, self.controller._lm_timeout_seconds()),
                api_key=os.getenv("LM_STUDIO_API_KEY", "lm-studio"),
                cancel_check=lambda: self.controller._raise_if_cancelled(request_id),
                include_vision=True,
            )
        except Exception as exc:
            self.controller._after_threadsafe(0, self._finish_workbench_error, request_id, str(exc), self.benchmark_button)
            return
        self.controller._after_threadsafe(0, self._finish_benchmark, request_id, result)

    def _finish_benchmark(self, request_id: int, result: dict[str, object]) -> None:
        if request_id != self.controller.active_request_id:
            return
        lines = [f"Model: {result['model']}", f"Score: {result['score']}/100", f"Time: {result['seconds']}s"]
        for case in result["cases"]:
            lines.append(f"\n{'PASS' if case.get('passed') else 'FAIL'} · {case['case']} · {case['seconds']}s\n{case.get('response') or case.get('error')}")
        self.benchmark_output.setPlainText("\n".join(lines))
        self.benchmark_button.setEnabled(True)
        self.controller.request_in_progress = False
        self.controller._set_request_controls(False)
        self.controller.status_var.set("Model benchmark complete")

    def load_profile_editor(self) -> None:
        name = self.profile_name.currentText().strip()
        profile = self.generator_profiles.get(name, GENERATOR_PROFILE_DEFAULTS.get(name, {}))
        self.profile_editor.setPlainText(json.dumps(profile, indent=2, sort_keys=True))

    def save_profile_editor(self) -> None:
        name = self.profile_name.currentText().strip()
        if not name:
            return
        try:
            profile = json.loads(self.profile_editor.toPlainText())
        except json.JSONDecodeError as exc:
            QMessageBox.warning(self, "Invalid profile JSON", str(exc))
            return
        if not isinstance(profile, dict):
            QMessageBox.warning(self, "Invalid profile", "A generator profile must be a JSON object.")
            return
        self.generator_profiles[name] = profile
        if self.profile_name.findText(name) < 0:
            self.profile_name.addItem(name)
        self.controller._save_settings()
        self.controller.status_var.set(f"Saved generator profile {name}")

    def apply_profile_editor(self) -> None:
        self.save_profile_editor()
        name = self.profile_name.currentText().strip()
        profile = self.generator_profiles.get(name)
        if not isinstance(profile, dict):
            return
        self.project["generator_profile"] = name
        self.project["generator_settings"] = profile.get("setup", {})
        style = str(profile.get("prompt_style", "")).strip()
        negative = profile.get("negative_prompt")
        instruction = f"Generator profile {name}: use {style or 'the profile-defined prompt structure'}."
        instruction += " Fold avoidances into the main prompt." if negative is False else " A separate negative prompt is supported."
        self._append_model_instruction(instruction)

    def choose_comfy_workflow(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose ComfyUI API workflow", "", "JSON files (*.json)")
        if path:
            self.workflow_path = path
            self.comfy_workflow_label.setText(path)

    def send_to_comfyui(self) -> None:
        _draft, corrected = self._current_texts()
        if not corrected or not self.workflow_path:
            QMessageBox.warning(self, "ComfyUI handoff", "Choose an API workflow and create a corrected prompt first.")
            return
        try:
            workflow = json.loads(Path(self.workflow_path).read_text(encoding="utf-8"))
            result = enqueue_comfyui(
                server_url=self.comfy_url.text(),
                workflow=workflow,
                prompt=corrected,
                positive_node_id=self.comfy_node.text().strip(),
            )
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
            QMessageBox.critical(self, "ComfyUI handoff failed", str(exc))
            return
        self.integration_output.setPlainText(json.dumps(result, indent=2))
        self.controller.status_var.set("Prompt enqueued in ComfyUI")
