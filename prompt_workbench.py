#!/usr/bin/env python3
"""Project and evaluation workflows for Image Prompt Corrector.

This module deliberately keeps the workbench data independent from Qt.  The
GUI can therefore persist projects, run audits, benchmark models, and enqueue
ComfyUI jobs without coupling those operations to widgets.
"""

from __future__ import annotations

import csv
import io
import json
import mimetypes
import re
import time
import urllib.error
import urllib.request
import uuid
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from krea_prompt_corrector import (
    chat_completion,
    fetch_image_data_url,
    final_compliance_issues,
    validate_explicit_adult_mode,
)
from nsfw_scene_contract import nsfw_image_audit_contract


PROJECT_SCHEMA_VERSION = 1
REFERENCE_ROLES = (
    "Identity",
    "Face",
    "Outfit",
    "Pose",
    "Composition",
    "Style",
    "Environment",
    "Color palette",
    "Prop",
    "Other",
)
CONTRACT_STATUSES = ("pass", "warning", "fail")
GENERATOR_PROFILE_DEFAULTS = {
    "Krea 2": {
        "prompt_style": "natural visual description",
        "negative_prompt": False,
        "setup": {"creativity": "raw", "intensity": 0, "movement": 0},
    },
    "FLUX.2 Klein 9B": {
        "prompt_style": "priority ordered explicit description",
        "negative_prompt": False,
        "setup": {"steps": 4, "guidance": 1.0},
    },
    "ComfyUI custom": {
        "prompt_style": "workflow supplied",
        "negative_prompt": True,
        "setup": {},
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def new_project(name: str = "Untitled project") -> dict[str, object]:
    now = utc_now()
    return {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "id": _id("project"),
        "name": name.strip() or "Untitled project",
        "created_at": now,
        "updated_at": now,
        "draft_prompt": "",
        "corrected_prompt": "",
        "generator_target": "Krea 2",
        "generator_profile": "Krea 2",
        "generator_settings": {},
        "references": [],
        "results": [],
        "reviews": [],
        "versions": [],
        "characters": [],
        "composition": [],
        "text_layout": [],
        "notes": "",
    }


def normalize_project(value: object) -> dict[str, object]:
    project = new_project()
    if not isinstance(value, dict):
        return project
    for key in project:
        if key in value:
            project[key] = value[key]
    project["schema_version"] = PROJECT_SCHEMA_VERSION
    project["name"] = str(project.get("name", "")).strip() or "Untitled project"
    for key in ("references", "results", "reviews", "versions", "characters", "composition", "text_layout"):
        if not isinstance(project.get(key), list):
            project[key] = []
    project["updated_at"] = str(project.get("updated_at") or utc_now())
    return project


def add_version(
    project: dict[str, object],
    *,
    prompt: str,
    label: str = "Revision",
    parent_id: str = "",
    notes: str = "",
) -> dict[str, object]:
    version = {
        "id": _id("version"),
        "parent_id": parent_id,
        "label": label.strip() or "Revision",
        "prompt": prompt.strip(),
        "notes": notes.strip(),
        "created_at": utc_now(),
    }
    project.setdefault("versions", []).append(version)
    project["updated_at"] = utc_now()
    return version


def add_reference(
    project: dict[str, object],
    path: str,
    *,
    role: str = "Other",
    panel: int | None = None,
    subject: str = "",
    crop: tuple[float, float, float, float] | None = None,
    mask_path: str = "",
    notes: str = "",
) -> dict[str, object]:
    if role not in REFERENCE_ROLES:
        role = "Other"
    reference = {
        "id": _id("reference"),
        "path": str(path),
        "role": role,
        "panel": panel if panel and panel > 0 else None,
        "subject": subject.strip(),
        "crop": list(crop) if crop else None,
        "mask_path": str(mask_path),
        "notes": notes.strip(),
    }
    project.setdefault("references", []).append(reference)
    project["updated_at"] = utc_now()
    return reference


def reference_instruction(references: Iterable[dict[str, object]]) -> str:
    lines: list[str] = []
    for ref in references:
        role = str(ref.get("role", "Other"))
        subject = str(ref.get("subject", "")).strip()
        panel = ref.get("panel")
        crop = ref.get("crop")
        detail = f"Use as {role.lower()} reference"
        if subject:
            detail += f" for {subject}"
        if panel:
            detail += f" in Panel {panel}"
        if isinstance(crop, list) and len(crop) == 4:
            detail += f"; inspect normalized crop {crop}"
        notes = str(ref.get("notes", "")).strip()
        if notes:
            detail += f"; {notes}"
        lines.append(detail + ".")
    return "\n".join(lines)


def save_project_bundle(project: dict[str, object], destination: str | Path) -> Path:
    """Save a portable .ipcp bundle and copy available media into it."""
    destination = Path(destination)
    if destination.suffix.lower() != ".ipcp":
        destination = destination.with_suffix(".ipcp")
    normalized = normalize_project(project)
    stored = json.loads(json.dumps(normalized))
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        used_names: set[str] = set()
        for collection in ("references", "results"):
            for entry in stored.get(collection, []):
                source = Path(str(entry.get("path", "")))
                if not source.is_file():
                    continue
                base = re.sub(r"[^A-Za-z0-9._-]+", "_", source.name) or "asset"
                member = f"assets/{base}"
                counter = 2
                while member in used_names:
                    member = f"assets/{source.stem}_{counter}{source.suffix}"
                    counter += 1
                used_names.add(member)
                archive.write(source, member)
                entry["bundle_path"] = member
        archive.writestr("project.json", json.dumps(stored, indent=2, sort_keys=True))
    return destination


def load_project_bundle(source: str | Path, extract_dir: str | Path | None = None) -> dict[str, object]:
    source = Path(source)
    with zipfile.ZipFile(source, "r") as archive:
        names = set(archive.namelist())
        if "project.json" not in names:
            raise ValueError("Project bundle does not contain project.json")
        project = normalize_project(json.loads(archive.read("project.json").decode("utf-8")))
        if extract_dir is not None:
            root = Path(extract_dir)
            root.mkdir(parents=True, exist_ok=True)
            for collection in ("references", "results"):
                for entry in project.get(collection, []):
                    member = str(entry.get("bundle_path", ""))
                    if not member or member not in names or not member.startswith("assets/"):
                        continue
                    target = root / Path(member).name
                    target.write_bytes(archive.read(member))
                    entry["path"] = str(target)
    return project


def contract_dashboard(
    corrected_prompt: str,
    *,
    original_prompt: str = "",
    story_elements: str = "",
    content_format: str = "Auto",
    weighted_terms: str = "",
    model_instructions: str = "",
    safe_for_work: bool = False,
) -> list[dict[str, str]]:
    issues = final_compliance_issues(
        corrected_prompt,
        original_prompt=original_prompt,
        story_elements=story_elements,
        content_format=content_format,
        weighted_terms=weighted_terms,
        model_instructions=model_instructions,
        safe_for_work=safe_for_work,
    )
    groups = {
        "Structure": ("empty", "multiple lines", "syntax", "panel"),
        "Counts and placement": ("count", "spatial", "position", "side"),
        "Text and exclusions": ("rendered text", "quoted", "exclusion", "script"),
        "Characters and pose": ("entity", "role", "pose", "plausibility"),
        "Visual clarity": ("vague", "feeling", "slang", "weak", "spelling"),
        "Emphasis": ("weighted", "concept", "focus", "intent"),
    }
    rows: list[dict[str, str]] = []
    remaining = list(issues)
    for label, markers in groups.items():
        matched = [issue for issue in issues if any(marker in issue.lower() for marker in markers)]
        remaining = [issue for issue in remaining if issue not in matched]
        rows.append({
            "category": label,
            "status": "fail" if matched else "pass",
            "detail": "; ".join(matched) if matched else "No detected contract violations.",
        })
    if remaining:
        rows.append({"category": "Other", "status": "warning", "detail": "; ".join(remaining)})
    return rows


def clarification_questions(prompt: str, *, maximum: int = 3) -> list[str]:
    """Return only questions whose answer can materially change composition."""
    text = prompt.strip().lower()
    if not text:
        return ["What is the main subject and what should be visibly happening?"]
    questions: list[str] = []
    if len(text.split()) < 7:
        questions.append("What is the main action or decisive visual moment?")
    if re.search(r"\b(two|three|several|people|characters|they|them)\b", text) and not re.search(
        r"\b(left|right|foreground|background|center|wearing)\b", text
    ):
        questions.append("Which character performs each action, and where is each character positioned?")
    if re.search(r"\b(text|caption|sign|logo|says|reads|title)\b", text) and '"' not in prompt:
        questions.append("What exact text must be rendered, including capitalization and punctuation?")
    if re.search(r"\b(hand|arm|leg|foot|pose|holding|gripping|kicking)\b", text) and not re.search(
        r"\b(front|back|profile|camera|left|right)\b", text
    ):
        questions.append("From which camera view should the pose and limb orientation be seen?")
    return questions[:maximum]


def build_result_review_messages(
    *,
    original_prompt: str,
    corrected_prompt: str,
    reference_instruction_text: str = "",
    content_format: str = "Single Image",
    image_data_urls: Iterable[str],
    safe_for_work: bool = False,
    explicit_nsfw: bool = False,
) -> list[dict[str, object]]:
    if safe_for_work and explicit_nsfw:
        raise RuntimeError("Safe for work and Explicit adult (NSFW) cannot both be enabled.")
    if explicit_nsfw:
        validate_explicit_adult_mode(f"{original_prompt}\n{corrected_prompt}")
    system = (
        "You are a strict visual result inspector. Compare generated images with the source request, "
        "generator prompt, and intentional reference roles. Do not reward beauty when a requirement is wrong. "
        "Return JSON only with keys score (0-100), summary, passed (array), failed (array), warnings (array), "
        "and repair_prompt. When explicit adult mode is enabled, also return nsfw_fidelity as an object "
        "with participant_count, action_roles, contact_targets, object_separation, visible_phase, reactions, "
        "and discrepancies; each check value must be pass, fail, or not_applicable, and discrepancies must "
        "be an array. The repair_prompt must preserve successful content and change only failed items. "
        "Check counts, identity, pose and viewpoint, spatial side, props, exact text, exclusions, composition, "
        "style, and panel-to-beat mapping."
    )
    text = (
        f"Content format: {content_format}\n\nOriginal request:\n{original_prompt}\n\n"
        f"Generator prompt:\n{corrected_prompt}"
    )
    if reference_instruction_text.strip():
        text += f"\n\nReference roles:\n{reference_instruction_text.strip()}"
    if safe_for_work:
        text += (
            "\n\nMandatory safety contract: the generated image must be safe for work, with no explicit nudity, "
            "exposed intimate anatomy, sexual activity, fetish framing, or graphic gore."
        )
    elif explicit_nsfw:
        text += (
            "\n\nExplicit adult mode: evaluate requested adult content directly without censoring or "
            "penalizing explicitness. Every sexualized subject must be unambiguously age 18 or older."
            "\n\n"
            + nsfw_image_audit_contract(
                original_prompt,
                corrected_prompt,
                content_format=content_format,
            )
        )
    content: list[dict[str, object]] = [{"type": "text", "text": text}]
    for data_url in image_data_urls:
        content.append({"type": "image_url", "image_url": {"url": data_url}})
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


def parse_review_response(text: str) -> dict[str, object]:
    cleaned = text.strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        cleaned = match.group(0)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return {
            "score": 0,
            "summary": text.strip() or "The model returned an empty review.",
            "passed": [],
            "failed": ["Review response was not valid JSON."],
            "warnings": [],
            "repair_prompt": "",
        }
    if not isinstance(parsed, dict):
        parsed = {}
    try:
        score = max(0, min(100, int(parsed.get("score", 0))))
    except (TypeError, ValueError):
        score = 0
    result: dict[str, object] = {"score": score, "summary": str(parsed.get("summary", "")).strip()}
    for key in ("passed", "failed", "warnings"):
        value = parsed.get(key, [])
        result[key] = [str(item).strip() for item in value if str(item).strip()] if isinstance(value, list) else []
    result["repair_prompt"] = str(parsed.get("repair_prompt", "")).strip()
    nsfw_fidelity = parsed.get("nsfw_fidelity", {})
    if isinstance(nsfw_fidelity, dict):
        normalized_fidelity: dict[str, object] = {}
        for key in (
            "participant_count",
            "action_roles",
            "contact_targets",
            "object_separation",
            "visible_phase",
            "reactions",
        ):
            value = str(nsfw_fidelity.get(key, "not_applicable")).strip().lower()
            normalized_fidelity[key] = (
                value if value in {"pass", "fail", "not_applicable"} else "not_applicable"
            )
        discrepancies = nsfw_fidelity.get("discrepancies", [])
        normalized_fidelity["discrepancies"] = (
            [str(item).strip() for item in discrepancies if str(item).strip()]
            if isinstance(discrepancies, list)
            else []
        )
        result["nsfw_fidelity"] = normalized_fidelity
    return result


def review_generated_images(
    *,
    base_url: str,
    model: str,
    image_paths: Iterable[str],
    original_prompt: str,
    corrected_prompt: str,
    references: Iterable[dict[str, object]] = (),
    content_format: str = "Single Image",
    timeout: float = 600,
    api_key: str = "lm-studio",
    cancel_check: Callable[[], None] | None = None,
    safe_for_work: bool = False,
    explicit_nsfw: bool = False,
) -> dict[str, object]:
    started = time.monotonic()
    image_paths = list(image_paths)
    fetch_started = time.monotonic()
    data_urls = [
        fetch_image_data_url(
            Path(path).resolve().as_uri(),
            timeout=timeout,
            max_bytes=20_000_000,
        )
        for path in image_paths
    ]
    messages = build_result_review_messages(
        original_prompt=original_prompt,
        corrected_prompt=corrected_prompt,
        reference_instruction_text=reference_instruction(references),
        content_format=content_format,
        image_data_urls=data_urls,
        safe_for_work=safe_for_work,
        explicit_nsfw=explicit_nsfw,
    )
    fetch_seconds = round(time.monotonic() - fetch_started, 3)
    model_started = time.monotonic()
    response = chat_completion(
        base_url=base_url,
        model=model,
        messages=messages,
        temperature=0.1,
        max_tokens=2048,
        timeout=timeout,
        api_key=api_key,
        cancel_check=cancel_check,
    )
    model_seconds = round(time.monotonic() - model_started, 3)
    result = parse_review_response(response)
    result["created_at"] = utc_now()
    result["image_paths"] = image_paths
    result["diagnostics"] = {
        "image_load_seconds": fetch_seconds,
        "model_review_seconds": model_seconds,
        "total_seconds": round(time.monotonic() - started, 3),
        "image_count": len(image_paths),
    }
    return result


def targeted_repair_prompt(review: dict[str, object], current_prompt: str) -> str:
    explicit = str(review.get("repair_prompt", "")).strip()
    if explicit:
        return explicit
    failed = review.get("failed", [])
    failures = (
        [str(item) for item in failed if str(item).strip()]
        if isinstance(failed, list)
        else []
    )
    nsfw_fidelity = review.get("nsfw_fidelity", {})
    if isinstance(nsfw_fidelity, dict):
        discrepancies = nsfw_fidelity.get("discrepancies", [])
        if isinstance(discrepancies, list):
            failures.extend(
                "NSFW fidelity: " + str(item)
                for item in discrepancies
                if str(item).strip()
            )
    if not failures:
        return current_prompt.strip()
    return (
        "Preserve every successful visual element and the existing composition. Correct only these failures: "
        + "; ".join(failures)
        + ". Updated prompt: "
        + current_prompt.strip()
    )


def character_continuity_issues(characters: Iterable[dict[str, object]], panel_texts: Iterable[str]) -> list[str]:
    panels = list(panel_texts)
    issues: list[str] = []
    for character in characters:
        name = str(character.get("name", "")).strip()
        anchors = [str(item).strip() for item in character.get("anchors", []) if str(item).strip()]
        if not name:
            continue
        relevant = [panel for panel in panels if name.lower() in panel.lower()]
        for index, panel in enumerate(panels, start=1):
            if name.lower() not in panel.lower():
                continue
            missing = [anchor for anchor in anchors if anchor.lower() not in panel.lower()]
            if missing:
                issues.append(f"Panel {index}: {name} is missing continuity anchors: {', '.join(missing)}")
        if not relevant:
            issues.append(f"Character {name} does not appear in any panel description.")
    return issues


def composition_instruction(items: Iterable[dict[str, object]]) -> str:
    lines: list[str] = []
    for item in items:
        kind = str(item.get("kind", "subject"))
        label = str(item.get("label", kind)).strip()
        x = round(float(item.get("x", 0)), 3)
        y = round(float(item.get("y", 0)), 3)
        width = round(float(item.get("width", 0)), 3)
        height = round(float(item.get("height", 0)), 3)
        panel = item.get("panel")
        scope = f"Panel {panel}: " if panel else ""
        lines.append(f"{scope}{label} {kind} occupies normalized box x={x}, y={y}, width={width}, height={height}.")
    return "\n".join(lines)


def text_layout_instruction(items: Iterable[dict[str, object]]) -> str:
    lines: list[str] = []
    for item in items:
        exact = str(item.get("text", "")).strip()
        if not exact:
            continue
        speaker = str(item.get("speaker", "")).strip()
        kind = str(item.get("kind", "speech bubble"))
        panel = item.get("panel")
        placement = str(item.get("placement", "auto")).strip()
        prefix = f"Panel {panel}: " if panel else ""
        owner = f" spoken by {speaker}" if speaker else ""
        lines.append(f'{prefix}{kind}{owner} must render exactly "{exact}" at {placement}.')
    return "\n".join(lines)


def parse_batch_csv(text: str) -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(text))
    rows: list[dict[str, str]] = []
    for index, row in enumerate(reader, start=1):
        prompt = str(row.get("prompt", "") or "").strip()
        if not prompt:
            continue
        rows.append({
            "id": str(row.get("id", "") or index),
            "prompt": prompt,
            "goal": str(row.get("goal", "") or "").strip(),
            "focus": str(row.get("focus", "") or "").strip(),
            "status": "pending",
            "result": "",
            "error": "",
        })
    return rows


def prompt_variant_instructions(prompt: str) -> list[dict[str, str]]:
    locks = "Preserve all subjects, counts, identities, positions, exclusions, exact quoted text, and required actions."
    return [
        {"label": "Faithful", "instruction": f"{locks} Make the smallest useful cleanup. Source: {prompt}"},
        {"label": "Composition", "instruction": f"{locks} Strengthen hierarchy, framing, depth, and readable staging. Source: {prompt}"},
        {"label": "Camera", "instruction": f"{locks} Clarify camera view, perspective, lens feel, and pose readability. Source: {prompt}"},
        {"label": "Atmosphere", "instruction": f"{locks} Strengthen coherent lighting, palette, material response, and atmosphere. Source: {prompt}"},
    ]


@dataclass
class DiagnosticTrace:
    stages: list[dict[str, object]] = field(default_factory=list)
    _active_name: str = ""
    _active_start: float = 0.0

    def start(self, name: str) -> None:
        if self._active_name:
            self.finish("interrupted")
        self._active_name = name
        self._active_start = time.monotonic()

    def finish(self, status: str = "ok", detail: str = "") -> None:
        if not self._active_name:
            return
        self.stages.append({
            "name": self._active_name,
            "seconds": round(time.monotonic() - self._active_start, 3),
            "status": status,
            "detail": detail,
        })
        self._active_name = ""
        self._active_start = 0.0

    def total_seconds(self) -> float:
        return round(sum(float(stage.get("seconds", 0)) for stage in self.stages), 3)


def benchmark_model(
    *,
    base_url: str,
    model: str,
    timeout: float = 120,
    api_key: str = "lm-studio",
    cancel_check: Callable[[], None] | None = None,
    include_vision: bool = False,
) -> dict[str, object]:
    cases = [
        ("instruction", 'Return exactly: red cube left of two blue spheres; no text.', ("red cube", "two blue spheres", "no text")),
        ("quoted text", 'Return exactly: a sign reading "NORTH GATE"; no other lettering.', ("north gate", "no other lettering")),
        ("spatial", 'Return exactly: front-facing knight, shield in subject-left hand, sword in subject-right hand.', ("front-facing", "subject-left", "subject-right")),
        ("panels", 'Return exactly: Panel 1: cat finds key. Panel 2: same cat opens door.', ("panel 1", "panel 2", "cat", "key", "door")),
    ]
    results: list[dict[str, object]] = []
    started = time.monotonic()
    for label, request, required in cases:
        case_started = time.monotonic()
        try:
            response = chat_completion(
                base_url=base_url,
                model=model,
                messages=[
                    {"role": "system", "content": "Follow the user's exact output contract. Do not explain."},
                    {"role": "user", "content": request},
                ],
                temperature=0.0,
                max_tokens=160,
                timeout=timeout,
                api_key=api_key,
                cancel_check=cancel_check,
            ).strip()
            haystack = response.lower()
            passed = all(token in haystack for token in required)
            results.append({"case": label, "passed": passed, "seconds": round(time.monotonic() - case_started, 3), "response": response})
        except RuntimeError as exc:
            results.append({"case": label, "passed": False, "seconds": round(time.monotonic() - case_started, 3), "error": str(exc)})
    if include_vision:
        # A tiny known-red PNG makes the probe deterministic and avoids relying
        # on any project image semantics.
        red_png = (
            "data:image/png;base64,"
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl2n2cAAAAASUVORK5CYII="
        )
        case_started = time.monotonic()
        try:
            response = chat_completion(
                base_url=base_url,
                model=model,
                messages=[
                    {"role": "system", "content": "Inspect the supplied image. Return one color word only."},
                    {"role": "user", "content": [
                        {"type": "text", "text": "What is the dominant color?"},
                        {"type": "image_url", "image_url": {"url": red_png}},
                    ]},
                ],
                temperature=0.0,
                max_tokens=20,
                timeout=timeout,
                api_key=api_key,
                cancel_check=cancel_check,
            ).strip()
            results.append({"case": "vision", "passed": "red" in response.lower(), "seconds": round(time.monotonic() - case_started, 3), "response": response})
        except RuntimeError as exc:
            results.append({"case": "vision", "passed": False, "seconds": round(time.monotonic() - case_started, 3), "error": str(exc)})
    passed_count = sum(bool(item.get("passed")) for item in results)
    return {
        "model": model,
        "score": round(100 * passed_count / len(results)),
        "passed": passed_count,
        "total": len(results),
        "seconds": round(time.monotonic() - started, 3),
        "cases": results,
        "created_at": utc_now(),
    }


def load_generator_profiles(value: object) -> dict[str, dict[str, object]]:
    profiles = json.loads(json.dumps(GENERATOR_PROFILE_DEFAULTS))
    if isinstance(value, dict):
        for name, profile in value.items():
            if isinstance(name, str) and name.strip() and isinstance(profile, dict):
                profiles[name.strip()] = profile
    return profiles


def enqueue_comfyui(
    *,
    server_url: str,
    workflow: dict[str, object],
    prompt: str,
    positive_node_id: str,
    client_id: str = "promptcorrector",
    timeout: float = 30,
) -> dict[str, object]:
    workflow_copy = json.loads(json.dumps(workflow))
    node = workflow_copy.get(str(positive_node_id))
    if not isinstance(node, dict):
        raise ValueError(f"Workflow does not contain node {positive_node_id}")
    inputs = node.setdefault("inputs", {})
    if not isinstance(inputs, dict):
        raise ValueError(f"Workflow node {positive_node_id} has no inputs object")
    inputs["text"] = prompt
    payload = json.dumps({"prompt": workflow_copy, "client_id": client_id}).encode("utf-8")
    endpoint = server_url.rstrip("/") + "/prompt"
    request = urllib.request.Request(endpoint, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"ComfyUI enqueue failed: {exc}") from exc
    if not isinstance(result, dict):
        raise RuntimeError("ComfyUI returned an unexpected response")
    return result
