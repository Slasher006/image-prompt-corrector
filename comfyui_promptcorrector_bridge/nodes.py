"""ComfyUI node that pulls finished text from Image Prompt Corrector."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


WORKSPACE_CHOICES = (
    "Latest result",
    "Prompt Corrector",
    "Comic Story",
    "Meme Creator",
)
TRANSFER_MODES = (
    "Refresh on queue",
    "Use displayed text",
)
WORKSPACE_KEYS = {
    "Prompt Corrector": "prompt",
    "Comic Story": "comic",
    "Meme Creator": "meme",
}
TOP_LEVEL_RESULT_KEYS = {
    "Prompt Corrector": "corrected_prompt",
    "Comic Story": "comic_result",
    "Meme Creator": "meme_result",
}
MAX_SETTINGS_BYTES = 20 * 1024 * 1024
MAX_PUSHED_PROMPT_CHARACTERS = 1_000_000


class PromptCorrectorBridgeError(RuntimeError):
    """Raised when the PromptCorrector state cannot provide usable text."""


def validate_bridge_push_payload(payload: Any) -> dict[str, str]:
    """Validate a result sent by the desktop app before broadcasting it."""

    if not isinstance(payload, dict):
        raise PromptCorrectorBridgeError("The push payload must be a JSON object.")
    prompt = payload.get("prompt")
    workspace = str(payload.get("workspace", "")).strip()
    if not isinstance(prompt, str) or not prompt.strip():
        raise PromptCorrectorBridgeError("The pushed prompt is empty.")
    prompt = prompt.strip()
    if len(prompt) > MAX_PUSHED_PROMPT_CHARACTERS:
        raise PromptCorrectorBridgeError(
            "The pushed prompt exceeds the bridge size limit."
        )
    if workspace not in WORKSPACE_CHOICES[1:]:
        raise PromptCorrectorBridgeError(
            f"Unsupported pushed workspace: {workspace}"
        )
    return {
        "prompt": prompt,
        "workspace": workspace,
        "source": workspace,
    }


def _resolve_settings_path(settings_path: Path | None = None) -> Path:
    if settings_path is not None:
        return Path(settings_path).expanduser()

    configured = os.getenv("PROMPTCORRECTOR_SETTINGS_PATH", "").strip()
    if configured:
        return Path(configured).expanduser()

    candidates = (
        Path(__file__).resolve().parent.parent / "promptcorrector_settings.json",
        Path.home() / "promptcorrector" / "promptcorrector_settings.json",
        Path.home() / "image-prompt-corrector" / "promptcorrector_settings.json",
    )
    return next((path for path in candidates if path.is_file()), candidates[1])


def _load_settings(settings_path: Path | None = None) -> dict[str, Any]:
    path = _resolve_settings_path(settings_path)
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise PromptCorrectorBridgeError(
            f"PromptCorrector settings were not found at {path}. Set "
            "PROMPTCORRECTOR_SETTINGS_PATH before starting ComfyUI if your "
            "PromptCorrector checkout is elsewhere."
        ) from exc
    if size > MAX_SETTINGS_BYTES:
        raise PromptCorrectorBridgeError(
            f"PromptCorrector settings are unexpectedly large ({size} bytes)."
        )
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PromptCorrectorBridgeError(
            f"PromptCorrector settings could not be read: {exc}"
        ) from exc
    if not isinstance(loaded, dict):
        raise PromptCorrectorBridgeError(
            "PromptCorrector settings do not contain a JSON object."
        )
    return loaded


def _history_prompt(
    settings: dict[str, Any],
    workspace: str | None,
) -> tuple[str, str, str]:
    history = settings.get("prompt_history", [])
    if not isinstance(history, list):
        return "", "", ""
    for entry in history:
        if not isinstance(entry, dict):
            continue
        entry_workspace = str(entry.get("workspace", "prompt")).strip().casefold()
        if workspace is not None and entry_workspace != workspace:
            continue
        prompt = str(
            entry.get("corrected_prompt", entry.get("prompt", ""))
        ).strip()
        if not prompt:
            continue
        source = {
            "prompt": "Prompt Corrector",
            "comic": "Comic Story",
            "meme": "Meme Creator",
        }.get(entry_workspace, "Prompt Corrector")
        return prompt, source, str(entry.get("created_at", "")).strip()
    return "", "", ""


def read_promptcorrector_result(
    workspace: str = "Latest result",
    *,
    settings_path: Path | None = None,
) -> dict[str, str]:
    """Read one finished prompt without exposing unrelated saved state."""

    if workspace not in WORKSPACE_CHOICES:
        raise PromptCorrectorBridgeError(f"Unsupported workspace: {workspace}")
    path = _resolve_settings_path(settings_path)
    settings = _load_settings(path)
    requested_workspace = WORKSPACE_KEYS.get(workspace)
    prompt, source, created_at = _history_prompt(settings, requested_workspace)

    if not prompt and workspace != "Latest result":
        prompt = str(settings.get(TOP_LEVEL_RESULT_KEYS[workspace], "")).strip()
        source = workspace
    if not prompt and workspace == "Latest result":
        for source_name in ("Prompt Corrector", "Comic Story", "Meme Creator"):
            candidate = str(
                settings.get(TOP_LEVEL_RESULT_KEYS[source_name], "")
            ).strip()
            if candidate:
                prompt = candidate
                source = source_name
                break
    if not prompt:
        raise PromptCorrectorBridgeError(
            f"No saved corrected text is available for {workspace}."
        )

    try:
        updated_ns = path.stat().st_mtime_ns
    except OSError:
        updated_ns = 0
    return {
        "prompt": prompt,
        "source": source or workspace,
        "created_at": created_at,
        "settings_updated_ns": str(updated_ns),
    }


class PromptCorrectorBridge:
    """Expose PromptCorrector output as a normal ComfyUI STRING connection."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": True,
                        "tooltip": (
                            "Corrected text pulled from Image Prompt Corrector. "
                            "You may edit it and choose Use displayed text."
                        ),
                    },
                ),
                "transfer_mode": (
                    list(TRANSFER_MODES),
                    {
                        "default": "Refresh on queue",
                        "tooltip": (
                            "Refresh on queue always uses the newest saved result. "
                            "Use displayed text keeps edits made in this node."
                        ),
                    },
                ),
                "workspace": (
                    list(WORKSPACE_CHOICES),
                    {
                        "default": "Latest result",
                        "tooltip": "Select which PromptCorrector workspace to pull.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "source")
    FUNCTION = "transfer"
    CATEGORY = "text/PromptCorrector"
    DESCRIPTION = (
        "Pull the latest corrected Prompt, Comic, or Meme text from Image Prompt "
        "Corrector and pass it to any ComfyUI STRING input."
    )
    SEARCH_ALIASES = [
        "prompt corrector",
        "corrected prompt",
        "transfer prompt",
        "prompt bridge",
    ]

    def transfer(
        self,
        prompt: str,
        transfer_mode: str = "Refresh on queue",
        workspace: str = "Latest result",
    ) -> tuple[str, str]:
        displayed = str(prompt or "").strip()
        if transfer_mode == "Use displayed text":
            if not displayed:
                raise PromptCorrectorBridgeError(
                    "The displayed prompt is empty. Pull a result or enter text."
                )
            return displayed, "Displayed text"

        try:
            result = read_promptcorrector_result(workspace)
        except PromptCorrectorBridgeError:
            if displayed:
                return displayed, "Displayed text fallback"
            raise
        return result["prompt"], result["source"]

    @classmethod
    def IS_CHANGED(
        cls,
        prompt: str,
        transfer_mode: str = "Refresh on queue",
        workspace: str = "Latest result",
    ):
        if transfer_mode == "Use displayed text":
            return str(prompt or "")
        try:
            result = read_promptcorrector_result(workspace)
        except PromptCorrectorBridgeError as exc:
            return f"unavailable:{exc}:{prompt}"
        return (
            result["settings_updated_ns"],
            workspace,
            result["prompt"],
        )


NODE_CLASS_MAPPINGS = {
    "PromptCorrectorBridge": PromptCorrectorBridge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptCorrectorBridge": "PromptCorrector Bridge",
}
