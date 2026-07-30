"""ComfyUI node that pulls finished text from Image Prompt Corrector."""

from __future__ import annotations

import json
import importlib
import os
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any


WORKSPACE_CHOICES = (
    "Latest result",
    "Prompt Corrector",
    "Comic Story",
    "Meme Creator",
)
PUSH_WORKSPACE_CHOICES = (*WORKSPACE_CHOICES[1:], "FLUX Image Edit")
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
MAX_REFERENCE_IMAGES = 8
MAX_REFERENCE_NAME_CHARACTERS = 1024


class PromptCorrectorBridgeError(RuntimeError):
    """Raised when the PromptCorrector state cannot provide usable text."""


def validate_bridge_push_payload(payload: Any) -> dict[str, object]:
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
    if workspace not in PUSH_WORKSPACE_CHOICES:
        raise PromptCorrectorBridgeError(
            f"Unsupported pushed workspace: {workspace}"
        )
    queue_after_send = payload.get("queue_after_send", False)
    if not isinstance(queue_after_send, bool):
        raise PromptCorrectorBridgeError(
            "queue_after_send must be true or false."
        )
    result = {
        "prompt": prompt,
        "workspace": workspace,
        "source": workspace,
    }
    raw_references = payload.get("reference_images", [])
    if not isinstance(raw_references, list):
        raise PromptCorrectorBridgeError(
            "reference_images must be a list of uploaded ComfyUI filenames."
        )
    if len(raw_references) > MAX_REFERENCE_IMAGES:
        raise PromptCorrectorBridgeError(
            f"The bridge supports up to {MAX_REFERENCE_IMAGES} reference images."
        )
    references: list[str] = []
    for value in raw_references:
        if not isinstance(value, str) or not value.strip():
            raise PromptCorrectorBridgeError(
                "Every reference image must have an uploaded ComfyUI filename."
            )
        name = value.strip().replace("\\", "/")
        path = PurePosixPath(name)
        if (
            len(name) > MAX_REFERENCE_NAME_CHARACTERS
            or path.is_absolute()
            or ".." in path.parts
            or "\x00" in name
        ):
            raise PromptCorrectorBridgeError(
                "A reference image filename is unsafe or too long."
            )
        references.append(name)
    if references and workspace != "FLUX Image Edit":
        raise PromptCorrectorBridgeError(
            "Reference images may only be pushed to FLUX Image Edit."
        )
    if workspace == "FLUX Image Edit" and not references:
        raise PromptCorrectorBridgeError(
            "FLUX Image Edit requires at least one reference image."
        )
    if references:
        result["reference_images"] = references
    if queue_after_send:
        result["queue_after_send"] = True
    return result


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


def _empty_reference_image():
    try:
        torch = importlib.import_module("torch")
    except ImportError as exc:
        raise PromptCorrectorBridgeError(
            "PyTorch is unavailable inside ComfyUI."
        ) from exc
    return (
        torch.zeros((1, 1, 1, 3), dtype=torch.float32),
        torch.zeros((1, 1, 1), dtype=torch.float32),
    )


def load_comfyui_reference_image(filename: str):
    """Load one uploaded ComfyUI input image or return an empty placeholder."""

    filename = str(filename or "").strip()
    if not filename:
        return _empty_reference_image()
    try:
        comfy_nodes = importlib.import_module("nodes")
        loader = comfy_nodes.LoadImage()
        return loader.load_image(filename)
    except Exception as exc:
        raise PromptCorrectorBridgeError(
            f"Could not load ComfyUI reference image {filename}: {exc}"
        ) from exc


class PromptCorrectorFluxImageEditBridge:
    """Expose a FLUX edit prompt plus up to eight separately connectable images."""

    @classmethod
    def INPUT_TYPES(cls):
        required: dict[str, object] = {
            "prompt": (
                "STRING",
                {
                    "default": "",
                    "multiline": True,
                    "dynamicPrompts": True,
                    "tooltip": (
                        "FLUX edit prompt pushed from the Image Edit tab."
                    ),
                },
            ),
        }
        for index in range(1, MAX_REFERENCE_IMAGES + 1):
            required[f"reference_image_{index}"] = (
                "STRING",
                {
                    "default": "",
                    "tooltip": (
                        f"Uploaded ComfyUI input filename for reference Image {index}."
                    ),
                },
            )
        return {"required": required}

    RETURN_TYPES = (
        "STRING",
        "STRING",
        *(item for _index in range(MAX_REFERENCE_IMAGES) for item in ("IMAGE", "MASK")),
    )
    RETURN_NAMES = (
        "prompt",
        "source",
        *(
            item
            for index in range(1, MAX_REFERENCE_IMAGES + 1)
            for item in (f"reference_{index}", f"reference_{index}_mask")
        ),
    )
    FUNCTION = "transfer"
    CATEGORY = "image/PromptCorrector"
    DESCRIPTION = (
        "Receive a FLUX.2 Klein edit prompt and up to eight reference images "
        "from Image Prompt Corrector. Connect each IMAGE output to the VAE/"
        "reference-latent path used by your FLUX.2 Klein workflow."
    )
    SEARCH_ALIASES = [
        "flux image edit",
        "prompt corrector image bridge",
        "multi reference",
        "klein image edit",
    ]

    def transfer(
        self,
        prompt: str,
        reference_image_1: str = "",
        reference_image_2: str = "",
        reference_image_3: str = "",
        reference_image_4: str = "",
        reference_image_5: str = "",
        reference_image_6: str = "",
        reference_image_7: str = "",
        reference_image_8: str = "",
    ):
        cleaned_prompt = str(prompt or "").strip()
        if not cleaned_prompt:
            raise PromptCorrectorBridgeError(
                "The FLUX Image Edit prompt is empty. Send it from PromptCorrector."
            )
        filenames = (
            reference_image_1,
            reference_image_2,
            reference_image_3,
            reference_image_4,
            reference_image_5,
            reference_image_6,
            reference_image_7,
            reference_image_8,
        )
        if not any(str(filename or "").strip() for filename in filenames):
            raise PromptCorrectorBridgeError(
                "No FLUX Image Edit references were received."
            )
        outputs: list[object] = [cleaned_prompt, "FLUX Image Edit"]
        for filename in filenames:
            image, mask = load_comfyui_reference_image(filename)
            outputs.extend((image, mask))
        return tuple(outputs)

    @classmethod
    def IS_CHANGED(
        cls,
        prompt: str,
        reference_image_1: str = "",
        reference_image_2: str = "",
        reference_image_3: str = "",
        reference_image_4: str = "",
        reference_image_5: str = "",
        reference_image_6: str = "",
        reference_image_7: str = "",
        reference_image_8: str = "",
    ):
        return (
            str(prompt or ""),
            reference_image_1,
            reference_image_2,
            reference_image_3,
            reference_image_4,
            reference_image_5,
            reference_image_6,
            reference_image_7,
            reference_image_8,
        )


NODE_CLASS_MAPPINGS = {
    "PromptCorrectorBridge": PromptCorrectorBridge,
    "PromptCorrectorFluxImageEditBridge": PromptCorrectorFluxImageEditBridge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptCorrectorBridge": "PromptCorrector Bridge",
    "PromptCorrectorFluxImageEditBridge": "PromptCorrector FLUX Image Edit Bridge",
}
