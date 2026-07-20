"""ComfyUI PromptCorrector Bridge custom-node package."""

from __future__ import annotations

from .nodes import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    PromptCorrectorBridgeError,
    read_promptcorrector_result,
)

WEB_DIRECTORY = "./web"


def _register_latest_prompt_route() -> None:
    try:
        from aiohttp import web
        from server import PromptServer
    except ImportError:
        return

    async def latest_prompt(request):
        workspace = request.query.get("workspace", "Latest result")
        try:
            result = read_promptcorrector_result(workspace)
        except PromptCorrectorBridgeError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=404)
        return web.json_response({"ok": True, **result})

    PromptServer.instance.routes.get(
        "/promptcorrector_bridge/latest"
    )(latest_prompt)


_register_latest_prompt_route()

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]
