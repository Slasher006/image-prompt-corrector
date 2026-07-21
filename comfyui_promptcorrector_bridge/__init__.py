"""ComfyUI PromptCorrector Bridge custom-node package."""

from __future__ import annotations

from .nodes import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    PromptCorrectorBridgeError,
    read_promptcorrector_result,
    validate_bridge_push_payload,
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

    async def push_prompt(request):
        try:
            payload = await request.json()
            result = validate_bridge_push_payload(payload)
        except (ValueError, TypeError, PromptCorrectorBridgeError) as exc:
            return web.json_response(
                {"ok": False, "error": str(exc)},
                status=400,
            )
        PromptServer.instance.send_sync(
            "promptcorrector_bridge_prompt",
            result,
        )
        return web.json_response(
            {
                "ok": True,
                "workspace": result["workspace"],
                "characters": len(result["prompt"]),
                "queue_requested": bool(result.get("queue_after_send")),
            }
        )

    PromptServer.instance.routes.get(
        "/promptcorrector_bridge/latest"
    )(latest_prompt)
    PromptServer.instance.routes.post(
        "/promptcorrector_bridge/push"
    )(push_prompt)


_register_latest_prompt_route()

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]
