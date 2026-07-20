import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

function matchingBridgeNodes(workspace) {
    return (app.graph?._nodes || []).filter((node) => {
        if (node.type !== "PromptCorrectorBridge") {
            return false;
        }
        const workspaceWidget = node.widgets?.find(
            (widget) => widget.name === "workspace",
        );
        const selected = workspaceWidget?.value || "Latest result";
        return selected === "Latest result" || selected === workspace;
    });
}

function updateBridgeNode(node, payload, status) {
    const promptWidget = node.widgets?.find(
        (widget) => widget.name === "prompt",
    );
    if (!promptWidget) {
        return;
    }
    promptWidget.value = payload.prompt;
    promptWidget.callback?.(payload.prompt);
    if (node.promptCorrectorBridgeButton) {
        node.promptCorrectorBridgeButton.name = status;
        window.setTimeout(() => {
            if (node.promptCorrectorBridgeButton) {
                node.promptCorrectorBridgeButton.name =
                    "Pull latest corrected prompt";
                node.setDirtyCanvas(true, true);
            }
        }, 2500);
    }
    node.setDirtyCanvas(true, true);
}

api.addEventListener("promptcorrector_bridge_prompt", ({ detail }) => {
    if (!detail?.prompt || !detail?.workspace) {
        return;
    }
    const nodes = matchingBridgeNodes(detail.workspace);
    for (const node of nodes) {
        updateBridgeNode(node, detail, `Pushed: ${detail.source}`);
    }
    app.graph?.setDirtyCanvas(true, true);
});

app.registerExtension({
    name: "promptcorrector.bridge",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "PromptCorrectorBridge") {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated
                ? onNodeCreated.apply(this, arguments)
                : undefined;
            const button = this.addWidget(
                "button",
                "Pull latest corrected prompt",
                null,
                async () => {
                    const promptWidget = this.widgets?.find(
                        (widget) => widget.name === "prompt",
                    );
                    const workspaceWidget = this.widgets?.find(
                        (widget) => widget.name === "workspace",
                    );
                    if (!promptWidget || !workspaceWidget) {
                        return;
                    }

                    const originalName = button.name;
                    button.name = "Pulling...";
                    this.setDirtyCanvas(true, true);
                    try {
                        const query = new URLSearchParams({
                            workspace: workspaceWidget.value,
                        });
                        const response = await api.fetchApi(
                            `/promptcorrector_bridge/latest?${query.toString()}`,
                            { cache: "no-store" },
                        );
                        const payload = await response.json();
                        if (!response.ok || !payload.ok) {
                            throw new Error(
                                payload.error || "PromptCorrector result is unavailable.",
                            );
                        }
                        promptWidget.value = payload.prompt;
                        promptWidget.callback?.(payload.prompt);
                        button.name = `Loaded: ${payload.source}`;
                        app.graph.setDirtyCanvas(true, true);
                    } catch (error) {
                        console.error("[PromptCorrector Bridge]", error);
                        button.name = "Pull failed - see console";
                        this.setDirtyCanvas(true, true);
                    }
                    window.setTimeout(() => {
                        button.name = originalName;
                        this.setDirtyCanvas(true, true);
                    }, 2500);
                },
            );
            this.promptCorrectorBridgeButton = button;
            this.size = [
                Math.max(this.size[0], 360),
                Math.max(this.size[1], 300),
            ];
            return result;
        };
    },
});
