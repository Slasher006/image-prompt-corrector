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

api.addEventListener("promptcorrector_bridge_prompt", async ({ detail }) => {
    if (!detail?.prompt || !detail?.workspace) {
        return;
    }
    const nodes = matchingBridgeNodes(detail.workspace);
    for (const node of nodes) {
        updateBridgeNode(node, detail, `Pushed: ${detail.source}`);
    }
    app.graph?.setDirtyCanvas(true, true);
    if (!detail.queue_after_send) {
        return;
    }
    if (!nodes.length) {
        console.warn(
            "[PromptCorrector Bridge] Queue skipped because no matching bridge node is open.",
        );
        app.extensionManager?.toast?.add?.({
            severity: "warn",
            summary: "PromptCorrector queue skipped",
            detail: "No matching PromptCorrector Bridge node is open in this workflow.",
            life: 5000,
        });
        return;
    }
    try {
        await Promise.resolve();
        await app.queuePrompt();
        app.extensionManager?.toast?.add?.({
            severity: "success",
            summary: "PromptCorrector queued",
            detail: `${detail.source} was sent and the current workflow was queued.`,
            life: 3000,
        });
    } catch (error) {
        console.error("[PromptCorrector Bridge] Queue failed", error);
        app.extensionManager?.toast?.add?.({
            severity: "error",
            summary: "PromptCorrector queue failed",
            detail: error?.message || "ComfyUI could not queue the current workflow.",
            life: 6000,
        });
    }
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
