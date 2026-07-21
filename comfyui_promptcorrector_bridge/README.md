# ComfyUI PromptCorrector Bridge

This bundled custom node transfers finished text from Image Prompt Corrector
into ComfyUI without using the clipboard.

## Install

Copy or symlink `comfyui_promptcorrector_bridge` into ComfyUI's `custom_nodes`
directory, then restart ComfyUI:

```bash
ln -s /path/to/image-prompt-corrector/comfyui_promptcorrector_bridge \
  /path/to/ComfyUI/custom_nodes/comfyui_promptcorrector_bridge
```

The bridge automatically checks these settings locations:

- `PROMPTCORRECTOR_SETTINGS_PATH`, when set
- the parent Image Prompt Corrector checkout
- `~/promptcorrector/promptcorrector_settings.json`
- `~/image-prompt-corrector/promptcorrector_settings.json`

## Use

1. Add **PromptCorrector Bridge** from `text > PromptCorrector`.
2. Choose **Latest result**, **Prompt Corrector**, **Comic Story**, or
   **Meme Creator**.
3. Leave **Refresh on queue** selected to always output the newest saved result.
4. Connect the `prompt` output to a text encoder or any other `STRING` input.

Use **Pull latest corrected prompt** to copy the selected saved result into the
visible multiline field. Switch to **Use displayed text** when you want ComfyUI
to keep manual edits made in that field.

PromptCorrector's result panes also provide **Send to ComfyUI**. This saves the
visible result and immediately updates every open bridge node set to either
**Latest result** or the matching workspace. The node's transfer mode is not
changed: **Refresh on queue** remains available and continues to load the newest
saved result whenever the workflow runs.

In PromptCorrector, enable **ComfyUI > Auto-send completed results** to push each
successful Prompt Corrector, Comic Story, or Meme Creator result automatically.
Enable **Queue workflow after sending** to queue the currently open ComfyUI
workflow after its matching bridge node is updated. The browser bridge waits
briefly for ComfyUI to commit and serialize the new widget value before it sends
the queue command. For safety, the bridge skips automatic queueing if the open
workflow has no matching bridge
node. Keep only the ComfyUI page you intend to run connected when automatic
queueing is enabled.

## Privacy

The node definition keeps its default prompt empty so ComfyUI's global node
metadata does not expose saved text. The bridge endpoint returns only the
selected corrected result, its workspace label, and timestamp.
