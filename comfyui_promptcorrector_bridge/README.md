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

## Privacy

The node definition keeps its default prompt empty so ComfyUI's global node
metadata does not expose saved text. The bridge endpoint returns only the
selected corrected result, its workspace label, and timestamp.
