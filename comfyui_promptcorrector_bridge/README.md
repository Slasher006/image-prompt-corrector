# ComfyUI PromptCorrector Bridge

This bundled custom node transfers finished text from Image Prompt Corrector
into ComfyUI without using the clipboard. Dedicated nodes also receive FLUX
edit references and MiniMax H3 I2V keyframes plus generation controls.

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

### FLUX.2 Klein image editing

1. Add **PromptCorrector FLUX Image Edit Bridge** from
   `image > PromptCorrector`.
2. Connect its `prompt` output to the FLUX.2 Klein text encoder.
3. Connect `reference_1`, `reference_2`, and `reference_3` to three VAE Encode
   and **Set Reference Latent** paths. Chain those reference-latent nodes into
   the model conditioning. Additional outputs are available through
   `reference_8`.
4. Open the main **FLUX Image Edit** tab in PromptCorrector, add and role-label
   the reference images, optionally paint masks, prepare the prompt, and send
   it.

PromptCorrector uploads the images to ComfyUI first. The connector event then
updates the FLUX bridge node with the uploaded filenames and waits for all
prompt/reference widget callbacks before an optional automatic queue. Each
reference is exposed separately as `IMAGE` plus its alpha-derived `MASK`.
PromptCorrector embeds painted masks into temporary RGBA transport copies, so
the original reference files remain unchanged. Unused outputs receive an empty
placeholder.

### MiniMax H3 image to video

1. Update ComfyUI and open its official `video_minimax_h3_i2v` template.
2. Add **PromptCorrector MiniMax H3 I2V Bridge** from
   `video > PromptCorrector`.
3. Connect `prompt`, `first_frame`, optional `last_frame`, `duration`, `width`,
   `height`, and `seed` to the matching inputs on ComfyUI's
   **Image to Video (MiniMax H3)** subgraph.
4. In PromptCorrector's **MiniMax H3 I2V** tab, choose the exact opening frame,
   optionally choose an ending frame, prepare or locally correct the motion and
   native-audio prompt, then click **Send H3 I2V to ComfyUI**.

The workspace defaults to 864 x 480 for a practical local low-VRAM starting
point and also exposes H3's 768-pixel-short-edge native landscape, portrait,
and square sizes. Duration is restricted to H3's 1-15 second range. The bridge
uploads one or two keyframes without changing the originals and keeps sampler
implementation details outside H3's natural-language prompt.

Official references: [MiniMax H3 overview](https://www.minimax.io/blog/minimax-h3),
[ComfyUI H3 I2V workflow](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_i2v.json),
and [Comfy-Org model files](https://huggingface.co/Comfy-Org/MiniMax-H3).

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

The node definitions keep their default prompt and reference filenames empty,
so ComfyUI's global node metadata does not expose saved content. FLUX reference
and H3 keyframe files are uploaded to ComfyUI's normal input directory; the push
event carries only their ComfyUI filenames, not local filesystem paths.
