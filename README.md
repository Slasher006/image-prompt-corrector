# Image Prompt Corrector

Image Prompt Corrector is a Python/Qt desktop app with a native dark interface. It uses an LM Studio chat model to turn rough input into a faithful prompt for either Krea 2 or FLUX.2 Klein 9B. The default workflow prioritizes instruction adherence over decorative expansion.

It is built for a workflow where LM Studio runs locally or on another machine, and Krea or ComfyUI runs separately.

The app is under active development. Its offline test suite covers the correction
contracts, mode isolation, preset libraries, GUI behavior, and Workbench helpers.
See [Contributing](CONTRIBUTING.md) for development checks and
[Security](SECURITY.md) for data-flow and deployment guidance.

## Features

- Dark Qt GUI with separate Prompt Corrector, Comic Story, Meme Creator, Model Chat, and Workbench modes.
- Task-focused control groups for the creative brief, model guidance, rewrite rules, quality, and research.
- LM Studio connection over localhost or network.
- Model dropdown populated from LM Studio's `/models` endpoint.
- Input-wide spelling and language cleanup for the draft, concepts, goal, focus, weighted terms, story beats, model instructions, and generation feedback.
- Entity-centered prompt organization that keeps an object beside its attributes, actions, position, and direct effects.
- Stronger prompt-specific creative direction with internal concept exploration instead of generic adjective inflation.
- Krea 2 style formatting with natural language instead of Stable Diffusion keyword soup.
- Exact, Improve, and Explore workflows, with fidelity-first Exact mode as the default.
- Generator target selector for **Krea 2** and **FLUX.2 Klein 9B**.
- Separate **Prompt Corrector**, **Comic Story**, **Meme Creator**, and **Model Chat** workspaces; comics and memes work with either generator.
- A global **Settings** drawer keeps shared generation, processing, web-research, and LM Studio connection controls outside the individual mode panes.
- Dedicated image-macro builder with coordinated style presets, humor tones, a scene brief, optional exact top and bottom caption positions, caption style, aspect ratio, and its own saved result.
- Creative-response brief that turns a pasted situation, message, or event into a tailored meme concept with model-invented scene and captions.
- Hard checks for explicit counts, object-side assignments, exclusions, quoted text, scripts/languages, required concepts, and panel mappings.
- Logic and plausibility checks for contradictions, unclear action, impossible framing, and prompt drift.
- Slang and vague phrasing translation into concrete visual language.
- Feeling interpretation that turns abstract emotions into visible expression, posture, gesture, lighting, palette, and framing cues.
- Optional model-first grounded web verification and reference image analysis.
- Knowledge checks for concepts, actions, objects, materials, places, characters, and style terms.
- Action and pose mechanics research when grounded verification and action enhancement are enabled.
- Concept keyword integration.
- Goal headline, focus field, story elements, model instructions, and generation feedback.
- Controlled story invention and extension for single images and multi-panel sequences.
- Output length, detail level, risk level, prompt preset, and Krea setting controls.
- Prompt pair history with saved settings per prompt.
- Autosaved draft recovery, live word/token estimates, and target-range warnings.
- A word-level before/after Changes view.
- Searchable, renameable, pinnable prompt history.
- Reusable custom setup presets with JSON import/export.
- Local reference-image drag-and-drop plus web-reference thumbnails.
- Direct multi-turn chat with the selected LM Studio model, including a system instruction and cancellable responses.
- A project-based Workbench that closes the prompt-to-image feedback loop.
- Generated-image visual audits with explicit pass/fail lists, timing diagnostics, and minimal repair prompts.
- Portable `.ipcp` project bundles containing prompts, versions, references, results, reviews, and media assets.
- Reference roles for identity, face, outfit, pose, composition, style, environment, palette, and props, with optional panel, crop, and mask metadata.
- Character bibles and per-panel continuity inspection for recurring identities, clothing, props, and other anchors.
- A visible contract dashboard for structure, counts, placement, exact text, exclusions, pose, clarity, and emphasis.
- Controlled A/B prompt variants with a recorded project winner.
- A drawable normalized composition canvas and exact speech-bubble, caption, sign, title, and sound-effect contracts.
- Resumable CSV batch correction with JSON result export.
- Loaded-model fidelity and vision benchmarking.
- Editable generator profiles and optional ComfyUI API-workflow handoff.
- Mutually exclusive persistent **Safe for work** and **Explicit adult (NSFW)** modes for prompt correction, comics, memes, batches, A/B variants, and generated-image review.

## Requirements

- Python 3.10 or newer.
- PySide6.
- LM Studio with the local server enabled.
- A loaded LM Studio model. The default model is:

```text
qwen3-vl-4b-instruct
```

Install the GUI dependency with:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate` and run the
GUI with `python krea_prompt_gui.py`.

## LM Studio Setup

1. Open LM Studio.
2. Load your model.
3. Start the local server.
4. Make sure the server exposes the OpenAI-compatible API.

Default local URL:

```text
http://127.0.0.1:1234/v1
```

For another machine on your network, use that machine's IP address or hostname in the app:

```text
192.168.1.50
```

Port:

```text
1234
```

On the LM Studio machine, allow network access and make sure the firewall allows port `1234`.

If the remote machine is running the model on CPU, increase **Timeout** in the host/port row. The GUI defaults to `600` seconds.

## Run The GUI

From this folder:

```bash
./launch_promptcorrector.sh
```

Fallback command:

```bash
python3 krea_prompt_gui.py
```

The GUI stores settings and prompt history in:

```text
promptcorrector_settings.json
```

This runtime file is intentionally ignored by Git because it can contain drafts,
results, chat history, connection settings, and local reference-image paths.
Workbench `.ipcp` bundles and extracted media folders are ignored for the same
reason. Back them up or share them deliberately, not as part of a source commit.

Window size is not restored by default, which works better with tiling window managers such as i3wm. Enable **View > Remember window size** if you want the app to save and restore its last size.

Program options are in the top menu bar:

- **File**: save, import or export setup presets, and exit safely (`Ctrl+S`, `Ctrl+Q`).
- **Edit**: undo, redo, or clear one specific creative workspace.
- **Create**: grouped Prompt Corrector, Comic Story, Meme Creator, and Model Chat actions.
- **Model**: LM Studio connection, rewrite and safety rules, and generation passes.
- **Research**: grounded web verification plus separate reference-analysis toggles for Prompt, Comic, and Meme.
- **Library**: History, persisted Activity diagnostics, and workspace-isolated References.
- **View**: show or hide shared settings and the Activity/History/References dock, and control window-size restore.

## Basic Workflow

1. Select **Krea 2** or **FLUX.2 Klein 9B** under **Generator**.
2. Use **Prompt Corrector** for one still image, **Comic Story** for a multi-panel page, or **Meme Creator** for an image macro with exact top text, bottom text, or both.
3. Leave **Workflow** on **Exact** when following the request closely matters most.
4. Enter the image brief, or choose a comic panel count and complete each visible panel editor.
5. Generate and copy the result.
6. To develop it further, click **Iterate result…** (`Ctrl+Shift+R`), describe the next change, and run another pass. The current result becomes the next draft while the active concepts, references, and settings remain in place; each successful iteration stays available in History.
7. Apply the separately displayed generator setup recommendation in the image tool itself.

Every **Invent** button uses text already entered in its own field as a mandatory creative seed: it preserves that idea while expanding, completing, or polishing it with compatible details from the other fields. Leave the target field blank to invent it from scratch. After a successful invention, the adjacent **Recall** button restores the exact value that field had before Invent replaced it, including an originally blank value. **Invent all panels** preserves every entered panel beat and creates only the blank beats around them; **Recall all** restores the complete pre-Invent panel set.

The **Presets…** button beside Visual direction opens a searchable, multi-select creative-direction library shared by Prompt Corrector, Comic Story, and Meme Creator. Its 420 standard entries cover mood and emotional tone, lighting, color palette, time and season, weather and atmosphere, composition and hierarchy, depth and focus, motion and energy, environment, surfaces and texture, art direction and genre, image finish and color grade, subject presentation, and optical effects. Combine entries from any categories, preview the generated direction, then replace the current field or append the selection to manual direction text. Each workspace keeps its direction and selected presets separate.

The **Concepts…** button beside Concepts opens a separate searchable content library in Prompt Corrector and Comic Story. Its 484 entries span 22 categories: character archetypes, professions, relationships, actions, life moments, animals, folklore, science fiction, environments, architecture, interiors, props, transport, natural forms, narrative conflicts, symbolism, historical worldbuilding, fashion, crafts, food, product and graphic design, and abstract spatial ideas. Select up to eight active concepts, preview the exact comma-separated field text, then replace the current concepts or append the selection to manual concepts. Prompt and comic selections are saved independently.

The **Actions…** and **Emotions…** buttons provide dedicated narrative preset libraries beside the Prompt Corrector Story beat, Comic Story Premise, and Meme Creator Scene fields. The action library contains 252 concrete visible beats in 14 categories covering movement, object interaction, conversation, work, discovery, conflict, rescue, sport, performance, daily life, travel, science, nature, and ceremony. The emotion library contains 216 visibly expressed reactions in 12 categories covering joy, affection, calm, wonder, confidence, surprise, fear, anger, grief, shame, distrust, and mixed or changing emotions. Select up to six presets from either library and append them without duplicates or replace the current narrative. Each workspace and library keeps its selection independently.

Enable **Safe for work** under **Advanced setup > Rules & research > Rewrite rules** or from the **Processing** menu when the final prompt must be non-explicit. The option preserves the main subject, identity, action, composition, and tone while converting nudity, exposed intimate anatomy, sexual activity, erotic or fetish framing, and graphic gore into concrete non-explicit visual alternatives. It is saved with settings, presets, and prompt history. The final compliance gate treats it as mandatory and performs deterministic cleanup if the selected local model ignores the instruction, but safety-policy labels and audience boilerplate are removed before the image prompt is shown.

Enable **Explicit adult (NSFW)** in the same location when adult content must be preserved without euphemizing, censoring, or automatically adding clothing. The corrector keeps the requested sexual action as one short literal visual core—plain generator-ready wording such as `dildo in vagina` remains plain—while expansion adds compatible setting, composition, lighting, and style around it. It does not automatically append toy geometry, insertion mechanics, or a multi-stage sexual progression. A 244-rule supported-language catalog can still normalize unclear adult slang and idioms while actor/receiver roles, contact targets, objects, and intensity remain fixed. It is mutually exclusive with **Safe for work**, persists with settings, presets, and prompt history, and is used by correction, comics, memes, controlled variants, batches, and generated-image review. Requests containing underage or ambiguous-age terms are rejected before a model call; clearly adult wording such as `adult`, `age 24`, or `young adult` remains valid. Internal mode and age-guard instructions are not copied into the visible image prompt.

While **Explicit adult (NSFW)** is enabled, the Actions, Emotions, Concepts, Visual direction, and percentage-mixer libraries gain 474 adult-only entries across 31 categories: 109 actions, 72 emotional reactions, 147 concepts, and 146 visual directions. They cover adult relationships, solo/partner/group scenarios, seduction, foreplay, sexual actions and positions, adult toys, consensual power exchange, pacing, climax, aftercare, expressions, settings, wardrobe, props, composition, lighting, palette, texture, and erotic art direction. These categories are omitted from every picker while Explicit adult mode is off.

Sexual content involving an underage or ambiguous-age subject is rejected before any model call in every mode, even when **Explicit adult (NSFW)** is disabled. Non-sexual scenes containing children remain supported.

Use **Advanced setup** only when you need concepts, research, custom rewrite rules, or model controls. **Improve** permits restrained polish. **Explore** permits new supporting ideas and story development.

Under **Advanced setup > Processing > Rewrite rules**, use **Rewrite rule strength** to reduce optional rewrite, polish, and advisory audit pressure when a result feels over-constrained. `100` applies all enabled guidance firmly; lower values allow progressively more flexibility. Explicit user requirements, requested counts and positions, exact quoted text, selected safety mode, and removal of private control language always remain strict.

## Model Chat

Open the **Model Chat** work-mode tab to talk directly to the selected LM Studio model without the prompt-correction instructions or repair passes.

1. Optionally change the system instruction, temperature, and maximum response tokens.
2. Type a message and click **Send**, or press `Ctrl+Shift+Enter` while the message editor is focused.
3. The answer appears as it streams. Continue sending messages for a multi-turn conversation; the complete visible conversation is sent with each follow-up.
4. Use **Stop** to cancel the active stream, **Copy last response** to copy the latest answer, or **New chat** to clear the conversation.

Chat preferences and up to 100 recent user/assistant messages are stored in `promptcorrector_settings.json`. Prompt correction and chat share the selected model, LM Studio connection, timeout, and one-request-at-a-time cancellation flow.

## Meme Creator

Open **Meme Creator** to build a meme without mixing its fields into the normal prompt editor. It supports two workflows:

- For a creative response, paste or summarize the cause under **Situation to respond to** and optionally explain the **Desired response**. Leave Scene, Top text, and Bottom text blank; the model invents the visual analogy and the strongest one- or two-caption structure.
- For a manual meme, leave the response situation blank, describe the Scene, and enter exact **Top text**, **Bottom text**, or both.
- In either workflow, choose a preset and humor tone, adjust the visual controls if needed, then click **Generate meme prompt**.
- Use **Invent top** or **Invent bottom** to ask the selected LM Studio model for one caption without generating the full meme prompt. An entered target caption is treated as a seed to improve; a blank target caption is invented from the current situation, desired response, scene, humor tone, caption style, and opposite caption.

Built-in presets include **Classic Sarcasm**, **Deadpan Irony**, **Relatable Reaction**, **Absurdist Chaos**, **Self-own**, **Wholesome Punchline**, and **Demotivational Irony**. Humor tones can also be selected independently: Auto, Sarcastic, Ironic, Deadpan, Dry observational, Absurdist, Self-deprecating, Wholesome, and Dark comedy. Choose **Custom** to keep a manually configured combination.

Every manually supplied caption is emitted as a quoted rendered-text contract. Meme generation uses a dedicated creative-director path instead of the normal fidelity ranker. In creative-response mode, the model internally explores multiple joke mechanisms, then must return a finished image prompt with a newly invented scene, one or two concise quoted captions, and explicit top or bottom placement. Smaller 2B/4B models receive a shorter one-line contract. Common outputs such as `TOP TEXT: ...`, `BOTTOM: ...`, curly quotes, Markdown captions, and scene/caption blocks separated by blank lines are normalized automatically. If two full attempts produce a usable scene but miss the caption contract, the best scene is retained for a caption-only repair instead of being discarded. Rejected candidates and their validation issues are recorded in Activity for diagnosis. An echoed or lightly edited production brief is rejected; the source context must not become visible image text. The response brief, meme draft, and result are saved in `promptcorrector_settings.json`.

Meme Creator and Comic Story share the selected model, generator, workflow, and processing behavior with Prompt Corrector. They do not inherit Prompt Corrector's concepts, concept mix, goal, focus, weighted terms, model instructions, generation feedback, or references. Each creative workspace now owns its own local references and reference-analysis toggle.

In **Prompt Corrector**, open **Single-image options** for creative direction, weighted words, model guidance, generation feedback, and reference-image analysis. Those content-bearing controls belong only to the single-image workspace. Use the top-level **Settings** drawer for controls shared across image-prompt modes.

## Workbench

The **Workbench** is the fifth main workspace. It keeps experimental and project-level tools away from the normal Prompt Corrector screen.

### Project and generated-image review

1. Create a corrected single-image or comic prompt.
2. Open **Workbench > Project & Review** and click **Sync current prompt**.
3. Add one or more images produced by Krea, FLUX, ComfyUI, or another generator.
4. Click **Audit selected/all results**.
5. Review the score, successful requirements, failures, warnings, and timing diagnostics.
6. Click **Use repair as feedback** to place the minimal revision in the existing Generation feedback field.

The visual audit uses the selected LM Studio model's vision support. It checks the generated image instead of merely re-reading the prompt. Counts, identity, viewpoint-aware pose, spatial side, props, exact text, exclusions, composition, style, and panel mapping are evaluated independently.

Projects can be saved as `.ipcp` bundles. A bundle includes its JSON project record and copies of currently available reference and result images. Opening a bundle extracts those media assets beside the bundle so it remains usable on another machine.

### References and continuity

Role-aware references state why an image is present. A face reference cannot silently become a composition template, and a style reference does not redefine character identity. References can target one comic panel and can include a normalized crop (`x,y,width,height`) or a separate mask image.

Character bibles store a name, recurring identity anchors, and forbidden drift. The continuity inspector checks every panel that names the character and reports missing anchors per panel.

### Contracts, variants, composition, and text

The contract dashboard exposes the deterministic checks already used by the final correction gate. The clarification tool asks at most three questions and only raises missing decisions that can materially change the image.

The A/B lab creates Faithful, Composition, Camera, and Atmosphere variants while explicitly locking counts, identities, positions, exclusions, rendered text, and required actions. A generated variant can be selected as the project winner and is retained in the version history.

On the composition canvas, choose a region type and label, click **Draw box**, then drag a rectangle. **Apply to model instructions** converts each box into normalized spatial constraints. The adjacent exact-text editor binds rendered text to its kind, speaker, panel, and placement.

### Batch correction

Paste or import CSV with a required `prompt` column and optional `id`, `goal`, and `focus` columns:

```csv
id,prompt,goal,focus
1,"red robot in rain","cinematic portrait","robot face"
2,"two cats beside a blue door","storybook scene","clear count and door color"
```

Batch correction runs the fidelity-first single-image path. Stop returns an active item to pending state, so **Run/resume** can continue it. Results and per-item errors can be exported as JSON.

### Model benchmark and integrations

The model benchmark probes exact instruction following, quoted text, subject-relative spatial language, panel mapping, and vision input. It reports individual timings and responses instead of hiding a failed case behind one score.

Generator profiles are editable JSON objects containing prompt style, negative-prompt capability, and external setup values. Applying one stores it in the project and adds its compatibility rule to Model instructions.

For ComfyUI, export a workflow in API format, choose its JSON file, enter the positive CLIP text node ID, and click **Enqueue current prompt**. PromptCorrector copies the corrected prompt into that node and posts the workflow to the configured `/prompt` endpoint. No ComfyUI dependency is required inside PromptCorrector.

For a pull-based workflow, install the bundled
`comfyui_promptcorrector_bridge` custom node. It exposes the newest saved
Prompt Corrector, Comic Story, or Meme Creator result as a normal ComfyUI
`STRING` output, with automatic queue-time refresh and an editable manual mode.
See [`comfyui_promptcorrector_bridge/README.md`](comfyui_promptcorrector_bridge/README.md)
for installation and usage.

The draft and latest result are autosaved while you work and restored after the next launch. The counters below both editors show word totals and approximate token usage. Use the **Changes** tab to inspect additions, removals, and replacements. Concise, Balanced, and Detailed remain qualitative length preferences. **Expanded** is a concrete 140–280-word contract; if a local model returns less than 140 words, the correction pipeline performs one targeted expansion repair. With **Creative enhancement**, Expanded must add prompt-specific visual development rather than paraphrasing or adjective padding.

The final corrected output contains only the image prompt. Krea generation controls are deliberately kept outside it. Krea Turbo is useful for fast iteration; Krea Medium or Large is the better final pass when prompt and style fidelity matter most.

For **FLUX.2 Klein 9B**, the corrector follows Black Forest Labs' documented priority order: main subject, key action, critical style, essential context, then secondary details. Klein does not include prompt upsampling, so the corrected prompt explicitly contains the necessary visual information rather than relying on automatic expansion. The separate setup recommendation uses the official distilled-model defaults of four inference steps and guidance `1.0`. The 9B weights are distributed under the FLUX Non-Commercial License. See the [official FLUX.2 overview](https://docs.bfl.ai/flux_2/flux2_overview) and [FLUX.2 Klein 9B model card](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B).

Models whose names advertise 4B parameters or fewer automatically use a streamlined correction path. Their main instruction is reduced to a compact contract, and **Audit and repair** uses a short concrete audit instead of the large-model free-form audit. The path stays capped at two calls unless a hard contract still requires one targeted final repair. Missing counts, changed positions, violated exclusions, lost quoted text, panel errors, or other hard contracts still trigger repair or a conservative deterministic fallback.

Some Qwen3.5 reasoning fine-tunes ignore `/no_think` and can consume the entire completion budget in `reasoning_content` without producing a prompt. The app detects this response explicitly. Prefer a Qwen3 VL instruct variant for prompt correction; the local `huihui-qwen3-vl-4b-instruct-abliterated@q8_0` model has been live-tested with the streamlined workflow.

## Important Fields

**Your prompt**

The rough image idea. This can be messy, misspelled, duplicated, or mixed with instructions.

Example:

```text
knight at castle gate, wounded, holding sword, torch, cinematic, make sure armor is historical, no watermark
```

**Goal headline**

A short intent anchor, like a newspaper headline. It helps keep the model from drifting away from the point of the image.

Example:

```text
A wounded knight reaches the last safe gate
```

**Focus**

What the final image should emphasize.

Example:

```text
injury, readable sword pose, torchlit armor
```

**Story elements**

Visual storytelling beats that should be visible in the final image. Use this for action, motion direction, cause and effect, reactions, prop interaction, environmental response, and staging.

Example:

```text
the knight bursts through the gate, arrows hit the shield, villagers recoil, dust rises from the courtyard
```

The **Prompt Corrector** workspace converts its optional Story beat into one readable still-image moment and forbids panels, gutters, storyboards, or sequential frames. If its draft asks for multiple panels, correction stops immediately and tells you to use the Comic Story workspace instead of silently dropping the sequence.

The **Comic Story** workspace creates a multi-panel page for either Krea 2 or FLUX.2 Klein 9B. Choose 2 to 12 panels and the interface immediately shows one required editor per panel. Separate controls define the metadata-only working title, premise, layout, reading order, page aspect ratio, recurring continuity anchors, and shared visual direction. Each panel editor accepts its own subject, action or reaction, framing, dialogue, and caption requirements.

Action poses are audited as compact physical chains instead of generic “correct anatomy” wording: camera view, torso direction, the active shoulder-to-hand or hip-to-foot chain, contact point, and weight-bearing limb. Subject-relative anatomical left/right stays separate from image-left/image-right frame placement.

Explicitly labelled descriptions are mandatory panel contracts. For example, `Panel 1: she finds the key` must remain in Panel 1; it cannot be omitted, replaced by an invented beat, merged into another panel, or moved to Panel 2. The correction pass, model audit, and deterministic final gate all check this mapping. Exact quoted dialogue is also validated inside its assigned panel rather than only somewhere on the page.

You do not need to type panel labels in the GUI. The Comic Story workspace numbers every visible editor automatically and assembles them into mandatory `Panel 1:`, `Panel 2:`, and later contracts before calling LM Studio.

Multi-panel example:

```text
three-panel comic strip, panel 1 the red-cloaked knight discovers the broken gate, panel 2 the same knight raises her shield as arrows strike, panel 3 she reaches the courtyard and says "Close the gate!"
```

**Invent and extend story** is disabled in the default Exact and Improve workflows. Explore enables it. When enabled, the model may fill gaps with a coherent setup, motivation, escalation, reaction, transition, consequence, and payoff while preserving core identities, world, tone, required outcome, and fixed panel count. At Expanded + Creative enhancement, supplied Single Image story elements are mandatory visible content and the model must make the larger causal arc legible through the chosen decisive moment.

For a single image, story development chooses the strongest decisive moment and implies the wider arc through expressions, damage, tracks, displaced objects, environmental response, and other visible before-and-after evidence. For multi-panel work, it distributes the extended beats across the requested panels while maintaining continuity.

**Weighted words**

Comma-separated words or short phrases that should receive stronger visual priority. Use Krea-friendly natural emphasis, not Stable Diffusion syntax. The app accepts `term:weight`, `term=weight`, or `term*weight`, including two-decimal values such as `1.15`, clamps weights to `0.1` through `3.0`, and tells LM Studio to express the priority through composition, lighting, framing, detail, or action binding.

With the cursor inside a weighted term, press `Ctrl+Up` to increase its weight by `0.05` or `Ctrl+Down` to decrease it by `0.05`. If the term has no weight yet, the shortcut adds one.

You can also select a word or phrase in **Your prompt** and press `Ctrl+Up` to add or increase that phrase in **Weighted words**. Press `Ctrl+Down` on the selected phrase to reduce its weight.

Weighted words are highlighted directly inside **Your prompt**. Stronger weights use stronger highlight colors, so you can see at a glance what the final prompt correction will emphasize.

Example:

```text
face:1.6, red cloak:1.3, torchlight:1.2
```

**Concepts**

Comma-separated concepts that must be integrated into the final prompt.

Example:

```text
medieval armor, ruined castle, torchlight
```

With grounded web verification enabled, the app first asks the selected model what it knows about these concepts, checks those claims against web-search evidence, and reconciles disagreements and uncertainty. Only the reconciled factual and glossary guidance enters the correction prompt; raw search snippets remain out of the final correction pass.

**Concept/style mix**

Open **Single-image options**, then click **Mix...** beside **Concept/style mix**. Add up to six ingredients, assign percentages that total 100%, and save. Ingredients can be concepts, art styles, materials, moods, lighting languages, or other visual influences.

Inside the percentage editor, **Browse library…** opens an exhaustive searchable catalog assembled from the app's content concepts, action presets, visible-emotion presets, rendering modes and media, and creative-direction presets. Its 1,453 standard entries are organized into 63 categories covering subjects, roles, relationships, concrete actions, visible emotional reactions, creatures, places, objects, narrative ideas, historical worldbuilding, fashion, crafts, design, photography, illustration, painting, 3D media, genres, mood, lighting, palette, weather, composition, focus, motion, texture, finish, presentation, and optical effects. Explicit adult mode adds the same 474 gated adult ingredients available in the dedicated preset libraries. Select up to six ingredients and either add them to existing custom rows or replace all mixer rows. A changed library blend is balanced to 100% automatically; every share remains editable before saving.

Example:

```text
Art Nouveau:60%, cyberpunk:25%, botanical architecture:15%
```

The percentages express relative creative influence, not a promise that an image generator can divide pixels or features with mathematical precision. The app adds every nonzero ingredient to the required concept list, maps its share into the existing visual-priority system, and gives LM Studio an explicit coherent-blend instruction. Larger shares should control more of the composition, palette, materials, lighting, shape language, and detail. The final paste-ready image prompt does not contain numeric weights or percentages.

For quick iteration, keep the draft prompt focused on the subject and action, then change only the mix percentages between runs. Save useful blends as setup presets.

**Temperature**

Controls LM Studio rewrite randomness. Lower values are steadier and more literal; higher values allow more variation. Exact defaults to `0.10`.

**Sampling seed**

Sampling is random by default. Enable **Use fixed seed** in **Settings → Generation** and enter an integer seed to make the complete LM Studio prompt-generation path reproducible. Audit, repair, and meme-retry passes use deterministic offsets from that seed, so they remain different from one another while repeating consistently on the next run. Change the seed to explore a different reproducible result. The CLI equivalent is `--seed 42`.

**Context tokens**

Controls the approximate token budget for supporting research, concept research, and image-analysis context sent into LM Studio. **Auto (recommended)** reads the selected model's actually loaded `context_length`, reserves room for core instructions, the draft, and generated output, and assigns at most one quarter of the window (capped at `8192`) to supporting context. The dropdown also provides fixed `4K`, `8K`, `16K`, `32K`, and `64K` overrides. When supporting context exceeds the resolved budget, each non-empty section is reduced proportionally.

This setting does not change the context window of a model already loaded by LM Studio. If automatic detection is unavailable, PromptCorrector safely falls back to a `4096` supporting-context budget.

**Krea settings**

These are generation controls, not visual prompt content. **Show generator setup recommendation** displays them beside the result and **Copy generator setup** copies them separately. They are never appended to the image prompt.

- **Creativity**: broad style freedom. Lower values keep the result closer to the prompt; higher values allow more interpretation.
- **Intensity**: how strongly Krea should push the visual impact. Higher values generally mean stronger contrast, bolder mood, and more forceful styling.
- **Complexity**: how much visual density Krea should allow. Higher values can support richer environments and more detail; lower values favor simpler, cleaner images.
- **Movement**: how much motion or dynamic energy Krea should suggest. Higher values fit action, gestures, wind, fabric movement, and dynamic framing; lower values fit still portraits, products, and calm scenes.

The three sliders use Krea-style values from `-100` to `100`. `0` is neutral. Exact uses `creativity=raw`; increasing creativity allows Krea to reinterpret or expand the prompt more freely.

This behavior follows Krea's own guidance: [Krea 2 API controls](https://www.krea.ai/blog/krea-2-api-launch) documents raw creativity as no prompt expansion, while [Krea 2 Turbo](https://www.krea.ai/blog/krea-2-turbo) positions Turbo for rapid ideation and Medium/Large for higher-quality final generation.

When FLUX.2 Klein 9B is selected, the Krea sliders are disabled because they do not apply. The result instead shows the fixed FLUX setup recommendation separately.

**Model instructions**

Non-visual instructions for how to rewrite the prompt. These guide the correction but should not be copied into the final Krea prompt.

Example:

```text
Keep it realistic and medieval. Fix logic problems. Do not add monsters or magic.
```

**Generation feedback**

Use this after a bad image result. Describe what went wrong, then run the prompt again.

Example:

```text
Previous result looked too fantasy and the sword pose was unclear.
```

## Creativity and Logical Grouping

The corrector now treats creativity as visual concept development rather than adding generic words such as `epic`, `cinematic`, or `highly detailed`. Balanced improvement selects one coherent direction and adds a few useful details. Creative enhancement internally compares multiple interpretations, chooses the strongest, and develops a prompt-specific motif, relationship, environmental consequence, material behavior, light interaction, or staging idea.

Related details are organized into entity-centered clusters. An object stays beside its material, attributes, action, position, and visual effects. For example, a light bulb is described together with its fixture, glass, color temperature, glow, illuminated surfaces, cast shadows, and reflections. The model then orders those clusters by visual hierarchy instead of preserving a scrambled draft order.

For scenes with multiple people, the corrector gives each person a stable gender-or-role plus position label and repeats that label before the person's actions and attributes. Ambiguous pronouns and dropped male/female identities are hard validation failures, so the final repair pass must replace them before the prompt is returned.

Every user-authored text field is checked for clear spelling and language errors before it affects the final prompt. Exact quoted rendered text is preserved character-for-character, and uncertain names, brands, fictional terms, foreign words, or specialist vocabulary are not silently changed.

Accidental model-generated script changes are cleaned from corrected prompts, Single Image and Comic **Invent** results, all-panel inventions, Meme fields, captions, and meme recovery attempts. Han/Chinese characters, Kana, Hangul, Cyrillic, Arabic, and Devanagari remain untouched when the user supplied that script in the relevant source text; only an unexpected script introduced by the model is removed. **Model Chat** remains multilingual and is not filtered.

## Risk Levels

**Strict cleanup**

Conservative cleanup. Best when you want the prompt preserved closely.

**Balanced improvement**

Default mode. Chooses one coherent interpretation and adds a small number of distinctive, connected visual details while preserving intent.

**Creative enhancement**

Internally explores multiple visual directions, selects the strongest, and develops a memorable but coherent prompt-specific idea while keeping the main subject and goal anchored.

## Prompt Presets

Presets tell the model which kind of prompt quality matters most:

- Auto
- Photoreal portrait
- Cinematic action
- Product shot
- Character design
- Architecture
- Graphic poster
- Historical accuracy

Example: use **Cinematic action** when body mechanics, motion, contact points, and camera timing matter.

## Grounded Web Verification

The app uses a model-first verification sequence before correction:

1. It asks the selected LM Studio model what it knows and where it is uncertain.
2. It identifies up to twelve knowledge-sensitive targets across explicit concepts, actions, pose mechanics, objects, materials, places, characters, styles, weighted terms, and other important visual words.
3. It runs targeted web searches for those terms.
4. It asks the model to compare its prior knowledge with the web evidence, correct disagreements, and preserve unresolved uncertainty.
5. It sends only the reconciled factual and glossary guidance into the final correction pass. Raw search snippets are retained for the activity log but cannot donate another page's scene or wording to the prompt.

When a prompt is vague, grounded verification also adds clarification queries for the likely meaning, missing visual decisions, subject, setting, lighting, composition, and concrete anchor terms already present.

Available research-related options:

- **Grounded web verification**
- **Analyze reference images**
- **Enhance actions**

Grounded verification may use web sources such as Wikipedia, Bing, and DuckDuckGo. Use **Search engine** to limit verification to one provider when speed or provider reliability matters. If **Enhance actions** is also enabled, the app adds deeper action and pose research for body mechanics, balance, contact points, weight shift, motion timing, and viewpoint-aware body orientation. Automatic reference-image analysis uses only explicit entries from **Concepts**; it never searches for a matching version of the complete draft scene. Safe automatic sources are Yandex Images, DuckDuckGo Images, and Wikipedia/Wikimedia. Gelbooru and Rule34 remain available only when explicitly selected. If a provider fails, the app logs the failure and continues with usable safe sources.

Use **Image source** to limit concept-image lookup to one provider. This is faster than **Auto (safe sources)** and avoids waiting on providers you do not want to use.

Reference image analysis requires a vision-capable model in LM Studio.

The shared **References** dock accepts PNG, JPEG, WebP, and GIF files by drag-and-drop or with **Add images**. Select Prompt Corrector, Comic Story, or Meme Creator at the top of the dock; each workspace keeps its own analysis toggle and up to eight local paths. Local images are intentional references, but may clarify only requested identity, material, or style traits; they are not scene templates. When local images are present, the app skips automatic web image lookup. Without local images, automatic image research runs only for explicit concepts available in that workspace. For every source, the vision model must separate allowed facts from rejected scene details, and only the allowed glossary section reaches correction. The source image's unrelated pose, action, camera, crop, composition, layout, object placement, background, setting, palette, lighting arrangement, text, and story are excluded.

The **Stop** button immediately releases the GUI for a new request, closes the active LM Studio streaming connection, and discards any partial result. Old workers and research requests remain invalidated, so a late result cannot overwrite the restarted request.

## History and Activity

Every successful Prompt Corrector, Comic Story, and Meme Creator generation is stored in the shared history list with a workspace label. Loading an entry switches to the correct workspace and restores its editable fields, result, and still-available local references without overwriting the other modes.

Prompt Corrector history restores:

- Original requested prompt
- Corrected prompt
- Goal headline
- Focus
- Story elements
- Invent and extend story
- Weighted words
- Concepts
- Model instructions
- Generation feedback
- Model
- Mode, detail, output length
- Risk level and preset
- Toggle states
- Krea settings and sliders

Comic history restores the title, premise, continuity, concepts, style and dialogue direction, layout, panel count, every panel beat, result, and references. Meme history restores the response brief, scene, focus, captions, humor settings, visual direction, result, and references.

Activity is a timestamped persisted diagnostic history rather than a one-run console. Filter it by Prompt Corrector, Comic Story, Meme Creator, or System. It records pre-model input rejections, model-call difficulties, candidate validation failures, optional-audit failures, repair attempts, deterministic fallbacks, and final hard-contract rejections. Entries summarize named validation issues without copying private model instructions into the log. Model errors identify the workspace, failed stage, low-level detail, and a concrete next action. A failed retry keeps the previous successful result visible.

You can search, load, copy, rename, pin, delete, or clear history entries. Pinned entries remain at the top of the visible list.

## Saved Setup Presets

The **Generation** tab includes a **Saved setup** manager. **Save as** captures the current prompt fields, correction toggles, generation controls, research choices, and Krea sliders. Presets can be loaded or deleted locally, and imported or exported as JSON for backup or sharing.

## CLI Usage

You can also use the prompt corrector from the terminal:

```bash
python3 krea_prompt_corrector.py --prompt "a knight at a castle gate, torchlight, fix anatomy"
```

FLUX.2 Klein target:

```bash
python3 krea_prompt_corrector.py \
  --target "FLUX.2 Klein 9B" \
  --prompt "a knight at a castle gate, torchlight, historically accurate armor"
```

Comic Story with FLUX.2 Klein:

```bash
python3 krea_prompt_corrector.py \
  --target "FLUX.2 Klein 9B" \
  --format "Comic Story" \
  --prompt "a fox gets lost in a neon city and finds its way home"
```

CLI defaults match the GUI's Exact workflow: strict cleanup, preserved wording, no story invention, temperature `0.1`, balanced detail, and automatic supporting-context sizing. Use `--context-tokens auto` (the default) or a numeric override such as `8192`.

Remote LM Studio example:

```bash
python3 krea_prompt_corrector.py \
  --base-url http://192.168.1.50:1234/v1 \
  --model qwen3-vl-4b-instruct \
  --prompt "a wounded knight at a castle gate" \
  --goal-headline "A wounded knight reaches the last safe gate" \
  --focus "torchlit armor and readable sword pose" \
  --concepts "medieval armor, ruined castle" \
  --risk-level "Balanced improvement" \
  --preset "Historical accuracy"
```

Read a prompt from a file:

```bash
python3 krea_prompt_corrector.py --file prompt.txt
```

Pipe a prompt through stdin:

```bash
printf "a cinematic knight at a castle gate" | python3 krea_prompt_corrector.py
```

## Useful CLI Options

- `--base-url`
- `--model`
- `--target` (`Krea 2` or `FLUX.2 Klein 9B`)
- `--format` (`Single Image` or `Comic Story`)
- `--prompt`
- `--file`
- `--timeout`
- `--context-tokens`
- `--mode`
- `--detail`
- `--output-length`
- `--risk-level`
- `--preset`
- `--concepts`
- `--focus`
- `--goal-headline`
- `--model-instructions`
- `--grounded-web-verification` (`--live-concept-research` remains as a compatible alias)
- `--search-engine`
- `--analyze-reference-images`
- `--image-source`
- `--audit-repair`
- `--enhance-actions`
- `--no-story-development`
- `--develop-story`
- `--allow-creative-rewrite`
- `--thinking-mode`
- `--include-krea-settings`
- `--show-generator-setup` (clearer alias for the legacy option above)

## Testing

The test suite is offline: it mocks LM Studio and research-provider traffic, so a
running model server is not required.

Run syntax checks for all application modules:

```bash
python3 -m py_compile \
  action_emotion_presets.py \
  concept_presets.py \
  krea_prompt_corrector.py \
  krea_prompt_gui.py \
  mix_ingredient_presets.py \
  prompt_workbench.py \
  visual_direction_presets.py \
  workbench_gui.py
```

Run unit tests:

```bash
QT_QPA_PLATFORM=offscreen python3 -m unittest discover -v
```

GitHub Actions runs the same checks on Python 3.10 and 3.12.

## Repository Layout

- `krea_prompt_corrector.py`: correction engine, LM Studio client, research,
  validation, repair, and CLI.
- `krea_prompt_gui.py`: main PySide6 desktop interface and saved-state handling.
- `prompt_workbench.py`: project bundles, generated-image review, contracts,
  batch processing, benchmarks, and ComfyUI helpers.
- `workbench_gui.py`: Workbench interface.
- `*_presets.py`: built-in visual direction, concept, action, emotion, and mix
  catalogs.
- `tests/`: offline unit and GUI-contract tests.
- `launch_promptcorrector.sh`: Linux/macOS launcher.

## Privacy and Network Access

PromptCorrector does not require a hosted PromptCorrector service, but it is not
strictly offline when network-backed features are enabled:

- The configured LM Studio server receives prompt or chat inputs and any
  reference images selected for model analysis.
- Grounded research sends search terms to the selected search and image
  providers.
- ComfyUI handoff sends the corrected prompt and workflow to the configured
  ComfyUI endpoint.

Keep LM Studio and ComfyUI on localhost unless remote access is intentional and
properly protected. Read [SECURITY.md](SECURITY.md) before using a remote or
network-exposed model server.

## Troubleshooting

**LM Studio timeout**

Use **Test LM Studio** in the GUI. If it fails, check that the LM Studio server is running and that the selected model is loaded. For CPU-only remote inference, raise the GUI **Timeout** value or pass `--timeout 900` or higher in CLI mode.

**Network LM Studio does not connect**

Use the remote machine IP or hostname, keep port `1234`, and allow the port through the firewall.

**Research fails with SSL errors**

The app does not bypass TLS certificate validation. Check the system clock and
operating-system CA certificates, or disable grounded web verification and use
concepts manually until the certificate problem is fixed.

**Final output contains notes or audit text**

The corrector has cleanup and final repair passes to prevent this. Keep **Audit and repair** enabled for stricter final output.

**Krea ignores a concept**

Put the concept in **Concepts**, not only in the draft prompt, and keep **Audit and repair** enabled.

## License

No license has been selected yet. Until a `LICENSE` file is added, normal
copyright restrictions apply. The repository owner should choose the intended
open-source or source-available terms before inviting reuse or contributions.
