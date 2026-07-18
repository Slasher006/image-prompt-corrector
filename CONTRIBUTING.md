# Contributing

Thanks for helping improve Image Prompt Corrector. Bug reports, focused fixes,
tests, and documentation improvements are welcome.

## Development setup

Image Prompt Corrector requires Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

LM Studio is required for live prompt generation, image analysis, and model
benchmarks. It is not required for the unit test suite.

## Run checks

Run the complete offline test suite:

```bash
QT_QPA_PLATFORM=offscreen python -m unittest discover -v
```

Compile every application module:

```bash
python -m py_compile \
  action_emotion_presets.py \
  concept_presets.py \
  krea_prompt_corrector.py \
  krea_prompt_gui.py \
  mix_ingredient_presets.py \
  prompt_workbench.py \
  visual_direction_presets.py \
  workbench_gui.py
```

Before opening a pull request, also run:

```bash
git diff --check
```

## Change guidelines

- Preserve explicit user instructions and exact rendered-text contracts.
- Keep Prompt Corrector, Comic Story, and Meme Creator content isolated unless a
  setting is intentionally shared.
- Keep web and image research limited to factual or glossary support; reference
  material must not silently replace the requested scene, pose, composition, or
  story.
- Add or update tests for behavior changes. Network-dependent behavior should be
  mocked in unit tests.
- Add a description and an example to tooltips for new visible GUI controls.
- Do not commit `promptcorrector_settings.json`, `.ipcp` project bundles,
  generated media, model files, API keys, personal prompts, or local image
  paths.

## Pull requests

Keep each pull request focused. Describe the user-visible result, important
implementation choices, and the exact verification commands you ran. Include
screenshots for material GUI changes when practical.
