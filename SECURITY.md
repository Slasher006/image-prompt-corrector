# Security Policy

## Supported versions

Security fixes are applied to the current `main` branch. This project does not
yet publish versioned releases with separate maintenance windows.

## Reporting a vulnerability

Please do not open a public issue for a suspected vulnerability. Use GitHub's
private vulnerability reporting for the repository when it is available, or
contact the maintainer through a private channel. Include reproduction steps,
the affected component, and the potential impact. Do not include real API keys,
private prompts, or personal images.

## Local data and network boundaries

PromptCorrector is a local desktop application, but some enabled features make
network requests:

- Prompt generation, chat, benchmarking, and image analysis send their inputs
  to the configured LM Studio server. A remote LM Studio host can therefore
  receive prompts, conversation history included in the active chat request,
  and reference images selected for analysis.
- Grounded research sends search terms to the selected public search and image
  providers.
- ComfyUI handoff sends the corrected prompt and selected API workflow to the
  configured ComfyUI server.

The app stores drafts, results, chat history, prompt history, presets, connection
settings, and local reference paths in `promptcorrector_settings.json`. Workbench
`.ipcp` bundles can contain copies of reference and generated images. Both are
excluded by the repository's `.gitignore`; review any files you deliberately
force-add before publishing them.

## Deployment recommendations

- Keep LM Studio and ComfyUI bound to localhost unless remote access is needed.
- If either service is exposed to a network, use firewall rules, an authenticated
  reverse proxy, or another trusted access-control layer.
- Set `LM_STUDIO_API_KEY` in the environment when the server requires a bearer
  token. Do not place credentials in source files or shared shell scripts.
- Keep operating-system CA certificates current. Research requests fail closed
  when TLS certificate validation fails.
- Treat prompts, chat history, reference images, generated images, exported
  presets, and project bundles as potentially sensitive user data.
