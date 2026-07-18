#!/usr/bin/env python3
"""Qt GUI for correcting Krea 2 prompts with LM Studio."""

from __future__ import annotations

import base64
import difflib
import html
import json
import os
import re
from pathlib import Path
import threading
import urllib.parse
from datetime import datetime
import time

from PySide6.QtCore import QObject, QSize, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QActionGroup, QColor, QDesktopServices, QIcon, QPixmap, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from krea_prompt_corrector import (
    CONTEXT_TOKEN_AUTO,
    CONTEXT_TOKEN_MAX,
    CONTEXT_TOKEN_MIN,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    CREATIVE_SESSION_TTL_SECONDS,
    GENERATOR_TARGETS,
    CONTENT_FORMATS,
    CREATIVITY_LEVELS,
    DETAIL_LEVELS,
    OUTPUT_LENGTHS,
    PROMPT_PRESETS,
    PROMPT_MODES,
    REFERENCE_IMAGE_SOURCES,
    RISK_LEVELS,
    TEXT_RESEARCH_ENGINES,
    adjust_named_weighted_term,
    adjust_weighted_terms_text,
    analyze_reference_images,
    classify_prompt_parts,
    collect_action_pose_research,
    collect_integrated_concept_research,
    collect_reference_image_diagnostics,
    collect_targeted_prompt_research,
    collect_vague_prompt_research,
    concept_mix_instruction,
    concept_mix_to_concepts,
    concept_mix_to_weighted_terms,
    build_all_comic_panels_suggestion_messages,
    build_comic_field_suggestion_messages,
    build_invent_field_repair_messages,
    build_meme_caption_suggestion_messages,
    build_meme_field_suggestion_messages,
    build_single_image_field_suggestion_messages,
    chat_completion,
    estimate_audit_max_tokens,
    estimate_max_tokens,
    enforce_comic_speech_bubble_contract,
    extract_panel_descriptions,
    fetch_image_data_url,
    format_generator_recommendation,
    list_lm_studio_models,
    is_small_model,
    normalize_all_comic_panel_suggestions,
    normalize_and_validate_invent,
    canonicalize_saved_invent_value,
    normalize_lm_studio_base_url,
    parse_concepts,
    parse_concept_mix,
    parse_weighted_terms,
    post_chat_completion,
    probe_model_visual_knowledge,
    prompt_research_targets,
    reconcile_model_knowledge_with_web,
    resolve_comic_layout,
    slider_value,
    strip_unexpected_scripts,
    strip_private_prompt_guidance,
    unload_lm_studio_model,
    vague_prompt_issues,
    vague_prompt_needs_clarification_research,
)
from action_emotion_presets import (
    ACTION_PRESET_KEYS,
    ACTION_PRESETS,
    EMOTION_PRESET_KEYS,
    EMOTION_PRESETS,
    NARRATIVE_PRESET_LIMIT,
    format_narrative_presets,
    merge_narrative_text,
    narrative_preset_key,
)
from concept_presets import (
    CONCEPT_PRESET_KEYS,
    CONCEPT_PRESETS,
    CONCEPT_SELECTION_LIMIT,
    concept_preset_key,
    format_concept_presets,
    merge_concept_text,
)
from mix_ingredient_presets import (
    MIX_INGREDIENT_KEYS,
    MIX_INGREDIENT_LIMIT,
    MIX_INGREDIENT_PRESETS,
    format_mix_ingredient_names,
    mix_ingredient_key,
    mix_ingredient_keys_for_names,
)
from visual_direction_presets import (
    VISUAL_DIRECTION_PRESET_KEYS,
    VISUAL_DIRECTION_PRESETS,
    format_visual_direction_presets,
    visual_preset_key,
)
from workbench_gui import PromptWorkbench

WORKFLOW_PROFILES = ("Exact", "Improve", "Explore")
CAMERA_CONTROL_AUTO = "Auto (use prompt)"
CAMERA_CONTROL_PRESETS = (
    CAMERA_CONTROL_AUTO,
    "Eye-level medium shot, 50mm lens",
    "Low-angle medium-wide shot, 35mm lens",
    "High-angle wide shot, 24mm lens",
    "Wide establishing shot, 24mm lens",
    "Overhead top-down shot, 35mm lens",
    "Close-up portrait, 85mm lens",
    "Extreme close-up, macro lens",
    "Extreme close-up facial detail, 100mm macro lens",
    "Tight headshot, 105mm portrait lens",
    "Head-and-shoulders portrait, 85mm lens",
    "Bust portrait, 70mm lens",
    "Cowboy shot, 50mm lens",
    "Medium close-up, 70mm lens",
    "Medium shot, 50mm lens",
    "Medium-wide shot, 35mm lens",
    "Full-body shot, 50mm lens",
    "Wide full-scene shot, 28mm lens",
    "Extreme-wide environmental shot, 18mm lens",
    "Ultra-wide establishing shot, 14mm lens",
    "Environmental portrait, 35mm lens",
    "Detail insert shot, 100mm macro lens",
    "Two-shot, eye level, 50mm lens",
    "Three-person group shot, 35mm lens",
    "Large group portrait, elevated 28mm lens",
    "Over-the-shoulder shot, 50mm lens",
    "Point-of-view shot, natural 35mm perspective",
    "First-person hands-in-frame view, 24mm lens",
    "Selfie perspective, arm's-length 24mm lens",
    "Mirror-reflection composition, eye level, 50mm lens",
    "Straight-on symmetrical shot, 50mm lens",
    "Front three-quarter view, 50mm lens",
    "Rear three-quarter view, 50mm lens",
    "Side-profile view, 85mm lens",
    "Direct rear view, 50mm lens",
    "Low-angle hero shot, 24mm lens",
    "Extreme low worm's-eye view, 14mm lens",
    "Ground-level shot, 24mm lens",
    "Knee-height low shot, 35mm lens",
    "Hip-height candid shot, 35mm lens",
    "Shoulder-height natural view, 50mm lens",
    "High-angle medium shot, 50mm lens",
    "Bird's-eye view, 24mm lens",
    "Oblique aerial view, 35mm equivalent",
    "Drone establishing view, 24mm equivalent",
    "Satellite-style orthographic top-down view",
    "Dutch-angle medium shot, 35mm lens",
    "Subtle canted angle, 50mm lens",
    "Centered one-point perspective, 24mm lens",
    "Two-point architectural perspective, 24mm tilt-shift lens",
    "Three-point dramatic perspective, 18mm lens",
    "Isometric orthographic view, no lens distortion",
    "Orthographic front elevation, no perspective",
    "Orthographic side elevation, no perspective",
    "Orthographic top plan, no perspective",
    "Exploded isometric view, orthographic projection",
    "Ultra-wide 14mm rectilinear perspective",
    "Wide-angle 18mm perspective",
    "Wide-angle 24mm perspective",
    "Documentary 28mm perspective",
    "Natural wide 35mm perspective",
    "Normal 50mm perspective",
    "Short-telephoto 70mm perspective",
    "Portrait 85mm compression",
    "Portrait 105mm compression",
    "Telephoto 135mm compression",
    "Long-telephoto 200mm compression",
    "Fisheye 8mm circular perspective",
    "Fisheye 15mm full-frame perspective",
    "Anamorphic wide shot, 40mm lens, 2.39:1 framing",
    "Tilt-shift miniature-effect view, 45mm lens",
    "Split-diopter composition, near and far subjects sharp",
    "Shallow-focus close-up, 85mm f/1.4 lens",
    "Deep-focus wide shot, 28mm lens",
    "Rack-focus moment, foreground to background",
    "Tracking side view, 35mm lens",
    "Head-on tracking shot, 50mm lens",
    "Rear follow shot, 35mm lens",
    "Dolly-in shot, 50mm lens",
    "Dolly-out reveal, 35mm lens",
    "Dolly-zoom perspective effect, 50mm lens",
    "Orbiting three-quarter shot, 35mm lens",
    "Crane-down establishing shot, 24mm lens",
    "Crane-up reveal, 24mm lens",
    "Handheld documentary shot, 28mm lens",
    "Steadicam follow shot, 35mm lens",
    "Whip-pan action frame, 35mm lens",
    "Panning action shot, 70mm lens",
    "Locked-off tripod shot, 50mm lens",
    "Long-exposure locked-off shot, 35mm lens",
    "Security-camera high corner view, ultra-wide lens",
    "Doorbell-camera view, ultra-wide lens",
    "Body-camera first-person view, ultra-wide lens",
    "Dashcam windshield view, wide lens",
    "Paparazzi telephoto candid, 200mm lens",
    "Direct-flash snapshot, eye level, 35mm lens",
    "Phone-camera candid, 26mm equivalent lens",
    "Webcam view, eye level, wide lens",
    "Underwater wide shot, dome-port 16mm lens",
    "Half-over half-under waterline shot, wide lens",
    "Through-window layered composition, 50mm lens",
    "Through-doorway frame-within-frame, 35mm lens",
    "Reflection-dominant shot, subject seen in glass or water",
    "Foreground-occluded observational shot, 85mm lens",
)
CONTEXT_TOKEN_AUTO_LABEL = "Auto (recommended)"
CONTEXT_TOKEN_CHOICES = (
    CONTEXT_TOKEN_AUTO_LABEL,
    "4K",
    "8K",
    "16K",
    "32K",
    "64K",
)
CONTEXT_TOKEN_CHOICE_VALUES = {
    CONTEXT_TOKEN_AUTO_LABEL: CONTEXT_TOKEN_AUTO,
    "4K": 4_096,
    "8K": 8_192,
    "16K": 16_384,
    "32K": 32_000,
    "64K": 64_000,
}
MEME_CAPTION_STYLES = (
    "Classic bold white with black outline",
    "Clean bold sans-serif",
    "Black text on white caption bars",
    "Demotivational black frame with white serif text",
    "Impact uppercase with heavy black outline",
    "Bold yellow subtitles with black outline",
    "White subtitle text on translucent black bar",
    "Minimal lowercase sans-serif",
    "Condensed tabloid headline",
    "News chyron lower third",
    "Social post card with clean UI text",
    "Comic speech balloon lettering",
    "Retro comic burst lettering",
    "Handwritten marker caption",
    "Typewriter-style caption",
    "Monospace terminal caption",
    "Elegant serif editorial caption",
    "Neon outlined display text",
    "Sticker-style bubble lettering",
    "Plain white caption without outline",
)
MEME_ASPECT_RATIOS = (
    "1:1 square",
    "4:5 portrait",
    "3:4 portrait",
    "2:3 portrait",
    "9:16 vertical story",
    "5:4 landscape",
    "4:3 landscape",
    "3:2 landscape",
    "16:9 landscape",
    "21:9 ultrawide",
)
MEME_TONES = (
    "Auto",
    "Sarcastic",
    "Ironic",
    "Deadpan",
    "Dry observational",
    "Absurdist",
    "Self-deprecating",
    "Wholesome",
    "Dark comedy",
    "Playful",
    "Witty",
    "Sardonic",
    "Satirical",
    "Parodic",
    "Surreal",
    "Chaotic",
    "Cringe comedy",
    "Awkward",
    "Exasperated",
    "Triumphant",
    "Mock inspirational",
    "Nostalgic",
    "Cute",
    "Anti-joke",
    "Reaction-only",
)
MEME_PRESETS: dict[str, dict[str, str]] = {
    "Custom": {},
    "Classic Sarcasm": {
        "tone": "Sarcastic",
        "caption_style": "Classic bold white with black outline",
        "aspect_ratio": "1:1 square",
        "visual_direction": "A highly readable reaction image with an exaggerated but believable expression and immediate visual contrast.",
    },
    "Deadpan Irony": {
        "tone": "Ironic",
        "caption_style": "Clean bold sans-serif",
        "aspect_ratio": "1:1 square",
        "visual_direction": "Understated documentary-like realism, restrained expression, and a visual situation that quietly contradicts the caption.",
    },
    "Relatable Reaction": {
        "tone": "Dry observational",
        "caption_style": "Classic bold white with black outline",
        "aspect_ratio": "1:1 square",
        "visual_direction": "A familiar everyday situation with a clear expressive reaction and uncluttered composition.",
    },
    "Absurdist Chaos": {
        "tone": "Absurdist",
        "caption_style": "Clean bold sans-serif",
        "aspect_ratio": "4:5 portrait",
        "visual_direction": "Surreal escalation, deliberately unexpected object combinations, energetic framing, and a visually obvious punchline.",
    },
    "Self-own": {
        "tone": "Self-deprecating",
        "caption_style": "Black text on white caption bars",
        "aspect_ratio": "1:1 square",
        "visual_direction": "A candid personal-failure reaction with warm imperfection and an immediately understandable situation.",
    },
    "Wholesome Punchline": {
        "tone": "Wholesome",
        "caption_style": "Clean bold sans-serif",
        "aspect_ratio": "4:5 portrait",
        "visual_direction": "Warm expressive subjects, gentle lighting, friendly color, and an affectionate visual payoff.",
    },
    "Demotivational Irony": {
        "tone": "Ironic",
        "caption_style": "Demotivational black frame with white serif text",
        "aspect_ratio": "4:5 portrait",
        "visual_direction": "A centered faux-inspirational photograph presented with overly serious polish that contrasts with the disappointing punchline.",
    },
    "Distracted Choice": {
        "tone": "Witty",
        "caption_style": "Clean bold sans-serif",
        "aspect_ratio": "3:2 landscape",
        "visual_direction": "Three clearly separated roles communicate temptation, rejection, and distraction through gaze direction and body language, with generous label-safe space.",
    },
    "Two Buttons": {
        "tone": "Exasperated",
        "caption_style": "Sticker-style bubble lettering",
        "aspect_ratio": "4:3 landscape",
        "visual_direction": "A sweating decision-maker faces two equally prominent choices, with a simple close framing and unmistakable visual dilemma.",
    },
    "Expanding Brain": {
        "tone": "Satirical",
        "caption_style": "Clean bold sans-serif",
        "aspect_ratio": "4:5 portrait",
        "visual_direction": "A vertically escalating sequence of increasingly radiant, surreal mental states with clear stage separation and rising visual intensity.",
    },
    "Drake Approval": {
        "tone": "Playful",
        "caption_style": "Impact uppercase with heavy black outline",
        "aspect_ratio": "1:1 square",
        "visual_direction": "A two-tier rejection-versus-approval reaction using one consistent subject, strongly differentiated gestures, and label-safe areas beside each reaction.",
    },
    "Change My Mind": {
        "tone": "Sardonic",
        "caption_style": "Black text on white caption bars",
        "aspect_ratio": "16:9 landscape",
        "visual_direction": "A confident person sits behind a plainly visible debate table in an open public setting, with a provocative statement area and documentary realism.",
    },
    "Mock News": {
        "tone": "Satirical",
        "caption_style": "News chyron lower third",
        "aspect_ratio": "16:9 landscape",
        "visual_direction": "A polished breaking-news frame with a serious presenter, clear lower-third area, newsroom graphics, and a visual situation that undermines the official tone.",
    },
    "Corporate Stock Photo": {
        "tone": "Cringe comedy",
        "caption_style": "Clean bold sans-serif",
        "aspect_ratio": "4:3 landscape",
        "visual_direction": "Overly enthusiastic office stock photography with forced smiles, implausibly pristine props, bright flat lighting, and obvious caption-safe negative space.",
    },
    "Surreal Escalation": {
        "tone": "Surreal",
        "caption_style": "Bold yellow subtitles with black outline",
        "aspect_ratio": "4:5 portrait",
        "visual_direction": "A believable everyday scene disrupted by one impossible premise that escalates coherently into a visually immediate surreal punchline.",
    },
    "Chaotic Screenshot": {
        "tone": "Chaotic",
        "caption_style": "Social post card with clean UI text",
        "aspect_ratio": "9:16 vertical story",
        "visual_direction": "A dense but readable phone-native composition of alerts, reactions, and escalating consequences, organized around one obvious comic failure.",
    },
    "Wholesome Animal": {
        "tone": "Cute",
        "caption_style": "Minimal lowercase sans-serif",
        "aspect_ratio": "1:1 square",
        "visual_direction": "An expressive animal performs a small relatable action in soft natural light, with simple surroundings and an affectionate visual payoff.",
    },
    "Retro Comic Reaction": {
        "tone": "Parodic",
        "caption_style": "Retro comic burst lettering",
        "aspect_ratio": "4:5 portrait",
        "visual_direction": "Bold halftone print texture, dramatic reaction pose, limited vintage palette, dynamic speed lines, and a large integrated reaction burst.",
    },
    "Text Message Reaction": {
        "tone": "Awkward",
        "caption_style": "Social post card with clean UI text",
        "aspect_ratio": "9:16 vertical story",
        "visual_direction": "A clean phone-message exchange paired with one restrained facial reaction, readable hierarchy, and realistic awkward timing.",
    },
    "Terminal Humor": {
        "tone": "Dry observational",
        "caption_style": "Monospace terminal caption",
        "aspect_ratio": "16:9 landscape",
        "visual_direction": "A dark developer workstation, one conspicuous terminal failure, restrained screen glow, and an exhausted reaction framed with technical clarity.",
    },
    "Mock Inspiration": {
        "tone": "Mock inspirational",
        "caption_style": "Elegant serif editorial caption",
        "aspect_ratio": "4:5 portrait",
        "visual_direction": "Grand aspirational landscape photography paired with a conspicuously mundane setback, polished like a premium motivational campaign.",
    },
    "Anti-joke Minimal": {
        "tone": "Anti-joke",
        "caption_style": "Minimal lowercase sans-serif",
        "aspect_ratio": "1:1 square",
        "visual_direction": "An extremely literal, visually plain scene with neutral light and deliberate under-staging that makes the lack of a conventional punchline the joke.",
    },
    "Reaction GIF Frame": {
        "tone": "Reaction-only",
        "caption_style": "Plain white caption without outline",
        "aspect_ratio": "16:9 landscape",
        "visual_direction": "A single loop-ready reaction beat with an unmistakable expression, clean silhouette, simple background, and no rendered caption unless supplied.",
    },
}

COMIC_LAYOUT_PRESETS = (
    "Auto grid",
    "Horizontal strip",
    "Vertical strip",
    "2 x 1 grid",
    "1 x 2 grid",
    "2 x 2 grid",
    "3 x 2 grid",
    "2 x 3 grid",
    "3 x 3 grid",
    "4 x 3 grid",
    "3 x 4 grid",
    "Manga page",
    "Western comic page",
    "European album page",
    "Sunday newspaper strip",
    "Four-panel yonkoma strip",
    "Full-width tiers",
    "Large splash with inset panels",
    "Central splash with surrounding panels",
    "Asymmetric cinematic panels",
    "Diagonal action panels",
    "Borderless montage",
    "Double-page spread",
)
COMIC_READING_ORDER_PRESETS = (
    "Left to right, top to bottom",
    "Right to left, top to bottom",
    "Top to bottom",
    "Bottom to top",
)
COMIC_ASPECT_RATIO_PRESETS = (
    "1:1 square",
    "5:6 portrait",
    "4:5 portrait",
    "3:4 portrait",
    "2:3 portrait",
    "9:16 vertical",
    "A-series portrait",
    "US comic portrait",
    "5:4 landscape",
    "4:3 landscape",
    "3:2 landscape",
    "16:9 landscape",
    "21:9 ultrawide",
    "A-series landscape",
)


SETTINGS_PATH = Path(__file__).with_name("promptcorrector_settings.json")
PROMPT_HISTORY_LIMIT = 50
CUSTOM_PRESET_LIMIT = 30
DARK_STYLESHEET = """
QWidget {
    background-color: #10131a;
    color: #e7eaf0;
    font-family: Inter, "Noto Sans", sans-serif;
    font-size: 13px;
}
QMainWindow, QMenuBar, QMenu { background-color: #0d1016; }
QToolTip { background-color: #202735; color: #f2f4f8; border: 1px solid #59657a; padding: 6px; }
QMenuBar { border-bottom: 1px solid #262d3a; }
QMenuBar::item:selected, QMenu::item:selected { background: #293248; }
QGroupBox {
    border: 1px solid #2a3240;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 12px;
    font-weight: 600;
    color: #cdd5e5;
}
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; }
QLineEdit, QTextEdit, QListWidget, QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: #171c25;
    border: 1px solid #30394a;
    border-radius: 6px;
    padding: 6px 8px;
    selection-background-color: #6d5dfc;
}
QLineEdit:focus, QTextEdit:focus, QListWidget:focus, QComboBox:focus,
QSpinBox:focus, QDoubleSpinBox:focus { border: 1px solid #7c6cff; }
QLineEdit[readOnly="true"] { color: #8993a5; background: #141820; }
QPushButton {
    background-color: #222a38;
    border: 1px solid #354057;
    border-radius: 6px;
    padding: 7px 13px;
    font-weight: 600;
}
QPushButton:hover { background-color: #2c3547; border-color: #56627a; }
QPushButton:pressed { background-color: #1c2330; }
QPushButton:disabled { color: #626b7b; background-color: #181d26; }
QPushButton#primaryButton { background-color: #6d5dfc; border-color: #8174ff; color: white; }
QPushButton#primaryButton:hover { background-color: #7a6bff; }
QTabWidget::pane { border: 1px solid #2a3240; border-radius: 7px; top: -1px; }
QTabBar::tab { background: #161b24; padding: 8px 14px; border: 1px solid #2a3240; }
QTabBar::tab:selected { background: #242b3a; color: #ffffff; border-bottom-color: #6d5dfc; }
QProgressBar { background: #171c25; border: 1px solid #30394a; border-radius: 4px; height: 8px; text-align: center; }
QProgressBar::chunk { background-color: #6d5dfc; border-radius: 3px; }
QSlider::groove:horizontal { height: 5px; background: #293142; border-radius: 2px; }
QSlider::handle:horizontal { width: 15px; margin: -5px 0; background: #7c6cff; border-radius: 7px; }
QCheckBox::indicator { width: 16px; height: 16px; }
QCheckBox::indicator:checked { background: #6d5dfc; border: 1px solid #9389ff; border-radius: 3px; }
QSplitter::handle { background: #242b38; width: 2px; }
QLabel#statusLabel { color: #a99fff; font-weight: 700; }
QScrollBar:vertical { background: #151a22; width: 10px; }
QScrollBar::handle:vertical { background: #3a4457; border-radius: 4px; min-height: 24px; }
"""
PROMPT_HISTORY_OPTION_KEYS = (
    "generator_target",
    "content_format",
    "workflow_profile",
    "camera_control",
    "model",
    "mode",
    "detail",
    "output_length",
    "risk_level",
    "prompt_preset",
    "lm_timeout",
    "variations",
    "temperature",
    "fixed_seed",
    "seed",
    "context_token_budget",
    "weighted_terms",
    "story_elements",
    "preserve_wording",
    "quote_rendered_text",
    "fix_logic",
    "enhance_actions",
    "develop_story",
    "clean_constraints",
    "safe_for_work",
    "explicit_nsfw",
    "altered_text_encoder",
    "thinking_mode",
    "live_research",
    "search_engine",
    "reference_image_analysis",
    "reference_image_source",
    "audit_repair",
    "include_krea_settings",
    "creativity",
    "intensity",
    "complexity",
    "movement",
)


UI_HELP: dict[str, tuple[str, str]] = {
    "LM Studio": ("Connection settings shared by prompt correction and chat.", "Use a model loaded on localhost:1234."),
    "Model": ("Choose or enter the LM Studio model identifier.", "Select qwen2.5-vl-7b-instruct."),
    "Host": ("Computer running the LM Studio API server.", "Use localhost for this computer."),
    "Port": ("Network port used by the LM Studio API.", "LM Studio normally uses 1234."),
    "Timeout": ("Maximum seconds to wait for one model request.", "Use 300 for a slower vision model."),
    "API URL": ("Resolved OpenAI-compatible API address.", "http://localhost:1234/v1"),
    "Test connection": ("Check the server and refresh the model list.", "Click after loading a model in LM Studio."),
    "Creative direction": ("Optional details that guide the rewritten prompt.", "Set Focus to dramatic rim lighting."),
    "Camera": (
        "Choose or type the camera framing applied to Single Image, Comic Story, and Meme Creator output.",
        "Select Low-angle medium-wide shot, 35mm lens.",
    ),
    "Concepts": ("Concepts the model must understand and preserve.", "Enter solarpunk, biophilic architecture."),
    "Goal headline": ("A one-line statement of the intended result.", "Enter heroic fantasy book cover."),
    "Focus": ("The most important visual subject or quality.", "Enter the knight's weathered armor."),
    "Story beat": ("Describe optional narrative evidence for the one still image.", "The knight reaches the gate as arrows strike her shield."),
    "Comic layout": ("Choose the physical arrangement of panels on the generated page.", "Use 2 x 2 grid for four equally sized panels."),
    "Comic reading order": ("Choose the order in which the panel sequence should be read.", "Use right to left for a manga-style page."),
    "Comic title": ("Optional working title that anchors the story without forcing visible cover text.", "The Last Gate"),
    "Comic page aspect ratio": ("Sets the full comic page shape so panel geometry has a stable canvas.", "4:5 portrait for a vertical four-panel page"),
    "Aspect ratio": ("Sets the full comic page shape so panel geometry has a stable canvas.", "4:5 portrait for a vertical four-panel page"),
    "Comic premise": ("Describe the cast, situation, setting, conflict, and intended outcome.", "A courier crosses a flooded city to deliver medicine."),
    "Comic continuity anchors": ("Define details that must remain stable in every relevant panel.", "Same red coat, brass satchel, rainy blue-hour city, bicycle always image-right."),
    "Comic concepts to integrate": (
        "List supporting visual concepts the model must incorporate without replacing the comic's story.",
        "Bioluminescent fungi, Stone Age astronomy, amber jewelry.",
    ),
    "Comic style direction": ("Set shared art, palette, lighting, camera, or lettering direction for the page.", "Clean European ink lines, muted watercolor, warm dialogue balloons."),
    "Comic dialogue direction": (
        "Control how the model writes newly invented character speech without changing visual narration.",
        "Make characters speak like cavemen using short grammar, simple words, and no modern slang.",
    ),
    "Comic speech bubbles": (
        "Allow invented or supplied dialogue to appear in concise, speaker-bound speech bubbles.",
        'A bubble reading "Stay behind me!" with its tail pointing clearly to the red-cloaked knight.',
    ),
    "Comic result": ("Shows the corrected generator-ready prompt for the complete comic page.", "Copy this result into Krea 2 or FLUX.2 Klein 9B."),
    "Comic Story": ("Construct a multi-panel comic request with one editor per panel.", "Choose four panels and describe one chronological beat in each editor."),
    "Page structure": ("Configure panel count, page layout, and reading direction.", "Use four panels in a 2 x 2 grid read left to right."),
    "Story and continuity": ("Define the overall story plus details shared across panels.", "Keep the same red coat and rainy city in every panel."),
    "Comic page prompt": ("Review the complete generator-ready comic page prompt.", "Copy the result into the selected generator."),
    "Meme Creator": ("Build a single image-macro prompt with an exact top caption, bottom caption, or both.", "Describe a surprised cat, then enter only the caption positions the joke needs."),
    "Invent top caption": (
        "Improve the entered top caption as a creative seed, or invent it from context when blank.",
        "Enter a rough setup line to refine, or leave it blank for a new caption.",
    ),
    "Invent bottom caption": (
        "Improve the entered bottom caption as a creative seed, or invent it from context when blank.",
        "Enter a rough punchline to refine, or leave it blank for a new caption.",
    ),
    "Invent meme situation": (
        "Expand the entered situation as a mandatory seed, or invent it from context when blank.",
        "Enter a rough situation to develop, or leave it blank for a new coherent meme setup.",
    ),
    "Invent meme response": (
        "Expand the entered response as a mandatory seed, or invent it from context when blank.",
        "Enter a rough stance to develop, or leave it blank for a reaction such as dry disbelief.",
    ),
    "Invent meme scene": (
        "Expand the entered scene as a mandatory seed, or invent it from context when blank.",
        "Enter a rough visual joke to develop, or leave it blank for a concrete still scene.",
    ),
    "Invent meme focus": (
        "Expand the entered focus as a mandatory seed, or invent it from context when blank.",
        "Enter a rough emphasis to refine, or leave it blank to derive one from the scene and joke.",
    ),
    "Invent meme visual direction": (
        "Expand the entered visual direction as a mandatory seed, or invent it from context when blank.",
        "Enter a rough treatment to refine, or leave it blank for a new style and camera direction.",
    ),
    "Meme setup": ("Define the meme canvas and caption appearance.", "Use a square canvas with classic bold outlined captions."),
    "Meme preset": ("Apply a coordinated humor tone, caption style, aspect ratio, and visual direction.", "Choose Classic Sarcasm or Deadpan Irony, then adjust individual controls if needed."),
    "Humor tone": ("Choose the comedic voice that should shape the relationship between image and caption.", "Use Sarcastic for mock praise or Ironic when the image contradicts the words."),
    "Meme temperature": (
        "Control LM Studio randomness for every Meme Creator request, including Invent buttons and final prompt generation.",
        "Use 0.3 for consistent wording, 0.7 for balanced invention, or 1.0 for wilder ideas.",
    ),
    "Creative response": ("Describe what prompted the meme and the reaction you want it to deliver.", "A coworker called a third emergency meeting about reducing meetings; respond with dry disbelief."),
    "Situation to respond to": ("Paste or summarize the message, event, claim, or behavior that the meme should answer.", "The manager announced another mandatory productivity meeting at 7 a.m."),
    "Desired response": ("Optionally state the point, stance, or emotional reaction the meme should communicate.", "Friendly sarcasm that says this will achieve the opposite of its goal."),
    "Meme content": ("Describe the visual joke and enter at least one exact caption.", "Scene: a dog at a laptop; bottom: NOW THERE ARE THREE."),
    "Meme scene": ("Describe the image underneath the captions.", "A confident orange cat presenting a chaotic spreadsheet."),
    "Meme focus": (
        "Name the subject, expression, action, prop, or joke detail that must receive the strongest visual emphasis.",
        "The exhausted employee's direct stare at the camera.",
    ),
    "Top text": ("Optionally enter the exact caption rendered across the top of the meme.", "WHEN THE MEETING COULD HAVE BEEN AN EMAIL"),
    "Bottom text": ("Optionally enter the exact caption rendered across the bottom of the meme.", "AND IT GETS A SEQUEL"),
    "Caption style": ("Choose how the top and bottom captions should look.", "Classic bold white text with a black outline."),
    "Meme aspect ratio": ("Choose the shape of the complete meme image.", "Use 1:1 square for a standard social-media meme."),
    "Meme visual direction": ("Add optional art, camera, lighting, or mood direction for the underlying image.", "Slightly awkward flash photography with a deadpan expression."),
    "Meme prompt": ("Review the generator-ready meme prompt with every supplied exact caption.", "Copy the prompt into Krea 2 or FLUX.2 Klein 9B."),
    "Generate meme prompt": ("Create a generator-ready prompt from the meme scene and supplied captions.", "Fill in the scene and at least one caption, then generate."),
    "Clear meme": ("Clear the meme scene, captions, visual direction, and result.", "Use before starting a different joke."),
    "Panels": ("Choose how many panel editors the comic workspace displays.", "Set 6 for a six-panel story."),
    "Layout": ("Choose how panels are arranged on the page.", "Select Horizontal strip for three panels in one row."),
    "Reading order": ("Choose the chronological direction of the panel sequence.", "Select right to left for manga ordering."),
    "Title": ("Optionally name the story to anchor its intent.", "The Last Gate"),
    "Premise": ("Describe the cast, setting, conflict, and intended outcome.", "A courier crosses a flooded city with medicine."),
    "Continuity anchors": ("List visual details that must remain stable between panels.", "Same coat, satchel, bicycle, and rainy blue-hour lighting."),
    "Visual direction": ("Describe the shared art style, palette, lighting, and lettering.", "Clean ink lines with muted watercolor."),
    "Panel beats": ("Describe one ordered visual event in each visible panel editor.", "Panel 1 establishes the character and conflict."),
    "Generate comic prompt": ("Assemble all comic fields and generate a validated page prompt.", "Fill four panel beats, then click Generate comic prompt."),
    "Copy result": ("Copy the generated comic prompt to the clipboard.", "Paste it into Krea 2 or FLUX.2 Klein 9B."),
    "Clear story": ("Clear the comic premise, continuity, panel beats, and result.", "Use before beginning a different comic."),
    "Model guidance": ("Extra instructions and feedback sent to the model.", "Emphasize motion and readable silhouettes."),
    "Weighted words": ("Terms to emphasize with prompt weights.", "Enter dragon:1.3, fog:0.8."),
    "Concept/style mix": ("Blend concepts or styles using relative percentages.", "Mix watercolor:60% with cyberpunk:40%."),
    "Mix...": ("Open the percentage mix editor.", "Set watercolor to 60% and cyberpunk to 40%."),
    "Model instructions": ("Direct instructions for how the model should rewrite.", "Keep the camera at eye level."),
    "Generation feedback": ("Describe what was wrong with an earlier image.", "The hands were hidden and the pose felt stiff."),
    "Mode": ("Choose the rewrite strategy for the prompt.", "Use Creative enhancement for richer invention."),
    "Detail": ("Control how much visual specificity is added.", "Use High for materials, lighting, and camera detail."),
    "Output length": ("Choose a general target size for the corrected prompt.", "Use Medium for a balanced prompt."),
    "Risk": ("Control how boldly the model may reinterpret the draft.", "Use Low when exact subject details matter."),
    "Creative freedom": ("Control how boldly the rewrite may develop the visual idea.", "Choose Creative enhancement for multiple internal concept directions."),
    "Artistic detail freedom": (
        "Let the model take bold control of secondary artistic details while preserving the requested subject, action, story, captions, and outcome.",
        "Enable for extravagant materials, surreal set dressing, unusual lighting, visual metaphors, and recurring motifs without changing the main scene.",
    ),
    "Preset": ("Apply a built-in prompt style profile.", "Choose Cinematic for film-like framing."),
    "Variations": ("Number of corrected prompt alternatives to request.", "Set 3 to compare three approaches."),
    "Temperature": ("Control response randomness; lower is more consistent.", "Use 0.3 for precise rewrites."),
    "Use fixed seed": ("Repeat LM Studio sampling with a reproducible seed instead of a random one.", "Enable it with seed 42 to reproduce the same prompt-generation path."),
    "Sampling seed": ("Integer seed sent to LM Studio when fixed-seed mode is enabled.", "Use 42, then change it to explore a different reproducible result."),
    "Context tokens": (
        "Auto reads the loaded model context and reserves room for instructions and output; fixed values override it.",
        "Keep Auto unless you need a known manual supporting-context limit.",
    ),
    "Saved setup": ("Select a saved collection of prompt settings.", "Choose Product shots to restore that setup."),
    "Creativity": ("Set the Krea creativity value included in the result.", "Choose Medium for balanced adherence."),
    "Intensity": ("Set the strength of Krea's visual transformation.", "Use 25 for a modest effect."),
    "Complexity": ("Set how visually elaborate the Krea result should be.", "Use 40 for a detailed environment."),
    "Movement": ("Set the amount of implied motion in the Krea result.", "Use 60 for a fast action scene."),
    "Preserve wording strictly": ("Keep the draft's wording wherever possible.", "Enable for a branded phrase that must not change."),
    "Quote rendered text": ("Put text that must appear in the image in quotes.", "Turn SIGN: OPEN into a sign reading \"OPEN\"."),
    "Fix logic conflicts": ("Repair contradictory spatial, lighting, or action details.", "Resolve both noon sunlight and a pitch-black sky."),
    "Enhance actions": ("Make poses and actions more visually explicit.", "Expand running into a forward lean and trailing coat."),
    "Invent and extend story": ("Add coherent narrative beats when the prompt has story intent.", "Add an obstacle and reaction to a rescue scene."),
    "Clean Krea constraints": ("Remove syntax or instructions that do not help Krea.", "Remove unsupported parameter flags."),
    "Clean generator constraints": ("Remove syntax or instructions unsupported by the selected generator.", "Turn negative-prompt boilerplate into a clear desired scene state."),
    "Altered encoder safe": ("Prefer plain wording that survives altered text encoders.", "Use red metal helmet instead of nested syntax."),
    "Safe for work": ("Convert explicit sexual content, nudity, fetish framing, and graphic gore into non-explicit visual alternatives.", "Keep the same character and composition but use complete opaque clothing and non-graphic implied injury."),
    "Explicit adult (NSFW)": (
        "Preserve explicitly requested adult nudity or sexual content without euphemizing it; underage or ambiguous-age subjects are rejected.",
        "Keep requested explicit content between clearly adult subjects while preserving the scene and composition.",
    ),
    "Thinking mode": ("Allow a reasoning pass before producing the corrected prompt.", "Enable for a complex multi-panel story."),
    "Audit and repair": ("Run a second pass to check and repair the answer.", "Catch a missing panel or lost character detail."),
    "Include Krea settings": ("Show recommended Krea controls separately from the image prompt.", "Use creativity raw without adding it to the prompt text."),
    "Show Krea setup recommendation": ("Show recommended Krea controls separately from the image prompt.", "Use creativity raw without adding it to the prompt text."),
    "Show generator setup recommendation": ("Show controls for the selected generator separately from its prompt.", "Show four steps and guidance 1.0 for FLUX.2 Klein 9B."),
    "Unload model after correction": ("Ask LM Studio to unload the model when work finishes.", "Enable to free GPU memory after one correction."),
    "Grounded web verification": ("Compare model knowledge with web evidence before rewriting.", "Verify a historical object or martial-arts action."),
    "Search engine": ("Choose the source used for grounded text research.", "Use Auto to try available providers."),
    "Analyze reference images": ("Analyze local references, or use web images only as concept glossaries for explicit Concepts.", "Add a costume photo, or enter Art Nouveau in Concepts for concept-only image research."),
    "Image source": ("Choose where automatic concept-glossary images are found.", "Use Wikipedia/Wikimedia to research an explicit architecture concept."),
    "Your prompt": ("Write or paste the image prompt to improve.", "A knight running through a stormy castle courtyard."),
    "Correct prompt": ("Send the draft and enabled settings to the model.", "Click to rewrite the prompt in Your prompt."),
    "Stop": ("Cancel the active model request and discard partial output.", "Click while a long correction or chat response streams."),
    "Weight −": ("Decrease emphasis for the selected draft term.", "Change (fog:1.2) to (fog:1.1)."),
    "Weight +": ("Increase emphasis for the selected draft term.", "Change dragon to (dragon:1.1)."),
    "−": ("Decrease the weighted term at the cursor.", "Change fog:1.0 to fog:0.9."),
    "+": ("Increase the weighted term at the cursor.", "Change dragon:1.0 to dragon:1.1."),
    "Copy corrected": ("Copy the corrected prompt to the clipboard.", "Paste it into the selected image generator afterward."),
    "Iterate result": ("Use the current result as the next draft and refine it again.", "Ask for warmer lighting while keeping the existing subject and settings."),
    "Corrected prompt": ("Review and optionally edit the model's final prompt.", "Adjust one camera detail before copying."),
    "Activity": ("Shows research, image, connection, and generation events.", "Check here to confirm an image reached the model."),
    "History": ("Browse earlier prompt corrections and their settings.", "Load yesterday's pinned castle prompt."),
    "References": ("Manage local and discovered images used as visual context.", "Drop a costume PNG here for analysis."),
    "Chat settings": ("Configure direct conversation with the selected model.", "Set a system instruction for concise answers."),
    "System instruction": ("Define the model's role or response style for this chat.", "You are a concise storyboard editor."),
    "Max response tokens": ("Limit how many tokens the chat model may generate.", "Set 2048 for a detailed answer."),
    "Conversation": ("Read the current multi-turn model chat.", "Review the model's previous storyboard advice."),
    "Message": ("Write a direct message to the selected model.", "Explain how to stage this three-panel scene."),
    "Send": ("Send the message and stream the model's reply.", "Press after typing a question below."),
    "New chat": ("Clear the current conversation and start fresh.", "Use before changing to an unrelated topic."),
    "Copy last response": ("Copy the newest model reply to the clipboard.", "Paste the advice into your notes."),
    "Single-image direction": ("Set creative direction used only by Prompt Corrector.", "Name the concept, focus, and story beat for one still image."),
    "Single-image options": ("Shows Prompt Corrector-only direction, guidance, and reference controls.", "Open it to add a goal, weighted words, or a local reference image."),
    "Generation": ("Control shared rewrite length, detail, variation, and sampling.", "Choose detailed output with two variations."),
    "Krea controls": ("Set Krea-specific values that may accompany the prompt.", "Raise Movement for an action scene."),
    "Processing": ("Choose shared rewrite safeguards, quality passes, and web research.", "Enable audit and repair for every image-prompt mode."),
    "Connection": ("Configure the LM Studio server shared by every mode.", "Use a model loaded on localhost port 1234."),
    "Rewrite rules": ("Choose what the model may change or improve.", "Enable Fix logic conflicts for impossible staging."),
    "Quality and session": ("Control extra model passes and session cleanup.", "Enable Audit and repair for complex prompts."),
    "Web research": ("Configure grounded text research shared by image-prompt modes.", "Verify a historical artifact before rewriting."),
    "Single-image references": ("Configure visual references used only by Prompt Corrector.", "Analyze a costume photo for one still-image prompt."),
    "Prompt Corrector": ("Turn a rough single-image idea into a production-ready prompt.", "Rewrite one cinematic still-image brief."),
    "Model Chat": ("Talk directly to the selected model without prompt correction.", "Ask for story ideas or visual explanations."),
    "Result": ("Shows the corrected prompt returned by the model.", "Review the final wording before copying."),
    "Changes": ("Highlights differences between the draft and corrected prompt.", "Look for added lighting or pose details."),
}
for _panel_number in range(1, 13):
    UI_HELP[f"Panel {_panel_number}"] = (
        "Define the mandatory visual beat for this numbered panel.",
        f"Panel {_panel_number} shows one clear action, reaction, or transition.",
    )


def ui_tooltip(description: str, example: str) -> str:
    """Return the consistent two-line help text used throughout the UI."""

    return f"{description}\nExample: {example}"


WORKSPACE_LABELS = {
    "prompt": "Prompt Corrector",
    "comic": "Comic Story",
    "meme": "Meme Creator",
    "system": "System",
}


def classify_workflow_error(
    error: str,
    *,
    workspace: str = "prompt",
    stage: str = "",
) -> dict[str, str]:
    """Turn low-level model failures into stable, actionable UI diagnostics."""

    detail = re.sub(r"\s+", " ", str(error or "")).strip() or "Unknown workflow error."
    lowered = detail.casefold()
    title = "Prompt generation failed"
    category = "generation"
    next_step = "Review Activity for the failed stage, adjust the current inputs, and retry."

    if any(marker in lowered for marker in (
        "failed to connect",
        "connection refused",
        "could not connect",
        "no connection could be made",
        "lm studio is unreachable",
    )):
        title = "LM Studio is unavailable"
        category = "connection"
        next_step = "Start the LM Studio local server, load the selected model, then retry."
    elif "timed out" in lowered or "timeout" in lowered:
        title = "LM Studio timed out"
        category = "timeout"
        next_step = "Keep the current inputs, raise the LM Studio timeout or use a faster loaded model, then retry."
    elif any(marker in lowered for marker in (
        "reasoning_content",
        "reasoning-only",
        "hidden reasoning",
        "finish_reason: length",
        "finish reason 'length'",
        "token budget",
        "returned no prompt",
        "empty response",
        "no usable value",
        "did not invent a usable",
    )):
        title = "The model returned no usable result"
        category = "model-output"
        next_step = "Use a non-reasoning instruct model or increase the response-token budget, then retry."
    elif "hard fidelity contract" in lowered:
        title = "The prompt contract could not be repaired"
        category = "contract"
        next_step = "The input is preserved. Review the named contract issue in Activity, make the disputed identity, count, or position explicit, then retry."
    elif any(marker in lowered for marker in (
        "base64 encoded image",
        "image input",
        "vision model",
        "reference image",
    )):
        title = "Reference-image analysis failed"
        category = "reference"
        next_step = "Use a vision-capable loaded model, remove the failing reference, or disable reference analysis and retry."
    elif any(marker in lowered for marker in (
        "model not found",
        "no model",
        "not loaded",
        "unknown model",
    )):
        title = "The selected model is not loaded"
        category = "model"
        next_step = "Load the selected language or vision model in LM Studio, refresh the model list, then retry."
    elif any(marker in lowered for marker in (
        "not allowed",
        "requires unambiguously adult",
        "safe for work and explicit adult",
        "incomplete comic",
        "incomplete meme",
        "single image format accepts",
    )):
        title = "The current input needs attention"
        category = "input"
        next_step = "Correct the named input conflict; the current draft and previous result have been kept."

    workspace_label = WORKSPACE_LABELS.get(workspace, WORKSPACE_LABELS["prompt"])
    stage_label = re.sub(r"\s+", " ", stage).strip() or "Model request"
    return {
        "title": title,
        "category": category,
        "workspace": workspace_label,
        "stage": stage_label,
        "detail": detail,
        "next_step": next_step,
        "message": (
            f"Workspace: {workspace_label}\n"
            f"Stage: {stage_label}\n\n"
            f"{detail}\n\n"
            f"Next: {next_step}"
        ),
    }


class CorrectionCancelled(RuntimeError):
    """Internal signal used to stop the active correction workflow."""


class Value:
    """Small Tk-variable-style value holder used by the existing workflow code."""

    def __init__(self, value: object = None) -> None:
        self._value = value
        self._callbacks: list[object] = []

    def get(self):
        return self._value

    def set(self, value: object) -> None:
        self._value = value
        for callback in tuple(self._callbacks):
            callback(value)

    def trace_add(self, _mode: str, callback) -> None:
        self._callbacks.append(lambda _value: callback())

    def subscribe(self, callback) -> None:
        self._callbacks.append(callback)


class UiDispatcher(QObject):
    invoke = Signal(object, object)


class QtButton(QPushButton):
    def configure(self, *, state: str) -> None:
        self.setEnabled(state != "disabled")


class QtComboBox(QComboBox):
    def configure(self, *, values: list[str]) -> None:
        current = self.currentText()
        self.clear()
        self.addItems(values)
        self.setCurrentText(current)


class QtTextEdit(QTextEdit):
    """QTextEdit with the tiny text API used by the correction workflow."""

    increase_weight_callback = None
    decrease_weight_callback = None

    def get(self, _start: str, _end: str) -> str:
        return self.toPlainText()

    def delete(self, _start: str, _end: str) -> None:
        self.clear()

    def insert(self, index: str, text: str) -> None:
        if index == "end":
            cursor = self.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            cursor.insertText(text)
            self.setTextCursor(cursor)
        else:
            self.setPlainText(text)

    def configure(self, **_kwargs: object) -> None:
        return

    def see(self, _index: str) -> None:
        self.moveCursor(QTextCursor.MoveOperation.End)
        self.ensureCursorVisible()

    def keyPressEvent(self, event) -> None:
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if event.key() == Qt.Key.Key_Up and self.increase_weight_callback is not None:
                self.increase_weight_callback()
                event.accept()
                return
            if event.key() == Qt.Key.Key_Down and self.decrease_weight_callback is not None:
                self.decrease_weight_callback()
                event.accept()
                return
        super().keyPressEvent(event)


class QtWeightedLineEdit(QLineEdit):
    def __init__(self, text: str, increase_callback, decrease_callback) -> None:
        super().__init__(text)
        self._increase_callback = increase_callback
        self._decrease_callback = decrease_callback

    def keyPressEvent(self, event) -> None:
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if event.key() == Qt.Key.Key_Up:
                self._increase_callback()
                event.accept()
                return
            if event.key() == Qt.Key.Key_Down:
                self._decrease_callback()
                event.accept()
                return
        super().keyPressEvent(event)


class ChatInputEdit(QTextEdit):
    send_requested = Signal()

    def keyPressEvent(self, event) -> None:
        if (
            event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
            and event.modifiers() & Qt.KeyboardModifier.ControlModifier
            and event.modifiers() & Qt.KeyboardModifier.ShiftModifier
        ):
            self.send_requested.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class QtHistoryList(QListWidget):
    def curselection(self) -> tuple[int, ...]:
        row = self.currentRow()
        return (row,) if row >= 0 else ()

    def selection_set(self, index: int) -> None:
        self.setCurrentRow(index)

    def activate(self, index: int) -> None:
        self.setCurrentRow(index)


class ReferencePreviewList(QListWidget):
    filesDropped = Signal(list)

    def __init__(self) -> None:
        super().__init__()
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls() and any(url.isLocalFile() for url in event.mimeData().urls()):
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:
        paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        if paths:
            self.filesDropped.emit(paths)
            event.acceptProposedAction()
            return
        super().dropEvent(event)


class MessageBox:
    @staticmethod
    def showerror(title: str, text: str) -> None:
        QMessageBox.critical(None, title, text)

    @staticmethod
    def showwarning(title: str, text: str) -> None:
        QMessageBox.warning(None, title, text)

    @staticmethod
    def askyesno(title: str, text: str) -> bool:
        return QMessageBox.question(
            None, title, text, QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        ) == QMessageBox.StandardButton.Yes


messagebox = MessageBox()


class PromptCorrectorWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.controller: PromptCorrectorApp | None = None

    def closeEvent(self, event) -> None:
        if self.controller is not None and not self.controller.closing:
            self.controller.closing = True
            self.controller.cancel_event.set()
            self.controller.active_request_id += 1
            self.controller._save_settings()
        event.accept()


class PromptCorrectorApp:
    def __init__(self, root: QMainWindow) -> None:
        self.root = root
        self.root.setWindowTitle("Image Prompt Corrector")
        self.root.setMinimumSize(980, 720)

        self.model_var = Value(os.getenv("LM_STUDIO_MODEL", DEFAULT_MODEL))
        self.generator_target_var = Value("Krea 2")
        self.content_format_var = Value("Single Image")
        self.camera_control_var = Value(CAMERA_CONTROL_AUTO)
        self.comic_panel_count_var = Value(4)
        self.comic_layout_var = Value("Auto grid")
        self.comic_reading_order_var = Value("Left to right, top to bottom")
        self.comic_aspect_ratio_var = Value("4:5 portrait")
        self.comic_title_var = Value("")
        self.comic_premise_var = Value("")
        self.comic_continuity_var = Value("")
        self.comic_concepts_var = Value("")
        self.comic_visual_direction_var = Value("")
        self.comic_dialogue_direction_var = Value("")
        self.comic_speech_bubbles_var = Value(True)
        self.comic_panel_vars = [Value("") for _index in range(12)]
        self.meme_scene_var = Value("")
        self.meme_top_text_var = Value("")
        self.meme_bottom_text_var = Value("")
        self.meme_response_context_var = Value("")
        self.meme_response_goal_var = Value("")
        self.meme_focus_var = Value("")
        self.meme_preset_var = Value("Custom")
        self.meme_tone_var = Value("Auto")
        self.meme_temperature_var = Value(0.7)
        self.meme_caption_style_var = Value("Classic bold white with black outline")
        self.meme_aspect_ratio_var = Value("1:1 square")
        self.meme_visual_direction_var = Value("")
        self.workflow_profile_var = Value("Exact")
        self.base_url_var = Value(
            value=normalize_lm_studio_base_url(os.getenv("LM_STUDIO_BASE_URL", DEFAULT_BASE_URL))
        )
        self.lm_host_var = Value("127.0.0.1")
        self.lm_port_var = Value("1234")
        self.lm_timeout_var = Value(600)
        self.concepts_var = Value("")
        self.concept_mix_var = Value("")
        self.visual_direction_var = Value("")
        self.goal_headline_var = Value("")
        self.focus_var = Value("")
        self.weighted_terms_var = Value("")
        self.story_elements_var = Value("")
        self.model_instructions_var = Value("")
        self.mode_var = Value("Auto")
        self.detail_var = Value("Balanced")
        self.output_length_var = Value("Balanced")
        self.risk_level_var = Value("Strict cleanup")
        self.prompt_preset_var = Value("Auto")
        self.generation_feedback_var = Value("")
        self.variation_var = Value(1)
        self.temperature_var = Value(0.1)
        self.fixed_seed_var = Value(False)
        self.seed_var = Value(42)
        self.context_token_budget_var = Value(CONTEXT_TOKEN_AUTO_LABEL)
        self.chat_system_prompt_var = Value("You are a helpful assistant.")
        self.chat_temperature_var = Value(0.7)
        self.chat_max_tokens_var = Value(2048)
        self.preserve_var = Value(True)
        self.quote_text_var = Value(True)
        self.fix_logic_var = Value(True)
        self.enhance_actions_var = Value(False)
        self.develop_story_var = Value(False)
        self.artistic_detail_freedom_var = Value(False)
        self.clean_constraints_var = Value(True)
        self.safe_for_work_var = Value(False)
        self.explicit_nsfw_var = Value(False)
        self.safe_for_work_var.subscribe(
            lambda enabled: (
                self.explicit_nsfw_var.set(False)
                if enabled and self.explicit_nsfw_var.get()
                else None
            )
        )
        self.explicit_nsfw_var.subscribe(
            lambda enabled: (
                self.safe_for_work_var.set(False)
                if enabled and self.safe_for_work_var.get()
                else None
            )
        )
        self.altered_encoder_var = Value(True)
        self.thinking_mode_var = Value(False)
        self.live_research_var = Value(False)
        self.search_engine_var = Value("Auto (all engines)")
        self.reference_images_var = Value(False)
        self.comic_reference_images_var = Value(False)
        self.meme_reference_images_var = Value(False)
        self.reference_image_source_var = Value("Auto (safe sources)")
        self.audit_repair_var = Value(True)
        self.include_settings_var = Value(True)
        self.unload_after_generation_var = Value(False)
        self.remember_window_size_var = Value(False)
        self.creativity_var = Value("raw")
        self.intensity_var = Value(0)
        self.complexity_var = Value(0)
        self.movement_var = Value(0)
        self.status_var = Value("Ready")
        self.progress_var = Value(0.0)
        self.progress_text_var = Value("Idle")
        self.progress_active = False
        self.progress_stage = "Idle"
        self.progress_stage_started_at = 0.0
        self.weighted_highlight_after_id: QTimer | None = None
        self.cancel_event = threading.Event()
        self.active_request_id = 0
        self.active_request_workspace = "system"
        self.closing = False
        self.available_models: list[str] = []
        self.prompt_history: list[dict[str, object]] = []
        self.activity_log: list[dict[str, str]] = []
        self.active_activity_workspace = "system"
        self.chat_messages: list[dict[str, str]] = []
        self.chat_stream_text = ""
        self.custom_presets: dict[str, dict[str, object]] = {}
        self.visual_preset_selections: dict[str, list[str]] = {
            "prompt": [],
            "comic": [],
            "meme": [],
        }
        self.concept_preset_selections: dict[str, list[str]] = {
            "prompt": [],
            "comic": [],
        }
        self.narrative_preset_selections: dict[str, dict[str, list[str]]] = {
            destination: {"action": [], "emotion": []}
            for destination in ("prompt", "comic", "meme")
        }
        self.recovered_draft = ""
        self.recovered_corrected = ""
        self.recovered_comic_result = ""
        self.recovered_meme_result = ""
        self.recovered_workbench_project: dict[str, object] | None = None
        self.recovered_generator_profiles: dict[str, dict[str, object]] | None = None
        self.local_reference_paths: list[str] = []
        self.comic_reference_paths: list[str] = []
        self.meme_reference_paths: list[str] = []
        self.web_reference_candidates: dict[str, list[dict[str, str]]] = {
            "prompt": [],
            "comic": [],
            "meme": [],
        }
        self.draft_autosave_timer: QTimer | None = None
        self.model_combo: QtComboBox | None = None
        self.weighted_terms_entry: QLineEdit | None = None
        self.seed_spin: QSpinBox | None = None
        self.story_elements_entry: QtTextEdit | None = None
        self.single_image_invent_buttons: list[QtButton] = []
        self.visual_preset_buttons: dict[str, QtButton] = {}
        self.concept_preset_buttons: dict[str, QtButton] = {}
        self.narrative_preset_buttons: dict[str, QtButton] = {}
        self.invent_recall_buttons: dict[str, QtButton] = {}
        self.invent_recall_values: dict[str, str] = {}
        self.invent_recall_groups: dict[str, dict[str, str]] = {}
        self.pending_invent_recall: dict[int, dict[str, str]] = {}
        self.pending_invent_recall_groups: dict[int, str] = {}
        self.single_invent_research_cache: dict[str, object] | None = None
        self.comic_panel_groups: list[QGroupBox] = []
        self.comic_panel_editors: list[QtTextEdit] = []
        self.comic_invent_buttons: list[QtButton] = []
        self.comic_panel_invent_buttons: list[QtButton] = []
        self.comic_invent_all_button: QtButton | None = None
        self.comic_layout_preview_label: QLabel | None = None
        self.comic_result_text: QtTextEdit | None = None
        self.comic_generate_button: QtButton | None = None
        self.comic_stop_button: QtButton | None = None
        self.meme_result_text: QtTextEdit | None = None
        self.meme_generate_button: QtButton | None = None
        self.meme_stop_button: QtButton | None = None
        self.meme_temperature_spin: QDoubleSpinBox | None = None
        self.meme_invent_buttons: list[QtButton] = []
        self.meme_response_context_button: QtButton | None = None
        self.meme_response_goal_button: QtButton | None = None
        self.meme_scene_button: QtButton | None = None
        self.meme_focus_button: QtButton | None = None
        self.meme_top_caption_button: QtButton | None = None
        self.meme_bottom_caption_button: QtButton | None = None
        self.meme_visual_direction_button: QtButton | None = None
        self.history_listbox: QtHistoryList | None = None
        self.history_search_entry: QLineEdit | None = None
        self.activity_scope_combo: QComboBox | None = None
        self.reference_workspace_combo: QComboBox | None = None
        self.reference_analysis_checkbox: QCheckBox | None = None
        self.library_tabs: QTabWidget | None = None
        self.library_dock: QDockWidget | None = None
        self.custom_preset_combo: QComboBox | None = None
        self.krea_recommendation_label: QLabel | None = None
        self.copy_krea_button: QPushButton | None = None
        self.generator_controls_page: QWidget | None = None
        self.generator_controls_tab_index: int | None = None
        self.prompt_guidance_page: QWidget | None = None
        self.prompt_options_button: QtButton | None = None
        self.setup_action = None
        self.reference_preview_list: QListWidget | None = None
        self.correct_button: QtButton | None = None
        self.stop_button: QtButton | None = None
        self.single_clear_button: QtButton | None = None
        self.iterate_button: QtButton | None = None
        self.chat_send_button: QtButton | None = None
        self.chat_stop_button: QtButton | None = None
        self.chat_transcript: QTextEdit | None = None
        self.chat_input: ChatInputEdit | None = None
        self.workbench_widget: PromptWorkbench | None = None
        self.request_in_progress = False
        self.dispatcher = UiDispatcher()
        self.dispatcher.invoke.connect(lambda callback, args: callback(*args))

        self._load_settings()
        self.generator_target_var.subscribe(self._apply_generator_target)
        self.content_format_var.subscribe(self._apply_content_format)
        self.workflow_profile_var.subscribe(self._apply_workflow_profile)
        self.meme_preset_var.subscribe(self._apply_meme_preset)
        self._build_ui()
        self._refresh_invent_recall_buttons()
        if self.recovered_draft:
            self.draft_text.setPlainText(self.recovered_draft)
            self.status_var.set("Recovered autosaved draft")
        if self.recovered_corrected:
            self.corrected_text.setPlainText(self.recovered_corrected)
        self._update_text_counters()
        self._update_diff_view(self.draft_text.toPlainText(), self.corrected_text.toPlainText())
        self.weighted_terms_var.trace_add("write", self._schedule_weighted_highlights)
        self._schedule_weighted_highlights()
        if SETTINGS_PATH.exists():
            self._refresh_model_list_in_background()

    def _settings_snapshot(self) -> dict[str, object]:
        workbench_state = self.workbench_widget.snapshot() if self.workbench_widget is not None else {
            "project": self.recovered_workbench_project,
            "generator_profiles": self.recovered_generator_profiles,
        }
        settings = {
            "generator_target": self.generator_target_var.get(),
            "content_format": self.content_format_var.get(),
            "camera_control": self.camera_control_var.get(),
            "comic_panel_count": self.comic_panel_count_var.get(),
            "comic_layout": self.comic_layout_var.get(),
            "comic_reading_order": self.comic_reading_order_var.get(),
            "comic_aspect_ratio": self.comic_aspect_ratio_var.get(),
            "comic_title": self.comic_title_var.get(),
            "comic_premise": self.comic_premise_var.get(),
            "comic_continuity": self.comic_continuity_var.get(),
            "comic_concepts": self.comic_concepts_var.get(),
            "comic_visual_direction": self.comic_visual_direction_var.get(),
            "comic_dialogue_direction": self.comic_dialogue_direction_var.get(),
            "comic_speech_bubbles": self.comic_speech_bubbles_var.get(),
            "comic_panels": [panel.get() for panel in self.comic_panel_vars],
            "comic_result": (
                self.comic_result_text.toPlainText()
                if self.comic_result_text is not None
                else self.recovered_comic_result
            ),
            "meme_scene": self.meme_scene_var.get(),
            "meme_top_text": self.meme_top_text_var.get(),
            "meme_bottom_text": self.meme_bottom_text_var.get(),
            "meme_response_context": self.meme_response_context_var.get(),
            "meme_response_goal": self.meme_response_goal_var.get(),
            "meme_focus": self.meme_focus_var.get(),
            "meme_preset": self.meme_preset_var.get(),
            "meme_tone": self.meme_tone_var.get(),
            "meme_temperature": self._meme_temperature(),
            "meme_caption_style": self.meme_caption_style_var.get(),
            "meme_aspect_ratio": self.meme_aspect_ratio_var.get(),
            "meme_visual_direction": self.meme_visual_direction_var.get(),
            "meme_result": (
                self.meme_result_text.toPlainText()
                if self.meme_result_text is not None
                else self.recovered_meme_result
            ),
            "workflow_profile": self.workflow_profile_var.get(),
            "model": self.model_var.get(),
            "available_models": self.available_models,
            "prompt_history": self.prompt_history,
            "activity_log": self.activity_log[-500:],
            "custom_presets": self.custom_presets,
            "invent_recall_values": self.invent_recall_values,
            "invent_recall_groups": self.invent_recall_groups,
            "draft_prompt": self.draft_text.toPlainText() if hasattr(self, "draft_text") else self.recovered_draft,
            "corrected_prompt": self.corrected_text.toPlainText() if hasattr(self, "corrected_text") else self.recovered_corrected,
            "local_reference_paths": self.local_reference_paths,
            "workspace_reference_paths": {
                "prompt": self.local_reference_paths,
                "comic": self.comic_reference_paths,
                "meme": self.meme_reference_paths,
            },
            "draft_saved_at": datetime.now().isoformat(timespec="seconds"),
            "base_url": self._current_base_url(),
            "lm_host": self.lm_host_var.get(),
            "lm_port": self.lm_port_var.get(),
            "lm_timeout": self._lm_timeout_seconds(),
            "concepts": self.concepts_var.get(),
            "concept_mix": self.concept_mix_var.get(),
            "concept_preset_selections": self.concept_preset_selections,
            "narrative_preset_selections": self.narrative_preset_selections,
            "visual_direction": self.visual_direction_var.get(),
            "visual_preset_selections": self.visual_preset_selections,
            "goal_headline": self.goal_headline_var.get(),
            "focus": self.focus_var.get(),
            "weighted_terms": self.weighted_terms_var.get(),
            "story_elements": self.story_elements_var.get(),
            "model_instructions": self.model_instructions_var.get(),
            "mode": self.mode_var.get(),
            "detail": self.detail_var.get(),
            "output_length": self.output_length_var.get(),
            "risk_level": self.risk_level_var.get(),
            "prompt_preset": self.prompt_preset_var.get(),
            "generation_feedback": self.generation_feedback_var.get(),
            "variations": self._variation_count(),
            "temperature": self._temperature_value(),
            "fixed_seed": self.fixed_seed_var.get(),
            "seed": self._configured_seed(),
            "context_token_budget": self._context_token_setting_value(),
            "chat_system_prompt": self.chat_system_prompt_var.get(),
            "chat_temperature": self._chat_temperature(),
            "chat_max_tokens": self._chat_max_tokens(),
            "chat_messages": self.chat_messages,
            "story_elements": self.story_elements_var.get(),
            "preserve_wording": self.preserve_var.get(),
            "quote_rendered_text": self.quote_text_var.get(),
            "fix_logic": self.fix_logic_var.get(),
            "enhance_actions": self.enhance_actions_var.get(),
            "develop_story": self.develop_story_var.get(),
            "artistic_detail_freedom": self.artistic_detail_freedom_var.get(),
            "clean_constraints": self.clean_constraints_var.get(),
            "safe_for_work": self.safe_for_work_var.get(),
            "explicit_nsfw": self.explicit_nsfw_var.get(),
            "altered_text_encoder": self.altered_encoder_var.get(),
            "thinking_mode": self.thinking_mode_var.get(),
            "live_research": self.live_research_var.get(),
            "search_engine": self.search_engine_var.get(),
            "reference_image_analysis": self.reference_images_var.get(),
            "comic_reference_image_analysis": self.comic_reference_images_var.get(),
            "meme_reference_image_analysis": self.meme_reference_images_var.get(),
            "reference_image_source": self.reference_image_source_var.get(),
            "audit_repair": self.audit_repair_var.get(),
            "include_krea_settings": self.include_settings_var.get(),
            "unload_after_generation": self.unload_after_generation_var.get(),
            "creativity": self.creativity_var.get(),
            "intensity": slider_value(self.intensity_var.get()),
            "complexity": slider_value(self.complexity_var.get()),
            "movement": slider_value(self.movement_var.get()),
            "remember_window_size": self.remember_window_size_var.get(),
            "workbench": workbench_state,
        }
        if self.remember_window_size_var.get():
            settings["geometry"] = f"{self.root.width()}x{self.root.height()}"
        return settings

    def _prompt_option_snapshot(self) -> dict[str, object]:
        return {
            "generator_target": self.generator_target_var.get(),
            "content_format": self.content_format_var.get(),
            "workflow_profile": self.workflow_profile_var.get(),
            "camera_control": self.camera_control_var.get(),
            "model": self.model_var.get(),
            "mode": self.mode_var.get(),
            "detail": self.detail_var.get(),
            "output_length": self.output_length_var.get(),
            "risk_level": self.risk_level_var.get(),
            "prompt_preset": self.prompt_preset_var.get(),
            "visual_direction": self.visual_direction_var.get(),
            "lm_timeout": self._lm_timeout_seconds(),
            "variations": self._variation_count(),
            "temperature": self._temperature_value(),
            "fixed_seed": self.fixed_seed_var.get(),
            "seed": self._configured_seed(),
            "context_token_budget": self._context_token_setting_value(),
            "preserve_wording": self.preserve_var.get(),
            "quote_rendered_text": self.quote_text_var.get(),
            "fix_logic": self.fix_logic_var.get(),
            "enhance_actions": self.enhance_actions_var.get(),
            "develop_story": self.develop_story_var.get(),
            "artistic_detail_freedom": self.artistic_detail_freedom_var.get(),
            "clean_constraints": self.clean_constraints_var.get(),
            "safe_for_work": self.safe_for_work_var.get(),
            "explicit_nsfw": self.explicit_nsfw_var.get(),
            "altered_text_encoder": self.altered_encoder_var.get(),
            "thinking_mode": self.thinking_mode_var.get(),
            "live_research": self.live_research_var.get(),
            "search_engine": self.search_engine_var.get(),
            "reference_image_analysis": self.reference_images_var.get(),
            "reference_image_source": self.reference_image_source_var.get(),
            "audit_repair": self.audit_repair_var.get(),
            "include_krea_settings": self.include_settings_var.get(),
            "creativity": self.creativity_var.get(),
            "intensity": slider_value(self.intensity_var.get()),
            "complexity": slider_value(self.complexity_var.get()),
            "movement": slider_value(self.movement_var.get()),
        }

    def _load_settings(self) -> None:
        if not SETTINGS_PATH.exists():
            return

        try:
            with SETTINGS_PATH.open("r", encoding="utf-8") as settings_file:
                settings = json.load(settings_file)
        except (OSError, json.JSONDecodeError):
            return

        self.invent_recall_values = self._invent_recall_values_setting(
            settings.get("invent_recall_values")
        )
        self.invent_recall_groups = self._invent_recall_groups_setting(
            settings.get("invent_recall_groups")
        )
        self.model_var.set(str(settings.get("model", self.model_var.get())))
        self.generator_target_var.set(
            self._choice_setting(
                settings.get("generator_target"),
                GENERATOR_TARGETS,
                self.generator_target_var.get(),
            )
        )
        self.content_format_var.set(
            self._choice_setting(
                settings.get("content_format"),
                CONTENT_FORMATS,
                self.content_format_var.get(),
            )
        )
        self.camera_control_var.set(
            str(settings.get("camera_control", CAMERA_CONTROL_AUTO)).strip()
            or CAMERA_CONTROL_AUTO
        )
        self.comic_panel_count_var.set(
            self._int_setting(settings.get("comic_panel_count"), 2, 12, 4)
        )
        self.comic_layout_var.set(
            self._choice_setting(
                settings.get("comic_layout"),
                COMIC_LAYOUT_PRESETS,
                "Auto grid",
            )
        )
        self.comic_reading_order_var.set(
            self._choice_setting(
                settings.get("comic_reading_order"),
                COMIC_READING_ORDER_PRESETS,
                "Left to right, top to bottom",
            )
        )
        self.comic_aspect_ratio_var.set(
            self._choice_setting(
                settings.get("comic_aspect_ratio"),
                COMIC_ASPECT_RATIO_PRESETS,
                "4:5 portrait",
            )
        )
        self.comic_title_var.set(str(settings.get("comic_title", "")))
        self.comic_premise_var.set(str(settings.get("comic_premise", "")))
        self.comic_continuity_var.set(str(settings.get("comic_continuity", "")))
        self.comic_concepts_var.set(str(settings.get("comic_concepts", "")))
        self.comic_visual_direction_var.set(str(settings.get("comic_visual_direction", "")))
        self.comic_dialogue_direction_var.set(
            str(settings.get("comic_dialogue_direction", ""))
        )
        self.comic_speech_bubbles_var.set(
            self._bool_setting(
                settings.get("comic_speech_bubbles"),
                self.comic_speech_bubbles_var.get(),
            )
        )
        self.recovered_comic_result = strip_private_prompt_guidance(
            str(settings.get("comic_result", ""))
        )
        stored_panels = settings.get("comic_panels", [])
        if isinstance(stored_panels, list):
            for index, value in enumerate(stored_panels[:12]):
                self.comic_panel_vars[index].set(str(value))
        self.meme_scene_var.set(str(settings.get("meme_scene", "")))
        self.meme_top_text_var.set(str(settings.get("meme_top_text", "")))
        self.meme_bottom_text_var.set(str(settings.get("meme_bottom_text", "")))
        self.meme_response_context_var.set(str(settings.get("meme_response_context", "")))
        self.meme_response_goal_var.set(str(settings.get("meme_response_goal", "")))
        self.meme_focus_var.set(str(settings.get("meme_focus", "")))
        self.meme_preset_var.set(
            self._choice_setting(
                settings.get("meme_preset"),
                tuple(MEME_PRESETS),
                "Custom",
            )
        )
        self.meme_tone_var.set(
            self._choice_setting(settings.get("meme_tone"), MEME_TONES, "Auto")
        )
        self.meme_temperature_var.set(
            self._float_setting(
                settings.get("meme_temperature"),
                0.0,
                2.0,
                self.meme_temperature_var.get(),
            )
        )
        self.meme_caption_style_var.set(
            self._choice_setting(
                settings.get("meme_caption_style"),
                MEME_CAPTION_STYLES,
                "Classic bold white with black outline",
            )
        )
        self.meme_aspect_ratio_var.set(
            self._choice_setting(
                settings.get("meme_aspect_ratio"),
                MEME_ASPECT_RATIOS,
                "1:1 square",
            )
        )
        self.meme_visual_direction_var.set(str(settings.get("meme_visual_direction", "")))
        self.recovered_meme_result = strip_private_prompt_guidance(
            str(settings.get("meme_result", ""))
        )
        stored_profile = settings.get("workflow_profile")
        profile = (
            stored_profile
            if isinstance(stored_profile, str) and stored_profile in WORKFLOW_PROFILES
            else "Exact"
        )
        self.workflow_profile_var.set(profile)
        self.available_models = self._model_list_setting(settings.get("available_models"))
        if self.model_var.get() and self.model_var.get() not in self.available_models:
            self.available_models.insert(0, self.model_var.get())
        self.prompt_history = self._history_setting(settings.get("prompt_history"))
        self.activity_log = self._activity_setting(settings.get("activity_log"))
        self.custom_presets = self._custom_presets_setting(settings.get("custom_presets"))
        stored_visual_presets = settings.get("visual_preset_selections", {})
        if isinstance(stored_visual_presets, dict):
            for destination in self.visual_preset_selections:
                values = stored_visual_presets.get(destination, [])
                if isinstance(values, list):
                    self.visual_preset_selections[destination] = [
                        str(value)
                        for value in values
                        if str(value) in VISUAL_DIRECTION_PRESET_KEYS
                    ]
        stored_concept_presets = settings.get("concept_preset_selections", {})
        if isinstance(stored_concept_presets, dict):
            for destination in self.concept_preset_selections:
                values = stored_concept_presets.get(destination, [])
                if isinstance(values, list):
                    self.concept_preset_selections[destination] = [
                        str(value)
                        for value in values
                        if str(value) in CONCEPT_PRESET_KEYS
                    ][:CONCEPT_SELECTION_LIMIT]
        stored_narrative_presets = settings.get("narrative_preset_selections", {})
        if isinstance(stored_narrative_presets, dict):
            valid_keys = {
                "action": ACTION_PRESET_KEYS,
                "emotion": EMOTION_PRESET_KEYS,
            }
            for destination in self.narrative_preset_selections:
                stored_destination = stored_narrative_presets.get(destination, {})
                if not isinstance(stored_destination, dict):
                    continue
                for kind in ("action", "emotion"):
                    values = stored_destination.get(kind, [])
                    if isinstance(values, list):
                        self.narrative_preset_selections[destination][kind] = [
                            str(value)
                            for value in values
                            if str(value) in valid_keys[kind]
                        ][:NARRATIVE_PRESET_LIMIT]
        self.recovered_draft = str(settings.get("draft_prompt", ""))
        self.recovered_corrected = strip_private_prompt_guidance(
            str(settings.get("corrected_prompt", ""))
        )
        stored_workbench = settings.get("workbench")
        if isinstance(stored_workbench, dict):
            project = stored_workbench.get("project")
            profiles = stored_workbench.get("generator_profiles")
            if isinstance(project, dict):
                self.recovered_workbench_project = project
            if isinstance(profiles, dict):
                self.recovered_generator_profiles = profiles
        stored_workspace_paths = settings.get("workspace_reference_paths", {})
        if not isinstance(stored_workspace_paths, dict):
            stored_workspace_paths = {}
        for workspace, attribute in (
            ("prompt", "local_reference_paths"),
            ("comic", "comic_reference_paths"),
            ("meme", "meme_reference_paths"),
        ):
            local_paths = stored_workspace_paths.get(
                workspace,
                settings.get("local_reference_paths", []) if workspace == "prompt" else [],
            )
            if isinstance(local_paths, list):
                setattr(
                    self,
                    attribute,
                    [
                        str(path)
                        for path in local_paths
                        if isinstance(path, str) and Path(path).is_file()
                    ][:8],
                )
        self.base_url_var.set(
            normalize_lm_studio_base_url(str(settings.get("base_url", self.base_url_var.get())))
        )
        self._set_host_port_from_base_url(self.base_url_var.get())
        self.lm_host_var.set(str(settings.get("lm_host", self.lm_host_var.get())))
        self.lm_port_var.set(str(settings.get("lm_port", self.lm_port_var.get())))
        self.lm_timeout_var.set(
            self._int_setting(settings.get("lm_timeout"), 30, 3600, self.lm_timeout_var.get())
        )
        self.concepts_var.set(str(settings.get("concepts", self.concepts_var.get())))
        self.concept_mix_var.set(
            canonicalize_saved_invent_value(
                "single",
                "concept_mix",
                settings.get("concept_mix", self.concept_mix_var.get()),
            )
        )
        self.visual_direction_var.set(
            str(settings.get("visual_direction", self.visual_direction_var.get()))
        )
        self.goal_headline_var.set(str(settings.get("goal_headline", self.goal_headline_var.get())))
        self.focus_var.set(str(settings.get("focus", self.focus_var.get())))
        self.weighted_terms_var.set(
            canonicalize_saved_invent_value(
                "single",
                "weighted_terms",
                settings.get("weighted_terms", self.weighted_terms_var.get()),
            )
        )
        self.story_elements_var.set(str(settings.get("story_elements", self.story_elements_var.get())))
        self.model_instructions_var.set(
            str(settings.get("model_instructions", self.model_instructions_var.get()))
        )
        self.mode_var.set(self._choice_setting(settings.get("mode"), PROMPT_MODES, self.mode_var.get()))
        self.detail_var.set(
            self._choice_setting(settings.get("detail"), DETAIL_LEVELS, self.detail_var.get())
        )
        self.output_length_var.set(
            self._choice_setting(
                settings.get("output_length"),
                OUTPUT_LENGTHS,
                self.output_length_var.get(),
            )
        )
        self.risk_level_var.set(
            self._choice_setting(settings.get("risk_level"), RISK_LEVELS, self.risk_level_var.get())
        )
        self.prompt_preset_var.set(
            self._choice_setting(settings.get("prompt_preset"), PROMPT_PRESETS, self.prompt_preset_var.get())
        )
        self.generation_feedback_var.set(
            str(settings.get("generation_feedback", self.generation_feedback_var.get()))
        )
        self.variation_var.set(self._int_setting(settings.get("variations"), 1, 3, self.variation_var.get()))
        self.temperature_var.set(
            self._float_setting(settings.get("temperature"), 0.0, 2.0, self.temperature_var.get())
        )
        self.fixed_seed_var.set(
            self._bool_setting(settings.get("fixed_seed"), self.fixed_seed_var.get())
        )
        self.seed_var.set(
            self._int_setting(settings.get("seed"), 0, 2_147_483_647, self.seed_var.get())
        )
        self.context_token_budget_var.set(
            self._context_token_setting(settings, self.context_token_budget_var.get())
        )
        self.chat_system_prompt_var.set(
            str(settings.get("chat_system_prompt", self.chat_system_prompt_var.get()))
        )
        self.chat_temperature_var.set(
            self._float_setting(
                settings.get("chat_temperature"),
                0.0,
                2.0,
                self.chat_temperature_var.get(),
            )
        )
        self.chat_max_tokens_var.set(
            self._int_setting(
                settings.get("chat_max_tokens"),
                1,
                CONTEXT_TOKEN_MAX,
                self.chat_max_tokens_var.get(),
            )
        )
        self.chat_messages = self._chat_history_setting(settings.get("chat_messages"))
        self.preserve_var.set(bool(settings.get("preserve_wording", self.preserve_var.get())))
        self.quote_text_var.set(bool(settings.get("quote_rendered_text", self.quote_text_var.get())))
        self.fix_logic_var.set(bool(settings.get("fix_logic", self.fix_logic_var.get())))
        self.enhance_actions_var.set(bool(settings.get("enhance_actions", self.enhance_actions_var.get())))
        self.develop_story_var.set(
            self._bool_setting(settings.get("develop_story"), self.develop_story_var.get())
        )
        self.artistic_detail_freedom_var.set(
            self._bool_setting(
                settings.get("artistic_detail_freedom"),
                self.artistic_detail_freedom_var.get(),
            )
        )
        self.clean_constraints_var.set(bool(settings.get("clean_constraints", self.clean_constraints_var.get())))
        self.safe_for_work_var.set(
            self._bool_setting(settings.get("safe_for_work"), self.safe_for_work_var.get())
        )
        self.explicit_nsfw_var.set(
            self._bool_setting(settings.get("explicit_nsfw"), self.explicit_nsfw_var.get())
        )
        self.altered_encoder_var.set(bool(settings.get("altered_text_encoder", self.altered_encoder_var.get())))
        self.thinking_mode_var.set(bool(settings.get("thinking_mode", self.thinking_mode_var.get())))
        self.live_research_var.set(bool(settings.get("live_research", self.live_research_var.get())))
        self.search_engine_var.set(
            self._choice_setting(
                settings.get("search_engine"),
                TEXT_RESEARCH_ENGINES,
                self.search_engine_var.get(),
            )
        )
        self.reference_images_var.set(
            bool(settings.get("reference_image_analysis", self.reference_images_var.get()))
        )
        self.comic_reference_images_var.set(
            bool(
                settings.get(
                    "comic_reference_image_analysis",
                    self.comic_reference_images_var.get(),
                )
            )
        )
        self.meme_reference_images_var.set(
            bool(
                settings.get(
                    "meme_reference_image_analysis",
                    self.meme_reference_images_var.get(),
                )
            )
        )
        self.reference_image_source_var.set(
            self._choice_setting(
                settings.get("reference_image_source"),
                REFERENCE_IMAGE_SOURCES,
                self.reference_image_source_var.get(),
            )
        )
        self.audit_repair_var.set(bool(settings.get("audit_repair", self.audit_repair_var.get())))
        self.include_settings_var.set(bool(settings.get("include_krea_settings", self.include_settings_var.get())))
        self.unload_after_generation_var.set(
            self._bool_setting(settings.get("unload_after_generation"), self.unload_after_generation_var.get())
        )
        self.creativity_var.set(
            self._choice_setting(settings.get("creativity"), CREATIVITY_LEVELS, self.creativity_var.get())
        )
        self.intensity_var.set(self._int_setting(settings.get("intensity"), -100, 100, self.intensity_var.get()))
        self.complexity_var.set(self._int_setting(settings.get("complexity"), -100, 100, self.complexity_var.get()))
        self.movement_var.set(self._int_setting(settings.get("movement"), -100, 100, self.movement_var.get()))
        self.remember_window_size_var.set(
            self._bool_setting(settings.get("remember_window_size"), self.remember_window_size_var.get())
        )

        # Settings created before workflow profiles used enrichment-heavy values.
        # Migrate them to the fidelity-first Exact profile once; future launches
        # preserve the explicitly stored profile and its advanced overrides.
        if not isinstance(stored_profile, str) or stored_profile not in WORKFLOW_PROFILES:
            self._apply_workflow_profile("Exact")

        geometry = settings.get("geometry")
        if self.remember_window_size_var.get() and isinstance(geometry, str) and "x" in geometry:
            match = re.match(r"^(\d+)x(\d+)", geometry)
            if match:
                self.root.resize(int(match.group(1)), int(match.group(2)))

    def _save_settings(self) -> None:
        try:
            with SETTINGS_PATH.open("w", encoding="utf-8") as settings_file:
                json.dump(self._settings_snapshot(), settings_file, indent=2, sort_keys=True)
        except OSError as exc:
            self._log_activity(f"Could not save settings: {exc}")

    def _apply_workflow_profile(self, profile: object) -> None:
        profile = str(profile)
        if profile == "Exact":
            values = {
                self.risk_level_var: "Strict cleanup",
                self.preserve_var: True,
                self.enhance_actions_var: False,
                self.develop_story_var: False,
                self.artistic_detail_freedom_var: False,
                self.thinking_mode_var: False,
                self.live_research_var: False,
                self.reference_images_var: False,
                self.audit_repair_var: True,
                self.creativity_var: "raw",
                self.temperature_var: 0.1,
                self.detail_var: "Balanced",
                self.output_length_var: "Balanced",
                self.context_token_budget_var: CONTEXT_TOKEN_AUTO_LABEL,
                self.include_settings_var: True,
            }
        elif profile == "Improve":
            values = {
                self.risk_level_var: "Balanced improvement",
                self.preserve_var: False,
                self.enhance_actions_var: False,
                self.develop_story_var: False,
                self.artistic_detail_freedom_var: False,
                self.thinking_mode_var: False,
                self.live_research_var: False,
                self.reference_images_var: False,
                self.audit_repair_var: True,
                self.creativity_var: "low",
                self.temperature_var: 0.15,
                self.detail_var: "Detailed",
                self.output_length_var: "Balanced",
                self.context_token_budget_var: CONTEXT_TOKEN_AUTO_LABEL,
                self.include_settings_var: True,
            }
        elif profile == "Explore":
            values = {
                self.risk_level_var: "Creative enhancement",
                self.preserve_var: False,
                self.enhance_actions_var: True,
                self.develop_story_var: True,
                self.artistic_detail_freedom_var: True,
                self.thinking_mode_var: False,
                self.live_research_var: False,
                self.reference_images_var: False,
                self.audit_repair_var: True,
                self.creativity_var: "medium",
                self.temperature_var: 0.25,
                self.detail_var: "Detailed",
                self.output_length_var: "Detailed",
                self.context_token_budget_var: CONTEXT_TOKEN_AUTO_LABEL,
                self.include_settings_var: True,
            }
        else:
            return
        for variable, value in values.items():
            variable.set(value)
        if self.krea_recommendation_label is not None:
            self._update_krea_recommendation()

    def _on_close(self) -> None:
        self.closing = True
        self.cancel_event.set()
        self.active_request_id += 1
        self._save_settings()
        self.root.close()

    def _invoke_focused_editor(self, method_name: str) -> None:
        widget = QApplication.focusWidget()
        method = getattr(widget, method_name, None)
        if callable(method):
            method()

    def clear_draft(self) -> None:
        if not hasattr(self, "draft_text") or not self.draft_text.toPlainText().strip():
            return
        if not messagebox.askyesno("Clear draft", "Clear the current draft prompt?"):
            return
        self.draft_text.clear()
        self.status_var.set("Draft cleared")
        self._save_settings()

    def _choice_setting(self, value: object, choices: tuple[str, ...], default: str) -> str:
        return value if isinstance(value, str) and value in choices else default

    @staticmethod
    def _valid_invent_recall_keys() -> set[str]:
        return {
            *(
                f"single:{field}"
                for field in (
                    "draft",
                    "concepts",
                    "concept_mix",
                    "visual_direction",
                    "goal_headline",
                    "focus",
                    "story_elements",
                    "weighted_terms",
                    "model_instructions",
                    "generation_feedback",
                )
            ),
            *(
                f"comic:{field}"
                for field in (
                    "title",
                    "premise",
                    "continuity",
                    "concepts",
                    "visual_direction",
                    "dialogue_direction",
                )
            ),
            *(f"comic:panel_{index}" for index in range(1, 13)),
            *(
                f"meme:{field}"
                for field in (
                    "response_context",
                    "response_goal",
                    "scene",
                    "focus",
                    "top",
                    "bottom",
                    "visual_direction",
                )
            ),
        }

    def _invent_recall_values_setting(self, value: object) -> dict[str, str]:
        """Load only recognized, bounded Recall values from settings."""

        if not isinstance(value, dict):
            return {}
        valid_keys = self._valid_invent_recall_keys()
        restored: dict[str, str] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key)
            if (
                key in valid_keys
                and isinstance(raw_value, str)
                and len(raw_value) <= 100_000
            ):
                restored[key] = raw_value
        return restored

    def _invent_recall_groups_setting(
        self,
        value: object,
    ) -> dict[str, dict[str, str]]:
        """Load recognized grouped Recall snapshots without cross-workspace keys."""

        if not isinstance(value, dict):
            return {}
        restored: dict[str, dict[str, str]] = {}
        panel_keys = {
            f"comic:panel_{index}"
            for index in range(1, 13)
        }
        raw_panels = value.get("comic:all_panels")
        if isinstance(raw_panels, dict):
            panels = {
                str(key): raw_value
                for key, raw_value in raw_panels.items()
                if (
                    str(key) in panel_keys
                    and isinstance(raw_value, str)
                    and len(raw_value) <= 100_000
                )
            }
            if panels:
                restored["comic:all_panels"] = panels
        return restored

    def _model_list_setting(self, value: object) -> list[str]:
        if not isinstance(value, list):
            return []

        models: list[str] = []
        for item in value:
            model = str(item).strip()
            if model and model not in models:
                models.append(model)
        return models

    def _custom_presets_setting(self, value: object) -> dict[str, dict[str, object]]:
        if not isinstance(value, dict):
            return {}
        presets: dict[str, dict[str, object]] = {}
        for raw_name, snapshot in value.items():
            name = re.sub(r"\s+", " ", str(raw_name)).strip()
            if not name or not isinstance(snapshot, dict):
                continue
            presets[name] = dict(snapshot)
            if len(presets) >= CUSTOM_PRESET_LIMIT:
                break
        return presets

    def _history_setting(self, value: object) -> list[dict[str, object]]:
        if not isinstance(value, list):
            return []

        history: list[dict[str, object]] = []
        seen: set[str] = set()
        for item in value:
            if isinstance(item, dict):
                corrected_prompt = strip_private_prompt_guidance(
                    str(item.get("corrected_prompt", item.get("prompt", "")))
                )
                requested_prompt = str(item.get("requested_prompt", "")).strip()
                goal_headline = str(item.get("goal_headline", "")).strip()
                created_at = str(item.get("created_at", "")).strip()
                focus = str(item.get("focus", "")).strip()
                weighted_terms = str(item.get("weighted_terms", "")).strip()
                story_elements = str(item.get("story_elements", "")).strip()
                concepts = str(item.get("concepts", "")).strip()
                concept_mix = str(item.get("concept_mix", "")).strip()
                model_instructions = str(item.get("model_instructions", "")).strip()
                generation_feedback = str(item.get("generation_feedback", "")).strip()
                risk_level = str(item.get("risk_level", "")).strip()
                prompt_preset = str(item.get("prompt_preset", "")).strip()
                title = str(item.get("title", "")).strip()
                pinned = self._bool_setting(item.get("pinned"), False)
                workspace = str(item.get("workspace", "")).strip().casefold()
                if workspace not in {"prompt", "comic", "meme"}:
                    workspace = {
                        "comic story": "comic",
                        "meme": "meme",
                    }.get(str(item.get("content_format", "")).strip().casefold(), "prompt")
                workspace_state = item.get("workspace_state", {})
                if not isinstance(workspace_state, dict):
                    workspace_state = {}
                option_values = {
                    key: item[key]
                    for key in PROMPT_HISTORY_OPTION_KEYS
                    if key in item
                }
            else:
                corrected_prompt = strip_private_prompt_guidance(str(item))
                requested_prompt = ""
                goal_headline = ""
                created_at = ""
                focus = ""
                weighted_terms = ""
                story_elements = ""
                concepts = ""
                concept_mix = ""
                model_instructions = ""
                generation_feedback = ""
                risk_level = ""
                prompt_preset = ""
                title = ""
                pinned = False
                workspace = "prompt"
                workspace_state = {}
                option_values = {}
            key = f"{workspace}\n{goal_headline}\n{requested_prompt}\n{corrected_prompt}"
            if not corrected_prompt or key in seen:
                continue
            entry: dict[str, object] = {
                "requested_prompt": requested_prompt,
                "goal_headline": goal_headline,
                "focus": focus,
                "weighted_terms": weighted_terms,
                "story_elements": story_elements,
                "concepts": concepts,
                "concept_mix": concept_mix,
                "model_instructions": model_instructions,
                "generation_feedback": generation_feedback,
                "risk_level": risk_level,
                "prompt_preset": prompt_preset,
                "title": title,
                "pinned": pinned,
                "workspace": workspace,
                "workspace_state": workspace_state,
                "corrected_prompt": corrected_prompt,
                "created_at": created_at,
            }
            entry.update(option_values)
            history.append(entry)
            seen.add(key)
            if len(history) >= PROMPT_HISTORY_LIMIT:
                break
        return history

    def _activity_setting(self, value: object) -> list[dict[str, str]]:
        if not isinstance(value, list):
            return []
        events: list[dict[str, str]] = []
        for item in value[-500:]:
            if not isinstance(item, dict):
                continue
            message = re.sub(r"\s+", " ", str(item.get("message", ""))).strip()
            workspace = str(item.get("workspace", "system")).strip().casefold()
            if not message or workspace not in WORKSPACE_LABELS:
                continue
            events.append(
                {
                    "time": str(item.get("time", "")).strip(),
                    "workspace": workspace,
                    "message": message,
                }
            )
        return events

    def _int_setting(self, value: object, minimum: int, maximum: int, default: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return max(minimum, min(maximum, parsed))

    def _context_token_setting(self, source: dict[str, object], default: object) -> str:
        if "context_token_budget" in source:
            value = source.get("context_token_budget")
            if str(value).strip().casefold() in {
                "auto",
                CONTEXT_TOKEN_AUTO_LABEL.casefold(),
            }:
                return CONTEXT_TOKEN_AUTO_LABEL
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                return (
                    str(default)
                    if str(default) in CONTEXT_TOKEN_CHOICES
                    else CONTEXT_TOKEN_AUTO_LABEL
                )
            if parsed <= CONTEXT_TOKEN_AUTO:
                return CONTEXT_TOKEN_AUTO_LABEL
            exact = next(
                (
                    label
                    for label, token_count in CONTEXT_TOKEN_CHOICE_VALUES.items()
                    if token_count == parsed
                ),
                None,
            )
            if exact is not None:
                return exact
            manual_choices = [
                (label, token_count)
                for label, token_count in CONTEXT_TOKEN_CHOICE_VALUES.items()
                if token_count > CONTEXT_TOKEN_AUTO
            ]
            return min(manual_choices, key=lambda item: abs(item[1] - parsed))[0]
        if "context_line_budget" in source:
            return CONTEXT_TOKEN_AUTO_LABEL
        return (
            str(default)
            if str(default) in CONTEXT_TOKEN_CHOICES
            else CONTEXT_TOKEN_AUTO_LABEL
        )

    def _chat_history_setting(self, value: object) -> list[dict[str, str]]:
        if not isinstance(value, list):
            return []
        messages: list[dict[str, str]] = []
        for item in value[-100:]:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", ""))
            content = str(item.get("content", "")).strip()
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})
        return messages

    def _float_setting(self, value: object, minimum: float, maximum: float, default: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        return max(minimum, min(maximum, parsed))

    def _weighted_tag_for_value(self, weight: float) -> str:
        if weight >= 2.0:
            return "weight_high"
        if weight >= 1.5:
            return "weight_medium"
        return "weight_low"

    def _schedule_weighted_highlights(self, *_args: object) -> None:
        if not hasattr(self, "draft_text"):
            return
        if self.weighted_highlight_after_id is not None:
            self.weighted_highlight_after_id.stop()
        timer = QTimer(self.root)
        timer.setSingleShot(True)
        timer.timeout.connect(self._highlight_weighted_terms)
        timer.start(80)
        self.weighted_highlight_after_id = timer

    def _on_draft_modified(self, _event: object | None = None) -> str:
        self._schedule_weighted_highlights()
        self._update_text_counters()
        self._update_diff_view(self.draft_text.toPlainText(), self.corrected_text.toPlainText())
        if self.draft_autosave_timer is not None:
            self.draft_autosave_timer.stop()
        timer = QTimer(self.root)
        timer.setSingleShot(True)
        timer.timeout.connect(self._save_settings)
        timer.start(700)
        self.draft_autosave_timer = timer
        return "break"

    @staticmethod
    def _text_counts(text: str) -> tuple[int, int]:
        words = len(re.findall(r"\b[\w'-]+\b", text, flags=re.UNICODE))
        approximate_tokens = (len(text.strip()) + 3) // 4 if text.strip() else 0
        return words, approximate_tokens

    def _update_text_counters(self) -> None:
        if not hasattr(self, "draft_counter_label"):
            return
        draft_words, draft_tokens = self._text_counts(self.draft_text.toPlainText())
        corrected_words, corrected_tokens = self._text_counts(self.corrected_text.toPlainText())
        self.draft_counter_label.setText(f"{draft_words} words  ·  ≈{draft_tokens} tokens")
        self.corrected_counter_label.setText(
            f"{corrected_words} words  ·  ≈{corrected_tokens} tokens"
        )
        self.corrected_counter_label.setStyleSheet("color: #8993a5;")

    def _update_diff_view(self, requested: str, corrected: str) -> None:
        if not hasattr(self, "diff_text"):
            return
        if not requested and not corrected:
            self.diff_text.setHtml("<span style='color:#8993a5'>Changes will appear here.</span>")
            return
        before = re.findall(r"\S+|\s+", requested)
        after = re.findall(r"\S+|\s+", corrected)
        matcher = difflib.SequenceMatcher(a=before, b=after, autojunk=False)
        chunks: list[str] = []
        for operation, a0, a1, b0, b1 in matcher.get_opcodes():
            old = html.escape("".join(before[a0:a1]))
            new = html.escape("".join(after[b0:b1]))
            if operation == "equal":
                chunks.append(old)
            elif operation == "delete":
                chunks.append(f"<span style='background:#522b35;color:#ffb3c0;text-decoration:line-through'>{old}</span>")
            elif operation == "insert":
                chunks.append(f"<span style='background:#244a3a;color:#a9f5c8'>{new}</span>")
            else:
                chunks.append(f"<span style='background:#522b35;color:#ffb3c0;text-decoration:line-through'>{old}</span>")
                chunks.append(f"<span style='background:#244a3a;color:#a9f5c8'>{new}</span>")
        self.diff_text.setHtml("".join(chunks))

    def _on_corrected_modified(self) -> None:
        self._update_text_counters()
        self._update_diff_view(self.draft_text.toPlainText(), self.corrected_text.toPlainText())
        if self.iterate_button is not None and not self.request_in_progress:
            self.iterate_button.setEnabled(bool(self.corrected_text.toPlainText().strip()))

    def _highlight_weighted_terms(self) -> None:
        self.weighted_highlight_after_id = None
        if not hasattr(self, "draft_text"):
            return

        selections: list[QTextEdit.ExtraSelection] = []
        document_text = self.draft_text.toPlainText()
        colors = {
            "weight_low": QColor("#554b24"),
            "weight_medium": QColor("#725d1e"),
            "weight_high": QColor("#8c4b22"),
        }
        terms = sorted(parse_weighted_terms(self.weighted_terms_var.get()), key=lambda item: len(item[0]), reverse=True)
        for term, weight in terms:
            if not term:
                continue
            tag = self._weighted_tag_for_value(weight)
            for match in re.finditer(re.escape(term), document_text, flags=re.IGNORECASE):
                selection = QTextEdit.ExtraSelection()
                selection.format.setBackground(colors[tag])
                cursor = self.draft_text.textCursor()
                cursor.setPosition(match.start())
                cursor.setPosition(match.end(), QTextCursor.MoveMode.KeepAnchor)
                selection.cursor = cursor
                selections.append(selection)
        self.draft_text.setExtraSelections(selections)

    def _adjust_weighted_term(self, delta: float) -> str:
        if self.weighted_terms_entry is None:
            return "break"
        cursor_index = self.weighted_terms_entry.cursorPosition()
        updated, new_cursor = adjust_weighted_terms_text(
            self.weighted_terms_var.get(),
            cursor_index,
            delta,
        )
        self.weighted_terms_var.set(updated)
        self.weighted_terms_entry.setCursorPosition(new_cursor)
        self.status_var.set(
            "Adjusted weighted word "
            + ("up" if delta > 0 else "down")
        )
        self._schedule_weighted_highlights()
        return "break"

    def _increase_weighted_term(self, _event: object | None = None) -> str:
        return self._adjust_weighted_term(0.1)

    def _decrease_weighted_term(self, _event: object | None = None) -> str:
        return self._adjust_weighted_term(-0.1)

    def _draft_weight_target(self) -> str:
        cursor = self.draft_text.textCursor()
        if cursor.hasSelection():
            return cursor.selectedText().strip()
        cursor.select(QTextCursor.SelectionType.WordUnderCursor)
        return cursor.selectedText().strip()

    def _adjust_draft_weighted_term(self, delta: float) -> str:
        target = re.sub(r"\s+", " ", self._draft_weight_target()).strip(" ,.;:()[]{}")
        if not target:
            self.status_var.set("Select a prompt word or phrase to weight")
            return "break"
        self.weighted_terms_var.set(
            adjust_named_weighted_term(
                self.weighted_terms_var.get(),
                target,
                delta,
            )
        )
        self._schedule_weighted_highlights()
        self.status_var.set(
            f"Weighted '{target}' " + ("up" if delta > 0 else "down")
        )
        return "break"

    def _increase_draft_weighted_term(self, _event: object | None = None) -> str:
        return self._adjust_draft_weighted_term(0.1)

    def _decrease_draft_weighted_term(self, _event: object | None = None) -> str:
        return self._adjust_draft_weighted_term(-0.1)

    def _bool_setting(self, value: object, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
        return default

    def _load_prompt_options_from_history(self, entry: dict[str, object]) -> None:
        self.generator_target_var.set(
            self._choice_setting(
                entry.get("generator_target"),
                GENERATOR_TARGETS,
                self.generator_target_var.get(),
            )
        )
        self.content_format_var.set(
            self._choice_setting(
                entry.get("content_format"),
                CONTENT_FORMATS,
                self.content_format_var.get(),
            )
        )
        self.camera_control_var.set(
            str(entry.get("camera_control", self.camera_control_var.get())).strip()
            or CAMERA_CONTROL_AUTO
        )
        profile = entry.get("workflow_profile")
        if isinstance(profile, str) and profile in WORKFLOW_PROFILES:
            self.workflow_profile_var.set(profile)
        model = str(entry.get("model", "")).strip()
        if model:
            self.model_var.set(model)
            if model not in self.available_models:
                self.available_models.insert(0, model)
                if self.model_combo is not None:
                    self.model_combo.configure(values=self.available_models)

        self.mode_var.set(self._choice_setting(entry.get("mode"), PROMPT_MODES, self.mode_var.get()))
        self.detail_var.set(self._choice_setting(entry.get("detail"), DETAIL_LEVELS, self.detail_var.get()))
        self.output_length_var.set(
            self._choice_setting(entry.get("output_length"), OUTPUT_LENGTHS, self.output_length_var.get())
        )
        self.risk_level_var.set(
            self._choice_setting(entry.get("risk_level"), RISK_LEVELS, self.risk_level_var.get())
        )
        self.prompt_preset_var.set(
            self._choice_setting(entry.get("prompt_preset"), PROMPT_PRESETS, self.prompt_preset_var.get())
        )
        self.visual_direction_var.set(
            str(entry.get("visual_direction", self.visual_direction_var.get()))
        )
        self.lm_timeout_var.set(
            self._int_setting(entry.get("lm_timeout"), 30, 3600, self.lm_timeout_var.get())
        )
        self.variation_var.set(self._int_setting(entry.get("variations"), 1, 3, self.variation_var.get()))
        self.temperature_var.set(
            self._float_setting(entry.get("temperature"), 0.0, 2.0, self.temperature_var.get())
        )
        self.fixed_seed_var.set(
            self._bool_setting(entry.get("fixed_seed"), self.fixed_seed_var.get())
        )
        self.seed_var.set(
            self._int_setting(entry.get("seed"), 0, 2_147_483_647, self.seed_var.get())
        )
        self.context_token_budget_var.set(
            self._context_token_setting(entry, self.context_token_budget_var.get())
        )
        self.weighted_terms_var.set(str(entry.get("weighted_terms", self.weighted_terms_var.get())))
        self.story_elements_var.set(str(entry.get("story_elements", self.story_elements_var.get())))
        self.preserve_var.set(self._bool_setting(entry.get("preserve_wording"), self.preserve_var.get()))
        self.quote_text_var.set(self._bool_setting(entry.get("quote_rendered_text"), self.quote_text_var.get()))
        self.fix_logic_var.set(self._bool_setting(entry.get("fix_logic"), self.fix_logic_var.get()))
        self.enhance_actions_var.set(self._bool_setting(entry.get("enhance_actions"), self.enhance_actions_var.get()))
        self.develop_story_var.set(
            self._bool_setting(entry.get("develop_story"), self.develop_story_var.get())
        )
        self.artistic_detail_freedom_var.set(
            self._bool_setting(
                entry.get("artistic_detail_freedom"),
                self.artistic_detail_freedom_var.get(),
            )
        )
        self.clean_constraints_var.set(self._bool_setting(entry.get("clean_constraints"), self.clean_constraints_var.get()))
        self.safe_for_work_var.set(
            self._bool_setting(entry.get("safe_for_work"), self.safe_for_work_var.get())
        )
        self.explicit_nsfw_var.set(
            self._bool_setting(entry.get("explicit_nsfw"), self.explicit_nsfw_var.get())
        )
        self.altered_encoder_var.set(self._bool_setting(entry.get("altered_text_encoder"), self.altered_encoder_var.get()))
        self.thinking_mode_var.set(self._bool_setting(entry.get("thinking_mode"), self.thinking_mode_var.get()))
        self.live_research_var.set(self._bool_setting(entry.get("live_research"), self.live_research_var.get()))
        self.search_engine_var.set(
            self._choice_setting(
                entry.get("search_engine"),
                TEXT_RESEARCH_ENGINES,
                self.search_engine_var.get(),
            )
        )
        self.reference_images_var.set(
            self._bool_setting(entry.get("reference_image_analysis"), self.reference_images_var.get())
        )
        self.reference_image_source_var.set(
            self._choice_setting(
                entry.get("reference_image_source"),
                REFERENCE_IMAGE_SOURCES,
                self.reference_image_source_var.get(),
            )
        )
        self.audit_repair_var.set(self._bool_setting(entry.get("audit_repair"), self.audit_repair_var.get()))
        self.include_settings_var.set(
            self._bool_setting(entry.get("include_krea_settings"), self.include_settings_var.get())
        )
        self.creativity_var.set(
            self._choice_setting(entry.get("creativity"), CREATIVITY_LEVELS, self.creativity_var.get())
        )
        self.intensity_var.set(self._int_setting(entry.get("intensity"), -100, 100, self.intensity_var.get()))
        self.complexity_var.set(self._int_setting(entry.get("complexity"), -100, 100, self.complexity_var.get()))
        self.movement_var.set(self._int_setting(entry.get("movement"), -100, 100, self.movement_var.get()))

    def _set_host_port_from_base_url(self, base_url: str) -> None:
        parsed = urllib.parse.urlsplit(normalize_lm_studio_base_url(base_url))
        if parsed.hostname:
            self.lm_host_var.set(parsed.hostname)
        if parsed.port:
            self.lm_port_var.set(str(parsed.port))

    def _current_base_url(self) -> str:
        host = self.lm_host_var.get().strip()
        port = self.lm_port_var.get().strip()
        if host:
            if ":" in host and not host.startswith("[") and not host.count(".") == 3:
                host = f"[{host}]"
            target = f"http://{host}:{port or '1234'}/v1"
        else:
            target = self.base_url_var.get()
        normalized = normalize_lm_studio_base_url(target)
        self.base_url_var.set(normalized)
        return normalized

    def _bind_line(self, variable: Value, *, read_only: bool = False) -> QLineEdit:
        widget = QLineEdit(str(variable.get()))
        widget.setReadOnly(read_only)
        if not read_only:
            widget.textChanged.connect(variable.set)
        variable.subscribe(lambda value, w=widget: self._set_line_value(w, value))
        return widget

    def _bind_text(self, variable: Value, *, maximum_height: int | None = None) -> QtTextEdit:
        widget = QtTextEdit()
        widget.setPlainText(str(variable.get()))
        if maximum_height is not None:
            widget.setMaximumHeight(maximum_height)
        widget.textChanged.connect(lambda w=widget, v=variable: v.set(w.toPlainText()))
        variable.subscribe(lambda value, w=widget: self._set_text_value(w, value))
        return widget

    @staticmethod
    def _set_help(widget: QWidget, description: str, example: str):
        widget.setToolTip(ui_tooltip(description, example))
        return widget

    def _help(self, widget: QWidget, key: str):
        description, example = UI_HELP[key]
        return self._set_help(widget, description, example)

    def _apply_known_tooltips(self, root: QWidget) -> None:
        """Apply shared help to titled controls, groups, labels, and tabs."""

        for label in root.findChildren(QLabel):
            if not label.toolTip() and label.text() in UI_HELP:
                self._help(label, label.text())
        for group in root.findChildren(QGroupBox):
            if not group.toolTip() and group.title() in UI_HELP:
                self._help(group, group.title())
        for button_type in (QPushButton, QCheckBox):
            for button in root.findChildren(button_type):
                if not button.toolTip() and button.text() in UI_HELP:
                    self._help(button, button.text())
        for tabs in root.findChildren(QTabWidget):
            if not tabs.toolTip():
                self._set_help(
                    tabs,
                    "Switch between the available views in this section.",
                    "Select a tab such as Result or History.",
                )
            for index in range(tabs.count()):
                title = tabs.tabText(index)
                if title in UI_HELP:
                    description, example = UI_HELP[title]
                    tabs.setTabToolTip(index, ui_tooltip(description, example))
        fallback_help = (
            (QLineEdit, "Enter or review the value used by this setting.", "Type a value, then run the correction."),
            (QComboBox, "Choose one of the available values for this setting.", "Open the list and select the desired option."),
            (QDoubleSpinBox, "Set the numeric value used by this setting.", "Use the arrows to make a small adjustment."),
            (QSpinBox, "Set the numeric value used by this setting.", "Use the arrows or type an exact number."),
            (QTextEdit, "Enter or review text used by this part of the workflow.", "Type text or inspect the generated result."),
            (QListWidget, "Select an item to use with the nearby actions.", "Click one row before choosing Load or Remove."),
            (QSlider, "Adjust this value across its available range.", "Drag right to increase the setting."),
            (QProgressBar, "Shows progress for the active request.", "Watch it advance during model and audit passes."),
        )
        for widget_type, description, example in fallback_help:
            for widget in root.findChildren(widget_type):
                if not widget.toolTip():
                    self._set_help(widget, description, example)
        for button in root.findChildren(QPushButton):
            if not button.toolTip():
                label = button.text().strip() or "this action"
                self._set_help(
                    button,
                    f"Run the {label} action for the current workspace.",
                    f"Click {label} after completing the nearby fields.",
                )
        for group in root.findChildren(QGroupBox):
            if not group.toolTip():
                title = group.title().strip() or "settings"
                self._set_help(
                    group,
                    f"Controls and results for {title}.",
                    f"Complete the fields in {title}, then use its action button.",
                )
        for label in root.findChildren(QLabel):
            if not label.toolTip():
                text = label.text().strip() or "this field"
                self._set_help(
                    label,
                    f"Explains or labels {text}.",
                    "Use the related field or action in this section.",
                )
        for tabs in root.findChildren(QTabWidget):
            for index in range(tabs.count()):
                if not tabs.tabToolTip(index):
                    title = tabs.tabText(index)
                    tabs.setTabToolTip(
                        index,
                        ui_tooltip(
                            f"Open the {title} workspace.",
                            f"Select {title} to use its tools.",
                        ),
                    )
        for action in self.root.findChildren(QAction):
            if action.isSeparator() or "Example:" in action.toolTip():
                continue
            title = action.text().replace("&", "")
            if title in UI_HELP:
                description, example = UI_HELP[title]
            else:
                description = f"Run the {title} command."
                example = f"Choose {title} from this menu."
            help_text = ui_tooltip(description, example)
            action.setToolTip(help_text)
            action.setStatusTip(help_text)
        for menu in self.root.findChildren(QMenu):
            menu.setToolTipsVisible(True)

    def _bind_weighted_terms_line(self) -> QtWeightedLineEdit:
        widget = QtWeightedLineEdit(
            str(self.weighted_terms_var.get()),
            self._increase_weighted_term,
            self._decrease_weighted_term,
        )
        widget.textChanged.connect(self.weighted_terms_var.set)
        self.weighted_terms_var.subscribe(lambda value, w=widget: self._set_line_value(w, value))
        return widget

    @staticmethod
    def _set_line_value(widget: QLineEdit, value: object) -> None:
        text = str(value)
        if widget.text() != text:
            widget.blockSignals(True)
            widget.setText(text)
            widget.blockSignals(False)

    @staticmethod
    def _set_text_value(widget: QTextEdit, value: object) -> None:
        text = str(value)
        if widget.toPlainText() != text:
            widget.blockSignals(True)
            widget.setPlainText(text)
            widget.blockSignals(False)

    def _bind_combo(self, variable: Value, choices, *, editable: bool = False) -> QtComboBox:
        widget = QtComboBox()
        widget.setEditable(editable)
        widget.setMaxVisibleItems(20)
        widget.setMinimumContentsLength(18)
        widget.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        widget.addItems(list(choices))
        widget.setCurrentText(str(variable.get()))
        widget.currentTextChanged.connect(variable.set)
        variable.subscribe(lambda value, w=widget: self._set_combo_value(w, value))
        return widget

    @staticmethod
    def _set_combo_value(widget: QComboBox, value: object) -> None:
        text = str(value)
        if widget.currentText() != text:
            widget.blockSignals(True)
            widget.setCurrentText(text)
            widget.blockSignals(False)

    def _bind_spin(self, variable: Value, minimum: int, maximum: int, step: int = 1) -> QSpinBox:
        widget = QSpinBox()
        widget.setRange(minimum, maximum)
        widget.setSingleStep(step)
        widget.setValue(int(variable.get()))
        widget.valueChanged.connect(variable.set)
        variable.subscribe(lambda value, w=widget: w.setValue(int(value)))
        return widget

    def _bind_double_spin(self, variable: Value) -> QDoubleSpinBox:
        widget = QDoubleSpinBox()
        widget.setRange(0.0, 2.0)
        widget.setSingleStep(0.05)
        widget.setDecimals(2)
        widget.setValue(float(variable.get()))
        widget.valueChanged.connect(variable.set)
        variable.subscribe(lambda value, w=widget: w.setValue(float(value)))
        return widget

    def _bind_check(self, label: str, variable: Value) -> QCheckBox:
        widget = QCheckBox(label)
        if label in UI_HELP:
            self._help(widget, label)
        widget.setChecked(bool(variable.get()))
        widget.toggled.connect(variable.set)
        variable.subscribe(lambda value, w=widget: w.setChecked(bool(value)))
        return widget

    def _open_mix_ingredient_picker(
        self,
        current_names: list[str],
    ) -> tuple[list[str], bool] | None:
        selected = {
            key
            for key in mix_ingredient_keys_for_names(current_names)
            if key in MIX_INGREDIENT_KEYS
        }

        dialog = QDialog(self.root)
        dialog.setWindowTitle("Mixer ingredient library")
        dialog.resize(860, 720)
        outer = QVBoxLayout(dialog)
        intro = QLabel(
            "Choose up to six ingredients from the combined concept and visual-style catalogs. "
            "The library includes content concepts, rendering media, genres, mood, lighting, "
            "palette, atmosphere, composition, texture, finish, and presentation."
        )
        intro.setWordWrap(True)
        outer.addWidget(intro)

        filters = QHBoxLayout()
        filters.addWidget(QLabel("Category"))
        category_combo = QComboBox()
        category_combo.addItems(("All categories", *MIX_INGREDIENT_PRESETS.keys()))
        self._set_help(
            category_combo,
            "Filters the mixer library to content concepts, media, or one visual direction.",
            "Choose Styles · Rendering modes and media to browse watercolor, 3D, and print styles.",
        )
        filters.addWidget(category_combo)
        filters.addWidget(QLabel("Search"))
        search_entry = QLineEdit()
        search_entry.setPlaceholderText(
            "Search watercolor, courier, moonlight, Art Nouveau, texture…"
        )
        self._set_help(
            search_entry,
            f"Searches category names and all {len(MIX_INGREDIENT_KEYS)} mixer ingredients.",
            "Type watercolor to find the watercolor rendering medium.",
        )
        filters.addWidget(search_entry, 1)
        outer.addLayout(filters)

        preset_list = QListWidget()
        preset_list.setMinimumHeight(390)
        self._set_help(
            preset_list,
            f"Selects up to {MIX_INGREDIENT_LIMIT} ingredients for percentage mixing.",
            "Combine Watercolor, cyberpunk, and bioluminescent fungi.",
        )
        outer.addWidget(preset_list, 1)

        selection_label = QLabel()
        outer.addWidget(selection_label)
        limit_notice = QLabel()
        limit_notice.setStyleSheet("color: #d59b45;")
        limit_notice.setWordWrap(True)
        outer.addWidget(limit_notice)
        preview = QTextEdit()
        preview.setReadOnly(True)
        preview.setMaximumHeight(100)
        preview.setPlaceholderText("Selected mixer ingredients will appear here.")
        self._set_help(
            preview,
            "Previews the ingredient names that will be inserted into the percentage editor.",
            "Watercolor, cyberpunk, bioluminescent fungi",
        )
        outer.addWidget(preview)

        apply_row = QHBoxLayout()
        clear_button = QtButton("Clear selection")
        self._set_help(
            clear_button,
            "Unchecks every selected library ingredient.",
            "Clear the current library selection before choosing a new blend.",
        )
        apply_row.addWidget(clear_button)
        apply_row.addStretch()
        apply_row.addWidget(QLabel("Apply mode"))
        apply_mode = QComboBox()
        apply_mode.addItems(
            ("Add to current mixer rows", "Replace all mixer rows")
        )
        self._set_help(
            apply_mode,
            "Adds library ingredients to free rows or replaces the current six-row mix.",
            "Use Replace all mixer rows when starting a completely different blend.",
        )
        apply_row.addWidget(apply_mode)
        outer.addLayout(apply_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText(
            "Use ingredients"
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        outer.addWidget(buttons)

        rebuilding = False

        def sync_visible_selection() -> None:
            if rebuilding:
                return
            for index in range(preset_list.count()):
                item = preset_list.item(index)
                key = str(item.data(Qt.ItemDataRole.UserRole))
                if item.checkState() == Qt.CheckState.Checked:
                    selected.add(key)
                else:
                    selected.discard(key)

        def refresh_preview() -> None:
            names = format_mix_ingredient_names(selected)
            selection_label.setText(
                f"{len(names)} of {MIX_INGREDIENT_LIMIT} ingredients selected"
            )
            preview.setPlainText(", ".join(names))

        def rebuild_list() -> None:
            nonlocal rebuilding
            sync_visible_selection()
            category_filter = category_combo.currentText()
            query = search_entry.text().strip().casefold()
            rebuilding = True
            preset_list.blockSignals(True)
            preset_list.clear()
            for category, values in MIX_INGREDIENT_PRESETS.items():
                if category_filter != "All categories" and category != category_filter:
                    continue
                for value in values:
                    searchable = f"{category} {value}".casefold()
                    if query and query not in searchable:
                        continue
                    key = mix_ingredient_key(category, value)
                    item = QListWidgetItem(f"{category}  ·  {value}")
                    item.setData(Qt.ItemDataRole.UserRole, key)
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    item.setCheckState(
                        Qt.CheckState.Checked
                        if key in selected
                        else Qt.CheckState.Unchecked
                    )
                    preset_list.addItem(item)
            preset_list.blockSignals(False)
            rebuilding = False
            refresh_preview()

        def on_item_changed(item: QListWidgetItem) -> None:
            key = str(item.data(Qt.ItemDataRole.UserRole))
            names = format_mix_ingredient_names(selected)
            selected_name = item.text().split("  ·  ", 1)[-1].casefold()
            already_represented = any(
                name.casefold() == selected_name for name in names
            )
            if (
                item.checkState() == Qt.CheckState.Checked
                and key not in selected
                and not already_represented
                and len(names) >= MIX_INGREDIENT_LIMIT
            ):
                preset_list.blockSignals(True)
                item.setCheckState(Qt.CheckState.Unchecked)
                preset_list.blockSignals(False)
                limit_notice.setText(
                    f"The mixer accepts {MIX_INGREDIENT_LIMIT} ingredients. "
                    "Uncheck one before adding another."
                )
                return
            limit_notice.clear()
            sync_visible_selection()
            refresh_preview()

        def clear_selection() -> None:
            nonlocal rebuilding
            selected.clear()
            limit_notice.clear()
            rebuilding = True
            preset_list.blockSignals(True)
            for index in range(preset_list.count()):
                preset_list.item(index).setCheckState(Qt.CheckState.Unchecked)
            preset_list.blockSignals(False)
            rebuilding = False
            rebuild_list()

        category_combo.currentTextChanged.connect(lambda _value: rebuild_list())
        search_entry.textChanged.connect(lambda _value: rebuild_list())
        preset_list.itemChanged.connect(on_item_changed)
        clear_button.clicked.connect(clear_selection)
        rebuild_list()

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        sync_visible_selection()
        names = format_mix_ingredient_names(selected)[:MIX_INGREDIENT_LIMIT]
        return names, apply_mode.currentText() == "Replace all mixer rows"

    def _open_concept_mix_editor(self, _checked: bool = False) -> None:
        dialog = QDialog(self.root)
        dialog.setWindowTitle("Concept and style mix")
        dialog.setMinimumWidth(680)
        outer = QVBoxLayout(dialog)
        intro = QLabel(
            "Name up to six concepts, styles, materials, moods, or visual influences. "
            "The active rows must total 100%. Browse the exhaustive ingredient library or "
            "continue entering custom ingredients."
        )
        intro.setWordWrap(True)
        outer.addWidget(intro)

        grid = QGridLayout()
        name_header = QLabel("Ingredient")
        share_header = QLabel("Share")
        grid.addWidget(name_header, 0, 0)
        grid.addWidget(share_header, 0, 1)
        parsed = parse_concept_mix(self.concept_mix_var.get())
        row_edits: list[QLineEdit] = []
        row_spins: list[QSpinBox] = []
        for row in range(6):
            name = parsed[row][0] if row < len(parsed) else ""
            share = parsed[row][1] if row < len(parsed) else (50 if not parsed and row < 2 else 0)
            edit = QLineEdit(name)
            edit.setObjectName(f"mixIngredient{row + 1}")
            edit.setPlaceholderText("e.g. Art Nouveau")
            spin = QSpinBox()
            spin.setObjectName(f"mixShare{row + 1}")
            spin.setRange(0, 100)
            spin.setSuffix("%")
            spin.setValue(share)
            grid.addWidget(edit, row + 1, 0)
            grid.addWidget(spin, row + 1, 1)
            row_edits.append(edit)
            row_spins.append(spin)
        grid.setColumnStretch(0, 1)
        outer.addLayout(grid)

        tools = QHBoxLayout()
        library_button = QPushButton("Browse library…")
        library_button.setObjectName("mixLibraryButton")
        self._set_help(
            library_button,
            "Open the searchable library of concepts, media, genres, mood, lighting, "
            "palette, composition, texture, and finish ingredients.",
            "Choose Watercolor, cyberpunk, and bioluminescent fungi before setting shares.",
        )
        equal_button = QPushButton("Balance equally")
        normalize_button = QPushButton("Normalize to 100%")
        total_label = QLabel()
        tools.addWidget(library_button)
        tools.addWidget(equal_button)
        tools.addWidget(normalize_button)
        tools.addStretch(1)
        tools.addWidget(total_label)
        outer.addLayout(tools)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        save_button = buttons.button(QDialogButtonBox.StandardButton.Save)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        outer.addWidget(buttons)

        def active_rows() -> list[int]:
            return [index for index, edit in enumerate(row_edits) if edit.text().strip()]

        def refresh_total() -> None:
            active = active_rows()
            total = sum(row_spins[index].value() for index in active)
            total_label.setText(f"Total: {total}%")
            total_label.setStyleSheet("color: #a9f5c8;" if total == 100 else "color: #ffb3c0;")
            save_button.setEnabled(bool(active) and total == 100)

        def set_equal() -> None:
            active = active_rows()
            if not active:
                return
            base, remainder = divmod(100, len(active))
            for position, index in enumerate(active):
                row_spins[index].setValue(base + (1 if position < remainder else 0))
            refresh_total()

        def normalize() -> None:
            active = active_rows()
            if not active:
                return
            current = [row_spins[index].value() for index in active]
            if sum(current) <= 0:
                set_equal()
                return
            exact = [value * 100.0 / sum(current) for value in current]
            values = [int(value) for value in exact]
            remainder = 100 - sum(values)
            order = sorted(
                range(len(exact)),
                key=lambda index: (exact[index] - values[index], -index),
                reverse=True,
            )
            for position in order[:remainder]:
                values[position] += 1
            for index, value in zip(active, values):
                row_spins[index].setValue(value)
            refresh_total()

        def browse_library() -> None:
            current_names = [
                edit.text().strip() for edit in row_edits if edit.text().strip()
            ]
            result = self._open_mix_ingredient_picker(current_names)
            if result is None:
                return
            library_names, replace = result
            if replace:
                combined = library_names
            else:
                combined = list(current_names)
                seen = {name.casefold() for name in combined}
                for name in library_names:
                    if name.casefold() not in seen:
                        combined.append(name)
                        seen.add(name.casefold())
            combined = combined[:MIX_INGREDIENT_LIMIT]
            changed = [name.casefold() for name in combined] != [
                name.casefold() for name in current_names
            ]
            for index, edit in enumerate(row_edits):
                edit.setText(combined[index] if index < len(combined) else "")
            if changed and combined:
                set_equal()
            else:
                refresh_total()

        for edit, spin in zip(row_edits, row_spins):
            edit.textChanged.connect(refresh_total)
            spin.valueChanged.connect(refresh_total)
        library_button.clicked.connect(browse_library)
        equal_button.clicked.connect(set_equal)
        normalize_button.clicked.connect(normalize)
        refresh_total()

        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.concept_mix_var.set(
                ", ".join(
                    f"{row_edits[index].text().strip()}:{row_spins[index].value()}%"
                    for index in active_rows()
                )
            )
            self._save_settings()
            self.status_var.set("Updated concept and style mix")

    def _narrative_preset_target(self, destination: str) -> Value:
        return {
            "comic": self.comic_premise_var,
            "meme": self.meme_scene_var,
        }.get(destination, self.story_elements_var)

    def _make_narrative_preset_button(
        self,
        destination: str,
        kind: str,
    ) -> QtButton:
        normalized_kind = "emotion" if kind == "emotion" else "action"
        label = "Emotions…" if normalized_kind == "emotion" else "Actions…"
        button = QtButton(label)
        button.clicked.connect(
            lambda _checked=False, target=destination, preset_kind=normalized_kind:
            self._open_narrative_preset_picker(target, preset_kind)
        )
        if normalized_kind == "emotion":
            self._set_help(
                button,
                "Open the searchable library of visible emotional states, reactions, "
                "transitions, facial cues, and body language.",
                "Add relief mixed with lingering fear to make the reaction visually readable.",
            )
        else:
            self._set_help(
                button,
                "Open the searchable library of concrete actions, interactions, work, "
                "movement, conflict, rescue, performance, and everyday story beats.",
                "Add reaching carefully for a fragile object as the decisive visible action.",
            )
        self.narrative_preset_buttons[f"{destination}:{normalized_kind}"] = button
        return button

    def _open_narrative_preset_picker(self, destination: str, kind: str) -> None:
        destination = destination if destination in {"prompt", "comic", "meme"} else "prompt"
        kind = "emotion" if kind == "emotion" else "action"
        catalog = EMOTION_PRESETS if kind == "emotion" else ACTION_PRESETS
        valid_keys = EMOTION_PRESET_KEYS if kind == "emotion" else ACTION_PRESET_KEYS
        target = self._narrative_preset_target(destination)
        selected = {
            key
            for key in self.narrative_preset_selections[destination][kind]
            if key in valid_keys
        }
        library_name = "Emotion" if kind == "emotion" else "Action"

        dialog = QDialog(self.root)
        dialog.setWindowTitle(f"{library_name} preset library")
        dialog.resize(850, 720)
        outer = QVBoxLayout(dialog)
        if kind == "emotion":
            intro_text = (
                "Choose visible emotional states and reactions with concrete facial expression "
                "and body-language cues. Mixed and changing emotions are included."
            )
        else:
            intro_text = (
                "Choose concrete visible actions and story beats across movement, interaction, "
                "work, discovery, conflict, rescue, performance, daily life, and ceremony."
            )
        intro = QLabel(
            f"{intro_text} Select up to {NARRATIVE_PRESET_LIMIT} presets."
        )
        intro.setWordWrap(True)
        outer.addWidget(intro)

        filters = QHBoxLayout()
        filters.addWidget(QLabel("Category"))
        category_combo = QComboBox()
        category_combo.addItems(("All categories", *catalog.keys()))
        self._set_help(
            category_combo,
            f"Filters the {library_name.lower()} library to one narrative dimension.",
            "Choose Investigation and discovery to browse clue-finding actions."
            if kind == "action"
            else "Choose Fear anxiety and vulnerability to browse readable fear reactions.",
        )
        filters.addWidget(category_combo)
        filters.addWidget(QLabel("Search"))
        search_entry = QLineEdit()
        search_entry.setPlaceholderText(
            "Search rescue, opening, running, repair, discovery…"
            if kind == "action"
            else "Search joy, fear, relief, anger, grief, wonder…"
        )
        self._set_help(
            search_entry,
            f"Searches category names and all {len(valid_keys)} {library_name.lower()} presets.",
            "Type rescue to find emergency actions."
            if kind == "action"
            else "Type relief to find pure and mixed relief reactions.",
        )
        filters.addWidget(search_entry, 1)
        outer.addLayout(filters)

        preset_list = QListWidget()
        preset_list.setMinimumHeight(390)
        self._set_help(
            preset_list,
            f"Selects up to {NARRATIVE_PRESET_LIMIT} concrete {library_name.lower()} presets.",
            "Combine opening a sealed container with reacting to a sudden revelation."
            if kind == "action"
            else "Combine quiet anxiety with courage emerging through visible fear.",
        )
        outer.addWidget(preset_list, 1)

        selection_label = QLabel()
        outer.addWidget(selection_label)
        limit_notice = QLabel()
        limit_notice.setStyleSheet("color: #d59b45;")
        limit_notice.setWordWrap(True)
        outer.addWidget(limit_notice)
        preview = QTextEdit()
        preview.setReadOnly(True)
        preview.setMaximumHeight(110)
        preview.setPlaceholderText(
            f"Selected {library_name.lower()} direction will appear here."
        )
        self._set_help(
            preview,
            f"Previews the exact {library_name.lower()} direction applied to the narrative field.",
            "opening a sealed container; reacting to a sudden revelation"
            if kind == "action"
            else "quiet anxiety with tightly clasped hands; courage emerging through visible anxiety",
        )
        outer.addWidget(preview)

        apply_row = QHBoxLayout()
        clear_button = QtButton("Clear selection")
        self._set_help(
            clear_button,
            f"Unchecks every selected {library_name.lower()} preset.",
            "Clear the selection before creating a different narrative combination.",
        )
        apply_row.addWidget(clear_button)
        apply_row.addStretch()
        apply_row.addWidget(QLabel("Apply mode"))
        apply_mode = QComboBox()
        apply_mode.addItems(
            ("Append to current narrative", "Replace current narrative")
        )
        self._set_help(
            apply_mode,
            "Appends the selection without duplicates or replaces the entire narrative field.",
            "Append emotion cues after selecting an action beat.",
        )
        apply_row.addWidget(apply_mode)
        outer.addLayout(apply_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Apply")
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        outer.addWidget(buttons)

        rebuilding = False

        def sync_visible_selection() -> None:
            if rebuilding:
                return
            for index in range(preset_list.count()):
                item = preset_list.item(index)
                key = str(item.data(Qt.ItemDataRole.UserRole))
                if item.checkState() == Qt.CheckState.Checked:
                    selected.add(key)
                else:
                    selected.discard(key)

        def refresh_preview() -> None:
            selection_label.setText(
                f"{len(selected)} of {NARRATIVE_PRESET_LIMIT} presets selected"
            )
            preview.setPlainText(format_narrative_presets(kind, selected))

        def rebuild_list() -> None:
            nonlocal rebuilding
            sync_visible_selection()
            category_filter = category_combo.currentText()
            query = search_entry.text().strip().casefold()
            rebuilding = True
            preset_list.blockSignals(True)
            preset_list.clear()
            for category, values in catalog.items():
                if category_filter != "All categories" and category != category_filter:
                    continue
                for value in values:
                    searchable = f"{category} {value}".casefold()
                    if query and query not in searchable:
                        continue
                    key = narrative_preset_key(kind, category, value)
                    item = QListWidgetItem(f"{category}  ·  {value}")
                    item.setData(Qt.ItemDataRole.UserRole, key)
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    item.setCheckState(
                        Qt.CheckState.Checked
                        if key in selected
                        else Qt.CheckState.Unchecked
                    )
                    preset_list.addItem(item)
            preset_list.blockSignals(False)
            rebuilding = False
            refresh_preview()

        def on_item_changed(item: QListWidgetItem) -> None:
            key = str(item.data(Qt.ItemDataRole.UserRole))
            if (
                item.checkState() == Qt.CheckState.Checked
                and key not in selected
                and len(selected) >= NARRATIVE_PRESET_LIMIT
            ):
                preset_list.blockSignals(True)
                item.setCheckState(Qt.CheckState.Unchecked)
                preset_list.blockSignals(False)
                limit_notice.setText(
                    f"This library applies up to {NARRATIVE_PRESET_LIMIT} presets at once. "
                    "Uncheck one before adding another."
                )
                return
            limit_notice.clear()
            sync_visible_selection()
            refresh_preview()

        def clear_selection() -> None:
            nonlocal rebuilding
            selected.clear()
            limit_notice.clear()
            rebuilding = True
            preset_list.blockSignals(True)
            for index in range(preset_list.count()):
                preset_list.item(index).setCheckState(Qt.CheckState.Unchecked)
            preset_list.blockSignals(False)
            rebuilding = False
            rebuild_list()

        category_combo.currentTextChanged.connect(lambda _value: rebuild_list())
        search_entry.textChanged.connect(lambda _value: rebuild_list())
        preset_list.itemChanged.connect(on_item_changed)
        clear_button.clicked.connect(clear_selection)
        rebuild_list()

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        sync_visible_selection()
        ordered_selection = [
            narrative_preset_key(kind, category, value)
            for category, values in catalog.items()
            for value in values
            if narrative_preset_key(kind, category, value) in selected
        ][:NARRATIVE_PRESET_LIMIT]
        narrative_text = format_narrative_presets(kind, ordered_selection)
        if apply_mode.currentText() == "Append to current narrative":
            narrative_text = merge_narrative_text(str(target.get()), narrative_text)
        target.set(narrative_text)
        self.narrative_preset_selections[destination][kind] = ordered_selection
        self._save_settings()
        workspace_label = {
            "prompt": "Prompt Corrector story beat",
            "comic": "Comic Story premise",
            "meme": "Meme Creator scene",
        }[destination]
        self.status_var.set(
            f"Applied {len(ordered_selection)} {library_name.lower()} presets to {workspace_label}"
        )

    def _concept_target(self, destination: str) -> Value:
        return (
            self.comic_concepts_var
            if destination == "comic"
            else self.concepts_var
        )

    def _make_concept_preset_button(self, destination: str) -> QtButton:
        button = QtButton("Concepts…")
        button.clicked.connect(
            lambda _checked=False, target=destination: self._open_concept_preset_picker(
                target
            )
        )
        self._set_help(
            button,
            "Open the exhaustive searchable content-concept library for characters, roles, "
            "relationships, actions, creatures, places, objects, narratives, eras, and design.",
            "Combine a courier, impossible delivery, floating village, and antique compass.",
        )
        self.concept_preset_buttons[destination] = button
        return button

    def _open_concept_preset_picker(self, destination: str) -> None:
        destination = destination if destination in {"prompt", "comic"} else "prompt"
        target = self._concept_target(destination)
        selected = {
            key
            for key in self.concept_preset_selections.get(destination, [])
            if key in CONCEPT_PRESET_KEYS
        }

        dialog = QDialog(self.root)
        dialog.setWindowTitle("Content concept library")
        dialog.resize(820, 720)
        outer = QVBoxLayout(dialog)
        intro = QLabel(
            "Build the content of the image from a broad catalog of subjects, roles, "
            "relationships, actions, creatures, environments, objects, narrative motifs, "
            f"eras, crafts, and design ideas. Select up to {CONCEPT_SELECTION_LIMIT} concepts "
            "so every selected concept remains active in the generation contract."
        )
        intro.setWordWrap(True)
        outer.addWidget(intro)

        filters = QHBoxLayout()
        filters.addWidget(QLabel("Category"))
        category_combo = QComboBox()
        category_combo.addItems(("All categories", *CONCEPT_PRESETS.keys()))
        self._set_help(
            category_combo,
            "Filters the content-concept library to one subject or story dimension.",
            "Choose Environments and biomes to browse locations.",
        )
        filters.addWidget(category_combo)
        filters.addWidget(QLabel("Search"))
        search_entry = QLineEdit()
        search_entry.setPlaceholderText(
            "Search courier, dragon, observatory, reunion, symbolism…"
        )
        self._set_help(
            search_entry,
            f"Searches category names and all {len(CONCEPT_PRESET_KEYS)} concept entries.",
            "Type observatory to find architectural and interior concepts.",
        )
        filters.addWidget(search_entry, 1)
        outer.addLayout(filters)

        preset_list = QListWidget()
        preset_list.setMinimumHeight(390)
        self._set_help(
            preset_list,
            f"Selects up to {CONCEPT_SELECTION_LIMIT} content concepts for the active workspace.",
            "Check courier, impossible delivery, floating village, and antique compass.",
        )
        outer.addWidget(preset_list, 1)

        selection_label = QLabel()
        outer.addWidget(selection_label)
        limit_notice = QLabel()
        limit_notice.setStyleSheet("color: #d59b45;")
        limit_notice.setWordWrap(True)
        outer.addWidget(limit_notice)
        preview = QTextEdit()
        preview.setReadOnly(True)
        preview.setMaximumHeight(110)
        preview.setPlaceholderText("Selected comma-separated concepts will appear here.")
        self._set_help(
            preview,
            "Previews the exact comma-separated Concepts field text that will be applied.",
            "courier, impossible delivery, floating village, antique compass",
        )
        outer.addWidget(preview)

        apply_row = QHBoxLayout()
        clear_button = QtButton("Clear selection")
        self._set_help(
            clear_button,
            "Unchecks every selected content concept.",
            "Clear the current set before building a different concept combination.",
        )
        apply_row.addWidget(clear_button)
        apply_row.addStretch()
        apply_row.addWidget(QLabel("Apply mode"))
        apply_mode = QComboBox()
        apply_mode.addItems(
            ("Replace current concepts", "Append to current concepts")
        )
        self._set_help(
            apply_mode,
            "Chooses whether the selection replaces or extends manually entered concepts.",
            "Append catalog ideas while preserving a custom character concept.",
        )
        apply_row.addWidget(apply_mode)
        outer.addLayout(apply_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Apply")
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        outer.addWidget(buttons)

        rebuilding = False

        def sync_visible_selection() -> None:
            if rebuilding:
                return
            for index in range(preset_list.count()):
                item = preset_list.item(index)
                key = str(item.data(Qt.ItemDataRole.UserRole))
                if item.checkState() == Qt.CheckState.Checked:
                    selected.add(key)
                else:
                    selected.discard(key)

        def refresh_preview() -> None:
            selection_label.setText(
                f"{len(selected)} of {CONCEPT_SELECTION_LIMIT} concepts selected"
            )
            preview.setPlainText(format_concept_presets(tuple(selected)))

        def rebuild_list() -> None:
            nonlocal rebuilding
            sync_visible_selection()
            category_filter = category_combo.currentText()
            query = search_entry.text().strip().casefold()
            rebuilding = True
            preset_list.blockSignals(True)
            preset_list.clear()
            for category, values in CONCEPT_PRESETS.items():
                if category_filter != "All categories" and category != category_filter:
                    continue
                for value in values:
                    searchable = f"{category} {value}".casefold()
                    if query and query not in searchable:
                        continue
                    key = concept_preset_key(category, value)
                    item = QListWidgetItem(f"{category}  ·  {value}")
                    item.setData(Qt.ItemDataRole.UserRole, key)
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    item.setCheckState(
                        Qt.CheckState.Checked
                        if key in selected
                        else Qt.CheckState.Unchecked
                    )
                    preset_list.addItem(item)
            preset_list.blockSignals(False)
            rebuilding = False
            refresh_preview()

        def on_item_changed(item: QListWidgetItem) -> None:
            key = str(item.data(Qt.ItemDataRole.UserRole))
            if (
                item.checkState() == Qt.CheckState.Checked
                and key not in selected
                and len(selected) >= CONCEPT_SELECTION_LIMIT
            ):
                preset_list.blockSignals(True)
                item.setCheckState(Qt.CheckState.Unchecked)
                preset_list.blockSignals(False)
                limit_notice.setText(
                    f"The correction pipeline accepts {CONCEPT_SELECTION_LIMIT} active concepts. "
                    "Uncheck one before adding another."
                )
                return
            limit_notice.clear()
            sync_visible_selection()
            refresh_preview()

        def clear_selection() -> None:
            nonlocal rebuilding
            selected.clear()
            limit_notice.clear()
            rebuilding = True
            preset_list.blockSignals(True)
            for index in range(preset_list.count()):
                preset_list.item(index).setCheckState(Qt.CheckState.Unchecked)
            preset_list.blockSignals(False)
            rebuilding = False
            rebuild_list()

        category_combo.currentTextChanged.connect(lambda _value: rebuild_list())
        search_entry.textChanged.connect(lambda _value: rebuild_list())
        preset_list.itemChanged.connect(on_item_changed)
        clear_button.clicked.connect(clear_selection)
        rebuild_list()

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        sync_visible_selection()
        ordered_selection = [
            concept_preset_key(category, value)
            for category, values in CONCEPT_PRESETS.items()
            for value in values
            if concept_preset_key(category, value) in selected
        ][:CONCEPT_SELECTION_LIMIT]
        concept_text = format_concept_presets(ordered_selection)
        current = str(target.get()).strip()
        if apply_mode.currentText() == "Append to current concepts":
            concept_text = merge_concept_text(current, concept_text)
        target.set(concept_text)
        self.concept_preset_selections[destination] = ordered_selection
        self._save_settings()
        workspace_label = {
            "prompt": "Prompt Corrector",
            "comic": "Comic Story",
        }[destination]
        self.status_var.set(
            f"Applied {len(ordered_selection)} content concepts to {workspace_label}"
        )

    def _visual_direction_target(self, destination: str) -> Value:
        return {
            "comic": self.comic_visual_direction_var,
            "meme": self.meme_visual_direction_var,
        }.get(destination, self.visual_direction_var)

    def _make_visual_preset_button(self, destination: str) -> QtButton:
        button = QtButton("Presets…")
        button.clicked.connect(
            lambda _checked=False, target=destination: self._open_visual_preset_picker(
                target
            )
        )
        self._set_help(
            button,
            "Open the searchable creative-direction library for mood, lighting, palette, "
            "atmosphere, composition, focus, texture, motion, genre, and finish.",
            "Combine ominous mood, moonlight, low fog, and a desaturated blue palette.",
        )
        self.visual_preset_buttons[destination] = button
        return button

    def _open_visual_preset_picker(self, destination: str) -> None:
        destination = destination if destination in {"prompt", "comic", "meme"} else "prompt"
        target = self._visual_direction_target(destination)
        selected = {
            key
            for key in self.visual_preset_selections.get(destination, [])
            if key in VISUAL_DIRECTION_PRESET_KEYS
        }

        dialog = QDialog(self.root)
        dialog.setWindowTitle("Creative direction presets")
        dialog.resize(820, 720)
        outer = QVBoxLayout(dialog)
        intro = QLabel(
            "Combine any number of concrete visual directions. Camera and lens choices remain "
            "in the global Camera control, so this library focuses on the complementary art direction."
        )
        intro.setWordWrap(True)
        outer.addWidget(intro)

        filters = QHBoxLayout()
        filters.addWidget(QLabel("Category"))
        category_combo = QComboBox()
        category_combo.addItems(("All categories", *VISUAL_DIRECTION_PRESETS.keys()))
        self._set_help(
            category_combo,
            "Filters the creative-direction library to one visual dimension.",
            "Choose Lighting to browse only lighting setups.",
        )
        filters.addWidget(category_combo)
        filters.addWidget(QLabel("Search"))
        search_entry = QLineEdit()
        search_entry.setPlaceholderText("Search lighting, fog, nostalgic, texture, noir…")
        self._set_help(
            search_entry,
            "Searches category names and all 420 preset descriptions.",
            "Type fog to find atmospheric fog and fog-related optical effects.",
        )
        filters.addWidget(search_entry, 1)
        outer.addLayout(filters)

        preset_list = QListWidget()
        preset_list.setMinimumHeight(390)
        self._set_help(
            preset_list,
            "Selects any combination of concrete creative-direction presets.",
            "Check moonlit night, low valley mist, and cool desaturated blues.",
        )
        outer.addWidget(preset_list, 1)

        selection_label = QLabel()
        outer.addWidget(selection_label)
        preview = QTextEdit()
        preview.setReadOnly(True)
        preview.setMaximumHeight(130)
        preview.setPlaceholderText("Selected direction text will appear here.")
        self._set_help(
            preview,
            "Previews the exact category-bound visual direction that will be applied.",
            "Lighting: warm golden-hour sunlight; Weather and atmosphere: low valley mist.",
        )
        outer.addWidget(preview)

        apply_row = QHBoxLayout()
        clear_button = QtButton("Clear selection")
        self._set_help(
            clear_button,
            "Unchecks every selected creative-direction preset.",
            "Clear the current combination before building a different look.",
        )
        apply_row.addWidget(clear_button)
        apply_row.addStretch()
        apply_row.addWidget(QLabel("Apply mode"))
        apply_mode = QComboBox()
        apply_mode.addItems(
            ("Replace current visual direction", "Append to current visual direction")
        )
        self._set_help(
            apply_mode,
            "Chooses whether the preset text replaces or extends manually entered direction.",
            "Append a lighting setup while preserving a custom wardrobe-art direction.",
        )
        apply_row.addWidget(apply_mode)
        outer.addLayout(apply_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Apply")
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        outer.addWidget(buttons)

        rebuilding = False

        def sync_visible_selection() -> None:
            if rebuilding:
                return
            for index in range(preset_list.count()):
                item = preset_list.item(index)
                key = str(item.data(Qt.ItemDataRole.UserRole))
                if item.checkState() == Qt.CheckState.Checked:
                    selected.add(key)
                else:
                    selected.discard(key)

        def refresh_preview() -> None:
            selection_label.setText(
                f"{len(selected)} preset{'s' if len(selected) != 1 else ''} selected"
            )
            preview.setPlainText(
                format_visual_direction_presets(tuple(selected))
            )

        def rebuild_list() -> None:
            nonlocal rebuilding
            sync_visible_selection()
            category_filter = category_combo.currentText()
            query = search_entry.text().strip().casefold()
            rebuilding = True
            preset_list.blockSignals(True)
            preset_list.clear()
            for category, values in VISUAL_DIRECTION_PRESETS.items():
                if category_filter != "All categories" and category != category_filter:
                    continue
                for value in values:
                    searchable = f"{category} {value}".casefold()
                    if query and query not in searchable:
                        continue
                    key = visual_preset_key(category, value)
                    item = QListWidgetItem(f"{category}  ·  {value}")
                    item.setData(Qt.ItemDataRole.UserRole, key)
                    item.setFlags(
                        item.flags()
                        | Qt.ItemFlag.ItemIsUserCheckable
                    )
                    item.setCheckState(
                        Qt.CheckState.Checked
                        if key in selected
                        else Qt.CheckState.Unchecked
                    )
                    preset_list.addItem(item)
            preset_list.blockSignals(False)
            rebuilding = False
            refresh_preview()

        def on_item_changed(_item: QListWidgetItem) -> None:
            sync_visible_selection()
            refresh_preview()

        def clear_selection() -> None:
            nonlocal rebuilding
            selected.clear()
            rebuilding = True
            preset_list.blockSignals(True)
            for index in range(preset_list.count()):
                preset_list.item(index).setCheckState(Qt.CheckState.Unchecked)
            preset_list.blockSignals(False)
            rebuilding = False
            rebuild_list()

        category_combo.currentTextChanged.connect(lambda _value: rebuild_list())
        search_entry.textChanged.connect(lambda _value: rebuild_list())
        preset_list.itemChanged.connect(on_item_changed)
        clear_button.clicked.connect(clear_selection)
        rebuild_list()

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        sync_visible_selection()
        ordered_selection = [
            visual_preset_key(category, value)
            for category, values in VISUAL_DIRECTION_PRESETS.items()
            for value in values
            if visual_preset_key(category, value) in selected
        ]
        direction = format_visual_direction_presets(ordered_selection)
        current = str(target.get()).strip()
        if (
            apply_mode.currentText() == "Append to current visual direction"
            and current
            and direction
        ):
            direction = f"{current.rstrip(' .')}. {direction}"
        elif (
            apply_mode.currentText() == "Append to current visual direction"
            and current
            and not direction
        ):
            direction = current
        target.set(direction)
        self.visual_preset_selections[destination] = ordered_selection
        if destination == "meme":
            self.meme_preset_var.set("Custom")
        self._save_settings()
        workspace_label = {
            "prompt": "Prompt Corrector",
            "comic": "Comic Story",
            "meme": "Meme Creator",
        }[destination]
        self.status_var.set(
            f"Applied {len(ordered_selection)} creative-direction presets to {workspace_label}"
        )

    def _effective_mix_inputs(self) -> tuple[str, str, str, str]:
        mix = self.concept_mix_var.get().strip()
        concepts = concept_mix_to_concepts(self.concepts_var.get().strip(), mix)
        weighted_terms = concept_mix_to_weighted_terms(self.weighted_terms_var.get().strip(), mix)
        instructions = self.model_instructions_var.get().strip()
        private_instructions = concept_mix_instruction(mix)
        return concepts, weighted_terms, instructions, private_instructions

    def _custom_preset_snapshot(self) -> dict[str, object]:
        return {
            **self._prompt_option_snapshot(),
            "concepts": self.concepts_var.get(),
            "concept_preset_selection": list(
                self.concept_preset_selections.get("prompt", [])
            ),
            "narrative_preset_selection": {
                kind: list(values)
                for kind, values in self.narrative_preset_selections["prompt"].items()
            },
            "concept_mix": self.concept_mix_var.get(),
            "visual_direction": self.visual_direction_var.get(),
            "visual_preset_selection": list(
                self.visual_preset_selections.get("prompt", [])
            ),
            "goal_headline": self.goal_headline_var.get(),
            "focus": self.focus_var.get(),
            "weighted_terms": self.weighted_terms_var.get(),
            "story_elements": self.story_elements_var.get(),
            "model_instructions": self.model_instructions_var.get(),
            "generation_feedback": self.generation_feedback_var.get(),
            "unload_after_generation": self.unload_after_generation_var.get(),
        }

    def _refresh_custom_preset_combo(self, selected: str = "") -> None:
        if self.custom_preset_combo is None:
            return
        current = selected or self.custom_preset_combo.currentText()
        self.custom_preset_combo.blockSignals(True)
        self.custom_preset_combo.clear()
        self.custom_preset_combo.addItems(sorted(self.custom_presets, key=str.casefold))
        if current in self.custom_presets:
            self.custom_preset_combo.setCurrentText(current)
        self.custom_preset_combo.blockSignals(False)

    def _store_custom_preset(self, name: str) -> bool:
        name = re.sub(r"\s+", " ", name).strip()
        if not name:
            return False
        if name not in self.custom_presets and len(self.custom_presets) >= CUSTOM_PRESET_LIMIT:
            messagebox.showwarning("Preset limit", f"You can save up to {CUSTOM_PRESET_LIMIT} custom presets.")
            return False
        self.custom_presets[name] = self._custom_preset_snapshot()
        self._refresh_custom_preset_combo(name)
        self._save_settings()
        self.status_var.set(f"Saved setup preset '{name}'")
        return True

    def save_custom_preset(self, _checked: bool = False) -> None:
        suggested = self.custom_preset_combo.currentText() if self.custom_preset_combo is not None else ""
        name, accepted = QInputDialog.getText(self.root, "Save setup preset", "Preset name:", text=suggested)
        if accepted:
            self._store_custom_preset(name)

    def _apply_custom_preset(self, name: str) -> bool:
        snapshot = self.custom_presets.get(name)
        if not snapshot:
            return False
        self._load_prompt_options_from_history(snapshot)
        for key, variable in (
            ("concepts", self.concepts_var),
            ("concept_mix", self.concept_mix_var),
            ("visual_direction", self.visual_direction_var),
            ("goal_headline", self.goal_headline_var),
            ("focus", self.focus_var),
            ("weighted_terms", self.weighted_terms_var),
            ("story_elements", self.story_elements_var),
            ("model_instructions", self.model_instructions_var),
            ("generation_feedback", self.generation_feedback_var),
        ):
            variable.set(str(snapshot.get(key, variable.get())))
        stored_visual_presets = snapshot.get("visual_preset_selection", [])
        if isinstance(stored_visual_presets, list):
            self.visual_preset_selections["prompt"] = [
                str(value)
                for value in stored_visual_presets
                if str(value) in VISUAL_DIRECTION_PRESET_KEYS
            ]
        stored_concept_presets = snapshot.get("concept_preset_selection", [])
        if isinstance(stored_concept_presets, list):
            self.concept_preset_selections["prompt"] = [
                str(value)
                for value in stored_concept_presets
                if str(value) in CONCEPT_PRESET_KEYS
            ][:CONCEPT_SELECTION_LIMIT]
        stored_narrative_presets = snapshot.get("narrative_preset_selection", {})
        if isinstance(stored_narrative_presets, dict):
            for kind, valid_keys in (
                ("action", ACTION_PRESET_KEYS),
                ("emotion", EMOTION_PRESET_KEYS),
            ):
                values = stored_narrative_presets.get(kind, [])
                if isinstance(values, list):
                    self.narrative_preset_selections["prompt"][kind] = [
                        str(value)
                        for value in values
                        if str(value) in valid_keys
                    ][:NARRATIVE_PRESET_LIMIT]
        self.unload_after_generation_var.set(
            self._bool_setting(snapshot.get("unload_after_generation"), self.unload_after_generation_var.get())
        )
        self._save_settings()
        self.status_var.set(f"Loaded setup preset '{name}'")
        return True

    def load_custom_preset(self, _checked: bool = False) -> None:
        if self.custom_preset_combo is not None:
            self._apply_custom_preset(self.custom_preset_combo.currentText())

    def delete_custom_preset(self, _checked: bool = False) -> None:
        name = self.custom_preset_combo.currentText() if self.custom_preset_combo is not None else ""
        if not name or name not in self.custom_presets:
            return
        if not messagebox.askyesno("Delete preset", f"Delete the saved setup preset '{name}'?"):
            return
        del self.custom_presets[name]
        self._refresh_custom_preset_combo()
        self._save_settings()
        self.status_var.set(f"Deleted setup preset '{name}'")

    def export_custom_presets(self, _checked: bool = False) -> None:
        if not self.custom_presets:
            messagebox.showwarning("No presets", "There are no custom presets to export.")
            return
        filename, _filter = QFileDialog.getSaveFileName(
            self.root, "Export setup presets", "promptcorrector-presets.json", "JSON files (*.json)"
        )
        if not filename:
            return
        try:
            Path(filename).write_text(json.dumps(self.custom_presets, indent=2, sort_keys=True), encoding="utf-8")
        except OSError as exc:
            messagebox.showerror("Export failed", str(exc))
            return
        self.status_var.set(f"Exported {len(self.custom_presets)} setup presets")

    def import_custom_presets(self, _checked: bool = False) -> None:
        filename, _filter = QFileDialog.getOpenFileName(
            self.root, "Import setup presets", "", "JSON files (*.json)"
        )
        if not filename:
            return
        try:
            loaded = json.loads(Path(filename).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            messagebox.showerror("Import failed", str(exc))
            return
        imported = self._custom_presets_setting(loaded)
        if not imported:
            messagebox.showwarning("Import presets", "That file does not contain valid setup presets.")
            return
        merged = {**self.custom_presets, **imported}
        self.custom_presets = dict(list(merged.items())[:CUSTOM_PRESET_LIMIT])
        self._refresh_custom_preset_combo()
        self._save_settings()
        self.status_var.set(f"Imported {len(imported)} setup presets")

    def _menu_check(self, menu, label: str, variable: Value) -> QAction:
        action = menu.addAction(label)
        action.setCheckable(True)
        action.setChecked(bool(variable.get()))
        action.toggled.connect(variable.set)
        variable.subscribe(lambda value, a=action: a.setChecked(bool(value)))
        return action

    def _menu_choices(self, menu, label: str, variable: Value, choices) -> None:
        submenu = menu.addMenu(label)
        group = QActionGroup(self.root)
        group.setExclusive(True)
        for choice in choices:
            action = submenu.addAction(choice)
            action.setCheckable(True)
            action.setChecked(variable.get() == choice)
            action.triggered.connect(lambda _checked=False, v=choice: variable.set(v))
            variable.subscribe(lambda value, a=action, v=choice: a.setChecked(value == v))
            group.addAction(action)

    def _build_menu(self) -> None:
        bar = self.root.menuBar()

        file_menu = bar.addMenu("File")
        save_action = file_menu.addAction("Save now")
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self._save_settings)
        file_menu.addSeparator()
        file_menu.addAction("Import setup presets…", self.import_custom_presets)
        file_menu.addAction("Export setup presets…", self.export_custom_presets)
        file_menu.addSeparator()
        exit_action = file_menu.addAction("Exit")
        exit_action.setShortcut("Ctrl+Q")
        exit_action.setMenuRole(QAction.MenuRole.QuitRole)
        exit_action.triggered.connect(self._on_close)

        edit_menu = bar.addMenu("Edit")
        undo_action = edit_menu.addAction("Undo")
        undo_action.setShortcut("Ctrl+Z")
        undo_action.triggered.connect(lambda: self._invoke_focused_editor("undo"))
        redo_action = edit_menu.addAction("Redo")
        redo_action.setShortcut("Ctrl+Shift+Z")
        redo_action.triggered.connect(lambda: self._invoke_focused_editor("redo"))
        edit_menu.addSeparator()
        clear_menu = edit_menu.addMenu("Clear workspace")
        clear_action = clear_menu.addAction("Clear Prompt Corrector")
        clear_action.setShortcut("Ctrl+Shift+Backspace")
        clear_action.triggered.connect(self.clear_single_image)
        clear_menu.addAction("Clear Comic Story", self.clear_comic_story)
        clear_menu.addAction("Clear Meme Creator", self.clear_meme)

        create = bar.addMenu("Create")
        prompt = create.addMenu("Prompt Corrector")
        correct_action = prompt.addAction("Correct prompt")
        correct_action.setShortcut("Ctrl+Return")
        correct_action.triggered.connect(self.correct_prompt)
        copy_action = prompt.addAction("Copy corrected prompt")
        copy_action.setShortcut("Ctrl+Shift+C")
        copy_action.triggered.connect(self.copy_corrected)
        iterate_action = prompt.addAction("Iterate corrected prompt")
        iterate_action.setShortcut("Ctrl+Shift+R")
        iterate_action.triggered.connect(self.iterate_corrected_prompt)
        focus_action = prompt.addAction("Focus prompt editor")
        focus_action.setShortcut("Ctrl+L")
        focus_action.triggered.connect(lambda: self.draft_text.setFocus())
        comic = create.addMenu("Comic Story")
        comic.addAction("Generate comic prompt", self.correct_comic_story)
        comic.addAction("Invent all comic panels", self.invent_all_comic_panels)
        comic.addAction("Copy comic result", self.copy_comic_result)
        meme = create.addMenu("Meme Creator")
        meme.addAction("Generate meme prompt", self.correct_meme)
        meme.addAction("Copy meme result", self.copy_meme_result)
        chat = create.addMenu("Model Chat")
        send_chat_action = chat.addAction("Send message")
        send_chat_action.setShortcut("Ctrl+Shift+Return")
        send_chat_action.triggered.connect(self.send_chat_message)
        chat.addAction("New chat", lambda _checked=False: self.clear_chat())
        chat.addAction("Copy last response", self.copy_last_chat_response)

        model_menu = bar.addMenu("Model")
        connection = model_menu.addMenu("Connection")
        connection.addAction("Test LM Studio", self.test_lm_studio_connection)
        connection.addAction("Save settings", self._save_settings)
        connection.addSeparator()
        self._menu_check(
            connection,
            "Unload model after correction",
            self.unload_after_generation_var,
        )
        processing = model_menu.addMenu("Rewrite and safety")
        self._menu_check(processing, "Preserve wording strictly", self.preserve_var)
        self._menu_check(
            processing,
            "Artistic detail freedom",
            self.artistic_detail_freedom_var,
        )
        self._menu_check(processing, "Quote rendered text", self.quote_text_var)
        processing.addSeparator()
        for label, variable in (
            ("Clean generator constraints", self.clean_constraints_var),
            ("Safe for work", self.safe_for_work_var),
            ("Explicit adult (NSFW)", self.explicit_nsfw_var),
            ("Fix logic conflicts", self.fix_logic_var),
            ("Enhance actions", self.enhance_actions_var),
            ("Invent and extend story", self.develop_story_var),
            ("Altered encoder safe", self.altered_encoder_var),
        ):
            self._menu_check(processing, label, variable)

        generation = model_menu.addMenu("Generation passes")
        self._menu_check(generation, "Use fixed seed", self.fixed_seed_var)
        self._menu_check(generation, "Thinking mode", self.thinking_mode_var)
        self._menu_check(generation, "Audit and repair", self.audit_repair_var)
        self._menu_check(generation, "Show generator setup recommendation", self.include_settings_var)

        research = bar.addMenu("Research")
        self._menu_check(research, "Grounded web verification", self.live_research_var)
        self._menu_choices(research, "Search engine", self.search_engine_var, TEXT_RESEARCH_ENGINES)
        research.addSeparator()
        references = research.addMenu("Reference images")
        references.setToolTipsVisible(True)
        self._set_help(
            references,
            "Reference-image controls separated by creative workspace.",
            "Analyze a costume image in Comic Story without adding it to Meme Creator.",
        )
        self._menu_check(
            references,
            "Analyze for Prompt Corrector",
            self.reference_images_var,
        )
        self._menu_check(
            references,
            "Analyze for Comic Story",
            self.comic_reference_images_var,
        )
        self._menu_check(
            references,
            "Analyze for Meme Creator",
            self.meme_reference_images_var,
        )
        self._menu_choices(
            references,
            "Web image source",
            self.reference_image_source_var,
            REFERENCE_IMAGE_SOURCES,
        )

        library = bar.addMenu("Library")
        history = library.addMenu("History")
        history.addAction("Load selected", self.load_selected_history_prompt)
        history.addAction("Copy selected prompt", self.copy_selected_history_prompt)
        history.addAction("Delete selected", self.delete_selected_history_prompt)
        history.addAction("Rename selected", self.rename_selected_history_prompt)
        history.addAction("Pin or unpin selected", self.toggle_selected_history_pin)
        history.addSeparator()
        history.addAction("Clear all", self.clear_prompt_history)
        library.addAction(
            "Show Activity",
            lambda _checked=False: self.show_library_tab("Activity"),
        )
        library.addAction(
            "Show History",
            lambda _checked=False: self.show_library_tab("History"),
        )
        library.addAction(
            "Show References",
            lambda _checked=False: self.show_library_tab("References"),
        )
        library.addSeparator()
        library.addAction("Clear Activity", self.clear_activity_history)

        self.view_menu = bar.addMenu("View")
        self.setup_action = self.view_menu.addAction("Show shared settings")
        self.setup_action.setCheckable(True)
        self.setup_action.setChecked(False)
        self.setup_action.setShortcut("Ctrl+Shift+Space")
        self.setup_action.toggled.connect(lambda visible: self.setup_tabs.setVisible(visible))
        self._menu_check(
            self.view_menu,
            "Remember window size",
            self.remember_window_size_var,
        )

    def _build_ui(self) -> None:
        self._build_menu()
        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(18, 14, 18, 18)
        outer.setSpacing(12)

        quick = QHBoxLayout()
        target_label = QLabel("Generator")
        self._set_help(
            target_label,
            "Selects the image model that will receive the corrected prompt.",
            "Choose FLUX.2 Klein 9B for a local Klein workflow.",
        )
        quick.addWidget(target_label)
        target_combo = self._bind_combo(self.generator_target_var, GENERATOR_TARGETS)
        self._set_help(
            target_combo,
            "Adapts prompt structure and setup guidance to the destination generator.",
            "FLUX.2 Klein uses explicit detailed prompts because it has no prompt upsampling.",
        )
        quick.addWidget(target_combo)
        workflow_label = QLabel("Workflow")
        self._set_help(
            workflow_label,
            "Selects how strictly the corrected prompt must preserve the draft.",
            "Exact preserves counts, positions, exclusions, and quoted text.",
        )
        quick.addWidget(workflow_label)
        profile_combo = self._bind_combo(self.workflow_profile_var, WORKFLOW_PROFILES)
        self._set_help(
            profile_combo,
            "Exact preserves every stated fact; Improve adds restrained polish; Explore permits invention.",
            "Choose Exact when the selected generator should follow the request as closely as possible.",
        )
        quick.addWidget(profile_combo)
        self.camera_label = QLabel("Camera")
        self._help(self.camera_label, "Camera")
        quick.addWidget(self.camera_label)
        self.camera_combo = self._bind_combo(
            self.camera_control_var,
            CAMERA_CONTROL_PRESETS,
            editable=True,
        )
        self._help(self.camera_combo, "Camera")
        self.camera_combo.setMinimumContentsLength(18)
        quick.addWidget(self.camera_combo)
        model_label = QLabel("Model")
        self._help(model_label, "Model")
        quick.addWidget(model_label)
        self.model_combo = self._bind_combo(self.model_var, self.available_models, editable=True)
        self._help(self.model_combo, "Model")
        quick.addWidget(self.model_combo, 1)
        self.profile_summary_label = QLabel()
        self._set_help(
            self.profile_summary_label,
            "Summarizes the active fidelity and creativity policy.",
            "Exact uses raw creativity and forbids invented content.",
        )
        self.workflow_profile_var.subscribe(lambda _value: self._update_profile_summary())
        quick.addWidget(self.profile_summary_label, 1)
        advanced_button = QtButton("Settings")
        self._set_help(
            advanced_button,
            "Shows settings shared by the image-prompt workspaces and the LM Studio connection.",
            "Open this to adjust generation, processing, research, or connection settings.",
        )
        advanced_button.setCheckable(True)
        advanced_button.setChecked(False)
        if self.setup_action is not None:
            advanced_button.toggled.connect(self.setup_action.setChecked)
            self.setup_action.toggled.connect(advanced_button.setChecked)
        quick.addWidget(advanced_button)
        outer.addLayout(quick)

        setup_tabs = QTabWidget()
        self.setup_tabs = setup_tabs
        setup_tabs.setVisible(False)
        advanced_button.toggled.connect(setup_tabs.setVisible)
        setup_tabs.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.mode_tabs = QTabWidget()
        prompt_mode = QWidget()
        prompt_outer = QVBoxLayout(prompt_mode)
        prompt_outer.setContentsMargins(0, 8, 0, 0)
        prompt_outer.setSpacing(12)

        prompt_options_bar = QHBoxLayout()
        self.prompt_options_button = QtButton("Single-image options")
        self.prompt_options_button.setCheckable(True)
        self.prompt_options_button.setChecked(False)
        self._help(self.prompt_options_button, "Single-image options")
        prompt_options_bar.addWidget(self.prompt_options_button)
        prompt_options_hint = QLabel(
            "Optional direction, weighting, model guidance, and reference images for Prompt Corrector only."
        )
        prompt_options_hint.setStyleSheet("color: #8993a5;")
        self._help(prompt_options_hint, "Single-image options")
        prompt_options_bar.addWidget(prompt_options_hint)
        prompt_options_bar.addStretch()
        prompt_outer.addLayout(prompt_options_bar)
        self.prompt_options_button.toggled.connect(
            lambda visible: advanced_button.setChecked(False) if visible else None
        )
        advanced_button.toggled.connect(
            lambda visible: self.prompt_options_button.setChecked(False) if visible else None
        )

        prompt_page = QWidget()
        self.prompt_guidance_page = prompt_page
        prompt_page.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        prompt_page.setVisible(False)
        self.prompt_options_button.toggled.connect(prompt_page.setVisible)
        prompt_page_layout = QHBoxLayout(prompt_page)
        direction_group = QGroupBox("Single-image direction")
        direction_grid = QGridLayout(direction_group)
        direction_fields = (
            ("Concepts", self.concepts_var, "concepts"),
            ("Concept/style mix", self.concept_mix_var, "concept_mix"),
            ("Visual direction", self.visual_direction_var, "visual_direction"),
            ("Goal headline", self.goal_headline_var, "goal_headline"),
            ("Focus", self.focus_var, "focus"),
            ("Story beat", self.story_elements_var, "story_elements"),
        )
        for row, (label, variable, field_name) in enumerate(direction_fields):
            direction_grid.addWidget(QLabel(label), row, 0)
            if variable is self.story_elements_var:
                field = self._help(self._bind_text(variable, maximum_height=84), label)
                self.story_elements_entry = field
                field.setPlaceholderText(
                    "Optional action, reaction, or story evidence to show in this one still image."
                )
                narrative_controls = QHBoxLayout()
                narrative_controls.addWidget(field, 1)
                narrative_presets = QVBoxLayout()
                narrative_presets.addWidget(
                    self._make_narrative_preset_button("prompt", "action")
                )
                narrative_presets.addWidget(
                    self._make_narrative_preset_button("prompt", "emotion")
                )
                narrative_controls.addLayout(narrative_presets)
                direction_grid.addLayout(narrative_controls, row, 1)
            elif variable is self.concepts_var:
                field = self._help(self._bind_line(variable), label)
                field.setPlaceholderText(
                    "Up to 8 comma-separated subjects, places, objects, motifs, or ideas"
                )
                concept_controls = QHBoxLayout()
                concept_controls.addWidget(field, 1)
                concept_controls.addWidget(
                    self._make_concept_preset_button("prompt")
                )
                direction_grid.addLayout(concept_controls, row, 1)
            elif variable is self.concept_mix_var:
                field = self._help(self._bind_line(variable), label)
                field.setPlaceholderText("watercolor:60%, cyberpunk:40%")
                mix_controls = QHBoxLayout()
                mix_controls.addWidget(field, 1)
                mix_button = QtButton("Mix...")
                self._help(mix_button, "Mix...")
                mix_button.clicked.connect(self._open_concept_mix_editor)
                mix_controls.addWidget(mix_button)
                direction_grid.addLayout(mix_controls, row, 1)
            elif variable is self.visual_direction_var:
                field = self._bind_line(variable)
                self._set_help(
                    field,
                    "Sets concrete mood, lighting, palette, atmosphere, composition, texture, "
                    "motion, genre, and finish direction for Prompt Corrector.",
                    "Ominous moonlight, low fog, desaturated blue, and deep layered composition.",
                )
                field.setPlaceholderText(
                    "Mood, lighting, palette, atmosphere, composition, texture, and finish"
                )
                visual_controls = QHBoxLayout()
                visual_controls.addWidget(field, 1)
                visual_controls.addWidget(
                    self._make_visual_preset_button("prompt")
                )
                direction_grid.addLayout(visual_controls, row, 1)
            else:
                field = self._help(self._bind_line(variable), label)
                direction_grid.addWidget(field, row, 1)
            invent_button = QtButton("Invent")
            invent_button.clicked.connect(
                lambda _checked=False, name=field_name: self.invent_single_image_field(name)
            )
            self._set_help(
                invent_button,
                f"Expand the entered {label.lower()} as a mandatory seed, or invent it when blank.",
                f"Enter a rough {label.lower()} to develop, or leave it blank to use the other fields.",
            )
            self.single_image_invent_buttons.append(invent_button)
            direction_grid.addWidget(invent_button, row, 2)
            direction_grid.addWidget(
                self._make_invent_recall_button(f"single:{field_name}"),
                row,
                3,
            )
        direction_grid.setColumnStretch(1, 1)
        prompt_page_layout.addWidget(direction_group, 1)

        guidance_group = QGroupBox("Model guidance and references")
        guidance_grid = QGridLayout(guidance_group)
        guidance_fields = (
            ("Weighted words", self.weighted_terms_var, "weighted_terms"),
            ("Model instructions", self.model_instructions_var, "model_instructions"),
            ("Generation feedback", self.generation_feedback_var, "generation_feedback"),
        )
        for row, (label, variable, field_name) in enumerate(guidance_fields):
            guidance_grid.addWidget(QLabel(label), row, 0)
            if variable is self.weighted_terms_var:
                field = self._help(self._bind_weighted_terms_line(), label)
                self.weighted_terms_entry = field
                weighted_controls = QHBoxLayout()
                weighted_controls.addWidget(field, 1)
                decrease_button = QtButton("−")
                self._help(decrease_button, "−")
                decrease_button.clicked.connect(self._decrease_weighted_term)
                weighted_controls.addWidget(decrease_button)
                increase_button = QtButton("+")
                self._help(increase_button, "+")
                increase_button.clicked.connect(self._increase_weighted_term)
                weighted_controls.addWidget(increase_button)
                guidance_grid.addLayout(weighted_controls, row, 1)
            else:
                field = self._help(self._bind_line(variable), label)
                guidance_grid.addWidget(field, row, 1)
            invent_button = QtButton("Invent")
            invent_button.clicked.connect(
                lambda _checked=False, name=field_name: self.invent_single_image_field(name)
            )
            self._set_help(
                invent_button,
                f"Expand the entered {label.lower()} as a mandatory seed, or invent it when blank.",
                f"Enter rough {label.lower()} to develop, or leave it blank to use the prompt context.",
            )
            self.single_image_invent_buttons.append(invent_button)
            guidance_grid.addWidget(invent_button, row, 2)
            guidance_grid.addWidget(
                self._make_invent_recall_button(f"single:{field_name}"),
                row,
                3,
            )
        reference_analysis = self._bind_check(
            "Analyze single-image references",
            self.reference_images_var,
        )
        self._set_help(
            reference_analysis,
            "Analyzes local reference images and explicit concept-glossary images only for Prompt Corrector.",
            "Add a costume image in References, then enable this for one still-image prompt.",
        )
        guidance_grid.addWidget(reference_analysis, len(guidance_fields), 0, 1, 2)
        guidance_grid.addWidget(QLabel("Reference source"), len(guidance_fields) + 1, 0)
        reference_source = self._bind_combo(
            self.reference_image_source_var,
            REFERENCE_IMAGE_SOURCES,
        )
        self._set_help(
            reference_source,
            "Chooses the source for automatic concept-glossary images in Prompt Corrector.",
            "Use Wikipedia/Wikimedia for an architecture concept.",
        )
        guidance_grid.addWidget(reference_source, len(guidance_fields) + 1, 1)
        guidance_grid.setColumnStretch(1, 1)
        prompt_page_layout.addWidget(guidance_group, 1)
        prompt_outer.addWidget(prompt_page)

        generation_page = QWidget()
        generation_grid = QGridLayout(generation_page)
        generation_grid.addWidget(QLabel("Mode"), 0, 0)
        generation_grid.addWidget(self._help(self._bind_combo(self.mode_var, PROMPT_MODES), "Mode"), 0, 1)
        generation_grid.addWidget(QLabel("Detail"), 0, 2)
        generation_grid.addWidget(self._help(self._bind_combo(self.detail_var, DETAIL_LEVELS), "Detail"), 0, 3)
        generation_grid.addWidget(QLabel("Output length"), 1, 0)
        generation_grid.addWidget(
            self._help(self._bind_combo(self.output_length_var, OUTPUT_LENGTHS), "Output length"),
            1,
            1,
        )
        generation_grid.addWidget(
            self._bind_check(
                "Artistic detail freedom",
                self.artistic_detail_freedom_var,
            ),
            1,
            2,
            1,
            2,
        )
        generation_grid.addWidget(QLabel("Creative freedom"), 2, 0)
        generation_grid.addWidget(
            self._help(self._bind_combo(self.risk_level_var, RISK_LEVELS), "Creative freedom"),
            2,
            1,
        )
        generation_grid.addWidget(QLabel("Preset"), 2, 2)
        generation_grid.addWidget(
            self._help(self._bind_combo(self.prompt_preset_var, PROMPT_PRESETS), "Preset"),
            2,
            3,
        )
        generation_grid.addWidget(QLabel("Variations"), 3, 0)
        generation_grid.addWidget(self._help(self._bind_spin(self.variation_var, 1, 3), "Variations"), 3, 1)
        tuning = QHBoxLayout()
        tuning.addWidget(QLabel("Temperature"))
        tuning.addWidget(self._help(self._bind_double_spin(self.temperature_var), "Temperature"))
        tuning.addWidget(QLabel("Context tokens"))
        context_tokens = self._bind_combo(
            self.context_token_budget_var,
            CONTEXT_TOKEN_CHOICES,
        )
        self._help(context_tokens, "Context tokens")
        tuning.addWidget(context_tokens)
        generation_grid.addLayout(tuning, 3, 2, 1, 2)
        fixed_seed = self._bind_check("Use fixed seed", self.fixed_seed_var)
        generation_grid.addWidget(fixed_seed, 4, 0)
        generation_grid.addWidget(QLabel("Sampling seed"), 4, 2)
        self.seed_spin = self._bind_spin(self.seed_var, 0, 2_147_483_647)
        self._help(self.seed_spin, "Sampling seed")
        self.seed_spin.setEnabled(bool(self.fixed_seed_var.get()))
        self.fixed_seed_var.subscribe(
            lambda enabled: self.seed_spin.setEnabled(bool(enabled))
            if self.seed_spin is not None
            else None
        )
        generation_grid.addWidget(self.seed_spin, 4, 3)
        generation_grid.addWidget(QLabel("Saved setup"), 5, 0)
        self.custom_preset_combo = QComboBox()
        self._help(self.custom_preset_combo, "Saved setup")
        self._refresh_custom_preset_combo()
        generation_grid.addWidget(self.custom_preset_combo, 5, 1)
        preset_controls = QHBoxLayout()
        for label, callback, description, example in (
            ("Load", self.load_custom_preset, "Apply the selected saved setup.", "Load a saved Product shots setup."),
            ("Save as…", self.save_custom_preset, "Save the current controls as a named setup.", "Save these values as Cinematic comic."),
            ("Delete", self.delete_custom_preset, "Delete the selected saved setup.", "Remove an obsolete test setup."),
            ("Import", self.import_custom_presets, "Import saved setups from a JSON file.", "Import presets shared from another computer."),
            ("Export", self.export_custom_presets, "Export all saved setups to a JSON file.", "Create a backup before reinstalling."),
        ):
            button = QtButton(label)
            self._set_help(button, description, example)
            button.clicked.connect(callback)
            preset_controls.addWidget(button)
        generation_grid.addLayout(preset_controls, 5, 2, 1, 2)
        generation_grid.setColumnStretch(1, 1)
        generation_grid.setColumnStretch(3, 1)
        setup_tabs.addTab(generation_page, "Generation")

        krea_page = QWidget()
        krea_grid = QGridLayout(krea_page)
        krea_grid.addWidget(QLabel("Creativity"), 0, 0)
        krea_grid.addWidget(
            self._help(self._bind_combo(self.creativity_var, CREATIVITY_LEVELS), "Creativity"),
            0,
            1,
        )
        self._add_slider(krea_grid, "Intensity", self.intensity_var, 1)
        self._add_slider(krea_grid, "Complexity", self.complexity_var, 2)
        self._add_slider(krea_grid, "Movement", self.movement_var, 3)
        krea_grid.setColumnStretch(1, 1)
        self.generator_controls_page = krea_page
        self.generator_controls_tab_index = setup_tabs.addTab(krea_page, "Krea controls")

        options_page = QWidget()
        options_layout = QHBoxLayout(options_page)

        rewrite_group = QGroupBox("Rewrite rules")
        rewrite_grid = QGridLayout(rewrite_group)
        rewrite_vars = (
            ("Preserve wording strictly", self.preserve_var),
            ("Quote rendered text", self.quote_text_var),
            ("Fix logic conflicts", self.fix_logic_var),
            ("Enhance actions", self.enhance_actions_var),
            ("Invent and extend story", self.develop_story_var),
            ("Clean generator constraints", self.clean_constraints_var),
            ("Safe for work", self.safe_for_work_var),
            ("Explicit adult (NSFW)", self.explicit_nsfw_var),
            ("Altered encoder safe", self.altered_encoder_var),
        )
        for index, (label, variable) in enumerate(rewrite_vars):
            rewrite_grid.addWidget(self._bind_check(label, variable), index // 2, index % 2)
        options_layout.addWidget(rewrite_group, 2)

        quality_group = QGroupBox("Quality and session")
        quality_grid = QGridLayout(quality_group)
        quality_vars = (
            ("Thinking mode", self.thinking_mode_var),
            ("Audit and repair", self.audit_repair_var),
            ("Show generator setup recommendation", self.include_settings_var),
            ("Unload model after correction", self.unload_after_generation_var),
        )
        for row, (label, variable) in enumerate(quality_vars):
            control = self._bind_check(label, variable)
            if label == "Show generator setup recommendation":
                self._set_help(
                    control,
                    "Shows generation controls separately from the image prompt.",
                    "Copy creativity=raw into Krea instead of placing it in the prompt.",
                )
            quality_grid.addWidget(control, row, 0)
        options_layout.addWidget(quality_group, 1)

        research_group = QGroupBox("Web research")
        research_grid = QGridLayout(research_group)
        research_grid.addWidget(self._bind_check("Grounded web verification", self.live_research_var), 0, 0, 1, 2)
        research_grid.addWidget(QLabel("Search engine"), 1, 0)
        research_grid.addWidget(
            self._help(self._bind_combo(self.search_engine_var, TEXT_RESEARCH_ENGINES), "Search engine"),
            1,
            1,
        )
        research_grid.setColumnStretch(1, 1)
        options_layout.addWidget(research_group, 2)
        setup_tabs.addTab(options_page, "Processing")

        connection_page = QWidget()
        connection_grid = QGridLayout(connection_page)
        connection_hint = QLabel(
            "The selected model is shown in the global bar above. These connection "
            "settings are shared by Prompt Corrector, Comic Story, Meme Creator, "
            "Model Chat, and Workbench."
        )
        connection_hint.setWordWrap(True)
        connection_hint.setStyleSheet("color: #8993a5;")
        self._help(connection_hint, "Connection")
        connection_grid.addWidget(connection_hint, 0, 0, 1, 6)
        connection_grid.addWidget(QLabel("Host"), 1, 0)
        connection_grid.addWidget(self._help(self._bind_line(self.lm_host_var), "Host"), 1, 1)
        connection_grid.addWidget(QLabel("Port"), 1, 2)
        connection_grid.addWidget(self._help(self._bind_line(self.lm_port_var), "Port"), 1, 3)
        connection_grid.addWidget(QLabel("Timeout"), 1, 4)
        connection_grid.addWidget(
            self._help(self._bind_spin(self.lm_timeout_var, 30, 3600, 30), "Timeout"),
            1,
            5,
        )
        connection_grid.addWidget(QLabel("API URL"), 2, 0)
        connection_grid.addWidget(
            self._help(self._bind_line(self.base_url_var, read_only=True), "API URL"),
            2,
            1,
            1,
            4,
        )
        self.test_connection_button = QtButton("Test connection")
        self._help(self.test_connection_button, "Test connection")
        self.test_connection_button.clicked.connect(self.test_lm_studio_connection)
        connection_grid.addWidget(self.test_connection_button, 2, 5)
        connection_grid.setColumnStretch(1, 3)
        connection_grid.setColumnStretch(3, 1)
        setup_tabs.addTab(connection_page, "Connection")
        outer.addWidget(setup_tabs)

        workspace = QSplitter(Qt.Orientation.Horizontal)
        prompt_panel = QWidget()
        prompt_layout = QVBoxLayout(prompt_panel)
        prompt_layout.setContentsMargins(0, 0, 0, 0)
        draft_group = QGroupBox("Your prompt")
        draft_layout = QVBoxLayout(draft_group)
        draft_invent_row = QHBoxLayout()
        draft_invent_row.addStretch()
        self.single_draft_invent_button = QtButton("Invent prompt")
        self.single_draft_invent_button.clicked.connect(
            lambda: self.invent_single_image_field("draft")
        )
        self._set_help(
            self.single_draft_invent_button,
            "Expand the entered prompt as a mandatory creative seed, or invent a prompt when blank.",
            "Enter a rough concept to develop, or leave the prompt blank to invent from the other fields.",
        )
        self.single_image_invent_buttons.append(self.single_draft_invent_button)
        draft_invent_row.addWidget(self.single_draft_invent_button)
        draft_invent_row.addWidget(
            self._make_invent_recall_button("single:draft")
        )
        draft_layout.addLayout(draft_invent_row)
        self.draft_text = QtTextEdit()
        self.draft_text.setPlaceholderText("Paste or type a rough image prompt…")
        self._help(self.draft_text, "Your prompt")
        self.draft_text.textChanged.connect(self._on_draft_modified)
        self.draft_text.increase_weight_callback = self._increase_draft_weighted_term
        self.draft_text.decrease_weight_callback = self._decrease_draft_weighted_term
        draft_layout.addWidget(self.draft_text)
        self.draft_counter_label = QLabel()
        self.draft_counter_label.setStyleSheet("color: #8993a5;")
        self._set_help(
            self.draft_counter_label,
            "Shows the current draft's word and character count.",
            "Use it to stay below a desired prompt length.",
        )
        draft_layout.addWidget(self.draft_counter_label)
        prompt_layout.addWidget(draft_group, 1)

        controls = QHBoxLayout()
        self.correct_button = QtButton("Correct prompt")
        self.correct_button.setObjectName("primaryButton")
        self.correct_button.clicked.connect(self.correct_prompt)
        controls.addWidget(self.correct_button)
        self.stop_button = QtButton("Stop")
        self.stop_button.clicked.connect(self.stop_current_request)
        self.stop_button.setEnabled(False)
        controls.addWidget(self.stop_button)
        self.single_clear_button = QtButton("Clear all")
        self._set_help(
            self.single_clear_button,
            "Clears the Single Image prompt, result, optional direction, guidance, presets, and local references.",
            "Use before starting a completely different single-image prompt.",
        )
        self.single_clear_button.clicked.connect(self.clear_single_image)
        controls.addWidget(self.single_clear_button)
        weight_down_button = QtButton("Weight −")
        self._help(weight_down_button, "Weight −")
        weight_down_button.clicked.connect(self._decrease_draft_weighted_term)
        controls.addWidget(weight_down_button)
        weight_up_button = QtButton("Weight +")
        self._help(weight_up_button, "Weight +")
        weight_up_button.clicked.connect(self._increase_draft_weighted_term)
        controls.addWidget(weight_up_button)
        copy_button = QtButton("Copy corrected")
        copy_button.clicked.connect(self.copy_corrected)
        controls.addWidget(copy_button)
        self.iterate_button = QtButton("Iterate result…")
        self._help(self.iterate_button, "Iterate result")
        self.iterate_button.clicked.connect(self.iterate_corrected_prompt)
        self.iterate_button.setEnabled(False)
        controls.addWidget(self.iterate_button)
        controls.addStretch()
        self.status_label = QLabel(str(self.status_var.get()))
        self.status_label.setObjectName("statusLabel")
        self._set_help(
            self.status_label,
            "Shows the current connection or generation status.",
            "Look here for Stopped or Correction complete.",
        )
        self.status_var.subscribe(lambda value: self.status_label.setText(str(value)))
        controls.addWidget(self.status_label)
        prompt_layout.addLayout(controls)
        self.progress_bar = QProgressBar()
        self._set_help(
            self.progress_bar,
            "Shows approximate progress through research and model passes.",
            "A second pass advances it during audit and repair.",
        )
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_var.subscribe(lambda value: self.progress_bar.setValue(round(float(value))))
        prompt_layout.addWidget(self.progress_bar)
        self.progress_label = QLabel(str(self.progress_text_var.get()))
        self._set_help(
            self.progress_label,
            "Names the workflow step currently in progress.",
            "It may show Analyzing reference images.",
        )
        self.progress_text_var.subscribe(lambda value: self.progress_label.setText(str(value)))
        prompt_layout.addWidget(self.progress_label)

        corrected_group = QGroupBox("Corrected prompt")
        corrected_layout = QVBoxLayout(corrected_group)
        result_tabs = QTabWidget()
        self.corrected_text = QtTextEdit()
        self.corrected_text.setPlaceholderText("The corrected prompt will appear here.")
        self._help(self.corrected_text, "Corrected prompt")
        self.corrected_text.textChanged.connect(self._on_corrected_modified)
        result_tabs.addTab(self.corrected_text, "Result")
        self.diff_text = QtTextEdit()
        self.diff_text.setReadOnly(True)
        self._help(self.diff_text, "Changes")
        result_tabs.addTab(self.diff_text, "Changes")
        corrected_layout.addWidget(result_tabs)
        self.corrected_counter_label = QLabel()
        self._set_help(
            self.corrected_counter_label,
            "Shows the corrected prompt's word and character count.",
            "Compare it with the requested maximum length.",
        )
        corrected_layout.addWidget(self.corrected_counter_label)
        self.krea_recommendation_label = QLabel()
        self.krea_recommendation_label.setWordWrap(True)
        self.krea_recommendation_label.setStyleSheet("color: #9aa6bd;")
        self._set_help(
            self.krea_recommendation_label,
            "Shows controls for the selected generator outside the image prompt.",
            "Use creativity=raw for Krea or four steps with guidance 1.0 for FLUX Klein.",
        )
        corrected_layout.addWidget(self.krea_recommendation_label)
        self.copy_krea_button = QtButton("Copy generator setup")
        self._set_help(
            self.copy_krea_button,
            "Copies only the separate generator control recommendation.",
            "Paste the prompt into the selected generator, then apply this setup independently.",
        )
        self.copy_krea_button.clicked.connect(self.copy_krea_recommendation)
        corrected_layout.addWidget(self.copy_krea_button)
        for variable in (
            self.include_settings_var,
            self.creativity_var,
            self.intensity_var,
            self.complexity_var,
            self.movement_var,
        ):
            variable.subscribe(lambda _value: self._update_krea_recommendation())
        prompt_layout.addWidget(corrected_group, 1)
        workspace.addWidget(prompt_panel)

        side_tabs = QTabWidget()
        self.library_tabs = side_tabs
        activity_page = QWidget()
        activity_layout = QVBoxLayout(activity_page)
        activity_controls = QHBoxLayout()
        activity_controls.addWidget(QLabel("Show"))
        self.activity_scope_combo = QComboBox()
        self.activity_scope_combo.addItems(
            (
                "All workspaces",
                "Prompt Corrector",
                "Comic Story",
                "Meme Creator",
                "System",
            )
        )
        self.activity_scope_combo.currentTextChanged.connect(
            self._refresh_activity_text
        )
        activity_controls.addWidget(self.activity_scope_combo, 1)
        clear_activity_button = QtButton("Clear")
        clear_activity_button.clicked.connect(self.clear_activity_history)
        self._set_help(
            clear_activity_button,
            "Clears the persisted workflow activity history.",
            "Clear old connection diagnostics after resolving a server problem.",
        )
        activity_controls.addWidget(clear_activity_button)
        activity_layout.addLayout(activity_controls)
        self.activity_text = QtTextEdit()
        self.activity_text.setReadOnly(True)
        self._help(self.activity_text, "Activity")
        activity_layout.addWidget(self.activity_text)
        side_tabs.addTab(activity_page, "Activity")
        history_page = QWidget()
        history_layout = QVBoxLayout(history_page)
        self.history_search_entry = QLineEdit()
        self.history_search_entry.setPlaceholderText("Search history…")
        self._set_help(
            self.history_search_entry,
            "Filter saved corrections by title or prompt text.",
            "Type castle to show matching history entries.",
        )
        self.history_search_entry.setClearButtonEnabled(True)
        self.history_search_entry.textChanged.connect(self._refresh_history_listbox)
        history_layout.addWidget(self.history_search_entry)
        self.history_listbox = QtHistoryList()
        self._set_help(
            self.history_listbox,
            "Select a previous correction; double-click to load it.",
            "Select a pinned comic prompt and click Load.",
        )
        self.history_listbox.itemDoubleClicked.connect(self.load_selected_history_prompt)
        history_layout.addWidget(self.history_listbox)
        history_buttons = QGridLayout()
        for index, (label, callback, description, example) in enumerate((
            ("Load", self.load_selected_history_prompt, "Load the selected correction and its settings.", "Resume a saved castle prompt."),
            ("Copy", self.copy_selected_history_prompt, "Copy the selected corrected prompt.", "Paste an older result into the selected generator."),
            ("Pin", self.toggle_selected_history_pin, "Pin or unpin the selected history entry.", "Keep a favorite setup at the top."),
            ("Rename", self.rename_selected_history_prompt, "Rename the selected history entry.", "Rename it Three-panel rescue."),
            ("Delete", self.delete_selected_history_prompt, "Delete the selected history entry.", "Remove an unwanted test correction."),
            ("Clear all", self.clear_prompt_history, "Delete every saved history entry.", "Clear history before starting a new project."),
        )):
            button = QtButton(label)
            self._set_help(button, description, example)
            button.clicked.connect(callback)
            history_buttons.addWidget(button, index // 2, index % 2)
        history_layout.addLayout(history_buttons)
        side_tabs.addTab(history_page, "History")
        references_page = QWidget()
        references_layout = QVBoxLayout(references_page)
        reference_workspace_row = QHBoxLayout()
        reference_workspace_row.addWidget(QLabel("Workspace"))
        self.reference_workspace_combo = QComboBox()
        self.reference_workspace_combo.addItems(
            ("Prompt Corrector", "Comic Story", "Meme Creator")
        )
        self.reference_workspace_combo.currentTextChanged.connect(
            self._on_reference_workspace_changed
        )
        reference_workspace_row.addWidget(self.reference_workspace_combo, 1)
        references_layout.addLayout(reference_workspace_row)
        self.reference_analysis_checkbox = QCheckBox(
            "Analyze these references in this workspace"
        )
        self._set_help(
            self.reference_analysis_checkbox,
            "Includes this workspace's isolated local references in its next model request.",
            "Enable it for Comic Story without affecting Prompt Corrector or Meme Creator.",
        )
        self.reference_analysis_checkbox.toggled.connect(
            self._set_current_reference_analysis
        )
        references_layout.addWidget(self.reference_analysis_checkbox)
        references_hint = QLabel(
            "References are isolated per workspace. Local images provide requested identity, "
            "material, or style facts; they never donate an unrelated scene, pose, camera, or story."
        )
        references_hint.setWordWrap(True)
        self._help(references_hint, "References")
        references_layout.addWidget(references_hint)
        reference_controls = QHBoxLayout()
        for label, callback, description, example in (
            ("Add images…", self.add_local_reference_images, "Choose local reference images to send for analysis.", "Add a PNG showing the desired costume."),
            ("Remove selected", self.remove_selected_reference_image, "Remove the selected local reference from this request.", "Remove the wrong lighting reference."),
            ("Clear local", self.clear_local_reference_images, "Remove all local reference images.", "Clear the set before a new project."),
        ):
            button = QtButton(label)
            self._set_help(button, description, example)
            button.clicked.connect(callback)
            reference_controls.addWidget(button)
        references_layout.addLayout(reference_controls)
        self.reference_preview_list = ReferencePreviewList()
        self._help(self.reference_preview_list, "References")
        self.reference_preview_list.setIconSize(QSize(112, 84))
        self.reference_preview_list.itemDoubleClicked.connect(self.open_reference_preview)
        self.reference_preview_list.filesDropped.connect(self.add_local_reference_paths)
        references_layout.addWidget(self.reference_preview_list)
        side_tabs.addTab(references_page, "References")
        workspace.setStretchFactor(0, 1)
        prompt_outer.addWidget(workspace, 1)
        self.mode_tabs.addTab(prompt_mode, "Prompt Corrector")
        self.mode_tabs.addTab(self._build_comic_story_page(), "Comic Story")
        self.mode_tabs.addTab(self._build_meme_creator_page(), "Meme Creator")
        self.mode_tabs.addTab(self._build_chat_page(), "Model Chat")
        self.workbench_widget = PromptWorkbench(self)
        self.mode_tabs.addTab(self.workbench_widget, "Workbench")
        self.mode_tabs.currentChanged.connect(self._on_workspace_changed)
        self._on_workspace_changed(self.mode_tabs.currentIndex())
        outer.addWidget(self.mode_tabs, 1)
        self.root.setCentralWidget(central)
        self.library_dock = QDockWidget("Activity, History, and References", self.root)
        self.library_dock.setObjectName("projectLibraryDock")
        self.library_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea
            | Qt.DockWidgetArea.BottomDockWidgetArea
        )
        self.library_dock.setWidget(side_tabs)
        self.root.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea,
            self.library_dock,
        )
        self.view_menu.addAction(self.library_dock.toggleViewAction())
        self._apply_known_tooltips(self.library_dock)
        self._apply_generator_target(self.generator_target_var.get())
        self._update_profile_summary()
        self._update_krea_recommendation()
        self._apply_known_tooltips(central)
        self._refresh_history_listbox()
        self._refresh_activity_text()
        self._on_reference_workspace_changed()
        self._refresh_local_reference_previews()
        self._refresh_chat_transcript()

    def _build_comic_story_page(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(8, 12, 8, 8)
        outer.setSpacing(10)

        intro = QLabel(
            "Construct one comic page panel by panel. The selected Generator and Workflow above apply to this workspace."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #9aa6bd;")
        self._set_help(
            intro,
            "Comic Story creates a structured multi-panel request for either Krea 2 or FLUX.2 Klein 9B.",
            "Choose four panels, describe one chronological beat per panel, then generate the final page prompt.",
        )
        outer.addWidget(intro)

        workspace = QSplitter(Qt.Orientation.Horizontal)
        builder_scroll = QScrollArea()
        builder_scroll.setWidgetResizable(True)
        builder = QWidget()
        builder_layout = QVBoxLayout(builder)
        builder_layout.setContentsMargins(4, 4, 8, 4)
        builder_layout.setSpacing(10)

        structure = QGroupBox("Page structure")
        structure_grid = QGridLayout(structure)
        structure_grid.addWidget(QLabel("Panels"), 0, 0)
        panel_count = self._bind_spin(self.comic_panel_count_var, 2, 12)
        self._set_help(
            panel_count,
            "Controls how many dedicated panel editors are visible and required.",
            "Set 6 to construct a six-panel page.",
        )
        structure_grid.addWidget(panel_count, 0, 1)
        structure_grid.addWidget(QLabel("Layout"), 0, 2)
        structure_grid.addWidget(
            self._help(
                self._bind_combo(
                    self.comic_layout_var,
                    COMIC_LAYOUT_PRESETS,
                ),
                "Comic layout",
            ),
            0,
            3,
        )
        structure_grid.addWidget(QLabel("Reading order"), 1, 0)
        structure_grid.addWidget(
            self._help(
                self._bind_combo(
                    self.comic_reading_order_var,
                    COMIC_READING_ORDER_PRESETS,
                ),
                "Comic reading order",
            ),
            1,
            1,
            1,
            3,
        )
        structure_grid.addWidget(QLabel("Aspect ratio"), 2, 0)
        structure_grid.addWidget(
            self._help(
                self._bind_combo(
                    self.comic_aspect_ratio_var,
                    COMIC_ASPECT_RATIO_PRESETS,
                ),
                "Comic page aspect ratio",
            ),
            2,
            1,
            1,
            3,
        )
        speech_bubbles = self._bind_check(
            "Comic speech bubbles",
            self.comic_speech_bubbles_var,
        )
        structure_grid.addWidget(speech_bubbles, 3, 0, 1, 4)
        self.comic_layout_preview_label = QLabel()
        self.comic_layout_preview_label.setWordWrap(True)
        self.comic_layout_preview_label.setStyleSheet("color: #9aa6bd;")
        self._set_help(
            self.comic_layout_preview_label,
            "Shows the exact panel arrangement that Auto grid will send to the model.",
            "For three panels, Auto uses two panels across the top and one full-width panel below.",
        )
        structure_grid.addWidget(self.comic_layout_preview_label, 4, 0, 1, 4)
        structure_grid.setColumnStretch(1, 1)
        structure_grid.setColumnStretch(3, 2)
        builder_layout.addWidget(structure)

        brief = QGroupBox("Story and continuity")
        brief_grid = QGridLayout(brief)
        brief_grid.addWidget(QLabel("Title"), 0, 0)
        title = self._bind_line(self.comic_title_var)
        title.setPlaceholderText("Optional working title")
        brief_grid.addWidget(self._help(title, "Comic title"), 0, 1)
        title_invent = QtButton("Invent")
        title_invent.clicked.connect(
            lambda: self.invent_comic_field("title")
        )
        self._set_help(
            title_invent,
            "Expand the entered title as a mandatory seed, or invent one when blank.",
            "Enter a rough title to refine, or leave it blank to derive one from the story.",
        )
        self.comic_invent_buttons.append(title_invent)
        brief_grid.addWidget(title_invent, 0, 2)
        brief_grid.addWidget(
            self._make_invent_recall_button("comic:title"),
            0,
            3,
        )
        brief_grid.addWidget(QLabel("Premise"), 1, 0)
        premise = self._bind_text(self.comic_premise_var, maximum_height=90)
        premise.setPlaceholderText("Who is involved, what happens, where it happens, and the intended outcome")
        premise_controls = QHBoxLayout()
        premise_controls.addWidget(self._help(premise, "Comic premise"), 1)
        premise_presets = QVBoxLayout()
        premise_presets.addWidget(
            self._make_narrative_preset_button("comic", "action")
        )
        premise_presets.addWidget(
            self._make_narrative_preset_button("comic", "emotion")
        )
        premise_controls.addLayout(premise_presets)
        brief_grid.addLayout(premise_controls, 1, 1)
        premise_invent = QtButton("Invent")
        premise_invent.clicked.connect(
            lambda: self.invent_comic_field("premise")
        )
        self._set_help(
            premise_invent,
            "Expand the entered premise as a mandatory seed, or invent one when blank.",
            "Enter a rough story idea to develop, or leave it blank to derive one from the other fields.",
        )
        self.comic_invent_buttons.append(premise_invent)
        brief_grid.addWidget(premise_invent, 1, 2)
        brief_grid.addWidget(
            self._make_invent_recall_button("comic:premise"),
            1,
            3,
        )
        brief_grid.addWidget(QLabel("Continuity anchors"), 2, 0)
        continuity = self._bind_text(self.comic_continuity_var, maximum_height=90)
        continuity.setPlaceholderText("Recurring character identity, wardrobe, props, environment, colors, and screen direction")
        brief_grid.addWidget(self._help(continuity, "Comic continuity anchors"), 2, 1)
        continuity_invent = QtButton("Invent")
        continuity_invent.clicked.connect(
            lambda: self.invent_comic_field("continuity")
        )
        self._set_help(
            continuity_invent,
            "Expand the entered continuity anchors as mandatory seeds, or invent them when blank.",
            "Enter rough recurring details to develop, or leave the field blank to create them.",
        )
        self.comic_invent_buttons.append(continuity_invent)
        brief_grid.addWidget(continuity_invent, 2, 2)
        brief_grid.addWidget(
            self._make_invent_recall_button("comic:continuity"),
            2,
            3,
        )
        brief_grid.addWidget(QLabel("Concepts to integrate"), 3, 0)
        comic_concepts = self._bind_line(self.comic_concepts_var)
        comic_concepts.setPlaceholderText(
            "Supporting visual concepts, comma-separated, for example: bioluminescent fungi, Stone Age astronomy"
        )
        concept_controls = QHBoxLayout()
        concept_controls.addWidget(
            self._help(comic_concepts, "Comic concepts to integrate"),
            1,
        )
        concept_controls.addWidget(
            self._make_concept_preset_button("comic")
        )
        brief_grid.addLayout(concept_controls, 3, 1)
        concepts_invent = QtButton("Invent")
        concepts_invent.clicked.connect(
            lambda: self.invent_comic_field("concepts")
        )
        self._set_help(
            concepts_invent,
            "Keep and expand the entered supporting concepts, or invent a concept list when blank.",
            "Enter concepts to preserve and enrich, or leave the field blank for a new compatible list.",
        )
        self.comic_invent_buttons.append(concepts_invent)
        brief_grid.addWidget(concepts_invent, 3, 2)
        brief_grid.addWidget(
            self._make_invent_recall_button("comic:concepts"),
            3,
            3,
        )
        brief_grid.addWidget(QLabel("Style direction"), 4, 0)
        visual = self._bind_line(self.comic_visual_direction_var)
        visual.setPlaceholderText("Optional art style, palette, lighting, camera, or lettering direction")
        visual_controls = QHBoxLayout()
        visual_controls.addWidget(
            self._help(visual, "Comic style direction"),
            1,
        )
        visual_controls.addWidget(
            self._make_visual_preset_button("comic")
        )
        brief_grid.addLayout(visual_controls, 4, 1)
        visual_invent = QtButton("Invent")
        visual_invent.clicked.connect(
            lambda: self.invent_comic_field("visual_direction")
        )
        self._set_help(
            visual_invent,
            "Expand the entered visual direction as a mandatory seed, or invent it when blank.",
            "Enter a rough art direction to develop, or leave it blank for a new treatment.",
        )
        self.comic_invent_buttons.append(visual_invent)
        brief_grid.addWidget(visual_invent, 4, 2)
        brief_grid.addWidget(
            self._make_invent_recall_button("comic:visual_direction"),
            4,
            3,
        )
        brief_grid.addWidget(QLabel("Dialogue direction"), 5, 0)
        dialogue_direction = self._bind_line(self.comic_dialogue_direction_var)
        dialogue_direction.setPlaceholderText(
            "How invented speech should sound, for example: short caveman grammar, simple words, no modern slang"
        )
        brief_grid.addWidget(
            self._help(dialogue_direction, "Comic dialogue direction"),
            5,
            1,
        )
        dialogue_invent = QtButton("Invent")
        dialogue_invent.clicked.connect(
            lambda: self.invent_comic_field("dialogue_direction")
        )
        self._set_help(
            dialogue_invent,
            "Expand the entered dialogue direction as a mandatory seed, or invent it when blank.",
            "Enter a rough speech rule to develop, or leave it blank for a new voice.",
        )
        self.comic_invent_buttons.append(dialogue_invent)
        brief_grid.addWidget(dialogue_invent, 5, 2)
        brief_grid.addWidget(
            self._make_invent_recall_button("comic:dialogue_direction"),
            5,
            3,
        )
        brief_grid.setColumnStretch(1, 1)
        builder_layout.addWidget(brief)

        panels_header = QHBoxLayout()
        panels_heading = QLabel("Panel beats")
        panels_heading.setStyleSheet("font-weight: 700; color: #ffffff;")
        panels_header.addWidget(panels_heading)
        panels_header.addStretch()
        self.comic_invent_all_button = QtButton("Invent all panels")
        self.comic_invent_all_button.clicked.connect(self.invent_all_comic_panels)
        self._set_help(
            self.comic_invent_all_button,
            "Expand every entered panel beat as a mandatory seed and invent only the blank panels.",
            "Enter any beats you already know; blank panels are created around them as one sequence.",
        )
        self.comic_invent_buttons.append(self.comic_invent_all_button)
        panels_header.addWidget(self.comic_invent_all_button)
        panels_header.addWidget(
            self._make_invent_recall_button(
                "comic:all_panels",
                label="Recall all",
                group=True,
            )
        )
        builder_layout.addLayout(panels_header)
        for index, variable in enumerate(self.comic_panel_vars, start=1):
            group = QGroupBox(f"Panel {index}")
            panel_layout = QVBoxLayout(group)
            panel_editor_row = QHBoxLayout()
            editor = self._bind_text(variable, maximum_height=115)
            editor.setPlaceholderText(
                "Describe this panel's subject, action or reaction, setting, framing, and exact quoted dialogue or caption."
            )
            self._set_help(
                editor,
                f"Panel {index} beat",
                f"Panel {index}: the same red-cloaked knight raises her shield and says \"Stay behind me!\"",
            )
            panel_editor_row.addWidget(editor, 1)
            panel_invent = QtButton("Invent")
            panel_invent.clicked.connect(
                lambda _checked=False, panel=index: self.invent_comic_field(
                    f"panel_{panel}"
                )
            )
            self._set_help(
                panel_invent,
                f"Expand panel {index}'s entered beat as a mandatory seed, or invent it when blank.",
                f"Enter a rough beat to develop, or leave panel {index} blank to use the surrounding story.",
            )
            self.comic_invent_buttons.append(panel_invent)
            self.comic_panel_invent_buttons.append(panel_invent)
            panel_editor_row.addWidget(panel_invent)
            panel_editor_row.addWidget(
                self._make_invent_recall_button(f"comic:panel_{index}")
            )
            panel_layout.addLayout(panel_editor_row)
            self.comic_panel_groups.append(group)
            self.comic_panel_editors.append(editor)
            builder_layout.addWidget(group)
        builder_layout.addStretch()
        builder_scroll.setWidget(builder)
        workspace.addWidget(builder_scroll)

        result_panel = QWidget()
        result_layout = QVBoxLayout(result_panel)
        result_layout.setContentsMargins(8, 4, 4, 4)
        result_group = QGroupBox("Comic page prompt")
        result_group_layout = QVBoxLayout(result_group)
        self.comic_result_text = QtTextEdit()
        self.comic_result_text.setPlainText(self.recovered_comic_result)
        self.comic_result_text.setPlaceholderText(
            "The generator-ready multi-panel comic prompt will appear here."
        )
        self._help(self.comic_result_text, "Comic result")
        result_group_layout.addWidget(self.comic_result_text)
        result_layout.addWidget(result_group, 1)

        buttons = QHBoxLayout()
        self.comic_generate_button = QtButton("Generate comic prompt")
        self.comic_generate_button.setObjectName("primaryButton")
        self.comic_generate_button.clicked.connect(self.correct_comic_story)
        buttons.addWidget(self.comic_generate_button)
        self.comic_stop_button = QtButton("Stop")
        self.comic_stop_button.clicked.connect(self.stop_current_request)
        self.comic_stop_button.setEnabled(False)
        buttons.addWidget(self.comic_stop_button)
        copy_button = QtButton("Copy result")
        copy_button.clicked.connect(self.copy_comic_result)
        buttons.addWidget(copy_button)
        clear_button = QtButton("Clear story")
        clear_button.clicked.connect(self.clear_comic_story)
        buttons.addWidget(clear_button)
        result_layout.addLayout(buttons)
        comic_status = QLabel(str(self.status_var.get()))
        comic_status.setObjectName("statusLabel")
        self._set_help(
            comic_status,
            "Shows the current comic generation or connection status.",
            "Look here for Done, Error, or Stopped.",
        )
        self.status_var.subscribe(lambda value: comic_status.setText(str(value)))
        result_layout.addWidget(comic_status)
        comic_progress = QProgressBar()
        comic_progress.setRange(0, 100)
        comic_progress.setValue(round(float(self.progress_var.get())))
        self.progress_var.subscribe(lambda value: comic_progress.setValue(round(float(value))))
        result_layout.addWidget(comic_progress)
        comic_progress_text = QLabel(str(self.progress_text_var.get()))
        self._set_help(
            comic_progress_text,
            "Names the current comic-generation workflow stage.",
            "It may show Waiting for LM Studio correction.",
        )
        self.progress_text_var.subscribe(lambda value: comic_progress_text.setText(str(value)))
        result_layout.addWidget(comic_progress_text)
        workspace.addWidget(result_panel)
        workspace.setStretchFactor(0, 3)
        workspace.setStretchFactor(1, 2)
        outer.addWidget(workspace, 1)

        self.comic_panel_count_var.subscribe(self._update_comic_panel_visibility)
        self.comic_layout_var.subscribe(
            lambda _value: self._update_comic_layout_preview()
        )
        self._update_comic_panel_visibility(self.comic_panel_count_var.get())
        return page

    def _update_comic_panel_visibility(self, count: object) -> None:
        try:
            visible_count = max(2, min(12, int(count)))
        except (TypeError, ValueError):
            visible_count = 4
        for index, group in enumerate(self.comic_panel_groups):
            group.setVisible(index < visible_count)
        self._update_comic_layout_preview()

    def _effective_comic_layout(self) -> str:
        return resolve_comic_layout(
            self.comic_layout_var.get(),
            int(self.comic_panel_count_var.get()),
        )

    def _update_comic_layout_preview(self) -> None:
        try:
            visible_count = max(
                2,
                min(12, int(self.comic_panel_count_var.get())),
            )
        except (TypeError, ValueError):
            visible_count = 4
        selected = str(self.comic_layout_var.get()).strip()
        fixed_grid_counts = {
            "2 x 2 grid": 4,
            "3 x 2 grid": 6,
        }
        if (
            selected in fixed_grid_counts
            and fixed_grid_counts[selected] != visible_count
        ):
            self.comic_layout_var.set("Auto grid")
            return
        if self.comic_layout_preview_label is not None:
            self.comic_layout_preview_label.setText(
                f"Effective layout: {resolve_comic_layout(selected, visible_count)}"
            )

    def _on_workspace_changed(self, index: int) -> None:
        camera_visible = index in {0, 1, 2}
        self.camera_label.setVisible(camera_visible)
        self.camera_combo.setVisible(camera_visible)
        if index == 0:
            self.content_format_var.set("Single Image")
        elif index == 1:
            self.content_format_var.set("Comic Story")
        elif index == 2:
            self.content_format_var.set("Meme")
        if index in {0, 1, 2} and self.reference_workspace_combo is not None:
            self.reference_workspace_combo.setCurrentIndex(index)

    def _current_workspace_key(self) -> str:
        if hasattr(self, "mode_tabs"):
            return {0: "prompt", 1: "comic", 2: "meme"}.get(
                self.mode_tabs.currentIndex(),
                "system",
            )
        return {
            "single image": "prompt",
            "comic story": "comic",
            "meme": "meme",
        }.get(str(self.content_format_var.get()).strip().casefold(), "system")

    def show_library_tab(self, name: str) -> None:
        if self.library_tabs is None or self.library_dock is None:
            return
        for index in range(self.library_tabs.count()):
            if self.library_tabs.tabText(index) == name:
                self.library_tabs.setCurrentIndex(index)
                break
        self.library_dock.show()
        self.library_dock.raise_()

    def _apply_meme_preset(self, preset_name: object) -> None:
        name = str(preset_name)
        preset = MEME_PRESETS.get(name, {})
        if not preset:
            return
        self.meme_tone_var.set(preset["tone"])
        self.meme_caption_style_var.set(preset["caption_style"])
        self.meme_aspect_ratio_var.set(preset["aspect_ratio"])
        self.meme_visual_direction_var.set(preset["visual_direction"])
        self.visual_preset_selections["meme"] = []
        self.status_var.set(f"Applied meme preset: {name}")

    def _build_meme_creator_page(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(8, 12, 8, 8)
        outer.setSpacing(10)

        intro = QLabel(
            "Describe what you are responding to and let the model invent a tailored "
            "meme, or provide your own scene and exact captions. The selected Generator "
            "and Workflow above apply to this workspace."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #9aa6bd;")
        self._help(intro, "Meme Creator")
        outer.addWidget(intro)

        workspace = QSplitter(Qt.Orientation.Horizontal)
        builder_scroll = QScrollArea()
        builder_scroll.setWidgetResizable(True)
        builder = QWidget()
        builder_layout = QVBoxLayout(builder)
        builder_layout.setContentsMargins(4, 4, 8, 4)
        builder_layout.setSpacing(10)

        setup = QGroupBox("Meme setup")
        setup_grid = QGridLayout(setup)
        setup_grid.addWidget(QLabel("Preset"), 0, 0)
        preset = self._bind_combo(self.meme_preset_var, tuple(MEME_PRESETS))
        setup_grid.addWidget(self._help(preset, "Meme preset"), 0, 1)
        setup_grid.addWidget(QLabel("Humor tone"), 1, 0)
        tone = self._bind_combo(self.meme_tone_var, MEME_TONES)
        setup_grid.addWidget(self._help(tone, "Humor tone"), 1, 1)
        setup_grid.addWidget(QLabel("Temperature"), 2, 0)
        self.meme_temperature_spin = self._bind_double_spin(
            self.meme_temperature_var
        )
        setup_grid.addWidget(
            self._help(self.meme_temperature_spin, "Meme temperature"),
            2,
            1,
        )
        setup_grid.addWidget(QLabel("Aspect ratio"), 3, 0)
        aspect_ratio = self._bind_combo(
            self.meme_aspect_ratio_var,
            MEME_ASPECT_RATIOS,
        )
        setup_grid.addWidget(self._help(aspect_ratio, "Meme aspect ratio"), 3, 1)
        setup_grid.addWidget(QLabel("Caption style"), 4, 0)
        caption_style = self._bind_combo(
            self.meme_caption_style_var,
            MEME_CAPTION_STYLES,
        )
        setup_grid.addWidget(self._help(caption_style, "Caption style"), 4, 1)
        setup_grid.setColumnStretch(1, 1)
        builder_layout.addWidget(setup)

        response = QGroupBox("Creative response")
        response_grid = QGridLayout(response)
        response_grid.addWidget(QLabel("Situation to respond to"), 0, 0)
        response_context = self._bind_text(
            self.meme_response_context_var,
            maximum_height=115,
        )
        response_context.setPlaceholderText(
            "Paste or summarize what happened, what someone said, or the post/message "
            "you want to answer with a meme."
        )
        response_grid.addWidget(
            self._help(response_context, "Situation to respond to"),
            0,
            1,
        )
        self.meme_response_context_button = QtButton("Invent")
        self.meme_response_context_button.clicked.connect(
            lambda: self.invent_meme_field("response_context")
        )
        self.meme_invent_buttons.append(self.meme_response_context_button)
        response_grid.addWidget(
            self._help(self.meme_response_context_button, "Invent meme situation"),
            0,
            2,
        )
        response_grid.addWidget(
            self._make_invent_recall_button("meme:response_context"),
            0,
            3,
        )
        response_grid.addWidget(QLabel("Desired response"), 1, 0)
        response_goal = self._bind_line(self.meme_response_goal_var)
        response_goal.setPlaceholderText(
            "Optional: what the meme should communicate, such as playful disagreement"
        )
        response_grid.addWidget(
            self._help(response_goal, "Desired response"),
            1,
            1,
        )
        self.meme_response_goal_button = QtButton("Invent")
        self.meme_response_goal_button.clicked.connect(
            lambda: self.invent_meme_field("response_goal")
        )
        self.meme_invent_buttons.append(self.meme_response_goal_button)
        response_grid.addWidget(
            self._help(self.meme_response_goal_button, "Invent meme response"),
            1,
            2,
        )
        response_grid.addWidget(
            self._make_invent_recall_button("meme:response_goal"),
            1,
            3,
        )
        response_grid.setColumnStretch(1, 1)
        builder_layout.addWidget(response)

        content = QGroupBox("Meme content")
        content_grid = QGridLayout(content)
        content_grid.addWidget(QLabel("Scene"), 0, 0)
        scene = self._bind_text(self.meme_scene_var, maximum_height=150)
        scene.setPlaceholderText(
            "Optional when a response situation is supplied; otherwise describe the "
            "subject, expression, action, setting, and visual joke."
        )
        scene_controls = QHBoxLayout()
        scene_controls.addWidget(self._help(scene, "Meme scene"), 1)
        scene_presets = QVBoxLayout()
        scene_presets.addWidget(
            self._make_narrative_preset_button("meme", "action")
        )
        scene_presets.addWidget(
            self._make_narrative_preset_button("meme", "emotion")
        )
        scene_controls.addLayout(scene_presets)
        content_grid.addLayout(scene_controls, 0, 1)
        self.meme_scene_button = QtButton("Invent")
        self.meme_scene_button.clicked.connect(
            lambda: self.invent_meme_field("scene")
        )
        self.meme_invent_buttons.append(self.meme_scene_button)
        content_grid.addWidget(
            self._help(self.meme_scene_button, "Invent meme scene"),
            0,
            2,
        )
        content_grid.addWidget(
            self._make_invent_recall_button("meme:scene"),
            0,
            3,
        )
        content_grid.addWidget(QLabel("Focus"), 1, 0)
        focus = self._bind_line(self.meme_focus_var)
        focus.setPlaceholderText(
            "Optional: the subject, expression, action, or joke detail to emphasize most"
        )
        content_grid.addWidget(self._help(focus, "Meme focus"), 1, 1)
        self.meme_focus_button = QtButton("Invent")
        self.meme_focus_button.clicked.connect(
            lambda: self.invent_meme_field("focus")
        )
        self.meme_invent_buttons.append(self.meme_focus_button)
        content_grid.addWidget(
            self._help(self.meme_focus_button, "Invent meme focus"),
            1,
            2,
        )
        content_grid.addWidget(
            self._make_invent_recall_button("meme:focus"),
            1,
            3,
        )
        content_grid.addWidget(QLabel("Top text"), 2, 0)
        top_text = self._bind_line(self.meme_top_text_var)
        top_text.setPlaceholderText(
            "Optional exact words; leave both captions blank for model-written text"
        )
        self.meme_top_caption_button = QtButton("Invent top")
        self.meme_top_caption_button.clicked.connect(
            lambda: self.invent_meme_caption("top")
        )
        self.meme_invent_buttons.append(self.meme_top_caption_button)
        content_grid.addWidget(
            self._help(self.meme_top_caption_button, "Invent top caption"),
            2,
            2,
        )
        content_grid.addWidget(
            self._make_invent_recall_button("meme:top"),
            2,
            3,
        )
        content_grid.addWidget(self._help(top_text, "Top text"), 2, 1)
        content_grid.addWidget(QLabel("Bottom text"), 3, 0)
        bottom_text = self._bind_line(self.meme_bottom_text_var)
        bottom_text.setPlaceholderText(
            "Optional exact words; leave both captions blank for model-written text"
        )
        content_grid.addWidget(self._help(bottom_text, "Bottom text"), 3, 1)
        self.meme_bottom_caption_button = QtButton("Invent bottom")
        self.meme_bottom_caption_button.clicked.connect(
            lambda: self.invent_meme_caption("bottom")
        )
        self.meme_invent_buttons.append(self.meme_bottom_caption_button)
        content_grid.addWidget(
            self._help(self.meme_bottom_caption_button, "Invent bottom caption"),
            3,
            2,
        )
        content_grid.addWidget(
            self._make_invent_recall_button("meme:bottom"),
            3,
            3,
        )
        content_grid.addWidget(QLabel("Visual direction"), 4, 0)
        visual_direction = self._bind_text(
            self.meme_visual_direction_var,
            maximum_height=100,
        )
        visual_direction.setPlaceholderText(
            "Optional style, camera, lighting, palette, or mood for the underlying image"
        )
        visual_direction_controls = QHBoxLayout()
        visual_direction_controls.addWidget(
            self._help(visual_direction, "Meme visual direction"),
            1,
        )
        visual_direction_controls.addWidget(
            self._make_visual_preset_button("meme")
        )
        content_grid.addLayout(visual_direction_controls, 4, 1)
        self.meme_visual_direction_button = QtButton("Invent")
        self.meme_visual_direction_button.clicked.connect(
            lambda: self.invent_meme_field("visual_direction")
        )
        self.meme_invent_buttons.append(self.meme_visual_direction_button)
        content_grid.addWidget(
            self._help(
                self.meme_visual_direction_button,
                "Invent meme visual direction",
            ),
            4,
            2,
        )
        content_grid.addWidget(
            self._make_invent_recall_button("meme:visual_direction"),
            4,
            3,
        )
        content_grid.setColumnStretch(1, 1)
        builder_layout.addWidget(content)
        builder_layout.addStretch()
        builder_scroll.setWidget(builder)
        workspace.addWidget(builder_scroll)

        result_panel = QWidget()
        result_layout = QVBoxLayout(result_panel)
        result_layout.setContentsMargins(8, 4, 4, 4)
        result_group = QGroupBox("Meme prompt")
        result_group_layout = QVBoxLayout(result_group)
        self.meme_result_text = QtTextEdit()
        self.meme_result_text.setPlainText(self.recovered_meme_result)
        self.meme_result_text.setPlaceholderText(
            "The generator-ready meme prompt will appear here."
        )
        self._help(self.meme_result_text, "Meme prompt")
        result_group_layout.addWidget(self.meme_result_text)
        result_layout.addWidget(result_group, 1)

        buttons = QHBoxLayout()
        self.meme_generate_button = QtButton("Generate meme prompt")
        self.meme_generate_button.setObjectName("primaryButton")
        self.meme_generate_button.clicked.connect(self.correct_meme)
        buttons.addWidget(self.meme_generate_button)
        self.meme_stop_button = QtButton("Stop")
        self.meme_stop_button.clicked.connect(self.stop_current_request)
        self.meme_stop_button.setEnabled(False)
        buttons.addWidget(self.meme_stop_button)
        copy_button = QtButton("Copy result")
        copy_button.clicked.connect(self.copy_meme_result)
        buttons.addWidget(copy_button)
        clear_button = QtButton("Clear meme")
        clear_button.clicked.connect(self.clear_meme)
        buttons.addWidget(clear_button)
        result_layout.addLayout(buttons)

        meme_status = QLabel(str(self.status_var.get()))
        meme_status.setObjectName("statusLabel")
        self._set_help(
            meme_status,
            "Shows the current meme generation or connection status.",
            "Look here for Done, Error, or Stopped.",
        )
        self.status_var.subscribe(lambda value: meme_status.setText(str(value)))
        result_layout.addWidget(meme_status)
        meme_progress = QProgressBar()
        meme_progress.setRange(0, 100)
        meme_progress.setValue(round(float(self.progress_var.get())))
        self.progress_var.subscribe(
            lambda value: meme_progress.setValue(round(float(value)))
        )
        result_layout.addWidget(meme_progress)
        meme_progress_text = QLabel(str(self.progress_text_var.get()))
        self._set_help(
            meme_progress_text,
            "Names the current meme-generation workflow stage.",
            "It may show Waiting for LM Studio correction.",
        )
        self.progress_text_var.subscribe(
            lambda value: meme_progress_text.setText(str(value))
        )
        result_layout.addWidget(meme_progress_text)

        workspace.addWidget(result_panel)
        workspace.setStretchFactor(0, 3)
        workspace.setStretchFactor(1, 2)
        outer.addWidget(workspace, 1)
        return page

    def _build_chat_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 12, 8, 8)
        layout.setSpacing(12)

        settings_group = QGroupBox("Chat settings")
        settings_grid = QGridLayout(settings_group)
        settings_grid.addWidget(QLabel("System instruction"), 0, 0)
        system_prompt = self._bind_line(self.chat_system_prompt_var)
        system_prompt.setPlaceholderText("Optional instruction that defines how the model should respond")
        self._help(system_prompt, "System instruction")
        settings_grid.addWidget(system_prompt, 0, 1, 1, 5)
        settings_grid.addWidget(QLabel("Temperature"), 1, 0)
        settings_grid.addWidget(
            self._help(self._bind_double_spin(self.chat_temperature_var), "Temperature"),
            1,
            1,
        )
        settings_grid.addWidget(QLabel("Max response tokens"), 1, 2)
        settings_grid.addWidget(
            self._help(
                self._bind_spin(self.chat_max_tokens_var, 1, CONTEXT_TOKEN_MAX, 256),
                "Max response tokens",
            ),
            1,
            3,
        )
        model_hint = QLabel("Uses the LM Studio model and connection selected above.")
        model_hint.setStyleSheet("color: #8993a5;")
        self._set_help(
            model_hint,
            "Chat uses the model, host, port, and timeout selected above.",
            "Change Model above before beginning a specialized chat.",
        )
        settings_grid.addWidget(model_hint, 1, 4, 1, 2)
        settings_grid.setColumnStretch(1, 2)
        settings_grid.setColumnStretch(4, 3)
        layout.addWidget(settings_group)

        transcript_group = QGroupBox("Conversation")
        transcript_layout = QVBoxLayout(transcript_group)
        self.chat_transcript = QTextEdit()
        self.chat_transcript.setReadOnly(True)
        self.chat_transcript.setPlaceholderText("Your conversation with the model will appear here.")
        self._help(self.chat_transcript, "Conversation")
        transcript_layout.addWidget(self.chat_transcript)
        layout.addWidget(transcript_group, 1)

        composer_group = QGroupBox("Message")
        composer_layout = QVBoxLayout(composer_group)
        self.chat_input = ChatInputEdit()
        self.chat_input.setPlaceholderText("Talk directly to the model…  Ctrl+Shift+Enter sends")
        self._help(self.chat_input, "Message")
        self.chat_input.setMaximumHeight(140)
        self.chat_input.send_requested.connect(self.send_chat_message)
        composer_layout.addWidget(self.chat_input)
        controls = QHBoxLayout()
        self.chat_send_button = QtButton("Send")
        self.chat_send_button.setObjectName("primaryButton")
        self.chat_send_button.clicked.connect(self.send_chat_message)
        controls.addWidget(self.chat_send_button)
        self.chat_stop_button = QtButton("Stop")
        self.chat_stop_button.clicked.connect(self.stop_current_request)
        self.chat_stop_button.setEnabled(False)
        controls.addWidget(self.chat_stop_button)
        new_chat_button = QtButton("New chat")
        new_chat_button.clicked.connect(lambda: self.clear_chat())
        controls.addWidget(new_chat_button)
        copy_button = QtButton("Copy last response")
        copy_button.clicked.connect(self.copy_last_chat_response)
        controls.addWidget(copy_button)
        controls.addStretch()
        composer_layout.addLayout(controls)
        layout.addWidget(composer_group)
        return page

    def _refresh_chat_transcript(self) -> None:
        if self.chat_transcript is None:
            return
        visible_messages = list(self.chat_messages)
        if self.chat_stream_text:
            visible_messages.append({"role": "assistant", "content": self.chat_stream_text})
        if not visible_messages:
            self.chat_transcript.clear()
            return
        blocks: list[str] = []
        for message in visible_messages:
            is_user = message["role"] == "user"
            label = "You" if is_user else "Model"
            background = "#252d3c" if is_user else "#1c2330"
            border = "#6d5dfc" if is_user else "#3c485f"
            content = html.escape(message["content"]).replace("\n", "<br>")
            blocks.append(
                f'<div style="margin:8px 2px;padding:10px 12px;background:{background};'
                f'border-left:3px solid {border};border-radius:5px;">'
                f'<div style="color:#a99fff;font-weight:700;margin-bottom:5px;">{label}</div>'
                f'<div style="color:#e7eaf0;white-space:pre-wrap;">{content}</div></div>'
            )
        self.chat_transcript.setHtml("".join(blocks))
        self.chat_transcript.moveCursor(QTextCursor.MoveOperation.End)

    def send_chat_message(self) -> None:
        if self.request_in_progress:
            self.status_var.set("Another model request is already running")
            return
        if self.chat_input is None:
            return
        content = self.chat_input.toPlainText().strip()
        if not content:
            messagebox.showwarning("Missing message", "Type a message for the model first.")
            return
        model = self.model_var.get().strip()
        if not model:
            messagebox.showwarning("Missing model", "Select or enter an LM Studio model first.")
            return

        self.chat_messages.append({"role": "user", "content": content})
        self.chat_messages = self.chat_messages[-100:]
        self.chat_input.clear()
        self.chat_stream_text = ""
        self._refresh_chat_transcript()
        self._save_settings()

        self.active_request_id += 1
        request_id = self.active_request_id
        self.active_request_workspace = "system"
        self.cancel_event.clear()
        self.request_in_progress = True
        self._set_request_controls(True)
        self.status_var.set("Model is responding...")

        messages: list[dict[str, object]] = []
        system_prompt = self.chat_system_prompt_var.get().strip()
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(dict(message) for message in self.chat_messages)
        thread = threading.Thread(
            target=self._chat_worker,
            args=(request_id, self._current_base_url(), model, messages),
            daemon=True,
        )
        thread.start()

    def _chat_worker(
        self,
        request_id: int,
        base_url: str,
        model: str,
        messages: list[dict[str, object]],
    ) -> None:
        try:
            response = chat_completion(
                base_url=base_url,
                model=model,
                messages=messages,
                temperature=self._chat_temperature(),
                max_tokens=self._chat_max_tokens(),
                timeout=self._lm_timeout_seconds(),
                api_key=os.getenv("LM_STUDIO_API_KEY", "lm-studio"),
                cancel_check=lambda: self._raise_if_cancelled(request_id),
                chunk_callback=lambda chunk: self._after_threadsafe(
                    0, self._show_chat_chunk, request_id, chunk
                ),
            )
            self._raise_if_cancelled(request_id)
        except CorrectionCancelled:
            self._after_threadsafe(0, self._show_cancelled, request_id)
            return
        except Exception as exc:
            if self._request_cancelled(request_id):
                self._after_threadsafe(0, self._show_cancelled, request_id)
            else:
                self._after_threadsafe(
                    0,
                    self._show_error,
                    str(exc),
                    "system",
                    "Model chat",
                )
            return
        self._after_threadsafe(0, self._show_chat_response, request_id, response)

    def _show_chat_chunk(self, request_id: int, chunk: str) -> None:
        if request_id != self.active_request_id or self.cancel_event.is_set():
            return
        self.chat_stream_text += chunk
        self._refresh_chat_transcript()

    def _show_chat_response(self, request_id: int, response: str) -> None:
        if request_id != self.active_request_id or self.cancel_event.is_set():
            return
        self.chat_stream_text = ""
        self.chat_messages.append({"role": "assistant", "content": response.strip()})
        self.chat_messages = self.chat_messages[-100:]
        self._refresh_chat_transcript()
        self.request_in_progress = False
        self._set_request_controls(False)
        self.status_var.set("Chat response complete")
        self._save_settings()
        if self.chat_input is not None:
            self.chat_input.setFocus()

    def clear_chat(self, confirm: bool = True) -> None:
        if self.request_in_progress:
            self.status_var.set("Stop the current request before starting a new chat")
            return
        if self.chat_messages and confirm and not messagebox.askyesno(
            "New chat", "Clear the current model conversation?"
        ):
            return
        self.chat_messages = []
        self.chat_stream_text = ""
        self._refresh_chat_transcript()
        self._save_settings()
        self.status_var.set("New chat started")

    def copy_last_chat_response(self) -> None:
        response = next(
            (
                message["content"]
                for message in reversed(self.chat_messages)
                if message["role"] == "assistant"
            ),
            "",
        )
        if response:
            QApplication.clipboard().setText(response)
            self.status_var.set("Copied last model response")

    def _add_slider(self, layout: QGridLayout, label: str, variable: Value, row: int) -> None:
        layout.addWidget(QLabel(label), row, 0)
        slider = QSlider(Qt.Orientation.Horizontal)
        self._help(slider, label)
        slider.setRange(-100, 100)
        slider.setValue(int(variable.get()))
        spin = self._bind_spin(variable, -100, 100)
        self._help(spin, label)
        slider.valueChanged.connect(variable.set)
        variable.subscribe(lambda value, w=slider: w.setValue(int(value)))
        controls = QHBoxLayout()
        controls.addWidget(slider, 1)
        controls.addWidget(spin)
        layout.addLayout(controls, row, 1, 1, 3)

    def test_lm_studio_connection(self) -> None:
        base_url = self._current_base_url()
        self._save_settings()
        self.active_activity_workspace = "system"
        self._log_activity("Started LM Studio connection test.", "system")
        self.status_var.set("Testing LM Studio...")
        self._log_activity(f"Testing LM Studio connection at {base_url}...")
        self.test_connection_button.configure(state="disabled")

        thread = threading.Thread(
            target=self._test_lm_studio_worker,
            args=(base_url,),
            daemon=True,
        )
        thread.start()

    def _test_lm_studio_worker(self, base_url: str) -> None:
        try:
            models = list_lm_studio_models(
                base_url=base_url,
                timeout=8.0,
                api_key=os.getenv("LM_STUDIO_API_KEY", "lm-studio"),
            )
        except Exception as exc:
            self._log_activity_threadsafe(f"Connection failed: {exc}")
            self._after_threadsafe(0, self._finish_connection_test, False, str(exc))
            return

        if models:
            self._log_activity_threadsafe("LM Studio reachable. Models:")
            for model in models:
                self._log_activity_threadsafe(f"- {model}")
            self._after_threadsafe(0, self._update_available_models, models)
        else:
            self._log_activity_threadsafe("LM Studio reachable, but no language models were found.")
        self._after_threadsafe(0, self._finish_connection_test, True, "")

    def _refresh_model_list_in_background(self) -> None:
        thread = threading.Thread(
            target=self._refresh_model_list_worker,
            args=(self._current_base_url(),),
            daemon=True,
        )
        thread.start()

    def _refresh_model_list_worker(self, base_url: str) -> None:
        try:
            models = list_lm_studio_models(
                base_url=base_url,
                timeout=5.0,
                api_key=os.getenv("LM_STUDIO_API_KEY", "lm-studio"),
            )
        except Exception:
            return
        self._after_threadsafe(0, self._update_available_models, models)

    def _finish_connection_test(self, ok: bool, error: str) -> None:
        self.test_connection_button.configure(state="normal")
        self.status_var.set("LM Studio reachable" if ok else "LM Studio connection failed")
        if error:
            messagebox.showerror("LM Studio connection", error)

    def _single_image_field_context(self) -> dict[str, object]:
        draft = (
            self.draft_text.toPlainText().strip()
            if hasattr(self, "draft_text")
            else self.recovered_draft.strip()
        )
        return {
            "draft": draft,
            "concepts": self.concepts_var.get().strip(),
            "concept_mix": self.concept_mix_var.get().strip(),
            "visual_direction": self.visual_direction_var.get().strip(),
            "goal_headline": self.goal_headline_var.get().strip(),
            "focus": self.focus_var.get().strip(),
            "story_elements": self.story_elements_var.get().strip(),
            "weighted_terms": self.weighted_terms_var.get().strip(),
            "model_instructions": self.model_instructions_var.get().strip(),
            "generation_feedback": self.generation_feedback_var.get().strip(),
            "mode": self.mode_var.get().strip(),
            "generator_target": self.generator_target_var.get().strip(),
            "camera_direction": self._camera_direction(),
            "artistic_detail_freedom": bool(
                self.artistic_detail_freedom_var.get()
            ),
        }

    def _single_image_invent_research_options(self) -> dict[str, object]:
        reference_image_analysis = bool(self.reference_images_var.get())
        effective_concepts, _, _, _ = self._effective_mix_inputs()
        return {
            "live_research": bool(self.live_research_var.get()),
            "search_engine": self.search_engine_var.get().strip(),
            "concepts": effective_concepts,
            "reference_image_analysis": reference_image_analysis,
            "reference_image_source": self.reference_image_source_var.get().strip(),
            "local_reference_candidates": (
                self._local_reference_candidates("prompt")
                if reference_image_analysis
                else []
            ),
            "enhance_actions": bool(self.enhance_actions_var.get()),
        }

    def _make_invent_recall_button(
        self,
        key: str,
        *,
        label: str = "Recall",
        group: bool = False,
    ) -> QtButton:
        """Build a one-step restore button for a model-invented input field."""

        button = QtButton(label)
        button.setEnabled(False)
        if group:
            button.clicked.connect(
                lambda _checked=False, name=key: self.recall_invented_group(name)
            )
            description = (
                "Restore every panel beat to the values they had immediately "
                "before the last successful Invent all panels request."
            )
            example = "Recover all panel beats after trying a generated sequence."
        else:
            button.clicked.connect(
                lambda _checked=False, name=key: self.recall_invented_input(name)
            )
            description = (
                "Restore this field to the value it had immediately before its "
                "last successful Invent request."
            )
            example = "Recover your rough text after trying an invented expansion."
        self._set_help(button, description, example)
        self.invent_recall_buttons[key] = button
        return button

    def _invent_input_value(self, key: str) -> str:
        namespace, _, field = key.partition(":")
        if namespace == "single":
            if field == "draft":
                return self.draft_text.toPlainText()
            targets = {
                "concepts": self.concepts_var,
                "concept_mix": self.concept_mix_var,
                "visual_direction": self.visual_direction_var,
                "goal_headline": self.goal_headline_var,
                "focus": self.focus_var,
                "story_elements": self.story_elements_var,
                "weighted_terms": self.weighted_terms_var,
                "model_instructions": self.model_instructions_var,
                "generation_feedback": self.generation_feedback_var,
            }
        elif namespace == "comic":
            panel_match = re.fullmatch(r"panel_(\d+)", field)
            if panel_match:
                panel_index = int(panel_match.group(1)) - 1
                if 0 <= panel_index < len(self.comic_panel_vars):
                    return str(self.comic_panel_vars[panel_index].get())
                return ""
            targets = {
                "title": self.comic_title_var,
                "premise": self.comic_premise_var,
                "continuity": self.comic_continuity_var,
                "concepts": self.comic_concepts_var,
                "visual_direction": self.comic_visual_direction_var,
                "dialogue_direction": self.comic_dialogue_direction_var,
            }
        elif namespace == "meme":
            targets = {
                "response_context": self.meme_response_context_var,
                "response_goal": self.meme_response_goal_var,
                "scene": self.meme_scene_var,
                "focus": self.meme_focus_var,
                "top": self.meme_top_text_var,
                "bottom": self.meme_bottom_text_var,
                "visual_direction": self.meme_visual_direction_var,
            }
        else:
            return ""
        target = targets.get(field)
        return str(target.get()) if target is not None else ""

    def _set_invent_input_value(self, key: str, value: str) -> bool:
        namespace, _, field = key.partition(":")
        if namespace == "single":
            if field == "draft":
                self.draft_text.setPlainText(value)
                return True
            targets = {
                "concepts": self.concepts_var,
                "concept_mix": self.concept_mix_var,
                "visual_direction": self.visual_direction_var,
                "goal_headline": self.goal_headline_var,
                "focus": self.focus_var,
                "story_elements": self.story_elements_var,
                "weighted_terms": self.weighted_terms_var,
                "model_instructions": self.model_instructions_var,
                "generation_feedback": self.generation_feedback_var,
            }
        elif namespace == "comic":
            panel_match = re.fullmatch(r"panel_(\d+)", field)
            if panel_match:
                panel_index = int(panel_match.group(1)) - 1
                if 0 <= panel_index < len(self.comic_panel_vars):
                    self.comic_panel_vars[panel_index].set(value)
                    return True
                return False
            targets = {
                "title": self.comic_title_var,
                "premise": self.comic_premise_var,
                "continuity": self.comic_continuity_var,
                "concepts": self.comic_concepts_var,
                "visual_direction": self.comic_visual_direction_var,
                "dialogue_direction": self.comic_dialogue_direction_var,
            }
        elif namespace == "meme":
            targets = {
                "response_context": self.meme_response_context_var,
                "response_goal": self.meme_response_goal_var,
                "scene": self.meme_scene_var,
                "focus": self.meme_focus_var,
                "top": self.meme_top_text_var,
                "bottom": self.meme_bottom_text_var,
                "visual_direction": self.meme_visual_direction_var,
            }
        else:
            return False
        target = targets.get(field)
        if target is None:
            return False
        target.set(value)
        return True

    def _stage_invent_recall(
        self,
        request_id: int,
        keys: list[str],
        *,
        group: str = "",
    ) -> None:
        self.pending_invent_recall[request_id] = {
            key: self._invent_input_value(key) for key in keys
        }
        if group:
            self.pending_invent_recall_groups[request_id] = group

    def _discard_pending_invent_recall(self, request_id: int) -> None:
        self.pending_invent_recall.pop(request_id, None)
        self.pending_invent_recall_groups.pop(request_id, None)

    def _commit_invent_recall(self, request_id: int) -> None:
        values = self.pending_invent_recall.pop(request_id, {})
        group = self.pending_invent_recall_groups.pop(request_id, "")
        if not values:
            return
        self.invent_recall_values.update(values)
        if group:
            self.invent_recall_groups[group] = dict(values)
        self._refresh_invent_recall_buttons()

    def _refresh_invent_recall_buttons(self) -> None:
        for key, button in self.invent_recall_buttons.items():
            available = (
                key in self.invent_recall_groups
                if key.endswith(":all_panels")
                else key in self.invent_recall_values
            )
            button.setEnabled(bool(available) and not self.request_in_progress)

    def recall_invented_input(self, key: str) -> None:
        if self.request_in_progress:
            self.status_var.set("Stop the current request before recalling input")
            return
        if key not in self.invent_recall_values:
            return
        value = self.invent_recall_values.pop(key)
        if not self._set_invent_input_value(key, value):
            self.invent_recall_values[key] = value
            return
        self._refresh_invent_recall_buttons()
        self.status_var.set("Recalled input from before Invent")
        self._save_settings()

    def recall_invented_group(self, key: str) -> None:
        if self.request_in_progress:
            self.status_var.set("Stop the current request before recalling input")
            return
        values = self.invent_recall_groups.pop(key, None)
        if values is None:
            return
        restored = 0
        for field_key, value in values.items():
            if self._set_invent_input_value(field_key, value):
                restored += 1
                self.invent_recall_values.pop(field_key, None)
        self._refresh_invent_recall_buttons()
        self.status_var.set(f"Recalled {restored} panel inputs from before Invent all")
        self._save_settings()

    def _show_invent_error(self, request_id: int, error: str) -> None:
        self._discard_pending_invent_recall(request_id)
        self._show_error(
            error,
            workspace=self.active_request_workspace,
            stage="Invent field",
        )

    def _normalize_or_repair_invent(
        self,
        *,
        request_id: int,
        base_url: str,
        model: str,
        workspace: str,
        field: str,
        response: str,
        seed_value: str,
        temperature: float,
        seed: int | None,
    ) -> str:
        """Apply the central Invent gate and make at most one repair request."""

        canonical = normalize_and_validate_invent(
            workspace,
            field,
            response,
            seed_value=seed_value,
        )
        if canonical:
            return canonical
        repair_response = chat_completion(
            base_url=base_url,
            model=model,
            messages=build_invent_field_repair_messages(
                workspace=workspace,
                field=field,
                candidate=response,
                issues=["The value failed its field schema or semantic contract."],
                seed_value=seed_value,
            ),
            temperature=min(0.2, max(0.0, float(temperature))),
            max_tokens=192,
            timeout=self._lm_timeout_seconds(),
            api_key=os.getenv("LM_STUDIO_API_KEY", "lm-studio"),
            seed=(None if seed is None else (int(seed) + 1) % 2_147_483_648),
            ttl=CREATIVE_SESSION_TTL_SECONDS,
            cancel_check=lambda: self._raise_if_cancelled(request_id),
        )
        self._raise_if_cancelled(request_id)
        return normalize_and_validate_invent(
            workspace,
            field,
            repair_response,
            seed_value=seed_value,
        )

    def invent_single_image_field(self, field: str) -> None:
        normalized_field = str(field).strip().casefold()
        field_labels = {
            "draft": "prompt",
            "concepts": "concepts",
            "concept_mix": "concept and style mix",
            "visual_direction": "visual direction",
            "goal_headline": "goal headline",
            "focus": "focus",
            "story_elements": "story beat",
            "weighted_terms": "weighted words",
            "model_instructions": "model instructions",
            "generation_feedback": "generation feedback",
        }
        if normalized_field not in field_labels:
            return
        if self.request_in_progress:
            self.status_var.set("Another model request is already running")
            return
        model = self.model_var.get().strip()
        if not model:
            messagebox.showwarning(
                "Missing model",
                "Select or enter an LM Studio model first.",
            )
            return

        self.active_request_id += 1
        request_id = self.active_request_id
        self.active_request_workspace = "prompt"
        self.cancel_event.clear()
        self.request_in_progress = True
        self._stage_invent_recall(
            request_id,
            [f"single:{normalized_field}"],
        )
        self._set_request_controls(True)
        label = field_labels[normalized_field]
        self.status_var.set(f"Inventing Single Image {label}...")
        self._save_settings()
        thread = threading.Thread(
            target=self._invent_single_image_field_worker,
            args=(
                request_id,
                self._current_base_url(),
                model,
                normalized_field,
                self._single_image_field_context(),
                self._single_image_invent_research_options(),
                self._sampling_seed(),
            ),
            daemon=True,
        )
        thread.start()

    def _collect_single_image_invent_research(
        self,
        *,
        request_id: int,
        base_url: str,
        model: str,
        context: dict[str, object],
        options: dict[str, object],
    ) -> dict[str, object]:
        """Collect the research selected for a user-seeded Invent prompt pass."""

        draft = str(context.get("draft", "")).strip()
        concepts = str(
            options.get("concepts") or context.get("concepts", "")
        ).strip()
        story_elements = str(context.get("story_elements", "")).strip()
        weighted_terms = str(context.get("weighted_terms", "")).strip()
        local_candidates = list(options.get("local_reference_candidates") or [])
        research_prompt = draft or (
            "Invent a still-image prompt using these requested concepts: " + concepts
            if concepts
            else ""
        )
        image_prompt = research_prompt or (
            "Use only identity, material, technique, and concept facts from the "
            "user-provided reference images."
            if local_candidates
            else ""
        )
        result: dict[str, object] = {
            "research_context": "",
            "image_context": "",
            "concept_context": "",
            "web_completed": bool(
                options.get("live_research") and research_prompt
            ),
            "image_completed": bool(
                options.get("reference_image_analysis") and image_prompt
            ),
        }
        if options.get("live_research") and research_prompt:
            self._set_status_threadsafe("Researching before Invent...")
            self._log_activity_threadsafe(
                "Running grounded web research inside the Single Image Invent prompt pass."
            )
            model_knowledge = probe_model_visual_knowledge(
                base_url=base_url or DEFAULT_BASE_URL,
                model=model or DEFAULT_MODEL,
                prompt=research_prompt,
                concept_keywords=concepts,
                story_elements=story_elements,
                weighted_terms=weighted_terms,
                timeout=float(self._lm_timeout_seconds()),
                api_key=os.getenv("LM_STUDIO_API_KEY", "lm-studio"),
                cancel_check=lambda: self._raise_if_cancelled(request_id),
            )
            self._raise_if_cancelled(request_id)
            targets = prompt_research_targets(
                research_prompt,
                model_knowledge,
                concept_keywords=concepts,
                weighted_terms=weighted_terms,
            )
            web_research = collect_targeted_prompt_research(
                targets,
                max_results=2,
                timeout=10.0,
                search_engine=str(options.get("search_engine", "")),
            )
            self._raise_if_cancelled(request_id)
            if (
                vague_prompt_issues(research_prompt)
                and vague_prompt_needs_clarification_research(research_prompt)
            ):
                vague_context = collect_vague_prompt_research(
                    research_prompt,
                    timeout=10.0,
                    search_engine=str(options.get("search_engine", "")),
                )
                if vague_context:
                    web_research = (
                        f"{web_research}\n\n{vague_context}"
                        if web_research
                        else vague_context
                    )
            self._raise_if_cancelled(request_id)
            if options.get("enhance_actions"):
                action_context = collect_action_pose_research(
                    research_prompt,
                    timeout=10.0,
                    search_engine=str(options.get("search_engine", "")),
                )
                if action_context:
                    web_research = (
                        f"{web_research}\n\n{action_context}"
                        if web_research
                        else action_context
                    )
            self._raise_if_cancelled(request_id)
            reconciled = reconcile_model_knowledge_with_web(
                base_url=base_url or DEFAULT_BASE_URL,
                model=model or DEFAULT_MODEL,
                prompt=research_prompt,
                model_probe=model_knowledge,
                web_research=web_research,
                timeout=float(self._lm_timeout_seconds()),
                api_key=os.getenv("LM_STUDIO_API_KEY", "lm-studio"),
                cancel_check=lambda: self._raise_if_cancelled(request_id),
            )
            result["research_context"] = (
                "Grounded concept glossary and factual verification only:\n"
                + reconciled.strip()
            )
            self._log_activity_threadsafe("Invent-pass grounded web research result:")
            self._log_activity_threadsafe(str(result["research_context"]))

        if options.get("reference_image_analysis") and image_prompt:
            if local_candidates:
                self._set_status_threadsafe("Analyzing images before Invent...")
                image_candidates, diagnostics = self._collect_reference_images_for_prompt(
                    image_prompt,
                    str(options.get("reference_image_source", "")),
                    local_candidates,
                )
                self._after_threadsafe(
                    0,
                    self._set_reference_candidates,
                    request_id,
                    "prompt",
                    image_candidates,
                )
                for diagnostic in diagnostics:
                    self._log_activity_threadsafe(diagnostic)
                result["image_context"] = analyze_reference_images(
                    base_url=base_url or DEFAULT_BASE_URL,
                    model=model or DEFAULT_MODEL,
                    concept=image_prompt,
                    image_candidates=image_candidates,
                    timeout=float(self._lm_timeout_seconds()),
                    api_key=os.getenv("LM_STUDIO_API_KEY", "lm-studio"),
                    max_images=2,
                    cancel_check=lambda: self._raise_if_cancelled(request_id),
                )
            elif concepts:
                self._set_status_threadsafe("Researching concept images before Invent...")
                result["concept_context"] = collect_integrated_concept_research(
                    concepts,
                    timeout=15.0,
                    text_research=False,
                    search_engine=str(options.get("search_engine", "")),
                    image_analysis=True,
                    image_source=str(options.get("reference_image_source", "")),
                    image_timeout=float(self._lm_timeout_seconds()),
                    base_url=base_url or DEFAULT_BASE_URL,
                    model=model or DEFAULT_MODEL,
                    api_key=os.getenv("LM_STUDIO_API_KEY", "lm-studio"),
                    cancel_check=lambda: self._raise_if_cancelled(request_id),
                )
            self._raise_if_cancelled(request_id)
            self._log_activity_threadsafe("Invent-pass image research result:")
            self._log_activity_threadsafe(
                str(result["image_context"] or result["concept_context"])
                or "No applicable image research target was found."
            )
        return result

    def _invent_single_image_field_worker(
        self,
        request_id: int,
        base_url: str,
        model: str,
        field: str,
        context: dict[str, object],
        research_options: dict[str, object],
        seed: int | None,
    ) -> None:
        try:
            research_bundle: dict[str, object] = {}
            if field == "draft" and (
                research_options.get("live_research")
                or research_options.get("reference_image_analysis")
            ):
                research_bundle = self._collect_single_image_invent_research(
                    request_id=request_id,
                    base_url=base_url,
                    model=model,
                    context=context,
                    options=research_options,
                )
            response = chat_completion(
                base_url=base_url,
                model=model,
                messages=build_single_image_field_suggestion_messages(
                    field=field,
                    research_context=str(
                        research_bundle.get("research_context", "")
                    ),
                    image_context=str(research_bundle.get("image_context", "")),
                    concept_context=str(
                        research_bundle.get("concept_context", "")
                    ),
                    **context,
                ),
                temperature=self._temperature_value(),
                max_tokens=384 if field in {"draft", "story_elements"} else 192,
                timeout=self._lm_timeout_seconds(),
                api_key=os.getenv("LM_STUDIO_API_KEY", "lm-studio"),
                seed=seed,
                ttl=CREATIVE_SESSION_TTL_SECONDS,
                cancel_check=lambda: self._raise_if_cancelled(request_id),
            )
            self._raise_if_cancelled(request_id)
            suggestion = strip_unexpected_scripts(
                self._normalize_or_repair_invent(
                    request_id=request_id,
                    base_url=base_url,
                    model=model,
                    workspace="single",
                    field=field,
                    response=response,
                    seed_value=str(context.get(field, "")),
                    temperature=self._temperature_value(),
                    seed=seed,
                ),
                json.dumps(context, ensure_ascii=False, default=str),
            )
            if not suggestion:
                raise RuntimeError(
                    "LM Studio did not invent a usable value for this Single Image field."
                )
        except CorrectionCancelled:
            self._after_threadsafe(0, self._show_cancelled, request_id)
            return
        except Exception as exc:
            if self._request_cancelled(request_id):
                self._after_threadsafe(0, self._show_cancelled, request_id)
            else:
                self._after_threadsafe(
                    0,
                    self._show_invent_error,
                    request_id,
                    str(exc),
                )
            return
        self._after_threadsafe(
            0,
            self._show_invented_single_image_field,
            request_id,
            field,
            suggestion,
            research_bundle,
            str(research_options.get("concepts", "")).strip(),
        )

    def _show_invented_single_image_field(
        self,
        request_id: int,
        field: str,
        suggestion: str,
        research_bundle: dict[str, object] | None = None,
        source_concepts: str = "",
    ) -> None:
        if request_id != self.active_request_id or self.cancel_event.is_set():
            self._discard_pending_invent_recall(request_id)
            return
        self._commit_invent_recall(request_id)
        if field == "draft":
            self.draft_text.setPlainText(suggestion)
            self.single_invent_research_cache = (
                {
                    **(research_bundle or {}),
                    "draft": suggestion.strip(),
                    "concepts": source_concepts.strip(),
                }
                if research_bundle
                else None
            )
        else:
            targets = {
                "concepts": self.concepts_var,
                "concept_mix": self.concept_mix_var,
                "visual_direction": self.visual_direction_var,
                "goal_headline": self.goal_headline_var,
                "focus": self.focus_var,
                "story_elements": self.story_elements_var,
                "weighted_terms": self.weighted_terms_var,
                "model_instructions": self.model_instructions_var,
                "generation_feedback": self.generation_feedback_var,
            }
            target = targets.get(field)
            if target is None:
                return
            target.set(suggestion)
        self.request_in_progress = False
        self._set_request_controls(False)
        self.status_var.set("Invented Single Image field")
        self._save_settings()

    def _comic_field_context(self) -> dict[str, object]:
        count = max(2, min(12, int(self.comic_panel_count_var.get())))
        return {
            "title": self.comic_title_var.get().strip(),
            "premise": self.comic_premise_var.get().strip(),
            "continuity": self.comic_continuity_var.get().strip(),
            "concepts": self.comic_concepts_var.get().strip(),
            "visual_direction": self.comic_visual_direction_var.get().strip(),
            "dialogue_direction": self.comic_dialogue_direction_var.get().strip(),
            "panels": [
                self.comic_panel_vars[index].get().strip()
                for index in range(count)
            ],
            "panel_count": count,
            "layout": self._effective_comic_layout(),
            "reading_order": self.comic_reading_order_var.get().strip(),
            "aspect_ratio": self.comic_aspect_ratio_var.get().strip(),
            "generator_target": self.generator_target_var.get().strip(),
            "camera_direction": self._camera_direction(),
            "speech_bubbles": bool(self.comic_speech_bubbles_var.get()),
            "artistic_detail_freedom": bool(
                self.artistic_detail_freedom_var.get()
            ),
            "research_concepts": bool(self.live_research_var.get()),
            "search_engine": self.search_engine_var.get().strip(),
        }

    def _collect_grounded_comic_concept_research(
        self,
        *,
        request_id: int,
        base_url: str,
        model: str,
        concepts: str,
        search_engine: str,
    ) -> str:
        normalized_concepts = ", ".join(parse_concepts(concepts))
        if not normalized_concepts:
            return ""

        research_prompt = (
            "Research only these user-requested comic concept keywords as a visual glossary: "
            f"{normalized_concepts}"
        )
        self._set_status_threadsafe("Researching comic concepts...")
        self._log_activity_threadsafe(
            "Checking the model and web for grounded comic concept facts: "
            + normalized_concepts
        )
        model_knowledge = probe_model_visual_knowledge(
            base_url=base_url or DEFAULT_BASE_URL,
            model=model or DEFAULT_MODEL,
            prompt=research_prompt,
            concept_keywords=normalized_concepts,
            story_elements="",
            weighted_terms="",
            timeout=float(self._lm_timeout_seconds()),
            api_key=os.getenv("LM_STUDIO_API_KEY", "lm-studio"),
            cancel_check=lambda: self._raise_if_cancelled(request_id),
        )
        self._raise_if_cancelled(request_id)
        research_targets = prompt_research_targets(
            research_prompt,
            model_knowledge,
            concept_keywords=normalized_concepts,
            weighted_terms="",
        )
        web_research = collect_targeted_prompt_research(
            research_targets,
            max_results=2,
            timeout=10.0,
            search_engine=search_engine,
        )
        self._raise_if_cancelled(request_id)
        reconciled = reconcile_model_knowledge_with_web(
            base_url=base_url or DEFAULT_BASE_URL,
            model=model or DEFAULT_MODEL,
            prompt=research_prompt,
            model_probe=model_knowledge,
            web_research=web_research,
            timeout=float(self._lm_timeout_seconds()),
            api_key=os.getenv("LM_STUDIO_API_KEY", "lm-studio"),
            cancel_check=lambda: self._raise_if_cancelled(request_id),
        )
        self._raise_if_cancelled(request_id)
        grounded = (
            "Grounded concept glossary and factual verification only:\n"
            f"{reconciled.strip()}"
        )
        self._log_activity_threadsafe("Comic concept research result:")
        self._log_activity_threadsafe(grounded)
        return grounded

    def invent_all_comic_panels(self) -> None:
        if self.request_in_progress:
            self.status_var.set("Another model request is already running")
            return
        model = self.model_var.get().strip()
        if not model:
            messagebox.showwarning(
                "Missing model",
                "Select or enter an LM Studio model first.",
            )
            return

        self.active_request_id += 1
        request_id = self.active_request_id
        self.active_request_workspace = "comic"
        self.cancel_event.clear()
        self.request_in_progress = True
        count = max(2, min(12, int(self.comic_panel_count_var.get())))
        panel_keys = [f"comic:panel_{index}" for index in range(1, count + 1)]
        self._stage_invent_recall(
            request_id,
            panel_keys,
            group="comic:all_panels",
        )
        self._set_request_controls(True)
        self.status_var.set(f"Inventing all {count} Comic Story panels...")
        self._save_settings()
        thread = threading.Thread(
            target=self._invent_all_comic_panels_worker,
            args=(
                request_id,
                self._current_base_url(),
                model,
                self._comic_field_context(),
                self._sampling_seed(),
            ),
            daemon=True,
        )
        thread.start()

    def _invent_all_comic_panels_worker(
        self,
        request_id: int,
        base_url: str,
        model: str,
        context: dict[str, object],
        seed: int | None,
    ) -> None:
        panel_count = max(2, min(12, int(context.get("panel_count", 4))))
        try:
            suggestion_context = dict(context)
            research_concepts = bool(
                suggestion_context.pop("research_concepts", False)
            )
            search_engine = str(
                suggestion_context.pop("search_engine", "Auto (all engines)")
            )
            concept_research = (
                self._collect_grounded_comic_concept_research(
                    request_id=request_id,
                    base_url=base_url,
                    model=model,
                    concepts=str(suggestion_context.get("concepts", "")),
                    search_engine=search_engine,
                )
                if research_concepts
                else ""
            )
            response = chat_completion(
                base_url=base_url,
                model=model,
                messages=build_all_comic_panels_suggestion_messages(
                    **suggestion_context,
                    concept_research=concept_research,
                ),
                temperature=self._temperature_value(),
                max_tokens=max(512, min(2304, panel_count * 192)),
                timeout=self._lm_timeout_seconds(),
                api_key=os.getenv("LM_STUDIO_API_KEY", "lm-studio"),
                seed=seed,
                ttl=CREATIVE_SESSION_TTL_SECONDS,
                cancel_check=lambda: self._raise_if_cancelled(request_id),
            )
            self._raise_if_cancelled(request_id)
            panels = normalize_all_comic_panel_suggestions(
                response,
                panel_count=panel_count,
                speech_bubbles=bool(context.get("speech_bubbles", True)),
            )
            source_script_context = json.dumps(
                context,
                ensure_ascii=False,
                default=str,
            )
            panels = [
                strip_unexpected_scripts(panel, source_script_context)
                for panel in panels
            ]
            if len(panels) != panel_count or any(
                not panel.strip() for panel in panels
            ):
                raise RuntimeError(
                    "LM Studio did not return every requested comic panel. "
                    f"Expected Panel 1 through Panel {panel_count} in order."
                )
        except CorrectionCancelled:
            self._after_threadsafe(0, self._show_cancelled, request_id)
            return
        except Exception as exc:
            if self._request_cancelled(request_id):
                self._after_threadsafe(0, self._show_cancelled, request_id)
            else:
                self._after_threadsafe(
                    0,
                    self._show_invent_error,
                    request_id,
                    str(exc),
                )
            return
        self._after_threadsafe(
            0,
            self._show_invented_all_comic_panels,
            request_id,
            panels,
        )

    def _show_invented_all_comic_panels(
        self,
        request_id: int,
        panels: list[str],
    ) -> None:
        if request_id != self.active_request_id or self.cancel_event.is_set():
            self._discard_pending_invent_recall(request_id)
            return
        self._commit_invent_recall(request_id)
        for index, panel in enumerate(panels):
            if index >= len(self.comic_panel_vars):
                break
            self.comic_panel_vars[index].set(panel)
        self.request_in_progress = False
        self._set_request_controls(False)
        self.status_var.set(f"Invented all {len(panels)} Comic Story panels")
        self._save_settings()

    def invent_comic_field(self, field: str) -> None:
        normalized_field = str(field).strip().casefold()
        if (
            normalized_field not in {
                "title",
                "premise",
                "continuity",
                "concepts",
                "visual_direction",
                "dialogue_direction",
            }
            and not re.fullmatch(r"panel_\d+", normalized_field)
        ):
            return
        if self.request_in_progress:
            self.status_var.set("Another model request is already running")
            return
        model = self.model_var.get().strip()
        if not model:
            messagebox.showwarning(
                "Missing model",
                "Select or enter an LM Studio model first.",
            )
            return
        panel_match = re.fullmatch(r"panel_(\d+)", normalized_field)
        if panel_match and int(panel_match.group(1)) > int(
            self.comic_panel_count_var.get()
        ):
            return

        self.active_request_id += 1
        request_id = self.active_request_id
        self.active_request_workspace = "comic"
        self.cancel_event.clear()
        self.request_in_progress = True
        self._stage_invent_recall(
            request_id,
            [f"comic:{normalized_field}"],
        )
        self._set_request_controls(True)
        label = normalized_field.replace("_", " ")
        self.status_var.set(f"Inventing Comic Story {label}...")
        self._save_settings()
        thread = threading.Thread(
            target=self._invent_comic_field_worker,
            args=(
                request_id,
                self._current_base_url(),
                model,
                normalized_field,
                self._comic_field_context(),
                self._sampling_seed(),
            ),
            daemon=True,
        )
        thread.start()

    def _invent_comic_field_worker(
        self,
        request_id: int,
        base_url: str,
        model: str,
        field: str,
        context: dict[str, object],
        seed: int | None,
    ) -> None:
        try:
            suggestion_context = dict(context)
            research_concepts = bool(
                suggestion_context.pop("research_concepts", False)
            )
            search_engine = str(
                suggestion_context.pop("search_engine", "Auto (all engines)")
            )
            concept_research = (
                self._collect_grounded_comic_concept_research(
                    request_id=request_id,
                    base_url=base_url,
                    model=model,
                    concepts=str(suggestion_context.get("concepts", "")),
                    search_engine=search_engine,
                )
                if research_concepts
                else ""
            )
            response = chat_completion(
                base_url=base_url,
                model=model,
                messages=build_comic_field_suggestion_messages(
                    field=field,
                    **suggestion_context,
                    concept_research=concept_research,
                ),
                temperature=self._temperature_value(),
                max_tokens=(
                    384
                    if field in {"premise", "continuity"} or field.startswith("panel_")
                    else 192
                ),
                timeout=self._lm_timeout_seconds(),
                api_key=os.getenv("LM_STUDIO_API_KEY", "lm-studio"),
                seed=seed,
                ttl=CREATIVE_SESSION_TTL_SECONDS,
                cancel_check=lambda: self._raise_if_cancelled(request_id),
            )
            self._raise_if_cancelled(request_id)
            panel_match = re.fullmatch(r"panel_(\d+)", field)
            if panel_match:
                panel_index = int(panel_match.group(1)) - 1
                panels = suggestion_context.get("panels", [])
                seed_value = (
                    str(panels[panel_index])
                    if isinstance(panels, list) and 0 <= panel_index < len(panels)
                    else ""
                )
            else:
                seed_value = str(suggestion_context.get(field, ""))
            suggestion = strip_unexpected_scripts(
                self._normalize_or_repair_invent(
                    request_id=request_id,
                    base_url=base_url,
                    model=model,
                    workspace="comic",
                    field=field,
                    response=response,
                    seed_value=seed_value,
                    temperature=self._temperature_value(),
                    seed=seed,
                ),
                json.dumps(suggestion_context, ensure_ascii=False, default=str),
            )
            if field.startswith("panel_"):
                suggestion = enforce_comic_speech_bubble_contract(
                    suggestion,
                    speech_bubbles=bool(
                        suggestion_context.get("speech_bubbles", True)
                    ),
                )
            if not suggestion:
                raise RuntimeError(
                    "LM Studio did not invent a usable value for this Comic Story field."
                )
        except CorrectionCancelled:
            self._after_threadsafe(0, self._show_cancelled, request_id)
            return
        except Exception as exc:
            if self._request_cancelled(request_id):
                self._after_threadsafe(0, self._show_cancelled, request_id)
            else:
                self._after_threadsafe(
                    0,
                    self._show_invent_error,
                    request_id,
                    str(exc),
                )
            return
        self._after_threadsafe(
            0,
            self._show_invented_comic_field,
            request_id,
            field,
            suggestion,
        )

    def _show_invented_comic_field(
        self,
        request_id: int,
        field: str,
        suggestion: str,
    ) -> None:
        if request_id != self.active_request_id or self.cancel_event.is_set():
            self._discard_pending_invent_recall(request_id)
            return
        self._commit_invent_recall(request_id)
        panel_match = re.fullmatch(r"panel_(\d+)", field)
        if panel_match:
            panel_index = int(panel_match.group(1)) - 1
            if panel_index < 0 or panel_index >= len(self.comic_panel_vars):
                return
            self.comic_panel_vars[panel_index].set(suggestion)
        else:
            targets = {
                "title": self.comic_title_var,
                "premise": self.comic_premise_var,
                "continuity": self.comic_continuity_var,
                "concepts": self.comic_concepts_var,
                "visual_direction": self.comic_visual_direction_var,
                "dialogue_direction": self.comic_dialogue_direction_var,
            }
            target = targets.get(field)
            if target is None:
                return
            target.set(suggestion)
        self.request_in_progress = False
        self._set_request_controls(False)
        self.status_var.set("Invented Comic Story field")
        self._save_settings()

    def correct_prompt(self) -> None:
        draft = self.draft_text.get("1.0", "end").strip()
        if not draft:
            messagebox.showwarning("Missing prompt", "Paste or type a prompt first.")
            return
        self.content_format_var.set("Single Image")
        self._start_prompt_correction(
            draft=draft,
            story_elements=self.story_elements_var.get().strip(),
            destination="prompt",
        )

    def _comic_story_inputs(self) -> tuple[str, str]:
        count = max(2, min(12, int(self.comic_panel_count_var.get())))
        title = self.comic_title_var.get().strip()
        premise = self.comic_premise_var.get().strip()
        continuity = self.comic_continuity_var.get().strip()
        concepts = ", ".join(parse_concepts(self.comic_concepts_var.get()))
        visual_direction = self.comic_visual_direction_var.get().strip()
        dialogue_direction = self.comic_dialogue_direction_var.get().strip()
        layout = self._effective_comic_layout()
        reading_order = self.comic_reading_order_var.get().strip()
        aspect_ratio = self.comic_aspect_ratio_var.get().strip()
        panels = [self.comic_panel_vars[index].get().strip() for index in range(count)]
        missing = [str(index + 1) for index, panel in enumerate(panels) if not panel]
        if missing:
            raise ValueError("Describe every visible panel before generating. Missing panel(s): " + ", ".join(missing))
        if not premise and not title:
            raise ValueError("Add a story premise or title before generating.")

        draft_parts = [f"A {count}-panel comic story page."]
        if title:
            draft_parts.append(f"Working title metadata only: {title}.")
        if premise:
            draft_parts.append(f"Story premise: {premise}.")
        draft_parts.append(f"Page layout: {layout}.")
        draft_parts.append(f"Reading order: {reading_order}.")
        draft_parts.append(f"Aspect ratio: {aspect_ratio}.")
        if self.comic_speech_bubbles_var.get():
            draft_parts.append(
                "Speech bubbles are allowed. The model may invent concise dialogue when "
                "it strengthens a panel beat. Put every spoken line in straight double "
                "quotes, identify its speaker and panel, place it in a clearly readable "
                "speech bubble, and point the bubble tail unambiguously to that speaker."
            )
            if dialogue_direction:
                draft_parts.append(
                    "Mandatory dialogue wording contract for newly invented speech only: "
                    f"{dialogue_direction}. Every newly invented quoted line must visibly "
                    "demonstrate this vocabulary, grammar, sentence length, rhythm, and tone. "
                    "Do not fall back to neutral modern speech. Preserve any dialogue already "
                    "supplied in straight double quotes exactly, and do not apply this writing "
                    "style to narration or visual descriptions."
                )
        else:
            draft_parts.append(
                "Do not invent speech bubbles, thought bubbles, dialogue, or extra visible text."
            )
        if continuity:
            draft_parts.append(f"Shared continuity anchors: {continuity}.")
        if concepts:
            draft_parts.append(
                "Required concepts to integrate across the comic page: "
                f"{concepts}. Represent every concept visibly in at least one appropriate "
                "panel and repeat it only where continuity benefits. Use these concepts as "
                "supporting design language and story texture. Never let them replace the "
                "requested cast, actions, setting, panel beats, or outcome."
            )
        if visual_direction:
            draft_parts.append(
                "Mandatory shared comic style direction: "
                f"{visual_direction}. Apply it consistently to every panel while keeping "
                "the story and action readable."
            )
        story_elements = "\n".join(
            f"Panel {index}: {panel}" for index, panel in enumerate(panels, start=1)
        )
        return " ".join(draft_parts), story_elements

    def correct_comic_story(self) -> None:
        try:
            draft, story_elements = self._comic_story_inputs()
        except ValueError as exc:
            messagebox.showwarning("Incomplete comic story", str(exc))
            return
        self.content_format_var.set("Comic Story")
        self._start_prompt_correction(
            draft=draft,
            story_elements=story_elements,
            destination="comic",
        )

    def _meme_inputs(self) -> str:
        scene = self.meme_scene_var.get().strip()
        top_text = self.meme_top_text_var.get().strip()
        bottom_text = self.meme_bottom_text_var.get().strip()
        response_context = self.meme_response_context_var.get().strip()
        response_goal = self.meme_response_goal_var.get().strip()
        focus = self.meme_focus_var.get().strip()
        if not scene and not response_context:
            raise ValueError(
                "Describe the meme scene or add a situation to respond to before generating."
            )
        if not top_text and not bottom_text and not response_context:
            raise ValueError(
                "Enter top text, bottom text, or a response situation that lets the "
                "model invent the captions."
            )
        if '"' in top_text or '"' in bottom_text:
            raise ValueError(
                "Captions cannot contain straight double quotes because those quotes mark "
                "the exact rendered-text contract. Use curly quotes instead."
            )

        parts = [
            f"A single image-macro meme in {self.meme_aspect_ratio_var.get().strip()} format.",
        ]
        if response_context:
            safe_context = re.sub(r"\s+", " ", response_context).replace('"', "”")
            parts.append(
                "Create an original meme response tailored specifically to this background "
                f"situation, which is context and not text to render: {safe_context}."
            )
            if response_goal:
                safe_goal = re.sub(r"\s+", " ", response_goal).replace('"', "”")
                parts.append(f"Intended response or stance: {safe_goal}.")
            parts.append(
                "Find a concise visual analogy, reaction, reversal, or contrast that answers "
                "the situation creatively instead of merely restating it."
            )
        if scene:
            parts.append(f"Underlying visual scene: {scene}.")
        else:
            parts.append(
                "Invent the clearest, funniest underlying visual scene for this response, "
                "with a specific subject, readable expression or action, and immediate setup."
            )
        if focus:
            parts.append(
                f"Primary visual focus: {focus}. Give this the strongest compositional "
                "emphasis and make it immediately readable."
            )
        tone = self.meme_tone_var.get().strip()
        if tone and tone != "Auto":
            parts.append(
                f"Humor tone: {tone}. Communicate that tone through the contrast "
                "between the image, expression, situation, and caption without adding "
                "explanatory visible text."
            )
        if top_text:
            parts.append(
                "Place one centered top caption at the upper edge reading exactly "
                f'"{top_text}".'
            )
        if bottom_text:
            parts.append(
                "Place one centered bottom caption at the lower edge reading exactly "
                f'"{bottom_text}".'
            )
        caption_count = int(bool(top_text)) + int(bool(bottom_text))
        parts.append(f"Caption treatment: {self.meme_caption_style_var.get().strip()}.")
        if caption_count:
            parts.extend(
                [
                (
                    "Keep every supplied caption large, sharply legible, fully inside "
                    "the canvas, and visually separated from the underlying image."
                ),
                (
                    "The only visible words are the exact quoted caption."
                    if caption_count == 1
                    else "The only visible words are the two exact quoted captions."
                ),
                ]
            )
        else:
            parts.extend(
                [
                    (
                        "Invent either one concise top caption, one concise bottom caption, "
                        "or a two-part top-and-bottom setup, choosing whichever structure "
                        "delivers the strongest response."
                    ),
                    (
                        "The final image prompt must state each invented caption's exact "
                        "wording in straight double quotes and identify whether it belongs "
                        "at the top or bottom."
                    ),
                    (
                        "Use only the newly invented caption words as visible text. Do not "
                        "render the response context, instructions, labels, or alternatives."
                    ),
                ]
            )
        visual_direction = self.meme_visual_direction_var.get().strip()
        if visual_direction:
            parts.append(f"Visual direction: {visual_direction}.")
        return " ".join(parts)

    def _meme_field_context(self) -> dict[str, object]:
        return {
            "response_context": self.meme_response_context_var.get().strip(),
            "response_goal": self.meme_response_goal_var.get().strip(),
            "scene": self.meme_scene_var.get().strip(),
            "focus": self.meme_focus_var.get().strip(),
            "tone": self.meme_tone_var.get().strip(),
            "caption_style": self.meme_caption_style_var.get().strip(),
            "aspect_ratio": self.meme_aspect_ratio_var.get().strip(),
            "visual_direction": self.meme_visual_direction_var.get().strip(),
            "camera_direction": self._camera_direction(),
            "top_caption": self.meme_top_text_var.get().strip(),
            "bottom_caption": self.meme_bottom_text_var.get().strip(),
            "artistic_detail_freedom": bool(
                self.artistic_detail_freedom_var.get()
            ),
        }

    def invent_meme_field(self, field: str) -> None:
        normalized_field = str(field).strip().casefold()
        field_labels = {
            "response_context": "situation",
            "response_goal": "desired response",
            "scene": "scene",
            "focus": "focus",
            "visual_direction": "visual direction",
        }
        if normalized_field not in field_labels:
            return
        if self.request_in_progress:
            self.status_var.set("Another model request is already running")
            return
        model = self.model_var.get().strip()
        if not model:
            messagebox.showwarning(
                "Missing model",
                "Select or enter an LM Studio model first.",
            )
            return

        self.active_request_id += 1
        request_id = self.active_request_id
        self.active_request_workspace = "meme"
        self.cancel_event.clear()
        self.request_in_progress = True
        self._stage_invent_recall(
            request_id,
            [f"meme:{normalized_field}"],
        )
        self._set_request_controls(True)
        label = field_labels[normalized_field]
        self.status_var.set(f"Inventing meme {label}...")
        self._save_settings()
        thread = threading.Thread(
            target=self._invent_meme_field_worker,
            args=(
                request_id,
                self._current_base_url(),
                model,
                normalized_field,
                self._meme_field_context(),
                self._sampling_seed(),
            ),
            daemon=True,
        )
        thread.start()

    def _invent_meme_field_worker(
        self,
        request_id: int,
        base_url: str,
        model: str,
        field: str,
        context: dict[str, object],
        seed: int | None,
    ) -> None:
        try:
            response = chat_completion(
                base_url=base_url,
                model=model,
                messages=build_meme_field_suggestion_messages(
                    field=field,
                    **context,
                ),
                temperature=self._meme_temperature(),
                max_tokens=192,
                timeout=self._lm_timeout_seconds(),
                api_key=os.getenv("LM_STUDIO_API_KEY", "lm-studio"),
                seed=seed,
                ttl=CREATIVE_SESSION_TTL_SECONDS,
                cancel_check=lambda: self._raise_if_cancelled(request_id),
            )
            self._raise_if_cancelled(request_id)
            suggestion = strip_unexpected_scripts(
                self._normalize_or_repair_invent(
                    request_id=request_id,
                    base_url=base_url,
                    model=model,
                    workspace="meme",
                    field=field,
                    response=response,
                    seed_value=str(context.get(field, "")),
                    temperature=self._meme_temperature(),
                    seed=seed,
                ),
                json.dumps(context, ensure_ascii=False, default=str),
            )
            if not suggestion:
                raise RuntimeError(
                    "LM Studio did not invent a usable value for this Meme Creator field."
                )
        except CorrectionCancelled:
            self._after_threadsafe(0, self._show_cancelled, request_id)
            return
        except Exception as exc:
            if self._request_cancelled(request_id):
                self._after_threadsafe(0, self._show_cancelled, request_id)
            else:
                self._after_threadsafe(
                    0,
                    self._show_invent_error,
                    request_id,
                    str(exc),
                )
            return
        self._after_threadsafe(
            0,
            self._show_invented_meme_field,
            request_id,
            field,
            suggestion,
        )

    def _show_invented_meme_field(
        self,
        request_id: int,
        field: str,
        suggestion: str,
    ) -> None:
        if request_id != self.active_request_id or self.cancel_event.is_set():
            self._discard_pending_invent_recall(request_id)
            return
        self._commit_invent_recall(request_id)
        targets = {
            "response_context": self.meme_response_context_var,
            "response_goal": self.meme_response_goal_var,
            "scene": self.meme_scene_var,
            "focus": self.meme_focus_var,
            "visual_direction": self.meme_visual_direction_var,
        }
        target = targets.get(field)
        if target is None:
            return
        target.set(suggestion)
        self.request_in_progress = False
        self._set_request_controls(False)
        field_label = {
            "response_context": "situation",
            "response_goal": "desired response",
            "scene": "scene",
            "focus": "focus",
            "visual_direction": "visual direction",
        }[field]
        self.status_var.set(f"Invented meme {field_label}")
        self._save_settings()

    def invent_meme_caption(self, position: str) -> None:
        normalized_position = str(position).strip().casefold()
        if normalized_position not in {"top", "bottom"}:
            return
        if self.request_in_progress:
            self.status_var.set("Another model request is already running")
            return
        response_context = self.meme_response_context_var.get().strip()
        scene = self.meme_scene_var.get().strip()
        model = self.model_var.get().strip()
        if not model:
            messagebox.showwarning(
                "Missing model",
                "Select or enter an LM Studio model first.",
            )
            return

        self.active_request_id += 1
        request_id = self.active_request_id
        self.active_request_workspace = "meme"
        self.cancel_event.clear()
        self.request_in_progress = True
        self._stage_invent_recall(
            request_id,
            [f"meme:{normalized_position}"],
        )
        self._set_request_controls(True)
        self.status_var.set(f"Inventing {normalized_position} caption...")
        self._save_settings()
        other_caption = (
            self.meme_bottom_text_var.get().strip()
            if normalized_position == "top"
            else self.meme_top_text_var.get().strip()
        )
        current_caption = (
            self.meme_top_text_var.get().strip()
            if normalized_position == "top"
            else self.meme_bottom_text_var.get().strip()
        )
        thread = threading.Thread(
            target=self._invent_meme_caption_worker,
            args=(
                request_id,
                self._current_base_url(),
                model,
                normalized_position,
                response_context,
                self.meme_response_goal_var.get().strip(),
                scene,
                self.meme_focus_var.get().strip(),
                self.meme_tone_var.get().strip(),
                self.meme_caption_style_var.get().strip(),
                self._camera_direction(),
                current_caption,
                other_caption,
                self._sampling_seed(),
            ),
            daemon=True,
        )
        thread.start()

    def _invent_meme_caption_worker(
        self,
        request_id: int,
        base_url: str,
        model: str,
        position: str,
        response_context: str,
        response_goal: str,
        scene: str,
        focus: str,
        tone: str,
        caption_style: str,
        camera_direction: str,
        current_caption: str,
        other_caption: str,
        seed: int | None,
    ) -> None:
        try:
            response = chat_completion(
                base_url=base_url,
                model=model,
                messages=build_meme_caption_suggestion_messages(
                    position=position,
                    response_context=response_context,
                    response_goal=response_goal,
                    scene=scene,
                    focus=focus,
                    tone=tone,
                    caption_style=caption_style,
                    camera_direction=camera_direction,
                    current_caption=current_caption,
                    other_caption=other_caption,
                ),
                temperature=self._meme_temperature(),
                max_tokens=96,
                timeout=self._lm_timeout_seconds(),
                api_key=os.getenv("LM_STUDIO_API_KEY", "lm-studio"),
                seed=seed,
                ttl=CREATIVE_SESSION_TTL_SECONDS,
                cancel_check=lambda: self._raise_if_cancelled(request_id),
            )
            self._raise_if_cancelled(request_id)
            caption = strip_unexpected_scripts(
                self._normalize_or_repair_invent(
                    request_id=request_id,
                    base_url=base_url,
                    model=model,
                    workspace="meme",
                    field=position,
                    response=response,
                    seed_value=current_caption,
                    temperature=self._meme_temperature(),
                    seed=seed,
                ),
                "\n".join(
                    (
                        response_context,
                        response_goal,
                        scene,
                        focus,
                        current_caption,
                        other_caption,
                    )
                ),
            )
            if not caption:
                raise RuntimeError(
                    "LM Studio did not invent usable caption text from the Meme Creator context."
                )
        except CorrectionCancelled:
            self._after_threadsafe(0, self._show_cancelled, request_id)
            return
        except Exception as exc:
            if self._request_cancelled(request_id):
                self._after_threadsafe(0, self._show_cancelled, request_id)
            else:
                self._after_threadsafe(
                    0,
                    self._show_invent_error,
                    request_id,
                    str(exc),
                )
            return
        self._after_threadsafe(
            0,
            self._show_invented_meme_caption,
            request_id,
            position,
            caption,
        )

    def _show_invented_meme_caption(
        self,
        request_id: int,
        position: str,
        caption: str,
    ) -> None:
        if request_id != self.active_request_id or self.cancel_event.is_set():
            self._discard_pending_invent_recall(request_id)
            return
        self._commit_invent_recall(request_id)
        target = (
            self.meme_top_text_var
            if position == "top"
            else self.meme_bottom_text_var
        )
        target.set(caption)
        self.request_in_progress = False
        self._set_request_controls(False)
        self.status_var.set(f"Invented {position} caption")
        self._save_settings()

    def correct_meme(self) -> None:
        try:
            draft = self._meme_inputs()
        except ValueError as exc:
            messagebox.showwarning("Incomplete meme", str(exc))
            return
        self.content_format_var.set("Meme")
        self._start_prompt_correction(
            draft=draft,
            story_elements="",
            destination="meme",
        )

    def _start_prompt_correction(
        self,
        *,
        draft: str,
        story_elements: str,
        destination: str,
    ) -> None:
        if self.request_in_progress:
            self.status_var.set("Another model request is already running")
            return

        requested_prompt = draft
        draft = self._apply_camera_direction(draft, destination)
        draft = self._apply_visual_direction(draft, destination)
        base_url = self._current_base_url()
        if destination == "prompt":
            (
                effective_concepts,
                effective_weighted_terms,
                effective_model_instructions,
                effective_private_model_instructions,
            ) = self._effective_mix_inputs()
            effective_goal_headline = self.goal_headline_var.get().strip()
            effective_focus = self.focus_var.get().strip()
            effective_generation_feedback = self.generation_feedback_var.get().strip()
            effective_reference_image_analysis = self.reference_images_var.get()
            effective_local_references = (
                self._local_reference_candidates("prompt")
                if effective_reference_image_analysis
                else []
            )
        elif destination == "comic":
            effective_concepts = ", ".join(
                parse_concepts(self.comic_concepts_var.get())
            )
            effective_weighted_terms = ""
            comic_style = self.comic_visual_direction_var.get().strip()
            effective_model_instructions = (
                "Mandatory shared comic style direction: "
                f"{comic_style}. Apply it consistently to every panel without "
                "replacing the requested story, cast, action, or concepts."
                if comic_style
                else ""
            )
            effective_private_model_instructions = ""
            effective_goal_headline = ""
            effective_focus = ""
            effective_generation_feedback = ""
            effective_reference_image_analysis = (
                self.comic_reference_images_var.get()
            )
            effective_local_references = (
                self._local_reference_candidates("comic")
                if effective_reference_image_analysis
                else []
            )
        else:
            # Meme Creator has its own content fields. Keep the shared
            # model/workflow configuration, but never inherit creative
            # direction or reference material from Prompt Corrector.
            effective_concepts = ""
            effective_weighted_terms = ""
            effective_model_instructions = ""
            effective_private_model_instructions = ""
            effective_goal_headline = ""
            effective_focus = ""
            effective_generation_feedback = ""
            effective_reference_image_analysis = (
                self.meme_reference_images_var.get()
            )
            effective_local_references = (
                self._local_reference_candidates("meme")
                if effective_reference_image_analysis
                else []
            )
        precomputed_research: dict[str, object] = {}
        if destination == "prompt" and self.single_invent_research_cache:
            cache = self.single_invent_research_cache
            if (
                str(cache.get("draft", "")).strip() == requested_prompt.strip()
                and str(cache.get("concepts", "")).strip()
                == effective_concepts.strip()
            ):
                if self.live_research_var.get() and cache.get("web_completed"):
                    precomputed_research["research_context"] = str(
                        cache.get("research_context", "")
                    )
                    precomputed_research["web_completed"] = True
                if (
                    effective_reference_image_analysis
                    and cache.get("image_completed")
                ):
                    precomputed_research["image_context"] = str(
                        cache.get("image_context", "")
                    )
                    precomputed_research["concept_context"] = str(
                        cache.get("concept_context", "")
                    )
                    precomputed_research["image_completed"] = True
        self.active_request_id += 1
        request_id = self.active_request_id
        self.active_request_workspace = destination
        self.cancel_event.clear()
        self.request_in_progress = True
        self._set_request_controls(True)
        self.status_var.set("Correcting...")
        self._set_progress(5.0, "Starting correction")
        self._start_progress_timer()
        # Keep the previous successful result visible until a replacement has
        # actually passed validation. A recoverable server or contract error
        # must not destroy the user's last usable output.
        self._refresh_local_reference_previews()
        self._save_settings()
        self.active_activity_workspace = destination
        self._log_activity("Started prompt correction.", destination)
        self._log_activity(f"Settings saved to {SETTINGS_PATH.name}.")
        self._log_activity(f"LM Studio URL: {base_url}")
        self._log_activity(f"LM Studio timeout: {self._lm_timeout_seconds()} seconds")
        self._log_activity(
            f"Mode: {self.mode_var.get()} | Detail: {self.detail_var.get()} | "
            f"Length: {self.output_length_var.get()} | Variations: {self._variation_count()} | "
            f"Temperature: {self._temperature_value():.2f} | "
            f"Seed: {self._sampling_seed() if self._sampling_seed() is not None else 'random'} | "
            f"Context tokens: {self._context_token_display()}"
        )
        if is_small_model(self.model_var.get()):
            self._log_activity(
                "Small-model optimization: skipping the free-form audit and allowing at most one targeted repair."
            )
        self._log_activity(
            f"Target: {self.generator_target_var.get()} | Format: {self.content_format_var.get()} | Risk: {self.risk_level_var.get()} | Preset: {self.prompt_preset_var.get()}"
        )
        enabled = [
            label
            for label, enabled_flag in (
                ("fix logic", self.fix_logic_var.get()),
                ("enhance actions", self.enhance_actions_var.get()),
                ("invent and extend story", self.develop_story_var.get()),
                (
                    "artistic detail freedom",
                    self.artistic_detail_freedom_var.get(),
                ),
                ("clean constraints", self.clean_constraints_var.get()),
                ("safe-for-work conversion", self.safe_for_work_var.get()),
                ("explicit adult mode", self.explicit_nsfw_var.get()),
                ("altered encoder safe", self.altered_encoder_var.get()),
                ("thinking mode", self.thinking_mode_var.get()),
                ("grounded web verification", self.live_research_var.get()),
                ("reference image analysis", effective_reference_image_analysis),
                ("audit and repair", self.audit_repair_var.get()),
                ("show generator setup recommendation", self.include_settings_var.get()),
                ("unload model after correction", self.unload_after_generation_var.get()),
            )
            if enabled_flag
        ]
        self._log_activity("Enabled: " + (", ".join(enabled) if enabled else "none"))
        if effective_concepts:
            self._log_activity(f"Concept keywords: {effective_concepts}")
            parsed_concepts = parse_concepts(effective_concepts)
            self._log_activity("Normalized concepts: " + ", ".join(parsed_concepts))
        if (
            destination == "prompt"
            and effective_concepts
            and self.concept_mix_var.get().strip()
        ):
            mix_summary = ", ".join(
                f"{name}:{percentage}%"
                for name, percentage in parse_concept_mix(self.concept_mix_var.get())
            )
            self._log_activity(f"Concept/style mix: {mix_summary}")
        if effective_goal_headline:
            self._log_activity(f"Goal headline: {effective_goal_headline}")
        if effective_focus:
            self._log_activity(f"Result focus: {effective_focus}")
        if story_elements:
            self._log_activity(f"Story elements: {story_elements}")
            panel_descriptions = extract_panel_descriptions(story_elements)
            if panel_descriptions:
                self._log_activity(
                    f"Mandatory panel content detected for {len(panel_descriptions)} panel(s):"
                )
                for number, description in panel_descriptions:
                    self._log_activity(f"- Panel {number}: {description}")
        if effective_weighted_terms:
            weighted_terms = parse_weighted_terms(effective_weighted_terms)
            parsed = ", ".join(f"{term}:{weight:g}" for term, weight in weighted_terms)
            self._log_activity(f"Weighted words: {parsed or effective_weighted_terms}")
        if effective_model_instructions:
            self._log_activity(f"Model instructions: {effective_model_instructions}")
        if effective_generation_feedback:
            self._log_activity(f"Generation feedback: {effective_generation_feedback}")
        classified = classify_prompt_parts(draft)
        if any(classified.values()):
            self._log_activity(
                "Instruction classifier: "
                f"visual={len(classified['visual_content'])}, "
                f"instructions={len(classified['model_instructions'])}, "
                f"avoidances={len(classified['avoidances'])}, "
                f"styles={len(classified['style_references'])}, "
                f"rendered_text={len(classified['rendered_text'])}"
            )
        vague_issues = vague_prompt_issues(draft)
        if vague_issues:
            self._log_activity("Vague prompt analysis: " + ", ".join(vague_issues))
            if vague_prompt_needs_clarification_research(draft) and not self.live_research_var.get():
                self._log_activity(
                    "Prompt is vague enough for clarification research, but grounded web verification is disabled."
                )
        if precomputed_research:
            self._log_activity(
                "Reusing grounded research completed in the Invent prompt pass."
            )

        thread = threading.Thread(
            target=self._correct_prompt_worker,
            args=(
                request_id,
                draft,
                self.generator_target_var.get(),
                self.content_format_var.get(),
                effective_concepts,
                effective_goal_headline,
                effective_focus,
                effective_weighted_terms,
                story_elements,
                effective_model_instructions,
                effective_private_model_instructions,
                effective_generation_feedback,
                self.model_var.get().strip(),
                base_url,
                self.mode_var.get(),
                self.detail_var.get(),
                self.output_length_var.get(),
                None,
                None,
                self.risk_level_var.get(),
                self.prompt_preset_var.get(),
                self._lm_timeout_seconds(),
                self._variation_count(),
                (
                    self._meme_temperature()
                    if destination == "meme"
                    else self._temperature_value()
                ),
                self._sampling_seed(),
                self._context_token_budget(),
                self.preserve_var.get(),
                self.quote_text_var.get(),
                self.fix_logic_var.get(),
                self.enhance_actions_var.get(),
                self.develop_story_var.get(),
                self.artistic_detail_freedom_var.get(),
                self.clean_constraints_var.get(),
                self.safe_for_work_var.get(),
                self.explicit_nsfw_var.get(),
                self.altered_encoder_var.get(),
                self.thinking_mode_var.get(),
                self.live_research_var.get(),
                self.search_engine_var.get(),
                effective_reference_image_analysis,
                self.reference_image_source_var.get(),
                effective_local_references,
                self.audit_repair_var.get(),
                self.include_settings_var.get(),
                self.unload_after_generation_var.get(),
                self.creativity_var.get(),
                slider_value(self.intensity_var.get()),
                slider_value(self.complexity_var.get()),
                slider_value(self.movement_var.get()),
                requested_prompt,
                destination,
                precomputed_research,
            ),
            daemon=True,
        )
        thread.start()

    def stop_current_request(self) -> None:
        if not self.request_in_progress or self.cancel_event.is_set():
            return

        self.cancel_event.set()
        self._discard_pending_invent_recall(self.active_request_id)
        # Permanently invalidate the old worker before making the controls usable
        # again.  A newly started request clears cancel_event, so the request id is
        # what keeps a slow old LM Studio worker stale after that point.
        self.active_request_id += 1
        self.request_in_progress = False
        self.chat_stream_text = ""
        self._refresh_chat_transcript()
        self.status_var.set("Stopped - ready for a new request")
        self._finish_progress(False)
        self._set_request_controls(False)
        if self.workbench_widget is not None:
            self.workbench_widget.on_request_stopped()
        self._log_activity(
            "Stopped. The old LM Studio stream is being closed, its partial result will be discarded, and a new request can start now."
        )

    def _set_request_controls(self, running: bool) -> None:
        if self.correct_button is not None:
            self.correct_button.configure(state="disabled" if running else "normal")
        if self.stop_button is not None:
            self.stop_button.configure(state="normal" if running else "disabled")
        if self.iterate_button is not None:
            has_result = hasattr(self, "corrected_text") and bool(
                self.corrected_text.toPlainText().strip()
            )
            self.iterate_button.configure(
                state="disabled" if running or not has_result else "normal"
            )
        for button in self.single_image_invent_buttons:
            button.configure(state="disabled" if running else "normal")
        if self.comic_generate_button is not None:
            self.comic_generate_button.configure(state="disabled" if running else "normal")
        for button in self.comic_invent_buttons:
            button.configure(state="disabled" if running else "normal")
        if self.comic_stop_button is not None:
            self.comic_stop_button.configure(state="normal" if running else "disabled")
        if self.meme_generate_button is not None:
            self.meme_generate_button.configure(state="disabled" if running else "normal")
        for button in self.meme_invent_buttons:
            button.configure(state="disabled" if running else "normal")
        if self.meme_stop_button is not None:
            self.meme_stop_button.configure(state="normal" if running else "disabled")
        if self.chat_send_button is not None:
            self.chat_send_button.configure(state="disabled" if running else "normal")
        if self.chat_stop_button is not None:
            self.chat_stop_button.configure(state="normal" if running else "disabled")
        self._refresh_invent_recall_buttons()

    def _request_cancelled(self, request_id: int) -> bool:
        return self.cancel_event.is_set() or request_id != self.active_request_id

    def _raise_if_cancelled(self, request_id: int) -> None:
        if self._request_cancelled(request_id):
            raise CorrectionCancelled()

    def _clear_activity(self) -> None:
        self.clear_activity_history()

    def clear_activity_history(self, _checked: bool = False) -> None:
        self.activity_log = []
        self._refresh_activity_text()
        self._save_settings()
        self.status_var.set("Cleared activity history")

    def _activity_scope_key(self) -> str:
        if self.activity_scope_combo is None:
            return "all"
        label = self.activity_scope_combo.currentText()
        return {
            "Prompt Corrector": "prompt",
            "Comic Story": "comic",
            "Meme Creator": "meme",
            "System": "system",
        }.get(label, "all")

    def _refresh_activity_text(self, _value: object = None) -> None:
        if not hasattr(self, "activity_text"):
            return
        scope = self._activity_scope_key()
        lines: list[str] = []
        for event in self.activity_log:
            workspace = event.get("workspace", "system")
            if scope != "all" and workspace != scope:
                continue
            time_label = event.get("time", "")
            prefix = f"[{time_label}] " if time_label else ""
            prefix += f"[{WORKSPACE_LABELS.get(workspace, 'System')}] "
            lines.append(prefix + event.get("message", ""))
        self.activity_text.setPlainText("\n".join(lines))
        cursor = self.activity_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.activity_text.setTextCursor(cursor)

    def _log_activity(self, message: str, workspace: str | None = None) -> None:
        message = str(message or "").strip()
        if not message:
            return
        workspace = workspace or self.active_activity_workspace
        if workspace not in WORKSPACE_LABELS:
            workspace = "system"
        self.activity_log.append(
            {
                "time": datetime.now().strftime("%H:%M:%S"),
                "workspace": workspace,
                "message": message,
            }
        )
        del self.activity_log[:-500]
        self._refresh_activity_text()

    def _log_activity_threadsafe(self, message: str) -> None:
        self._after_threadsafe(0, self._log_activity, message)

    def _set_status_threadsafe(self, status: str) -> None:
        self._after_threadsafe(0, self.status_var.set, status)

    def _set_progress(self, value: float, stage: str) -> None:
        self.progress_var.set(max(0.0, min(100.0, value)))
        self.progress_stage = stage
        self.progress_stage_started_at = time.monotonic()
        self.progress_text_var.set(stage)

    def _set_progress_threadsafe(self, value: float, stage: str) -> None:
        self._after_threadsafe(0, self._set_progress, value, stage)

    def _after_threadsafe(self, delay_ms: int, callback, *args) -> None:
        if self.closing:
            return
        if delay_ms:
            self.dispatcher.invoke.emit(lambda: QTimer.singleShot(delay_ms, callback), ())
        else:
            self.dispatcher.invoke.emit(callback, args)

    def _start_progress_timer(self) -> None:
        self.progress_active = True
        self._tick_progress_timer()

    def _tick_progress_timer(self) -> None:
        if not self.progress_active:
            return
        elapsed = int(max(0.0, time.monotonic() - self.progress_stage_started_at))
        if elapsed:
            self.progress_text_var.set(f"{self.progress_stage} ({elapsed}s)")
        QTimer.singleShot(1000, self._tick_progress_timer)

    def _finish_progress(self, ok: bool) -> None:
        self.progress_active = False
        self.progress_var.set(100.0 if ok else 0.0)
        self.progress_text_var.set("Done" if ok else "Stopped")

    def _history_label(self, entry: dict[str, object]) -> str:
        title = re.sub(r"\s+", " ", str(entry.get("title", ""))).strip()
        goal = re.sub(r"\s+", " ", str(entry.get("goal_headline", ""))).strip()
        requested = re.sub(r"\s+", " ", str(entry.get("requested_prompt", ""))).strip()
        corrected = re.sub(r"\s+", " ", str(entry.get("corrected_prompt", ""))).strip()
        prompt = title or goal or requested or corrected
        if len(prompt) > 72:
            prompt = prompt[:69].rstrip() + "..."
        created_at = str(entry.get("created_at", ""))
        workspace = str(entry.get("workspace", "prompt"))
        workspace_label = {
            "prompt": "Prompt",
            "comic": "Comic",
            "meme": "Meme",
        }.get(workspace, "Prompt")
        label = f"[{workspace_label}] {prompt}"
        label = f"{created_at}  {label}" if created_at else label
        return f"★  {label}" if self._bool_setting(entry.get("pinned"), False) else label

    def _refresh_history_listbox(self, _text: str = "") -> None:
        if self.history_listbox is None:
            return
        query = self.history_search_entry.text().strip().casefold() if self.history_search_entry is not None else ""
        self.history_listbox.clear()
        ordered = sorted(
            enumerate(self.prompt_history),
            key=lambda pair: (not self._bool_setting(pair[1].get("pinned"), False), pair[0]),
        )
        for original_index, entry in ordered:
            searchable = " ".join(str(value) for value in entry.values()).casefold()
            if query and query not in searchable:
                continue
            item = QListWidgetItem(self._history_label(entry))
            item.setData(Qt.ItemDataRole.UserRole, original_index)
            self.history_listbox.addItem(item)

    def _selected_history_index(self) -> int | None:
        if self.history_listbox is None:
            return None
        item = self.history_listbox.currentItem()
        if item is None:
            return None
        index = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(index, int) or index < 0 or index >= len(self.prompt_history):
            return None
        return index

    def _selected_history_entry(self) -> dict[str, object] | None:
        index = self._selected_history_index()
        return self.prompt_history[index] if index is not None else None

    def _workspace_history_state(self, workspace: str) -> dict[str, object]:
        if workspace == "comic":
            return {
                "title": self.comic_title_var.get(),
                "premise": self.comic_premise_var.get(),
                "continuity": self.comic_continuity_var.get(),
                "concepts": self.comic_concepts_var.get(),
                "visual_direction": self.comic_visual_direction_var.get(),
                "dialogue_direction": self.comic_dialogue_direction_var.get(),
                "panel_count": self.comic_panel_count_var.get(),
                "layout": self.comic_layout_var.get(),
                "reading_order": self.comic_reading_order_var.get(),
                "aspect_ratio": self.comic_aspect_ratio_var.get(),
                "speech_bubbles": self.comic_speech_bubbles_var.get(),
                "panels": [panel.get() for panel in self.comic_panel_vars],
                "reference_paths": list(self.comic_reference_paths),
                "reference_image_analysis": self.comic_reference_images_var.get(),
            }
        if workspace == "meme":
            return {
                "scene": self.meme_scene_var.get(),
                "top_text": self.meme_top_text_var.get(),
                "bottom_text": self.meme_bottom_text_var.get(),
                "response_context": self.meme_response_context_var.get(),
                "response_goal": self.meme_response_goal_var.get(),
                "focus": self.meme_focus_var.get(),
                "preset": self.meme_preset_var.get(),
                "tone": self.meme_tone_var.get(),
                "caption_style": self.meme_caption_style_var.get(),
                "aspect_ratio": self.meme_aspect_ratio_var.get(),
                "visual_direction": self.meme_visual_direction_var.get(),
                "reference_paths": list(self.meme_reference_paths),
                "reference_image_analysis": self.meme_reference_images_var.get(),
            }
        return {
            "reference_paths": list(self.local_reference_paths),
            "reference_image_analysis": self.reference_images_var.get(),
        }

    def _restore_history_reference_paths(
        self,
        workspace: str,
        state: dict[str, object],
    ) -> None:
        paths = state.get("reference_paths", [])
        if not isinstance(paths, list):
            paths = []
        restored = [
            str(path)
            for path in paths
            if isinstance(path, str) and Path(path).is_file()
        ][:8]
        if workspace == "comic":
            self.comic_reference_paths[:] = restored
        elif workspace == "meme":
            self.meme_reference_paths[:] = restored
        else:
            self.local_reference_paths[:] = restored
        self._reference_analysis_var(workspace).set(
            self._bool_setting(
                state.get("reference_image_analysis"),
                bool(restored),
            )
        )
        self._refresh_local_reference_previews()

    def _load_comic_history_state(self, state: dict[str, object]) -> None:
        for key, variable in (
            ("title", self.comic_title_var),
            ("premise", self.comic_premise_var),
            ("continuity", self.comic_continuity_var),
            ("concepts", self.comic_concepts_var),
            ("visual_direction", self.comic_visual_direction_var),
            ("dialogue_direction", self.comic_dialogue_direction_var),
        ):
            if key in state:
                variable.set(str(state.get(key, "")))
        self.comic_panel_count_var.set(
            self._int_setting(state.get("panel_count"), 2, 12, 4)
        )
        self.comic_layout_var.set(
            self._choice_setting(
                state.get("layout"),
                COMIC_LAYOUT_PRESETS,
                self.comic_layout_var.get(),
            )
        )
        self.comic_reading_order_var.set(
            self._choice_setting(
                state.get("reading_order"),
                COMIC_READING_ORDER_PRESETS,
                self.comic_reading_order_var.get(),
            )
        )
        self.comic_aspect_ratio_var.set(
            self._choice_setting(
                state.get("aspect_ratio"),
                COMIC_ASPECT_RATIO_PRESETS,
                self.comic_aspect_ratio_var.get(),
            )
        )
        self.comic_speech_bubbles_var.set(
            self._bool_setting(
                state.get("speech_bubbles"),
                self.comic_speech_bubbles_var.get(),
            )
        )
        panels = state.get("panels", [])
        if isinstance(panels, list):
            for index, panel in enumerate(self.comic_panel_vars):
                panel.set(str(panels[index]) if index < len(panels) else "")
        self._restore_history_reference_paths("comic", state)

    def _load_meme_history_state(self, state: dict[str, object]) -> None:
        if "preset" in state:
            self.meme_preset_var.set(
                self._choice_setting(
                    state.get("preset"),
                    tuple(MEME_PRESETS),
                    "Custom",
                )
            )
        for key, variable in (
            ("scene", self.meme_scene_var),
            ("top_text", self.meme_top_text_var),
            ("bottom_text", self.meme_bottom_text_var),
            ("response_context", self.meme_response_context_var),
            ("response_goal", self.meme_response_goal_var),
            ("focus", self.meme_focus_var),
            ("visual_direction", self.meme_visual_direction_var),
        ):
            if key in state:
                variable.set(str(state.get(key, "")))
        self.meme_tone_var.set(
            self._choice_setting(
                state.get("tone"),
                MEME_TONES,
                self.meme_tone_var.get(),
            )
        )
        self.meme_caption_style_var.set(
            self._choice_setting(
                state.get("caption_style"),
                MEME_CAPTION_STYLES,
                self.meme_caption_style_var.get(),
            )
        )
        self.meme_aspect_ratio_var.set(
            self._choice_setting(
                state.get("aspect_ratio"),
                MEME_ASPECT_RATIOS,
                self.meme_aspect_ratio_var.get(),
            )
        )
        self._restore_history_reference_paths("meme", state)

    def _add_prompt_history(
        self,
        requested_prompt: str,
        corrected_prompt: str,
        story_elements_override: str | None = None,
        content_format_override: str | None = None,
        workspace: str | None = None,
    ) -> None:
        workspace = workspace or {
            "comic story": "comic",
            "meme": "meme",
        }.get(str(content_format_override or "").strip().casefold(), "prompt")
        goal_headline = (
            self.goal_headline_var.get().strip()
            if workspace == "prompt"
            else ""
        )
        requested_prompt = requested_prompt.strip()
        corrected_prompt = corrected_prompt.strip()
        if not corrected_prompt:
            return

        key = f"{workspace}\n{goal_headline}\n{requested_prompt}\n{corrected_prompt}"
        existing_metadata = next(
            (
                {
                    "title": str(entry.get("title", "")),
                    "pinned": self._bool_setting(entry.get("pinned"), False),
                }
                for entry in self.prompt_history
                if (
                    f"{str(entry.get('workspace', 'prompt')).strip()}\n"
                    f"{str(entry.get('goal_headline', '')).strip()}\n"
                    f"{str(entry.get('requested_prompt', '')).strip()}\n"
                    f"{str(entry.get('corrected_prompt', entry.get('prompt', ''))).strip()}"
                ) == key
            ),
            {"title": "", "pinned": False},
        )
        self.prompt_history = [
            entry
            for entry in self.prompt_history
            if (
                f"{str(entry.get('workspace', 'prompt')).strip()}\n"
                f"{str(entry.get('goal_headline', '')).strip()}\n"
                f"{str(entry.get('requested_prompt', '')).strip()}\n"
                f"{str(entry.get('corrected_prompt', entry.get('prompt', ''))).strip()}"
            ) != key
        ]
        option_snapshot = self._prompt_option_snapshot()
        if workspace != "prompt":
            option_snapshot.update(
                {
                    "visual_direction": "",
                    "weighted_terms": "",
                    "story_elements": "",
                    "reference_image_analysis": self._reference_analysis_var(
                        workspace
                    ).get(),
                }
            )
        self.prompt_history.insert(
            0,
            {
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "workspace": workspace,
                "workspace_state": self._workspace_history_state(workspace),
                "requested_prompt": requested_prompt,
                "goal_headline": goal_headline,
                "focus": self.focus_var.get().strip() if workspace == "prompt" else "",
                "weighted_terms": (
                    self.weighted_terms_var.get().strip()
                    if workspace == "prompt"
                    else ""
                ),
                "concepts": self.concepts_var.get().strip() if workspace == "prompt" else "",
                "concept_mix": (
                    self.concept_mix_var.get().strip()
                    if workspace == "prompt"
                    else ""
                ),
                "model_instructions": (
                    self.model_instructions_var.get().strip()
                    if workspace == "prompt"
                    else ""
                ),
                "generation_feedback": (
                    self.generation_feedback_var.get().strip()
                    if workspace == "prompt"
                    else ""
                ),
                "risk_level": self.risk_level_var.get().strip(),
                "prompt_preset": self.prompt_preset_var.get().strip(),
                "corrected_prompt": corrected_prompt,
                **existing_metadata,
                **option_snapshot,
                "content_format": content_format_override or self.content_format_var.get(),
                "story_elements": (
                    story_elements_override
                    if story_elements_override is not None
                    else self.story_elements_var.get().strip()
                ),
            },
        )
        del self.prompt_history[PROMPT_HISTORY_LIMIT:]
        self._refresh_history_listbox()
        self._save_settings()

    def load_selected_history_prompt(self, _event: object | None = None) -> None:
        entry = self._selected_history_entry()
        if not entry:
            return
        requested_prompt = str(entry.get("requested_prompt", "")).strip()
        goal_headline = str(entry.get("goal_headline", "")).strip()
        corrected_prompt = str(entry.get("corrected_prompt", entry.get("prompt", ""))).strip()
        workspace = str(entry.get("workspace", "prompt")).strip().casefold()
        state = entry.get("workspace_state", {})
        if not isinstance(state, dict):
            state = {}
        if workspace == "comic":
            self._load_comic_history_state(state)
            if self.comic_result_text is not None:
                self.comic_result_text.setPlainText(corrected_prompt)
            self.mode_tabs.setCurrentIndex(1)
            self.status_var.set("Loaded Comic Story history")
            return
        if workspace == "meme":
            self._load_meme_history_state(state)
            if self.meme_result_text is not None:
                self.meme_result_text.setPlainText(corrected_prompt)
            self.mode_tabs.setCurrentIndex(2)
            self.status_var.set("Loaded Meme Creator history")
            return
        self._load_prompt_options_from_history(entry)
        self.goal_headline_var.set(goal_headline)
        self.focus_var.set(str(entry.get("focus", "")))
        self.weighted_terms_var.set(str(entry.get("weighted_terms", "")))
        self.story_elements_var.set(str(entry.get("story_elements", "")))
        self.concepts_var.set(str(entry.get("concepts", "")))
        self.concept_mix_var.set(str(entry.get("concept_mix", "")))
        self.model_instructions_var.set(str(entry.get("model_instructions", "")))
        self.generation_feedback_var.set(str(entry.get("generation_feedback", "")))
        if requested_prompt:
            self.draft_text.delete("1.0", "end")
            self.draft_text.insert("1.0", requested_prompt)
        self.corrected_text.delete("1.0", "end")
        self.corrected_text.insert("1.0", corrected_prompt)
        self._restore_history_reference_paths("prompt", state)
        self.mode_tabs.setCurrentIndex(0)
        self.status_var.set("Loaded Prompt Corrector history")

    def copy_selected_history_prompt(self) -> None:
        entry = self._selected_history_entry()
        prompt = str(entry.get("corrected_prompt", entry.get("prompt", ""))).strip() if entry else ""
        if not prompt:
            return
        QApplication.clipboard().setText(prompt)
        self.status_var.set("Copied history prompt")

    def delete_selected_history_prompt(self) -> None:
        index = self._selected_history_index()
        if index is None:
            return

        del self.prompt_history[index]
        self._refresh_history_listbox()
        if self.history_listbox is not None and self.history_listbox.count():
            self.history_listbox.setCurrentRow(min(index, self.history_listbox.count() - 1))
        self._save_settings()
        self.status_var.set("Deleted selected history prompt")

    def rename_selected_history_prompt(self, _checked: bool = False) -> None:
        index = self._selected_history_index()
        if index is None:
            return
        entry = self.prompt_history[index]
        current = str(entry.get("title", ""))
        title, accepted = QInputDialog.getText(self.root, "Rename history entry", "Title:", text=current)
        if not accepted:
            return
        entry["title"] = re.sub(r"\s+", " ", title).strip()
        self._refresh_history_listbox()
        self._save_settings()
        self.status_var.set("Renamed history entry")

    def toggle_selected_history_pin(self, _checked: bool = False) -> None:
        index = self._selected_history_index()
        if index is None:
            return
        entry = self.prompt_history[index]
        pinned = not self._bool_setting(entry.get("pinned"), False)
        entry["pinned"] = pinned
        self._refresh_history_listbox()
        self._save_settings()
        self.status_var.set("Pinned history entry" if pinned else "Unpinned history entry")

    def clear_prompt_history(self) -> None:
        if not self.prompt_history:
            return
        if not messagebox.askyesno(
            "Clear prompt history",
            "Delete all saved prompt history entries?",
        ):
            return

        self.prompt_history = []
        self._refresh_history_listbox()
        self._save_settings()
        self.status_var.set("Cleared prompt history")

    def add_local_reference_images(self, _checked: bool = False) -> None:
        filenames, _filter = QFileDialog.getOpenFileNames(
            self.root,
            "Add reference images",
            "",
            "Images (*.png *.jpg *.jpeg *.webp *.gif);;All files (*)",
        )
        self.add_local_reference_paths(filenames)

    def _reference_workspace_key(self) -> str:
        if self.reference_workspace_combo is not None:
            return {
                "Prompt Corrector": "prompt",
                "Comic Story": "comic",
                "Meme Creator": "meme",
            }.get(self.reference_workspace_combo.currentText(), "prompt")
        workspace = self._current_workspace_key()
        return workspace if workspace in {"prompt", "comic", "meme"} else "prompt"

    def _reference_paths(self, workspace: str | None = None) -> list[str]:
        workspace = workspace or self._reference_workspace_key()
        return {
            "prompt": self.local_reference_paths,
            "comic": self.comic_reference_paths,
            "meme": self.meme_reference_paths,
        }.get(workspace, self.local_reference_paths)

    def _reference_analysis_var(self, workspace: str | None = None) -> Value:
        workspace = workspace or self._reference_workspace_key()
        return {
            "prompt": self.reference_images_var,
            "comic": self.comic_reference_images_var,
            "meme": self.meme_reference_images_var,
        }.get(workspace, self.reference_images_var)

    def _on_reference_workspace_changed(self, _label: object = None) -> None:
        if self.reference_analysis_checkbox is not None:
            self.reference_analysis_checkbox.blockSignals(True)
            self.reference_analysis_checkbox.setChecked(
                bool(self._reference_analysis_var().get())
            )
            self.reference_analysis_checkbox.blockSignals(False)
        self._refresh_local_reference_previews()

    def _set_current_reference_analysis(self, enabled: bool) -> None:
        self._reference_analysis_var().set(bool(enabled))
        self._save_settings()

    def add_local_reference_paths(self, paths: list[str]) -> None:
        supported = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
        workspace = self._reference_workspace_key()
        workspace_paths = self._reference_paths(workspace)
        for raw_path in paths:
            path = str(Path(raw_path).expanduser().resolve())
            if (
                Path(path).is_file()
                and Path(path).suffix.lower() in supported
                and path not in workspace_paths
            ):
                workspace_paths.append(path)
            if len(workspace_paths) >= 8:
                break
        if workspace_paths:
            self._reference_analysis_var(workspace).set(True)
        self._on_reference_workspace_changed()
        self._refresh_local_reference_previews()
        self._save_settings()

    def _refresh_local_reference_previews(self) -> None:
        if self.reference_preview_list is None:
            return
        self.reference_preview_list.clear()
        local_role = Qt.ItemDataRole.UserRole.value + 1
        workspace = self._reference_workspace_key()
        for path in self._reference_paths(workspace):
            pixmap = QPixmap(path)
            item = QListWidgetItem(Path(path).name)
            item.setData(Qt.ItemDataRole.UserRole, Path(path).as_uri())
            item.setData(local_role, path)
            if not pixmap.isNull():
                item.setIcon(QIcon(pixmap.scaled(112, 84, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)))
            self.reference_preview_list.addItem(item)
        for candidate in self.web_reference_candidates.get(workspace, []):
            title = candidate.get("title", "Reference image").strip() or "Reference image"
            summary = candidate.get("summary", "").strip()
            item = QListWidgetItem(title + (f"\n{summary}" if summary else ""))
            item.setData(Qt.ItemDataRole.UserRole, candidate.get("url", ""))
            self.reference_preview_list.addItem(item)

    def remove_selected_reference_image(self, _checked: bool = False) -> None:
        if self.reference_preview_list is None:
            return
        item = self.reference_preview_list.currentItem()
        path = str(item.data(Qt.ItemDataRole.UserRole.value + 1) or "") if item is not None else ""
        workspace_paths = self._reference_paths()
        if path in workspace_paths:
            workspace_paths.remove(path)
            self._refresh_local_reference_previews()
            self._save_settings()

    def clear_local_reference_images(self, _checked: bool = False) -> None:
        workspace = self._reference_workspace_key()
        workspace_paths = self._reference_paths(workspace)
        if not workspace_paths:
            return
        workspace_paths.clear()
        self.web_reference_candidates[workspace] = []
        self._refresh_local_reference_previews()
        self._save_settings()

    def _local_reference_candidates(
        self,
        workspace: str | None = None,
    ) -> list[dict[str, str]]:
        return [
            {"title": Path(path).name, "url": Path(path).as_uri(), "summary": "User-provided local reference"}
            for path in self._reference_paths(workspace)
            if Path(path).is_file()
        ]

    def _collect_reference_images_for_prompt(
        self,
        draft: str,
        source: str,
        local_candidates: list[dict[str, str]],
    ) -> tuple[list[dict[str, str]], list[str]]:
        if local_candidates:
            return local_candidates, [
                f"Using {len(local_candidates)} user-provided local reference image(s).",
                "Automatic web image lookup skipped so unrelated search results cannot alter the prompt.",
            ]
        return collect_reference_image_diagnostics(
            draft,
            max_images=4,
            timeout=15.0,
            source=source,
        )

    def _set_reference_candidates(
        self,
        request_id: int,
        workspace: str,
        candidates: list[dict[str, str]],
    ) -> None:
        if request_id != self.active_request_id or self.reference_preview_list is None:
            return
        local_urls = {
            candidate["url"]
            for candidate in self._local_reference_candidates(workspace)
        }
        web_candidates = [
            candidate
            for candidate in candidates
            if candidate.get("url", "") not in local_urls
        ][:4]
        self.web_reference_candidates[workspace] = web_candidates
        if workspace != self._reference_workspace_key():
            return
        self._refresh_local_reference_previews()
        offset = len(self._reference_paths(workspace))
        if web_candidates:
            threading.Thread(
                target=self._fetch_reference_preview_worker,
                args=(request_id, offset, web_candidates),
                daemon=True,
            ).start()

    def _fetch_reference_preview_worker(
        self, request_id: int, offset: int, candidates: list[dict[str, str]]
    ) -> None:
        for index, candidate in enumerate(candidates):
            if self._request_cancelled(request_id):
                return
            url = candidate.get("url", "")
            if not url:
                continue
            try:
                data_url = fetch_image_data_url(url, timeout=8.0, max_bytes=750_000)
            except Exception:
                continue
            self._after_threadsafe(0, self._show_reference_preview_image, request_id, offset + index, url, data_url)

    def _show_reference_preview_image(self, request_id: int, index: int, url: str, data_url: str) -> None:
        if request_id != self.active_request_id or self.reference_preview_list is None:
            return
        if index < 0 or index >= self.reference_preview_list.count():
            return
        try:
            encoded = data_url.split(",", 1)[1]
            image_data = base64.b64decode(encoded)
        except (IndexError, ValueError):
            return
        pixmap = QPixmap()
        if not pixmap.loadFromData(image_data):
            return
        icon_pixmap = pixmap.scaled(
            112,
            84,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        item = self.reference_preview_list.item(index)
        if item is not None and item.data(Qt.ItemDataRole.UserRole) == url:
            item.setIcon(QIcon(icon_pixmap))

    def open_reference_preview(self, item: QListWidgetItem) -> None:
        url = str(item.data(Qt.ItemDataRole.UserRole) or "").strip()
        if url:
            QDesktopServices.openUrl(QUrl(url))

    def _update_available_models(self, models: list[str]) -> None:
        current_model = self.model_var.get().strip()
        refreshed: list[str] = []
        for model in models:
            model = model.strip()
            if model and model not in refreshed:
                refreshed.append(model)

        self.available_models = refreshed
        if self.model_combo is not None:
            self.model_combo.configure(values=self.available_models)
        if refreshed and current_model not in refreshed:
            self.model_var.set(refreshed[0])
        self._save_settings()

    def _variation_count(self) -> int:
        try:
            return max(1, min(3, int(self.variation_var.get())))
        except ValueError:
            return 1

    def _temperature_value(self) -> float:
        try:
            return round(max(0.0, min(2.0, float(self.temperature_var.get()))), 2)
        except ValueError:
            return 0.2

    def _meme_temperature(self) -> float:
        try:
            return round(
                max(0.0, min(2.0, float(self.meme_temperature_var.get()))),
                2,
            )
        except (TypeError, ValueError):
            return 0.7

    def _configured_seed(self) -> int:
        try:
            return max(0, min(2_147_483_647, int(self.seed_var.get())))
        except (TypeError, ValueError):
            return 42

    def _sampling_seed(self) -> int | None:
        return self._configured_seed() if self.fixed_seed_var.get() else None

    def _chat_temperature(self) -> float:
        try:
            return round(max(0.0, min(2.0, float(self.chat_temperature_var.get()))), 2)
        except (TypeError, ValueError):
            return 0.7

    def _chat_max_tokens(self) -> int:
        try:
            return max(1, min(CONTEXT_TOKEN_MAX, int(self.chat_max_tokens_var.get())))
        except (TypeError, ValueError):
            return 2048

    def _context_token_budget(self) -> int:
        selected = str(self.context_token_budget_var.get())
        if selected in CONTEXT_TOKEN_CHOICE_VALUES:
            return CONTEXT_TOKEN_CHOICE_VALUES[selected]
        try:
            return max(
                CONTEXT_TOKEN_MIN,
                min(CONTEXT_TOKEN_MAX, int(selected)),
            )
        except (TypeError, ValueError):
            return CONTEXT_TOKEN_AUTO

    def _context_token_setting_value(self) -> str | int:
        budget = self._context_token_budget()
        return "auto" if budget == CONTEXT_TOKEN_AUTO else budget

    def _context_token_display(self) -> str:
        budget = self._context_token_budget()
        return "Auto" if budget == CONTEXT_TOKEN_AUTO else f"{budget:,}"

    def _lm_timeout_seconds(self) -> int:
        try:
            return max(30, min(3600, int(self.lm_timeout_var.get())))
        except ValueError:
            return 600

    def _camera_direction(self) -> str:
        direction = re.sub(
            r"\s+",
            " ",
            str(self.camera_control_var.get() or ""),
        ).strip(" .")
        if not direction or direction.casefold().startswith("auto"):
            return ""
        return direction

    def _apply_camera_direction(self, draft: str, destination: str) -> str:
        direction = self._camera_direction()
        source = str(draft or "").strip()
        if not direction or direction.casefold() in source.casefold():
            return source
        labels = {
            "prompt": "Camera framing and viewpoint",
            "comic": "Shared camera framing and viewpoint across the comic panels",
            "meme": "Camera framing and viewpoint for the underlying meme image",
        }
        label = labels.get(destination, "Camera framing and viewpoint")
        separator = " " if source.endswith((".", "!", "?")) else ". "
        return f"{source}{separator}{label}: {direction}."

    def _apply_visual_direction(self, draft: str, destination: str) -> str:
        source = str(draft or "").strip()
        if destination != "prompt":
            return source
        direction = re.sub(
            r"\s+",
            " ",
            str(self.visual_direction_var.get() or ""),
        ).strip(" .")
        if not direction or direction.casefold() in source.casefold():
            return source
        separator = " " if source.endswith((".", "!", "?")) else ". "
        return f"{source}{separator}Visual direction: {direction}."

    def _correct_prompt_worker(
        self,
        request_id: int,
        draft: str,
        generator_target: str,
        content_format: str,
        concepts: str,
        goal_headline: str,
        focus: str,
        weighted_terms: str,
        story_elements: str,
        model_instructions: str,
        private_model_instructions: str,
        generation_feedback: str,
        model: str,
        base_url: str,
        mode: str,
        detail_level: str,
        output_length: str,
        output_min_words: int | None,
        output_max_words: int | None,
        risk_level: str,
        prompt_preset: str,
        lm_timeout: int,
        variation_count: int,
        temperature: float,
        seed: int | None,
        context_token_budget: int,
        preserve_strictly: bool,
        optimize_quoted_text: bool,
        fix_logic: bool,
        enhance_actions: bool,
        develop_story: bool,
        artistic_detail_freedom: bool,
        clean_constraints: bool,
        safe_for_work: bool,
        explicit_nsfw: bool,
        altered_text_encoder: bool,
        thinking_mode: bool,
        live_research: bool,
        search_engine: str,
        reference_image_analysis: bool,
        reference_image_source: str,
        local_reference_candidates: list[dict[str, str]],
        audit_repair: bool,
        include_krea_settings: bool,
        unload_after_generation: bool,
        creativity: str,
        intensity: int,
        complexity: int,
        movement: int,
        requested_prompt: str,
        destination: str,
        precomputed_research: dict[str, object] | None = None,
    ) -> None:
        try:
            self._raise_if_cancelled(request_id)
            precomputed_research = precomputed_research or {}
            research_context = str(
                precomputed_research.get("research_context", "")
            )
            image_context = str(precomputed_research.get("image_context", ""))
            web_already_completed = bool(
                precomputed_research.get("web_completed")
            )
            image_already_completed = bool(
                precomputed_research.get("image_completed")
            )
            if live_research and not web_already_completed:
                self._set_status_threadsafe("Checking model knowledge...")
                self._set_progress_threadsafe(10.0, "Checking model knowledge")
                self._log_activity_threadsafe(
                    "Asking the selected model what it knows about the prompt's concepts, actions, objects, materials, places, and styles..."
                )
                model_knowledge = probe_model_visual_knowledge(
                    base_url=base_url or DEFAULT_BASE_URL,
                    model=model or DEFAULT_MODEL,
                    prompt=draft,
                    concept_keywords=concepts,
                    story_elements=story_elements,
                    weighted_terms=weighted_terms,
                    timeout=float(lm_timeout),
                    api_key=os.getenv("LM_STUDIO_API_KEY", "lm-studio"),
                    cancel_check=lambda: self._raise_if_cancelled(request_id),
                )
                self._raise_if_cancelled(request_id)
                self._log_activity_threadsafe("Model prior-knowledge probe:")
                self._log_activity_threadsafe(model_knowledge)
                research_targets = prompt_research_targets(
                    draft,
                    model_knowledge,
                    concept_keywords=concepts,
                    weighted_terms=weighted_terms,
                )
                self._log_activity_threadsafe(
                    "Web verification targets: "
                    + ", ".join(
                        f"{target['category']}={target['term']}" for target in research_targets
                    )
                )
                self._set_status_threadsafe("Checking model knowledge against the web...")
                self._set_progress_threadsafe(18.0, "Verifying concepts and visual terms")
                research_context = collect_targeted_prompt_research(
                    research_targets,
                    max_results=2,
                    timeout=10.0,
                    search_engine=search_engine,
                )
                self._raise_if_cancelled(request_id)
                self._log_activity_threadsafe("Targeted web verification result:")
                self._log_activity_threadsafe(research_context)
                vague_issues = vague_prompt_issues(draft)
                if vague_issues and vague_prompt_needs_clarification_research(draft):
                    self._raise_if_cancelled(request_id)
                    self._set_progress_threadsafe(22.0, "Researching vague prompt meaning")
                    self._log_activity_threadsafe(
                        "Vague prompt issues found: " + ", ".join(vague_issues)
                    )
                    self._log_activity_threadsafe(
                        f"Running additional clarification research for the vague prompt using {search_engine}..."
                    )
                    vague_context = collect_vague_prompt_research(
                        draft,
                        timeout=10.0,
                        search_engine=search_engine,
                    )
                    self._raise_if_cancelled(request_id)
                    if vague_context:
                        self._log_activity_threadsafe("Vague prompt clarification research result:")
                        self._log_activity_threadsafe(vague_context)
                        research_context = f"{research_context}\n\n{vague_context}"
                if enhance_actions:
                    self._raise_if_cancelled(request_id)
                    self._set_progress_threadsafe(28.0, "Researching action and pose mechanics")
                    self._log_activity_threadsafe(
                        f"Running action and pose mechanics research using {search_engine}..."
                    )
                    action_context = collect_action_pose_research(
                        draft,
                        timeout=10.0,
                        search_engine=search_engine,
                    )
                    self._raise_if_cancelled(request_id)
                    if action_context:
                        self._log_activity_threadsafe("Action and pose research result:")
                        self._log_activity_threadsafe(action_context)
                        research_context = f"{research_context}\n\n{action_context}" if research_context else action_context
                    else:
                        self._log_activity_threadsafe("No explicit action or pose research targets found.")
                self._raise_if_cancelled(request_id)
                self._set_status_threadsafe("Reconciling model and web knowledge...")
                self._set_progress_threadsafe(38.0, "Reconciling model knowledge with web evidence")
                self._log_activity_threadsafe(
                    "Comparing the model's prior knowledge with all collected web-search evidence..."
                )
                reconciled_knowledge = reconcile_model_knowledge_with_web(
                    base_url=base_url or DEFAULT_BASE_URL,
                    model=model or DEFAULT_MODEL,
                    prompt=draft,
                    model_probe=model_knowledge,
                    web_research=research_context,
                    timeout=float(lm_timeout),
                    api_key=os.getenv("LM_STUDIO_API_KEY", "lm-studio"),
                    cancel_check=lambda: self._raise_if_cancelled(request_id),
                )
                self._raise_if_cancelled(request_id)
                self._log_activity_threadsafe("Grounded knowledge reconciliation:")
                self._log_activity_threadsafe(reconciled_knowledge)
                research_context = (
                    "Grounded concept glossary and factual verification only:\n"
                    f"{reconciled_knowledge}"
                )
                self._set_status_threadsafe("Correcting...")
            elif live_research:
                self._log_activity_threadsafe(
                    "Using grounded web research from the Invent prompt pass."
                )

            if (
                reference_image_analysis
                and local_reference_candidates
                and not image_already_completed
            ):
                self._raise_if_cancelled(request_id)
                self._set_status_threadsafe("Analyzing reference images...")
                self._set_progress_threadsafe(45.0, "Analyzing main reference images")
                self._log_activity_threadsafe(
                    "Analyzing the user-provided local reference images for the main prompt..."
                )
                image_candidates, image_diagnostics = self._collect_reference_images_for_prompt(
                    draft,
                    reference_image_source,
                    local_reference_candidates,
                )
                self._raise_if_cancelled(request_id)
                self._after_threadsafe(
                    0,
                    self._set_reference_candidates,
                    request_id,
                    destination,
                    image_candidates,
                )
                for diagnostic in image_diagnostics:
                    self._log_activity_threadsafe(diagnostic)
                self._log_activity_threadsafe(
                    f"Found {len(image_candidates)} main prompt reference image(s)."
                )
                if image_candidates:
                    self._log_activity_threadsafe(
                        f"Selected reference image for analysis: {image_candidates[0].get('title', 'reference image')}"
                    )
                image_context = analyze_reference_images(
                    base_url=base_url or DEFAULT_BASE_URL,
                    model=model or DEFAULT_MODEL,
                    concept=draft,
                    image_candidates=image_candidates,
                    timeout=float(lm_timeout),
                    api_key=os.getenv("LM_STUDIO_API_KEY", "lm-studio"),
                    max_images=2,
                    cancel_check=lambda: self._raise_if_cancelled(request_id),
                )
                self._raise_if_cancelled(request_id)
                self._log_activity_threadsafe("Main prompt reference image analysis:")
                self._log_activity_threadsafe(image_context)
                self._set_status_threadsafe("Correcting...")
            elif reference_image_analysis and not image_already_completed:
                self._log_activity_threadsafe(
                    "Whole-prompt web image matching is disabled. Automatic web images are searched only "
                    "for explicit concept keywords and reduced to concept-only glossary facts."
                )

            concept_context = str(
                precomputed_research.get("concept_context", "")
            )
            concept_image_analysis = reference_image_analysis and not local_reference_candidates
            concept_research_enabled = bool(
                concepts and concept_image_analysis and not image_already_completed
            )
            if concept_research_enabled:
                self._raise_if_cancelled(request_id)
                self._set_status_threadsafe("Researching concept keywords...")
                self._set_progress_threadsafe(52.0, "Analyzing concept keyword references")
                if reference_image_analysis and local_reference_candidates:
                    self._log_activity_threadsafe(
                        "Skipping automatic concept web images because local references are active."
                    )
                self._log_activity_threadsafe(
                    f"Running reference image analysis for concept keywords using {reference_image_source}; text evidence was already handled by the grounded knowledge preflight."
                )
                concept_context = collect_integrated_concept_research(
                    concepts,
                    timeout=15.0,
                    text_research=False,
                    search_engine=search_engine,
                    image_analysis=concept_image_analysis,
                    image_source=reference_image_source,
                    image_timeout=float(lm_timeout),
                    base_url=base_url or DEFAULT_BASE_URL,
                    model=model or DEFAULT_MODEL,
                    api_key=os.getenv("LM_STUDIO_API_KEY", "lm-studio"),
                    cancel_check=lambda: self._raise_if_cancelled(request_id),
                )
                self._raise_if_cancelled(request_id)
                self._log_activity_threadsafe("Concept keyword research result:")
                self._log_activity_threadsafe(concept_context)
                self._set_status_threadsafe("Correcting...")
            elif reference_image_analysis and image_already_completed:
                self._log_activity_threadsafe(
                    "Using grounded image research from the Invent prompt pass."
                )
            elif concepts:
                if reference_image_analysis and local_reference_candidates:
                    self._log_activity_threadsafe(
                        "Skipping automatic concept web images because local references are active."
                    )
                if live_research:
                    self._log_activity_threadsafe(
                        "Explicit concept keywords were included in the model-first grounded web verification."
                    )
                else:
                    self._log_activity_threadsafe(
                        "Concept keywords will be integrated without live web or image research."
                    )

            if audit_repair:
                self._set_progress_threadsafe(55.0, "Waiting for LM Studio correction, audit, and repair")
                self._log_activity_threadsafe("Sending prompt to LM Studio for correction and audit/repair...")
            else:
                self._set_progress_threadsafe(65.0, "Waiting for LM Studio correction")
                self._log_activity_threadsafe("Sending prompt to LM Studio for correction...")

            self._raise_if_cancelled(request_id)
            corrected = post_chat_completion(
                base_url=base_url or DEFAULT_BASE_URL,
                model=model or DEFAULT_MODEL,
                prompt=draft,
                generator_target=generator_target,
                content_format=content_format,
                temperature=temperature,
                seed=seed,
                max_tokens=(
                    estimate_audit_max_tokens(detail_level, variation_count, output_length, output_max_words)
                    if audit_repair
                    else estimate_max_tokens(detail_level, variation_count, output_length, output_max_words)
                ),
                timeout=float(lm_timeout),
                api_key=os.getenv("LM_STUDIO_API_KEY", "lm-studio"),
                mode=mode,
                detail_level=detail_level,
                output_length=output_length,
                output_min_words=output_min_words,
                output_max_words=output_max_words,
                risk_level=risk_level,
                prompt_preset=prompt_preset,
                variation_count=variation_count,
                preserve_strictly=preserve_strictly,
                optimize_quoted_text=optimize_quoted_text,
                fix_logic=fix_logic,
                enhance_actions=enhance_actions,
                develop_story=develop_story,
                artistic_detail_freedom=artistic_detail_freedom,
                clean_constraints=clean_constraints,
                safe_for_work=safe_for_work,
                explicit_nsfw=explicit_nsfw,
                altered_text_encoder=altered_text_encoder,
                thinking_mode=thinking_mode,
                include_krea_settings=include_krea_settings,
                creativity=creativity,
                intensity=intensity,
                complexity=complexity,
                movement=movement,
                audit_repair=audit_repair,
                research_context=research_context,
                image_context=image_context,
                concept_context=concept_context,
                goal_headline=goal_headline,
                focus=focus,
                concept_keywords=concepts,
                model_instructions=model_instructions,
                private_model_instructions=private_model_instructions,
                generation_feedback=generation_feedback,
                weighted_terms=weighted_terms,
                story_elements=story_elements,
                context_token_budget=context_token_budget,
                cancel_check=lambda: self._raise_if_cancelled(request_id),
                diagnostic_callback=self._log_activity_threadsafe,
            )
            self._raise_if_cancelled(request_id)
            if unload_after_generation:
                self._set_progress_threadsafe(92.0, "Unloading LM Studio model")
                self._log_activity_threadsafe("Unloading LM Studio model after correction...")
                try:
                    unloaded = unload_lm_studio_model(
                        base_url=base_url or DEFAULT_BASE_URL,
                        model=model or DEFAULT_MODEL,
                        timeout=30.0,
                        api_key=os.getenv("LM_STUDIO_API_KEY", "lm-studio"),
                    )
                    self._raise_if_cancelled(request_id)
                    self._log_activity_threadsafe(
                        "Unloaded LM Studio model instance(s): " + ", ".join(unloaded)
                    )
                except Exception as unload_error:
                    self._log_activity_threadsafe(f"Model unload failed: {unload_error}")
        except CorrectionCancelled:
            self._log_activity_threadsafe("Correction stopped. Late worker result ignored.")
            self._after_threadsafe(0, self._show_cancelled, request_id)
            return
        except Exception as exc:
            if self._request_cancelled(request_id):
                self._log_activity_threadsafe("Correction stopped after an error. Error ignored.")
                self._after_threadsafe(0, self._show_cancelled, request_id)
                return
            self._log_activity_threadsafe(f"Error: {exc}")
            self._after_threadsafe(
                0,
                self._show_error,
                str(exc),
                destination,
                self.progress_stage,
            )
            return

        if self._request_cancelled(request_id):
            self._log_activity_threadsafe("Correction stopped. Late LM Studio result ignored.")
            self._after_threadsafe(0, self._show_cancelled, request_id)
            return
        self._set_progress_threadsafe(95.0, "Finalizing prompt")
        self._log_activity_threadsafe("Correction finished. Final prompt updated.")
        self._after_threadsafe(
            0,
            self._show_corrected,
            request_id,
            requested_prompt,
            corrected,
            destination,
            story_elements,
        )

    def _show_corrected(
        self,
        request_id: int,
        requested_prompt: str,
        corrected: str,
        destination: str = "prompt",
        story_elements: str | None = None,
    ) -> None:
        if request_id != self.active_request_id or self.cancel_event.is_set():
            return
        if destination == "comic" and self.comic_result_text is not None:
            self.comic_result_text.setPlainText(corrected)
        elif destination == "meme" and self.meme_result_text is not None:
            self.meme_result_text.setPlainText(corrected)
        else:
            self.corrected_text.delete("1.0", "end")
            self.corrected_text.insert("1.0", corrected)
        self._update_krea_recommendation()
        self._add_prompt_history(
            requested_prompt,
            corrected,
            story_elements,
            (
                "Comic Story"
                if destination == "comic"
                else "Meme"
                if destination == "meme"
                else "Single Image"
            ),
            destination,
        )
        self.status_var.set("Done")
        self._finish_progress(True)
        self.request_in_progress = False
        self._set_request_controls(False)

    def _show_cancelled(self, request_id: int) -> None:
        self._discard_pending_invent_recall(request_id)
        if request_id != self.active_request_id:
            return
        self.chat_stream_text = ""
        self._refresh_chat_transcript()
        self.status_var.set("Stopped")
        self._finish_progress(False)
        self.request_in_progress = False
        self._set_request_controls(False)

    def _show_error(
        self,
        error: str,
        workspace: str = "prompt",
        stage: str = "",
    ) -> None:
        self.chat_stream_text = ""
        self._refresh_chat_transcript()
        diagnostic = classify_workflow_error(
            error,
            workspace=workspace,
            stage=stage,
        )
        self.active_activity_workspace = (
            workspace if workspace in WORKSPACE_LABELS else "system"
        )
        self._log_activity(
            f"{diagnostic['title']} at {diagnostic['stage']}: "
            f"{diagnostic['detail']} Next: {diagnostic['next_step']}",
            self.active_activity_workspace,
        )
        self.status_var.set(diagnostic["title"])
        self._finish_progress(False)
        self.progress_text_var.set(f"Failed: {diagnostic['stage']}")
        self.request_in_progress = False
        self._set_request_controls(False)
        self.show_library_tab("Activity")
        if diagnostic["category"] == "input":
            messagebox.showwarning(diagnostic["title"], diagnostic["message"])
        else:
            messagebox.showerror(diagnostic["title"], diagnostic["message"])

    def copy_corrected(self) -> None:
        corrected = self.corrected_text.get("1.0", "end").strip()
        if not corrected:
            return

        QApplication.clipboard().setText(corrected)
        self.status_var.set("Copied corrected prompt")

    def iterate_corrected_prompt(self) -> None:
        """Promote the current result to a new draft and run another pass."""

        if self.request_in_progress:
            self.status_var.set("Stop the current request before iterating the result")
            return
        corrected = self.corrected_text.toPlainText().strip()
        if not corrected:
            messagebox.showwarning("Nothing to iterate", "Generate or enter a corrected prompt first.")
            return
        feedback, accepted = QInputDialog.getMultiLineText(
            self.root,
            "Iterate corrected prompt",
            "What should change in the next iteration? Leave this blank to refine it again with the current settings.",
            self.generation_feedback_var.get().strip(),
        )
        if not accepted:
            return
        self.draft_text.setPlainText(corrected)
        self.generation_feedback_var.set(feedback.strip())
        self.status_var.set("Starting another prompt iteration...")
        self.correct_prompt()

    def copy_comic_result(self) -> None:
        corrected = self.comic_result_text.toPlainText().strip() if self.comic_result_text else ""
        if not corrected:
            return
        QApplication.clipboard().setText(corrected)
        self.status_var.set("Copied comic prompt")

    def clear_single_image(self) -> None:
        if self.request_in_progress:
            self.status_var.set("Stop the current request before clearing Single Image")
            return
        self.draft_text.clear()
        self.corrected_text.clear()
        for variable in (
            self.concepts_var,
            self.concept_mix_var,
            self.visual_direction_var,
            self.goal_headline_var,
            self.focus_var,
            self.weighted_terms_var,
            self.story_elements_var,
            self.model_instructions_var,
            self.generation_feedback_var,
        ):
            variable.set("")
        self.concept_preset_selections["prompt"] = []
        self.narrative_preset_selections["prompt"] = {
            "action": [],
            "emotion": [],
        }
        self.visual_preset_selections["prompt"] = []
        self.local_reference_paths.clear()
        self.web_reference_candidates["prompt"] = []
        self.reference_images_var.set(False)
        self._refresh_local_reference_previews()
        self.status_var.set("Single Image cleared")
        self._save_settings()

    def clear_comic_story(self) -> None:
        if self.request_in_progress:
            self.status_var.set("Stop the current request before clearing the comic")
            return
        self.comic_title_var.set("")
        self.comic_premise_var.set("")
        self.narrative_preset_selections["comic"] = {
            "action": [],
            "emotion": [],
        }
        self.comic_continuity_var.set("")
        self.comic_concepts_var.set("")
        self.concept_preset_selections["comic"] = []
        self.comic_visual_direction_var.set("")
        self.visual_preset_selections["comic"] = []
        self.comic_dialogue_direction_var.set("")
        for panel in self.comic_panel_vars:
            panel.set("")
        if self.comic_result_text is not None:
            self.comic_result_text.clear()
        self.comic_reference_paths.clear()
        self.web_reference_candidates["comic"] = []
        self.comic_reference_images_var.set(False)
        self._refresh_local_reference_previews()
        self.status_var.set("Comic story cleared")
        self._save_settings()

    def copy_meme_result(self) -> None:
        corrected = self.meme_result_text.toPlainText().strip() if self.meme_result_text else ""
        if not corrected:
            return
        QApplication.clipboard().setText(corrected)
        self.status_var.set("Copied meme prompt")

    def clear_meme(self) -> None:
        if self.request_in_progress:
            self.status_var.set("Stop the current request before clearing the meme")
            return
        self.meme_scene_var.set("")
        self.narrative_preset_selections["meme"] = {
            "action": [],
            "emotion": [],
        }
        self.meme_top_text_var.set("")
        self.meme_bottom_text_var.set("")
        self.meme_response_context_var.set("")
        self.meme_response_goal_var.set("")
        self.meme_focus_var.set("")
        self.meme_visual_direction_var.set("")
        self.visual_preset_selections["meme"] = []
        if self.meme_result_text is not None:
            self.meme_result_text.clear()
        self.meme_reference_paths.clear()
        self.web_reference_candidates["meme"] = []
        self.meme_reference_images_var.set(False)
        self._refresh_local_reference_previews()
        self.status_var.set("Meme cleared")
        self._save_settings()

    def _update_profile_summary(self) -> None:
        if not hasattr(self, "profile_summary_label"):
            return
        target = str(self.generator_target_var.get())
        content_format = str(self.content_format_var.get())
        format_note = (
            "four-panel default with continuity"
            if content_format == "Comic Story"
            else "finished one-image meme"
            if content_format == "Meme"
            else "one still only"
        )
        setup = (
            "FLUX prompt is explicit because Klein has no prompt upsampling."
            if target == "FLUX.2 Klein 9B"
            else "Krea creativity raw."
        )
        summaries = {
            "Exact": f"Fidelity first: no invented content; {format_note}; {setup}",
            "Improve": f"Faithful polish; {format_note}; target is {target}.",
            "Explore": f"Creative exploration; {format_note}; target is {target}.",
        }
        self.profile_summary_label.setText(summaries.get(str(self.workflow_profile_var.get()), ""))

    def _apply_content_format(self, _content_format: object) -> None:
        self._update_profile_summary()

    def _apply_generator_target(self, target: object) -> None:
        target = str(target)
        is_krea = target == "Krea 2"
        if self.generator_controls_page is not None:
            self.generator_controls_page.setEnabled(is_krea)
        if (
            hasattr(self, "setup_tabs")
            and self.generator_controls_tab_index is not None
        ):
            self.setup_tabs.setTabText(
                self.generator_controls_tab_index,
                "Krea controls" if is_krea else "FLUX setup (fixed)",
            )
            self.setup_tabs.setTabToolTip(
                self.generator_controls_tab_index,
                (
                    "Krea creativity and motion controls. Example: creativity raw for exact adherence."
                    if is_krea
                    else "FLUX.2 Klein distilled setup is fixed guidance shown with the result. Example: 4 steps, guidance 1.0."
                ),
            )
        self._update_profile_summary()
        self._update_krea_recommendation()

    def _krea_recommendation_text(self) -> str:
        return format_generator_recommendation(
            str(self.generator_target_var.get()),
            creativity=str(self.creativity_var.get()),
            intensity=slider_value(self.intensity_var.get()),
            complexity=slider_value(self.complexity_var.get()),
            movement=slider_value(self.movement_var.get()),
        )

    def _update_krea_recommendation(self) -> None:
        if self.krea_recommendation_label is None:
            return
        visible = bool(self.include_settings_var.get())
        self.krea_recommendation_label.setVisible(visible)
        self.krea_recommendation_label.setText(
            self._krea_recommendation_text() if visible else ""
        )
        if self.copy_krea_button is not None:
            self.copy_krea_button.setVisible(visible)

    def copy_krea_recommendation(self) -> None:
        recommendation = self._krea_recommendation_text()
        QApplication.clipboard().setText(recommendation)
        self.status_var.set("Copied generator setup recommendation")


def main() -> None:
    application = QApplication.instance() or QApplication([])
    application.setStyle("Fusion")
    application.setStyleSheet(DARK_STYLESHEET)
    root = PromptCorrectorWindow()
    controller = PromptCorrectorApp(root)
    root.controller = controller
    root.show()
    application.exec()


if __name__ == "__main__":
    main()
