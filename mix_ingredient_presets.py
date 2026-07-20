"""Searchable ingredient catalog for the percentage-based concept/style mixer."""

from __future__ import annotations

from action_emotion_presets import (
    ACTION_PRESETS,
    EMOTION_PRESETS,
    EXPLICIT_ADULT_ACTION_PRESETS,
    EXPLICIT_ADULT_EMOTION_PRESETS,
)
from concept_presets import CONCEPT_PRESETS, EXPLICIT_ADULT_CONCEPT_PRESETS
from krea_prompt_corrector import PROMPT_MODES
from visual_direction_presets import (
    EXPLICIT_ADULT_VISUAL_DIRECTION_PRESETS,
    VISUAL_DIRECTION_PRESETS,
)
from nsfw_scene_contract import infer_nsfw_preset_metadata


MIX_INGREDIENT_LIMIT = 6
_KEY_SEPARATOR = "\x1f"


def _unique(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = " ".join(value.split())
        key = cleaned.casefold()
        if cleaned and key not in seen:
            unique.append(cleaned)
            seen.add(key)
    return tuple(unique)


MIX_INGREDIENT_PRESETS: dict[str, tuple[str, ...]] = {
    "Styles · Rendering modes and media": _unique(
        [mode for mode in PROMPT_MODES if mode != "Auto" and "," not in mode]
    ),
    **{
        f"Concepts · {category}": _unique(list(values))
        for category, values in CONCEPT_PRESETS.items()
    },
    **{
        f"Actions · {category}": _unique(list(values))
        for category, values in ACTION_PRESETS.items()
    },
    **{
        f"Emotions · {category}": _unique(list(values))
        for category, values in EMOTION_PRESETS.items()
    },
    **{
        f"Directions · {category}": _unique(
            [value for value in values if "," not in value]
        )
        for category, values in VISUAL_DIRECTION_PRESETS.items()
    },
}


def mix_ingredient_key(category: str, value: str) -> str:
    """Return the stable serialized identity for one mixer ingredient."""

    return f"{category}{_KEY_SEPARATOR}{value}"


MIX_INGREDIENT_KEYS = frozenset(
    mix_ingredient_key(category, value)
    for category, values in MIX_INGREDIENT_PRESETS.items()
    for value in values
)


def mix_ingredient_preset_catalog(
    *,
    explicit_nsfw: bool = False,
) -> dict[str, tuple[str, ...]]:
    """Return mixer ingredients visible under the active content mode."""

    if not explicit_nsfw:
        return MIX_INGREDIENT_PRESETS
    adult_categories = {
        **{
            f"Actions · {category}": _unique(list(values))
            for category, values in EXPLICIT_ADULT_ACTION_PRESETS.items()
        },
        **{
            f"Emotions · {category}": _unique(list(values))
            for category, values in EXPLICIT_ADULT_EMOTION_PRESETS.items()
        },
        **{
            f"Concepts · {category}": _unique(list(values))
            for category, values in EXPLICIT_ADULT_CONCEPT_PRESETS.items()
        },
        **{
            f"Directions · {category}": _unique(
                [value for value in values if "," not in value]
            )
            for category, values in EXPLICIT_ADULT_VISUAL_DIRECTION_PRESETS.items()
        },
    }
    return {**MIX_INGREDIENT_PRESETS, **adult_categories}


EXPLICIT_ADULT_MIX_INGREDIENT_PRESET_METADATA = {
    mix_ingredient_key(category, value): infer_nsfw_preset_metadata(
        category.split(" · ", 1)[0].lower(),
        category,
        value,
    )
    for category, values in mix_ingredient_preset_catalog(
        explicit_nsfw=True
    ).items()
    if "NSFW" in category
    for value in values
}


def format_mix_ingredient_names(
    keys: list[str] | tuple[str, ...] | set[str],
    *,
    explicit_nsfw: bool = False,
) -> list[str]:
    """Return selected ingredient names in catalog order without duplicates."""

    catalog = mix_ingredient_preset_catalog(explicit_nsfw=explicit_nsfw)
    selected = set(keys)
    names: list[str] = []
    seen: set[str] = set()
    for category, values in catalog.items():
        for value in values:
            if mix_ingredient_key(category, value) not in selected:
                continue
            normalized = value.casefold()
            if normalized not in seen:
                names.append(value)
                seen.add(normalized)
    return names


def mix_ingredient_keys_for_names(
    names: list[str] | tuple[str, ...],
    *,
    explicit_nsfw: bool = False,
) -> list[str]:
    """Map existing mixer names to their first matching catalog entries."""

    catalog = mix_ingredient_preset_catalog(explicit_nsfw=explicit_nsfw)
    wanted = {name.strip().casefold() for name in names if name.strip()}
    found: list[str] = []
    matched: set[str] = set()
    for category, values in catalog.items():
        for value in values:
            normalized = value.casefold()
            if normalized in wanted and normalized not in matched:
                found.append(mix_ingredient_key(category, value))
                matched.add(normalized)
    return found
