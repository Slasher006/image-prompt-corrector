"""Searchable ingredient catalog for the percentage-based concept/style mixer."""

from __future__ import annotations

from action_emotion_presets import ACTION_PRESETS, EMOTION_PRESETS
from concept_presets import CONCEPT_PRESETS
from krea_prompt_corrector import PROMPT_MODES
from visual_direction_presets import VISUAL_DIRECTION_PRESETS


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


def format_mix_ingredient_names(
    keys: list[str] | tuple[str, ...] | set[str],
) -> list[str]:
    """Return selected ingredient names in catalog order without duplicates."""

    selected = set(keys)
    names: list[str] = []
    seen: set[str] = set()
    for category, values in MIX_INGREDIENT_PRESETS.items():
        for value in values:
            if mix_ingredient_key(category, value) not in selected:
                continue
            normalized = value.casefold()
            if normalized not in seen:
                names.append(value)
                seen.add(normalized)
    return names


def mix_ingredient_keys_for_names(names: list[str] | tuple[str, ...]) -> list[str]:
    """Map existing mixer names to their first matching catalog entries."""

    wanted = {name.strip().casefold() for name in names if name.strip()}
    found: list[str] = []
    matched: set[str] = set()
    for category, values in MIX_INGREDIENT_PRESETS.items():
        for value in values:
            normalized = value.casefold()
            if normalized in wanted and normalized not in matched:
                found.append(mix_ingredient_key(category, value))
                matched.add(normalized)
    return found
