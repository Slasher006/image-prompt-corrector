import unittest

from visual_direction_presets import (
    EXPLICIT_ADULT_VISUAL_DIRECTION_PRESETS,
    VISUAL_DIRECTION_PRESET_KEYS,
    VISUAL_DIRECTION_PRESETS,
    format_visual_direction_presets,
    visual_direction_preset_catalog,
    visual_preset_key,
)


class VisualDirectionPresetTests(unittest.TestCase):
    def test_catalog_is_exhaustive_unique_and_covers_major_direction_dimensions(self):
        self.assertGreaterEqual(len(VISUAL_DIRECTION_PRESETS), 14)
        self.assertGreaterEqual(
            sum(len(values) for values in VISUAL_DIRECTION_PRESETS.values()),
            400,
        )
        self.assertEqual(
            len(VISUAL_DIRECTION_PRESET_KEYS),
            sum(len(values) for values in VISUAL_DIRECTION_PRESETS.values()),
        )
        for category in (
            "Mood and emotional tone",
            "Lighting",
            "Color palette",
            "Weather and atmosphere",
            "Composition and hierarchy",
            "Depth and focus",
            "Motion and energy",
            "Surface and texture",
            "Art direction and genre",
            "Image finish and color grade",
        ):
            self.assertIn(category, VISUAL_DIRECTION_PRESETS)

    def test_selected_presets_format_in_catalog_order_with_category_binding(self):
        selected = [
            visual_preset_key("Lighting", "warm golden-hour sunlight"),
            visual_preset_key(
                "Mood and emotional tone",
                "nostalgic and bittersweet",
            ),
            visual_preset_key("Weather and atmosphere", "low valley mist"),
        ]

        result = format_visual_direction_presets(selected)

        self.assertTrue(result.startswith("Mood and emotional tone:"))
        self.assertIn("Lighting: warm golden-hour sunlight", result)
        self.assertIn("Weather and atmosphere: low valley mist", result)
        self.assertTrue(result.endswith("."))

    def test_explicit_adult_directions_are_only_formatted_when_enabled(self):
        self.assertGreaterEqual(
            len(EXPLICIT_ADULT_VISUAL_DIRECTION_PRESETS),
            8,
        )
        self.assertGreaterEqual(
            sum(map(len, EXPLICIT_ADULT_VISUAL_DIRECTION_PRESETS.values())),
            140,
        )
        self.assertFalse(
            any(
                category.startswith("NSFW")
                for category in visual_direction_preset_catalog()
            )
        )
        adult_catalog = visual_direction_preset_catalog(explicit_nsfw=True)
        self.assertTrue(any(category.startswith("NSFW") for category in adult_catalog))
        category, values = next(
            iter(EXPLICIT_ADULT_VISUAL_DIRECTION_PRESETS.items())
        )
        key = visual_preset_key(category, values[0])
        self.assertEqual(format_visual_direction_presets([key]), "")
        self.assertIn(
            values[0],
            format_visual_direction_presets([key], explicit_nsfw=True),
        )
        self.assertNotIn(
            "NSFW",
            format_visual_direction_presets([key], explicit_nsfw=True),
        )


if __name__ == "__main__":
    unittest.main()
