import unittest

from mix_ingredient_presets import (
    MIX_INGREDIENT_KEYS,
    MIX_INGREDIENT_LIMIT,
    MIX_INGREDIENT_PRESETS,
    format_mix_ingredient_names,
    mix_ingredient_key,
    mix_ingredient_keys_for_names,
)


class MixIngredientPresetTests(unittest.TestCase):
    def test_catalog_combines_concepts_styles_and_directions(self):
        self.assertGreaterEqual(len(MIX_INGREDIENT_PRESETS), 63)
        self.assertGreaterEqual(
            sum(len(values) for values in MIX_INGREDIENT_PRESETS.values()),
            1450,
        )
        self.assertEqual(
            len(MIX_INGREDIENT_KEYS),
            sum(len(values) for values in MIX_INGREDIENT_PRESETS.values()),
        )
        self.assertIn(
            "Styles · Rendering modes and media",
            MIX_INGREDIENT_PRESETS,
        )
        self.assertTrue(
            any(category.startswith("Concepts · ") for category in MIX_INGREDIENT_PRESETS)
        )
        self.assertTrue(
            any(category.startswith("Directions · ") for category in MIX_INGREDIENT_PRESETS)
        )
        self.assertTrue(
            any(category.startswith("Actions · ") for category in MIX_INGREDIENT_PRESETS)
        )
        self.assertTrue(
            any(category.startswith("Emotions · ") for category in MIX_INGREDIENT_PRESETS)
        )
        self.assertEqual(MIX_INGREDIENT_LIMIT, 6)
        for values in MIX_INGREDIENT_PRESETS.values():
            self.assertTrue(all("," not in value for value in values))

    def test_selected_names_are_deduplicated_and_follow_catalog_order(self):
        keys = [
            mix_ingredient_key(
                "Directions · Lighting",
                "soft window light",
            ),
            mix_ingredient_key(
                "Concepts · Professions and social roles",
                "courier",
            ),
            mix_ingredient_key(
                "Styles · Rendering modes and media",
                "Watercolor",
            ),
        ]

        self.assertEqual(
            format_mix_ingredient_names(keys),
            ["Watercolor", "courier", "soft window light"],
        )

    def test_existing_names_map_to_first_catalog_match(self):
        keys = mix_ingredient_keys_for_names(
            ["watercolor", "courier", "custom user style"]
        )

        self.assertEqual(
            format_mix_ingredient_names(keys),
            ["Watercolor", "courier"],
        )


if __name__ == "__main__":
    unittest.main()
