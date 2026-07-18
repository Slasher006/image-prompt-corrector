import unittest

from concept_presets import (
    CONCEPT_PRESET_KEYS,
    CONCEPT_PRESETS,
    CONCEPT_SELECTION_LIMIT,
    concept_preset_key,
    format_concept_presets,
    merge_concept_text,
)


class ConceptPresetTests(unittest.TestCase):
    def test_catalog_is_broad_unique_and_comma_safe(self):
        self.assertGreaterEqual(len(CONCEPT_PRESETS), 22)
        self.assertGreaterEqual(
            sum(len(values) for values in CONCEPT_PRESETS.values()),
            480,
        )
        self.assertEqual(
            len(CONCEPT_PRESET_KEYS),
            sum(len(values) for values in CONCEPT_PRESETS.values()),
        )
        for category in (
            "Character archetypes",
            "Relationships and groups",
            "Animals and wildlife",
            "Mythical beings and folklore",
            "Science fiction entities and technology",
            "Environments and biomes",
            "Architecture and built spaces",
            "Narrative situations and conflicts",
            "Symbols and visual metaphors",
            "Historical eras and worldbuilding",
            "Fashion costume and adornment",
            "Food drink and culinary culture",
            "Product and graphic design",
            "Abstract systems and spatial ideas",
        ):
            self.assertIn(category, CONCEPT_PRESETS)
        for values in CONCEPT_PRESETS.values():
            self.assertTrue(all("," not in value for value in values))
        self.assertEqual(CONCEPT_SELECTION_LIMIT, 8)

    def test_selected_concepts_format_in_catalog_order_without_category_labels(self):
        selected = [
            concept_preset_key("Objects props and tools", "antique compass"),
            concept_preset_key("Professions and social roles", "courier"),
            concept_preset_key(
                "Narrative situations and conflicts",
                "impossible delivery",
            ),
        ]

        self.assertEqual(
            format_concept_presets(selected),
            "courier, antique compass, impossible delivery",
        )

    def test_append_merge_preserves_manual_concepts_and_removes_duplicates(self):
        self.assertEqual(
            merge_concept_text(
                "custom sky city, Courier",
                "courier, antique compass",
            ),
            "custom sky city, Courier, antique compass",
        )


if __name__ == "__main__":
    unittest.main()
