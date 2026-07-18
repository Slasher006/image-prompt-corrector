import unittest

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


class ActionEmotionPresetTests(unittest.TestCase):
    def test_action_catalog_is_broad_unique_and_concrete(self):
        self.assertGreaterEqual(len(ACTION_PRESETS), 14)
        self.assertGreaterEqual(sum(map(len, ACTION_PRESETS.values())), 250)
        self.assertEqual(
            len(ACTION_PRESET_KEYS),
            sum(map(len, ACTION_PRESETS.values())),
        )
        for category in (
            "Movement and locomotion",
            "Hands props and object interaction",
            "Investigation and discovery",
            "Conflict defense and pursuit",
            "Rescue survival and emergency",
            "Sport dance and physical skill",
            "Science medicine and technology",
            "Ceremony celebration and ritual",
        ):
            self.assertIn(category, ACTION_PRESETS)
        self.assertTrue(
            all("," not in value for values in ACTION_PRESETS.values() for value in values)
        )

    def test_emotion_catalog_is_broad_unique_and_visibly_expressed(self):
        self.assertGreaterEqual(len(EMOTION_PRESETS), 12)
        self.assertGreaterEqual(sum(map(len, EMOTION_PRESETS.values())), 210)
        self.assertEqual(
            len(EMOTION_PRESET_KEYS),
            sum(map(len, EMOTION_PRESETS.values())),
        )
        for category in (
            "Joy delight and amusement",
            "Love affection and tenderness",
            "Curiosity wonder and awe",
            "Fear anxiety and vulnerability",
            "Anger frustration and defiance",
            "Sadness grief and loneliness",
            "Complex mixed and changing emotions",
        ):
            self.assertIn(category, EMOTION_PRESETS)
        self.assertTrue(
            all("," not in value for values in EMOTION_PRESETS.values() for value in values)
        )
        self.assertEqual(NARRATIVE_PRESET_LIMIT, 6)

    def test_format_and_merge_preserve_catalog_order_without_duplicates(self):
        action_keys = [
            narrative_preset_key(
                "action",
                "Investigation and discovery",
                "opening a long-forgotten archive",
            ),
            narrative_preset_key(
                "action",
                "Movement and locomotion",
                "running uphill while glancing back",
            ),
        ]
        formatted = format_narrative_presets("action", action_keys)
        self.assertEqual(
            formatted,
            "running uphill while glancing back; opening a long-forgotten archive",
        )
        self.assertEqual(
            merge_narrative_text(
                "A courier reaches the gate; running uphill while glancing back",
                formatted,
            ),
            "A courier reaches the gate; running uphill while glancing back; "
            "opening a long-forgotten archive",
        )


if __name__ == "__main__":
    unittest.main()
