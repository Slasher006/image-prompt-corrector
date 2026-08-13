import unittest

from minimax_h3_i2v_gui import (
    DEFAULT_H3_RESOLUTION,
    build_minimax_h3_i2v_prompt,
    enforce_minimax_h3_i2v_prompt_contract,
    h3_resolution_dimensions,
    minimax_h3_i2v_prompt_issues,
    normalize_minimax_h3_i2v_state,
)


class MiniMaxH3I2VTests(unittest.TestCase):
    def test_prompt_binds_first_and_optional_last_frame_with_audio(self):
        prompt = build_minimax_h3_i2v_prompt(
            scene="A transparent gaming mouse in a black studio.",
            motion="The camera pushes in as blue and orange rim lights pulse.",
            audio="Tactile clicks, a glassy whoosh, and low stereo room tone.",
            duration=5,
            has_last_frame=True,
        )

        self.assertIn("<Picture 1> as the exact first frame", prompt)
        self.assertIn("<Picture 2> as the final frame", prompt)
        self.assertIn("Over 5 seconds", prompt)
        self.assertIn("Audio:", prompt)
        self.assertNotIn("Preserve its subject", prompt)
        self.assertIn("Preserve Picture 1's subject", prompt)
        self.assertEqual(
            minimax_h3_i2v_prompt_issues(prompt, has_last_frame=True),
            [],
        )

    def test_contract_adds_only_missing_frame_bindings(self):
        repaired = enforce_minimax_h3_i2v_prompt_contract(
            "The subject turns toward the camera while rain falls.",
            has_last_frame=True,
        )
        self.assertTrue(repaired.startswith("Use <Picture 1> as the exact first frame."))
        self.assertTrue(repaired.endswith("<Picture 2> as the final frame."))

    def test_sampler_controls_are_rejected_from_natural_prompt(self):
        issues = minimax_h3_i2v_prompt_issues(
            "Use Picture 1 as the first frame. CFG: 5. The camera pans left."
        )
        self.assertTrue(any("sampler controls" in issue for issue in issues))

    def test_ambiguous_it_is_rejected_but_clear_its_is_expanded(self):
        ambiguous = (
            "Use <Picture 1> as the exact first frame. "
            "A cat sits beside a lamp. It glows."
        )
        self.assertTrue(
            any(
                "unresolved entity references" in issue
                for issue in minimax_h3_i2v_prompt_issues(ambiguous)
            )
        )

        repaired = enforce_minimax_h3_i2v_prompt_contract(
            "The penis is visible. Its tip releases semen."
        )
        self.assertIn("The penis's tip", repaired)
        self.assertEqual(minimax_h3_i2v_prompt_issues(repaired), [])

    def test_state_defaults_to_480p_for_local_low_vram_workflow(self):
        state = normalize_minimax_h3_i2v_state(
            {"duration": 99, "resolution": "unknown", "seed": -1}
        )
        self.assertEqual(state["duration"], 15.0)
        self.assertEqual(state["resolution"], DEFAULT_H3_RESOLUTION)
        self.assertEqual(state["seed"], 0)
        self.assertEqual(h3_resolution_dimensions(state["resolution"]), (864, 480))


if __name__ == "__main__":
    unittest.main()
