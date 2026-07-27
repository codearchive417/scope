import random
import unittest

from scope.data_utils import retained_teacher_records, sample_disjoint_index
from scope.data_collator import SCOPEDataCollator
from scope.prompts import build_teacher_privileged_user_message


class _FakeTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, **_kwargs):
        return messages[0]["content"]

    def __call__(self, texts, padding=False, return_tensors=None, **_kwargs):
        import torch

        if isinstance(texts, str):
            texts = [texts]
        ids = [[index + 1 for index, _ in enumerate(text)] for text in texts]
        if padding:
            width = max(len(row) for row in ids)
            ids = [[0] * (width - len(row)) + row for row in ids]
        masks = [[int(token != 0) for token in row] for row in ids]
        if return_tensors == "pt":
            ids, masks = torch.tensor(ids), torch.tensor(masks)
        return {"input_ids": ids, "attention_mask": masks}


class CalibrationSamplingTests(unittest.TestCase):
    def test_sample_never_returns_target(self):
        rng = random.Random(7)
        for target in range(8):
            samples = [sample_disjoint_index(target, 8, rng) for _ in range(100)]
            self.assertNotIn(target, samples)
            self.assertTrue(all(0 <= sample < 8 for sample in samples))

    def test_sampling_requires_two_examples(self):
        with self.assertRaises(ValueError):
            sample_disjoint_index(0, 1)

    def test_teacher_prompt_separates_calibration_and_target(self):
        prompt = build_teacher_privileged_user_message(
            "target problem", "calibration problem", "calibrated reasoning"
        )
        self.assertLess(prompt.index("calibration problem"), prompt.index("target problem"))
        self.assertIn("calibrated reasoning", prompt)

    def test_collator_uses_a_different_calibration_row(self):
        dataset = [
            {
                "_scope_index": index,
                "problem": f"problem {index}",
                "solution": f"solution {index}",
                "teacher_reasoning": f"reasoning {index}",
            }
            for index in range(3)
        ]
        collator = SCOPEDataCollator(_FakeTokenizer(), calibration_dataset=dataset)
        batch = collator([dataset[1]])
        self.assertNotEqual(batch["calibration_indices"][0], 1)
        self.assertEqual(batch["problems"], ["problem 1"])
        self.assertNotEqual(batch["calibration_problems"], batch["problems"])


class TeacherFilteringTests(unittest.TestCase):
    def test_filtered_and_empty_records_are_not_retained(self):
        records = [
            {"problem": "p0", "solution": "s0", "teacher_reasoning": "r0"},
            {"problem": "p1", "solution": "s1", "filtered": True},
            {"problem": "p2", "solution": "s2", "teacher_reasoning": ""},
        ]
        self.assertEqual(
            retained_teacher_records(records),
            [{"problem": "p0", "solution": "s0", "teacher_reasoning": "r0"}],
        )


if __name__ == "__main__":
    unittest.main()
