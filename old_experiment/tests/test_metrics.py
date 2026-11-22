import unittest
import torch

from extractor.training.utils import metrics


def logits_from_mask(mask):
    mask = torch.tensor(mask, dtype=torch.float32)
    return torch.where(mask > 0, torch.tensor(10.0), torch.tensor(-10.0)).unsqueeze(0).unsqueeze(0)


class MetricsTests(unittest.TestCase):
    def test_known_confusion_matrix(self):
        # target: 2 foreground pixels; prediction: 2 TP, 1 FP, 0 FN, 1 TN
        target = torch.tensor([[[[1, 1], [0, 0]]]], dtype=torch.float32)
        pred = torch.tensor([[[[1, 1], [1, 0]]]], dtype=torch.float32)
        got = metrics(logits_from_mask(pred[0, 0]), target)
        self.assertEqual((got["tp"], got["tn"], got["fp"], got["fn"]), (2, 1, 1, 0))
        self.assertAlmostEqual(got["dice"], 4 / 5)
        self.assertAlmostEqual(got["iou"], 2 / 3)
        self.assertAlmostEqual(got["precision"], 2 / 3)
        self.assertAlmostEqual(got["recall"], 1.0)
        self.assertAlmostEqual(got["specificity"], 1 / 2)
        self.assertAlmostEqual(got["accuracy"], 3 / 4)

    def test_empty_and_all_foreground_conventions(self):
        empty = torch.zeros((1, 1, 2, 2))
        all_fg = torch.ones((1, 1, 2, 2))
        self.assertEqual(metrics(logits_from_mask(empty[0, 0]), empty)["dice"], 1.0)
        self.assertEqual(metrics(logits_from_mask(all_fg[0, 0]), empty)["dice"], 0.0)
        self.assertEqual(metrics(logits_from_mask(empty[0, 0]), all_fg)["recall"], 0.0)
        self.assertEqual(metrics(logits_from_mask(all_fg[0, 0]), all_fg)["specificity"], 1.0)

    def test_threshold_is_applied_to_probability(self):
        target = torch.tensor([[[[1, 0], [0, 0]]]], dtype=torch.float32)
        # sigmoid(0)=0.5; at threshold .6 this is background.
        logits = torch.tensor([[[[0.0, -10.0], [-10.0, -10.0]]]])
        got = metrics(logits, target, threshold=.6)
        self.assertEqual((got["tp"], got["fn"]), (0, 1))


if __name__ == "__main__":
    unittest.main()
