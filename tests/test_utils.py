import unittest
import torch
from qml_mor.utils import probabilities_to_dictionary, samples_to_dictionary


class TestUtils(unittest.TestCase):
    def test_probabilities_to_dictionary(self):
        probs = torch.tensor([0.1, 0.2, 0.7, 0.0])
        result = probabilities_to_dictionary(probs)
        expected = {"00": 0.1, "01": 0.2, "10": 0.7, "11": 0.0}
        self.assertDictEqual(result, expected)

    def test_probabilities_to_dictionary_error(self):
        probs = torch.tensor([0.1, 0.2, 0.7])
        with self.assertRaises(ValueError):
            probabilities_to_dictionary(probs)

    def test_samples_to_dictionary_error(self):
        samples = torch.tensor(
            [[0, 0, 1], [1, 1.0, 0], [0, 0, 1], [1, 1, 0], [1, 1, 0]]
        )
        with self.assertRaises(ValueError):
            samples_to_dictionary(samples)

    def test_samples_to_dictionary(self):
        samples = torch.tensor(
            [[0, 0, 1], [1, 1, 0], [0, 0, 1], [1, 1, 0], [1, 1, 0]], dtype=torch.int
        )
        result = samples_to_dictionary(samples)
        expected = {"001": 0.4, "110": 0.6}
        result = samples_to_dictionary(samples)
        self.assertDictEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
