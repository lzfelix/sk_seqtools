import unittest

from sk_seqtools import SequenceLabelEncoder


class TestSequentialLabelEncoder(unittest.TestCase):

    def setUp(self):
        self.label_encoder = SequenceLabelEncoder()

    @staticmethod
    def compare_lists(left_lists, right_lists):
        """Returns if two lists have the same elements in the same order"""
        for left, right in zip(left_lists, right_lists):
            match = all(x == y for x, y in zip(left, right))
            if not match:
                return False
        return True

    def encode_decode(self, original_labels):
        """Encodes original_labels and then decodes them using self.label_encoder"""
        return self.label_encoder.inverse_transform(
            self.label_encoder.fit_transform(original_labels)
        )

    def test_encode_sequence(self):
        original_labels = [['a', 'a', 'c', 'd', 'b'], ['b', 'a', 'd', 'c', 'b']]

        decoded = self.encode_decode(original_labels)
        self.assertTrue(self.compare_lists(original_labels, decoded))

    def test_encode_different_sequences(self):
        original_labels = [['a', 'a', 'c', 'd', 'b'], ['a']]

        decoded = self.encode_decode(original_labels)
        self.assertTrue(self.compare_lists(original_labels, decoded))

    def test_encode_different_sequences_new_labels(self):
        original_labels = [['a', 'a', 'c', 'd', 'b'], ['z']]

        decoded = self.encode_decode(original_labels)
        self.assertTrue(self.compare_lists(original_labels, decoded))

    def test_enconde_simple_list(self):
        """Only sequences of sequences can be encoded."""

        with self.assertRaises(ValueError):
            invalid_labels = ['a', 'b', 'c']
            self.label_encoder.fit_transform(invalid_labels)

    def test_stepwise_work(self):
        original_labels = [['a', 'a', 'c', 'd', 'b'], ['b', 'a', 'd', 'c', 'b']]
        new_labels = [['c', 'c', 'd', 'b', 'a'], ['a', 'a', 'd', 'c', 'b']]

        self.label_encoder.fit(original_labels)
        decoded = self.encode_decode(new_labels)

        self.assertTrue(self.compare_lists(new_labels, decoded), "Encode/decode failed")


if __name__ == '__main__':
    unittest.main()
