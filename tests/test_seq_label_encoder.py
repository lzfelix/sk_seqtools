import unittest

from sk_seqtools import SequenceLabelEncoder


class TestSequentialLabelEncoder(unittest.TestCase):

    def setUp(self):
        self.label_encoder = SequenceLabelEncoder()

    @staticmethod
    def compare_lists(left_lists, right_lists):
        """Returns if two lists of lists have the same elements in the same order"""
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

    def test_encode_with_pad(self):
        original_labels = [['a', 'b'], ['b', 'b']]

        self.label_encoder.fit(original_labels, pad_label='<PADL>')
        self.assertListEqual(['<PADL>', 'a', 'b'],
                             self.label_encoder.classes_.tolist(),
                             "The encoder should contain all categorical labels and the <PADL> label")

    def test_encoder_without_pad(self):
        original_labels = [['a', 'b'], ['b']]

        self.label_encoder.fit(original_labels)
        self.assertListEqual(['a', 'b'],
                             self.label_encoder.classes_.tolist(),
                             'The encoder should *not* contain the <PADL> label')

    def test_encode_invalid_pad(self):
        original_labels = [['a', 'b'], ['b']]

        with self.assertRaises(ValueError):
            self.label_encoder.fit(original_labels, pad_label=123)

    def test_decode_with_pad(self):
        original_labels = [['a', 'b', 'b', 'b', 'b', 'a'], ['b', 'a', 'z']]
        self.label_encoder.fit(original_labels, pad_label='<PADL>')

        decoded_labels = self.encode_decode(original_labels)
        self.assertTrue(self.compare_lists(decoded_labels, original_labels))

        padded_labels = [['a', 'b', 'b', 'b', 'b', 'a'], ['b', 'a', 'z', '<PADL>', '<PADL>', '<PADL>']]
        decoded_padded_labels = self.encode_decode(padded_labels)
        self.assertTrue(padded_labels, decoded_padded_labels)


if __name__ == '__main__':
    unittest.main()
