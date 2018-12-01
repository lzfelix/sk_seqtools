from sklearn.preprocessing import LabelEncoder
import numpy as np


class SequenceLabelEncoder(LabelEncoder):
    """Extends sklearn LabelEncoder for sequences of labels."""
    
    def __init__(self):
        super().__init__()

    def fit(self, sequence, pad_label=None):
        unique_items = set()
        if pad_label:
            if not isinstance(pad_label, str):
                raise ValueError('pad_label should be either str or None')
            unique_items.add('<PADL>')

        for sample in sequence:
            unique_items.update(sample)

        super().fit(list(unique_items))
        return self

    def transform(self, sequence):
        f = super().transform
        return list([f(sample) for sample in sequence])

    def fit_transform(self, sequence):
        self.fit(sequence)
        return self.transform(sequence)
    
    def inverse_transform(self, sequences):
        itransformed = list()
        for sample in sequences:
            itransformed.append(super().inverse_transform(sample))

        return np.asarray(itransformed)
