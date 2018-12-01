from sklearn.preprocessing import LabelEncoder
import numpy as np


class SequenceLabelEncoder(LabelEncoder):
    """Extends sklearn LabelEncoder for sequences of labels allowing to encode an extra <PAD> label, which can be useful
    to make all sequences of labels have the same length. Notice that internally sklean uses the builtin sort() function
    from Python, so it is not possible to ensure that the PAD label is going to be mapped to the label 0. If, for
    instance, each class corresponds to a letter from a-z, then setting the PAD label to <PAD> is going to map it to the
    index 0, since sorted('a', 'z', 'c', '<PAD>') = ['<PAD>', 'a', 'c', 'z'].
    """
    
    def __init__(self):
        super().__init__()

    def fit(self, sequences, pad_label=None):
        unique_items = set()

        for sample in sequences:
            unique_items.update(sample)

        if pad_label:
            if not isinstance(pad_label, str):
                raise ValueError('pad_label should be either str or None')
            if pad_label in unique_items:
                raise RuntimeError(f'The pad_label {pad_label} must be different from the labels in the dataset')

        unique_items = list(unique_items)

        if pad_label is not None:
            unique_items = [pad_label] + list(unique_items)

        super().fit(unique_items)
        return self

    def transform(self, sequence):
        f = super().transform
        return list([f(sample) for sample in sequence])

    def fit_transform(self, sequence, pad_label=None):
        self.fit(sequence, pad_label)
        return self.transform(sequence)
    
    def inverse_transform(self, sequences):
        itransformed = list()
        for sample in sequences:
            itransformed.append(super().inverse_transform(sample))

        return np.asarray(itransformed)
