# sk_seqtools
Extending the LabelEncoder from [sklearn](https://github.com/scikit-learn/scikit-learn) to handle sequences of labels
to perform, for instance, NER tagging tasks.

## Installing
You can install from remote with

```bash
pip install git+https://github.com/lzfelix/sk_seqtools
```

## Usage examples

For now, please refer to the tests folder for examples on how to use the SequenceLabelEncoder.

## Running tests

Since the repository consists only of `SequentialLabelEncoder` so far, you can run the tests from the root folder with:

```bash
 python -m unittest tests/test_seq_label_encoder.py
```
