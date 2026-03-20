import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class Tokenizer:
    """
    A basic word-level tokenizer that maps unique words to integer IDs
    and reconstructs text from token sequences.
    """
    def __init__(self, texts):
        """
        Initializes the tokenizer by building a vocabulary from the input corpus.
        :param texts: texts (list of str):
        """
        # Create a unique sorted list of all words in the provided corpus
        all_text = " ".join(texts)
        self.words = sorted(set(all_text.split()))
        # Build lookup dictionaries for encoding and decoding
        self.word2idx = {w: i for i, w in enumerate(self.words)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self.vocab_size = len(self.words)

    def encode(self, text):
        """Converts a string of text into a list of integer tokens."""
        return [self.word2idx[w] for w in text.split() if w in self.word2idx]

    def decode(self, tokens):
        """Converts a list of integer tokens back into a human-readable string."""
        return " ".join([self.idx2word[int(t)] for t in tokens])
