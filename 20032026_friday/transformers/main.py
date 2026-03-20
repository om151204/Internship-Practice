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

class TransformerBlock(layers.Layer):
    """
    A standard Transformer layer consisting of Causal Multi-Head Attention
    followed by a Position-wise Feed-Forward Network.
    """
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super().__init__()
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dense(d_model)
        ])
        self.layer_norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, x, training=False):
        """
        Executes the forward pass of the Transformer block using causal self-attention.
        :param x: input sequence
        :param training: boolean indicating if we are training or inference
        :return: tf.Tensor: Processed tensor with the same shape as the input.
        """
        seq_len = tf.shape(x)[1]

        # Create a lower-triangular causal mask to prevent the model
        # from attending to future tokens during training/inference.
        i = tf.range(seq_len)[:, tf.newaxis]
        j = tf.range(seq_len)
        mask = i >= j
        mask = tf.reshape(mask, (1, seq_len, seq_len))

        # Multi-head self-attention with residual connection and normalization
        attn_output = self.mha(query=x, value=x, key=x, attention_mask=mask, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layer_norm1(x + attn_output)

        # Feed-forward network with residual connection and normalization
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layer_norm2(out1 + ffn_output)


