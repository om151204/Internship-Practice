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


class MiniGPT(tf.keras.Model):
    """
    A simplified GPT-style decoder-only model for autoregressive text generation.
    """
    def __init__(self, vocab_size, d_model=64, num_heads=4, num_layers=2, max_len=100):
        """
        Initializes the MiniGPT model architecture.
        :param vocab_size: Total number of unique tokens in the vocabulary.
        :param d_model: Dimensionality of the embedding and hidden states.
        :param num_heads: Number of attention heads in the Transformer blocks.
        :param num_layers: Number of Transformer blocks to stack.
        :param max_len: Maximum sequence length supported by positional embeddings.
        """
        super().__init__()
        self.d_model = d_model
        # Learned embeddings for both token identity and token position
        self.embedding = layers.Embedding(vocab_size, d_model)
        self.pos_emb = layers.Embedding(max_len, d_model)
        # Stack of multiple Transformer blocks for deep feature extraction
        self.blocks = [TransformerBlock(d_model, num_heads, d_model * 4) for _ in range(num_layers)]
        self.dropout = layers.Dropout(0.1)
        # Final linear layer to project back to vocabulary size for word prediction
        self.final_layer = layers.Dense(vocab_size)

    def call(self, x, training=False):
        """
        Performs the forward pass to generate logits for the next token in a sequence.
        :param x: Input tensor of token indices with shape (batch_size, seq_len).
        :param training: If True, applies dropout layers.
        :return: Logits for each token in the vocabulary with shape (batch_size, seq_len, vocab_size).
        """
        seq_len = tf.shape(x)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)

        # Sum the token and position embeddings
        x = self.embedding(x) + self.pos_emb(positions)
        x = self.dropout(x, training=training)

        # Pass through the sequential stack of Transformer blocks
        for block in self.blocks:
            x = block(x, training=training)

        return self.final_layer(x)

