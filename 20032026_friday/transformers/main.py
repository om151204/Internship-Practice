import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense,Embedding,LayerNormalization

sep = f"\n{'--'*70}\n\n"

class DataLoader:
    def __init__(self,sentences):
        self.sentences = sentences
        self.tokenizer = Tokenizer(filters = "")

    def load_data(self):
        """
        This method loads the data and performs tokenization on it
        :return: padded: Pad sequences of data
        """
        try:
            self.tokenizer.fit_on_texts(self.sentences)

            sequences = self.tokenizer.texts_to_sequences(self.sentences)

            padded = pad_sequences(sequences, padding="post")
            print("Padded Sequence:-\n")
            print(padded)
            print("Data Loaded and Preprocessed Successfully!!",end = sep)
            return padded
        except Exception as e:
            print(f"Error Loading Data {e}",end = sep)

    def get_vocab_size(self):
        """
        This method returns the size of the vocabulary
        :return: Size of the vocabulary
        """
        try:
            print("Vocabulary Size :-",len(self.tokenizer.word_index) + 1)
            return len(self.tokenizer.word_index) + 1
        except Exception as e:
            print(f"Error Returning Vocabulary size {e}",end = sep)

