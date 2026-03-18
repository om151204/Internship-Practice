import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,LSTM,Dense


class SentimentLSTM:
    def __init__(self):
        self.df = None
        self.tokenizer = Tokenizer()
        self.X = None
        self.y = None
        self.model = None
        self.max_len = 5

    def load_data(self):
        data = {
            'Review': [
                'I love this product',
                'This is amazing',
                'Very bad experience',
                'I hate this item',
                'Excellent quality',
                'Worst purchase ever',
                'Really happy with this',
                'Not good at all',
                'Superb performance',
                'Terrible service'
            ],
            'Sentiment': [1, 1, 0, 0, 1, 0, 1, 0, 1, 0]
        }
        self.df = pd.DataFrame(data)
        print(self.df.head())

    def preprocess_data(self):
        texts = self.df["Review"]
        self.y = self.df["Sentiment"]

        self.tokenizer.fit_on_texts(texts)

        sequences = self.tokenizer.texts_to_sequences(texts)

        self.X = pad_sequences(sequences, maxlen=self.max_len)

    def build_model(self):
        vocab_size = len(self.tokenizer.word_index) + 1
        self.model = Sequential()

        self.model.add(
            Embedding(
                input_dim=vocab_size, output_dim=8, input_length=self.max_len
            ))

        self.model.add(LSTM(16))

        self.model.add(Dense(1, activation='sigmoid'))

        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        print("Model Built Successfully")

    def train_model(self):
        self.model.fit(self.X, self.y, epochs=20, verbose=1)

    def predict(self):
        test_text = ["I really love this"]

        seq = self.tokenizer.texts_to_sequences(test_text)
        padded = pad_sequences(seq, maxlen=self.max_len)

        prediction = self.model.predict(padded)
        print(f"\nPrediction:- {prediction}")

        if prediction > 0.5:
            print("Positive Sentiment")
        else:
            print("Negative Sentiment")


obj = SentimentLSTM()
obj.load_data()
obj.preprocess_data()
obj.build_model()
obj.train_model()
obj.predict()