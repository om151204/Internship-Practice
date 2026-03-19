import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

sep = f"\n{'---'*70}\n\n"
class SimpleRNNModel:
    def __init__(self):
        self.sentences = [
            "movie was good",
            "movie was bad",
            "i like this film",
            "i hate this film",
            "this movie is amazing",
            "this movie is terrible",
            "film was nice",
            "film was boring",
            "good acting",
            "bad acting"
        ]

        self.labels = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

        self.tokenizer = Tokenizer()
        self.padded_sequences = None
        self.model = None

    def prepare_data(self):
        self.tokenizer.fit_on_texts(self.sentences)
        sequences = self.tokenizer.texts_to_sequences(self.sentences)

        self.padded_sequences = pad_sequences(sequences, padding="post")

        print("Word Index:-")
        print(self.tokenizer.word_index)

        print("\nSequences")
        print(sequences)

        print("\nPadded Sequences:-")
        print(self.padded_sequences)

    def build_model(self):
        self.prepare_data()
        vocab_size = len(self.tokenizer.word_index) + 1

        self.model = Sequential([
            Embedding(input_dim=vocab_size, output_dim=8),
            SimpleRNN(16),
            Dense(1, activation="sigmoid")
        ])

        self.model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        self.model.summary()

    def train_model(self):
        self.build_model()
        self.model.fit(
            self.padded_sequences,
            self.labels,
            epochs=20,
            verbose=1
        )

    def prediction(self):
        self.train_model()
        test = ["movie was amazing"]

        seq = self.tokenizer.texts_to_sequences(test)

        padded = pad_sequences(seq, padding="post",
                               maxlen=self.padded_sequences.shape[1])

        prediction = self.model.predict(padded)

        print(f"Prediction Score:- {prediction[0][0]}")

        if prediction > 0.5:
            print("Sentiment: Positive")
        else:
            print("Sentiment: Negative")


def main():
    rnn = SimpleRNNModel()
    rnn.prediction()


if __name__ == "__main__":
    main()
