import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

sep = f"\n{'---' * 60}\n\n"

class SimpleRNNModel:
    def __init__(self):
        """
        Initializes the dataset (sentences and binary labels).
        1 = Positive Sentiment, 0 = Negative Sentiment.
        """
        self.sentences = [
            "movie was good", "movie was bad",
            "i like this film", "i hate this film",
            "this movie is amazing", "this movie is terrible",
            "film was nice", "film was boring",
            "good acting", "bad acting"
        ]

        self.labels = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

        # Initialize Tokenizer and placeholders for data and model
        self.tokenizer = Tokenizer()
        self.padded_sequences = None
        self.model = None

    def prepare_data(self):
        """
        Converts raw text into numeric sequences and pads them to a uniform length.
        """
        try:
            # Create a vocabulary index based on word frequency
            self.tokenizer.fit_on_texts(self.sentences)

            # Convert sentences into lists of integers (sequences)
            sequences = self.tokenizer.texts_to_sequences(self.sentences)

            # Pad sequences with zeros at the end so all inputs have the same length
            self.padded_sequences = pad_sequences(sequences, padding="post")

            print("Word Index (Vocabulary):-")
            print(self.tokenizer.word_index)

            print("\nInteger Sequences:-")
            print(sequences)

            print("\nPadded Sequences (Input Matrix):-")
            print(self.padded_sequences)
            print("\nData Preparation Completed!!", end=sep)
        except Exception as e:
            print(f"Error in preparing data: {e}", end=sep)

    def build_model(self):
        """
        Defines the Neural Network architecture: Embedding -> RNN -> Dense.
        """
        try:
            self.prepare_data()
            # vocab_size is total words + 1 (to account for the 0-padding index)
            vocab_size = len(self.tokenizer.word_index) + 1

            self.model = Sequential([
                # Turns integers into dense vectors of fixed size (8)
                Embedding(input_dim=vocab_size, output_dim=8),

                # Simple Recurrent layer to process sequence data with 16 units
                SimpleRNN(16),

                # Output layer with Sigmoid for binary classification (0 to 1)
                Dense(1, activation="sigmoid")
            ])

            # Compile with Adam optimizer and Binary Cross-entropy loss
            self.model.compile(
                optimizer="adam",
                loss="binary_crossentropy",
                metrics=["accuracy"]
            )

            self.model.summary()
            print("Model built successfully!!", end=sep)
        except Exception as e:
            print(f"Error in building model: {e}", end=sep)

    def train_model(self):
        """
        Trains the model on the prepared padded sequences.
        """
        try:
            self.build_model()
            # Training the model for 20 iterations over the dataset
            self.model.fit(
                self.padded_sequences,
                self.labels,
                epochs=20,
                verbose=1
            )
            print("Training Model Completed!!", end=sep)
        except Exception as e:
            print(f"Error in training model: {e}", end=sep)

    def prediction(self):
        """
        Preprocesses a new test string and predicts its sentiment.
        """
        try:
            self.train_model()
            test = ["movie was amazing"]

            # Transform test text using the SAME tokenizer used for training
            seq = self.tokenizer.texts_to_sequences(test)

            # Pad to match the exact input length the model expects
            padded = pad_sequences(seq, padding="post",
                                   maxlen=self.padded_sequences.shape[1])

            # Generate prediction (a value between 0 and 1)
            prediction = self.model.predict(padded)

            print(f"Prediction Score:- {prediction[0][0]:.4f}")

            # Threshold of 0.5 to determine class
            if prediction > 0.5:
                print("Sentiment: Positive")
            else:
                print("Sentiment: Negative")
            print("Model Prediction Completed!!", end=sep)
        except Exception as e:
            print(f"Error in prediction: {e}", end=sep)


def main():
    """
    Entry point to run the sentiment analysis pipeline.
    """
    rnn = SimpleRNNModel()
    rnn.prediction()


if __name__ == "__main__":
    main()
