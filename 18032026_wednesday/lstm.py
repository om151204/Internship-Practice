import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,LSTM,Dense

sep = f"\n{'---'*70}\n\n"

class SentimentLSTM:
    """
    A class to perform sentiment analysis using an LSTM network.
    """
    def __init__(self):
        """
        Initializes the model parameters and tokenizer.
        """
        self.df = None
        self.tokenizer = Tokenizer()
        self.X = None
        self.y = None
        self.model = None
        self.max_len = 5

    def load_data(self):
        """
        Creates a sample dataset and loads it into a pandas DataFrame.
        :return: None
        """
        try:
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
            print("\nData Loaded Successfully!!",end=sep)
        except Exception as e:
            print(f"Error in loading data {e}")

    def preprocess_data(self):
        """
        Tokenizes text and pads sequences for model input.
        :return: None
        """
        try:
            self.load_data()
            texts = self.df["Review"]
            self.y = self.df["Sentiment"]

            self.tokenizer.fit_on_texts(texts)

            sequences = self.tokenizer.texts_to_sequences(texts)

            self.X = pad_sequences(sequences, maxlen=self.max_len)
            print(self.X.shape)
            print(self.y.shape)
            print("\nData Preprocessed Successfully!!",end=sep)
        except Exception as e:
            print(f"Error in preprocessing data {e}")

    def build_model(self):
        """
        Constructs the LSTM model architecture.
        :return: None
        """
        try:
            self.preprocess_data()
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
            print("\nModel Built Successfully!!",end=sep)
        except Exception as e:
            print(f"Error in building model {e}")

    def train_model(self):
        """
        Trains the model on the preprocessed data.
        :return: None
        """
        try:
            self.build_model()
            self.model.fit(self.X, self.y, epochs=20, verbose=1)
            print("\nModel Trained Successfully!!",end=sep)
        except Exception as e:
            print(f"Error in training model {e}")

    def predict(self):
        """
        Predicts the sentiment of a given list of string.
        :return: None
        """
        try:
            self.train_model()
            test_text = ["I really love this"]

            seq = self.tokenizer.texts_to_sequences(test_text)
            padded = pad_sequences(seq, maxlen=self.max_len)

            prediction = self.model.predict(padded)
            print(f"\nPrediction:- {prediction}")

            if prediction > 0.5:
                print("Positive Sentiment")
            else:
                print("Negative Sentiment")
            print("\nModel Prediction Completed!!",end=sep)
        except Exception as e:
            print(f"Error in prediction {e}")

def main():
    """
    Orchestrates the SentimentLSTM pipeline: initializes the class,
    loads and preprocesses data, builds the model, trains it,
    and performs sentiment prediction on sample text.
    :return: None
    """
    obj = SentimentLSTM()
    obj.predict()

if __name__ == "__main__":
    main()
