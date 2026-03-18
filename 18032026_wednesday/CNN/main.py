import tensorflow as tf
from tensorflow.keras import layers,models
import matplotlib.pyplot as plt

sep = f"\n{'---' * 70}\n\n"

class SimpleCNN:
    """
    Class to train and evaluate a simple convolutional neural network on mnist dataset
    """

    def __init__(self):
        """
        Constructor that initializes required variables
        """
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.history = None

    def load_dataset(self):
        """
        Method that loads MNIST dataset
        :return: None
        """
        try:
            (self.X_train, self.y_train), (self.X_test, self.y_test) = tf.keras.datasets.mnist.load_data()

            # Normalize pixel values (0-255 -> 0-1)
            self.X_train = self.X_train / 255.0
            self.X_test = self.X_test / 255.0

            # Add channel dimension for CNN
            self.X_train = self.X_train.reshape(-1, 28, 28, 1)
            self.X_test = self.X_test.reshape(-1, 28, 28, 1)

            print(f"Training Shape:- {self.X_train.shape}")
            print(f"Testing Shape:- {self.X_test.shape}")
            print("\n Dataset Loaded Successfully!!", end=sep)
        except Exception as e:
            print(f"Error while loading dataset: {e}")

    def build_model(self):
        """
        Method that builds simple convolutional neural network
        :return: None
        """
        try:
            self.load_dataset()
            self.model = models.Sequential([
                layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
                layers.MaxPooling2D((2, 2)),

                layers.Flatten(),

                layers.Dense(64, activation='relu'),

                layers.Dense(10, activation="softmax")
            ])

            self.model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

            self.model.summary()
            print("\nModel built Successfully!!", end=sep)
        except Exception as e:
            print(f"Error while building model: {e}")

    def train_model(self):
        """
        Method that trains simple convolutional neural network
        :return: None
        """
        try:
            self.build_model()
            self.history = self.model.fit(
                self.X_train,
                self.y_train,
                epochs=5,
                validation_data=(self.X_test, self.y_test)
            )

            print("\nModel Trained Successfully!!", end=sep)
        except Exception as e:
            print(f"Error while training model: {e}")

    def evaluate_model(self):
        """
        Method that evaluates simple convolutional neural network
        :return: None
        """
        try:
            self.train_model()
            loss, accuracy = self.model.evaluate(self.X_test, self.y_test)
            print(f"Test Accuracy:- {accuracy:.2%}")
            print("\nModel Evaluated Successfully!!", end=sep)
        except Exception as e:
            print(f"Error while evaluating model: {e}")

    def plot_performance(self):
        try:
            """
            Method that plots training and validation performance
            :return: None
            """
            self.evaluate_model()
            plt.figure(figsize=(12, 4))
            # plot for accuracy
            plt.subplot(1, 2, 1)
            plt.plot(self.history.history['accuracy'], label='Train Accuracy')
            plt.plot(self.history.history['val_accuracy'], label='Val Accuracy')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            # plot for loss
            plt.subplot(1, 2, 2)
            plt.plot(self.history.history['loss'], label='Train Loss')
            plt.plot(self.history.history['val_loss'], label='Val Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()

            plt.tight_layout()
            plt.show()
            print("\nPlotting done successfully!!")
        except Exception as e:
            print(f"Error while plotting: {e}")


def main():
    """
    Function to control the flow of code
    :return:
    """
    cnn = SimpleCNN()
    cnn.plot_performance()


if __name__ == "__main__":
    main()
