import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.evaluate import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

sep = f"\n{'--'*80}\n\n"

class TitanicDeepLearning:
    """
    A deep learning pipeline for predicting Titanic passenger survival.
    """
    def __init__(self):
        """
        Loads the Titanic dataset from a CSV file.
        """
        self.df = None
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown="ignore")
        self.preprocessing_pipeline = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.test_size = 0.2
        self.random_state = 42
        self.model = None
        self.input_shape = None #  Calculation: 4(numeric) + 2(Sex) + 3(Embarked) + 1(Possible LeftOver or Bias)
        self.history = None

    def load_data(self):
        """
        Loads the Titanic dataset from a CSV file.
        :return: None
        """
        try:
            self.df = pd.read_csv("Titanic-Dataset.csv")
            print("Data set Loaded Successfully\n")
            print(self.df.head(),end=sep)
        except FileNotFoundError:
            print("File not found please check the filepath",end=sep)
        except Exception as e:
            print("Error file loading the dataset",e,end=sep)

    def preprocess_data(self):
        """
        Cleans data, handles missing values, and performs feature engineering.
        :return: None
        """
        try:
            self.load_data()
            print("Dataset Info\n")
            print(self.df.describe())
            print("\nDataset Description\n")
            self.df.info()
            self.df.drop(columns=["Name","Ticket","PassengerId","Cabin"],inplace=True) # Dropping Irrelevant Columns
            print("\nChecking Null Values\n")
            print(self.df.isnull().sum())
            self.df["Age"] = self.df["Age"].fillna(self.df["Age"].median())# Handling missing values in the Age Column
            # Handling missing values in Embarked Column
            self.df["Embarked"] = self.df["Embarked"].fillna(self.df["Embarked"].mode()[0])
            # Feature Engineering
            self.df["FamilySize"] = self.df["SibSp"] + self.df["Parch"] + 1
            self.df.drop(columns=["SibSp","Parch"],inplace=True)
            print(f"\nChecking Duplicate Values {self.df.duplicated().sum()}")
            print("\nPreprocessing Completed",end=sep)
        except Exception as e:
            print("Error while preprocessing dataset",e,end=sep)

    def eda(self):
        """
        Generates a correlation heatmap to visualize feature relationships.
        :return: None
        """
        try:
            self.preprocess_data()
            corr = self.df.corr(numeric_only=True)
            sns.heatmap(corr,annot=True,cmap="YlGnBu")
            plt.title("Correlation Heatmap",fontsize=24,fontweight="bold",color="Blue")
            plt.show()
        except Exception as e:
            print("Error while performing eda",e,end=sep)

    def build_pipeline(self):
        """
        Initializes the ColumnTransformer for scaling and encoding features.
        :return: None
        """
        try:
            self.eda()
            numeric_features = ["Age","Fare","Pclass","FamilySize"]
            categorical_features = ["Embarked","Sex"]

            numeric_pipeline = Pipeline([("scaler",self.scaler)])
            categorical_pipeline = Pipeline([("encoder",self.encoder)])
            self.preprocessing_pipeline = ColumnTransformer([("numeric",numeric_pipeline,numeric_features),
                                                             ("categorical",categorical_pipeline,categorical_features),
                                                             ],remainder = "passthrough")
            print("Preprocessing Pipeline Created",end=sep)
        except BrokenPipeError as e:
            print("BrokenPipeError",e,end=sep)
        except Exception as e:
            print("Error in creating preprocessing pipeline",e,end=sep)

    def train_test_split(self):
        """
        Splits the dataset into stratified training and testing sets.
        :return: None
        """
        try:
            self.build_pipeline()
            self.X = self.df.drop(columns="Survived")
            self.y = self.df["Survived"]
            self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(
                self.X,self.y,test_size=self.test_size,random_state=self.random_state,stratify=self.y)
            print("Train test split Completed",end=sep)
        except Exception as e:
            print("Error in train_test_split",e,end=sep)

    def apply_pipeline(self):
        """
        Fits and transforms the features using the preprocessing pipeline.
        :return: None
        """
        try:
            self.train_test_split()
            self.X_train = self.preprocessing_pipeline.fit_transform(self.X_train)
            self.X_test = self.preprocessing_pipeline.transform(self.X_test)
            print("Pipeline Applied Successfully to training data",end=sep)
        except Exception as e:
            print("Error in applying pipeline to training data",e,end=sep)

    def build_model(self):
        """
        Defines and compiles the Sequential neural network architecture.
        :return:None
        """
        try:
            self.apply_pipeline()
            self.input_shape = self.X_train.shape[1]
            self.model = Sequential([
                Dense(32,activation="relu",input_shape=(self.input_shape,)),
                Dense(16,activation="relu"),
                Dense(1,activation="sigmoid"),
            ])

            self.model.compile(
                optimizer="adam",
                loss="binary_crossentropy",
                metrics=["accuracy"])
            print("Building Model Completed",end=sep)
        except Exception as e:
            print("Error in building model",e,end=sep)

    def train_model(self):
        """
        Trains the model using EarlyStopping to prevent overfitting.
        :return:None
        """
        try:
            self.build_model()
            early_stop = EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
            )

            self.history = self.model.fit(
                self.X_train,
                self.y_train,
                batch_size=16,
                epochs=50,
                validation_data=(self.X_test,self.y_test),
                callbacks=[early_stop]
            )
            print("Model Training Completed",end=sep)
        except Exception as e:
            print("Error in training model",e,end=sep)

    def evaluate_model(self):
        """
        Predicts test outcomes and prints performance metrics.
        :return:None
        """
        try:
            self.train_model()
            predictions = self.model.predict(self.X_test)
            predictions = (predictions > 0.5).astype(int)

            print("Accuracy Score:\n")
            print(accuracy_score(self.y_test,predictions))
            print("\nConfusion Matrix:\n")
            print(confusion_matrix(self.y_test,predictions))
            print("\nClassification Report:\n")
            print(classification_report(self.y_test,predictions))
        except Exception as e:
            print("Error in evaluating model",e,end=sep)

    def plot_history(self):
        """
        Visualizes training and validation loss over epochs.
        :return:None
        """
        try:
            self.evaluate_model()
            plt.figure(figsize = (8,5))
            plt.plot(self.history.history["loss"],label="Training Loss")
            plt.plot(self.history.history["val_loss"],label="Validation Loss")
            plt.title("Loss Curve",fontsize=24,fontweight="bold")
            plt.xlabel("Epoch",fontsize=24,fontweight="bold")
            plt.ylabel("Loss",fontsize=24,fontweight="bold")
            plt.legend()
            plt.show()
        except Exception as e:
            print("Error in plotting history",e,end=sep)

def main():
    """
    Orchestrates the complete machine learning workflow:
    1. Instantiates the TitanicDeepLearning class.
    2. Executes the pipeline from data loading to model training.
    3. Displays evaluation metrics and loss curves.
    :return: None
    """
    preprocessor = TitanicDeepLearning()
    preprocessor.plot_history()

if __name__ == "__main__":
    main()




