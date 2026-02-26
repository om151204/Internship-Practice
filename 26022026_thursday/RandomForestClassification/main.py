import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import warnings
warnings.filterwarnings("ignore")

separator = f"\n{"--"*70}\n"
class Classification:
    """
        A class to handle the end-to-end machine learning workflow for hotel booking classification.

        Includes data loading, preprocessing, feature engineering, EDA, and model evaluation.
    """
    def __init__(self):
        """Initializes the Classification class with default parameters and placeholders."""
        self.df = None
        self.ohe = OneHotEncoder(handle_unknown='ignore',drop="first")
        self.preprocessor = None
        self.train_dataset = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.test_size = 0.3
        self.randomstate = 43
        self.pipeline = None
        self.model = RandomForestClassifier()
        self.y_prediction = None

    def load_dataset(self):
        """Loads the hotel_bookings.csv dataset from the local directory."""
        try:
            self.df = pd.read_csv("hotel_bookings.csv")
            print(self.df.head())
            print("Dataset Loaded Successfully",end=separator)
        except FileNotFoundError:
            print("Dataset Not Found",end=separator)

    def data_preprocessing(self):
        """Cleans the dataset by dropping irrelevant columns, null values, and duplicates."""
        print(f"\nSize of the data: {self.df.shape}")
        print("\nData Description\n")
        print(self.df.describe())
        print("\nData Statistics\n")
        print(self.df.info())
        print("\nFinding Null Values\n")
        print(self.df.isnull().sum())
        print("\nDropped company,agent and reservation status column\n")
        self.df.drop(columns = ["company","agent","reservation_status"], inplace=True)
        self.df.dropna(inplace=True)
        print(f"Dropped Null values from dataset")
        print(f"\nRechecking for Null Values\n{self.df.isnull().sum()}\n")
        print(f"\nDuplicates in the dataset:- {self.df.duplicated().sum()}")
        self.df.drop_duplicates(inplace=True, keep="first")
        print(f"Dropped Duplicates in the dataset:- {self.df.duplicated().sum()}",end=separator)

    def feature_engineering(self):
        """Creates new features like 'total_night_stays' and 'room_mismatch' while dropping raw columns."""
        self.df["total_night_stays"] = self.df["stays_in_weekend_nights"] + self.df["stays_in_week_nights"]
        self.df["total_guests"] = self.df["adults"] + self.df["children"] + self.df["babies"]
        # Combine columns into a single string column first
        # Format will look like: "2015-July-1"
        self.df['arrival_date'] = (
                self.df['arrival_date_year'].astype(str) + '-' +
                self.df['arrival_date_month'] + '-' +
                self.df['arrival_date_day_of_month'].astype(str)
        )

        # Create a 'room_mismatch' flag:
        # 1 if the rooms are different, 0 if they are the same
        self.df['room_mismatch'] = np.where(self.df['reserved_room_type'] != self.df['assigned_room_type'], 1, 0)

        self.df.drop(columns = ["stays_in_weekend_nights","stays_in_week_nights","adults","children","babies","arrival_date_year","arrival_date_month","arrival_date_day_of_month","assigned_room_type","reserved_room_type"], inplace=True)

        print(self.df.columns)
        print("Feature Engineering Done!",end=separator)

    def eda(self):
        """Performs Exploratory Data Analysis including correlation heatmaps and box plots."""
        corr = self.df.corr(numeric_only=True)  # Correlation Matrix
        sns.heatmap(corr, annot=True,cmap='rocket')
        plt.show()

        numeric_cols = self.df.select_dtypes(include=["int64","float64"]).columns
        for i,cols in enumerate(numeric_cols):
            plt.subplot(3,5,i+1)
            sns.boxplot(self.df[cols],orient="h")
            plt.title(cols,weight = "bold",color = "red",fontsize=12)
        plt.tight_layout()
        plt.show()

    def encoder(self):
        """Initializes the ColumnTransformer for categorical feature encoding."""
        ohe_cols = self.df.select_dtypes(include=["str"]).columns
        self.preprocessor = ColumnTransformer(transformers=[("categorical",self.ohe,ohe_cols)],remainder="passthrough")
        print("Encoding Pipeline Created Successfully!",end=separator)

    def train_test_split(self):
        """Splits the data into training and testing sets."""
        independent_features = self.df.columns.to_list()
        independent_features.remove("is_canceled")
        self.X = self.df[independent_features]
        self.y = self.df["is_canceled"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.y,test_size=self.test_size,random_state = self.randomstate)
        print("Train Test Split Done!",end=separator)

    def model_training(self):
        """Builds and fits the Scikit-learn Pipeline on the training data."""
        self.pipeline = Pipeline(steps = [("preprocessor",self.preprocessor),("model",self.model)])
        self.pipeline.fit(self.X_train,self.y_train)
        print("Model Training Completed!",end=separator)

    def model_testing(self):
        """Predicts on test data and prints performance metrics."""
        self.y_prediction = self.pipeline.predict(self.X_test)
        print(f"Accuracy:- {accuracy_score(self.y_test,self.y_prediction)*100:.2f}%")
        print("\nClassification Report:-\n")
        print(classification_report(self.y_test,self.y_prediction))
        print("\nConfusion Matrix:-\n")
        print(confusion_matrix(self.y_test,self.y_prediction),end=separator)


def main():
    """Main execution block to run the classification workflow."""
    obj = Classification()
    obj.load_dataset()
    obj.data_preprocessing()
    obj.feature_engineering()
    obj.eda()
    obj.encoder()
    obj.train_test_split()
    obj.model_training()
    obj.model_testing()

if __name__ == "__main__":
    main()
