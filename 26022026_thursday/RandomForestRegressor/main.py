import pandas as pd
import seaborn as sns
import logging
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

logging.basicConfig(filename='logs.log',level=logging.INFO,filemode='a',format = '%(name)s - %(levelname)s - %(message)s')

seperator = f"\n{'--'*70}\n"
class Regressor:
    """
        A machine learning pipeline for flight price prediction using Random Forest.

        This class handles the end-to-end process including data loading,
        preprocessing, exploratory data analysis with outlier clipping,
        categorical encoding, and model evaluation.

        Attributes:
            filepath (str): Path to the CSV dataset.
            df (pd.DataFrame): The loaded dataset.
            model (RandomForestRegressor): The underlying regressor model.
            regressor (Pipeline): The final Scikit-learn pipeline object.
        """
    def __init__(self):
        """Initializes the Regressor with default parameters and model settings."""
        self.filepath = "flight_price_prediction.csv"
        self.df = None
        self.encoder = OneHotEncoder(handle_unknown='ignore',drop="first")
        self.preprocessing = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.test_size = 0.2
        self.randomstate = 42
        self.model = RandomForestRegressor(n_estimators=25,ccp_alpha=0.001,min_samples_split=5,max_depth=10,verbose=2)
        self.regressor = None
        self.y_prediction = None

    def load_dataset(self):
        """
           Loads the dataset from the CSV file specified in self.filepath.

           Raises:
               FileNotFoundError: If the CSV file is missing from the directory.
        """
        try:
            self.df = pd.read_csv(self.filepath)
            print(self.df.head(),end=seperator)
            logging.info("Dataset Loaded")
        except FileNotFoundError:
            print("File not found. Please check the path and try again.")
            logging.error("File not found")

    def data_preprocessing(self):
        """
        Cleans the dataset by removing irrelevant columns and checking for duplicates.
        Logs the shape of the data and identifies duplicate records.
        """
        logging.info("Data Preprocessing Started")
        print(f"Dataset shape:- {self.df.shape}")
        print(f"\nDuplicate Records:- {self.df.duplicated().sum()}")
        print("\nData Statistics")
        print(self.df.describe())
        print("\nData Description")
        self.df.info()
        self.df.drop(self.df.columns[0], axis=1, inplace=True)
        self.df.drop("flight", axis=1, inplace=True)
        print("\nDropped ID and flight Column")
        logging.info("Data Preprocessing Completed")

    def eda(self):
        """
           Performs Exploratory Data Analysis and Outlier Handling.

           Visualizes correlations via a heatmap and handles outliers in numeric
           columns by clipping them to the 1.5 * IQR (Interquartile Range) bounds.
        """

        logging.info("EDA Started")
        corr = self.df.corr(numeric_only=True)
        sns.heatmap(corr, annot=True, cmap="rocket")
        plt.title("Correlation Matrix",fontsize = 30,color = "red",fontweight = "bold")
        plt.show()

        categorical_cols = self.df.select_dtypes(include=["str"]).columns
        plt.figure(figsize=(18, 10))
        plt.title("Count-Plot", weight='bold', fontsize=20, color='Red')
        for i, cols in enumerate(categorical_cols):
            plt.subplot(4, 2, i + 1)
            sns.countplot(self.df[cols])
            plt.xlabel(cols, weight='semibold', fontsize=12, color='black')
        plt.tight_layout()
        plt.show()

        numeric_cols = self.df.select_dtypes(include=["int64","float64"]).columns.drop("price")
        print("\nOutliers:-")
        for i,cols in enumerate(numeric_cols):
            plt.subplot(2,2,i+1)
            sns.boxplot(self.df[cols])
            plt.title(cols,fontsize = 20,color = "red",fontweight = "bold")
            q1 = self.df[cols].quantile(0.25)
            q3 = self.df[cols].quantile(0.75)
            iqr = q3-q1
            lb = q1-1.5*iqr
            ub = q3+1.5*iqr
            outliers = self.df[(self.df[cols]<lb) | (self.df[cols]>ub)]
            print(f"{cols} ---> {len(outliers)}")
            self.df[cols] = self.df[cols].clip(lower=lb, upper=ub)
        plt.tight_layout()
        plt.show()
        print("\nOutlier Removed Successfully",end=seperator)
        logging.info("EDA Completed")

    def encoding(self):
        """
            Initializes the ColumnTransformer for categorical data.

            Identifies object-type columns and applies OneHotEncoding while
            handling unknown categories during transform.
        """
        logging.info("Encoder Pipline Started")
        cat_cols = self.df.select_dtypes(include=["str"]).columns
        self.preprocessing = ColumnTransformer(transformers=[("categorical",self.encoder,cat_cols)],remainder="passthrough")
        print("Encoding Pipeline Created",end=seperator)
        logging.info("Encoder Pipline Completed")

    def train_test_split(self):
        """
           Splits the dataframe into training and testing feature sets and targets.

           Uses self.test_size and self.randomstate for reproducibility.
        """
        logging.info("train_test_split Started")
        self.X = self.df.drop(columns=["price"])
        self.y = self.df["price"]
        self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.X,self.y,test_size=self.test_size,random_state=self.randomstate)
        print("Train Test Split Completed",end=seperator)
        logging.info("Train Test Split Completed")

    def model_training(self):
        """
            Constructs and fits the Scikit-learn Pipeline.

            The pipeline chains the preprocessing (encoding) step and the
            RandomForestRegressor model.
        """
        logging.info("Model Training Started")
        self.regressor = Pipeline(steps = [("preprocessor",self.preprocessing),("model",self.model)])
        self.regressor.fit(self.X_train,self.y_train)
        print("Model Training Completed",end=seperator)
        logging.info("Model Training Completed")

    def model_evaluation(self):
        """
            Evaluates the trained model on the test dataset.

            Prints and logs the R2 Score, Mean Absolute Error (MAE),
            and Mean Squared Error (MSE).
        """
        logging.info("Model Evaluation Started")
        self.y_prediction = self.regressor.predict(self.X_test)
        print(f"R2_Score:- {r2_score(self.y_test,self.y_prediction)*100:.2f}%")
        print(f"Mean Absolute Error:- {mean_absolute_error(self.y_test,self.y_prediction)}")
        print(f"Mean Squared Error:- {mean_squared_error(self.y_test,self.y_prediction)}")
        print("Model Evaluation Completed",end=seperator)
        logging.info("Model Evaluation Completed")

def main():
    reg = Regressor()
    reg.load_dataset()
    reg.data_preprocessing()
    reg.eda()
    reg.encoding()
    reg.train_test_split()
    reg.model_training()
    reg.model_evaluation()

if __name__ == "__main__":
    main()