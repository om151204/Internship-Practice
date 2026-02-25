import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error


seperator = f"\n{"--"*70}\n"
class Regressor:
    def __init__(self):
        self.df = None
        self.numeric_features = []
        self.label_encoder_features = []
        self.onehot_encoder_features = []
        self.preprocessor = None
        self.le = OrdinalEncoder()
        self.ohe = OneHotEncoder(drop="first")
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.test_size = 0.7
        self.random_state = 42
        self.model = DecisionTreeRegressor(max_depth=5,min_samples_leaf=10,ccp_alpha=0.01)
        self.pipeline = None
        self.y_prediction = None


    def load_dataset(self):
        self.df = pd.read_csv('../../20022026_friday/data.csv')
        print(self.df.head())
        print("\nDataset Loaded Successfully!",end=seperator)

    def data_preprocessing(self):
        print("Statistics of the Dataset")
        print(self.df.describe(),end=seperator)
        print("Description of the Dataset")
        print(self.df.info(),end=seperator)
        print("Checking Null Values")
        print(self.df.isnull().sum(),end=seperator)
        print(f"Checking Duplicate Values: {self.df.duplicated().sum()}")
        self.df.drop_duplicates(keep='first',inplace=True)
        print(f"Dropped the duplicate value: {self.df.duplicated().sum()}", end=seperator)

    def eda(self):
        corr = self.df.corr(numeric_only=True)  # Correlation Matrix
        sns.heatmap(corr, annot=True)
        plt.show()

        self.numeric_features = [cols for cols in self.df.columns if self.df[cols].dtype in ['int64','float64']]
        for i,cols in enumerate(self.numeric_features):   # Hist Plot
            plt.subplot(2,2,i+1)
            sns.histplot(self.df[cols],bins=50,kde=True)
            plt.title(f'Distribution of {cols}')
        plt.tight_layout()
        plt.show()

        for i, col in enumerate(self.numeric_features):
            plt.subplot(2, 2, i + 1)
            sns.boxplot(y=self.df[col])       # Box Plot
            plt.title(f'Boxplot of {col}')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def outliers(self):
        print("Outlier Count")
        for col in self.numeric_features:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            lb = q1 - 1.5*iqr
            ub = q3 + 1.5*iqr
            outliers = self.df[(self.df[col]<lb) | (self.df[col]>ub)]
            print(f"{col} --> {len(outliers)}")
        print("Outlier Detection Completed",end=seperator)

    def encoder(self):
        self.label_encoder_features = list(
        self.df.select_dtypes(include=['str']).columns.drop('region', errors='ignore'))
        self.onehot_encoder_features = ['region']
        self.preprocessor = ColumnTransformer(transformers=[("cat1",self.le,self.label_encoder_features),("cat2",self.ohe,self.onehot_encoder_features)],remainder='passthrough')
        print("Encoding Done Successfully!",end=seperator)

    def train_test_split(self):
        self.X = self.df.drop('charges',axis=1)
        self.y = self.df['charges']
        self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.X,self.y,test_size=self.test_size,random_state=self.random_state)
        print(f"X_train shape: {self.X_train.shape} and y_train shape: {self.y_train.shape}")
        print("Train Test Split Done Successfully!",end=seperator)

    def model_training(self):
        self.pipeline = Pipeline(steps=[("preprocessor",self.preprocessor),("model",self.model)])
        self.pipeline.fit(self.X_train,self.y_train)
        print("Model Training Completed!",end=seperator)

    def performance_evaluation(self):
        self.y_prediction = self.pipeline.predict(self.X_test)
        print(f"Mean Squared Error: {mean_squared_error(self.y_test,self.y_prediction)}")
        print(f"Mean Absolute Error: {mean_absolute_error(self.y_test,self.y_prediction)}")
        print(f"R2 Score: {r2_score(self.y_test,self.y_prediction)}",end=seperator)

def main():
    obj = Regressor()
    obj.load_dataset()
    obj.data_preprocessing()
    obj.eda()
    obj.outliers()
    obj.encoder()
    obj.train_test_split()
    obj.model_training()
    obj.performance_evaluation()

if __name__ == "__main__":
    main()

