import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


class HeartDiseasePipline:
    def __init__(self,filepath):
        """
        Initialization for basic variables
        :param filepath:
        """
        self.filepath = filepath
        self.df = None
        self.X = None
        self.Y = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.scalar = None

    def load_dataset(self):
        """
        Loads dataset from csv file
        :return: None
        """
        try:
            self.df = pd.read_csv(self.filepath)
            print("Dataset loaded successfully")
        except FileNotFoundError:
            print("Dataset not found")

    def perform_eda(self):
        """
        Checks shape,Description,Duplicate and null values
        :return: None
        """
        try:
            print(f"\nFirst Five Rows:-\n {self.df.head(5)}")
            print(f"\nDataset Info:- \n")
            print(self.df.info())
            print(f"\nDataset Shape:- {self.df.shape}")
            print(f"\nDataset Description:-\n {self.df.describe()}")
            print(f"\nFinding all null values:- \n{self.df.isnull().sum()}")
            print(f"\nDuplicate values:- {self.df.duplicated().sum()}")
        except Exception:
            print("Error performing eda")

    def fixing_dataset(self):
        """
        This method fixes the dataset liking handling missing values
        :return: None
        """
        try:
            self.df.fillna(self.df.median(numeric_only = True), inplace=True)
            categorical_columns = self.df.select_dtypes(include=['str']).columns
            for col in categorical_columns:
                mode_value = self.df[col].mode()
                if not mode_value.empty:
                    self.df[col] = self.df[col].fillna(mode_value[0])
            print(f"\nFinding all null values:- \n{self.df.isnull().sum()}\n")
        except Exception:
            print("Error fixing dataset")

    def detecting_outliers(self):
        """
        This method detects outliers in the dataset
        :return:
        """
        numeric_columns = self.df.select_dtypes(include=['float64']).columns
        for i,col in enumerate(numeric_columns):
            plt.subplot(5,4,i+1)
            sns.boxplot(y=self.df[col])
            plt.title(col)
        plt.show()
        corr = self.df[numeric_columns].corr()
        sns.heatmap(corr, annot=True)
        plt.show()

        for col in numeric_columns:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = self.df[(self.df[col] < lower) | (self.df[col] > upper)]
            print(f"{col} --> {len(outliers)} outliers")

    def encoding_target(self):
        """
        This method encodes the target variable
        :return: None
        """
        self.df["Heart Disease Status"] = self.df["Heart Disease Status"].map({"Yes":1,"No":0})

    def train_test_split(self):
        """
        This method splits the dataset into train and test
        :return: None
        """
        self.X = self.df.drop("Heart Disease Status",axis=1)
        self.Y = self.df["Heart Disease Status"]
        self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.X,self.Y,test_size=0.3,random_state=42)

    def encoding_standard_pipeline(self):
        """
        This method creates pipeline for the numerical and categorical columns
        :return: preprocessor object
        """
        numerical_cols = self.X.select_dtypes(include=['float64']).columns
        categorical_cols = self.X.select_dtypes(include=['str']).columns

        numeric_pipeline = Pipeline(steps =[("scalar",StandardScaler())])
        categorical_pipeline = Pipeline(steps =[("encoder",OneHotEncoder(drop="first"))])
        preprocessor = ColumnTransformer(transformers=[("num",numeric_pipeline,numerical_cols),("cat",categorical_pipeline,categorical_cols)])
        return preprocessor

    def training_model(self,transformer):
        """
        This method trains the model
        :param transformer: preprocessor object
        :return: model pipeline
        """
        model_pipeline = Pipeline(steps =[("preprocessor",transformer),("classifier",DecisionTreeClassifier( max_depth=None,min_samples_split=6,class_weight="balanced",random_state=42))])
        model_pipeline.fit(self.X_train,self.y_train)
        print("\nModel Trained Successfully")
        return model_pipeline

    def model_evaluation(self,model_pipeline):
        """
        This method evaluates the model.
        :param model_pipeline: Model Object
        :return: Nonemain.py
        """
        y_pred = model_pipeline.predict(self.X_test)
        print("\nAccuracy:")
        print(accuracy_score(self.y_test,y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(self.y_test,y_pred))
        print("\nClassification Report:")
        print(classification_report(self.y_test,y_pred))



if __name__ == "__main__":
    obj1 = HeartDiseasePipline("../23022026_monday/logistic_regression/heart_disease.csv")
    obj1.load_dataset()
    obj1.perform_eda()
    obj1.fixing_dataset()
    obj1.detecting_outliers()
    obj1.encoding_target()
    obj1.train_test_split()
    pre = obj1.encoding_standard_pipeline()
    model = obj1.training_model(pre)
    obj1.model_evaluation(model)

