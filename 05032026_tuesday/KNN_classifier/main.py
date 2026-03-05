import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.neighbors import KNeighborsClassifier
import joblib

separator = f"\n{'--'*70}\n"

class KNNClassifier:
    """
    This class implements KNN classifier.
    """
    def __init__(self,fpath):
        """
        This constructor initializes required variables
        :param fpath: path of the dataset
        """
        self.df = None
        self.filepath = fpath
        self.preprocessing = None
        self.scaler = StandardScaler()
        self.encoder = OrdinalEncoder()
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.random_state = 42
        self.test_size = 0.2
        self.model = KNeighborsClassifier(n_neighbors=5)
        self.pipeline = None


    def load_data(self):
        """
        This method loads dataset and transforms it to pandas dataframe
        :return: None
        """
        try:
            self.df = pd.read_csv(self.filepath)
            print(self.df.head(),end = separator)
        except FileNotFoundError:
            print(f"File not found: {self.filepath}")

    def data_preprocessing(self):
        """
        This method preprocess the data and drops irrelevant columns
        :return: None
        """
        try:
            self.load_data()
            print(f"Dataset Size:- {self.df.shape}\n")
            print("Statistics of the dataset:-\n")
            print(self.df.describe())
            print("\nDescription of the dataset:-\n")
            self.df.info()
            print(f"\nNull values in the dataset:-\n{self.df.isnull().sum()}\n")
            self.df.drop("user_id", axis = 1, inplace = True)
            print("Dropped id column")
            print(f"\nChecking the unique values in the dataset:-\n{self.df.nunique()}",end = separator)
        except Exception as e:
            print("Error while preprocessing data",e)


    def eda(self):
        """
        This method visualizes data distribution
        :return: None
        """
        try:
            self.data_preprocessing()
            corr = self.df.corr(numeric_only=True)
            sns.heatmap(corr, annot=True)
            plt.title("Correlation Matrix",weight = "bold",fontsize=25)
            plt.show()
            sns.pairplot(self.df,hue= 'purchased')
            plt.show()
        except Exception as e:
            print("Error while performing eda",e)

    def checking_outliers(self):
        """
        This method checks outlier in the dataset
        :return: None
        """
        try:
            self.eda()
            print("\nChecking Outliers\n")
            numeric_features = self.df.select_dtypes(include = 'int64').columns
            for i,col in enumerate(numeric_features):
                plt.subplot(2,2,i+1)
                sns.boxplot(y=self.df[col])
                plt.title(col)
            plt.show()
            for col in numeric_features:
                q1 = self.df[col].quantile(0.25)
                q3 = self.df[col].quantile(0.75)
                iqr = q3 - q1
                lb = q1 - 1.5 * iqr
                ub = q3 + 1.5 * iqr
                outliers = self.df[(self.df[col]<lb) | (self.df[col]>ub)]
                print(f"{col} ---> {len(outliers)}")
            print(end=separator)
        except Exception as e:
            print("Error while checking outliers",e)

    def pipeline_creation(self):
        """
        This method creates pipeline for preprocessing and transforms data
        :return: None
        """
        try:
            self.checking_outliers()
            numeric_features = self.df.select_dtypes(include = 'int64').columns.drop("purchased")
            self.preprocessing = ColumnTransformer(transformers = [("categorical",self.encoder,["gender"]),("numeric",self.scaler,numeric_features)])
            print("Preprocessing Pipeline created",end = separator)
        except Exception as e:
            print("Error while creating pipeline",e)

    def train_test_split(self):
        """
        This method splits dataset into train and test set
        :return: None
        """
        try:
            self.pipeline_creation()
            self.X = self.df.iloc[:,:-1]
            self.y = self.df.iloc[:,-1]
            self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.X,self.y,test_size=self.test_size,random_state=self.random_state)
            print("Train Test Split Done",end = separator)
        except Exception as e:
            print("Error while performing train test split",e)

    def model_training(self):
        """
        This method trains the model using
        :return:
        """
        try:
            self.train_test_split()
            self.pipeline = Pipeline(steps = [("preprocessing", self.preprocessing),("training",self.model)])

            param_grid = {
                "training__n_neighbors": [3, 5, 7, 9, 11],
                "training__weights": ["uniform", "distance"],
                "training__metric": ["euclidean", "manhattan", "minkowski"]
            }
            print("Starting Grid Search CV... this may take a while...\n")
            grid_search = GridSearchCV(self.pipeline,param_grid,refit=True,verbose=2,cv=5)
            grid_search.fit(self.X_train,self.y_train)
            self.pipeline = grid_search.best_estimator_
            self.model = grid_search.best_estimator_.named_steps["training"]
            print(end=separator)
            print(f"Best Parameters:{grid_search.best_params_}\n")
            print("Model Training Done",end = separator)
        except Exception as e:
            print("Error while training model training",e)

    def model_performance(self):
        """
        This method checks model performance on test data
        :return: None
        """
        try:
            self.model_training()
            y_prediction = self.pipeline.predict(self.X_test)
            print(f"Confusion Matrix\n")
            print(confusion_matrix(self.y_test,y_prediction))
            print("\nClassification Report\n")
            print(classification_report(self.y_test,y_prediction))
        except Exception as e:
            print("Error checking model performance",e)

    def save_model(self):
        """
        This method saves the model
        :return: None
        """
        self.model_performance()
        joblib.dump(self.pipeline,"knn_model.pkl")
        print("Model saved successfully",end = separator)

def main():
    """
    This function executes the end-to-end KNN Classification pipeline.
    :return: None
    """
    obj = KNNClassifier("../../03032026_tuesday/data/user_data.csv")
    obj.save_model()

if __name__ == "__main__":
    main()




