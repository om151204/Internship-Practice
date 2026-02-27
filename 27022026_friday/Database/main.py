from sqlalchemy import create_engine,text
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.tree import plot_tree

separator = f"\n\n{'--'*70}\n\n"

class MSSQLDataLoading:
    def __init__(self,server,database,driver ="ODBC+Driver+17+for+SQL+Server"):
        """
        Initialize database connection
        """
        connection_string = (
            f"mssql+pyodbc://{server}/{database}?driver={driver}"
        )
        self.df = None
        try:
            self.engine = create_engine(connection_string)
            print("Engine created successfully!",end=separator)
        except ConnectionError:
            print("Connection Error")

    def load_table(self, table_name):
        """
        This method loads data from MSSQL database
        return: df : dataframe created
        """
        try:
            query = f"SELECT * FROM {table_name}"
            self.df = pd.read_sql(query, self.engine)
            print("Data Loaded Successfully",end=separator)
            return self.df,self.engine
        except FileNotFoundError:
            print("Table not found")

def save_predictions(table_name,engine,df, predictions):
    """
    This function saves model predictions to the table
    :param table_name: str
    :param engine: engine object
    :param df: dataframe
    :param predictions: model predictions
    :return: None
    """

    df["predicted_class"] = predictions
    df.to_sql(table_name,engine, if_exists="replace", index=False) # Save back to database
    print("Predictions column added/updated successfully")



class Titanic:
    def __init__(self,df):
        self.df = df
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.tree = None

    def data_preprocessing(self):
        """
        This method preprocess the data and drops unnecessary columns
        :return: None
        """

        print("Data Statistics\n")
        print(self.df.describe(), end=separator)
        print("Data Description\n")
        print(self.df.info(),end=separator)
        self.df.drop(columns=["Name","Ticket","PassengerId"],inplace=True)
        print("Dropped irrelevant from the dataset\n")
        print(self.df.head(),end=separator)
        print("Finding Null Values in each column\n")
        print(self.df.isnull().sum(),end=separator)
        self.df.drop("Cabin",axis=1,inplace=True)
        print("Dropped Cabin as it has many null values\n")
        print(self.df.head(),end=separator)
        median_age = self.df["Age"].median()
        mode_embarked = self.df["Embarked"].mode()[0]
        self.df["Age"] = self.df["Age"].fillna(value=median_age)
        self.df["Embarked"] = self.df["Embarked"].fillna(value=mode_embarked)
        print("Imputed Age Column with Median Age and Embarked Column with Mode\n")
        print(self.df.isnull().sum(),end=separator)


    def feature_engineering(self):
        """
        Created new feature from the existing features
        :return: None
        """

        self.df["FamilySize"] = self.df["SibSp"] + self.df["Parch"] + 1
        self.df.drop(columns=["SibSp","Parch"],inplace=True)
        print("Feature Engineering Done\n")
        print(self.df.head(),end=separator)

        corr = self.df.corr(numeric_only=True)
        sns.heatmap(corr,annot=True)
        plt.show()

        sns.pairplot(self.df,hue="Survived")
        plt.show()


    def outliers(self):
        """
        This method detects outlier and display the number of outliers in the dataset
        :return: None
        """

        print("Outlier Detection\n")
        cols = [col for col in self.df if self.df[col].dtypes == "int64" or self.df[col].dtypes == "float64"]
        for i, col in enumerate(cols):
            plt.subplot(3,2,i+1)
            sns.boxplot(x=col,data=self.df)
            plt.title(col)
        plt.tight_layout()
        plt.show()
        print(self.df["Age"].describe(),end=separator)
        print(self.df["Fare"].describe(),end=separator)

    def splitting(self):
        """
        This method splits the dataset into train and test and separating independent and dependent variables
        :return :None
        """

        self.X = self.df[[cols for cols in self.df if cols != "Survived"]]
        self.y = self.df[["Survived"]]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.y,test_size=0.3,random_state=42)
        print("Train Test Split Done",end=separator)

    def encoding(self):
        """
        This method encodes the categorical columns into numerical columns
        :return: None
        """

        encoder = LabelEncoder()
        categorical_cols = [col for col in self.X_train.columns if self.X_train[col].dtypes == "str"]
        for i in categorical_cols:
            self.X_train[i] = encoder.fit_transform(self.X_train[i])
            self.X_test[i] = encoder.transform(self.X_test[i])
        print("Encoding Done",end=separator)

    def model_training(self):
        """
        This method performs model training and display the plot of the tree created during model training
        :return: None
        """

        self.tree = DecisionTreeClassifier(max_depth=12,max_leaf_nodes=5,min_samples_split=4,ccp_alpha=0.01,random_state=42)
        self.tree.fit(self.X_train,self.y_train)
        print("Model Training Done",end=separator)
        plot_tree(self.tree)
        plt.title("Decision Tree Model")
        plt.show()

    def model_testing(self):
        """
        This method evaluates the model and displays the report
        :return: y_prediction
        """

        y_prediction = self.tree.predict(self.X_test)
        print("Accuracy Metrics\n")
        print(accuracy_score(self.y_test,y_prediction))
        print("Confusion Matrix\n")
        print(confusion_matrix(self.y_test,y_prediction))
        print("\nClassification Report\n")
        print(classification_report(self.y_test,y_prediction),end=separator)

        return y_prediction


def main():
    server = "localhost"
    database = "om"
    loader = MSSQLDataLoading(server, database)   # Initialize loader
    df,engine = loader.load_table("titanic_dataset")

    model = Titanic(df)
    model.data_preprocessing()
    model.feature_engineering()
    model.outliers()
    model.splitting()
    model.encoding()
    model.model_training()
    y_prediction = model.model_testing()
    model_prediction = ["survived" if pred == False else "not survived" for pred in y_prediction]

    full_x = model.X.copy()

    # Handle encoding for full_X if not already done
    encoder = LabelEncoder()
    for col in full_x.select_dtypes(include='str').columns:
        full_x[col] = encoder.fit_transform(full_x[col].astype(str))

    # Generate predictions for ALL rows
    all_predictions_raw = model.tree.predict(full_x)

    # Map the labels
    all_predictions_mapped = ["survived" if prediction == 1 else "not survived" for prediction in all_predictions_raw]

    # Save back to database
    save_predictions("titanic_dataset", engine, df, all_predictions_mapped)


if __name__ == "__main__":
    main()