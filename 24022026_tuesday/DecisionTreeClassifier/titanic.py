import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

separator = f"\n{"--"*80}\n"

def load_dataset():
    """
    Loading the dataset
    :return: dataframe
    """

    df = pd.read_csv("Titanic-Dataset.csv")
    print("Dataset Loaded")
    print(df.head(),end=separator)
    return df

def data_preprocessing(df):
    """
    Data Preprocessing
    :param df: dataframe
    :return: None
    """

    print("Data Statistics\n")
    print(df.describe(),end=separator)
    print("Data Description\n")
    print(df.info(),end=separator)
    df.drop(columns=["Name","Ticket","PassengerId"],inplace=True)
    print("Dropped irrelevant from the dataset\n")
    print(df.head(),end=separator)
    print("Finding Null Values in each column\n")
    print(df.isnull().sum(),end=separator)
    df.drop("Cabin",axis=1,inplace=True)
    print("Dropped Cabin as it has many null values\n")
    print(df.head(),end=separator)
    median_age = df["Age"].median()
    mode_embarked = df["Embarked"].mode()[0]
    df["Age"] = df["Age"].fillna(value=median_age)
    df["Embarked"] = df["Embarked"].fillna(value=mode_embarked)
    print("Imputed Age Column with Median Age and Embarked Column with Mode\n")
    print(df.isnull().sum(),end=separator)


def feature_engineering(df):
    """
    Applying feature engineering
    :param df: dataframe
    :return: None
    """

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df.drop(columns=["SibSp","Parch"],inplace=True)
    print("Feature Engineering Done\n")
    print(df.head(),end=separator)

    corr = df.corr(numeric_only=True)
    sns.heatmap(corr,annot=True)
    plt.show()

    sns.pairplot(df,hue="Survived")
    plt.show()


def outliers(df):
    """
    Detecting outliers
    :param df: dataframe
    :return: None
    """

    print("Outlier Detection\n")
    cols = [col for col in df if df[col].dtypes == "int64" or df[col].dtypes == "float64"]
    for i, col in enumerate(cols):
        plt.subplot(3,2,i+1)
        sns.boxplot(x=col,data=df)
        plt.title(col)
    plt.tight_layout()
    plt.show()
    print(df["Age"].describe(),end=separator)
    print(df["Fare"].describe(),end=separator)

def splitting(df):
    """
    Split Data
    :param df: dataframe
    :return: xtrain,xtest,ytrain,ytest
    """

    x = df[[cols for cols in df if cols != "Survived"]]
    y = df[["Survived"]]
    xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.3,random_state=42)
    print("Train Test Split Done",end=separator)
    return xtrain,xtest,ytrain,ytest

def encoding(xtrain,xtest):
    """
    Using Label Encoder
    :param xtrain:
    :param xtest:
    :return: xtrain,xtest
    """

    encoder = LabelEncoder()
    categorical_cols = [col for col in xtrain.columns if xtrain[col].dtypes == "str"]
    for i in categorical_cols:
        xtrain[i] = encoder.fit_transform(xtrain[i])
        xtest[i] = encoder.transform(xtest[i])
    print("Encoding Done",end=separator)
    return xtrain,xtest

def model_training(xtrain,ytrain):
    """
    Model Training
    :param xtrain:
    :param ytrain:
    :return: tree
    """

    tree = DecisionTreeClassifier(max_depth=12,max_leaf_nodes=5,min_samples_split=4,ccp_alpha=0.01,random_state=42)
    tree.fit(xtrain,ytrain)
    print("Model Training Done",end=separator)
    return tree

def model_testing(xtest,ytest,trained_model):
    """
    Evaluating model
    :param xtest:
    :param ytest:
    :param trained_model: tree
    :return:  None
    """

    y_prediction = trained_model.predict(xtest)
    print("Accuracy Metrics\n")
    print(accuracy_score(ytest,y_prediction))
    print("Confusion Matrix\n")
    print(confusion_matrix(ytest,y_prediction))
    print("Classification Report\n")
    print(classification_report(ytest,y_prediction),end=separator)


if __name__ == "__main__":
    data = load_dataset()
    data_preprocessing(data)
    feature_engineering(data)
    outliers(data)
    X_train, X_test, y_train, y_test = splitting(data)
    X_train,X_test = encoding(X_train,X_test)
    model = model_training(X_train,y_train)
    model_testing(X_test,y_test,model)

