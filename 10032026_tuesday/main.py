import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

sep = f"\n{'---'*70}\n\n"
class CustomerSegmentation:
    """
    This class contains core logic for implementing K-Means Clustering (Unsupervised Learning)
    """
    def __init__(self):
        """
        This constructor initializes all the required variables and parameters
        """
        self.df = None
        self.numeric_cols = []
        self.encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.columns = None
        self.wcss = []
        self.model = None
        self.randomstate = 42

    def load_data(self):
        """
        This function loads the dataset into pandas dataframe
        :return: None
        """
        try:
            self.df = pd.read_csv('Mall_Customers.csv')
            print("Data loaded successfully\n")
            print(self.df.head(),end=sep)
        except FileNotFoundError:
            print("File not found, please try again\n")

    def data_preprocessing(self):
        """
        This method performs basic pre-processing and checks shape,statistics and drops unnecessary columns
        :return: None
        """
        try:
            self.load_data()
            print(f"Dataset shape:- {self.df.shape}")
            print("Data Description:-\n")
            self.df.info()
            print("\nDataset Statistics:-\n")
            print(self.df.describe())
            print(f"\nChecking null values:-\n{self.df.isnull().sum()}\n")
            print(f"Duplicates in the dataset:- {self.df.duplicated().sum()}\n")
            print("Dropping Customer ID\n")
            self.df.drop(columns=['CustomerID'], inplace=True)
            print(f"Checking duplicates after dropping the id column:- {self.df.duplicated().sum()}",end=sep)
        except Exception as e:
            print("Error in data preprocessing",e)

    def eda(self):
        """
        This method performs eda using heatmap,boxplot and scatter plot
        :return: None
        """
        try:
            self.data_preprocessing()
            corr = self.df.corr(numeric_only=True)
            sns.heatmap(corr, annot=True)
            plt.title("Correlation Matrix",color='red',fontsize=20,weight='bold')
            plt.show()

            self.numeric_cols = self.df.select_dtypes(include=["int64","float64"]).columns
            for i,cols in enumerate(self.numeric_cols):
                plt.subplot(3,1,i+1)
                sns.boxplot(self.df[cols])
                plt.title(cols,fontsize=12,weight='bold')
            plt.show()

            plt.figure(figsize = (10,10))
            plt.scatter(self.df["Annual Income (k$)"],self.df["Spending Score (1-100)"])
            plt.title("Scatter plot",color='red',fontsize=20,weight='bold')
            plt.xlabel("Annual Income (k$)",fontsize=20,weight='bold')
            plt.ylabel("Spending Score (1-100)",fontsize=15,weight='bold')
            plt.show()
            print("EDA Completed",end=sep)
        except Exception as e:
            print("Error in eda",e)

    def feature_encoding(self):
        """
        This method performs label encoding on gender column
        :return: None
        """
        try:
            self.eda()
            self.df["Gender"] = self.encoder.fit_transform(self.df["Gender"])
            print("Encoding completed for gender column",end=sep)
        except Exception as e:
            print("Error in feature encoding",e)

    def feature_scaling(self):
        """
        This method performs standard scaling on all the numeric columns
        :return: None
        """
        try:
            self.feature_encoding()
            self.columns = self.df.columns
            scaled_data = self.scaler.fit_transform(self.df)
            self.df = pd.DataFrame(scaled_data,columns=self.columns)
            print("Scaling Completed",end=sep)
        except Exception as e:
            print("Error in feature scaling",e)

    def elbow_method(self):
        """
        This method performs elbow method to find the optimum number of clusters
        :return: None
        """
        try:
            self.feature_scaling()
            self.wcss = []
            for i in range(1,11):
                kmeans = KMeans(n_clusters=i, random_state=self.randomstate,init='k-means++')
                kmeans.fit(self.df)
                self.wcss.append(kmeans.inertia_)
            plt.plot(range(1,11),self.wcss,marker='o')
            plt.title("Elbow Method",color='red',fontsize=20,weight='bold')
            plt.xlabel("Number of clusters",fontsize=20,weight='bold')
            plt.ylabel("WCSS",fontsize=20,weight='bold')
            plt.show()
        except Exception as e:
            print("Error in elbow method",e)

    def train_kmeans(self):
        """
        This method trains the k-means clustering algorithm on the dataset
        :return: None
        """
        try:
            self.elbow_method()
            self.model = KMeans(n_clusters = 6, random_state=self.randomstate,init='k-means++')
            self.df["Clusters_formed"] = self.model.fit_predict(self.df)
            print(self.df["Clusters_formed"].value_counts())
            # Silhouette score has range from 1 to -1 (alternative of elbow method used to determine appropriate value of k)
            score = silhouette_score(self.df, self.df["Clusters_formed"])  # Closer to 1 is better
            print(f'\nSilhouette Score: {score:.3f}\n')
            print("Training completed and clusters assigned",end=sep)
        except Exception as e:
            print("Error in training the model",e)

    def visualize_clusters(self):
        """
        This method visualizes the formation of clusters
        :return: None
        """
        try:
            self.train_kmeans()
            plt.figure(figsize = (10,7))
            sns.scatterplot(x = self.df["Annual Income (k$)"],y = self.df["Spending Score (1-100)"],hue = self.df["Clusters_formed"],palette="Set1")
            centers = self.model.cluster_centers_ # Centroids
            plt.scatter(centers[:,2],centers[:,3],s = 300,marker='*',c='black',label='Centroids')
            plt.title("Customer Segments",color='red',fontsize=20,weight='bold')
            plt.legend()
            plt.show()
            print("Customer Segmentation Visualization Completed",end=sep)
        except Exception as e:
            print("Error in visualizing clusters",e)

def main():
    """
    It is the driver function to execute main logic
    :return: None
    """
    customer_segmentation = CustomerSegmentation()
    customer_segmentation.visualize_clusters()

if __name__ == "__main__":
    main()
