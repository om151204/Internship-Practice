import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import matplotlib.pyplot as plt

seperator = f"\n\n{'--' * 70}\n\n"

class AprioriAlgorithm:
    """Class to implement the Apriori Algorithm for Market Basket Analysis."""
    def __init__(self):
        """
        Initialize data attributes and the Transaction Encoder.
        """
        self.df = None
        self.df_encoded = None
        self.basket = None
        self.transactions = None
        self.encoder = TransactionEncoder()
        self.algo = None
        self.rules = None

    def load_data(self):
        """
        Load the grocery dataset from a CSV file into a DataFrame.
        Returns - None
        """
        try:
            self.df = pd.read_csv("Groceries_Dataset.csv")
            print(f"\nDataset \n{self.df.head()}\n")
            print("Dataset loaded successfully", end = seperator)

        except FileNotFoundError:
            print("File not found")
        except Exception as error:
            print(error)

    def preprocess(self):
        """
        This method is used to preprocess the data
        return: None
        """
        try:
            self.load_data()
            print(f"Columns : {self.df.columns.tolist()}", end=seperator)
            self.df.info()
            print(end=seperator)
            print(f"Null-Values :\n\n{self.df.isnull().sum()}", end=seperator)
            print(f"Total-Duplicates : {self.df.duplicated().sum()}", end=seperator)
            print(f"Stats: \n\n{self.df.describe()}", end=seperator)
        except Exception as e:
            print(f"Error in preprocessing: {e}")

    def group_items(self):
        """
        Group individual items into lists representing unique transactions.
        Returns - None
        """
        try:
            # It groups the data by Member_number and Date (assuming one member on one day is one "shopping trip").
            self.preprocess()
            self.basket = self.df.groupby(["Member_number", "Date"])["itemDescription"].apply(list).reset_index()
            self.transactions = self.basket["itemDescription"].tolist()
            # print(f"Transactions = {self.transactions}")
            print("\nSuccessfully grouped items by transactions", end = seperator)

        except Exception as error:
            print(error)

    def feature_encoding(self):
        """
        Convert transaction lists into a One-Hot Encoded Boolean DataFrame.
        Returns - None
        """
        try:
            """
            Transaction Encoder creates a giant table of True/False (or 1s and 0s). Each column is a product, and 
            each row is a transaction. If a cell is True, that item was in that basket.
            """
            self.group_items()
            encoder_array = self.encoder.fit_transform(self.transactions)
            self.df_encoded = pd.DataFrame(encoder_array, columns = self.encoder.columns_)
            print("Feature encoding successful", end=seperator)

        except Exception as error:
            print(error)

    def run_algorithm(self):
        """
        Identify frequent item-sets using the Apriori algorithm.
        Returns - None
        """
        try:
            self.feature_encoding()
            self.algo = apriori(
                self.df_encoded,
                min_support = 0.01,
                use_colnames = True
            )
            print(f"Total Frequent Item-sets = {self.algo.shape[0]}\n")
            print("Model Training Completed", end = seperator)

        except Exception as error:
            print(error)

    def generate_association_rules(self):
        """
        Generate and filter 'If-Then' rules based on confidence and support.
        :return: None
        """
        try:
            self.run_algorithm()
            # 1. Generate rules
            # Confidence is the probability that item B is bought given that item A was bought.
            rules_df = association_rules(
                self.algo,
                metric="confidence",
                min_threshold=0.1
            )
            # 2. Guard Clause: Check if rules_df is valid before processing
            if rules_df is None or rules_df.empty:
                print("No association rules were generated.")
                return
            # 3. Filter rules
            """
            Antecedents: The "If" part (the item already in the basket).
            Consequents: The "Then" part (the item they are likely to add).
            This line ensures that both sides of the "If-Then" statement actually contain at least one item. 
            It prevents "ghost" rules from appearing in your results.
            """
            self.rules = rules_df[
                rules_df['antecedents'].apply(lambda x: len(x) >= 1) &
                rules_df['consequents'].apply(lambda x: len(x) >= 1)
                ]

            print("Association Rules:", self.rules.shape[0])
            """
            Support: How often the combination (A + B) appears in the entire dataset.
            Lift: This is the most important metric.
            Lift > 1: There is a strong positive relationship (A actually causes people to buy B).
            Lift = 1: The relationship is just random chance.
            """
            print(self.rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(5))
            print("\nAssociation rules generated successfully", end = seperator)

        except Exception as error:
            print(f"Error in generating rules: {error}")

    def visualize(self):
        """
        Plot a bar chart of the top 10 most purchased items.
        :return: None
        """
        try:
            self.generate_association_rules()
            top_items = self.df['itemDescription'].value_counts().head(10)
            top_items.plot(kind='bar', title='Top 10 Most Purchased Items')
            plt.xlabel("Item")
            plt.ylabel("Count")
            plt.show()
            print("Visualization Completed",end = seperator)
        except Exception as error:
            print(error)

def main():
    """
    Entry point to run the algorithm and display results.
    :return: None
    """
    algorithm = AprioriAlgorithm()
    algorithm.visualize()

if __name__ == "__main__":
    main()