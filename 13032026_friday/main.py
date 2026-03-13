import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import matplotlib.pyplot as plt

seperator = f"\n\n{'--' * 70}\n\n"

class AprioriAlgorithm:
    """Implementation of Apriori Algorithm on Groceries Dataset"""

    def __init__(self):
        self.df = None
        self.df_encoded = None
        self.basket = None
        self.transactions = None
        self.encoder = TransactionEncoder()
        self.algo = None
        self.rules = None

    def load_data(self):
        """
        Loading data from csv file
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

    def group_items(self):
        """
        Group items by transaction
        Returns - None
        """
        try:
            self.load_data()
            self.basket = self.df.groupby(["Member_number", "Date"])["itemDescription"].apply(list).reset_index()
            self.transactions = self.basket["itemDescription"].tolist()
            # print(f"Transactions = {self.transactions}")

            print("\nSuccessfully grouped items by transactions", end = seperator)

        except Exception as error:
            print(error)

    def feature_encoding(self):
        """
        One hot encoding of transactions
        Returns - None
        """
        try:
            self.group_items()
            encoder_array = self.encoder.fit_transform(self.transactions)
            self.df_encoded = pd.DataFrame(encoder_array, columns = self.encoder.columns_)

            print("Feature encoding successful", end=seperator)

        except Exception as error:
            print(error)

    def run_algorithm(self):
        """
        Run Apriori Algorithm
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
        try:
            self.run_algorithm()

            # 1. Generate rules
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
            self.rules = rules_df[
                rules_df['antecedents'].apply(lambda x: len(x) >= 1) &
                rules_df['consequents'].apply(lambda x: len(x) >= 1)
                ]

            print("Association Rules:", self.rules.shape[0])
            print(self.rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(5))
            print("\nAssociation rules generated successfully", end = seperator)

        except Exception as error:
            print(f"Error in generating rules: {error}")

    def visualize(self):
        """
        Visualize the results of Apriori Algorithm
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
    algorithm = AprioriAlgorithm()
    algorithm.visualize()

if __name__ == "__main__":
    main()