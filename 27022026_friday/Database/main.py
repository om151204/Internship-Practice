from sqlalchemy import create_engine
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

sep = f"\n\n{'-'*50}\n\n"

class MSSQLDataLoading:
    def __init__(self,server,database,driver ="ODBC+Driver+17+for+SQL+Server"):
        """
        Initialize database connection
        """
        connection_string = (
            f"mssql+pyodbc://{server}/{database}?driver={driver}"
        )
        try:
            self.engine = create_engine(connection_string)
            print("Engine created successfully!")
        except ConnectionError:
            print("Connection Error")

    def load_table(self, table_name):
        """
        This method loads data from MSSQL database
        """
        try:
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql(query, self.engine)
            return df
        except FileNotFoundError:
            print("Table not found")





def main():

    server = "localhost"
    database = "om"

    # Initialize loader
    loader = MSSQLDataLoading(server, database)
    df = loader.load_table("car_evaluation")





if __name__ == "__main__":
    main()