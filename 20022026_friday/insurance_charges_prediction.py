import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,r2_score

# Loading the dataset
df = pd.read_csv("data.csv")
print(df.head())

# Basic data info
print(f"Shape: {df.shape}")
print(f"\nColumn datatypes: \n{df.dtypes}")
print(f"\nMissing Values \n{df.isnull().sum()}")
print(f"\nDuplicate Values {df.duplicated().sum()}")

# Dropping Duplicate Records
df.drop_duplicates(keep="first", inplace=True)
print(f"\nDuplicate Values {df.duplicated().sum()}")

# Data Visualization (EDA)
# Histogram
plt.figure(figsize=(6,4))
sns.histplot(df["charges"],kde=True)
plt.title("Distribution of Charges")
plt.show()
# Insurance charges are heavily right-skewed.

# Scatter Plot
plt.figure(figsize=(6,4))
sns.scatterplot(x="age",y="charges",data=df)
plt.title("Age vs Charges")
plt.show()

# Insurance charges generally increase with age, but are separated into three distinct price bands based on other categorical factors.
# Pair Plot
plt.figure(figsize=(6,4))
sns.pairplot(df)
plt.show()

# Box Plot
plt.figure(figsize=(6,4))
sns.boxplot(x="smoker",y="charges",data=df)
plt.title("Distribution of Smoker vs Charges")
plt.show()

# This plot shows that there are outliers present in the dataset
def detect_outliers_iqr(new_df,column):
    q1 = new_df[column].quantile(0.25)
    q3 = new_df[column].quantile(0.75)
    iqr = q3 - q1
    lb = q1-1.5*iqr
    ub = q3 + 1.5*iqr
    outliers = new_df[(new_df[column] < ub) | (new_df[column] > lb)]
    print(f"Lower bound: {lb}, Upper bound: {ub}")
    return lb,ub
lower_bound,upper_bound = detect_outliers_iqr(df,"charges")
print(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}")

df_no_outliers = df[(df["charges"]>=lower_bound) & (df["charges"]<=upper_bound)]
print(f"Dataset after removing outliers {df_no_outliers.shape}")

df["log_charges"] = np.log(df["charges"])
print(df.head())

# Features and Target Selection
x = df.drop(["charges","log_charges"], axis=1)
y = df["log_charges"]

#Encoding and Scaling
numerical_features = ["age","bmi","children"]
categorical_features = ["sex","smoker","region"]

numeric_transformation = StandardScaler()
categorical_transformation = OneHotEncoder(drop="first")

preprocessor = ColumnTransformer(transformers=[("num",numeric_transformation,numerical_features),("cat",categorical_transformation,categorical_features)])

#Train Test Split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
print(f"Train.shape: {X_train.shape}")
print(f"Test.shape: {X_test.shape}")

#Model Training
lr_pipeline = Pipeline(steps=[("preprocessor",preprocessor),("model",LinearRegression())])
lr_pipeline.fit(X_train,y_train)

# Model Evaluation
y_pred = lr_pipeline.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test,y_pred)}")
print(f"MAE: {mean_absolute_error(y_test,y_pred)}")
print(f"RMSE: {np.sqrt(mean_absolute_error(y_test,y_pred))}")
print(f"R2 Score: {r2_score(y_test,y_pred)}")

# Data Point Testing
test_datapoint = [{"age":19,"sex":"female","bmi":27.9,"children":0,"smoker":"yes","region":"southwest"}]
test_df = pd.DataFrame(test_datapoint)
test_prediction = lr_pipeline.predict(test_df)
final_prediction = np.exp(test_prediction)
print(final_prediction)

plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Actual and Predicted Insurance Charges")
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
)
plt.show()