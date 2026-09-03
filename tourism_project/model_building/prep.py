# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder

# Define constants for the dataset and output paths
DATASET_PATH = "tourism_project/data/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop unique identifier column (not useful for modeling)
df.drop(df.columns[0], axis=1, inplace=True) # First Column in Excel file is not named
df.drop(columns=['CustomerID'], inplace=True)

# Encode categorical columns
label_encoder = LabelEncoder()
for col in ['TypeofContact', 'Occupation', 'Gender','ProductPitched','MaritalStatus','Designation']:
    df[col] = label_encoder.fit_transform(df[col])

# Target column
target_col = 'ProdTaken'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)

print("Data prepared: train/test splits written.")
