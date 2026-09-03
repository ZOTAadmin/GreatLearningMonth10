from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

import pandas as pd

RAW_PATH = "tourism_project/data/tourism.csv"

# Load the raw dataset
df = pd.read_csv(RAW_PATH)

# Validate that the expected columns are present before registering it
expected_columns = [
    "Unnamed", "CustomerID","ProdTaken",
    "Age", "CityTier", "DurationOfPitch","NumberOfPersonVisiting","NumberOfFollowups",
    "PreferredPropertyStar","NumberOfTrips","Passport","PitchSatisfactionScore","OwnCar",
    "NumberOfChildrenVisiting","MonthlyIncome","TypeofContact",
    "Occupation", "Gender","ProductPitched","MaritalStatus","Designation",
]
missing = [c for c in expected_columns if c not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing expected columns: {missing}")

print("Dataset registered successfully.")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("Columns:", list(df.columns))
print("Failure distribution:")
print(df["Failure"].value_counts())
