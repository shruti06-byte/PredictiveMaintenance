
import pandas as pd

# Load the dataset
df = pd.read_csv("data/ai4i2020.csv")

# Display basic information
print("Dataset loaded successfully!")
print("Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nMachine failure counts:")
print(df["Machine failure"].value_counts())
