
import pandas as pd

df = pd.read_csv("data/movielens_100k.csv")

print("Dataset shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 5 movies:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())
