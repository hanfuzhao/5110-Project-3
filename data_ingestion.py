import pandas as pd

url = "https://storage.googleapis.com/mlops-keming-123/Processed_data.csv"
df = pd.read_csv(url)

print(df.head())
