import urllib.request

url = "https://storage.googleapis.com/mlops-keming-123/Processed_data.csv"
local_filename = "Processed_data.csv"

urllib.request.urlretrieve(url, local_filename)

print("Successfully saved to:", local_filename)
