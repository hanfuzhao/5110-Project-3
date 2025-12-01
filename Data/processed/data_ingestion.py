import urllib.request
#Ingesting the processed data from the cloud storage

url = "https://storage.googleapis.com/mlops-keming-123/Processed_data.csv"
local_filename = "Processed_data.csv"

urllib.request.urlretrieve(url, local_filename)

print("Successfully saved data to:", local_filename)
