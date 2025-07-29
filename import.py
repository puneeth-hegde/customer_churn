
import pandas as pd

# Direct download and save locally
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)
df.to_csv("data/churn.csv", index=False)  # saves to your local data folder

print("✅ Dataset downloaded and saved to data/churn.csv")
print(df.head())
