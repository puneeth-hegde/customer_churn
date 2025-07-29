import pandas as pd
import random

# Sample values for categorical features
genders = ["Male", "Female"]
internet_services = ["DSL", "Fiber optic", "No"]
contracts = ["Month-to-month", "One year", "Two year"]
payment_methods = ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]

# Generate fake data
data = []
for _ in range(50):  # generate 50 records
    gender = random.choice(genders)
    tenure = random.randint(0, 72)
    monthly_charges = round(random.uniform(20, 120), 2)
    total_charges = round(monthly_charges * tenure, 2)
    internet = random.choice(internet_services)
    contract = random.choice(contracts)
    payment = random.choice(payment_methods)
    churn = random.choice(["Yes", "No"])  # random target
    
    data.append([gender, tenure, monthly_charges, total_charges, internet, contract, payment, churn])

# Create DataFrame
df = pd.DataFrame(data, columns=[
    "gender", "tenure", "MonthlyCharges", "TotalCharges",
    "InternetService", "Contract", "PaymentMethod", "Churn"
])

# Save CSV
df.to_csv("sample_churn_data.csv", index=False)
print("✅ Sample test dataset generated: 'sample_churn_data.csv'")
print(df.head())
