import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# 1️⃣ Load model and preprocessing artifacts
artifacts = pickle.load(open("churn_model.pkl", "rb"))
model = artifacts["model"]
scaler = artifacts["scaler"]
feature_names = artifacts["feature_names"]

st.title("📊 Customer Churn Prediction Dashboard")
st.write("Analyze churn data and make predictions using a trained ML model.")

# 2️⃣ File uploader for CSV
uploaded_file = st.file_uploader("Upload a CSV file for churn analysis", type="csv")

if uploaded_file:
    # Load and preview dataset
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview:", df.head())

    # Preprocess: Encode categorical and scale numerical features
    for col in df.select_dtypes(include=['object']).columns:
        if col != "Churn":  # Don't encode target column
            df[col] = df[col].astype('category').cat.codes

    # Align features with training columns
    df_processed = df.drop('Churn', axis=1)

    # Add any missing columns (fill with 0)
    for col in feature_names:
        if col not in df_processed.columns:
            df_processed[col] = 0

    # Reorder columns to match training
    df_processed = df_processed[feature_names]

    # Scale features
    df_scaled = scaler.transform(df_processed)

    # Churn distribution plot
    st.subheader("Churn Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='Churn', data=df)
    st.pyplot(fig)

    # Feature Importance plot
    st.subheader("Feature Importance")
    importances = model.feature_importances_
    feat_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    feat_df = feat_df.sort_values("Importance", ascending=False)
    st.bar_chart(feat_df.set_index("Feature"))

# 3️⃣ Manual Prediction Form
st.subheader("🔮 Predict Churn for a Single Customer")
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.selectbox("Partner", ["No", "Yes"])
dependents = st.selectbox("Dependents", ["No", "Yes"])
tenure = st.slider("Tenure (months)", 0, 72)
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["Yes", "No"])
online_backup = st.selectbox("Online Backup", ["Yes", "No"])
device_protection = st.selectbox("Device Protection", ["Yes", "No"])
tech_support = st.selectbox("Tech Support", ["Yes", "No"])
streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])
streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No"])
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0)

if st.button("Predict Churn"):
    # Build input row with ALL features
    input_df = pd.DataFrame([[
        gender, senior, partner, dependents, tenure, phone_service,
        multiple_lines, internet, online_security, online_backup,
        device_protection, tech_support, streaming_tv, streaming_movies,
        contract, paperless_billing, payment_method,
        monthly_charges, total_charges
    ]], columns=[
        "gender","SeniorCitizen","Partner","Dependents","tenure","PhoneService",
        "MultipleLines","InternetService","OnlineSecurity","OnlineBackup",
        "DeviceProtection","TechSupport","StreamingTV","StreamingMovies",
        "Contract","PaperlessBilling","PaymentMethod",
        "MonthlyCharges","TotalCharges"
    ])

    # Encode categorical variables
    for col in input_df.select_dtypes(include=['object']).columns:
        input_df[col] = input_df[col].astype('category').cat.codes

    # Ensure all training features exist
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns
    input_df = input_df[feature_names]

    # Scale features
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]
    st.success("✅ Customer is likely to CHURN" if prediction == 1 else "❌ Customer is NOT likely to churn")
