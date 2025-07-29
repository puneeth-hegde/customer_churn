
# Customer Churn Prediction Dashboard  

A Machine Learning project that predicts customer churn (whether a customer will leave the service) using the Telco Customer Churn dataset.  

It includes:  
- Exploratory Data Analysis (EDA)  
- Model training (Random Forest Classifier)  
- Interactive Streamlit dashboard for visualization and single-customer prediction  

---

## Problem Statement  

Telecom companies face high customer turnover (churn), impacting revenue.  
This project aims to:  
- Analyze customer behavior (EDA)  
- Identify factors contributing to churn  
- Predict churn using ML models  
- Provide an interactive dashboard for analysis and predictions  

---

## Tech Stack  

- Language: Python  
- Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Streamlit  
- ML Algorithm: Random Forest Classifier  
- Deployment: Streamlit (local dashboard)  

---

## Project Structure  

```
customer_churn/
├── data/
│   └── churn.csv                  # Dataset
├── outputs/
│   ├── churn_distribution.png      # EDA output
│   ├── confusion_matrix.png        # Model evaluation plot
│   ├── feature_importance.png      # Feature importance plot
│   └── churn_report.txt            # Classification report
├── main.py                         # Model training & preprocessing script
├── app.py                          # Streamlit dashboard (interactive UI)
├── churn_model.pkl                 # Saved model + scaler + features
├── requirements.txt                # Dependencies
└── README.md                       # Project documentation
```

---

## Dataset  

We used the **Telco Customer Churn dataset** from IBM:  
- 7043 rows, 21 features  
- Features: `gender`, `SeniorCitizen`, `Partner`, `tenure`, `InternetService`, `MonthlyCharges`, etc.  
- Target: **Churn** (`Yes`/`No`)  

Source: [IBM Telco Customer Churn Dataset (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  

---

## Dataset Import and Preprocessing  

### Importing the Dataset  

```python
df = pd.read_csv("data/churn.csv")
print(df.shape)
print(df.head())
```

### Data Cleaning  

```python
df.isnull().sum()
```

### Exploratory Data Analysis (EDA)  

```python
sns.countplot(x='Churn', data=df)
plt.savefig("outputs/churn_distribution.png")
```

### Preprocessing Steps  

```python
from sklearn.preprocessing import LabelEncoder
for col in df.select_dtypes(include=['object']).columns:
    df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop('Churn', axis=1)
y = df['Churn']

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Splitting the Dataset  

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
```

---

## Model Training  

```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
```

---

## Model Evaluation  

```python
from sklearn.metrics import confusion_matrix, classification_report
print(classification_report(y_test, model.predict(X_test)))
```

Outputs:  
- Confusion Matrix → `outputs/confusion_matrix.png`  
- Feature Importance → `outputs/feature_importance.png`  
- Classification Report → `outputs/churn_report.txt`  

---

## Streamlit Dashboard Features  

- CSV Upload: Upload churn data and view churn distribution.  
- Feature Importance: Visualizes top churn predictors.  
- Manual Prediction Form: Enter customer details and predict churn.  

---

## Running the Project  

### Clone repository  

```bash
git clone https://github.com/puneeth-hegde/customer_churn.git
cd customer_churn
```

### Install dependencies  

```bash
pip install -r requirements.txt
```

### Run training script  

```bash
python main.py
```

### Run Streamlit dashboard  

```bash
streamlit run app.py
```

Open in browser → `http://localhost:8501`  

---

## Outputs and Results  

### Churn Distribution  
![Churn Distribution](https://raw.githubusercontent.com/puneeth-hegde/customer_churn/main/outputs/churn_distribution.png)

### Feature Importance  
![Feature Importance](https://raw.githubusercontent.com/puneeth-hegde/customer_churn/main/outputs/feature_importance.png)

### Confusion Matrix  
![Confusion Matrix](https://raw.githubusercontent.com/puneeth-hegde/customer_churn/main/outputs/confusion_matrix.png)

---

## Manual Prediction Example  

Example input:  
- Gender: Male  
- Contract: Month-to-month  
- Internet: DSL  
- Tenure: 5 months  
- Monthly Charges: $70  

Prediction: Likely to Churn  

---

## Requirements  

```
pandas
numpy
matplotlib
seaborn
scikit-learn
streamlit
```

---

## Key Learnings  

- Data loading and cleaning  
- Encoding categorical and scaling numerical features  
- Training and evaluating ML models  
- Building an interactive Streamlit dashboard  

---

## Future Enhancements  

- Deploy dashboard online (Streamlit Cloud/Heroku)  
- Add hyperparameter tuning for better accuracy  
- Integrate SHAP for explainability  

---

## License  

MIT License  

---

## Contributor  

- Puneeth Hegde – Data preprocessing, model pipeline, dashboard integration

## Acknowledgments  
- IBM Telco Churn Dataset: Kaggle  
- Libraries: Scikit-learn, Pandas, Matplotlib, Streamlit  
- Project guidance inspired by open-source ML tutorials and educational resources  

