import sys
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from joblib import dump

# Kolom numerik yang dinormalisasi (Gender tidak ikut, sesuai README)
NORM_COL = ['AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI']
# Urutan fitur saat training, harus sama persis dengan app.py
FEATURES = NORM_COL + ['Gender']

# Load dataset
try:
    data = pd.read_csv("./dataset/Dataset of Diabetes.csv")
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: Dataset not found. Please check the file path.")
    sys.exit(1)

# Menghapus data yang tidak diperlukan
data = data.drop(['ID', 'No_Pation'], axis=1)

# Memperbaiki kesalahan entri data
data['Gender'] = data['Gender'].replace({'f': 'F'})
data['CLASS'] = data['CLASS'].replace({'N ': 'N', 'P': 'Y', 'Y ': 'Y'})
print("Gender:", data['Gender'].unique().tolist())
print("CLASS:", data['CLASS'].unique().tolist())

# Menangani data yang hilang
missing = data.isna().sum().sum()
if missing > 0:
    print(f"Error: {missing} missing values found in dataset.")
    sys.exit(1)

# Menangani data duplikat
print("\nRecords Before:", len(data))
data = data.drop_duplicates().reset_index(drop=True)
print("Records After:", len(data))

# Data transformation (encode Gender & CLASS)
data['Gender'] = data['Gender'].map({'F': 0, 'M': 1})
data['CLASS'] = data['CLASS'].map({'N': 0, 'Y': 1})

# Normalisasi
scaler = MinMaxScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data[NORM_COL]), columns=NORM_COL)
X = pd.concat([data_scaled, data[['Gender']]], axis=1)[FEATURES]
y = data['CLASS']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print("\nJumlah Baris Data Latih:", len(X_train))
print("Jumlah Baris Data Uji:", len(X_test))

# Model terpilih di README: Original - Random Forest (tanpa SMOTE)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluasi
y_pred = model.predict(X_test)
print("\n### Evaluation (Original - Random Forest) ###")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.6f}")
print(f"Precision: {precision_score(y_test, y_pred):.6f}")
print(f"Recall   : {recall_score(y_test, y_pred):.6f}")
print(f"F1 Score : {f1_score(y_test, y_pred):.6f}")
print("Confusion Matrix [[TN, FP], [FN, TP]]:")
print(confusion_matrix(y_test, y_pred))

# Save the model and scaler
dump(model, "diabetes_model.joblib")
dump(scaler, "scaler.joblib")
print("\nModel and scaler saved successfully!")
