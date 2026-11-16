import kagglehub
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. Download dataset and load it (using 200,000 rows to prevent mobile crash)
path = kagglehub.dataset_download("ealaxi/paysim1")
csv_file = os.path.join(path, "PS_20174392719_1491204439457_log.csv")
df = pd.read_csv(csv_file, nrows=200000)

# 2. Filter to only TRANSFER and CASH_OUT (where laundering actually happens)
aml_df = df[df['type'].isin(['TRANSFER', 'CASH_OUT'])].copy()

# 3. Convert text 'type' into numbers for the AI (TRANSFER=0, CASH_OUT=1)
aml_df['type'] = aml_df['type'].map({'TRANSFER': 0, 'CASH_OUT': 1})

# 4. Set clues (X) and the target answer (y)
features = ['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
X = aml_df[features]
y = aml_df['isFraud']

# 5. Split data (80% training for the AI, 20% testing to grade it)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Build and Train the Random Forest AI
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# 7. Test the AI and print the accuracy scoreboard
predictions = model.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
print("\nAccuracy Report:\n", classification_report(y_test, predictions))
