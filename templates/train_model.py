import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# 1. Load only the first 100k rows for speed
data_path = os.path.join('data', 'PS_20174392719_1491204439457_logs.csv')
print("Loading data (Fast Mode)...")
df = pd.read_csv(data_path, nrows=100000) 

# 2. Preprocess
type_map = {'CASH_OUT': 1, 'PAYMENT': 2, 'CASH_IN': 3, 'TRANSFER': 4, 'DEBIT': 5}
df['type'] = df['type'].map(type_map)

# 3. Features
features = ['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
X = df[features]
y = df['isFraud']

# 4. Train a smaller, faster model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training fast model...")
model = RandomForestClassifier(n_estimators=10, random_state=42) # Fewer trees = Faster
model.fit(X_train, y_train)

# 5. Save
with open('payments.pkl', 'wb') as f:
    pickle.dump(model, f)

print("SUCCESS! 'payments.pkl' is ready.")