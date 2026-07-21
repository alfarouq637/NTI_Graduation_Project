# import required libraries
import os
import joblib
from sklearn.linear_model import LogisticRegression
import numpy as np

# create models folder if not exists
os.makedirs("models", exist_ok=True)

# create simple dummy data
X_dummy = np.random.rand(100, 10)
y_dummy = np.random.randint(0, 2, size=100)

# train basic model
model = LogisticRegression()
model.fit(X_dummy, y_dummy)

# save valid pkl file
joblib.dump(model, "models/fraud_model.pkl")
print("Dummy model saved successfully")