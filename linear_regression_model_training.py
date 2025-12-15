import numpy as np
import pandas as pd
import joblib

class LinearRegression:
    def __init__(self):
        self.weights = None  
        self.bias = None     

    def fit(self, X, y):
        X_b = np.hstack([np.ones((X.shape[0], 1)), X])  
        theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y     
        self.bias = theta[0]
        self.weights = theta[1:]

    def predict(self, X):
        # X: (n_samples, n_features)
        return X @ self.weights + self.bias

# ---------- LOAD AND CLEAN DATA ----------
data = pd.read_csv("housing.csv")

# Choose features
feature_cols = ["total_rooms", "total_bedrooms", "households"]

# Drop rows with ANY missing values in features or target
data_clean = data[feature_cols + ["median_house_value"]].dropna()

X = data_clean[feature_cols].values         
y = data_clean["median_house_value"].values  

# ---------- TRAIN MODEL ----------
model = LinearRegression()
model.fit(X, y)

# ---------- SAVE MODEL ----------
joblib.dump(model, "multiple_linear_regression_model.joblib")


np.save("X_clean.npy", X)
np.save("y_clean.npy", y)
