import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from linear_regression_model_training import LinearRegression

# Load model
model = joblib.load('multiple_linear_regression_model.joblib')

# Load dataset
data = pd.read_csv('housing.csv')

# Select features and target
feature_cols = ['total_rooms', 'total_bedrooms', 'households']
target_col = 'median_house_value'

# Drop rows with missing values in features or target
data_clean = data[feature_cols + [target_col]].dropna()

X = data_clean[feature_cols].values
y = data_clean[target_col].values

st.title('House Prediction with Multiple Features')
st.write('Predict house value using multiple features: rooms, bedrooms, households.')

# User Input
rooms = st.number_input('Total Rooms:', min_value=1, value=2000)
bedrooms = st.number_input('Total Bedrooms:', min_value=1, value=500)
households = st.number_input('Households:', min_value=1, value=300)

if st.button('Predict'):
    X_input = np.array([[rooms, bedrooms, households]])
    prediction = model.predict(X_input)
    st.success(f"Predicted House Value: ${prediction[0]:,.2f}")

    # Metrics
    y_pred = model.predict(X)
    mae = np.mean(np.abs(y - y_pred))
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    r2 = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))

    # KPIs
    st.subheader('Model Performance')
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{mae:.2f}")
    col2.metric("RMSE", f"{rmse:.2f}")
    col3.metric("R²", f"{r2:.2f}")

    st.markdown("""
    - **MAE (Mean Absolute Error)**: Average absolute difference between actual and predicted values. Lower is better.
    - **RMSE (Root Mean Squared Error)**: Square root of average squared differences. Lower is better.
    - **R² (R-squared)**: Proportion of variance explained by the model. Closer to 1 is better.
    """)

    # Actual vs Predicted
    fig = px.scatter(x=y, y=y_pred, labels={'x': 'Actual', 'y': 'Predicted'}, title='Actual vs Predicted Prices')
    fig.add_shape(type='line', x0=y.min(), y0=y.min(), x1=y.max(), y1=y.max(), line=dict(color='red', dash='dash'))
    st.plotly_chart(fig)
    st.markdown("""
    **Actual vs Predicted Prices**: Shows how close the model's predictions are to actual values. Points close to the diagonal line indicate good predictions.
    """)

    # Residual Plot
    residuals = y - y_pred
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_pred, y=residuals, mode='markers', name='Residuals'))
    fig.update_layout(title='Residual Plot', xaxis_title='Predicted', yaxis_title='Residuals')
    st.plotly_chart(fig)
    st.markdown("""
    **Residual Plot**: Shows the difference (residuals) between actual and predicted values. Randomly scattered residuals around zero indicate a good model.
    """)

    # Histogram of Predictions
    fig = px.histogram(y_pred, nbins=20, title='Distribution of Predicted Prices')
    st.plotly_chart(fig)
    st.markdown("""
    **Distribution of Predicted Prices**: Shows the spread of predicted house values. A normal distribution is ideal for linear regression.
    """)
