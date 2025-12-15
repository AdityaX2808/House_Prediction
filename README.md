# House Price Prediction using Linear Regression

This repository contains a machine learning project that predicts house prices using **multiple linear regression**. The model is built from scratch and deployed as a web application using **Streamlit**. The goal is to predict the median house value based on features like total rooms, total bedrooms, and households.

## Problem Statement

House price prediction is a common regression problem in real estate. Accurately estimating house prices helps buyers, sellers, and investors make informed decisions. In this project, we aim to build a robust linear regression model that predicts the median house value using key features such as:

- Total rooms
- Total bedrooms
- Households

The model is trained on the California Housing dataset, which contains various attributes related to housing in California.

## How We Solve the Problem

### 1. Data Preparation
- We load the California Housing dataset and select relevant features.
- We clean the data by removing rows with missing values to ensure the model trains on high-quality data.

### 2. Model Training
- We implement a **multiple linear regression** model from scratch using NumPy.
- The model is trained to learn the relationship between the selected features and the median house value.

### 3. Model Evaluation
- We evaluate the model using key metrics:
  - **MAE (Mean Absolute Error)**
  - **RMSE (Root Mean Squared Error)**
  - **R² (R-squared)**
- These metrics help us understand the accuracy and reliability of our predictions.

### 4. Web Application
- We deploy the trained model as a web application using **Streamlit**.
- The app allows users to input house features and get real-time price predictions.
- The app also displays model metrics and visualizations to help users understand the model's performance.

## Features

- **Multiple Linear Regression Model**: Built from scratch using NumPy.
- **Streamlit Web App**: Interactive interface for house price prediction.
- **Model Metrics**: MAE, RMSE, and R² for performance evaluation.
- **Visualizations**: Actual vs predicted prices, residual plot, and distribution of predictions.

## Getting Started

### Prerequisites
- Python 3.x
- Required libraries: `numpy`, `pandas`, `joblib`, `plotly`, `streamlit`

### Installation
1. Clone the repository:
