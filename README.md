# Rainfall Prediction Using Machine Learning

## Project Overview
This project predicts next-day rainfall using historical weather data and supervised machine learning techniques. The objective is to analyze key atmospheric variables such as humidity, temperature, pressure, and cloud cover to generate accurate daily rainfall forecasts that support effective planning and risk mitigation.

## Objective
To understand and predict rainfall using weather-related data through analytics-based and machine learning methods, and to identify the key features influencing daily rainfall events.

## Dataset Description
Source: Kaggle  
Number of rows: Approximately 19000  
Data type: Daily weather observations

### Features
- Location – Geographical area of observation  
- Temperature – Average daily temperature  
- Humidity – Moisture level in the air  
- Wind Speed – Ground-level wind speed  
- Precipitation – Rainfall recorded for the day  
- Cloud Cover – Portion of sky covered by clouds  
- Pressure – Atmospheric pressure  

### Target Variable
RainTomorrow  
- 0 indicates no rain the next day  
- 1 indicates rain the next day

## Problem Statement
Rainfall is difficult to predict accurately using traditional methods due to complex and non-linear weather patterns. This project applies machine learning techniques to improve next-day rainfall prediction using historical weather data.

## Business Importance
Accurate rainfall prediction:
- Helps farmers optimize irrigation planning  
- Supports flood preparedness and risk management  
- Assists infrastructure and urban planning  
- Enables data-driven decision-making

## Data Preprocessing
- Missing values were identified and handled  
- Outliers were detected using the IQR method  
- Outliers were treated using capping  
- Class imbalance was handled using oversampling

## Exploratory Data Analysis
- Boxplots were used to visualize outliers  
- Correlation heatmaps identified strong relationships between rainfall and humidity, pressure, and cloud cover  
- Weak features were analyzed for possible exclusion

## Modeling Approach

### Models Implemented
- Logistic Regression  
- K-Nearest Neighbors  
- Naive Bayes  
- Decision Tree  
- Random Forest  
- AdaBoost  
- Gradient Boosting  
- XGBoost  

### Training and Validation
The dataset was split into training and testing sets to evaluate generalization performance.

## Model Evaluation
- Tree-based models achieved very high accuracy after handling class imbalance  
- Hyperparameter tuning was performed using GridSearchCV  
- Evaluation focused on stability and generalization

## Final Model Selection
XGBoost was selected as the final model due to its strong performance, efficiency, and reliability on complex datasets.

## Tech Stack
- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  
- XGBoost  

## Project Structure
- data/ – Dataset files  
- notebooks/ – EDA and experiments  
- models/ – Trained models  
- README.md – Documentation  

## Conclusion
This project demonstrates a robust machine learning approach for rainfall prediction. Key atmospheric features significantly improved forecasting accuracy, making the model useful for agriculture, public safety, and environmental planning.
