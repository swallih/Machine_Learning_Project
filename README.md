🌧️ Rainfall Prediction Using Machine Learning
📌 Project Overview

This project focuses on predicting next-day rainfall using historical weather data and supervised machine learning techniques. By analyzing key atmospheric features such as humidity, temperature, pressure, and cloud cover, the model aims to provide accurate daily rainfall forecasts to support better planning and risk reduction.

Author: Muhammed Swalih
Role: Data Analyst / Machine Learning Enthusiast

🎯 Objective

To analyze weather-related data and build machine learning models that can accurately predict whether it will rain the next day, while identifying the most influential features affecting rainfall events.

📂 Dataset Description

Source: Kaggle 

ml_project_rainfall_prediction

Total rows: ~19,000

Data type: Daily weather observations

Features

Location – Geographical area of observation

Temperature – Average daily temperature

Humidity – Moisture level in the air

Wind Speed – Ground-level wind speed

Precipitation – Rainfall recorded for the day

Cloud Cover – Portion of sky covered by clouds

Pressure – Atmospheric pressure

Target Variable

RainTomorrow

0 → No rain the next day

1 → Rain the next day

The dataset was preprocessed to handle missing values and outliers before modeling.

❓ Problem Statement

Traditional rainfall prediction methods often struggle with accuracy due to complex and non-linear weather patterns. This project applies machine learning techniques to improve next-day rainfall predictions using historical weather data.

🌍 Importance of the Project

Accurate rainfall prediction:

Helps farmers plan irrigation effectively

Supports flood preparedness and disaster management

Assists urban and infrastructure planning

Enables data-driven decision-making for individuals and organizations

🧹 Data Preprocessing

Missing Values: Identified and handled appropriately

Outlier Detection:

Technique used: Interquartile Range (IQR)

Outliers were handled using the capping method

Class Imbalance:

Dataset was imbalanced

Oversampling was applied to balance rainy and non-rainy cases

📊 Exploratory Data Analysis (EDA)

Boxplots: Used to visualize and detect outliers, especially in Precipitation and RainTomorrow

Correlation Heatmap:

Showed strong relationships between rainfall and features such as humidity, pressure, and cloud cover

Weak or less relevant features (e.g., Location) were identified for possible exclusion

🧠 Modeling Approach
Machine Learning Models Implemented

Logistic Regression

K-Nearest Neighbors (KNN)

Naïve Bayes (GaussianNB)

Decision Tree

Random Forest

AdaBoost

Gradient Boosting

XGBoost

Train–Test Strategy

Dataset split into training and testing sets to validate model performance.

📈 Initial Model Performance

Decision Tree, Random Forest, AdaBoost, Gradient Boosting achieved very high accuracy (~0.9997)

XGBoost achieved slightly lower but still excellent accuracy (~0.9992)

Oversampling significantly improved performance due to class imbalance.

⚙️ Model Evaluation & Hyperparameter Tuning

Hyperparameter tuning was performed using GridSearchCV to optimize model performance.

Best Parameters Found

Decision Tree:
criterion='gini', splitter='best', random_state=0

AdaBoost:
learning_rate=0.01, n_estimators=50, random_state=0

Gradient Boosting:
learning_rate=0.01, max_depth=3, n_estimators=50

Observations

Tree-based models showed minimal change after tuning, maintaining near-perfect accuracy

Logistic Regression improved slightly

KNN and GaussianNB performance decreased after tuning

🏆 Final Model Selection

Although several models performed exceptionally well, XGBoost was selected as the final model due to:

High accuracy

Better reliability

Efficiency in handling complex and large datasets

✅ Recommendations

Focus on strong predictors like humidity, pressure, and cloud cover

Balance rainy and dry day samples for better learning

Experiment with different model configurations

Avoid unnecessary complexity—simpler models can generalize better

Validate predictions with real-world weather data regularly

Handle missing values carefully based on data distribution

🧾 Conclusion

This project successfully developed a high-performing rainfall prediction system using machine learning techniques. After preprocessing, balancing the dataset, and evaluating multiple models, the final solution demonstrated excellent accuracy in predicting next-day rainfall. The insights gained from key weather features can support better decision-making in agriculture, public safety, and environmental planning.
