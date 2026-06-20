# Ecommerce Customer Churn Prediction using Artificial Neural Networks

## Overview

This project predicts whether an ecommerce customer is likely to churn using an Artificial Neural Network (ANN). Customer churn prediction helps businesses identify customers who may stop using their services and take proactive retention measures.

## Dataset

The dataset contains customer demographics, purchasing behavior, satisfaction scores, login activity, and other ecommerce-related features.

### Features Used

* Tenure
* Preferred Login Device
* City Tier
* Warehouse To Home Distance
* Preferred Payment Mode
* Hours Spent On App
* Number of Devices Registered
* Preferred Order Category
* Satisfaction Score
* Marital Status
* Number of Addresses
* Complaint Status
* Order Count
* Cashback Amount

## Data Preprocessing

* Handled categorical variables using Label Encoding.
* Standardized numerical features using StandardScaler.
* Split dataset into training and testing sets.
* Prepared data for ANN training.

## Model Architecture

Artificial Neural Network built using TensorFlow/Keras:

* Input Layer
* Hidden Layer with Sigmoid Activation
* Output Layer with Sigmoid Activation

### Loss Function

* Binary Cross Entropy

### Optimizer

* Adam

## Training

The model was trained for multiple epochs with validation data to monitor performance and reduce overfitting.

## Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score

## Results

The ANN model successfully learned customer behavior patterns and achieved competitive performance in predicting customer churn.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* TensorFlow / Keras
* Matplotlib
* Kaggle

## Future Improvements

* Hyperparameter tuning
* Deep neural network architectures
* Feature engineering
* Ensemble learning methods
* Model deployment using Spring Boot and Angular

## Author

Vinamra Gupta
