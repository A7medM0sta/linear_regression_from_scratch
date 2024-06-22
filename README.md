# Linear Regression from Scratch

This repository is part of a series of machine learning, artificial intelligence, and data science algorithms built from scratch as a learning experience.

In this part, we focus on implementing Linear Regression in Python.

## Linear Regression

Linear Regression is a supervised learning algorithm used for predicting a continuous outcome variable (also called the dependent variable) based on one or more predictor variables (also called independent variables).

The goal of Linear Regression is to find the best fit line that can predict the outcome variable with the minimum error.

## Implementation Details

The implementation includes the following steps:

- Data Preprocessing: The dataset is loaded and preprocessed. Preprocessing includes handling missing values, encoding categorical variables, feature scaling, etc.
- Train-Test Split: The dataset is split into a training set and a test set. The training set is used to train the model, and the test set is used to evaluate its performance.
- Model Training: The Linear Regression model is trained using the training set. The training process involves finding the weights (or coefficients) that result in the best fit line.
- Model Evaluation: The performance of the model is evaluated on the test set. The evaluation metrics used are Mean Squared Error (MSE) and R-squared.

## Datasets

The implementation includes tests on the following datasets:

- 'uscrime'
- 'BostonHousing'
- 'diamonds'

## Usage

To run the tests on the datasets, use the following command:

```python
python test-datasets.py

python test-learning-rate.py
