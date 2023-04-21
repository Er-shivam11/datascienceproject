# Credit Card Fraud Detection
## Introduction
This Python script aims to detect credit card fraud using various machine learning models. The script uses a dataset containing credit card transactions, where the objective is to predict whether a transaction is fraudulent or not.

## Libraries Used
Pandas: for loading and manipulating the dataset
Scikit-learn: for building and training the machine learning models
## Dataset
The credit card dataset used in this script contains records of 284,807 transactions, with 31 features. The dataset is available in CSV format and can be found here.

## Data Exploration
The script explores the dataset by performing the following steps:
Loading the dataset to a Pandas DataFrame
Displaying the first 5 and last 5 rows of the dataset
Displaying dataset information, such as the number of records and data types
Checking the number of missing values in each column
Displaying the distribution of legitimate and fraudulent transactions
Displaying statistical measures of the data, such as the mean and standard deviation of transaction amounts for both legitimate and fraudulent transactions

## Data Preprocessing
The script preprocesses the data by:

Separating the data into two dataframes: legitimate and fraudulent transactions
Creating a new dataset by randomly sampling the same number of records from the legitimate and fraudulent transactions (i.e., 492 records each)
Splitting the new dataset into training and test sets, with a test size of 0.2

## Machine Learning Models
The script trains and evaluates the following machine learning models:
Logistic Regression
Decision Tree Classifier
K-Nearest Neighbors Classifier

## Conclusion
The results show that the Logistic Regression and Decision Tree models perform better than the K-Nearest Neighbors model. Among the tested models, the Logistic Regression model achieved the highest accuracy score of 0.93, followed by the Decision Tree model with an accuracy score of 0.90, 
and the K-Nearest Neighbors model with an accuracy score of 0.60.
