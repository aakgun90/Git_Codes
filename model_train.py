
import numpy as numpy
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import matplotlib.pyplot as plt

"""
1- Train our Model
2- Create Web App using flask
3- Commit the code in Github
4- Create an Account in Heroku(PAAS)
5- Link the Github to Heroku
6- Deploy the Model
7- Web App is Ready
"""

dataset = pd.read_csv("C:\\Users\\alicn\\Desktop\\python_codes\\techpro\\deployment_series\\hiring.csv")
#Import Dataset and feature engineering

#print(dataset)

#dataset["experience"] = dataset[" experience"].astype(str)
#dataset["test_score"] = dataset[" test_score"].astype(float)
dataset["experience"].fillna(0, inplace= True)
dataset["test_score(out of 10)"].fillna(dataset["test_score(out of 10)"].mean(), inplace=True)
#print(dataset.columns)
#print(dataset.experience)
#print(dataset.isnull().sum())
#print(dataset.info())
# Create independent values
X = dataset.iloc[:, :3]
X

# Convert experience columns to int
def convert_to_int(word):
    word_dict = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
                "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "zero": 0, 0: 0
    }
    return word_dict[word]
X["experience"] = X["experience"].apply(lambda x: convert_to_int(x))
#print(X)    

# Create Dependent value
y = dataset.iloc[:, -1]
#print(y)

# Spliting Train and Test set
    # Since we have a very small dataset, we will train our model with all available data.
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

# Fitting model with traning data
regressor.fit(X,y)
#Saving model to disk
pickle.dump(regressor, open("modell.pkl", "wb"))

# Loading model to compare the results
model = pickle.load(open("model.pkl", "rb"))

#
#print(model.predict([[2, 9, 6]]))
#
