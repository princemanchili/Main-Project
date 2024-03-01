"""This module contains necessary function needed"""

# Import necessary modules
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import streamlit as st
from sklearn.ensemble import RandomForestClassifier


@st.cache_data()
def load_data():
    """This function returns the preprocessed data"""

    # Load the Diabetes dataset into DataFrame.
    df = pd.read_csv('pneumonia_dataset1.csv')
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    df['Pneumonia'] = df['Pneumonia'].map({'No Pneumonia':1,'Mild Pneumonia':2,'Moderate Pneumonia':3,'Severe Pneumonia':4})
    df.dropna()
    # Perform feature and target split
    X = df[["Age","Gender","Body_Temperature","Cough","Sore_Throat","Difficulty_Breathing","Chest_Pain","White_Blood_Cell_Count","Heart_Rate","Respiratory_Rate"]]
    y = df['Pneumonia']

    X  = np.array(X)
    y = np.array(y)

    print(type(df['Age']))


    return df, X, y

@st.cache_data()
def train_model(X, y):
    """This function trains the model and return the model and model score"""
    # # Create the model
    # model = DecisionTreeClassifier(
    #         ccp_alpha=0.0, class_weight=None, criterion='entropy',
    #         max_depth=4, max_features=None, max_leaf_nodes=None,
    #         min_impurity_decrease=0.0, min_samples_leaf=1, 
    #         min_samples_split=2, min_weight_fraction_leaf=0.0,
    #         random_state=42, splitter='best'
    #     )
    model = RandomForestClassifier(n_estimators=10, random_state=0) #randomforest model
    # Fit the data on model
    model.fit(X, y)
    # Get the model score
    score = model.score(X, y)

    # Return the values
    return model, score

def predict(X, y, features):
    # Get model and model score
    model, score = train_model(X, y)
    # Predict the value
    prediction = model.predict(np.array(features).reshape(1, -1))

    return prediction, score
