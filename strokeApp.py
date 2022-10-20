# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 12:04:01 2022

@author: Acer
"""

import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
from sklearn.metrics import confusion_matrix , accuracy_score, precision_score, recall_score

df = pd.read_csv("healthcare-dataset-stroke-data.csv")

st.title('Stroke Prediction')

if st.checkbox('About Data'):
    st.markdown('''
    1) id: unique identifier
    2) gender: "Male", "Female" or "Other"
    3) age: age of the patient
    4) hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension
    5) heart_disease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease
    6) ever_married: "No" or "Yes"
    7) work_type: "children", "Govt_jov", "Never_worked", "Private" or "Self-employed"
    8) Residence_type: "Rural" or "Urban"
    9) avg_glucose_level: average glucose level in blood
    10) bmi: body mass index
    11) smoking_status: "formerly smoked", "never smoked", "smokes" or "Unknown"
    12) stroke: 1 if the patient had a stroke or 0 if not
    ''')
if st.checkbox('Show dataframe'):
    st.dataframe(df)
    
if st.checkbox("Statistical Information"):
    st.write(df.describe())

def user_input_features():
    st.sidebar.header('Specified Input parameters')
    gender = st.sidebar.radio('Gender', ('Male', 'Female'))
    age = st.sidebar.number_input('Age', key = 'int', min_value = 0, max_value = 100)
    hypertension = st.sidebar.radio('Hypertension', ('Yes', 'No'))
    heart_disease = st.sidebar.radio('Heart Disease', ('Yes', 'No'))
    ever_married = st.sidebar.radio('Ever Married', ('Yes', 'No'))
    work_type = st.sidebar.selectbox('Work Type', options = ['children', 'Govt_job', 'Never_worked', 'Private', 'Self-employed'])
    residence_type = st.sidebar.radio('Residence Type', ('Rural', 'Urban'))
    avg_glucose_level = st.sidebar.number_input('Average Glucose Level', min_value = 0.0, max_value = 300.0)
    bmi = st.sidebar.slider('Body Mass Index', min_value = 0.0, max_value = 70.0)
    smoking_status = st.sidebar.selectbox('Smoking Status', options = ['formerly smoked', 'never smoked', 'smokes', 'Unknown'])
    st.sidebar.markdown("") 
     
    data = {
        'Gender': gender,
        'Age': age,
        'Hypertension': hypertension,
        'Heart Disease': heart_disease,
        'Ever Married': ever_married,
        'Work Type': work_type,
        'Residence Type': residence_type,
        'Avg Glucose Level': avg_glucose_level,
        'Bmi': bmi,
        'Smoking Status': smoking_status
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

data_df = user_input_features()
data_df['Gender'] = data_df['Gender'].map({"Male": 0, "Female": 1})
data_df['Hypertension'] = data_df['Hypertension'].map({'Yes': 1, 'No': 0})
data_df['Heart Disease'] = data_df['Heart Disease'].map({'Yes': 1, 'No': 0})
data_df['Ever Married'] = data_df['Ever Married'].map({'Yes': 1, 'No': 0})
data_df['Work Type'] = data_df['Work Type'].map({'Private': 0, 'Self-employed': 1, 'Govt_job':2, 'children':3, 'Never_worked':4})
data_df['Residence Type'] = data_df['Residence Type'].map({'Urban': 0, 'Rural': 1})
data_df['Smoking Status'] = data_df['Smoking Status'].map({'formerly smoked': 0, 'never smoked': 1, 'smokes':2, 'Unknown':3})
   
df['bmi'].fillna(df['bmi'].median(),inplace=True)
df=df.drop('id',axis=1)
df.drop(df[df['gender'] == 'Other'].index, inplace = True)
categorical_data=df.select_dtypes(include=['object']).columns
le=LabelEncoder()
df[categorical_data]=df[categorical_data].apply(le.fit_transform)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
smote=SMOTE()
X_train,y_train=smote.fit_resample(X_train,y_train)
rf = RandomForestClassifier(random_state=25)
rf.fit(X_train, y_train)
y_pred = rf.predict(data_df)

if st.sidebar.button('Predict'):
    if(y_pred[0]==0):
        st.success('Congatulation! You are free from Stroke')
    else:
        st.warning('You have Stroke')

st.markdown(""" The dataset can be download from: https://www.kaggle.com/fedesoriano/stroke-prediction-dataset""")
