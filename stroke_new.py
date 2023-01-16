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
from sklearn.linear_model import LogisticRegression
import streamlit as st

df = pd.read_csv("healthcare-dataset-stroke-data.csv")

def user_input_features():
    st.sidebar.header('Specify Input Parameters')
    gender = st.sidebar.radio('Gender', ('Male', 'Female'))
    age = st.sidebar.number_input('Age', key = 'int', min_value = 0, max_value = 100)
    hypertension = st.sidebar.select_slider('Hypertension', ('Yes', 'No'))
    heart_disease = st.sidebar.select_slider('Heart Disease', ('Yes', 'No'))
    ever_married = st.sidebar.select_slider('Ever Married', ('Yes', 'No'))
    work_type = st.sidebar.selectbox('Work Type', options = ['children', 'Govt_job', 'Never_worked', 'Private', 'Self-employed'])
    residence_type = st.sidebar.radio('Residence Type', ('Rural', 'Urban'))
    avg_glucose_level = st.sidebar.number_input('Average Glucose Level', min_value = 0.0, max_value = 300.0)
    bmi = st.sidebar.slider('Body Mass Index', min_value = 0.0, max_value = 60.0)
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
        'Smoking Status': smoking_status}
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
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
smote=SMOTE()
X_train,y_train=smote.fit_resample(X_train,y_train)
model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)
y_pred = model.predict(data_df)

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
if st.checkbox('Show Dataframe'):
    st.dataframe(df) 
if st.checkbox("Show Statistical Information"):
    st.write(df.describe())    
if st.checkbox("Show Data Visualization"): 
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Analysis of Stroke Target")
        fig, ax = plt.subplots(figsize=(3,3))
        ax = sns.countplot(x = 'stroke', data = df)
        st.pyplot(fig)
        st.info(" From the above plot, the number of people having a stroke is very high\
                as compared to not having stroke. The data is higly unbalanced. \
                ")
    with col2:
        st.write("")
    st.subheader("Analysis of Numerical Features")
    col1, col2, col3 = st.columns(3)
    with col1:
        fig, ax = plt.subplots(figsize=(3,3))
        ax = sns.distplot(df['age'], hist=True, kde=True) 
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(3,3))
        ax = sns.distplot(df['avg_glucose_level'], hist=True, kde=True)
        st.pyplot(fig)
    with col3:
        fig, ax = plt.subplots(figsize=(3,3))
        ax = sns.distplot(df['bmi'], hist=True, kde=True)
        st.pyplot(fig)
    st.info(" The distribution of age is closer to normal distribution while \
                the avg_glucose_level and bmi distributions have right skewed distribution.")
    st.subheader("Analysis of Categorical Features")
    col1, col2, col3 = st.columns(3)
    with col1:
        fig, axes = plt.subplots(figsize=(3,3))
        axes = sns.countplot(data=df,x='gender', hue='gender')
        st.pyplot(fig)
        st.info(" For the gender variable, there are three categories which are \
            Female, Male, and Other. There are more female gender than male gender \
            and there is only 1 other gender.")
    with col2:
        fig, ax = plt.subplots(figsize=(3,3))
        axes = sns.countplot(data=df,x='hypertension', hue='hypertension')
        st.pyplot(fig)
        st.info("For the hypertension variable, there are two categories which are Yes and No. \
            There are more patient having no hypertension than having hypertension.")
    with col3:
        fig, ax = plt.subplots(figsize=(3,3))
        axes = sns.countplot(data=df,x='heart_disease', hue='heart_disease')
        st.pyplot(fig)
        st.info("For the heart_disease variable, there are two categories which are Yes and No. \
            There are more patient having no heart disease than having a heart disease.")
    with col1:
        fig, ax = plt.subplots(figsize=(3,3))
        axes = sns.countplot(data=df,x='ever_married', hue='ever_married')
        st.pyplot(fig)
        st.info("For the ever_married variable, there are two levels which are Yes and No. \
            There are more people ever married than never married.")
    with col2:
        fig, ax = plt.subplots(figsize=(3,3))
        axes = sns.countplot(data=df,x='work_type', hue='work_type')
        st.pyplot(fig)
        st.info("For the work_type variable, there are five categories which are children, \
            Govt_job, Never_worked, Private, and Self-employed. The highest number is \
            private while the least number is Never_worked.")
    with col3:
        fig, ax = plt.subplots(figsize=(3,3))
        axes = sns.countplot(data=df,x='Residence_type', hue='Residence_type')
        st.pyplot(fig)
        st.info("For the Residence_type variable, there are two categories which are \
            Rural and Urban. The number of both Rural and Urban are almost equal.")
    with col1:
        fig, axes = plt.subplots(figsize=(3,3))
        axes = sns.countplot(data=df,x='smoking_status', hue='smoking_status')
        st.pyplot(fig) 
        st.info("For the smoking_status variable, there are has four categories which \
            are formely smoked, never smoked, smokes, and Unknown. The highest number \
            is never smoked while both of the least number is smokes.")

if st.sidebar.button('Predict'):
    if(y_pred ==0):
        st.success('Congratulation! You are free from Stroke')
    else:
        st.warning('You are having a stroke')
        
st.markdown(""" The dataset can be downloaded from: https://www.kaggle.com/fedesoriano/stroke-prediction-dataset""")
