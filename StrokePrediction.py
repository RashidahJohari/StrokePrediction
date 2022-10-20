# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 12:17:41 2022

@author: Acer
"""

#Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

#%% Step 1) Importing the data
# Reading the dataset
df = pd.read_csv("healthcare-dataset-stroke-data.csv")

# checking first 5 rows of dataset
print(df.head())

#%% Step 2) Data Exploration
# checking the shape of dataset
print(df.shape)
# **Conclusion**: There are 5110 rows and 12 columns in the dataset.

# checking missing values
print(df.isna().sum())
# **Conclusion:** The dataset has 1 column of missing value on bmi column

# Checking any null values, columns data types, number of entries
print(df.info())
# **Conclusion:**
# The dataset has 5110 total entries, 12 total columns, Only bmi column has missing values. There are 3 floating datatypes, 4 integer datatypes and 5 are the object datatypes.

# Finding duplicate Rows
df[df.duplicated()].sum()
# **Conclusion:** There are no duplicate rows present in the dataset

# Checking for Possible Outliers
fig, axs = plt.subplots(3, figsize = (7,7))
plt1 = sns.boxplot(df['age'], ax = axs[0])
plt4 = sns.boxplot(df['avg_glucose_level'], ax = axs[1])
plt5 = sns.boxplot(df['bmi'], ax = axs[2])
plt.tight_layout()
# **Conclusion:** 
# No outliers in age feature.
# Outliers are present in both bmi and avg_glucose_level.

#%% Step 3) Exploratory Data Analysis
# Analysis of Stroke 
plt.figure(figsize=(8,6))
sns.countplot(x = 'stroke', data = df)
plt.show()
# According to the output above, there are more individuals who do not suffer from stroke than those who do.

sns.distplot(df['age'], hist=True, kde=True)
# Most of the individuals of the dataset are of age 40 and above. The age distribution of female subjects is closer to normal distribution.

sns.distplot(df['avg_glucose_level'], hist=True, kde=True)
# Most of the individuals have 75-100 average glucose levels.
# avg_glucose_level has right skewed distribution

sns.distplot(df['bmi'], hist=True, kde=True)
# Most of the individuals of the dataset has BMI index between 20-30 kilogram/metresq.
# avg_glucose_level has right skewed distribution.

# Correlation plot
plt.figure(figsize=(15,10))
sns.heatmap(df.corr(),annot=True)
# **Conclusion:**
# No strong correlation is observed between any of the features.
# Variables that are showing some effective correlation are:  
# age, hypertension, heart_disease, avg_glucose_level.

#%% Step 4) Data Preprocessing
# Impute missing values
df['bmi'].fillna(df['bmi'].median(),inplace=True)

# drop irrelavant column
df=df.drop('id',axis=1)

# dropping 'Other' in gender
df.drop(df[df['gender'] == 'Other'].index, inplace = True)

# categorical data into numeric ones using Label Encoder
categorical_data=df.select_dtypes(include=['object']).columns
le=LabelEncoder()
df[categorical_data]=df[categorical_data].apply(le.fit_transform)

# Separating dataset into input and target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

#Splitting the dataset into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# oversampling the train datsets using SMOTE
smote=SMOTE()
X_train,y_train=smote.fit_resample(X_train,y_train)

#%% Step 5) Model Building
rf = RandomForestClassifier(random_state=25)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test) 

#%% Step 6) Model Performance
#Confusion matrix and classification report
con_mat = confusion_matrix(y_test, y_pred)
print(con_mat)

sns.heatmap(con_mat, annot=True, fmt="d")
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

print(classification_report(y_test, y_pred))