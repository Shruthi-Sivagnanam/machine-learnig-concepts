"""
Predicting whether a patient(female) has diabeties are not.
The dataset has few columns like glucose,insulin etc..,
"""

# importing necccessary library.
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

diabetes_df=pd.read_csv('diabetes.csv')

#separating the input and output dataset.
#input dataset.
x=diabetes_df.drop(columns=['Outcome'])
#output dataset
y=diabetes_df['Outcome']

# Standardization of data (which makes all records to single range).
scaler=StandardScaler()
standardised_data=scaler.fit_transform(x)

x=standardised_data

# Train set and text set are splitted.
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

#support vector model
model_svm = svm.SVC(kernel='linear')
model_svm.fit(x_train,y_train)
prediction_svm=model_svm.predict(x_test)

# decision tree model
model_dt= DecisionTreeClassifier()
model_dt.fit(x_train,y_train)
prediction_dt=model_dt.predict(x_test)

#Logistic Regression
model= LogisticRegression()
model.fit(x_train,y_train)
ascore=model.score(x_test,y_test)

print("Accuracy value for Logistic Regression model",ascore)
print("Accuracy value for decision tree model",accuracy_score(y_test,prediction_dt))
print("Accuracy value for support vector model",accuracy_score(y_test,prediction_svm))

# Predictive System

input_data=np.asarray((4,173,70,14,168,29.7,0.361,33))
input_data_reshaped=input_data.reshape(1,-1)

std_data=scaler.transform(input_data_reshaped)

#prediction in each model
print(model_svm.predict(std_data))
print(model_dt.predict(std_data))
print(model.predict(std_data))

# 0 means the person is not diabetic  1 means the person is diabetic.
