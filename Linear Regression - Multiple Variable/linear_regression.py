import pandas as pd
import numpy as np
from word2number import w2n
import math
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

"""
modifying the data set(missing values) converting that to have new csv file.

df = pd.read_csv('hiring.csv')
df.experience = df.experience.fillna("zero")
df.experience = df.experience.apply(w2n.word_to_num)
median_score = math.floor(df['test_score(out of 10)'].mean())

df['test_score(out of 10)'] = df['test_score(out of 10)'].fillna(median_score)
print(df)

df.to_csv('hiring_modified.csv')
"""

df=pd.read_csv('hiring_modified.csv')

x=df.drop(columns=['Unnamed: 0','salary($)'])
y=df.drop(columns=['experience','test_score(out of 10)','interview_score(out of 10)','Unnamed: 0'])

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

model = LinearRegression()
model.fit(x_train,y_train)

print("Enter the detail of the employee to predict the salary: ")
experience=int(input("Year(s) of experience : "))
test_score = float(input("Test score: (out of 10)"))
interview_score=int(input("Interview score: (out of 10)"))

if(0<=test_score<=10 and 0<=interview_score<=10 and experience>=0):
    data=[experience,test_score,interview_score]
    input_data=np.asarray(data)
    input_data_reshaped = input_data.reshape(1,-1)
    print("The salary expected: ",int(model.predict(input_data_reshaped)))
else:
    print("Please enter valid attributes!")


