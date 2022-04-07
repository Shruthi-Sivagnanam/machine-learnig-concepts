# importing the required libraries.
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

music_df=pd.read_csv('music.csv')

"""
Now the data set contains three columns.
['age','gender','genre']

the column age and gender is going to be the input set.
the column genre is going to be the output set.(prediction output).
"""
# x is the input dataset.
x=music_df.drop(columns=['genre'])
# y is the output dataset.
y=music_df['genre']

#spliting the test and train data set.
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)


model = DecisionTreeClassifier()
model.fit(x_train,y_train)
prediction = model.predict(x_test)
print(accuracy_score(y_test,prediction))
