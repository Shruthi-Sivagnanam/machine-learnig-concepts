import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression

df = pd.read_csv('canadapercapita.csv')

x=df.drop(columns=['per capita income (US$)'])
y=df['per capita income (US$)']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

model = LinearRegression()
model.fit(x_train,y_train)

print("The accuracy score of the model - ",model.score(x_test,y_test))

year = int(input("Enter the year to predict the per captia income in Canada : "))

if(year>1900):
    input_data=np.asarray(year)
    input_data_reshaped=input_data.reshape(1,-1)
    print(model.predict(input_data_reshaped))
else:
    print("Please enter a valid year")
