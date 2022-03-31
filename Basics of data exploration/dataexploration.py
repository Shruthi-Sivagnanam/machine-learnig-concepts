# library for reading data set
import pandas as pd
#library for array conversion
import numpy as np

#reading .csv file from the current directory.
df = pd.read_csv('SalaryGender.csv')

#for displaying the data set in a string format.
print(df.to_string())

#displays number of rows and columns in dataframe.
print(df.shape)

#displays the type of dataframe.
print(type(df))

#displays the dataypes of the each columns in dataframe
print(df.dtypes)

#slicing operation with data.
data=df.iloc[0:5]
print(data)

#displays all the unique data in the dataframe.
print(df['Salary'].unique())
print(df['Age'].unique())

#mean of all the columns in dataframe.
# ------ print(df.mean())

#median of all the columns of the dataframe.
# ------ print(df.median())

#mode value of columns
print(df.mode(axis=0))
#mode values of rows
print(df.mode(axis=1))




