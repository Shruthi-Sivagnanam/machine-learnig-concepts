import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('middle_tn_schools.csv')

#descibe the data statistically 
print(df.describe)
print(df.to_string())

# this give the statistical data with rows as column and column as row
print(df[['reduced_lunch','school_rating']].groupby(['school_rating']).describe().unstack())

#it is displays the correlation between the two attributes.
print(df[['reduced_lunch','school_rating']].corr())


#representing the dataset in bargraph
name=df['name'].head(10)
school_rating=df['school_rating'].head(10)

#fig=plt.figure(figsize=(50,7))

#plt.bar(name[0:10],school_rating[0:10])

#plt.show()

#representing the dataset in pie chart


fig=plt.figure(figsize=(10,7))
plt.pie(school_rating,labels=df['school_rating'])
plt.show()
