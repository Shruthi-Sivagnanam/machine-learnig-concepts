import pandas as pd
import matplotlib.pyplot as plt

north_america=pd.read_csv('north_america_2000_2010.csv')
south_america=pd.read_csv('south_america_2000_2010.csv')

#concating the both countries.
america=pd.concat([north_america,south_america])

#transposing the rows and columns for better view.
america_t=america.T
print(america_t)

#calculating mean.
mean_for_each_country=america.mean(axis=1)
country=america['Country']

#representing the data in the form pie cahrt.
fig=plt.figure(figsize=(10,7))
plt.pie(mean_for_each_country,labels=country)
plt.title(label="American countries' work hour")
plt.show();
