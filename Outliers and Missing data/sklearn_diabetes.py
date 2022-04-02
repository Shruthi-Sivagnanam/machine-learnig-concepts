"""
Checking the missing and outlier data in a dataset from sklearn.
"""

from sklearn.datasets import load_diabetes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=load_diabetes()

#this statement targets the data the import data in a dataset.
print(dataset.target)


#to display the dataset with training data and column set.
df=pd.DataFrame(data=np.c_[dataset['data'],dataset['target']],columns=dataset['feature_names']
+['target'])

print(df)

# it is used to diaply any missing data in any of column (false-if no missing and tru-if any missing)
print(df.isnull().any())

# to display the outlier

"""
Outlier - the dataset which is out of range.
which varies a lot from other observation.
"""


#this display the outlier data in form a boxplot so the outlier data can be spotted.
for column in df:
    plt.figure()
    df.boxplot(column)
plt.show()
