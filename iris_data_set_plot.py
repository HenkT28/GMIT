# Henk Tjalsma, 2019
# Iris Data Set - plotting iris data set

# https://www.kaggle.com/abhishekkrg/python-iris-data-visualization-and-explanation
# https://www.kaggle.com/jchen2186/machine-learning-with-iris-dataset
# https://machinelearningmastery.com/machine-learning-in-python-step-by-step/

# Importing Package 
import sys
import numpy
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings

np.random.seed(0)  # For reproducibility

# Reading Iris Dataset in Pandas Dataframe
data = pandas.read_csv("iris_original.csv")
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'species']

# Printing DataFrame -only first 20 rows to understand what data look like
# print(data.head(20)) - works!

# Peek at the data
# Data have 5 Columns - first four are features and fifth is Classfication of the Iris type
# print (data.head()) - works!

# Find out no of rows for each Species.
# print(data.groupby('species').size())  - works!

# Create 3 DataFrame for each Species  - works!

# setosa=data[data['species']=='setosa']
# versicolor =data[data['species']=='versicolor']
# virginica =data[data['species']=='virginica']

# print(setosa.describe())
# print(versicolor.describe())
# print(virginica.describe())

# The count tells that all the 4 features have 150 rows
# In general ,From Mean we can say that sepal is larger than petal.
# print(data.describe()) - works!

# Plotting Petal Length vs Petal Width & Sepal Length vs Sepal width
# Warnings.simplefilter("ignore")
# #Supress any warning - works!
# plt.figure()
# fig,ax=plt.subplots(1,2,figsize=(17, 9))
# data.plot(x="sepal_length",y="sepal_width",kind="scatter",ax=ax[0],sharex=False,sharey=False,label="sepal",color='r')
# data.plot(x="petal_length",y="petal_width",kind="scatter",ax=ax[1],sharex=False,sharey=False,label="petal",color='b')
# ax[0].set(title='Sepal comparasion ', ylabel='sepal-width')
# ax[1].set(title='Petal Comparasion',  ylabel='petal-width')
# ax[0].legend()
# ax[1].legend()
# plt.show()
# plt.close()

# We can see that  there are some petals which are smaller than rest of petal.
# Let's examine them.

# For each Species, let's check what is petal and sepal distibution - it works!
# setosa=data[data['species']=='setosa']
# versicolor =data[data['species']=='versicolor']
# virginica =data[data['species']=='virginica']

# plt.figure()
# fig,ax=plt.subplots(1,2,figsize=(21, 10))

# setosa.plot(x="sepal_length", y="sepal_width", kind="scatter",ax=ax[0],label='setosa',color='r')
# versicolor.plot(x="sepal_length",y="sepal_width",kind="scatter",ax=ax[0],label='versicolor',color='b')
# virginica.plot(x="sepal_length", y="sepal_width", kind="scatter", ax=ax[0], label='virginica', color='g')

# setosa.plot(x="petal_length", y="petal_width", kind="scatter",ax=ax[1],label='setosa',color='r')
# versicolor.plot(x="petal_length",y="petal_width",kind="scatter",ax=ax[1],label='versicolor',color='b')
# virginica.plot(x="petal_length", y="petal_width", kind="scatter", ax=ax[1], label='virginica', color='g')

# ax[0].set(title='Sepal comparasion ', ylabel='sepal-width')
# ax[1].set(title='Petal Comparasion',  ylabel='petal-width')
# ax[0].legend()
# ax[1].legend()
# plt.show()
# plt.close()

# Setosa - Setosa Petal are relatively smaller than rest of species - can easily be separated from the other species 
# Versicolor & virginica are also separate in petal comparison
# The Setosa sepal are smallest in length and largest in width compared to other species.


###############################Histogram# Plot all feature for all species###########################################################################
# - it works

setosa=data[data['species']=='setosa']
versicolor =data[data['species']=='versicolor']
virginica =data[data['species']=='virginica']

plt.figure()

fig,ax=plt.subplots(4,3,figsize=(17, 8))
setosa["sepal_length"].plot(kind="hist", ax=ax[0][0],label="setosa",color ='r',fontsize=10)
versicolor["sepal_length"].plot(kind="hist", ax=ax[0][1],label="versicolor",color='b',fontsize=10)
virginica["sepal_length"].plot( kind="hist",ax=ax[0][2],label="virginica",color='g',fontsize=10)

setosa["sepal_width"].plot(kind="hist", ax=ax[1][0],label="setosa",color ='r',fontsize=10)
versicolor["sepal_width"].plot(kind="hist", ax=ax[1][1],label="versicolor",color='b',fontsize=10)
virginica["sepal_width"].plot( kind="hist",ax=ax[1][2],label="virginica",color='g',fontsize=10)

setosa["petal_length"].plot(kind="hist", ax=ax[2][0],label="setosa",color ='r',fontsize=10)
versicolor["petal_length"].plot(kind="hist", ax=ax[2][1],label="versicolor",color='b',fontsize=10)
virginica["petal_length"].plot( kind="hist",ax=ax[2][2],label="virginica",color='g',fontsize=10)

setosa["petal_width"].plot(kind="hist", ax=ax[3][0],label="setosa",color ='r',fontsize=10)
versicolor["petal_width"].plot(kind="hist", ax=ax[3][1],label="versicolor",color='b',fontsize=10)
virginica["petal_width"].plot( kind="hist",ax=ax[3][2],label="virginica",color='g',fontsize=10)

plt.rcParams.update({'font.size': 10})
plt.tight_layout()

ax[0][0].set(title='SepalLengthCm')
ax[0][1].set(title='SepalLengthCm')
ax[0][2].set(title='SepalLengthCm')
ax[1][0].set(title='SepalWidthCm ')
ax[1][1].set(title='SepalWidthCm ')
ax[1][2].set(title='SepalWidthCm ')
ax[2][0].set(title='PetalLengthCm')
ax[2][1].set(title='PetalLengthCm')
ax[2][2].set(title='PetalLengthCm')
ax[3][0].set(title='PetalWidthCm')
ax[3][1].set(title='PetalWidthCm')
ax[3][2].set(title='PetalWidthCm')

ax[0][0].legend()
ax[0][1].legend()
ax[0][2].legend()
ax[1][0].legend()
ax[1][1].legend()
ax[1][2].legend()
ax[2][0].legend()
ax[2][1].legend()
ax[2][2].legend()
ax[3][0].legend()
ax[3][1].legend()
ax[3][2].legend()


plt.show()
plt.close()