# Henk Tjalsma, 2019
# Iris Data Set - plotting iris data set

# https://www.kaggle.com/abhishekkrg/python-iris-data-visualization-and-explanation
# https://www.kaggle.com/jchen2186/machine-learning-with-iris-dataset
# https://machinelearningmastery.com/machine-learning-in-python-step-by-step/

# Importing libraries
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Reading Iris Dataset in Pandas Dataframe
data = pd.read_csv("iris_original.csv")
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'species']

# matrix plot - scattered - it works
# scatter_matrix(data)
# plt.show()

# https://www.kaggle.com/anthonyhills/classifying-species-of-iris-flowers
# https://www.kaggle.com/farheen28/iris-dataset-analysis-using-knn
# - it works
# sns.FacetGrid(data, hue="species", height=10).map(plt.scatter, "sepal_length", "sepal_width").add_legend() 

# plt.show()

# From above scattered plot, we can distinctly distinguish Iris-setosa, but Iris-versicolor and Iris-verginica can't be distinguished based on thier sepal width and sepal length. 
# Thus to distinguish between versicolor and verginica, we have to analyse some other features.

# Here, we can use pairwise plotting - it works

# sns.pairplot(data, size=3, diag_kind="kde")

# plt.show()

# From the above pairwise plots, we see that Iris-setosa is distinguishable in all aspects. 
# As for differentiating between Iris-versicolor and Iris-virginica, they can be seperated on the basis of: Petal Length and Petal Width.
# In the plot between petal width verses petal length, the petal width and length of versicolor is smaller than that of virginica.

# 3D scattered plot - it works
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig=plt.figure()
ax=fig.add_subplot(111, projection='3d')

x=data["sepal_length"]
y=data["petal_length"]
z=data["petal_width"]

#color=("Iris-setosa", "Iris-virginica", "Iris-versicolor")
#color=['Iris-setosa', 'Iris-virginica', 'Iris-versicolor']
ax.scatter(x,y,z,c='r',marker='o')

ax.set_xlabel('sepal_length')
ax.set_ylabel('petal_length')
ax.set_zlabel('petal-width')

plt.show()

# https://www.kaggle.com/farheen28/iris-dataset-analysis-using-knn -> 10