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

# matrix plot - scattered - it works -> DONE
# scatter_matrix(data)
# plt.show()

# https://www.kaggle.com/anthonyhills/classifying-species-of-iris-flowers
# https://www.kaggle.com/farheen28/iris-dataset-analysis-using-knn
# - it works -> DONE
# sns.FacetGrid(data, hue="species", height=10).map(plt.scatter, "sepal_length", "sepal_width").add_legend() 

# plt.show()

# From above scattered plot, we can distinctly distinguish Iris-setosa, but Iris-versicolor and Iris-verginica can't be distinguished based on thier sepal width and sepal length. 
# Thus to distinguish between versicolor and verginica, we have to analyse some other features.

# Here, we can use pairwise plotting - it works -> DONE

# sns.pairplot(data, size=3, diag_kind="kde")

# plt.show()

# From the above pairwise plots, we see that Iris-setosa is distinguishable in all aspects. 
# As for differentiating between Iris-versicolor and Iris-virginica, they can be seperated on the basis of: Petal Length and Petal Width.
# In the plot between petal width verses petal length, the petal width and length of versicolor is smaller than that of virginica.

# 3D scattered plot - it works -> DONE
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt

# fig=plt.figure()
# ax=fig.add_subplot(111, projection='3d')

# x=data["sepal_length"]
# y=data["petal_length"]
# z=data["petal_width"]

#color=("Iris-setosa", "Iris-virginica", "Iris-versicolor")
#color=['Iris-setosa', 'Iris-virginica', 'Iris-versicolor']
# ax.scatter(x,y,z,c='r',marker='o')

# ax.set_xlabel('sepal_length')
# ax.set_ylabel('petal_length')
# ax.set_zlabel('petal-width')

# plt.show()

# These are some parametric and non parametric statistics of our dataset. Parametric: mean, std, min, max, count. Non Parametric: 25%, 50%, 75%
# print(data.describe()) - it works -> DONE

# 1D plot of SepalLength - it works
# import numpy as np
# plt.plot(data['sepal_length'], np.zeros_like(data['sepal_length']), 'o')
# plt.show()

# Through 1-D plots for feautures, we can find outliers.
# From the below box plot with whisker, we see the 3 species of flowers can be distinguised on the basis of petal length and petal width - it works

# sns.boxplot(x='species',y='petal_length', data=data)
# plt.show()

# sns.boxplot(x='species',y='petal_width', data=data)
# plt.show()

# sns.boxplot(x='species',y='sepal_length', data=data)
# plt.show()

# sns.boxplot(x='species',y='sepal_width', data=data)
# plt.show()

# Now, we can find the two most corelated features in the dataset by plotting qq plot or by calculating Pearson correlation coefficient or Spearman's rank correlation coefficient.
# - it works

# import statsmodels.graphics.gofplots as gof
# import pylab
# gof.qqplot_2samples(data['petal_length'] , data['sepal_length'], xlabel="PetalLengthCm", ylabel="SepalLengthCm")
# pylab.show()

# gof.qqplot_2samples(data['petal_width'] , data['sepal_length'], xlabel="PetalWidthCm", ylabel="SepalLengthCm")
# pylab.show()

# gof.qqplot_2samples(data['petal_width'] , data['sepal_width'], xlabel="PetalWidthCm", ylabel="SepalWidthCm")
# pylab.show()

# gof.qqplot_2samples(data['petal_length'] , data['sepal_width'], xlabel="PetalLengthCm", ylabel="SepalWidthCm")
# pylab.show()

# gof.qqplot_2samples(data['petal_width'] , data['petal_length'], xlabel="PetalWidthCm", ylabel="PetalLengthCm")
# pylab.show()

# From above sepal width and sepal length seems to be the most correlated feature. 
# Let's find out the corelation coefficient to see if our observation matches. 
# Pearson Coefficient: ρX,Y=cov(X,Y)/σXσY PCC tells us how two variables are correlated(for linear relationships).
# +1 implies perfectly correlated 
# -1 implies pefectly non correlated 
# Spearman's rank correlation coefficient: rs=ρrgX,rgY=cov(rgX,rgY)σrgXσrgY(for monotonic relationships).
# https://www.kaggle.com/farheen28/iris-dataset-analysis-using-knn -> 23 to 28


# Thus we get an accuracy of 95.4545
import numpy as np
import csv
import random
import math
import operator
import random

data=np.array(data)
np.ndarray
 
def euclideanDistance(train, crossValid, k):
    distance = 0
    for x in range(k):
        distance += pow((train[x] - crossValid[x]), 2)
    return math.sqrt(distance)
 
def getNeighbors(train, crossValid, k):
    distances = []
    length = len(crossValid)-1
    for x in range(len(train)):
        dist = euclideanDistance(crossValid, train[x], k)
        distances.append((train[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors
 
def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]
 
def getAccuracy(crossValid, predictions):
    correct = 0
    for x in range(len(crossValid)):
        if crossValid[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(crossValid))) * 100.0

def main():
    # prepare data by creating 3 datasets, training(65%), cross validation(20%) and test(15%) by using a random number generator.
    dataset = np.array(data)
    training_idx = np.random.randint(dataset.shape[0], size=98)
    test_idx = np.random.randint(dataset.shape[0], size=30)
    cross_idx = np.random.randint(dataset.shape[0], size=22)
    training, test, crossValid = dataset[training_idx,:], dataset[test_idx,:], dataset[cross_idx,:]
    # generate predictions
    predictions=[]
    k = 3
    for x in range(len(crossValid)):
        neighbors = getNeighbors(training, crossValid[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print('> predicted=' + repr(result) + ', actual=' + repr(crossValid[x][-1]))
    accuracy = getAccuracy(crossValid, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')
        
main()