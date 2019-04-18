# Henk Tjalsma, 2019
# Iris Data Set - removed headers csv file otherwise generates error "nan" -> not a number, it means the csv file contained column headers

# https://www.dataquest.io/blog/numpy-tutorial-python/

# Using numpy package to calculate the mean (not the same as average) of each column
# https://stackoverflow.com/questions/20054243/np-mean-vs-np-average-in-python-numpy
import numpy as np

# https://stackoverflow.com/a/3519314
# https://janakiev.com/blog/csv-in-python/
# Read the file into an array.
my_data_set = np.genfromtxt('iris.csv', delimiter=',')

# All of the values in the 1st column of the array
# firstcolumn = (my_data_set[:,0])
# Using mean function numpy package
mean_firstcolumn = np.mean(my_data_set[:,0]) 

# secondcolumn = (my_data_set[:,1])
mean_secondcolumn = np.mean(my_data_set[:,1])

# thirdcolumn = (my_data_set[:,2])
mean_thirdcolumn = np.mean(my_data_set[:,2])

# fourthcolumn = (my_data_set[:,3])
mean_fourthcolumn = np.mean(my_data_set[:,3])

print("The mean of 1st column is: ",mean_firstcolumn)
print("The mean of 2nd column is: ",mean_secondcolumn)
print("The mean of 3rd column is: ",mean_thirdcolumn)
print("The mean of 4th column is: ",mean_fourthcolumn)