# Henk Tjalsma, 2019
# Iris Data Set - removed headers csv file otherwise generates error "nan" -> not a number, it means the csv file contained column headers

# https://www.dataquest.io/blog/numpy-tutorial-python/

# Using numpy package to calculate the maximum value of each column
import numpy as np

# https://stackoverflow.com/a/3519314
# https://janakiev.com/blog/csv-in-python/
# Read the csv file into an array.
my_data_set = np.genfromtxt('iris.csv', delimiter=',')

# All of the values in the 1st column of the array
firstcolumn = (my_data_set[:,0]) 
# https://stackoverflow.com/questions/35503718/how-to-find-the-maximum-value-of-a-numpy-array-between-given-indicies
# Used max function numpy 
# max_firstcolumn = np.max(my_data_set[:,0]) 

# All of the values in the 2nd column of the array
secondcolumn = (my_data_set[:,1])
# max_secondcolumn = np.max(my_data_set[:,1])

# All of the values in the 3rd column of the array
thirdcolumn = (my_data_set[:,2])
# max_thirdcolumn = np.max(my_data_set[:,2])

# All of the values in the 4th column of the array
fourthcolumn = (my_data_set[:,3])
# max_fourthcolumn = np.max(my_data_set[:,3])

# print("The maximum value of 1st column is: ",max_firstcolumn)
# print("The maximum value of 2nd column is: ",max_secondcolumn)
# print("The maximum value of 3rd column is: ",max_thirdcolumn)
# print("The maximum value of 4th column is: ",max_fourthcolumn)

print("The output of 1st column is: ",firstcolumn)
print("The output of 2nd column is: ",secondcolumn)
print("The output of 3rd column is: ",thirdcolumn)
print("The output of 4th column is: ",fourthcolumn)