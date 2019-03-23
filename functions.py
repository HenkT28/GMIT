# Henk Tjalsma, 2019
# Solution to problem 10 - functions.py script
# Write a program that displays a plot of the functions x, x2 and 2x in the range [0, 4].
# https://realpython.com/python-matplotlib-guide/
# https://matplotlib.org/1.4.2/users/pyplot_tutorial.html
# https://stackoverflow.com/questions/46513843/plot-math-function-fx-with-varying-values-of-x-and-for-different-values-of-n
# https://techmonger.github.io/2/simple-parabola-matplotlib/
# https://stackoverflow.com/questions/22276066/how-to-plot-multiple-functions-on-the-same-figure-in-matplotlib
# https://www.tutorialspoint.com/numpy/numpy_matplotlib.htm
# https://matplotlib.org/users/pyplot_tutorial.html

# import numpy as np
import numpy as np
# import math
import math
#import matplotlib
import matplotlib.pyplot as plt

x= np.arange(0,4)
y_x = x
y_2x = 2*x
y_x2 = x*x

# Adding graph title and labels to X and Y axis
plt.title("Matplotlib x (red), 2x (green), x2 (blue) in the range [0, 4]")
plt.xlabel("x axis caption") 
plt.ylabel("y axis caption") 

# plot may or may not plot the lines, depending on the arguments
plt.plot(x,y_x,'r')
plt.plot(x,y_2x,'g')
plt.plot(x,y_x2,'b')

# scatter draws points without lines connecting
# plt.scatter(x, y_x)
# plt.scatter(x, y_2x)
# plt.scatter(x, y_x2)
plt.show()