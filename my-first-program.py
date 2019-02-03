i = 2

i = 3

i = i + 1

# Below is a compound statement - it contains other statements
# They go in a block together
# The colon says, do the following...while this condition is true -> while i < 10:
# So it prints value of i to the screen -> 4, and then adds 1 to the value of i -> 5
# ----------------
# Here is what is funny about the while statement. 
# Python will figure out that is the end of the statements that are part of the while loop. 
# Because the next line is a blank line, and the line after that is not indented with 2 spaces.
# ----------------
# What it will do is, run below 2 statements while this condition is true -> while i < 10:
#   print(i)
#   i = i + 1
# As i is 5, python will go back to the while statement, and checks the condition again. So i will be 6, 7, 8, 9, 10...
# 10 < 10, stictly speaking no it's not. The condition is false, and below statements will be ignored:
#   print(i)
#   i = i + 1
while i < 10:
    print(i)
    i = i + 1

if i == 10:
    print(i, "is ten")