import numpy as np

#Exercise 2 - Declare the variable L
L = [1,2,3.5,4, "Hello"]

#Exercise 3 - Python function
def myFunc(x,y,z):
        lst = [x,y,z]
        odd = [num for num in lst if num % 2 == 1]
        print(max(odd))

myFunc(13, 77, 11)