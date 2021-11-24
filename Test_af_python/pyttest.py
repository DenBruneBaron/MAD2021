import numpy as np
import scipy as sci
import sympy as sym
from sympy.abc import x,y
from sympy.interactive.printing import init_printing
init_printing()

#how to calculate definite integral with one variable in python
x = sym.symbols('x', real=True)
y=3*x
res = sym.integrate(y,(x,5,7)) #(x, lower limit, upper limit)
print("Result of an definite integral:", res)


#how to calculate indefinite integral with one variable in python
y = sym.sin(x)
res2 = sym.integrate(y,x)
print("Result of an indefinite integral:", res2)

