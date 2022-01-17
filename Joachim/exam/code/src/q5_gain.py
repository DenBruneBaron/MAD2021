#!/usr/bin/env python

import math

T = (12,5,8,9)
T_0 = (6,1,6,6)
T_1 = (6,4,2,3)

def H(T):
    """
    Entropy
    """
    N = sum(T)

    res  = 0

    for c in T:
        res -= c/N * math.log2(c/N)

    return res


print("H(T): %.4f" %H(T))
print("H(T_0): %.4f" %H(T_0))
print("H(T_1): %.4f" %H(T_1))

gain = H(T) - sum(T_0)/sum(T) * H(T_0) - sum(T_1)/sum(T) * H(T_1)
print("Gain(T,j): %.4f" %gain)

def gini(T):
    """
    Gini index
    """
    N = sum(T)

    res = 1

    for c in T:
        res -= (c/N)**2

    return res

Gini = gini(T) - sum(T_0)/sum(T) * gini(T_0) - sum(T_1)/sum(T) * gini(T_1)
print("gini(T): %.4f" %gini(T))
print("gini(T_0): %.4f" %gini(T_0))
print("gini(T_1): %.4f" %gini(T_1))

print("Gini(T,j): %.4f" %Gini)
