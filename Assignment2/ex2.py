import linweighreg
import numpy as np
import matplotlib.pyplot as plt


raw = np.genfromtxt('men-olympics-100.txt', delimiter=' ')

transposed = raw.T

#print(transposed)

#Extract the first "row", (index 1, since the array is 0-indexed) from raw
input_variables = raw[:,0].T
print(input_variables)
#Extract the second "row", (index 1, since the array is 0-indexed) from raw
target_values = raw[:,1].T
print(target_values)
lambda_values = np.logspace(-8, 0, 100, base=10)
#olympic years
x = input_variables 
#First place values
y = target_values

model_all = linweighreg.LinearRegression()
model_all.fit(x,y)
all_weights = model_all.w
print("----------------------------------------")
for i in range(len(all_weights)):
    print("\tw%i : %s" %(i, model_all.w[i]))
print("----------------------------------------")


plt.title('Mens 100m sprint results')
plt.xlabel('Olympic year')
plt.ylabel('1st place 100m track time')
plt.scatter(x, y)
xplot=np.linspace(1896,2008,100)
poly =np.polyfit(x,y,1)
xplot=np.linspace(1896,2008,100)
poly =np.polyfit(x,y,1)
print("values from polynomial fit:", poly)
yplot = poly[1]+poly[0]*(xplot)
plt.plot(xplot,yplot, c='r')
plt.show()

