import linweighreg
import numpy as np
import matplotlib.pyplot as plt




raw = np.genfromtxt('men-olympics-100.txt', delimiter=' ')

transposed = raw.T

#print(transposed)

#Extract the first "row", (index 1, since the array is 0-indexed) from raw
OL_year = raw[:,0].T
print(OL_year)
#Extract the second "row", (index 1, since the array is 0-indexed) from raw
OL_run_times = raw[:,1].T
print(OL_run_times)
lambda_values = np.logspace(-8, 0, 100, base=10)
print(lambda_values.shape)
#olympic years
x = OL_year 
#First place values
y = OL_run_times
N_len = len(x)

model_all = linweighreg.LinearRegression()
model_all.fit(x,y)
all_weights = model_all.w
print("----------------------------------------")
for i in range(len(all_weights)):
    print("\tw%i : %s" %(i, model_all.w[i]))
print("----------------------------------------")


model_LOOCV = linweighreg.LinearRegression()
fuck = model_LOOCV.fit_LOOCV(x,y,lambda_values, N_len)
print(fuck)



# def get_model(A, y, lamb=0):
#     n_col = A[1]
#     return np.linalg.lstsq(A.T.dot(A) + lamb * np.identity(n_col), A.T.dot(y))
# print(get_model(x,y,lamb=0))

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
#plt.show()

