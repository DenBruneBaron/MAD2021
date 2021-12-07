import linweighreg
import numpy as np
import matplotlib.pyplot as plt

# Reads the data from the file
raw = np.genfromtxt('men-olympics-100.txt', delimiter=' ')
transposed = raw.T

# Extract the first "row", (index 1, since the array is 0-indexed) from raw
OL_year = raw[:,0].T

# Extract the second "row", (index 1, since the array is 0-indexed) from raw
OL_run_times = raw[:,1].T

# Create lamda values for LOOCV
lambda_values = np.logspace(-8, 0, 100, base=10)
print("lambda shape:", lambda_values.shape)
#print(lambda_values)

#Olympic years
x = OL_year 

#First place values
y = OL_run_times

model_all = linweighreg.LinearRegression()
model_all.fit(x,y)
all_weights = model_all.w
print("----------------------------------------")
for i in range(len(all_weights)):
    print("\tw%i : %s" %(i, model_all.w[i]))
print("----------------------------------------")


N = len(y)

def RMSE(t,tp):
    res = np.sqrt(np.square(np.subtract(t,tp)).mean())
    print(res)
    return(res)

for i in lambda_values:
    model_LOOCV = linweighreg.LinearRegression()
    LOOCV_res = model_LOOCV.fit_LOOCV(x, y, i, N)
    loss = RMSE(LOOCV_res.w, y)
    print("lam=%.10f and loss=%.10f" % (lambda_values, loss))



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
