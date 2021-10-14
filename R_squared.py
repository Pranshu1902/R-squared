import numpy as np
from numpy.core.fromnumeric import var
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


rng = np.random.RandomState(1000)
x = rng.random(10)
y = x**2 + x + 1
plt.ylim(0,4)
plt.scatter(x,y)

mean = y.mean()

means = [mean]*10
plt.plot(x,means, color="black", label="mean")

# creating the model and fiting it

X = x[:,np.newaxis]

model = LinearRegression(fit_intercept=True)

model.fit(X,y)
m = model.coef_
c = model.intercept_
y_model = x*m + c

plt.plot(x,y_model,color="red", label="ML model")
plt.legend(loc="best")
plt.xlabel("X")
plt.ylabel("Y")


# Calculating R squared

var_mean = 0
for i in y:
    var_mean += (mean-i)**2


var_line = 0
def distance(x,y,models):
    global var_line
    slope = models.coef_
    intercept = models.intercept_
    val = abs(slope*x + intercept - y)
    root = (1+slope**2)**0.5
    var_line += val/root
for i in range(len(x)):
    distance(x[i],y[i],model)

# R squared calculation
r_squared = 1-(var_line/var_mean)
print("R squared =", r_squared[0])

plt.title("R squared = {}".format(r_squared[0]))

plt.savefig("output.png")

plt.show()
