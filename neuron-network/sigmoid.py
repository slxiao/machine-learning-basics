import math
import numpy as np
import matplotlib.pyplot as plt

upper, lower = 6, -6
num = 100

def sigmoid_activation(x):
    if x > upper:
        return 1
    elif x < lower:
        return 0
    #return 1/(1+math.pow(1000, -x))
    return 1/(1+np.exp(-x))

vals = [sigmoid_activation(x) for 
       x in np.linspace(lower, upper, num=num)]

plt.plot(np.linspace(lower, 
                     upper, 
                     num=num), vals)

plt.title('Sigmoid')
plt.savefig("sigmoid.png")