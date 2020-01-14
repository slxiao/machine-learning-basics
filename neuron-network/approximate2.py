from sknn.mlp import Regressor
from sknn.mlp import Layer
import numpy as np
import matplotlib.pyplot as plt

# Design Network
hiddenLayer = Layer("Rectifier", units=6)
outputLayer = Layer("Linear", units=1)

nn = Regressor([hiddenLayer, outputLayer], learning_rule='sgd', learning_rate=.001, batch_size=5, loss_type="mse")

# Generate Data
def cubic(x):
    return x**3 + x**2 - x - 1

def get_cubic_data(start, end, step_size):
    X = np.arange(start, end, step_size)
    X.shape = (len(X),1)
    y = np.array([cubic(X[i]) for i in range(len(X))])
    y.shape = (len(y),1)
    return X,y

# Train Model
X,y = get_cubic_data(-2,2,.1)
nn.fit(X,y)

# Predict
predictions = nn.predict(X)

# Visualize
plt.plot(predictions)
plt.plot(y)
plt.savefig("approximate2.png")