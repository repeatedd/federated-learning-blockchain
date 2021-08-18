import backprop as bp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer


df = pd.read_csv('data.csv')

X = np.array(df.drop('charges', axis=1))
y = np.array(df['charges'])
# X, y = load_breast_cancer(return_X_y=True)
y = y.reshape((len(y), 1))
print(X.shape)

# Split Data
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)
train_X = train_X.T
test_X = test_X.T

print(train_X.shape)
# Normalize
mean = np.mean(train_X, axis = 1, keepdims=True)
std_dev = np.std(train_X, axis = 1, keepdims=True)
train_X = (train_X - mean)/std_dev
test_X = (test_X - mean)/std_dev

train_y = train_y.T
test_y = test_y.T

train_X.shape, train_y.shape, test_X.shape, test_y.shape

description = [{"num_nodes" : 12, "activation" : "relu"},
            #    {"num_nodes" : 12, "activation" : "relu"},
               {"num_nodes" : 1, "activation" : "relu"}]

model = bp.NeuralNetwork(description,12,"mean_squared", train_X, train_y, learning_rate=0.001)

for i in range(len(model.layers)):
    print(model.layers[i].W)
    print(model.layers[i].b)

history = model.train(2000)    

plt.plot(history)

acc = model.calc_accuracy(train_X, train_y, "RMSE")
print("MSE on the training set is = {}".format(acc))

acc = model.calc_accuracy(test_X, test_y,"RMSE")
print("MSE on the test set is = {}".format(acc))



