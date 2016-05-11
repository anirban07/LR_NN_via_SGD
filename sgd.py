from math import exp
import random
import numpy as np

# TODO: Calculate logistic
def logistic(x):
    return (1.0 / (1 + np.exp(-x)))

# TODO: Calculate accuracy of predictions on data
def accuracy(data, predictions):
    total = len(predictions)
    correct = 0
    for i in range(0, len(predictions)):
        if ((predictions[i] <= 0.5 and data[i]['label'] == 0) or (predictions[i] > 0.5 and data[i]['label'] == 1)):
            correct += 1
    return (0.0 + correct) / total

class model:
    def __init__(self, structure):
        self.weights=[]
        self.bias = []
        for i in range(len(structure)-1):
            self.weights.append(np.random.normal(size=(structure[i], structure[i+1])))
            self.bias.append(np.random.normal(size=(1, structure[i+1])))
            
    # TODO: Calculate prediction based on model
    def predict(self, point):
        a = self.feedforward(point)
        return a[-1]

    # TODO: Update model using learning rate and L2 regularization
    def update(self, a, delta, eta, lam):
        for i in range(0, len(self.weights)):
            self.weights[i] = self.weights[i] - eta * (lam * self.weights[i] - np.transpose(a[i] * delta[i].item(0)))
            self.bias[i] = self.bias[i] - eta * np.transpose(-delta[i].item(0))

    # TODO: Perform the forward step of backpropagation
    def feedforward(self, point):
        a = []
        a.append(point['features'])
        for i in range(0, len(self.weights)):
            postsynaptic = logistic(np.dot(a[i], self.weights[i]) + self.bias[i])
            a.append(postsynaptic)
        return a
    
    # TODO: Backpropagate errors
    def backpropagate(self, a, label):
        result_delta = []
        L = len(a) - 1
        delta = label - a[-1]
        result_delta.insert(0, delta)
        while L > 1:
            delta = np.dot(delta, np.multiply(a[L - 1], 1 - a[L - 1]))
            delta = np.multiply(self.weights[L - 1], np.transpose(delta))
            L -= 1
            result_delta.insert(0, delta)
        return result_delta

    # TODO: Train your model
    def train(self, data, epochs, rate, lam):
        for i in range(0, epochs * len(data)):
            point = random.choice(data)
            a = self.feedforward(point)
            delta = self.backpropagate(a, point['label'])
            self.update(a, delta, rate, lam)

def logistic_regression(data, lam=0.00001):
    epochs = 100
    eta = 0.05
    m = model([data[0]["features"].shape[1], 1])
    m.train(data, epochs, eta, lam)
    return m
    
def neural_net(data, lam=0.00001):
    m = model([data[0]["features"].shape[1], 15, 1])
    m.train(data, 100, 0.05, lam)
    return m