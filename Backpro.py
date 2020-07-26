"""
Deep learning based on Backpropagation algorithm
Description: Writing a backpropagation algorithm from scratch.

@Author: Reza Torabi
"""


from math import exp
#from random import seed
from random import random
import matplotlib.pyplot as plt

# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
	hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	network.append(output_layer)
	return network

# Calculate neuron activation for an input
def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation

# Transfer neuron activation
def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))

# Forward propagate input to a network output
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs

# Calculate the derivative of an neuron output
def transfer_derivative(output):
	return output * (1.0 - output)

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(expected[j] - neuron['output'])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

# Update network weights with error
def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] += l_rate * neuron['delta']

# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
    RCE=[]
    for epoch in range(n_epoch):
    	sum_error = 0
    	for row in train:
    		outputs = forward_propagate(network, row)
    		expected = [0 for i in range(n_outputs)]
    		expected[row[-1]] = 1
    		sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
    		backward_propagate_error(network, expected)
    		update_weights(network, row, l_rate)
    	RCE.append(sum_error)
    	print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
    return RCE

# Make a prediction with a network
def predict(network, row):
	outputs = forward_propagate(network, row)
	return outputs.index(max(outputs))

#Parameters
n_hidden = 10
l_rate = 0.7
n_epoch = 200

# Test training backprop algorithm
dataset = [[1,0,0,0,0],
	[0,1,0,0,1],
	[0,0,1,0,2],
	[0,0,0,1,3]]


n_inputs = len(dataset[0]) - 1
n_outputs = len(set([row[-1] for row in dataset]))
network = initialize_network(n_inputs, n_hidden, n_outputs)
y = train_network(network, dataset, l_rate, n_epoch, n_outputs)
#for layer in network:
#	print(layer)

x = range(n_epoch)
#plot RCE versus epoch
plt.plot(x,y)
plt.title('Training the network')
plt.xlabel('Epoch')
plt.ylabel('Reconstruction error')
plt.show()

for row in dataset:
	prediction = predict(network, row)
	print('Expected=%d, Predicted=%d' % (row[-1], prediction))