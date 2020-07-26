"""
Deep learning based on Recirculation algorithm. For more informatin refer to Geoffrey paper
Description: Writing a Recirculation algorithm from scratch.

@Author: Reza Torabi
"""
####################################Libraries##################################
#Import neccessery libraries
from random import random
from math import exp
import matplotlib.pyplot as plt
#%matplotlib inline
####################################Parameters#################################
#Parameters in the code
r = 0.75 #Regression or projection coefficient
l_rate = 3  #Learning rate (epsilon in the paper)
n_hidden = 6 #Number of neurons in hidden layer
n_epoch = 100 #Number of epoch
###############################################################################
#                                   Functions
###############################################################################
##########################Step 1:Initialize a network##########################
#Function:Initialize a network containing initial weigths (plus bias for first layer or hidden layer)
#Descriptions: using random number between -0.5 and 0.5 to generate the weights
# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
	hidden_layer = [{'weights':[random()-0.5 for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[random()-0.5 for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	for j in range(n_outputs):
		output_layer[j]['weights'][-1]=0 #Note that the bias is zero for the second layer (output layer)
	network.append(output_layer)
	return network
###############################################################################
#######################Step 2:Propagation using Recirculation##################
#Function:Calculate neuron activation for an input
#Description:Activate the neurons
def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation
###############################################################################
#Function:Transfer neuron activation
#Description:We use a logistic function (sigmoid) according to equation (2) in the paper
def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))
###############################################################################
#Function:Propagation using Recirculation of input into a network in three time steps
def propagate_recirculate(network, row): #row is the input to the network and is a vector
	inputs = row
##time step 1 (t1)
	new_inputs = []
	for neuron in network[0]:
		activation = activate(neuron['weights'], inputs)
		neuron['aoR'] = transfer(activation) #a_oR real (Hidden)
		new_inputs.append(neuron['aoR'])
	inputs = new_inputs
##time step 2 (t2) 
	new_inputs = []
	out_feedforward = []
	count=0
	for neuron in network[1]:    
		activation = activate(neuron['weights'], inputs)   
		neuron['aiR'] = row[count]
		neuron['out'] = transfer(activation)
		out_feedforward.append(neuron['out'])
		neuron['aiI'] = r * neuron['aiR'] + (1 - r) * transfer(activation) #a_iI Hypothetical (Visible) 		
		new_inputs.append(neuron['aiI'])
		count +=1
	inputs = new_inputs #output of the first loop
##time step 3 (t3) 
	#new_inputs = []
	for neuron in network[0]:        
		activation = activate(neuron['weights'], inputs) 
		neuron['aoI'] = r * neuron['aoR'] + (1 - r) * transfer(activation) #a_oI Hypothetical (Hidden) 
		#new_inputs.append(neuron['aoI'])
	#return inputs #It returns the output of the first loop to calculate the error in the future
	return out_feedforward
###############################################################################
##############################Step 3:Trainig network###########################
#Function:Calculating recirculation error and storing in neurons
#Description:Calculating the difference between real part of visibl/hidden neurons and hypothetical one 
def errors(network):
	for i in reversed(range(len(network))):
		layer = network[i]        
		if i == len(network)-1:        
			for j in range(len(layer)):
				neuron = layer[j]                
				neuron['delta'] = neuron['aiR']-neuron['aiI']            
		else:            
			for j in range(len(layer)):
				neuron = layer[j]
				neuron['delta'] = neuron['aoR']-neuron['aoI']   
###############################################################################
# Function:Update network weights with error
def update_weights(network, l_rate):
	for i in range(len(network)):
		if i == 0:  #means hidden layer
			inputs = network[i+1]  #means output layer    
			for neuron in network[i]:
				for j in range(len(inputs)):  
					neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]['aiI'] 
					neuron['weights'][-1] += l_rate * neuron['delta'] 
		if i == 1:  #means output layer
			inputs = network[i-1]  #means hidden layer    
			for neuron in network[i]:
				for j in range(len(inputs)):  
					neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]['aoR'] 
###############################################################################
#Function:Training network
#Description:Training network for a fixed number of epochs. Errors is updated after a full propagation of the dataset
def train_network(network, train, l_rate, n_epoch, n_outputs):
	mse=[]
	for epoch in range(n_epoch):
		sum_error = 0
		for row in train:
			outputs = propagate_recirculate(network, row)
			expected = [0 for i in range(n_outputs)]
			expected[row[-1]] = 1
			sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
			errors(network)
			update_weights(network, l_rate)
		mse.append(sum_error)
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch+1, l_rate, sum_error))
	return mse
###############################################################################
# Make a prediction with a network
def predict(network, row):
	outputs =  propagate_recirculate(network, row)
	return outputs.index(max(outputs))
###############################################################################
############################# Main body of the code############################
###############################################################################
dataset = [[1,0,0,0,0],[0,1,0,0,1],[0,0,1,0,2],[0,0,0,1,3]] 

n_inputs = len(dataset[0]) - 1 #note that n_inputs should be equal to n_outputs for the recirculation algorithm
n_outputs = len(set([row[-1] for row in dataset]))
network = initialize_network(n_inputs, n_hidden, n_outputs)
y = train_network(network, dataset, l_rate, n_epoch, n_outputs)
x = range(n_epoch)
#plot MSE versus epoch
plt.plot(x,y)
plt.title('Training the network')
plt.xlabel('epoch')
plt.ylabel('mean square error')
plt.show()
#make a prediction
for row in dataset:
	prediction = predict(network, row)
	print('Expected=%d, Predicted=%d' % (row[-1], prediction))
###############################################################################

