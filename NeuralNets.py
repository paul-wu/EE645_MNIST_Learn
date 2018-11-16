'''
Implementation of neural network

11/15/2018   version 0.1
'''

import math
import random


class NeuralNet:

	'''
	The varaible 'Layer' is an integer array which specifies
	the number of neurals in each layer, the first layer is
	the input layer and the last is output layer.
	'''
	def __init__(self, Layer):
		self._layer = len(Layer)
		self.nets, Father = [], []
		for i in range(self._layer):
			current_layer = []
			for j in range(Layer[i]):
				current_layer.append(self.newNeural(Father))
			self.nets.append(current_layer)
			Father = current_layer
		
		
	# Build a new neural
	def newNeural(self, Father):
		data = {}
		data.update({'weights':[random.uniform(-1,1) for i in range(len(Father))]})
		data.update({'derivative':[0]*len(Father)})
		data.update({'father': Father})
		data.update({'value': 0})
		return data
		
	# We use sigmoid function as activation function
	def activation(self, x, a):
		x = math.exp(x*a)
		return x/(x+1)
	
	# 'a' is the pointor to previous layer, 'b' is the weights
	def dotProduct(self, a, b):
		l = len(a)
		if(len(b) != l):
			print("Dimension do not match for inner product.")
			return
		res = 0
		for i in range(l):
			res += a[i]['value']*b[i]
		return res
	
	# Evaluate the network with 'input' and output the binarilized output
	def evaluate(self, input):
		n = len(input)
		if len(self.nets[0]) != n:
			print('Inpute size do not match with the nets.')
			return
		# Initialize inpute layer
		for i in range(n):
			self.nets[0][i]['value'] = input[i]
		for j in range(1,self._layer):
			current_layer = self.nets[j]
			for k in current_layer:
				k['value'] = self.activation(self.dotProduct(k['father'], k['weights']), 1)
		# Binarilize the output
		output = []
		for t in self.nets[self._layer-1]:
			if t['value'] > 0.5:
				output.append(1)
			else:
				output.append(0)
		return output
	
	# Print values in each layer nicely
	def printLayers(self):
		for i in self.nets:
			v = []
			for j in i:
				v.append(j['value'])
			print(v)
			
	# Allow to change weights manually
	def modifyWeights(self, i,j, k, newWeight):
		self.nets[i][j]['weights'][k] = newWeight
	
	''' Using back propagation to calculate derivative,
		here we consider the l_2 loss
	'''
	def backPropagation(self, trueLabel):
		# Last layer
		layer = self.nets[self._layer-1]
		for i in len(layer):
			const = 2*(layer[i]['value'] - trueLabel[i])*layer[i]['value'](1-layer[i]['value'])
			for j in range(len(layer[i]['derivative'])):
				layer[i]['derivative'][j] = const*layer[i]['father'][j]['value']
		
		# Internal layers
		for k in range(2, self._layer - 1):
			index = self._layer - k
			layer = self.nets[index]
			for n in range(len(layer)):
				neural = layer[n]
				pre_neural = layer[index+1]
				const, temp = (1-neural['value']), 0
				for pre_n in pre_neural:
					temp += pre_n['weights'][n]
				const *= temp
				for t in range(len(neural['derivative'])):
					neural['derivative'][t] = const*neural['father'][t]['value']
	
	# Update weigth according to derivatives
	def updateWeights(self, l_step):
		for i in self.nets:
			for j in i:
				for k in range(j['weights']):
					j['weights'][k] -= l_step*j['derivative'][k]
					
	
	''' Gradient decent algorithm in one step, train[0] is the input,
		train[1] is the lable
	'''
	def gradientDecent(self, train, l_step):
		input, lable = tain[0], train[1]
		
		# Evaluate on current sample
		self.evaluate(input)
		
		# Compute derivative
		self.backPropagation(label)
		
		# Gradient Decent
		self.updateWeights(l_step)
	
	
	
a = NeuralNet([3,30,2])

b = a.evaluate([2,-10,-100])

#a.modifyWeights(2,0,1,-1)

print(b)

