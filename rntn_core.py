import numpy as np


class RNTNTree:

	def __init__(self, params=None):

		## Defining constants
		self.leaf = False		 # Boolean value to check if current node is a leaf node
		self.label = None		 # The class label for the current node
		self.left = None		 # Left child of the current node
		self.right = None		 # Right child of the current node
		self.word = None		 # Word of the current node if it's a leaf node
		self.word_idx = None	 # Index of the word in the current vocabulary
		self.params = params	 # Dictionary of parameters to be learnt and their gradients

		## Defining input vector
		self.input_vector = None # Concatenation of word vectors of
								 # left and right child

		## Defining output variables
		self.h = None			# Output of the tensor product of child nodes
								# or a word vector if it's a leaf node (1xd)
		self.probs = None		# Probabilities obtained after passing h through softmax (1xC)

	def softmax(self, vector):
		vector = np.exp(vector)
		return vector / np.sum(vector)

	def forward_propagate(self, node):

		## If it's a leaf, predict it's sentiment
		if node.leaf:
			node.h = self.params['L'][:, node.word_idx].reshape(-1, 1).copy()
			node.h = np.tanh(node.h)
			node.probs = self.softmax(np.dot(self.params['Ws'], node.h) + self.params['bs'])
			return

		## Recursing down
		self.forward_propagate(node.left)
		self.forward_propagate(node.right)

		## Computing class probabilities
		node.input_vector = np.vstack([node.left.h, node.right.h])
		node.h = np.dot(np.dot(node.input_vector.T, self.params['V'].T), node.input_vector)[0]
		node.h += np.dot(self.params['W'], node.input_vector)
		node.h = np.tanh(node.h)
		node.probs = self.softmax(np.dot(self.params['Ws'], node.h) + self.params['bs'])

	def back_propagate(self, node, lamda=0.01, del_down=None):
		## Derivative for cross entropy error
		delta = node.probs.copy()
		delta[node.label] -= 1

		## Gradients for softmax weights and bias
		self.params['dbs'] += delta
		self.params['dWs'] += np.dot(delta, node.h.T) + (lamda*self.params['Ws'])

		## Derivative of softmax error
		delta = np.dot(self.params['Ws'].T, delta) * (1 - node.h**2)

		## Add errors from parent (except for the root node)
		if del_down is not None:
			delta += del_down

		## If it's a leaf node, store derivative of word vector
		## and stop propagating
		if node.leaf:
			self.params['dL'][:, node.word_idx] += delta.flatten()  + (lamda*self.params['L'][:, node.word_idx])
			return
		else:
		## Continue propagation...

			## Gradients for weight matrix W and bias b
			self.params['db'] += delta
			self.params['dW'] += np.dot(delta, node.input_vector.T) + (lamda*self.params['W'])

			## Gradient of tensor V
			self.params['dV'] += (delta.T * np.outer(node.input_vector, node.input_vector)[:,:,np.newaxis]) \
								 + (lamda*self.params['V'])

			## Error for children
			S = (self.params['V'] + np.transpose(self.params['V'], axes=[1,0,2])).T
			S = np.dot(np.dot(S, node.input_vector).T, delta)[0]
			delta = (np.dot(self.params['W'].T, delta) + S) * (1 - np.tanh(node.input_vector)**2)

			## Backpropagate through structure
			self.back_propagate(node.left, lamda, delta[:self.params['d']])
			self.back_propagate(node.right, lamda, delta[self.params['d']:])

	def adagrad_update(self, step=0.001, epsilon=1e-12, batch_size=1):
		## Adding historical gradients
		self.params['hdWs'] += self.params['dWs'] ** 2
		self.params['hdbs'] += self.params['dbs'] ** 2
		self.params['hdW'] += self.params['dW'] ** 2
		self.params['hdb'] += self.params['db'] ** 2
		self.params['hdV'] += self.params['dV'] ** 2
		self.params['hdL'] += self.params['dL'] ** 2

		## Updating parameters
		self.params['Ws'] -= step * (self.params['dWs'] / (epsilon + np.sqrt(self.params['hdWs'])))
		self.params['bs'] -= step * (self.params['dbs'] / (epsilon + np.sqrt(self.params['hdbs'])))
		self.params['W'] -= step * (self.params['dW'] / (epsilon + np.sqrt(self.params['hdW'])))
		self.params['b'] -= step * (self.params['db'] / (epsilon + np.sqrt(self.params['hdb'])))
		self.params['V'] -= step * (self.params['dV'] / (epsilon + np.sqrt(self.params['hdV'])))
		self.params['L'] -= step * (self.params['dL'] / (epsilon + np.sqrt(self.params['hdL'])))
