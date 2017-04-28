'''
This file contains the implementation of the neural network using Backpropagation algorithm
'''
import numpy

class ANN:
	'''
	This class implements the Neural network.
	'''
	def __init__(self, hidden_neurons, learning_rate, epochs):
		'''
		This is the initialization method; initalizes basic hyper parameters of the Neural network
		'''
		self. hidden_neurons = hidden_neurons
		self.learning_rate = learning_rate
		self.epochs = epochs
		self.a = 1.7591
		self.b = 2.0/3
		self.hidden_layer_weights = None
		self.output_layer_weights= None

	def train(self, input, expected_output, cross_validation_input, cross_validation_expected_output):
		'''
		This is method that is used by the calling object for training the Neural network
		@return: The updated weights of the Neural network

		'''
		examples_length , feature_length = input.shape
		examples_length, output_nodes = expected_output.shape
		self._randomizewieghts(feature_length, output_nodes)
		self._train_network(input, expected_output, examples_length, feature_length, output_nodes)
		accuracy = self._get_cross_validation_accuracy(cross_validation_input, cross_validation_expected_output)
		return (self.hidden_layer_weights, self.output_layer_weights, accuracy)

	def _train_network(self,input, expected_output,examples_length, feature_length, output_nodes):
		'''
		Trains the network i.e. performs forward pass, error propagation and weight updation
		'''
		for j in xrange(self.epochs):
			for i in xrange(examples_length):

				output_layer_output_after_activation, input_to_output_layer_with_bias, hidden_layer_output_after_activation, input_with_bias = self._feed_forward(input[i], self.hidden_layer_weights, self.output_layer_weights)
				error = expected_output[i] - output_layer_output_after_activation

				weight_change_factor_hidden_layer, weight_change_factor_output_layer = self._calculate_weight_change_factor(error, input_to_output_layer_with_bias, hidden_layer_output_after_activation, input_with_bias)

				self.hidden_layer_weights = self.hidden_layer_weights + weight_change_factor_hidden_layer
				self.output_layer_weights = self.output_layer_weights + weight_change_factor_output_layer
				print 'Epoch no:{} Sample No:{} Error:{}'.format(j+1,i,error.dot(error.T) )

	def _feed_forward(self, input_signal, weights_hidden_layer, weights_output_layer):
		'''
		Performs forward pass of the Neural network
		@return: output_layer_output_after_activation   :   Output of the Neural network
				 input_to_output_layer_with_bias 	    :   Input to hidden layer with addition of bias
				 hidden_layer_output_after_activation	: 	Ouptut of hidden layer 				    
				 input_with_bias 						:	Input added with bias  
		'''
		input_with_bias = numpy.append(1, input_signal)
		hidden_layer_output = weights_hidden_layer.dot(input_with_bias)
		hidden_layer_output_after_activation = self._tan_sig(hidden_layer_output)

		input_to_output_layer_with_bias = numpy.append(1, hidden_layer_output_after_activation) 
		output_layer_output = weights_output_layer.dot(input_to_output_layer_with_bias)
		output_layer_output_after_activation = output_layer_output

		return (output_layer_output_after_activation, input_to_output_layer_with_bias, hidden_layer_output_after_activation, input_with_bias)

	def _calculate_weight_change_factor(self, error, input_to_output_layer_with_bias, hidden_layer_output_after_activation, input_with_bias):
		'''
		Back propagation core ideology implemented. Propagates error from output. 
		@return: The small weight change factor the weights of the two layers.(To help it network predict better)
		'''
		delta2 = error
		weight_change_factor_output_layer = self.learning_rate * (delta2[:,None]*input_to_output_layer_with_bias)

		delta1 = self.a * self.b * (1-(numpy.square(hidden_layer_output_after_activation))/numpy.square(self.a))*(self.output_layer_weights[:,1:self.hidden_neurons+1].T.dot(delta2))
		weight_change_factor_hidden_layer = self.learning_rate * delta1[:,None]*input_with_bias		

		return (weight_change_factor_hidden_layer, weight_change_factor_output_layer)

	def _get_cross_validation_accuracy(self, cross_validation_input, cross_validation_expected_output):
		'''
		Perfroms cross validation i.e. tests the given cross validation set for hyper paramter tuning
		'''
		return self._get_prediction_accuracy(cross_validation_input, cross_validation_expected_output, self.hidden_layer_weights, self.output_layer_weights)		
		

	def _get_prediction_accuracy(self, signals, output, weights_hidden_layer, weights_output_layer):
		'''
		Run forward pass using the weights acquired after training. Predicts the output class by taking the maximum value postion from the output of the Neural network
		@return: Prediction Accuracy 
		'''
		length = signals.shape[0]
		correctly_predicted_count = 0
		for i in xrange(length):
			predicted_class = numpy.argmax(self._feed_forward(signals[i],weights_hidden_layer, weights_output_layer)[0])
			expected_class = numpy.argmax(output[i])
			#print 'Expected class:{} Predicted class:{}'.format(expected_class+1,predicted_class+1)
			if (predicted_class == expected_class):
				correctly_predicted_count +=1
		return (correctly_predicted_count*1.0/length)*100

	def test(self, test_data, test_label, weights_hidden_layer, weights_output_layer):
		'''
		Test the trained Neural network
		'''
		return self._get_prediction_accuracy(test_data,test_label, weights_hidden_layer, weights_output_layer)

	def _randomizewieghts(self,feature_length,output_nodes):
		'''
		Utility method to randomly initialize weights with a small value
		'''
		delta = 0.12
		self.hidden_layer_weights = numpy.random.rand(self.hidden_neurons, feature_length+1)*2*delta - delta
		self.output_layer_weights = numpy.random.rand(output_nodes,self.hidden_neurons+1)*2*delta - delta

	def _tan_sig(self, value):
		'''
		TanSig function i.e. activation field implementation
		'''
		return self.a * numpy.tanh(self.b*value)

	
	






