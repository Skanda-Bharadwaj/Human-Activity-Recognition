'''
This file helps in training the HAR database
'''
import os
from numpy import loadtxt
from ANN import ANN 


directory_path = os.path.dirname(os.path.realpath(__file__))

'''
Get the training, testing, and cross-validation data and corresponding training_labels
The trainig  set is made up of 500 signals
The cross validation set is made up of 116 signals
The testing set is made up of 111 signals
NOTE : 
		1. The data loaded is shuffled randomly before hand 
'''
training_data = loadtxt(directory_path+'/HAR-Dataset/HAR_data_train.txt')
training_labels = loadtxt(directory_path+'/HAR-Dataset/HAR_labels_train.txt')

cross_validation_data = loadtxt(directory_path+'/HAR-Dataset/HAR_data_cv.txt')
cross_validation_labels = loadtxt(directory_path+'/HAR-Dataset/HAR_labels_cv.txt')

test_data = loadtxt(directory_path+'/HAR-Dataset/HAR_data_test.txt')
test_labels = loadtxt(directory_path+'/HAR-Dataset/HAR_labels_test.txt')

'''
Run the network
'''

no_hidden_neurons = 591
learning_rate = 0.002
no_of_epochs = 50

network = ANN(no_hidden_neurons, learning_rate, no_of_epochs)


(weights_hidden_layer, weights_output_layer, cross_validation_accuracy) = network.train(training_data, training_labels, cross_validation_data, cross_validation_labels)
print '\nFor the the given values:\n1.No of hidden neurons:{}\n2.Learning rate:{}\n3.No of Epochs:{}\n\tCV Accuracy is:{}%'.format(no_hidden_neurons, learning_rate,no_of_epochs,cross_validation_accuracy)

testing_accuracy = network.test(test_data, test_labels, weights_hidden_layer, weights_output_layer)
print '**********************************\nTesting Accuracy : {}%\n**********************************'.format(testing_accuracy)