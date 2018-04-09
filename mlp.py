
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 16:20:17 2018

@author: Shushan
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
import csv
import matplotlib

#function to read the data into a numpy matrix
def read_data(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    matrix = np.array(data, dtype = int)
	 # separate labels from samples
    data = matrix[:,1:]
    labels = matrix[:,0]
    return (data, labels)

#set up the network architecture
def NN(number_of_neurons, learning_rate, iter_number):
    mlp = MLPClassifier(hidden_layer_sizes=(number_of_neurons, ),
            max_iter=iter_number, solver='sgd', tol=1e-4, random_state=1,
            learning_rate_init=learning_rate)
    mlp.fit(images_train, labels_train)
    loss_lst = mlp.loss_curve_
    test_score = mlp.score(images_test, labels_test)
    return (loss_lst, test_score)

#import the training data and separate the images and labels
images_train, labels_train = read_data('train.csv')
#import the testing data and separate the images and labels
images_test, labels_test = read_data('test.csv')

#normalise the image pixel values so that they range from 0 to 1
images_train = images_train / 255
images_test  = images_test  / 255

results = {}
for n in range(10, 110, 10):
    for c in range(1, 12, 3):
        for it_num in [10, 50, 100]:
            print('Fitting NN with {} neurons, {} learning rate, {} max iteration number'.format(n, c/10, it_num))
            results[(n,c, it_num)] = NN(n, c/10, it_num)


