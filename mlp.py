# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 16:20:17 2018

@author: Shushan, Julia, Cornelis

source0: http://scikit-learn.org/stable/auto_examples/neural_networks/plot_mnist_filters.html#sphx-glr-auto-examples-neural-networks-plot-mnist-filters-py
source1: http://scikit-learn.org/stable/auto_examples/neural_networks/plot_mnist_filters.html#sphx-glr-auto-examples-neural-networks-plot-mnist-filters-py


"""

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import csv
import matplotlib.pyplot as plt

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


# set up the network architecture
    """
    mlp = MLPClassifier(hidden_layer_sizes=(number_of_neurons, ), max_iter=iter_number, solver='sgd', tol=1e-4, random_state=1, learning_rate_init=learning_rate)
    mlp.fit(images_train, labels_train)
    loss_lst = mlp.loss_curve_
    test_score = mlp.score(images_test, labels_test)
    return (loss_lst, test_score)
    """

# plot function
def plot_learning_curve(estimator, title, X, y, ylim , cv, n_jobs):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------

    title : string
        Title for the chart.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).

    train_sizes

    train_scores

    test_scores

    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes = np.linspace(.1, 1.0, 5)

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std  = np.std(train_scores,  axis=1)
    test_scores_mean  = np.mean(test_scores,  axis=1)
    test_scores_std   = np.std(test_scores,   axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


#import the data and separate the images and labels
images_train, labels_train = read_data('train.csv')
images_test,  labels_test  = read_data('test.csv')

#normalise the image pixel values so that they range from 0 to 1
images_train = images_train / 255
images_test  = images_test  / 255

N = np.arange(10,110,10);
C = np.arange(1,6,2);
I = np.array([10,50,100]);

train_scores = np.zeros([N.size, C.size, I.size]);
test_scores  = np.zeros([N.size, C.size, I.size]);

for i in range(N.size):              # number of neurons
    for j in range(C.size):          # learning rate
        for k in range(I.size):      # number of iterations
            n = N[i];
            c = C[j];
            it_num = I[k];
            mlp = MLPClassifier(hidden_layer_sizes=(n, ),
                    max_iter=it_num, solver='sgd', tol=1e-4,
                    random_state=1, learning_rate_init=(c/10))

            mlp.fit(images_train, labels_train)

            train_score = mlp.score(images_train, labels_train)
            test_score = mlp.score(images_test, labels_test)
            train_scores[i,j,k] = train_score;
            test_scores[i,j,k]  = test_score;
            print("Fitting NN with {} neurons, {} learning rate, {} max iteration number".format(n, c/10, it_num))
            print("Train set score (accuracy): %f" % train_score)
            print("Test set score (accuracy): %f" % test_score)

### for plotting

plt.plot(I, train_scores[8,0,:])
plt.plot(I, test_scores [8,0,:])
plt.show();

plt.plot(C/10, test_scores[0,:,0]);
plt.plot(C/10, test_scores[2,:,0]);
plt.plot(C/10, test_scores[4,:,0]);
plt.plot(C/10, test_scores[8,:,0]);
#plt.plot(C/10, train_scores[5,:,1]);
#plt.axis([10,100,.9,1]);
plt.grid();
plt.legend(["10 neurons", "30 neurons", "50 neurons", "90 neurons"]);
plt.title("Scores vs learning rate");
plt.ylabel("Score");
plt.xlabel("learning rate");
plt.show();

plt.plot(N, test_scores.max(2).max(1));
plt.plot(N, train_scores.max(2).max(1));
plt.axis([10,100,.9,1]);
plt.grid();
plt.legend(["Cross validation","Training"]);
plt.title("Scores vs number of neurons");
plt.ylabel("Best score");
plt.xlabel("Number of neurons");
plt.show();


ylim=(0.7, 1.01)
n_jobs=1
cv = None
#digits = load_digits()
#X, y = digits.data, digits.target
X = images_train;
y = labels_train;
title = "Learning Curves (MLPClassifier)"
estimator = MLPClassifier()
plot_learning_curve(estimator, title, X, y, ylim, cv, n_jobs)
plt.show()



