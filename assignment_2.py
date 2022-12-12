# Assignment 2: Breast cancer diagnosis using a Support Vector Machine
 
# https://uppsala.instructure.com/courses/65352/assignments/154750?module_item_id=627882
"""  
We want to find line/plane w^T*x + b = 0 such that w^T*x + b >= 1  for y_i = 1
 	   										       w^T*x + b <= -1 for y_i = -1
Here y_i are two classes of tumors, either malignant (bad) or benign (good).
Ideally we thus want the points to be completely separated into two clusters, which we seperate with the 
line/plane w^T*x + b = 0. This is however idealistic, outliers and missclassifications ruin this. Therefore
we relax the inequalities by a term xi_i, such that w^T*x + b >= 1  - xi_i for y_i = 1
 	   										        w^T*x + b <= -1 + xi_i for y_i = -1
This is equivalent to y_i*(w*x_i + b) >= 1
We can have two planes, one at the edge of one cluster and one at the edge of the other. The plane 
that then maximizes the margin between the plane and the clusters i then the plane in the middle of 
these two planes. The distance between these two planes is 2/norm(w), thus to maximize distance
we want to minimize norm(w) = w^T*w. 
This relaxation moves the lines towards eachother (even past?). If we also penalise this relaxation 
(because it is not ideal?) we get an optimisation problem:
minimise f(w,b) = 1/2*w^T*w + C*sum_{i=1}^m xi_i
sub. to y_i(w^T*x_i + b) >= 1 - xi_i, i = 1,...,m ; xi_i >= 0
This is a problem which can be solved with quadratic programming (https://en.wikipedia.org/wiki/Quadratic_programming).
But this is actually least squares?? Can use a least squares algorithm like gradient descent? 
INTRESSANT: https://towardsdatascience.com/implementing-svm-from-scratch-784e4ad0bc6a
QUESTIONS:
(*) How do we optimize to get w and b?
(*) Implementation???
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, model_selection

def gradient(lmb, w, b, sign, y, x):
	return np.array([lmb*w, 0]) if sign else np.array([(lmb*w-y*x),-y])

def constraint(x, y, w, b):
	return y*(np.dot(w,x)+b) >= 1

def SVM(X, y, lmb=0.01, gamma=0.001):

	maxiter = 1000
	w = np.zeros(X.shape[1])
	b = 0

	# massa skit i början

	for _ in range(maxiter): #eller tills nöjd typ?
		for i, x in enumerate(X): #loopa genom skiten
			D = gradient(lmb, w, b, constraint(x, y[i], w, b), y[i], x)

			w -= gamma*D[0]
			b -= gamma*D[1]

	return w, b

# Below from article to test model

X, y = datasets.make_blobs(
    n_samples=250, n_features=2, centers=2, cluster_std=1.05, random_state=1
)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1, shuffle=True, random_state=1)

y_train = np.where(y_train <= 0, -1, 1)

w,b = SVM(X_train,y_train)

est = np.dot(X_test, w) + b
prediction = np.sign(est)
result = np.where(prediction == -1, 0, 1)

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true==y_pred) / len(y_true)
    return accuracy

print("SVM Accuracy: ", accuracy(y_test, result))





  

 
