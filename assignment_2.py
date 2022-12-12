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

def gradient(lmb, w, b, sign, y, x):
	return np.array([lmb*w, 0]) if sign else np.array([(lmb*w-y*x),-y])

def SVM:
	

gamma = 0.001

print(gradient(1,1,1,True,1,1))





  

 
