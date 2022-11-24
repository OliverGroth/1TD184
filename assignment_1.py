# Optimisation - Assignment 1 - Levenberg-Marquardt

""" Write a implementation of the Levenberg-Marquardt algorithm 
	for the solution of an arbitratry nonlinear least squares problem
	The header of the function should look like this:

	function [x,resnorm,residual] = levmarq(func,x0) (MATLAB syntax)

	The output data is the solution vector x, the residual norm resnorm
	and the residual vector residual. The input data should be a string
	variable func with the name of the function defining your least 
	squares problem and an initial solution x0. I fyou find it useful,
	you can add a third argument options that passes various
	optimisation parameters to your function

	If the user supplies the gradient of the residual vector, this
	should of course be used. The function computing the residual should
	then have a header like function [r,gradr] = residualfunc(x). It 
	should also be possible to call your Levenberg-Marquardt function
	without supplying the gradient. In this case your implementation
	must calculate an approximation of the gradient using a
	numerical approximation such as finite differencing.

	The choise of the regularisation parameter mu in the algorithm
	needs particular attention. At an initial stage, you can use a
	fixed value, but in order for the function to be reliable for a 
	variety of problems, some kind of updating strategy will be
	necessary. Hints given on this in the instructions. """


""" Notes on the Levenberg-Marquardt algorithm

	For cases with multiple minima, algorithm converges to the global
	minima only if the starting guess beta_0 is close enough to the final
	solution.

	In each step, the parameter vector beta is replaced by a new estimate
	beta + delta. To determine delta, the function f(x_i,beta+delta) is
	approximated by its linearization f(x_i,beta) + J_i * delta, where
	J_i is the partial derivative of f with respext to beta (gradient of
	f with respect to beta).

	The sum S(beta) of square deviations has its minumum at a zero
	gradient with respect to beta. The first-order approximation of
	f(x_i,beta+delta) gives
	S(beta+delta)≈Sum(from 1 to m)[y_i - f(x_i,beta)-J_i*delta]^2
	or in vector notation
	S(beta+delta)≈||y-f(beta)-J*delta||^2 which can also be written as
	[y-f(beta)]^T[y-f(beta)]-2[y-f(beta)]^T*J*delta+delta^T*J^T*delta

	Taking the derivative of S(beta+delta) with respect to delta and
	setting the result to zero gives
	(J^T*J)delta=J^T[y-f(beta)] where J is the jacobian matrix, whose
	i-th row equals J_i, and where f(beta) and y are the vectors with
	the i-th component f(x_i,beta) and y_i respectively. The above
	expression for beta comes under the Gauss-Newton method. The Jacobian
	matrix as defined above is not (in general) a square matrix, but a 
	rectangular matrix of size mxn where n is the number of parameters.
	Matrix multiplication (J^T*J) yields the required nxn square matrix.

	Levenberg's contribution is to replace this equation by a "damped version"
	(J^T*J + lambda*I)delta = J^T[y-f(beta)]

	The (non-negative) damping factor lambda is adjusted at each iteration.
	If reduction of S is rapid, a smaller value can be used, bringing the
	algorithm closer to the Gauss-Newton algorithm, whereas if an iteration
	gives insufficient reduction in the residual, lambda can be increased,
	giving a step closer to the gradient-descent direction. Note that the
	gradient of S with respect to beta equals -2(J^T[y-f(beta)])^T, therefor,
	for large values of lambda, the step will be taken approximately in the 
	direction opposite of the gradient. If either the length of the
	calculated step delta or the reduction of sum of squares from the latest
	parameter vector beta+delta fall below predefined limits, iteration stops,
	and the last parameter vector beta is considered to be the solution.

	More to read on how to choose the damping parameter, but lets get it 
	working to begin with.
"""

import numpy as np

def levmarq(func, x, grad = None):

	# beta approximation
	# f(x_i,beta+delta) ≈ f(x_i,beta) + J_i*delta
	# J_i = df(x_i,beta)/dbeta

	# if gradient given, above is easy, however should work without given gradient
	# so we have to create an approximation of the gradient


def gradient():

	# Computes gradient by finite differencing
	# Gradient of S with respect to beta equals -2(J^T[y-f(beta)])^T
	# Gradient of 





return x, resnorm, residual
