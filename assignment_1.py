import numpy as np
import matplotlib.pyplot as plt

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

	"""	IMPLEMENTATION """
	"""	-------------- """

# Compute least squares approximation via Levenberg-Marquardt
def levmarq(func,x0):
	pass




# Compute residual
def residualfunc(x):
	pass




	