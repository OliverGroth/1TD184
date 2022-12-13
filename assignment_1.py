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
	J_i is the partial derivative of f with respect to beta (gradient of
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
import scipy.optimize as opt

func = lambda x, t: x[0]*np.exp(x[1]*t)

ver = lambda x, t, y: x[0]*np.exp(x[1]*t) - y


def levmarq(func, x, t, y, l, grad = None):
	
	n = len(x)
	m = len(t)

	J = jacobian(func, x, t)
	Jt = np.transpose(J)

	mat1 = np.matmul(Jt,J) + l*np.identity(n)
	mat1inv = np.linalg.inv(mat1)
	res = residual(t,y,func,x)
	mat2 = np.matmul(Jt,np.transpose(res[0]))
	delta = np.matmul(mat1inv,mat2)

	x = x+delta
	err = res[1]

	return x, res[0], res[1]

# Detta �r en kommentar 
def jacobian(func, x, t, h = 10**(-5)):

	n = len(x) # antal parametrar
	m = len(t) # antal datapunkter

	J = np.zeros([m,n]) # jacobianen

	for i in range(n):
		z = np.zeros(n)
		z[i] = h
		for j in range(m):
			d = (func(x+z,t[j])-func(x-z,t[j]))/(2*h)
			J[j,i] = d

	return J


def residual(t,y,func,x):

	m = len(t)
	res = np.zeros(m)

	for i in range(m):
		res[i] = y[i] - func(x,t[i])

	return res, np.linalg.norm(res)


def damp(func, x, t, y, l, nu, res0):
	# 1 förra värdets residualnorm (bra om sparat slippa räkna igen)
	# RETURNERA DETTA SÅ ATT DET KAN SKICKAS MED SOM INPUT

	# 2 beräkna residualnorm efter steg med l, kalla res1
	res1 = levmarq(func, x, t, y, l)[2]

	# 3 beräkna residualnorm efter steg med l/nu, kalla res2
	res2 = levmarq(func, x, t, y, l/nu)[2]

	# jämför 1, 2 och 3 

	# om 1 bäst:
	# while-loop där vi tar fram nya residualnormer för
	# efter steg med l*v^k (k antal iterationer)
	# (l senaste lambdat)

	res = res0 + 1

	if min(res1, res2) > res0:
		while res > res0 and l < 100:
			l = l*nu
			res = levmarq(func, x, t, y, l)[2]
		return l, res

	# om 2 eller 3 bäst, använd det deltat för 
	# nytt x och spara residualen för nästa jämförelse 	


	if res2 < res1:
		return l/nu, res2
	else:
		return l, res1
	

def main(func,t,y,x0):


	l0 = 0.5 #lambda
	nu = 1.5 # ny!!

	res = [0,0]	
	
	maxiter = 10**4
	TOL = 10**(-4)
	iters = 0
	err = 2*TOL
	res0 = residual(t,y,func,x0)[1]
	x = x0
	xdiff = 2*TOL

	while iters < maxiter and xdiff > TOL:
		xprev = x
		l, res0 = damp(func, x, t, y, l0, nu, res0)
		x, res[0], res[1] = levmarq(func, x, t, y, l)
		xdiff = np.linalg.norm(xprev-x)
		iters += 1
		err = res[1]
	print(iters)
	return x, err





t = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
y = np.array([7.2, 3.0, 1.5, 0.85, 0.48, 0.25, 0.20, 0.15])
x0 = np.array([-20, 5])

x, err = main(func,t,y,x0)
x_ver = opt.least_squares(lambda x: ver(x,t,y),x0)
print(x)
print(err)
print(x_ver)
print(np.linalg.norm(x_ver.fun))
