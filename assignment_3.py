"""
Interesting link: https://machinelearninggeek.com/solving-transportation-problem-using-linear-programming-in-python/#Transportation_Problem
More interesting: https://personal.utdallas.edu/~scniu/OPRE-6201/documents/TP1-Formulation.pdf
We want to find the optimal way (cost-wise) to transport goods from some suppliers to some demanders. A transportation
from some factory (supply) to some ware house (demand) is associated with some cost c_ij. A supplier i has supply S_i
and a warehouse has demand D_j. x_ij denotes the quantity transported. 
The total cost for shipping x_ij is c_ij*x_ij. Thus, we want to minimize the sum of this, giving objective function
minimize sum_i=1^m sum_j=1^n c_ij x_ij.
We do also need constraints. The total outflow from a given factory i is sum_j=1^n x_ij <= S_i, i.e the outflow
must be less than or equal to the supply of the factory. In the same way for a warehouse j we need the inflow
to be at least equal to the demand, i.e sum_i=1^m x_ij >= D_j. Thus we can formulate the linear programming problem as

minimize sum_i=1^m sum_j=1^n c_ij x_ij
Subject to:
	sum_j=1^m x_ij <= S_i for each i
	sum_i=1^n x_ij >= D_j for each j
	x_ij >= 0 for all i,j

This should be the solution to problem 1.

Now we want to solve an actual problem. We have three factories and four warehouses. Each has a given supply, demand,
and transportation cost in relation to another entity. Thus, i = 1,2,3 ; j = 1,2,3,4. Then our problem is

minimize 10*x_11 + 0*x_12 + 20*x_13 + 11*x_14 + 12*x_21 + 7*x_22 + 9*x_23 + 20*x_24 + 0*x_31 + 14*x_32 + 16*x_33 + 18*x_34
Subject to: 
	
"""

from scipy.optimize import linprog
import numpy as np

A_ub = np.array([[1,1,1,1,0,0,0,0,0,0,0,0],
				[0]*12,[0]*12,[0]*12,
				[0,0,0,0,1,1,1,1,0,0,0,0],
				[0]*12,[0]*12,[0]*12,
				[0,0,0,0,0,0,0,0,1,1,1,1],
				[0]*12,[0]*12,[0]*12])

A = np.array([[1,1,1],[1,1,1],[1,1,1],[1]*3])

print(A)

A_eq = np.array([[1,0,0,0,1,0,0,0,1,0,0,0],
				[0]*12,[0]*12,
				[0,1,0,0,0,1,0,0,0,1,0,0],
				[0]*12,[0]*12,
				[0,0,1,0,0,0,1,0,0,0,1,0],
				[0]*12,[0]*12,
				[0,0,0,1,0,0,0,1,0,0,0,1],
				[0]*12,[0]*12])
print(A_eq)
b_ub = np.array([25,0,0,0,55,0,0,0,35,0,0,0])
print(b_ub)
b_eq = np.array([15,0,0,45,0,0,30,0,0,25,0,0])
print(b_eq)

#b_eq = np.array([25,55,35])
#A_ub = np.array([[1,1,1],[1,1,1],[1,1,1],[1,1,1]])
#b_ub = np.array([15,45,30,25])
c = np.array([10,0,20,11,12,7,9,20,0,14,16,18])

#c = np.array([[10,0,20,11],[12,7,9,20],[0,14,16,18],[15,45,30,25]])

res = linprog(c,A_ub,b_ub,A_eq,b_eq)
print(res.x)

