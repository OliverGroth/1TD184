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