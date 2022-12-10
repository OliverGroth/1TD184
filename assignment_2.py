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
we want to minimize w. 

This relaxation moves the lines towards eachother (even past?). If we also penalise this relaxation 
(because it is not ideal?) we get an optimisation problem:

minimise f(w,b) = 1/2*w^T*w + C*sum_{i=1}^m xi_i
sub. to y_i(w^T*x_i + b) >= 1 - xi_i, i = 1,...,m ; xi_i >= 0
"""

  

 
