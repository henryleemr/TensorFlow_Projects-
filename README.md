# Convex Quadratic Solver
-prototype algorithm for solving a quadratic problem using a modified ADMM (alternating direction multiplier method) algorithm using TensorFlow

'''
Minimize    : 0.5 (x.T * P * x) + (q.T * x)
Subject to  :  l ≤ Ax ≤ u
'''

-run "main" in order to solve an example of a 3x3 convex quadratic optimisation problem


-settings can be changed as in example of "verbose=False"


-see also open source solver documentation: https://osqp.readthedocs.io/en/latest/
