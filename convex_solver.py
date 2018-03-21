"""Solve an Linear Quadratic problem using OSQP on tensorflow:
Minimize    : 0.5 (x.T * P * x) + (q.T * x)
Subject to  :  l ≤ Ax ≤ u

With acknowledgement from:
-Professor Stephen Boyd
-Professor Paul Goulart
-Dr. Matt Wytock
-Dr. Bartolomeo Stellato
-Nikitas Rontsis
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import namedtuple
from builtins import object
import numpy as np
import scipy.sparse as spa
import tensorflow as tf
import time
import scipy.linalg as la
from collections import namedtuple
import tensorflow as tf
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import timeline

class IterativeSolver(object):
    """Abstract base class for iterative solvers."""
    def __init__(self, name=None):
        self.name = name or type(self).__name__

    def init(self):
        raise NotImplementedError

    def iterate(self, state):
        raise NotImplementedError

    def stop(self, state):
        raise NotImplementedError

    def _iterate(self, state):
        with tf.name_scope(self.name):
            with tf.name_scope("iterate"):
                return self.iterate(state)

    def _stop(self, state):
        with tf.name_scope(self.name):
            with tf.name_scope("stop"):
                return self.stop(state)

    def _init(self):
        with tf.name_scope(self.name):
            with tf.name_scope("init"):
                return self.init()

    def solve(self):
        return tf.while_loop(
            lambda *args: tf.logical_not(self._stop(self.state(*args))),
            lambda *args: self._iterate(self.state(*args)),
            self._init())




class ConjugateGradient(IterativeSolver):
    def __init__(self, prob, sett, b, x_init, tol=1e-6, max_iterations=100, **kwargs):
        self.P = prob.P
        self.A = prob.A
        self.AT = prob.AT
        self.sigma = sett.sigma
        self.rho = sett.rho
        self.b = b
        self.x_init = x_init
        self.tol = tol
        self.max_iterations = max_iterations
        super(ConjugateGradient, self).__init__(**kwargs)
    @property

    def state(self):
        return namedtuple("State", ["x", "r", "p", "r_norm_sq", "k"])

    def M(self, x):
        Mx = self.rho * x
        Mx = tf.sparse_tensor_dense_matmul(self.A, Mx)
        Mx = tf.sparse_tensor_dense_matmul(self.AT, Mx)
        Mx = Mx + self.sigma * x
        Mx = Mx + tf.sparse_tensor_dense_matmul(self.P, x)
        return Mx

    def init(self):
        x = self.x_init
        r = self.b - self.M(x)
        p = r
        r_norm_sq = tf.reduce_sum(r*r)
        k = tf.constant(0)
        self.r_norm_sq0 = r_norm_sq
        return self.state(x, r, p, r_norm_sq, k)

    def iterate(self, state):
        Mp = self.M(state.p)
        alpha = state.r_norm_sq / tf.reduce_sum(state.p*Mp)
        x = state.x + alpha*state.p
        r = state.r - alpha*Mp
        r_norm_sq = tf.reduce_sum(r*r)
        beta = r_norm_sq / state.r_norm_sq
        p = r + beta*state.p
        return self.state(x, r, p, r_norm_sq, state.k+1)

    def stop(self, state):
        return tf.logical_or(
            state.k >= self.max_iterations,
            state.r_norm_sq <= self.tol*self.tol*self.r_norm_sq0)




OSQP_INFTY = 1.e+20          # OSQP Infinity
OSQP_NAN = 1.e+20            # OSQP Nan

class settings(object):
    """
    Solver settings
    """
    def __init__(self, **kwargs):
        self.constr = kwargs.pop('constr', 'box')  # 'box' or 'PSDcone'
        self.rho = kwargs.pop('rho', 0.1) # changed this from 1.6 to 0.1
        self.sigma = kwargs.pop('sigma', 1e-6)
        self.max_iter = kwargs.pop('max_iter', 5000)
        self.alpha = kwargs.pop('alpha', 1.6) # changed this from 1.0 to 1.6
        self.verbose = kwargs.pop('verbose', True)
        self.eps_abs = kwargs.pop('eps_abs', 1e-6)
        self.eps_rel = kwargs.pop('eps_rel', 1e-3)
        self.eps_pinf = kwargs.pop('eps_pinf', 1e-4) # changed this from 1e-3 to 1e-4
        self.eps_dinf = kwargs.pop('eps_dinf', 1e-4) # changed this from 1e-3 to 1e-4
        self.optimal = kwargs.pop('optimal', False)
        self.print_interval = kwargs.pop('print_interval', 100)
        self.check_interval = kwargs.pop('check_interval', 20)
        self.cg_tol = 1e-6
        self.cg_max_iter = 100

class problem(object):
    def __init__(self, P, q, A, l, u):
        # Function to convert matrices to tensors
        def convert_sparse_matrix_to_sparse_tensor(X):
            coo = X.tocoo()
            indices = np.mat([coo.row, coo.col]).transpose()
            return tf.cast(tf.SparseTensor(indices, coo.data, coo.shape), tf.float32)

        # Define infinity in tensor form
        self.inf = tf.reduce_max([-np.inf, np.inf])

        # Set problem data
        q = np.reshape(q, (q.shape[0], 1))
        try:
            l = np.reshape(l, (l.shape[0], 1))
        except Exception:
            pass
        try:
            u = np.reshape(u, (u.shape[0], 1))
        except Exception:
            pass

        # Define case for unbounded constraints
        if l is not None:
            self.l = tf.convert_to_tensor(l, dtype=tf.float32)
        else:
            -self.inf * tf.ones([m,1])

        if u is not None:
            self.u = tf.convert_to_tensor(u, dtype=tf.float32)
        else:
            self.inf * tf.ones([m,1])

        self.m_val = A.shape[0]
        self.n_val = A.shape[1]
        self.AT = convert_sparse_matrix_to_sparse_tensor(A.T)
        self.P = convert_sparse_matrix_to_sparse_tensor(P)
        self.A = convert_sparse_matrix_to_sparse_tensor(A)
        self.q = tf.convert_to_tensor(q, dtype=tf.float32)
        self.np_P = P
        self.np_A = A
        self.np_q = q

class state_while(object):
    def __init__(self, prob, sett):
        m = prob.m_val
        n = prob.n_val
        self.x = tf.Variable(tf.zeros((n,1), dtype=tf.float32))
        self.z = tf.Variable(tf.zeros((m,1), dtype=tf.float32))
        self.xz_tilde = tf.Variable(tf.zeros((n+m,1), dtype=tf.float32))
        self.x_prev = tf.Variable(tf.zeros((n,1), dtype=tf.float32))
        self.z_prev = tf.Variable(tf.zeros((m,1), dtype=tf.float32))
        self.y = tf.Variable(tf.zeros((m,1), dtype=tf.float32))
        self.dx = tf.Variable(tf.zeros((n,1), dtype=tf.float32))
        self.dy = tf.Variable(tf.zeros((m,1), dtype=tf.float32))
        self.obj_val = tf.Variable(0, dtype=tf.float32)
        self.pri_res = tf.Variable(0, dtype=tf.float32)
        self.dua_res = tf.Variable(0, dtype=tf.float32)
        self.cg_iter = tf.Variable([], dtype=tf.int32, validate_shape=False)
        self.cg_time = tf.Variable([], dtype=tf.float32, validate_shape=False)

def ATY_while(prob, y):
    return tf.sparse_tensor_dense_matmul(prob.AT, y)

def solve_linear_while(prob, sett, RHS, x_init):
    """Solve the linear system using conjugate gradient.
    x = (P + sigma*I + A'*A*rho)' * (r1 + rho*A'*r2)
    y = (A*x - r2) * rho

    where: KKT * [x;y] = rhs
    and rhs = [r1; r2]
    """
    # r1 is dense row1 of rhs vector from update_xz_tilde,
    # i.e. RHS = xz_tilde, the 3x1 vector [x_tilde; v_tilde]
    ## Solve linear system
    r1 = RHS[:prob.n_val]
    r2 = RHS[prob.n_val:]
    # Set up rhs
    rhs = r1 + sett.rho * tf.sparse_tensor_dense_matmul(prob.AT, r2)
    # Solve to obtain x first using conjugate gradient method solving Mx = b
    cg = ConjugateGradient(prob, sett, rhs, x_init, sett.cg_tol, sett.cg_max_iter)
    state = cg.solve()
    x = state.x
    cg_iter = state.k
    # Now find y = (A*x - r2) * rho
    y = tf.sparse_tensor_dense_matmul(prob.A, x) - r2
    y = y * sett.rho
    # Finally form the output [x; y] as the new xz_tilde
    RHS = tf.concat([x, y], 0)
    return RHS, cg_iter

def iterate_while(prob, sett, state):
    x = state.x
    z = state.z
    y = state.y
    dx = state.dx
    dy = state.dy
    x_prev = state.x_prev
    z_prev = state.z_prev
    xz_tilde = state.xz_tilde
    cg_iter = state.cg_iter
    cg_time = state.cg_time

    # # Update x_prev, z_prev
    # upd_x_prev = tf.assign(x_prev, upd_x)
    # upd_z_prev = tf.assign(z_prev, upd_z)
    """
    First ADMM step: update xz_tilde
    """
    # Row 1 of the rhs term
    tilderow1 = x * sett.sigma - prob.q
    # Row 2 of the rhs term
    tilderow2 = z_prev - (y / sett.rho)
    # Vertically stack tilderow1 and tilderow2
    tilde = tf.concat([tilderow1, tilderow2], axis=0)

    cg_init = time.time()

    # Solve linear system
    cg_xz_tilde, cg_iter_elem = solve_linear_while(prob, sett, tilde, x)

    # Append the new cg_time element to the cg_time array
    cg_time_elem = time.time() - cg_init
    cg_time_tensor = tf.convert_to_tensor(cg_time_elem, dtype=tf.float32)
    upd_cg_time = tf.concat([cg_time, [cg_time_tensor]], 0)

    # Append the new cg_iter element to the cg_iter array
    upd_cg_iter = tf.concat([cg_iter, [cg_iter_elem]], 0)
    # Update z_tilde
    v_new = cg_xz_tilde[prob.n_val:]
    z_tilde_new = z_prev + (1 / sett.rho) * (v_new - y)
    # And then put v_term back into xz_tilde
    upd_xz_tilde = tf.concat([cg_xz_tilde[:prob.n_val], z_tilde_new], 0)
    """
    ADMM step 2.1: Update variable x
    """
    # Extract top term i.e. new x
    upd_x = (sett.alpha * upd_xz_tilde[:prob.n_val]) + (1 - sett.alpha) * x_prev
    upd_dx = upd_x - x_prev
    """
    ADMM step 2.2: Update variable z
    """
    first = sett.alpha * upd_xz_tilde[prob.n_val:]
    second = (1 - sett.alpha) * z_prev
    third = (1 / sett.rho) * y
    z_pre_proj = first + second + third
    """
    Project z variable in set C (for now C = [l, u])
    """
    max_term = tf.maximum(z_pre_proj, prob.l)
    upd_z = tf.minimum(max_term, prob.u)
    """
    ADMM step 3: update dual variable y
    """
    term1 = sett.alpha * upd_xz_tilde[prob.n_val:]
    term2 = (1 - sett.alpha) * z_prev
    upd_dy = sett.rho * (term1 + term2 - z)
    upd_y = y + upd_dy
    """
    Compute quadratic objective value for the given x
    """
    xTPx = tf.matmul(tf.transpose(upd_x), tf.sparse_tensor_dense_matmul(prob.P, upd_x))
    upd_obj_val = 0.5 * xTPx + tf.tensordot(tf.transpose(prob.q), upd_x, axes=1)
    """
    Compute primal residual ||Ax - z||
    """
    normterm = tf.norm(tf.sparse_tensor_dense_matmul(prob.A, upd_x) - upd_z)
    if prob.m_val == 0: # No constraints
        upd_pri_res = 0
    else:
        upd_pri_res = normterm
    """
    Compute dual residual ||Px + q + A'y||
    """
    PX = tf.sparse_tensor_dense_matmul(prob.P, upd_x)
    upd_dua_res =  tf.norm(PX + prob.q + ATY_while(prob, upd_y))
    """
    Finally update all values
    """
    return  tf.group(tf.assign(state.x, upd_x), \
            tf.assign(state.z, upd_z), \
            tf.assign(state.y, upd_y), \
            tf.assign(state.dx, upd_dx), \
            tf.assign(state.dy, upd_dy), \
            tf.assign(state.xz_tilde, upd_xz_tilde), \
            tf.assign(state.x_prev, upd_x), \
            tf.assign(state.z_prev, upd_z), \
            tf.assign(state.obj_val, upd_obj_val, validate_shape=False), \
            tf.assign(state.pri_res, upd_pri_res, validate_shape=False), \
            tf.assign(state.dua_res, upd_dua_res, validate_shape=False), \
            tf.assign(state.cg_iter, upd_cg_iter, validate_shape=False), \
            tf.assign(state.cg_time, upd_cg_time, validate_shape=False))

class OSQP(object):
    def __init__(self, prob, sett):
        m = prob.m_val
        n = prob.n_val
        self.x = tf.Variable(tf.zeros((n,1), dtype=tf.float32))
        self.z = tf.Variable(tf.zeros((m,1), dtype=tf.float32))
        self.xz_tilde = tf.Variable(tf.zeros((n+m,1), dtype=tf.float32))
        self.x_prev = tf.Variable(tf.zeros((n,1), dtype=tf.float32))
        self.z_prev = tf.Variable(tf.zeros((m,1), dtype=tf.float32))
        self.y = tf.Variable(tf.zeros((m,1), dtype=tf.float32))
        self.dx = tf.Variable(tf.zeros((n,1), dtype=tf.float32))
        self.dy = tf.Variable(tf.zeros((m,1), dtype=tf.float32))
        self.obj_val = tf.Variable(0, dtype=tf.float32)
        self.pri_res = tf.Variable(0, dtype=tf.float32)
        self.dua_res = tf.Variable(0, dtype=tf.float32)
        self.cg_iter = tf.Variable([], dtype=tf.int32, validate_shape=False)
        self.cg_time = tf.Variable([], dtype=tf.float32, validate_shape=False)

    def ATY(self, prob, y):
        return tf.sparse_tensor_dense_matmul(prob.AT, y)

    def solve_linear(self, prob, sett, RHS, x_init):
        """Solve the linear system using conjugate gradient.
        x = (P + sigma*I + A'*A*rho)' * (r1 + rho*A'*r2)
        y = (A*x - r2) * rho

        where: KKT * [x;y] = rhs
        and rhs = [r1; r2]
        """
        # r1 is dense row1 of rhs vector from update_xz_tilde,
        # i.e. RHS = xz_tilde, the 3x1 vector [x_tilde; v_tilde]
        ## Solve linear system
        r1 = RHS[:prob.n_val]
        r2 = RHS[prob.n_val:]
        # Set up rhs
        rhs = r1 + sett.rho * tf.sparse_tensor_dense_matmul(prob.AT, r2)
        # Solve to obtain x first using conjugate gradient method solving Mx = b
        cg = ConjugateGradient(prob, sett, rhs, x_init, sett.cg_tol, sett.cg_max_iter)
        state = cg.solve()
        x = state.x
        cg_iter = state.k
        # Now find y = (A*x - r2) * rho
        y = tf.sparse_tensor_dense_matmul(prob.A, x) - r2
        y = y * sett.rho
        # Finally form the output [x; y] as the new xz_tilde
        RHS = tf.concat([x, y], 0)
        return RHS, cg_iter

    def iterate(self, prob, sett):
        x = self.x
        z = self.z
        y = self.y
        dx = self.dx
        dy = self.dy
        x_prev = self.x_prev
        z_prev = self.z_prev
        xz_tilde = self.xz_tilde
        cg_iter = self.cg_iter
        cg_time = self.cg_time

        # # Update x_prev, z_prev
        # upd_x_prev = tf.assign(x_prev, upd_x)
        # upd_z_prev = tf.assign(z_prev, upd_z)
        """
        First ADMM step: update xz_tilde
        """
        # Row 1 of the rhs term
        tilderow1 = x * sett.sigma - prob.q
        # Row 2 of the rhs term
        tilderow2 = z_prev - (y / sett.rho)
        # Vertically stack tilderow1 and tilderow2
        tilde = tf.concat([tilderow1, tilderow2], axis=0)

        cg_init = time.time()

        # Solve linear system
        cg_xz_tilde, cg_iter_elem = self.solve_linear(prob, sett, tilde, x)

        # Append the new cg_time element to the cg_time array
        cg_time_elem = time.time() - cg_init
        cg_time_tensor = tf.convert_to_tensor(cg_time_elem, dtype=tf.float32)
        upd_cg_time = tf.concat([cg_time, [cg_time_tensor]], 0)

        # Append the new cg_iter element to the cg_iter array
        upd_cg_iter = tf.concat([cg_iter, [cg_iter_elem]], 0)
        # Update z_tilde
        v_new = cg_xz_tilde[prob.n_val:]
        z_tilde_new = z_prev + (1 / sett.rho) * (v_new - y)
        # And then put v_term back into xz_tilde
        upd_xz_tilde = tf.concat([cg_xz_tilde[:prob.n_val], z_tilde_new], 0)
        """
        ADMM step 2.1: Update variable x
        """
        # Extract top term i.e. new x
        upd_x = (sett.alpha * upd_xz_tilde[:prob.n_val]) + (1 - sett.alpha) * x_prev
        upd_dx = upd_x - x_prev
        """
        ADMM step 2.2: Update variable z
        """
        first = sett.alpha * upd_xz_tilde[prob.n_val:]
        second = (1 - sett.alpha) * z_prev
        third = (1 / sett.rho) * y
        z_pre_proj = first + second + third
        """
        Project z variable in set C (for now C = [l, u])
        """
        max_term = tf.maximum(z_pre_proj, prob.l)
        upd_z = tf.minimum(max_term, prob.u)
        """
        ADMM step 3: update dual variable y
        """
        term1 = sett.alpha * upd_xz_tilde[prob.n_val:]
        term2 = (1 - sett.alpha) * z_prev
        upd_dy = sett.rho * (term1 + term2 - z)
        upd_y = y + upd_dy
        """
        Compute quadratic objective value for the given x
        """
        xTPx = tf.matmul(tf.transpose(upd_x), tf.sparse_tensor_dense_matmul(prob.P, upd_x))
        upd_obj_val = 0.5 * xTPx + tf.tensordot(tf.transpose(prob.q), upd_x, axes=1)
        """
        Compute primal residual ||Ax - z||
        """
        normterm = tf.norm(tf.sparse_tensor_dense_matmul(prob.A, upd_x) - upd_z)
        if prob.m_val == 0: # No constraints
            upd_pri_res = 0
        else:
            upd_pri_res = normterm
        """
        Compute dual residual ||Px + q + A'y||
        """
        PX = tf.sparse_tensor_dense_matmul(prob.P, upd_x)
        upd_dua_res =  tf.norm(PX + prob.q + self.ATY(prob, upd_y))
        """
        Finally update all values
        """
        return  tf.assign(self.x, upd_x), \
                tf.assign(self.z, upd_z), \
                tf.assign(self.y, upd_y), \
                tf.assign(self.dx, upd_dx), \
                tf.assign(self.dy, upd_dy), \
                tf.assign(self.xz_tilde, upd_xz_tilde), \
                tf.assign(self.x_prev, upd_x), \
                tf.assign(self.z_prev, upd_z), \
                tf.assign(self.obj_val, upd_obj_val, validate_shape=False), \
                tf.assign(self.pri_res, upd_pri_res, validate_shape=False), \
                tf.assign(self.dua_res, upd_dua_res, validate_shape=False), \
                tf.assign(self.cg_iter, upd_cg_iter, validate_shape=False), \
                tf.assign(self.cg_time, upd_cg_time, validate_shape=False)

def solve(P, q, A, l, u, **stgs):
    """Create tensorflow graph and solve."""
    tf.reset_default_graph()
    prob = problem(P, q, A, l, u)
    sett = settings(**stgs)
    solver = OSQP(prob, sett)

    # ops
    t0 = time.time()
    init_op = tf.global_variables_initializer()

    iterate_op = solver.iterate(prob, sett)

    graph_time = time.time() - t0
    t1 = time.time()

    with tf.Session() as sess:
        sess.run(init_op)
        def optimal(fout, prob, sett, iter):
            x, z, y, obj_val, pri_res, dua_res =  [fout[0],
                                                   fout[1],
                                                   fout[2],
                                                   fout[8],
                                                   fout[9],
                                                   fout[10]]
            # calculate eps_primal
            Ax = prob.np_A.dot(x)
            eps_primal = sett.eps_abs + \
                         sett.eps_rel * \
                         np.maximum(la.norm(Ax), la.norm(z))
            # calculate eps_dual
            PX = la.norm(prob.np_P.dot(x))
            ATY = la.norm(prob.np_A.T.dot(y))
            eps_dual = sett.eps_abs + \
                       sett.eps_rel * \
                       np.maximum(np.maximum(PX, ATY), la.norm(prob.np_q))

            # Print summary
            if sett.verbose & \
              ((iter % sett.print_interval == 0) \
               | (iter == 1) |(iter == sett.max_iter)):
               print(iter,(obj_val),(pri_res),(dua_res))

            return pri_res <= eps_primal and dua_res <= eps_dual
        for iter in range(1, sett.max_iter+1):
            fout = sess.run(iterate_op, feed_dict={})
            if iter % sett.check_interval == 0 and optimal(fout, prob, sett, iter):
                break
            else:
               optimal(fout, prob, sett, sett.max_iter+1)

        solve_time = time.time() - t1
        run_time = time.time() - t0

        dic = {'x': sess.run(solver.x),\
              'z': sess.run(solver.z),\
              'y': sess.run(solver.y),\
              'dx': sess.run(solver.dx),\
              'dy': sess.run(solver.dy),\
              'xz_tilde': sess.run(solver.xz_tilde),\
              'x_prev': sess.run(solver.x_prev),\
              'z_prev': sess.run(solver.z_prev),\
              'obj_val': sess.run(solver.obj_val),\
              'pri_res': sess.run(solver.pri_res),\
              'dua_res': sess.run(solver.dua_res),\
              'cg_iter': sess.run(solver.cg_iter),\
              'cg_time': sess.run(solver.cg_time),\
              'iter': iter,\
              'graph_time': graph_time,\
              'solve_time': solve_time,\
              'run_time': run_time}
    return dic

def main():
    #'optimal' example with different dimensions from exercise 4.3, pg189 of "Convex Optimisation" x* = (1, 0.5,-1)
    q = np.array([-22, -14.5, 13])
    A = spa.csc_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    l = np.array([-1])
    u = np.array([1])
    P = spa.csc_matrix([[13, 12, -2], [12, 17, 6], [-2, 6, 12]])

    res = solve(P, q, A, l, u,
                verbose=False,\
                )

    print("THIS IS tf_osqp.py from osqp_tensorflow tf version")
    print("Number of iterations : %3d" % res['iter'])
    print("graph_time :%12.4e" % res['graph_time'])
    print("solve_time :%12.4e" % res['solve_time'])
    print("run_time :%12.4e" % res['run_time'])
    print("     Objective value : %12.4e" % res['obj_val'])
    print("        || Ax - z || = %12.4e" % res['pri_res'])
    print("  || Px + q + A'y || = %12.4e" % res['dua_res'])
    print("                   x = ", end='')
    print(np.array(res['x'].T)[0])
    print("")
    print("                   y = ", end='')
    print(np.array(res['y'].T)[0])
    print("")

if __name__ == "__main__":
    main()
