"""Import high level packages."""
import numpy as np
import sys, os
import picos as pic
from mosek.fusion import *
import mosek
import cvxopt as cvx
from functools import wraps
import time
import timeout_decorator


def build_constants(n, s, L):
    Diag_s = np.diag(s)  # change s into a diagonal matrix
    Q = np.dot(np.dot(Diag_s, L), Diag_s)
    q = np.dot(np.dot(np.array(s), L), Diag_s)
    P = Q - np.diag(q)
    return P


def change_s(s, x):
    """Get the labels after flipping."""
    return [s[i] if x[i] == 0 else -s[i] for i in range(len(x))]


def diversity_index(L, s):
    """
    Compute eta (diversity index).

    retrun s.T*L*s
    """
    temp = 0
    for i in range(len(s)):
        for j in range(len(s)):
            temp += L[i][j]*s[i]*s[j]
    return temp


def correct_p(p):
    """Solve the problem when probability larger than 1 or smaller than 0."""
    return min(1, max(0, p))


@timeout_decorator.timeout(7200)
def sdp_relax(x, X, I, k, P, b):
    """
    Round technique in the paper. Multivariate rounding

    Loop, I times
    in this function, k is a value not a list
    """
    Sigma = X - np.outer(x, x)
    f = 0
    for i in range(I):
        flag = 1
        while flag == 1:
            z = np.random.multivariate_normal(x, Sigma)
            x_tilde = [np.random.choice([1, 0], 1,
                       p=[correct_p(t), 1 - correct_p(t)])[0] for t in z]
            if np.inner(x_tilde, b) <= k:
                flag = 0
        if f < np.dot(np.dot(x_tilde, P), x_tilde):
            f = np.dot(np.dot(x_tilde, P), x_tilde)
            x_right_tilde = x_tilde[:]  # maximum solution
    return x_right_tilde


@timeout_decorator.timeout(7200)
def simple_rounding(x, I, k, P, b):
    """
    An intuitive rounding technique, without drawing from Gaussian distribution.

    Loop, I times
    """
    np.random.seed(123)
    f = 0
    for i in range(I):
        flag = 1
        while flag == 1:
            x_tilde = [np.random.choice([1, 0], 1,
                       p=[t, 1 - t])[0] for t in x]
            if np.inner(x_tilde, b) <= k:
                flag = 0
        if f < np.dot(np.dot(x_tilde, P), x_tilde):
            f = np.dot(np.dot(x_tilde, P), x_tilde)
            x_right_tilde = x_tilde[:]  # maximum solution
    return x_right_tilde


#set a timer, should not run tooo long
@timeout_decorator.timeout(7200)
def mosek_scp(P, k, b_b_trans, n):
    """
    Use Mosek to solve the SCP-QBK Problem.

    return the optimal value
    Here k is a value not a list
    """

    with Model("M") as M:
        M.setSolverParam('optimizerMaxTime', 7200) # for test
        # set up the variables
        Z = M.variable("Z", Domain.inPSDCone(n+1))
        X = Z.slice([0,0], [n,n])
        x = Z.slice([0,n], [n,n+1])
        M.constraint(Expr.sub(X.diag(), x), Domain.equalsTo(0.))
        M.constraint(Z.index(n,n), Domain.equalsTo(1.))

        # set up the constants
        b_b_trans = Matrix.dense(b_b_trans)
        P = Matrix.dense(P)
        # constraints
        M.constraint("c2", Expr.dot(b_b_trans, X), Domain.lessThan(k*k))
        #objective
        M.objective(ObjectiveSense.Maximize, Expr.dot(P, X))
        #M.setLogHandler(sys.stdout)
        M.solve()

        return np.array(x.level()), np.reshape(X.level(), (n, n))

