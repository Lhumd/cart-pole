from numpy.linalg import inv
import numpy as np
import scipy.linalg


def LQR(A, B, Q, R):
    if not(A.shape[0] == A.shape[1] == B.shape[0] == Q.shape[0] == Q.shape[1] and B.shape[1] == R.shape[0] == R.shape[1]):
        return

    '''
    It is slow. Don't use it.
    
    #P = Matrix(MatrixSymbol('P', len(A), len(A)))
    #T1 = A.transpose() * P
    #T2 = P * A
    #T3 = P * B * inv(R) * B.transpose() * P
    #P = np.linalg.solve(T1 + T2 - T3, -1 * Q)
    #P = solve((T1 + T2 - T3 + Q, P - P.transpose()), P) 
    '''
    P = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))
    K = np.matrix(inv(R) * B.transpose() * P)
    return K