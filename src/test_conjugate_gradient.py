import numpy as np
import sys

from conjugate_gradient import solve

def test_conjugate_gradient():
    np.random.seed(0)

    D = 8192 # 1024
    M = 1.5 * np.random.randn(D, D)
    A = np.dot(M, M.T) + 10*np.identity(D)
    #D = 128
    #M = 1.5 * np.random.randn(D, D)
    #A = M + M.T + 100*np.identity(D)
    b = np.random.randn(D)
    print(f'A{A.shape} b{b.shape}')
    ans = solve(A, b, savelog = True)
    #print(f'ans={ans}')
    r = b - np.dot(A,ans)
    #print(f'b=Ax={b_a}')
    print(f'norm(b - A*x)={np.linalg.norm(r)}')
    for v in r:
        #print(v)
        assert np.abs(v) < 1.0E-5

def test_conjugate_gradient0():
    A = np.array([[1.1, -2.1, 3.0],
                  [0.0, 0.2,  0.5],
                  [1.5, -0.1, 2.0]])
    A = A * A.T
    b = np.array([1.1, 0.1,-2.4])
    ans = solve(A, b)
    print(f'rank(A)={np.linalg.matrix_rank(A)}')
    w, v = np.linalg.eig(A)
    print(f'eigen: {w}')
    #print(f'ans={ans}')
    r = b - np.dot(A,ans)
    #print(f'b=Ax={b_a}')
    print(f'norm(b - A*x)={np.linalg.norm(r)}')
    for v in r:
        print(v)
        assert np.abs(v) < 1.0E-5

if __name__ == "__main__":
    test_conjugate_gradient()