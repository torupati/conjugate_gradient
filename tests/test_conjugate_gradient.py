import numpy as np
import sys

from src.conjugate_gradient import solve

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

if __name__ == "__main__":
    test_conjugate_gradient()
