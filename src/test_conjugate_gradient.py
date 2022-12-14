import numpy as np
import sys

def conjugate_gradient(A, b, max_itr: int = 0):
    """
    Solve linear equation A*x = b by conjugate gradient method.
    """
    dim = len(b)
    x = np.zeros(dim) # initial_value
    r = b - np.dot(A, x)
    p = r
    r_ss = np.dot(r.T, r)
    k = 0
    print(f'p({p.shape}) {p}')
    _flog = open("cg.log", 'w')
    _flog.write('k,x,r,p,a,b\n')
    _flog.write(f'{k},{np.linalg.norm(x):10.4e},{np.linalg.norm(r):10.4e},{np.linalg.norm(p):10.4e},nan,nan\n')
    while True:
        print(f'k={k} res={r_ss} {len(p)}')
        if k >= min((lambda v : v if v > 0 else sys.maxsize)(max_itr), len(p)):
            break
        # Calculate A*p
        Ap = np.dot(A, p)
        # Add base vector p and add projection to solution
        _a = np.dot(r.T, p) / np.dot(p.T, Ap)
        x = x + _a * p
        r = r - _a * Ap
        r_ss_temp = np.dot(r.T, r)
        if r_ss_temp < 1.0E-10:
            break
        #
        _b = r_ss_temp / r_ss
        p = r + _b * p
        r_ss = r_ss_temp
        k = k + 1
        _flog.write(f'{k},{np.linalg.norm(x):10.4e},{np.linalg.norm(r):10.4e},{np.linalg.norm(p):10.4e}' \
        + f',{_a:10.4e},{_b:10.4e}\n')
    _flog.close()
    return x


def test_conjugate_gradient():
    #A = np.array([[1,0,3,0],
    #              [0,0,0,1],
    #              [3,0,0,2],
    #              [0,1,2,1]])
    #b = 100*np.array([6, 1, 4, 3])
    np.random.seed(0)

    D = 1024
    M = 1.5 * np.random.randn(D, D)
    A = M + M.T + 100*np.identity(D)
    D = 128
    M = 1.5 * np.random.randn(D, D)
    A = M + M.T + 100*np.identity(D)
    b = np.random.randn(D)
    print(f'A{A.shape} b{b.shape}')
    ans = conjugate_gradient(A, b)
    #print(f'ans={ans}')
    r = b - np.dot(A,ans)
    #print(f'b=Ax={b_a}')
    print(f'norm(b - A*x)={np.linalg.norm(r)}')
    for v in r:
        #print(v)
        assert np.abs(v) < 1.0E-5

if __name__ == "__main__":
    test_conjugate_gradient()