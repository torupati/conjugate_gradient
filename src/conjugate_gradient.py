import numpy as np


def solve(A, b, **args):
    """
    Solve linear equation A*x = b by conjugate gradient method.
    **args
      max_itr: int,  truncate iteration by this step. 0 is not trancate.
      savelog: bool, write log file.
    """
    dim = len(b)
    x = np.zeros(dim)  # initial_value
    r = b - np.dot(A, x)
    p = r
    r_ss = np.dot(r.T, r)
    k = 0
    print(f'p({p.shape}) {p}')
    max_itr = args.get('max_itr', 0)
    _flog = None
    if args.get('savelog', False):
        _flog = open("cg.log", 'w')
        _flog.write('k,x,r,p,a,b\n')
        _flog.write(f'{k},{np.linalg.norm(x):9.3e},{np.linalg.norm(r):9.3e},'
                    + f'{np.linalg.norm(p):9.3e},nan,nan\n')
    while True:
        print(f'k={k} res={r_ss} {len(p)}')
        if k >= min((lambda v: v if v > 0 else dim)(max_itr), len(p)):
            break
        # Calculate A*p
        Ap = np.dot(A, p)
        # Add base vector p and add projection to solution
        _a = np.dot(r.T, p) / np.dot(p.T, Ap)
        x = x + _a * p
        r = r - _a * Ap
        r_ss_temp = np.dot(r.T, r)
        if np.abs(r_ss_temp) < 1.0E-10:
            print(f'converge eps={np.abs(r_ss_temp)}')
            break
        #
        _b = r_ss_temp / r_ss
        p = r + _b * p
        r_ss = r_ss_temp
        k = k + 1
        if _flog is not None:
            _flog.write(f'{k},{np.linalg.norm(x):9.3e},'
                        + f'{np.linalg.norm(r):9.3e},'
                        + f'{np.linalg.norm(p):10.4e},{_a:10.4e},{_b:10.4e}\n')
    if _flog is not None:
        _flog.close()
    return x


if __name__ == '__main__':
    A = np.array([[1.1, -2.1, 3.0],
                  [0.0, 0.2,  0.5],
                  [1.5, -0.1, 2.0]])
    A = A * A.T
    b = np.array([1.1, 0.1, -2.4])
    ans = solve(A, b)
    print(f'rank(A)={np.linalg.matrix_rank(A)}')
    w, v = np.linalg.eig(A)
    print(f'eigen: {w}')
    # print(f'ans={ans}')
    r = b - np.dot(A, ans)
    # print(f'b=Ax={b_a}')
    print(f'norm(b - A*x)={np.linalg.norm(r)}')
    for v in r:
        print(v)
        assert np.abs(v) < 1.0E-5
