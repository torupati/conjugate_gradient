import numpy as np


def solve(A, b, **args):
    """
    Solve linear equation A*x = b by conjugate gradient method.
    **args
      max_itr: int,  truncate iteration by this step. 0 is not trancate.
      savelog: bool, write log file.
    """
    dim = len(b)
    x = np.zeros(dim) # initial_value
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
        _flog.write(f'{k},{np.linalg.norm(x):10.4e},{np.linalg.norm(r):10.4e},{np.linalg.norm(p):10.4e},nan,nan\n')
    while True:
        print(f'k={k} res={r_ss} {len(p)}')
        if k >= min((lambda v : v if v > 0 else dim)(max_itr), len(p)):
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
            _flog.write(f'{k},{np.linalg.norm(x):10.4e},{np.linalg.norm(r):10.4e},{np.linalg.norm(p):10.4e}' \
                + f',{_a:10.4e},{_b:10.4e}\n')
    if _flog is not None:
        _flog.close()
    return x
