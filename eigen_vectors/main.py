import numpy as np

def power_method(A: np.array, x0: np.array, phi, norm, max_iterations: int, epsilon: float):
    """
    Power method for dominant eigenvalue/eigenvector approximation.
    Returns:
      - r: final eigenvalue approximation
      - x: final eigenvector approximation
      - log: list of iteration dictionaries
    """
    x = x0 / norm(x0)
    err = float('inf')
    k = 1
    log = []

    while err > epsilon and k <= max_iterations:
        x1 = np.matmul(A, x)
        r = phi(x1) / phi(x)
        x1 = x1 / norm(x1)
        err = norm(x1 - x)

        log.append({
            "iteration": k,
            "x_prev": x.copy(),
            "x_new": x1.copy(),
            "eigenvalue_estimate": r,
            "error": err
        })

        x = x1
        k += 1

    return r, x, log


def forward_substitution(L, b):
    n = L.shape[0]
    x = np.zeros(n, dtype="complex")
    for i in range(n):
        numerator = b[i] - np.dot(L[i, :i], x[:i])
        x[i] = numerator / L[i, i]
    return x


def backward_substitution(U, b):
    n = U.shape[0]
    x = np.zeros(n, dtype="complex")
    for i in range(n - 1, -1, -1):
        numerator = b[i] - np.dot(U[i, i + 1:], x[i + 1:])
        x[i] = numerator / U[i, i]
    return x


def LUfactorization(A):
    n = len(A)
    L = np.zeros((n, n), dtype="complex")
    U = np.zeros((n, n), dtype="complex")

    L[0, 0] = 1.0
    U[0, 0] = A[0, 0]

    if n == 1:
        return L, U

    U[0, 1:n] = A[0, 1:n]
    L[1:n, 0] = A[1:n, 0] / U[0, 0]

    M = A[1:n, 1:n] - np.outer(L[1:n, 0], U[0, 1:n])
    L22, U22 = LUfactorization(M)
    L[1:n, 1:n] = L22
    U[1:n, 1:n] = U22

    return L, U


def inverse_power_method(A: np.array, x0: np.array, phi, norm, max_iterations: int, epsilon: float):
    """
    Inverse Power method for smallest eigenvalue/eigenvector approximation.
    Returns:
      - r: final eigenvalue approximation
      - x: final eigenvector approximation
      - log: list of iteration dictionaries
    """
    x = x0 / norm(x0)
    L, U = LUfactorization(A)
    err = float('inf')
    r_prev = float('inf')
    k = 1
    log = []

    while err > epsilon and k <= max_iterations:
        y = forward_substitution(L, x)
        x1 = backward_substitution(U, y)

        r = phi(x1) / phi(x)
        x1 = x1 / norm(x1)
        err = norm(r - r_prev)

        log.append({
            "iteration": k,
            "eigenvalue_estimate": r,
            "error": err,
            "x_prev": x.copy(),
            "x_new": x1.copy()
        })

        x = x1
        r_prev = r
        k += 1

    return r, x, log
