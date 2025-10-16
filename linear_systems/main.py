import numpy as np
from typing import Tuple

def forward_substitution(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solves the lower-triangular system Lx = b via forward substitution.

    Parameters
    ----------
    L : ndarray
        Lower-triangular n×n matrix.
    b : ndarray
        Right-hand side vector of length n.

    Returns
    -------
    x : ndarray
        Solution vector.
    """
    n = L.shape[0]
    x = np.zeros(n, dtype=float)
    for i in range(n):
        numerator = b[i] - np.dot(L[i, :i], x[:i])
        x[i] = numerator / L[i, i]
    return x


def backward_substitution(U: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solves the upper-triangular system Ux = b via backward substitution.
    """
    n = U.shape[0]
    x = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        numerator = b[i] - np.dot(U[i, i+1:], x[i+1:])
        x[i] = numerator / U[i, i]
    return x

def swap_rows(A: np.ndarray, b: np.ndarray, row1: int, row2: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Swaps two rows (1-indexed) in A and b in-place.
    """
    A[[row1-1, row2-1], :] = A[[row2-1, row1-1], :]
    b[row1-1], b[row2-1] = b[row2-1], b[row1-1]
    return A, b

def gaussian_elimination_elementary(A: np.ndarray, b: np.ndarray, return_steps: bool = False):
    """
    Performs Gaussian elimination (without pivoting) to transform A to upper-triangular form.

    Parameters
    ----------
    A : ndarray
        n×n matrix.
    b : ndarray
        Right-hand side vector.
    return_steps : bool, optional
        If True, return intermediate states for inspection.

    Returns
    -------
    U : ndarray
        Upper-triangular matrix.
    y : ndarray
        Modified RHS vector.
    steps : list[dict], optional
        List of intermediate elimination states.
    """
    A = A.astype(float).copy()
    b = b.astype(float).copy()
    n = A.shape[0]
    steps = [] if return_steps else None

    for k in range(n - 1):
        if A[k, k] == 0:
            # Partial pivoting
            max_idx = np.argmax(np.abs(A[k:, k])) + k
            if A[max_idx, k] == 0:
                raise ValueError("Matrix is singular.")
            A, b = swap_rows(A, b, k + 1, max_idx + 1)

        for i in range(k + 1, n):
            m = A[i, k] / A[k, k]
            A[i, k:] -= m * A[k, k:]
            b[i] -= m * b[k]

        if return_steps:
            steps.append({"k": k + 1, "A": A.copy(), "b": b.copy()})

    return (A, b, steps) if return_steps else (A, b)

def LUfactorization(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Recursive LU decomposition without pivoting.

    Parameters
    ----------
    A : ndarray
        n×n matrix.

    Returns
    -------
    L, U : ndarrays
        Lower and upper triangular matrices.
    """
    n = A.shape[0]
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    if n == 1:
        L[0, 0] = 1.0
        U[0, 0] = A[0, 0]
        return L, U

    L[0, 0] = 1.0
    U[0, :] = A[0, :]
    L[1:, 0] = A[1:, 0] / U[0, 0]

    M = A[1:, 1:] - np.outer(L[1:, 0], U[0, 1:])
    L22, U22 = LUfactorization(M)
    L[1:, 1:] = L22
    U[1:, 1:] = U22

    return L, U

def cholesky_decomposition(A: np.ndarray) -> np.ndarray:
    """
    Performs Cholesky decomposition on a symmetric positive-definite matrix A.

    Returns L such that A = L Lᵀ.
    """
    A = np.array(A, dtype=float)
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix A must be square")

    n = A.shape[0]
    L = np.zeros((n, n))

    for k in range(n):
        L[k, k] = np.sqrt(A[k, k] - np.dot(L[k, :k], L[k, :k]))
        for i in range(k + 1, n):
            L[i, k] = (A[i, k] - np.dot(L[i, :k], L[k, :k])) / L[k, k]

    return L

def norm_infinity(x: np.ndarray) -> float:
    """Infinity norm (max absolute value) of a vector."""
    return np.max(np.abs(x))


def jacobi_iteration_method(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    epsilon: float = 1e-8,
    max_iterations: int = 500,
    return_history: bool = False
):
    """
    Solves Ax = b using the Jacobi iterative method.

    Parameters
    ----------
    A : ndarray
        n×n matrix.
    b : ndarray
        n-vector.
    x0 : ndarray
        Initial guess.
    epsilon : float
        Convergence tolerance.
    max_iterations : int
        Maximum allowed iterations.
    return_history : bool
        If True, return all iteration values.

    Returns
    -------
    x : ndarray
        Approximate solution.
    k : int
        Number of iterations performed.
    converged : bool
        True if convergence criterion met.
    history : list[dict], optional
        Iteration history if requested.
    """
    n = A.shape[0]
    x = x0.astype(float).copy()
    history = [] if return_history else None
    converged = False

    for k in range(1, max_iterations + 1):
        x_new = np.zeros_like(x)
        for i in range(n):
            s = np.dot(A[i, :], x) - A[i, i] * x[i]
            x_new[i] = (b[i] - s) / A[i, i]

        err = norm_infinity(x_new - x) / max(norm_infinity(x_new), 1e-12)

        if return_history:
            history.append({"iteration": k, "x": x_new.copy(), "error": err})

        if err < epsilon:
            converged = True
            x = x_new
            break

        x = x_new

    return (x, k, converged, history)
