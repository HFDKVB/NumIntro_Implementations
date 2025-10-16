import math
from typing import Callable, List, Dict, Tuple

def interval_bisection(
    f: Callable[[float], float],
    a: float,
    b: float,
    epsilon: float,
    return_history: bool = False
) -> Tuple[float, int, List[Dict[str, float]] | None]:
    """
    Interval Bisection method for root-finding.

    Parameters
    ----------
    f : callable
        Function for which we want to find a root.
    a, b : float
        Lower and upper bounds of the initial interval.
    epsilon : float
        Desired precision.
    return_history : bool, optional
        If True, returns detailed iteration history for analysis or plotting.

    Returns
    -------
    root : float
        The estimated root.
    iterations : int
        Number of iterations performed.
    history : list of dict, optional
        Iteration details (only if return_history=True).
    """
    if f(a) * f(b) > 0:
        raise ValueError("Function has same sign at both ends of interval.")

    n_iterations = math.ceil(math.log((b - a) / epsilon, 2)) - 1

    ak, bk = a, b
    history = [] if return_history else None

    for k in range(n_iterations + 1):
        mk = (ak + bk) / 2
        fmk = f(mk)

        if return_history:
            history.append({
                "iteration": k,
                "a_k": ak,
                "b_k": bk,
                "m_k": mk,
                "f(m_k)": fmk,
                "|f(m_k)|<epsilon": abs(fmk) < epsilon
            })

        if abs(fmk) < epsilon:
            break

        if math.copysign(1, f(ak)) == math.copysign(1, fmk):
            ak = mk
        else:
            bk = mk

    return mk, k, history



def functional_iteration(
    f: Callable[[float], float],
    g: Callable[[float], float],
    x0: float,
    epsilon: float = 1e-5,
    K: int = 100,
    return_history: bool = False
) -> Tuple[float, int, bool, List[Dict[str, float]] | None]:
    """
    Fixed-point (functional) iteration method for solving f(x) = 0 via x = g(x).

    Parameters
    ----------
    f : callable
        Function for which we want to find the root.
    g : callable
        Iteration function such that x = g(x) at the root.
    x0 : float
        Initial guess.
    epsilon : float, optional
        Convergence tolerance for |f(x_k)|.
    K : int, optional
        Maximum number of iterations.
    return_history : bool, optional
        If True, returns iteration details (for plotting or diagnostics).

    Returns
    -------
    root : float
        Estimated root.
    iterations : int
        Number of iterations performed.
    converged : bool
        True if convergence achieved within tolerance, False otherwise.
    history : list of dict, optional
        Iteration details (only if return_history=True).
    """
    xk = x0
    converged = False
    history = [] if return_history else None

    for k in range(1, K + 1):
        xk1 = g(xk)
        fval = f(xk1)

        if return_history:
            history.append({
                "iteration": k,
                "x_k": xk,
                "x_k+1": xk1,
                "f(x_k+1)": fval,
                "|f(x_k+1)|<epsilon": abs(fval) < epsilon
            })

        if abs(fval) < epsilon:
            converged = True
            xk = xk1
            break

        xk = xk1

    return xk, k, converged, history


from typing import Callable, List, Dict, Tuple

def newtons_method(
    f: Callable[[float], float],
    fprime: Callable[[float], float],
    x0: float,
    epsilon: float = 1e-5,
    K: int = 100,
    return_history: bool = False
) -> Tuple[float, int, bool, List[Dict[str, float]] | None]:
    """
    Newton-Raphson method for finding a root of f(x) = 0.

    Parameters
    ----------
    f : callable
        Function for which to find the root.
    fprime : callable
        Derivative of f.
    x0 : float
        Initial guess.
    epsilon : float, optional
        Convergence tolerance.
    K : int, optional
        Maximum number of iterations.
    return_history : bool, optional
        If True, return iteration history for diagnostics.

    Returns
    -------
    root : float
        Estimated root.
    iterations : int
        Number of iterations performed.
    converged : bool
        True if converged within tolerance, False otherwise.
    history : list of dict, optional
        Iteration data if return_history=True.
    """
    xk = x0
    converged = False
    history = [] if return_history else None

    for k in range(1, K + 1):
        fval = f(xk)
        fprime_val = fprime(xk)

        if fprime_val == 0:
            raise ZeroDivisionError(f"Derivative is zero at x = {xk}")

        xk1 = xk - fval / fprime_val
        fval_next = f(xk1)

        if return_history:
            history.append({
                "iteration": k,
                "x_k": xk,
                "x_k+1": xk1,
                "f(x_k)": fval,
                "|x_k+1 - x_k|": abs(xk1 - xk),
                "|f(x_k+1)|<epsilon": abs(fval_next) < epsilon
            })

        if abs(xk1 - xk) < epsilon or abs(fval_next) < epsilon:
            converged = True
            xk = xk1
            break

        xk = xk1

    return xk, k, converged, history



def horners_method(a: dict, x0: float) -> Tuple[float, dict]:
    """
    Evaluate a polynomial p(x) using Horner's method.

    Parameters
    ----------
    a : dict
        Polynomial coefficients {0:a0, 1:a1, ..., n:an}.
    x0 : float
        Point at which to evaluate.

    Returns
    -------
    p(x0) : float
        Value of the polynomial at x0.
    b : dict
        Coefficients of the reduced polynomial q(x).
    """
    n = len(a) - 1
    b = {n: a[n]}
    for k in range(n - 1, -1, -1):
        b[k] = a[k] + b[k + 1] * x0
    return b[0], b

def newton_horners_method(
    a: dict,
    x0: float,
    epsilon: float = 1e-5,
    K: int = 100,
    return_history: bool = False
) -> Tuple[float, int, bool, List[Dict[str, float]] | None]:
    """
    Newton's method specialized for polynomials using Horner's scheme.

    Parameters
    ----------
    a : dict
        Polynomial coefficients {0:a0, 1:a1, ..., n:an}.
    x0 : float
        Initial guess.
    epsilon : float, optional
        Convergence tolerance.
    K : int, optional
        Maximum number of iterations.
    return_history : bool, optional
        If True, return iteration details.

    Returns
    -------
    root : float
        Estimated root.
    iterations : int
        Number of iterations performed.
    converged : bool
        True if converged within tolerance.
    history : list of dict, optional
        Iteration data (if return_history=True).
    """
    xi = x0
    converged = False
    history = [] if return_history else None

    for i in range(1, K + 1):
        n = len(a) - 1
        b, c = {}, {}
        b[n] = a[n]
        c[n] = b[n]

        # Build Horner chains
        for k in range(n - 1, 0, -1):
            b[k] = a[k] + b[k + 1] * xi
            c[k] = b[k] + c[k + 1] * xi

        b0 = a[0] + b[1] * xi
        denom = c.get(1, None)
        if denom == 0:
            raise ZeroDivisionError("Derivative (q(x)) became zero during iteration.")

        x_next = xi - b0 / denom

        if return_history:
            history.append({
                "iteration": i,
                "x_i": xi,
                "x_i+1": x_next,
                "p(x_i)": b0,
                "|x_i+1 - x_i|": abs(x_next - xi),
                "|p(x_i)|<epsilon": abs(b0) < epsilon
            })

        if abs(x_next - xi) < epsilon or abs(b0) < epsilon:
            converged = True
            xi = x_next
            break

        xi = x_next

    return xi, i, converged, history
