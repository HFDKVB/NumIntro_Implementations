from typing import Callable
import numpy as np


def trapezoid_rule(f: Callable[[float], float], a: float, b: float) -> float:
    """
    Compute the single-segment trapezoid approximation of the integral of f(x) over [a, b].

    Parameters
    ----------
    f : callable
        Function to integrate.
    a : float
        Lower limit of integration.
    b : float
        Upper limit of integration.

    Returns
    -------
    float
        Approximation of the integral ∫_a^b f(x) dx.
    """
    return (b - a) * 0.5 * (f(a) + f(b))


def composite_trapezoid_rule(
    f: Callable[[float], float], 
    a: float, 
    b: float, 
    n: int, 
    return_info: bool = False
):
    """
    Compute the composite trapezoid approximation of ∫_a^b f(x) dx
    using n uniform subintervals.

    Parameters
    ----------
    f : callable
        Function to integrate.
    a : float
        Lower limit of integration.
    b : float
        Upper limit of integration.
    n : int
        Number of subintervals.
    return_info : bool, optional
        If True, returns a dict containing sample points and f-values for inspection.

    Returns
    -------
    float or tuple
        - The integral approximation if return_info=False.
        - A tuple (integral, info_dict) if return_info=True.
    """
    h = (b - a) / n
    x_vals = np.linspace(a, b, n + 1)
    f_vals = np.array([f(x) for x in x_vals])

    integral = (h / 2) * (f_vals[0] + 2 * np.sum(f_vals[1:-1]) + f_vals[-1])

    if not return_info:
        return integral

    info = {
        "x": x_vals,
        "f(x)": f_vals,
        "h": h,
        "n": n,
        "approximation": integral
    }
    return integral, info


def simpsons_rule(f: Callable[[float], float], a: float, b: float) -> float:
    """
    Compute the single-segment trapezoid approximation of the integral of f(x) over [a, b].

    Parameters
    ----------
    f : callable
        Function to integrate.
    a : float
        Lower limit of integration.
    b : float
        Upper limit of integration.

    Returns
    -------
    float
        Approximation of the integral ∫_a^b f(x) dx.
    """
    return (b - a) * (1.0/6) * (f(a) + 4*f((a+b)*0.5) + f(b))



def composite_simpsons_rule(
    f: Callable[[float], float], 
    a: float, 
    b: float, 
    n: int, 
    return_info: bool = False
):
    """
    Compute the composite Simpson's rule approximation of ∫_a^b f(x) dx
    using n uniform subintervals (n will be made even if odd).

    Parameters
    ----------
    f : callable
        Function to integrate.
    a : float
        Lower limit of integration.
    b : float
        Upper limit of integration.
    n : int
        Number of subintervals (will be made even if odd).
    return_info : bool, optional
        If True, returns a dict containing sample points and f-values for inspection.

    Returns
    -------
    float or tuple
        - The integral approximation if return_info=False.
        - A tuple (integral, info_dict) if return_info=True.
    """
    if n % 2 == 1:
        n += 1  # ensure even number of subintervals

    h = (b - a) / n
    x_vals = np.linspace(a, b, n + 1)
    f_vals = np.array([f(x) for x in x_vals])

    # Simpson’s rule weights: 1, 4, 2, 4, ..., 4, 1
    integral = (h / 3) * (
        f_vals[0]
        + 4 * np.sum(f_vals[1:-1:2])
        + 2 * np.sum(f_vals[2:-1:2])
        + f_vals[-1]
    )

    if not return_info:
        return integral

    info = {
        "x": x_vals,
        "f(x)": f_vals,
        "h": h,
        "n": n,
        "approximation": integral,
    }
    return integral, info

from typing import Callable

def trapezoid_asymptotic_error(
    f_prime: Callable[[float], float], 
    a: float, 
    b: float, 
    n: int
) -> float:
    """
    Estimates the asymptotic (leading term) error of the composite trapezoidal rule
    for integrating f(x) over [a, b], based on f'(x).

    The asymptotic error is approximated by:
        E_T ≈ -(h² / 12) * [f'(b) - f'(a)]

    Parameters
    ----------
    f_prime : callable
        Derivative of the integrand f'(x).
    a : float
        Lower limit of integration.
    b : float
        Upper limit of integration.
    n : int
        Number of subintervals in the composite trapezoidal rule.

    Returns
    -------
    float
        Estimated asymptotic error term.
    """
    if n <= 0:
        raise ValueError("n must be a positive integer.")
    if a == b:
        return 0.0

    h = (b - a) / n
    error_estimate = -(h**2 / 12.0) * (f_prime(b) - f_prime(a))
    return error_estimate

from typing import Callable
import numpy as np

def romberg(
    f: Callable[[float], float],
    a: float,
    b: float,
    J: int,
    return_table: bool = False
):
    """
    Computes the Romberg integration of f(x) over [a, b] up to level J.

    Parameters
    ----------
    f : callable
        Function to integrate.
    a : float
        Lower limit.
    b : float
        Upper limit.
    J : int
        Number of Romberg levels.
    return_table : bool, optional
        If True, returns the full Romberg table for inspection.

    Returns
    -------
    float or tuple
        - Romberg approximation at level J if return_table=False.
        - (Romberg approximation, table) if return_table=True.
    """
    # Initialize Romberg table
    R = np.zeros((J, J), dtype=float)
    
    # Level 1: trapezoid with 1 subinterval
    for k in range(J):
        n_intervals = 2**k
        R[k, 0] = composite_trapezoid_rule(f, a, b, n_intervals)
    
    # Fill the Romberg table using Richardson extrapolation
    for j in range(1, J):
        for k in range(J - j):
            R[k, j] = R[k+1, j-1] + (R[k+1, j-1] - R[k, j-1]) / (4**j - 1)
        
    if return_table:
        return R[0, J-1], R
    else:
        return R[0, J-1]

    


    