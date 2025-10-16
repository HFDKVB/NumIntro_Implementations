import numpy as np

def evaluate_by_nested_multiplication(t: float, x: list, c: list):
    """
    Evaluates a Newton-form polynomial at a given point t using nested multiplication.
    
    Args:
        t: Point where the polynomial should be evaluated.
        x: List of x-values [x_0, ..., x_n].
        c: List of coefficients [c_0, ..., c_n].
    
    Returns:
        float: The polynomial value at t.
    """
    n = len(x)
    u = c[n - 1]
    for i in range(n - 2, -1, -1):
        u = c[i] + (t - x[i]) * u
    return u


def coefficients_by_nested_multiplication(x: list, y: list):
    """
    Computes coefficients of the Newton interpolating polynomial using
    the nested multiplication approach.
    
    Args:
        x: List of x-coordinates of data points.
        y: List of y-coordinates of data points.
    
    Returns:
        dict: {
            'coefficients': list of coefficients [c_0, ..., c_n],
            'steps': list of per-step computation logs for educational tracing.
        }
    """
    n = len(x)
    c = [0] * n
    c[0] = y[0]
    steps = []

    for k in range(1, n):
        d = x[k] - x[k - 1]
        u = c[k - 1]
        for i in range(k - 2, -1, -1):
            a = x[k] - x[i]
            u = c[i] + u * a
            d = a * d
        c[k] = (y[k] - u) / d

        steps.append({
            "k": k,
            "d": d,
            "u": u,
            "x_k": x[k],
            "y_k": y[k],
            "coefficients": c.copy()
        })

    return {"coefficients": c, "steps": steps}


def coefficients_by_divided_differences(x: list, y: list):
    """
    Computes coefficients of the Newton interpolating polynomial using divided differences.
    
    Args:
        x: List of x-coordinates of data points.
        y: List of y-coordinates of data points.
    
    Returns:
        dict: {
            'coefficients': list of divided-difference coefficients,
            'table': the full divided-difference table as np.ndarray
        }
    """
    n = len(x)
    f = np.zeros((n, n))
    f[:, 0] = y

    for col in range(1, n):
        for row in range(n - col):
            f[row, col] = (f[row + 1, col - 1] - f[row, col - 1]) / (x[row + col] - x[row])

    coefficients = f[0, :].tolist()

    return {"coefficients": coefficients, "table": f}
