import torch

def legendre_polynomial(n: int, x: torch.Tensor) -> torch.Tensor:
    """
    Evaluate the nth order Legendre polynomial at the given x values.

    Args:
        n (int): The order of the Legendre polynomial.
        x (torch.Tensor): The x values to evaluate the polynomial.

    Returns:
        torch.Tensor: The evaluated polynomial.
    """
    if n == 0:
        return torch.ones_like(x)
    if n == 1:
        return x

    P_n_2 = 1
    P_n_1 = x

    for i in range(2, n + 1):
        P_n = ((2 * i - 1) * x * P_n_1 - (i - 1) * P_n_2) / i
        P_n_2 = P_n_1
        P_n_1 = P_n

    return P_n

def chebyshev_polynomial(n: int, x: torch.Tensor) -> torch.Tensor:
    """
    Evaluate the nth order Chebyshev polynomial at the given x values.

    Args:
        n (int): The order of the Chebyshev polynomial.
        x (torch.Tensor): The x values to evaluate the polynomial.

    Returns:
        torch.Tensor: The evaluated polynomial.
    """
    if n == 0:
        return torch.ones_like(x)
    if n == 1:
        return x

    P_n_2 = 1
    P_n_1 = x

    for i in range(2, n + 1):
        P_n = 2 * x * P_n_1 - P_n_2
        P_n_2 = P_n_1
        P_n_1 = P_n
    
    return P_n

# Milton Abramowitz and Irene A. Stegun, eds.
# Handbook of Mathematical Functions with Formulas, Graphs, and Mathematical Tables.
# New York: Dover, 1972.
def hermite_polynomial(n: int, x: torch.Tensor) -> torch.Tensor:
    """
    Evaluate the nth order Hermite probabilistic polynomial at the given x values.
    The physical version of hermite could be calculated using:
        physical = evaluate_hermite(n, math.sqrt(2) * x) * math.pow(2, n / 2.0)

    Args:
        n (int): The order of the Hermite polynomial.
        x (torch.Tensor): The x values to evaluate the polynomial.

    Returns:
        torch.Tensor: The evaluated polynomial.
    """
    if isinstance(x, float) and (x != x):  # Check if x is NaN
        return x
    if n < 0:
        raise ValueError("Polynomial only defined for nonnegative n")

    if n == 0:
        return 1.0

    if n == 1:
        return x

    y3 = 0.0
    y2 = 1.0
    for k in range(n, 1, -1):  
        y1 = x * y2 - k * y3
        y3 = y2
        y2 = y1

    return x * y2 - y3

def standard_polynomial(n: int, x: torch.Tensor) -> torch.Tensor:
    """
    Evaluate the nth order standard polynomial at the given x values.

    Args:
        n (int): The order of the standard polynomial.
        x (torch.Tensor): The x values to evaluate the polynomial.

    Returns:
        torch.Tensor: The evaluated polynomial.
    """
    if n == 0:
        return torch.ones_like(x)
    if n == 1:
        return x
    return torch.pow(x, n)

polynomials = {
    "Legendre": legendre_polynomial,
    "Chebyshev": chebyshev_polynomial,
    "Hermite": hermite_polynomial,
    "Standard": standard_polynomial
}

def calculate_polynomial_matrix(n: int, x: torch.Tensor,
                                normalize_x=False,
                                polynomial=legendre_polynomial) -> torch.Tensor:
    """
    Calculate the matrix of polynomials up to the nth order.

    Args:
        n (int): The order of the polynomial.
        x (torch.Tensor): The x values to evaluate the polynomial.
        polynomial (Callable): The polynomial function to use. Defaults to `legendre_polynomial`.

    Returns:
        torch.Tensor: The matrix of polynomials
    """
    if normalize_x:
        x_min = x.min()
        x_max = x.max()
        x = 2 * (x - x_min) / (x_max - x_min) - 1

    # Initialize a tensor of zeros with shape (len(x), n + 1)
    P = torch.zeros((x.shape[0], n + 1), dtype=x.dtype, device=x.device)

    # Compute the polynomial values
    for j in range(n + 1):
        P[:, j] = polynomial(j, x)

    return P

def calculate_polynomial_fit(n: int, x: torch.Tensor, z: torch.Tensor, device,
                             polynomial=legendre_polynomial,
                             evaluate=False) -> torch.Tensor:
    """
    Calculate the coefficients of the polynomial fit.

    Args:
        n (int): The order of the polynomial.
        x (torch.Tensor): The x values to evaluate the polynomial.
        z (torch.Tensor): The y values to fit the polynomial.
        device: The device to use.
        polynomial (Callable): The polynomial function to use. Defaults to `legendre_polynomial`.
        evaluate (bool): Whether to evaluate the polynomial fit. If true returns the fitted values.

    Returns:
        torch.Tensor: The coefficients of the polynomial fit.
    """
    A = calculate_polynomial_matrix(n, x, normalize_x=True, polynomial=polynomial)

    batchA = A.unsqueeze(0).repeat(z.shape[0], 1, 1)
    if device != z.device:
        batchA = batchA.to(device=device)
        z = z.to(device)

    coefficients = torch.linalg.lstsq(batchA, z).solution

    if evaluate:
        return (A @ coefficients.T).T

    return coefficients