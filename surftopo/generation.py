import torch
import numpy as np
from scipy.interpolate import interp1d

def generate_test_surface(nx=1000, ny=1000, device="cuda"):
    """
    Generate a test surface with a sinusoidal pattern and a lipss pattern.

    Parameters
    ----------
    nx : int
        The number of points in the x-axis.
    ny : int
        The number of points in the y-axis.
    device : str
        The device to use for the tensors.

    Returns
    -------
    X : torch.Tensor
        The x-coordinates of the surface.
    Y : torch.Tensor
        The y-coordinates of the surface.
    Z : torch.Tensor
        The z-coordinates of the surface.
    """
    np.random.seed(0)
    step = 0.1
    period = 20 / step
    period_lipss = 1 / step
    x = np.arange(nx).astype(float)
    y = np.arange(ny).astype(float)
    x, y = np.meshgrid(x, y)
    z = np.sin(x / period * 2 * np.pi)
    z += 0.5 * (np.sin((x - period / 2) / period * 2 * np.pi) + 1) * 0.3 * np.sin(
        (np.sin(y / 10) + x) / period_lipss * 2 * np.pi)
    z += np.random.normal(size=z.shape) / 5
    z *= 5
    z[:, nx//9:9*nx//10] += 100    

    X = torch.from_numpy(x).to(device).float()
    Y = torch.from_numpy(y).to(device).float()
    Z = torch.from_numpy(z).to(device).float()
    return X, Y, Z

def generate_sinusoidal_surface(N=1000, M=1000, dtype=torch.float64, params=None, interpolation="linear", z_offset = 1000, borders = 100, device="cuda", slope_x=0, slope_y=0):
    """Generate a 2D sinusoidal surface with interpolation.

    Parameters:
        N (int): Number of rows (samples).
        M (int): Number of columns.
        params (list of tuples): List of (column index, period T, amplitude A).
        
    Returns:
        surface (torch.Tensor): Generated 2D surface.
    """
    if params is None:
        params = [
            (0, 200, 100),        # First column
            (M // 3, 300, 0.0),   # Middle column
            (M // 2, 300, 0.0),   # Middle column
            (2*M // 3, 300, 0.0),   # Middle column
            (M - 1, 400, 100.0)     # Last column
        ]

    t = torch.arange(N, dtype=dtype, device=device)  # Time index
    z = torch.zeros((N, M), dtype=dtype, device=device)

    # Assign sinusoidal values for the specified columns
    known_x = []
    known_y = []
    for col, T, A in params:
        signal = A/2 * torch.sin(2 * torch.pi * t / T)
        z[:, col] = signal
        known_x.append(col)
        known_y.append(signal.cpu().numpy())

    known_x = torch.tensor(known_x, dtype=dtype, device=device).cpu().numpy()
    known_y = torch.stack([torch.tensor(y) for y in known_y], dim=1).cpu().numpy()

    # Interpolate row-wise
    for i in range(N):
        interpolator = interp1d(known_x, known_y[i], kind=interpolation)
        z[i] = torch.tensor(interpolator(torch.arange(M, dtype=dtype))).to(device)

    # num_rows, num_cols = z.shape
    x = torch.arange(N, dtype=dtype, device=device)
    y = torch.arange(M, dtype=dtype, device=device)
    x, y = torch.meshgrid(x, y, indexing='xy')
    z += slope_x * x + slope_y * y
    z += z_offset
    set_border_columns_value(borders, 0, z)

    return x, y, z

def set_border_columns_value(N, value, *matrices):
    """
    Set the first and last N columns of the input matrices to a specific value.

    Parameters
    ----------
    N : int
        The number of columns to set to the specified value.
    value : float
        The value to set the columns to.
    matrices : torch.Tensor
        The matrices to update.
    """
    updated_matrices = []
    for matrix in matrices:
        _, cols = matrix.shape

        # Ensure N is valid
        if N < 0 or 2 * N > cols:
            raise ValueError(f"N={N} is too large for a matrix with {cols} columns.")

        # Set first and last N columns to zero
        matrix[:, :N] = value
        matrix[:, -N:] = value
        updated_matrices.append(matrix)

    return updated_matrices