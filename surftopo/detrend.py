import torch

def remove_plane(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, use_center=True):
    """
    Remove a plane from a 2D surface.

    Args:
        x (torch.Tensor): The x-coordinates of the surface.
        y (torch.Tensor): The y-coordinates of the surface.
        z (torch.Tensor): The z-coordinates of the surface.
        use_center (bool): If True, only the center half of the surface is used to fit the plane.

    Returns:
        torch.Tensor: The detrended z-coordinates.
    """
    if use_center:
        num_cols = x.shape[1]
        center_cols = num_cols // 2
        half_width = center_cols // 2

        start_idx = center_cols - half_width
        end_idx = center_cols + half_width
    else:
        start_idx = 0
        end_idx = x.shape[1]

    # Step 1: Flatten the input data
    x_flat = x[:, start_idx:end_idx].flatten()
    y_flat = y[:, start_idx:end_idx].flatten()
    z_flat = z[:, start_idx:end_idx].flatten()

    # Step 2: Construct the design matrix A
    A = torch.stack([x_flat, y_flat, torch.ones_like(x_flat)], dim=1)  # Shape: [n, 3]

    # Step 3: Solve for plane coefficients [a, b, c] using least squares
    solution = torch.linalg.lstsq(A, z_flat.unsqueeze(1))  # Returns a result object
    coefficients = solution.solution.squeeze()  # Extract the coefficients
    a, b, c = coefficients

    # Step 4: Calculate the plane values and distances
    z_plane = a * x + b * y + c  # Plane values at each (x, y)
    z_detrend = (z - z_plane) / torch.sqrt(a**2 + b**2 + 1)  # Perpendicular distance    

    return z_detrend

def remove_fiber_mean(z: torch.Tensor):
    """
    Remove the mean of a fiber from the z-coordinates.

    Args:
        z (torch.Tensor): The z-coordinates of the surface.

    Returns:
        torch.Tensor: The detrended z-coordinates.
    """
    fiber_mean = torch.mean(z, dim=0)
    z_detrend = z - fiber_mean

    return z_detrend

def remove_fiber_median(z: torch.Tensor):
    """
    Remove the mean of a fiber from the z-coordinates.

    Args:
        z (torch.Tensor): The z-coordinates of the surface.

    Returns:
        torch.Tensor: The detrended z-coordinates.
    """
    fiber_median = torch.median(z, dim=0).values
    z_detrend = z - fiber_median

    return z_detrend

def remove_fiber_linear_regression(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
    """
    Remove the linear trend from a fiber.

    Args:
        x (torch.Tensor): The x-coordinates of the fiber.
        y (torch.Tensor): The y-coordinates of the fiber.

    Returns:
        torch.Tensor: The detrended y-coordinates.
    """
    y = y[:, 0].squeeze()

    assert y.ndim == 1
    assert y.shape[0] == z.shape[0]

    A = torch.stack([y, torch.ones_like(y)], dim=1)
    B = z

    AtA = A.T @ A
    AtB = A.T @ B

    # Computes the solution of a square system of linear equations
    coefficients = torch.linalg.solve(AtA, AtB)

    trend = A @ coefficients
    z_detrend = z - trend

    return z_detrend

detrends = {
    "Surface_plane": remove_plane,
    "Fiber_linear_regression": remove_fiber_linear_regression,
}