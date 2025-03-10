import torch

def rotate_2d(x: torch.Tensor, y: torch.Tensor, angle_radians: torch.Tensor):
    """
    Rotate a 2D tensor of points around the origin by a given angle.

    Parameters
    ----------
    x : torch.Tensor
        The x-coordinates of the points.
    y : torch.Tensor
        The y-coordinates of the points.
    angle_radians : torch.Tensor
        The angle to rotate the points by, in radians.

    Returns
    -------
    torch.Tensor
        The x-coordinates of the rotated points.
    torch.Tensor
        The y-coordinates of the rotated points.
    """

    cos = torch.cos(angle_radians)
    sin = torch.sin(angle_radians)
    rotation_matrix = torch.tensor([
        [cos, -sin],
        [sin, cos]
    ], device=x.device, dtype=x.dtype)

    # Concatenate x and y into a 2D tensor (each column is a point)
    points = torch.stack([x.flatten(), y.flatten()], dim=0)

    # Apply the rotation matrix to all points at once (matrix multiplication)
    rotated_points = torch.matmul(rotation_matrix, points)

    # Reshape the result back to the original shape of x and y
    rotated_X = rotated_points[0, :].reshape(x.shape)
    rotated_Y = rotated_points[1, :].reshape(y.shape)

    return rotated_X, rotated_Y

