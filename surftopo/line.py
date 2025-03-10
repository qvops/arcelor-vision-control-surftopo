import torch

def line_eval(origin: torch.Tensor, direction: torch.Tensor, x: torch.Tensor, axis: int) -> torch.tensor:
    """
    Evaluate a line at a given value and axis.
    Args:
        origin: origin of the line
        direction: direction of the line
        x: x value to evaluate the line at
        axis: axis of the x value
    Returns:
        y: y value of the line at x given an axis
    """
    l = (x - origin[axis]) / direction[axis]
    data = origin + l[..., torch.newaxis] * direction
    return data

def line_eval_x(origin: torch.Tensor, direction: torch.Tensor, y: torch.Tensor) -> torch.tensor:
    """
    Evaluate a line at a given y value.
    Args:
        origin: origin of the line
        direction: direction of the line
        y: y value to evaluate the line at
    Returns:
        x: x value of the line at y
    """
    x = line_eval(origin, direction, y, axis=1)[...,0]
    return x

def line_eval_y(origin: torch.Tensor, direction: torch.Tensor, x: torch.Tensor) -> torch.tensor:
    """
    Evaluate a line at a given y value.
    Args:
        origin: origin of the line
        direction: direction of the line
        y: y value to evaluate the line at
    Returns:
        x: x value of the line at y
    """
    y = line_eval(origin, direction, x, axis=0)[...,1]
    return y

def line_distance_point(origin: torch.Tensor, direction: torch.Tensor, point: torch.Tensor) -> torch.tensor:
    """
    Calculate the distance between a line and a point.
    Args:
        origin: origin of the line
        direction: direction of the line
        point: point to calculate the distance to
    Returns:
        distance: distance between the line and the point
    """
    # Vector from line origin to point
    v = point - origin
    
    # Project v onto the direction
    projection = torch.dot(v, direction) / torch.dot(direction, direction) * direction
    
    # Perpendicular vector
    perpendicular_vector = v - projection
    
    # Distance is the norm of the perpendicular vector
    distance = torch.linalg.vector_norm(perpendicular_vector)
    return distance