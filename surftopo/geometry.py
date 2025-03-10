import numpy as np
import torch

def vector_flip_towards(vector: torch.Tensor, orientation: torch.Tensor) -> torch.Tensor:
    """
    Flip the vector towards the orientation vector.

    Args:
        vector (torch.tensor): The vector to flip.
        orientation (np.array): The orientation vector.

    Returns:
        torch.tensor: The flipped vector
    """
    if torch.dot(vector, orientation) < 0:
        return -vector
    return vector

def vector_flip_up(vector: torch.Tensor) -> torch.Tensor:
    """
    Flip the vector towards the up vector.

    Args:
        vector (torch.tensor): The vector to flip.

    Returns:
        torch.tensor: The flipped vector
    """
    up_vector = torch.tensor([0, 1], device=vector.device, dtype=vector.dtype)
    return vector_flip_towards(vector, up_vector)

def vector_normalize(vector: torch.Tensor) -> torch.Tensor:
    """
    Normalizes a given vector to have a unit norm.
    
    Args:
        vector (torch.tensor): The input vector to normalize.
        
    Returns:
        torch.tensor: The normalized vector with a unit norm.
    """
    norm = torch.linalg.vector_norm(vector, ord=2)  # Compute the Euclidean (L2) norm of the vector
    if norm == 0:
        return vector   
    return vector / norm

def vector_paralelism(vector1: torch.Tensor, vector2: torch.Tensor) -> torch.Tensor:
    """
    Calculate the paralelism between two vectors.

    Args:
        vector1 (np.ndarray): The first vector.
        vector2 (np.ndarray): The second vector.

    Returns:
        float: The paralelism value.
    """
    vector1 = vector_normalize(vector1)
    vector2 = vector_normalize(vector2)
    vector2_oriented = vector_flip_towards(vector2, vector1)
    return torch.dot(vector1, vector2_oriented)

def vector_angle_radian(vector: torch.Tensor) -> torch.Tensor:
    """
    Calculate the angle of a vector in radians.

    Args:
        vector (np.ndarray): The vector.

    Returns:
        float: The angle in radians
    """
    return torch.arctan2(vector[1], vector[0])

def calculate_vertical_angle_deviation(left_direction: torch.Tensor, right_direction: torch.Tensor):
    """
    Calculate the vertical angle deviation between two directions and the parallelism.

    Args:
        left_direction: The left direction.
        right_direction: The right direction.

    Returns:
        tuple: The vertical angle deviation and the parallelism.
    """
    left_direction = vector_flip_up(left_direction)
    left_direction = vector_normalize(left_direction)

    right_direction = vector_flip_up(right_direction)
    right_direction = vector_normalize(right_direction)

    parallelism = vector_paralelism(left_direction, right_direction)

    average_direction = vector_normalize(left_direction + right_direction)
    angle_radians = vector_angle_radian(average_direction)
    theta = torch.pi / 2 - angle_radians

    return theta, parallelism
