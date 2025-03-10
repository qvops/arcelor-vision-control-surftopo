import numpy as np
import torch

def validate_inputs(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    """
    Validate the input matrices.
    """
    # Ensure all matrices are numpy arrays
    if not all(isinstance(m, np.ndarray) for m in [x, y, z]):
        raise TypeError("All inputs must be numpy arrays.")
    
    # Ensure all matrices are 2D
    if not all(m.ndim == 2 for m in [x, y, z]):
        raise ValueError("All input matrices must be 2-dimensional.")    

    # Ensure matrices are the same size
    if x.shape != y.shape or y.shape != z.shape:
        raise ValueError("All input matrices must be the same size.")
    
def validate_lines(theta, parallelism, options):
    """
    Ensure that the lines are parallel and the angle is within a reasonable range.
    """
    # Ensure that the parallelism is above the minimum threshold
    if parallelism < options.min_valid_parallelism:
        raise ValueError(f"Lines are not parallel enough. Parallelism: {parallelism}, Min Parallelism: {options.min_valid_parallelism}")
    
    # Ensure that the angle is within a reasonable range
    theta_degrees = torch.rad2deg(theta)
    if abs(theta_degrees) > options.min_valid_theta_deg:
        raise ValueError(f"Vertical edges deviation is too large. Angle: {theta_degrees}º, Min valid deviation: {options.min_valid_theta_deg}º")
    
def validate_bounding_box(bbox, options):
    """
    Ensure that the bounding box is large enough.
    """
    bbox_left, bbox_top, bbox_right, bbox_bottom = bbox

    width = bbox_right - bbox_left
    if width < options.min_width_mm:
        raise ValueError(f"Bounding box width is too small. Width: {bbox_right - bbox_left}, Min Width: {options.min_width_mm}")

    height = bbox_top - bbox_bottom
    if height < options.min_height_mm:
        raise ValueError(f"Bounding box height is too small. Height: {bbox_top - bbox_bottom}, Min Height: {options.min_height_mm}")
    
    if bbox_left > bbox_right:
        raise ValueError("Bounding box left edge is on the right side of the right edge.")
    
    if bbox_bottom > bbox_top:
        raise ValueError("Bounding box bottom edge is on the top side of the top edge.")
    
def validate_dimensions(tensor, options):
    """
    Ensure that the tensor has a sufficient number of rows and columns.
    """
    rows, cols = tensor.shape

    if rows < options.min_valid_rows:
        raise ValueError(f"Number of valid rows is too small. Rows: {rows}, Min Rows: {options.min_valid_rows}")
    
    if cols < options.min_valid_cols:
        raise ValueError(f"Number of valid columns is too small. Columns: {cols}, Min Columns: {options.min_valid_cols}")
    
def validate_values(tensor):
    """
    Ensure that the tensor does not contain any invalid values (nan or inf).
    """
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        raise ValueError("Invalid values in tensor.")
    
def validate_shapes(x, y, z):
    """
    Ensure that all matrices have the same shape
    """
    if x.shape != y.shape or y.shape != z.shape:
        raise ValueError(f"All matrices must be the same size: {x.shape}, {y.shape}, {z.shape}")

def validate_left_right(x_left, x_right):
    """
    Ensure that the left edge is on the left side of the right edge.
    """
    if x_left.mean() > x_right.mean():
        raise ValueError("Left edge is on the right side of the right edge.")
    
    if len(x_left) < 2:
        raise ValueError(f"Left edge is too short: {len(x_left)} edges.")
    
    if len(x_right) < 2:
        raise ValueError(f"Right edge is too short: {len(x_right)} edges.")
