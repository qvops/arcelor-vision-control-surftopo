import torch
import torch.nn.functional as F

import numpy as np
import skimage

from surftopo.coordinate import Coordinate

def detect_vertical_edge_candidates(z: torch.Tensor, threshold: float = 100, invalid_value=0):
    """
    Detect the leftmost and rightmost edge candidates of a surface in a 2D tensor, starting from the center of each row.

    For each row in the input tensor, the function calculates the gradient and identifies the first and last indices 
    where the gradient exceeds the specified threshold, starting from the center of the row and moving towards 
    the left and right edges. These indices are considered as potential edge candidates.

    Args:
        z (torch.Tensor): A 2D tensor representing the surface, where each row corresponds to a data slice.
        threshold (float): The gradient magnitude threshold for detecting edge candidates. Defaults to 100.

    Returns:
        tuple: A tuple of two lists:
            - The first list contains the indices of the leftmost edge candidates for each row, starting from the center.
            - The second list contains the indices of the rightmost edge candidates for each row, starting from the center.
            If no edge candidate is detected in a row, the center index is returned
    """
    kernel = [[-1, 0, 1]]
    kernel = torch.tensor(kernel, dtype=z.dtype, device=z.device)
    
    z_batch = z.unsqueeze(0).unsqueeze(0)
    kernel_batch = kernel.unsqueeze(0).unsqueeze(0)  # 1x1x1x3
    gradient = F.conv2d(z_batch, kernel_batch, padding=(0, 1))

    gradient = gradient.squeeze(0).squeeze(0).abs()
    gradient[z == invalid_value] = 0

    center_idx = gradient.shape[1] // 2
    gradient_left = gradient[:, :center_idx]
    gradient_right = gradient[:, center_idx:]

    edges_left = (gradient_left > threshold).int()
    edges_left_reversed = edges_left.flip(dims=[1])
    edges_left_idx_reversed = torch.argmax(edges_left_reversed, dim=1)
    edges_left_idx = edges_left.size(1) - 1 - edges_left_idx_reversed

    edges_right = (gradient_right > threshold).int()
    edge_right_idx = torch.argmax(edges_right, dim=1) + center_idx

    edges_left_valid = edges_left.sum(dim=1) > 0
    edges_right_valid = edges_right.sum(dim=1) > 0
    edges_left_rows = torch.arange(edges_left.size(0), device=z.device)[edges_left_valid]
    edges_right_rows = torch.arange(edges_right.size(0), device=z.device)[edges_right_valid]
    edges_left_cols = edges_left_idx[edges_left_valid]
    edges_right_cols = edge_right_idx[edges_right_valid]

    return Coordinate(edges_left_rows, edges_left_cols), Coordinate(edges_right_rows, edges_right_cols)

def find_line_contour(x: torch.Tensor, y: torch.Tensor, residual_threshold=10, 
                      max_trials=200, stop_sample_percentage=75):
    """
    Fit a line to a set of 2D points using RANSAC and return the origin, direction, and inliers of the line.

    Args:
        x (torch.tensor): The x-coordinates of the points
        y (torch.tensor): The y-coordinates of the points
        residual_threshold (float): The maximum distance for a point to be considered an inlier. Defaults to 10.
        max_trials (int): The maximum number of iterations for RANSAC. Defaults to 100.
        stop_sample_percentage (int): The percentage of inliers required to stop RANSAC. Defaults to 75.

    Returns:
        tuple: A tuple containing the origin (x0, y0), direction (dx, dy), and inliers of the line.
    """
    device = x.device
    x_np = x.cpu().numpy()
    y_np = y.cpu().numpy()
    points = np.column_stack([x_np, y_np])

    num_points = points.shape[0]
    stop_sample_num = stop_sample_percentage * num_points // 100.0

    model_robust, inliers = skimage.measure.ransac(
        points, skimage.measure.LineModelND, min_samples=2, residual_threshold=residual_threshold, 
        max_trials=max_trials, stop_sample_num=stop_sample_num
    )
    origin, direction = model_robust.params

    origin = torch.tensor(origin, device=device)
    direction = torch.tensor(direction, device=device)
    inliers = torch.tensor(inliers, device=device)

    return origin, direction, inliers

def remove_outside_bbox(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, 
                        bbox: torch.Tensor, invalid_value=0, fill_value=torch.inf, residual_threshold=10):
    """
    Remove the points outside the bounding box defined by the input coordinates.

    Args:
        x (torch.Tensor): The x-coordinates of the points.
        y (torch.Tensor): The y-coordinates of the points.
        z (torch.Tensor): The z-coordinates of the points.
        bbox (torch.Tensor): The bounding box defined by [left, top, right, bottom].
        invalid_value (float): The value to ignore when removing points. Defaults to 0.
        fill_value (float): The value to fill the removed points with. Defaults to torch.inf.
        residual_threshold (float): The residual threshold for removing points. Defaults to 10.

    Returns:
        tuple: A tuple containing the updated x, y, and z coordinates.
    """

    bbox_left, bbox_top, bbox_right, bbox_bottom = bbox
    mask = (x < bbox_left-residual_threshold) | (x > bbox_right+residual_threshold) | (y < bbox_bottom) | (y > bbox_top)
    z = z.masked_fill(mask, invalid_value)
    x = x.masked_fill(mask, invalid_value)
    y = y.masked_fill(mask, invalid_value)

    mask = (z == invalid_value)
    x[mask] = fill_value
    y[mask] = fill_value
    z[mask] = fill_value

    min_values = torch.min(torch.masked_fill(x, x == -torch.inf, torch.inf), dim=1).values
    max_values = torch.max(torch.masked_fill(x, x == +torch.inf, -torch.inf), dim=1).values

    valid_rows = ((min_values - bbox_left) < residual_threshold) & ((bbox_right - max_values) < residual_threshold)
    z = z[valid_rows]
    x = x[valid_rows]
    y = y[valid_rows]

    return x, y, z

def fill_outside_boundary(matrix: torch.Tensor, rows_for_min_cols: torch.Tensor, min_cols: torch.Tensor, 
                                 rows_for_max_cols: torch.Tensor, max_cols: torch.Tensor,
                                 fill_value=0):
    """
    Removes the values outside the boundary defined by the minimum and maximum columns.

    Args:
        matrix (torch.Tensor): The input matrix.
        rows_for_min_cols (torch.Tensor): The rows for the minimum columns.
        min_cols (torch.Tensor): The minimum columns.
        rows_for_max_cols (torch.Tensor): The rows for the maximum columns.
        max_cols (torch.Tensor): The maximum columns.
        fill_value (float): The value to fill the outside boundary. Defaults to 0.

    Returns:
        torch.Tensor: The matrix with the values outside the boundary replaced by
    """
    result = matrix.clone()
    cols = torch.arange(result.size(1), device=result.device).unsqueeze(0)  # Shape: (1, num_cols)

    mask = cols < min_cols.unsqueeze(1)  # Mask for columns below threshold
    result[rows_for_min_cols] *= ~mask
    result[rows_for_min_cols] += mask * fill_value

    mask = cols > max_cols.unsqueeze(1)  # Mask for columns above or equal to threshold
    result[rows_for_max_cols] *= ~mask
    result[rows_for_max_cols] += mask * fill_value

    # Zero out rows outside the min_row and max_row range
    min_row = torch.min(rows_for_min_cols.min(), rows_for_max_cols.min())
    max_row = torch.max(rows_for_min_cols.max(), rows_for_max_cols.max())

    # Zero the rows above min_row and below max_row
    result[0:min_row] = fill_value  # Zero values above min_row
    result[max_row+1:] = fill_value  # Zero values below max_row

    return result

def fill_outliers_iqr(tensor: torch.Tensor, invalid_value=0, fill_value=0, scale_factor=1.5):
    """
    Replace outliers in a tensor with a specified fill value.
    Outliers are detected based on the interquartile range (IQR) method using a scale factor.
    The IQR method is more useful for data with stronger skew or heavier tails.

    Args:
        tensor (torch.Tensor): The input tensor.
        invalid_value (float): The value to ignore when detecting outliers. Defaults to 0.
        fill_value (float): The value to replace the outliers with. Defaults to 0.
        scale_factor (float): The scale factor for the IQR method. Defaults to 1.5.
            If set to 0, no outlier removal will be applied.

    Returns:
        torch.Tensor: The tensor with the outliers replaced by the fill value.
    """
    if scale_factor == 0:
        return tensor
    
    # Flatten the tensor and ignore invalid_value
    valid_values = tensor[tensor != invalid_value]
    if valid_values.numel() == 0:
        return tensor

    # to include nan values as invalid:
    # valid_values = tensor[(tensor != invalid_value) & ~torch.isnan(tensor)]
    
    # If there are no valid values, return the original tensor
    if valid_values.numel() == 0:
        return tensor

    # Calculate Q1, Q3 and IQR
    Q1 = torch.quantile(valid_values, 0.25)
    Q3 = torch.quantile(valid_values, 0.75)
    IQR = Q3 - Q1
    
    # Calculate lower and upper bounds for outliers
    lower_bound = Q1 - scale_factor * IQR
    upper_bound = Q3 + scale_factor * IQR
    
    # Create mask for outliers
    mask_outliers = (tensor < lower_bound) | (tensor > upper_bound)
    
    # Replace outliers with fill_value
    masked_tensor = tensor.masked_fill(mask_outliers, fill_value)
    
    return masked_tensor

def fill_outliers_zscore(tensor: torch.Tensor, invalid_value=0, fill_value=0, scale_factor=3.0):
    """
    Replace outliers in a tensor with a specified fill value.
    Outliers are detected based on the Z-score method using a scale factor.
    Given slightly skewed data the Z-score method might be a better choice for outlier removal.
    Much faster than fill_outliers_iqr.

    Args:
        tensor (torch.Tensor): The input tensor.
        invalid_value (float): The value to ignore when detecting outliers. Defaults to 0.
        fill_value (float): The value to replace the outliers with. Defaults to 0.
        scale_factor (float): The Z-score threshold for detecting outliers. Defaults to 3.
            If set to 0, no outlier removal will be applied.

    Returns:
        torch.Tensor: The tensor with the outliers replaced by the fill value.
    """
    if scale_factor == 0:
        return tensor
        
    # Flatten the tensor and ignore invalid_value
    valid_values = tensor[tensor != invalid_value]
    
    # If there are no valid values, return the original tensor
    if valid_values.numel() == 0:
        return tensor

    # Calculate mean and standard deviation
    mean = valid_values.mean()
    std = valid_values.std()

    # Avoid division by zero if standard deviation is zero
    if std == 0:
        return tensor

    # Calculate Z-scores for the tensor
    z_scores = (tensor - mean) / std

    # Create mask for outliers based on scale_factor threshold
    mask_outliers = torch.abs(z_scores) > scale_factor

    # Replace outliers with fill_value
    masked_tensor = tensor.masked_fill(mask_outliers, fill_value)

    return masked_tensor

def fill_burr(matrix, n=1):
    """
    Sets every value in the matrix to zero if it is within N neighbors of a zero value.
    Fill burr (rebaba in spanish) is a filter that removes values next to invalid values.

    Args:
        matrix (torch.Tensor): The input matrix.
        n (int): The number of neighbors to consider. Defaults to 1.

    Returns:
        torch.Tensor: The matrix with the zeros filled.
    """
    if n == 0:
        return matrix
    
    matrix = matrix.float().unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

    # Create a kernel of size (2n+1) x (2n+1) to check the neighborhood
    kernel_size = 2 * n + 1
    kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=matrix.dtype, device=matrix.device)
    kernel[:, :, n, n] = 0  # Exclude the center (self) value
    
    # Create a binary mask where the input matrix has zeros
    zero_mask = (matrix == 0).float()

    # Perform convolution to detect N-neighborhood of zeros
    neighbor_mask = F.conv2d(zero_mask, kernel, padding=n)
    neighbor_mask = (neighbor_mask > 0).float()  # Any non-zero value means within N neighbors of a zero

    # Set values within N neighbors of zeros to zero
    result = matrix * (1 - neighbor_mask)

    # Remove added dimensions and return result
    return result.squeeze(0).squeeze(0)

def crop(X, Y, Z, min_col, max_col, min_row, max_row):
    """
    Crop the input arrays X, Y, Z by the specified row and column range.
    The slice includes max_row and max_col (hence +1 in slicing).

    Args:
        X (torch.Tensor): The x-coordinates.
        Y (torch.Tensor): The y-coordinates.
        Z (torch.Tensor): The z-coordinates.
        min_col (int): The minimum column index.
        max_col (int): The maximum column index.
        min_row (int): The minimum row index.
        max_row (int): The maximum row index.

    Returns:
        tuple: A tuple containing the cropped X, Y, and Z arrays.
    """
    X_cropped = X[min_row:max_row+1, min_col:max_col+1]
    Y_cropped = Y[min_row:max_row+1, min_col:max_col+1]
    Z_cropped = Z[min_row:max_row+1, min_col:max_col+1]
    
    return X_cropped, Y_cropped, Z_cropped

def fill_invalid_values(X, Y, invalid_mask, fill_value=0):
    """
    Fill the invalid values in the input X and Y tensors with the specified fill value.

    Args:
        X (torch.Tensor): The input tensor X.
        Y (torch.Tensor): The input tensor Y.
        invalid_mask (torch.Tensor): The mask for invalid values.
        fill_value (float): The value to fill the invalid values with. Defaults to 0.

    Returns:
        tuple: A tuple containing the filled X and Y tensors.
    """
    x_filled = X.masked_fill(invalid_mask, fill_value)
    y_filled = Y.masked_fill(invalid_mask, fill_value)

    return x_filled, y_filled

def reduced_bbox(bbox_left, bbox_top, bbox_right, bbox_bottom, margin_left, margin_top, margin_right, margin_bottom):
    """
    Reduces the size of the working bounding box by a specified margin.

    Args:
        bbox_left (torch.Tensor): The left coordinate of the bounding box.
        bbox_right (torch.Tensor): The right coordinate of the bounding box.
        bbox_top (torch.Tensor): The top coordinate of the bounding box.
        bbox_bottom (torch.Tensor): The bottom coordinate of the bounding box.
        margin_left (int): The margin to reduce the left side of the bounding box.
        margin_right (int): The margin to reduce the right side of the bounding box.
        margin_top (int): The margin to reduce the top side of the bounding box.
        margin_bottom (int): The margin to reduce the bottom side of the bounding box.

    Returns:
        torch.Tensor: The reduced bounding box coordinates.
    """

    bbox_left = bbox_left + margin_left
    bbox_top = bbox_top - margin_top
    bbox_right = bbox_right - margin_right
    bbox_bottom = bbox_bottom + margin_bottom

    bbox = torch.tensor([bbox_left, bbox_top, bbox_right, bbox_bottom])
    return bbox

def remove_top_bottom_margin(X: torch.Tensor, Y: torch.Tensor, Z: torch.Tensor, top_bottom_margin_removal_mm):
    """
    Remove the top and bottom margin from the input X, Y, and Z tensors.

    Args:
        X (torch.Tensor): The x-coordinates.
        Y (torch.Tensor): The y-coordinates.
        Z (torch.Tensor): The z-coordinates.
        top_bottom_margin_removal_mm (float): The margin to remove from the top and bottom.

    Returns:
        tuple: A tuple containing the updated X, Y, and Z tensors.
    """
    length = Y[:, 0].contiguous()
    lower_limit = Y[0, 0] + top_bottom_margin_removal_mm
    upper_limit = Y[-1, 0] - top_bottom_margin_removal_mm

    lower_idx = torch.searchsorted(length, lower_limit, right=False)
    upper_idx = torch.searchsorted(length, upper_limit, right=True)
    valid_row_idx = torch.tensor([lower_idx, upper_idx], device=Y.device)

    X_valid, Y_valid, Z_valid = X[valid_row_idx[0]:valid_row_idx[1], :], Y[valid_row_idx[0]:valid_row_idx[1], :], Z[valid_row_idx[0]:valid_row_idx[1], :]
    return X_valid, Y_valid, Z_valid, valid_row_idx

def remove_top_bottom_rows(X: torch.Tensor, Y: torch.Tensor, Z: torch.Tensor, valid_row_idx):
    """
    Remove the top and bottom margin from the input X, Y, and Z tensors.

    Args:
        X (torch.Tensor): The x-coordinates.
        Y (torch.Tensor): The y-coordinates.
        Z (torch.Tensor): The z-coordinates.
        valid_col_idx (torch.Tensor): The valid column indices

    Returns:
        tuple: A tuple containing the updated X, Y, and Z tensors.
    """
    X_valid, Y_valid, Z_valid = X[valid_row_idx[0]:valid_row_idx[1], :], Y[valid_row_idx[0]:valid_row_idx[1], :], Z[valid_row_idx[0]:valid_row_idx[1], :]
    return X_valid, Y_valid, Z_valid

def create_uniform_grid(X: torch.Tensor, Y: torch.Tensor,bbox_left, bbox_right, bbox_bottom, bbox_top, resolution_mm=1, tolerance=1e-4):
    """
    Create a uniform grid of coordinates within the specified bounding box.

    Args:
        X (torch.Tensor): The x-coordinates.
        Y (torch.Tensor): The y-coordinates.
        bbox_left (torch.Tensor): The left coordinate of the bounding box.
        bbox_right (torch.Tensor): The right coordinate of the bounding box.
        bbox_top (torch.Tensor): The top coordinate of the bounding box.
        bbox_bottom (torch.Tensor): The bottom coordinate of the bounding box.
        resolution_mm (float): The resolution of the grid in millimeters.
        tolerance (float): The tolerance for the bbox limits. Defaults to 1e-4.

    Returns:
        tuple: A tuple containing the X and Y coordinate grids.
    """


    newx = torch.arange(bbox_left, bbox_right + tolerance, resolution_mm, device=X.device, dtype=X.dtype)
    newy = torch.arange(bbox_bottom, bbox_top + tolerance, resolution_mm, device=Y.device, dtype=Y.dtype)
    X_uni, Y_uni = torch.meshgrid(newx, newy, indexing='xy')
    return X_uni, Y_uni, newx, newy

def tolerance_ceil(value, tolerance=1e-5):
    """
    Round the input value to the nearest integer with a specified tolerance.

    Args:
        value (float): The input value.
        tolerance (float): The tolerance for rounding. Defaults to 1e-5.
    
    Returns:
        float: The rounded value.
    """
    return (value-tolerance).ceil()