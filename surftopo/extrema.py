import torch
import surftopo.interpolation
import scipy.signal
import numpy as np

def find_fiber_local_extrema_mask_simple(z: torch.Tensor):
    """
    Find local minima and maxima mask per fiber.

    Parameters
    ----------
    z : torch.Tensor
        The input tensor, each col is a fiber.

    Returns
    -------
    minima_mask
        The mask of local minima.
    maxima_mask
        The mask of local maxima.
    """
    # Compute first-order differences along dim
    diff = torch.diff(z, dim=0)

    # Find sign changes
    sign_change = torch.sign(diff[:-1, :]) - torch.sign(diff[1:, :])

    # Local maxima: sign changes from +1 to -1
    maxima_mask = sign_change > 0
    
    # Local minima: sign changes from -1 to +1
    minima_mask = sign_change < 0

    # Pad the masks to match the size of z
    # Since diff reduces the size by 2, we pad with False (no extrema at boundaries)
    maxima_mask = torch.nn.functional.pad(maxima_mask, (0, 0, 1, 1), mode='constant', value=False)
    minima_mask = torch.nn.functional.pad(minima_mask, (0, 0, 1, 1), mode='constant', value=False)

    return minima_mask, maxima_mask

def find_fiber_local_extrema_mask_not_repeated(z: torch.Tensor):
    """
    Find local minima and maxima mask per fiber. Only keeps the first of any adjacent pairs of equal values

    Parameters
    ----------
    z : torch.Tensor
        The input tensor, each col is a fiber.

    Returns
    -------
    minima_mask
        The mask of local minima.
    maxima_mask
        The mask of local maxima.
    """
    not_repeated = z[:-1, :] != z[1:, :]

    # Create a row of True values with the same number of columns
    true_row = torch.ones((1, not_repeated.shape[1]), dtype=torch.bool, device=z.device)

    # Insert the row at the beginning
    not_repeated_mask = torch.cat((true_row, not_repeated), dim=0)

    z_not_repeated = torch.masked_fill(z, ~not_repeated_mask, torch.inf)

    order = not_repeated_mask.to(dtype=torch.uint8).argsort(dim=0, descending=True, stable=True)
    z_not_repeated_sort = torch.gather(z_not_repeated, dim=0, index=order)

    diff = torch.diff(z_not_repeated_sort, dim=0)

    sign_change = torch.sign(diff[:-1, :]) - torch.sign(diff[1:, :])

    maxima_mask = sign_change > 0
    minima_mask = sign_change < 0

    maxima_mask = torch.nn.functional.pad(maxima_mask, (0, 0, 1, 1), mode='constant', value=False)
    minima_mask = torch.nn.functional.pad(minima_mask, (0, 0, 1, 1), mode='constant', value=False)

    reverse_order = order.argsort(dim=0)

    maxima_mask = torch.gather(maxima_mask, dim=0, index=reverse_order)
    maxima_mask[~not_repeated_mask] = False

    minima_mask = torch.gather(minima_mask, dim=0, index=reverse_order)
    minima_mask[~not_repeated_mask] = False

    return minima_mask, maxima_mask

def find_fiber_local_extrema_mask_simple_prominence(z: torch.Tensor, prominence_threshold=0):
    """
    Find local minima and maxima mask per fiber considering prominence to avoid noise. Only keeps the first of any adjacent pairs of equal values
    Prominence is calculate as the min distance from adjacent extrema (classical definition of prominence)

    Parameters
    ----------
    z : torch.Tensor
        The input tensor, each col is a fiber.

    Returns
    -------
    minima_mask
        The mask of local minima.
    maxima_mask
        The mask of local maxima.
    """    
    not_repeated = z[:-1, :] != z[1:, :]

    # Create a row of True values with the same number of columns
    true_row = torch.ones((1, not_repeated.shape[1]), dtype=torch.bool, device=z.device)

    # Insert the row at the beginning
    not_repeated_mask = torch.cat((true_row, not_repeated), dim=0)

    z_not_repeated = torch.masked_fill(z, ~not_repeated_mask, torch.inf)

    order_repeated = not_repeated_mask.to(dtype=torch.uint8).argsort(dim=0, descending=True, stable=True)
    z_not_repeated_sort = torch.gather(z_not_repeated, dim=0, index=order_repeated)

    # Compute differences and find extrema
    diff = torch.diff(z_not_repeated_sort, dim=0)
    sign_change = torch.sign(diff[:-1, :]) - torch.sign(diff[1:, :])
    extrema_mask = sign_change != 0

    # Pad first and last rows with False
    false_row = torch.full((1, extrema_mask.shape[1]), False, dtype=torch.bool, device=z.device)
    extrema_mask = torch.cat([false_row, extrema_mask, false_row], dim=0)

    # Generate row indices
    num_rows, num_cols = extrema_mask.shape
    row_indices = torch.arange(num_rows, device=z.device).view(-1, 1).expand(-1, num_cols)

    # Mask and sort extrema indices
    extrema_indices = torch.masked_fill(row_indices, ~extrema_mask, 0)

    # Sort extrema indices
    order = torch.argsort(extrema_indices, dim=0)
    extrema_indices_sorted = torch.gather(extrema_indices, 0, index=order)

    extrema_sorted_values = torch.gather(z_not_repeated_sort, 0, extrema_indices_sorted)
    extrema_sorted_values = torch.masked_fill(extrema_sorted_values, extrema_indices_sorted==0, -torch.inf)
    extrema_diff_prev = torch.diff(extrema_sorted_values, dim=0)
    extrema_diff_next = extrema_sorted_values[:-1] - extrema_sorted_values[1:]

    pad_row = torch.full((1, extrema_sorted_values.shape[1]), torch.inf, device=extrema_sorted_values.device)
    extrema_diff_prev = torch.cat([
        pad_row,
        extrema_diff_prev
    ], dim=0)
    extrema_diff_next = torch.cat([
        extrema_diff_next,
        pad_row, 
    ], dim=0)
    prominence = torch.min(extrema_diff_prev, extrema_diff_next)

    reverse_order = order.argsort(dim=0)
    prominence = torch.gather(prominence, 0, index=reverse_order)
    prominence = torch.masked_fill(prominence, ~extrema_mask, -torch.inf)

    maxima_mask = prominence > prominence_threshold
    minima_mask = (prominence < -prominence_threshold) & (~torch.isinf(prominence))

    reverse_order_repeated = order_repeated.argsort(dim=0)

    maxima_mask = torch.gather(maxima_mask, dim=0, index=reverse_order_repeated)
    maxima_mask[~not_repeated_mask] = False

    minima_mask = torch.gather(minima_mask, dim=0, index=reverse_order_repeated)
    minima_mask[~not_repeated_mask] = False

    return minima_mask, maxima_mask

def torch_nanminmax(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute element-wise minimum and maximum of two tensors, handling NaN values.

    Parameters
    ----------
    a : torch.Tensor
        The first input tensor.
    b : torch.Tensor
        The second input tensor.

    Returns
    -------
    torch.Tensor
        The element-wise minimum tensor ignoring nans.
    torch.Tensor
        The element-wise maximum tensor ignoring nans.
    """
    # Handle NaNs by replacing them with values from the other tensor
    a_nan = torch.isnan(a)
    b_nan = torch.isnan(b)

    # Replace NaN values with corresponding values from the other tensor
    a_fixed = torch.where(a_nan, b, a)
    b_fixed = torch.where(b_nan, a, b)

    # Compute element-wise minimum
    return torch.minimum(a_fixed, b_fixed), torch.maximum(a_fixed, b_fixed)

def find_fiber_local_extrema_mask(z: torch.Tensor, prominence_threshold=0):
    """
    Find local minima and maxima mask per fiber considering prominence to avoid noise. Only keeps the first of any adjacent pairs of equal values
    Prominence is calculate as the max distance from adjacent extrema that is the best approach for valley depth calculations

    Parameters
    ----------
    z : torch.Tensor
        The input tensor, each col is a fiber.

    Returns
    -------
    minima_mask
        The mask of local minima.
    maxima_mask
        The mask of local maxima.
    """
    not_repeated = z[:-1, :] != z[1:, :]

    # Create a row of True values with the same number of columns
    true_row = torch.ones((1, not_repeated.shape[1]), dtype=torch.bool, device=z.device)

    # Insert the row at the beginning
    not_repeated_mask = torch.cat((true_row, not_repeated), dim=0)

    z_not_repeated = torch.masked_fill(z, ~not_repeated_mask, torch.inf)

    order_not_repeated = not_repeated_mask.to(dtype=torch.uint8).argsort(dim=0, descending=True, stable=True)
    z_not_repeated_sort = torch.gather(z_not_repeated, dim=0, index=order_not_repeated)

    # Compute differences and find extrema
    diff = torch.diff(z_not_repeated_sort, dim=0)
    sign_change = torch.sign(diff[:-1, :]) - torch.sign(diff[1:, :])
    maxima_mask = sign_change > 0
    minima_mask = sign_change < 0
    extrema_mask = sign_change != 0

    # Pad first and last rows with False
    false_row = torch.full((1, extrema_mask.shape[1]), False, dtype=torch.bool, device=z.device)
    extrema_mask = torch.cat([false_row, extrema_mask, false_row], dim=0)

    # Generate row indices
    num_rows, num_cols = extrema_mask.shape
    row_indices = torch.arange(num_rows, device=extrema_mask.device).view(-1, 1).expand(-1, num_cols)

    # Mask and sort extrema indices
    extrema_indices = torch.masked_fill(row_indices, ~extrema_mask, 0)

    # Sort extrema indices
    order = torch.argsort(extrema_indices, dim=0)
    extrema_indices_sorted = torch.gather(extrema_indices, 0, index=order)

    extrema_sorted_values = torch.gather(z_not_repeated_sort, 0, extrema_indices_sorted)
    extrema_sorted_values = torch.masked_fill(extrema_sorted_values, extrema_indices_sorted==0, torch.nan)
    extrema_diff_prev = torch.diff(extrema_sorted_values, dim=0)
    extrema_diff_next = extrema_sorted_values[:-1] - extrema_sorted_values[1:]

    pad_row = torch.full((1, extrema_sorted_values.shape[1]), torch.nan, device=extrema_sorted_values.device)
    extrema_diff_prev = torch.cat([
        pad_row,
        extrema_diff_prev
    ], dim=0)
    extrema_diff_next = torch.cat([
        extrema_diff_next,
        pad_row, 
    ], dim=0)
    prominence_sort_min, prominence_sort_max = torch_nanminmax(extrema_diff_prev, extrema_diff_next)

    reverse_order = order.argsort(dim=0)
    prominence_max = torch.gather(prominence_sort_max, 0, index=reverse_order)
    prominence_max = torch.masked_fill(prominence_max, ~extrema_mask, -torch.inf)

    maxima_mask = prominence_max > prominence_threshold

    prominence_min = torch.gather(prominence_sort_min, 0, index=reverse_order)
    prominence_min = torch.masked_fill(prominence_min, ~extrema_mask, +torch.inf)

    minima_mask = (prominence_min < -prominence_threshold) 

    # Revert back to the original order
    reverse_order_repeated = order_not_repeated.argsort(dim=0)

    maxima_mask = torch.gather(maxima_mask, dim=0, index=reverse_order_repeated)
    maxima_mask[~not_repeated_mask] = False

    minima_mask = torch.gather(minima_mask, dim=0, index=reverse_order_repeated)
    minima_mask[~not_repeated_mask] = False

    return minima_mask, maxima_mask

def find_fiber_local_extrema(z: torch.Tensor):
    """
    Find local minima and maxima per fiber.

    Parameters
    ----------
    z : torch.Tensor
        The input tensor, each col is a fiber.

    Returns
    -------
    minima_row_col : tuple
        The row and column indices of the local minima.
    maxima_row_col : tuple
        The row and column indices of the local maxima.
    """
    minima_mask, maxima_mask = find_fiber_local_extrema_mask(z)
    minima = torch.nonzero(minima_mask, as_tuple=True)
    maxima = torch.nonzero(maxima_mask, as_tuple=True)
    minima_row_col = (minima[0], minima[1])
    maxima_row_col = (maxima[0], maxima[1])

    return minima_row_col, maxima_row_col


def calculate_max_fiber_valley_depth_and_wavelength_loop(z: torch.Tensor, y: torch.Tensor):
    """
    Compute the max valley depth per fiber and its corresponding wavelength.

    Parameters
    ----------
    z : torch.Tensor
        The input tensor where each column is a fiber.
    y : torch.Tensor
        The position tensor (same shape as z) representing spatial positions.

    Returns
    -------
    tuple of torch.Tensor
        - Max valley depth per fiber (1D tensor)
        - Wavelength of the valley where max depth occurs (1D tensor)
    """
    minima_row_col, maxima_row_col = find_fiber_local_extrema(z)

    min_rows, min_cols = minima_row_col
    max_rows, max_cols = maxima_row_col

    # Create tensors to store results per fiber
    max_depths = torch.zeros(z.shape[1], device=z.device)
    corresponding_wavelengths = torch.zeros(z.shape[1], device=z.device)

    # Unique fibers (columns) where extrema exist
    unique_cols = torch.unique(max_cols)

    for col in unique_cols:
        # Extract indices for this fiber
        max_idx = max_rows[max_cols == col]
        min_idx = min_rows[min_cols == col]

        if len(max_idx) < 2 or len(min_idx) == 0:
            continue  # No valid valley

        # Sort indices to ensure order
        max_idx = torch.sort(max_idx).values
        min_idx = torch.sort(min_idx).values

        # Pair consecutive maxima
        left_max = max_idx[:-1]
        right_max = max_idx[1:]

        # Find valid minima between each pair of maxima
        valid_min_mask = (min_idx[:, None] > left_max) & (min_idx[:, None] < right_max)
        valid_min_vals = torch.where(valid_min_mask, z[min_idx, col][:, None], torch.inf)
        valid_min_idx = torch.where(valid_min_mask, min_idx[:, None], -1)

        # Get minimum values and their indices for each valley region
        min_vals, min_indices = torch.min(valid_min_vals, dim=0)
        min_pos = valid_min_idx[min_indices, torch.arange(len(left_max))]

        # Ensure valid minima exist
        valid_mask = min_pos != -1
        if not torch.any(valid_mask):
            continue

        # Interpolate expected height at each minimum position
        h_left = z[left_max[valid_mask], col]
        h_right = z[right_max[valid_mask], col]
        x_left = left_max[valid_mask].float()
        x_right = right_max[valid_mask].float()
        x_min = min_pos[valid_mask].float()

        interpolated_heights = h_left + (h_right - h_left) * ((x_min - x_left) / (x_right - x_left))

        # Compute valley depths
        depths = interpolated_heights - min_vals[valid_mask]

        # Compute wavelengths (horizontal width between two maxima)
        y_left = y[left_max[valid_mask], col]
        y_right = y[right_max[valid_mask], col]
        wavelengths = y_right - y_left

        # Find the max depth and its corresponding wavelength
        max_depth_idx = torch.argmax(depths)
        max_depths[col] = depths[max_depth_idx]
        corresponding_wavelengths[col] = wavelengths[max_depth_idx]

    return max_depths, corresponding_wavelengths

def calculate_max_fiber_valley_depth_and_wavelength(z: torch.Tensor, y: torch.Tensor, prominence_threshold=0, testing=False):
    """
    Calculate the max valley depth per fiber and its corresponding wavelength.

    Parameters
    ----------
    z : torch.Tensor
        The input tensor where each column is a fiber.
    y : torch.Tensor
        The position tensor (same shape as z) representing spatial positions.

    Returns
    -------
    tuple of torch.Tensor
        - Max valley depth per fiber (1D tensor)
        - Wavelength of the valley where max depth occurs (1D tensor)
        - Indices of the max valley depth per fiber (1D tensor)
        - Indices of the previous maxima (1D tensor)
        - Indices of the next maxima (1D tensor)
    """
    minima_mask, maxima_mask = find_fiber_local_extrema_mask(z, prominence_threshold)

    y_maxima = y.masked_fill(~maxima_mask, torch.inf)
    z_maxima = z.masked_fill(~maxima_mask, torch.inf)

    z_maxima = z_maxima.T
    y_maxima = y_maxima.T

    _, y_maxima, z_maxima = surftopo.interpolation.sort_by(None, y_maxima, z_maxima, order_by=y_maxima, dim=1)
    z_inter = surftopo.interpolation.interp1d(y_maxima, z_maxima, y.T, extrapolate=False)

    z_inter = z_inter.T

    depths = z_inter - z
    depths[~minima_mask | torch.isinf(z_inter)] = -torch.inf
    max_depths = depths.max(axis=0)
    max_depth_indices = max_depths.indices

    num_rows, num_cols = maxima_mask.shape
    col_indices = torch.arange(num_rows, device=z.device).view(-1, 1).expand(-1, num_cols)
    maxima_indices = maxima_mask * col_indices

    max_depth_indices_2d = torch.atleast_2d(max_depth_indices)
    max_depth_indices_2d = max_depth_indices_2d.contiguous()

    sorted_maxima_indices, _ = torch.sort(maxima_indices.T, dim=1)
    sorted_maxima_indices = sorted_maxima_indices.contiguous()
    next_maxima_ind = torch.searchsorted(sorted_maxima_indices, max_depth_indices_2d.T)
    prev_maxima_ind = next_maxima_ind - 1

    next_maxima_ind = torch.clamp(next_maxima_ind, 0, num_rows - 1)
    prev_maxima_ind = torch.clamp(prev_maxima_ind, 0, num_rows - 1)

    next_maxima_ind_y = torch.gather(sorted_maxima_indices, 1, next_maxima_ind)
    prev_maxima_ind_y = torch.gather(sorted_maxima_indices, 1, prev_maxima_ind)

    next_maxima = torch.gather(y, 0, next_maxima_ind_y)
    prev_maxima = torch.gather(y, 0, prev_maxima_ind_y)

    corresponding_wavelengths = next_maxima - prev_maxima

    max_depth_values = max_depths.values
    max_depth_indices = max_depths.indices
    max_depth_values[torch.isinf(max_depth_values)] = 0

    if testing:
        return max_depth_values, corresponding_wavelengths, max_depth_indices, prev_maxima_ind_y, next_maxima_ind_y, z_inter
    
    return max_depth_values, corresponding_wavelengths, max_depth_indices, prev_maxima_ind_y, next_maxima_ind_y

def calculate_max_fiber_valley_depth_and_wavelength_scipy(z: torch.Tensor, y: torch.Tensor, prominence_threshold = 0):
    """
    Calculates the maximum valley depth and corresponding wavelength for each fiber, using SciPy and vectorized operations to minimize inner loops.
    
    Parameters
    ----------
    z : torch.Tensor
        The input tensor where each column is a fiber.
    y : torch.Tensor
        The position tensor (same shape as z) representing spatial positions.
    prominence_threshold : float, optional
        Prominence threshold for maximum detection (default 0).

    Returns
    -------
    tuple of torch.Tensor
        - Max valley depth per fiber (1D tensor)
        - Wavelength of the valley where max depth occurs (1D tensor)
        - Index of the valley (minimum) that produces the maximum depth (1D tensor).
    """
    z_np = z.cpu().numpy() if z.is_cuda else z.numpy()
    y_np = y.cpu().numpy() if y.is_cuda else y.numpy()
    
    _, num_fibers = z_np.shape
    max_depths = np.zeros(num_fibers)
    wavelengths = np.zeros(num_fibers)
    valley_indices = -np.ones(num_fibers, dtype=int)
    
    for col in range(num_fibers):
        z_fiber = z_np[:, col]
        y_fiber = y_np[:, col]
        
        maxima, _ = scipy.signal.find_peaks(z_fiber, prominence=prominence_threshold)
        minima, _ = scipy.signal.find_peaks(-z_fiber)
        
        if len(maxima) < 2 or minima.size == 0:
            continue
        
        I = np.sort(maxima)
        
        idx = np.searchsorted(I, minima)
        valid_mask = (idx > 0) & (idx < len(I))
        if not np.any(valid_mask):
            continue
        
        valid_minima = minima[valid_mask]
        left_max = I[idx[valid_mask] - 1]
        right_max = I[idx[valid_mask]]
        
        interp_heights = z_fiber[left_max] + (z_fiber[right_max] - z_fiber[left_max]) * (
            (valid_minima - left_max) / (right_max - left_max))
        
        depths = interp_heights - z_fiber[valid_minima]
        
        if depths.size == 0:
            continue
        
        best_idx = np.argmax(depths)
        best_depth = depths[best_idx]
        best_valley = valid_minima[best_idx]
        best_wavelength = y_fiber[right_max[best_idx]] - y_fiber[left_max[best_idx]]
        
        max_depths[col] = best_depth
        wavelengths[col] = best_wavelength
        valley_indices[col] = best_valley
    
    return (torch.tensor(max_depths, device=z.device),
            torch.tensor(wavelengths, device=z.device),
            torch.tensor(valley_indices, device=z.device))