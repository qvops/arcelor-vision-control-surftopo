import torch
import torch.nn.functional as F
import functools

def calculate_open_profile_gaussian_filter_weight(x_values: torch.Tensor, lc: float):
    """
    Calculate Gaussian filter weights for an open profile.
    ISO 16610-21 Open profile Gaussian filter weight function value
    
    Parameters:
        x_values (array): Input x-ordinates (range values), distance from center of weight function
        lc (float): Cutoff wavelength.
        
    Returns:
        weight (array): Gaussian weights.
    """
    alc = 0.469718639349826 * lc  # ALFA * LC
    weight = torch.exp(-torch.pi * (x_values / alc) ** 2) / alc
    return weight

def create_open_profile_gaussian_kernel_1d(distance_between_samples_mm: float, cutoff_min_distance_mm: float,
                                    device: str, dtype: torch.dtype):
    """
    Create a 1D Gaussian kernel for a given size and standard deviation.

    Parameters:
        distance_between_samples_mm (float): Distance between samples in mm.
        cutoff_min_distance_mm (float): Minimum cutoff distance in mm.
        device (str): Device to use.
        dtype (torch.dtype): Data type to use.

    Returns:
        kernel (torch.Tensor): Gaussian kernel.
    """
    dx = distance_between_samples_mm   # Distance between scans in mm
    lcut = cutoff_min_distance_mm    # Minimum cutoff distance in mm

    ni_DIV_Ln = 1 / dx            # Inverse of scan spacing

    Lc = 0.5                      # Default truncating constant by ISO 16610-21
    lcuttr = Lc * lcut            # Truncated cutoff wavelength

    # Cutoff wavelength in point intervals
    nicut = ni_DIV_Ln * lcut
    nicuttr = ni_DIV_Ln * lcuttr  # Truncated cutoff wavelength in intervals

    # Truncated Gaussian range limits
    gaussxlim = int(nicuttr)      # Convert to integer

    # Generate symmetric weight function x-ordinates
    gaussx = torch.arange(-gaussxlim, gaussxlim + 1, 1.0, device=device, dtype=dtype)

    gaussy = calculate_open_profile_gaussian_filter_weight(gaussx, nicut)

    gaussy = gaussy / torch.sum(gaussy)
    return gaussy

@functools.cache    
def create_gaussian_kernel_2d(step_x: float, step_y: float, cutoff_x: float, cutoff_y: float, device: str, dtype: torch.dtype):
    """
    Create a 2D Gaussian kernel for a given size and standard deviation.

    Parameters:
        step_x (float): Step size in the x-direction.
        step_y (float): Step size in the y-direction.
        cutoff_x (float): Cutoff distance in the x-direction.
        cutoff_y (float): Cutoff distance in the y-direction.
        device (str): Device to use.
        dtype (torch.dtype): Data type to use.

    Returns:
        kernel_x (torch.Tensor): Gaussian kernel in the x-direction.
        kernel_y (torch.Tensor): Gaussian kernel in the y-direction.
    """
    kernel_x = create_open_profile_gaussian_kernel_1d(step_x, cutoff_x, device, dtype).view(1, 1, 1, -1)  # Shape: (1, 1, 1, size_x)
    kernel_y = create_open_profile_gaussian_kernel_1d(step_y, cutoff_y, device, dtype).view(1, 1, -1, 1)  # Shape: (1, 1, size_y, 1)

    return kernel_x, kernel_y

def gaussian_filter(surface: torch.Tensor, kernel_x: torch.Tensor, kernel_y: torch.Tensor):
    """
    Apply a 2D Gaussian filter to a surface (separable kernel).
    In the y-direction, a permutation of the kernel is used to apply the filter.

    Parameters:
        surface (torch.Tensor): The input surface.
        kernel_x (torch.Tensor): Gaussian kernel in the x-direction.
        kernel_y (torch.Tensor): Gaussian kernel in the y-direction.

    Returns:
        result (torch.Tensor): The filtered surface.
    """
    size_x = kernel_x.numel()
    size_y = kernel_y.numel()

    # Add batch and channel dimensions to the tensor
    surface = surface.unsqueeze(0).unsqueeze(0)

    # Apply separable convolutions for efficiency
    padded_tensor_x = F.pad(surface, (size_x // 2, size_x // 2, 0, 0), mode='replicate')
    result = F.conv2d(padded_tensor_x, kernel_x, padding=0, groups=1)

    # Second convolution with replicate padding and permutation
    padded_result_y = result.permute(0, 1, 3, 2)
    padded_result_y = F.pad(padded_result_y, (size_y // 2, size_y // 2, 0, 0), mode='replicate')
    kernel_y = kernel_y.view(1, 1, 1, -1)
    result = F.conv2d(padded_result_y, kernel_y, padding=0, groups=1)
    result = result.permute(0, 1, 3, 2)

    return result.squeeze()

def gaussian_filter_no_permute(surface: torch.Tensor, kernel_x: torch.Tensor, kernel_y: torch.Tensor):
    """
    Apply a 2D Gaussian filter to a surface (separable kernel).
    This version does not use permute the convolution on the y-axis.
    It seems to be slower than the version with permute due to memory access

    Parameters:
        surface (torch.Tensor): The input surface.
        kernel_x (torch.Tensor): Gaussian kernel in the x-direction.
        kernel_y (torch.Tensor): Gaussian kernel in the y-direction.

    Returns:
        result (torch.Tensor): The filtered surface.
    """
    size_x = kernel_x.numel()
    size_y = kernel_y.numel()

    # Add batch and channel dimensions to the tensor
    surface = surface.unsqueeze(0).unsqueeze(0)

    # Apply separable convolutions for efficiency
    padded_tensor_x = F.pad(surface, (size_x // 2, size_x // 2, 0, 0), mode='replicate') 
    result = F.conv2d(padded_tensor_x, kernel_x, padding=0, groups=1)

    # Second convolution with replicate padding
    padded_result_y = F.pad(result, (0, 0, size_y // 2, size_y // 2), mode='replicate')
    result = F.conv2d(padded_result_y, kernel_y, padding=0, groups=1)

    return result.squeeze()