import torch
import functools
import surftopo.detrend
import surftopo.extrema

@functools.lru_cache
def calculate_profile_widths(z: torch.Tensor, dx: float):
    """
    Calculate the lengths of all profiles.

    Parameters
    ----------
    z : torch.Tensor (MxN)
        The z-coordinates of the surface.
    dx : float
        The distance between the points in the x-axis.

    Returns
    -------
    torch.tensor
        The length of the profiles in mm, a value per profile.
    """
    z = z.T
    profile_lengths = calculate_fiber_lengths(z, dy=dx)
    return profile_lengths

def calculate_fiber_lengths(z: torch.Tensor, dy: float):
    """
    Calculate the lengths of all fibers.

    Parameters
    ----------
    z : torch.Tensor (MxN)
        The z-coordinates of the surface.
    dy : float
        The distance between the points in the y-axis.

    Returns
    -------
    torch.tensor
        The length of the fibers in i-units, a value per fiber.
    """
    dz = z[1:, :] - z[:-1, :]
    distances = torch.sqrt(dy * dy + dz * dz)
    fiber_lengths = distances.sum(dim=0)

    return fiber_lengths

def calculate_flatness(z: torch.Tensor, dy: float):
    """
    Calculate the flatness of a surface.

    Parameters
    ----------
    z : torch.Tensor (MxN)
        The z-coordinates of the surface.
    dy : float
        The distance between the points in the y-axis.

    Returns
    -------
    torch.tensor
        The flatness of the surface in i-units, a value per fiber.
    """
    fiber_lengths = calculate_fiber_lengths(z, dy)
    reference_length = fiber_lengths.min()

    i_units = 1e5 * (fiber_lengths - reference_length) / reference_length    

    return i_units

def calculate_profile_tilt(z: torch.Tensor, dx: float):
    """
    Calculate the tilt angle of each profile in degrees based on the left and right edge points.

    Parameters
    ----------
    z : torch.Tensor (MxN)
        The z-coordinates of the surface.
    dx : float
        The distance between the points in the x-axis.

    Returns
    -------
    torch.Tensor
        The tilt angle in degrees for each profile.
    """

    left_edges = z[:, 0]
    right_edges = z[:, -1]
    
    delta_z = right_edges - left_edges
    
    n_points = z.size(1)
    horizontal_distance = (n_points - 1) * dx
    
    tilt_radians = torch.atan(delta_z / horizontal_distance)
    tilt_degrees = torch.rad2deg(tilt_radians)
    
    return tilt_degrees

def calculate_mean_axis(matrix: torch.Tensor, dim=1):
    """
    Calculate the mean of a matrix along a given axis ignoring infs.

    Parameters
    ----------
    matrix : torch.Tensor
        The matrix to calculate the mean.
    dim : int
        The axis to calculate the mean.

    Returns
    -------
    torch.tensor
        The mean of the matrix along the given axis.
    """
    matrix_nan = torch.masked_fill(matrix, torch.isinf(matrix), torch.nan)
    mean_axis = torch.nanmean(matrix_nan, dim=dim)
    return mean_axis    

@functools.lru_cache
def calculate_profile_min_max(x: torch.Tensor):
    """
    Calculate the min and max position of every profile ignoring infs.

    Parameters
    ----------
    x : torch.Tensor
        The x coordinates of the profiles.

    Returns
    -------
    torch.tensor
        The min and max position of the profiles.
    """
    min = torch.min(torch.masked_fill(x, x == -torch.inf, torch.inf), dim=1).values
    max = torch.max(torch.masked_fill(x, x == +torch.inf, -torch.inf), dim=1).values

    return min, max

def calculate_profile_widths_x(x: torch.Tensor):
    """
    Calculate the width of every profile ignoring infs.

    Parameters
    ----------
    x : torch.Tensor
        The x coordinates of the profiles.

    Returns
    -------
    torch.tensor
        The width of the profiles.
    """
    min, max = calculate_profile_min_max(x)
    width = max - min

    return width

def calculate_profile_center_deviations_deprecated(x: torch.Tensor):
    """
    Calculate the center deviation of every profile ignoring infs.
    
    Parameters
    ----------
    x : torch.Tensor
        The x coordinates of the profiles.

    Returns
    -------
    torch.tensor
        The center deviation of the profiles.
    """
    min, max = calculate_profile_min_max(x)

    # Center for each row
    row_centers = (max + min) / 2.0

    # Mean center across all rows
    mean_center = torch.mean(row_centers)

    center_deviation = row_centers - mean_center

    return center_deviation

def calculate_profile_center_deviations(x: torch.Tensor):
    """
    Calculate the center for each row of every profile ignoring infs.
    
    Parameters
    ----------
    x : torch.Tensor
        The x coordinates of the profiles.

    Returns
    -------
    torch.tensor
        The center of each row of the profiles.
    """

    min, max = calculate_profile_min_max(x)
    center_line = min + (max - min) / 2


    return center_line

def calculate_fiber_amplitudes(z: torch.Tensor):
    """
    Compute surface amplitude parameters for surface fibers according to the ISO standard.

    This function calculates various amplitude parameters based on the z-coordinates of a surface,
    where each column of the input tensor represents a fiber.

    Parameters
    ----------
    z : torch.Tensor
        A 2D tensor of shape (M, N), where:
            - M: Number of points along each fiber (rows).
            - N: Number of fibers (columns).
        Each element represents the z-coordinate (height) of the surface.

    Returns
    -------
    Returns a dictionary of the height parameters.
        Ra : torch.Tensor
            A 1D tensor of length N containing the arithmetic average of the absolute values of heights 
            for each fiber (column).
        Rq : torch.Tensor
            A 1D tensor of length N containing the root mean square (RMS) of the heights for each fiber.
        Rp : torch.Tensor
            A 1D tensor of length N containing the maximum peak height for each fiber.
        Rv : torch.Tensor
            A 1D tensor of length N containing the absolute value of the maximum valley depth for each fiber.
        Rt : torch.Tensor
            A 1D tensor of length N containing the total peak-to-valley height (Rp + Rv) for each fiber.

    Notes
    -----
    - Ra, Rq, Rp, Rv, and Rt are commonly used roughness parameters in surface metrology.
    - These calculations are performed for each column (fiber) independently.
    """
    n = z.shape[0]
    r = surftopo.detrend.remove_fiber_mean(z)

    # Compute Ra (Arithmetic average of absolute values)
    Ra = torch.abs(r).sum(dim=0) / n

    # Compute Rq (Root Mean Square)
    Rq = torch.sqrt((r ** 2).sum(dim=0) / n)

    # Compute Rp (Maximum value of r)
    Rp = torch.max(r, dim=0).values

    # Compute Rv (Absolute value of the minimum of r)
    Rv = torch.abs(torch.min(r, dim=0).values)

    # Compute Rt (Total peak-to-valley height)
    Rt = Rp + Rv

    results = {'Ra': Ra, 'Rq': Rq, 'Rp': Rp, 'Rv': Rv, 'Rt': Rt}

    # Prepend 'fiber_iso_' to each key
    results = {f"fiber_iso_{key}": value for key, value in results.items()}
    return results

def calculate_surface_amplitudes(z: torch.Tensor):
    """
    Calculates the amplitude parameters of the surface according to ISO-25178
    Returns a dictionary of the height parameters.

    Parameters
    ----------
    data : torch.Tensor
        A 2D tensor representing surface heights.

    Returns
    -------
    - Sa: Arithmetic mean of absolute deviations.
    - Sq: Root mean square roughness.
    - Sv: Maximum valley depth (absolute minimum value).
    - Sp: Maximum peak height.
    - Sz: Total height (Sp + Sv).
    - Ssk: Skewness of the height distribution.
    - Sku: Kurtosis of the height distribution.
    """
    mean = torch.mean(z)
    centered_data = z - mean
    abs_centered_data = torch.abs(centered_data)
    centered_data_sq = centered_data ** 2

    # Total number of elements in the tensor
    numel = z.numel()  

    # Arithmetic mean of absolute deviations
    sa = torch.sum(abs_centered_data) / numel  

     # Root mean square roughness
    sq = torch.sqrt(torch.sum(centered_data_sq) / numel) 

    # Maximum valley depth (absolute minimum)
    sv = torch.abs(torch.min(centered_data))  

    # Maximum peak height
    sp = torch.max(centered_data)  

    # Total height (peak-to-valley height)
    sz = sp + sv  

    # Skewness
    ssk = torch.sum(centered_data_sq * centered_data) / numel / (sq ** 3)  

    # Kurtosis
    sku = torch.sum(centered_data_sq ** 2) / numel / (sq ** 4)  

    results = {'Sa': sa.item(), 'Sq': sq.item(), 'Sv': sv.item(), 
            'Sp': sp.item(), 'Sz': sz.item(), 'Ssk': ssk.item(), 'Sku': sku.item()}

    # Prepend 'surface_iso_' to each key
    results = {f"surface_iso_{key}": value for key, value in results.items()}
    return results

def calculate_profile_offset(z: torch.Tensor, Zr: float):
    """
    Calculate the Offset for each profile, which is the difference between the mean height (Zm) 
    of the profile (ignoring infs) and the reference height Zr.

    Parameters
    ----------
    z : torch.Tensor (MxN)
        The z-coordinates of the surface, each row is a profile.
    Zr : float
        Reference height for all profiles.

    Returns
    -------
    torch.Tensor
        Offset (Zm - Zr) for each profile.
    """
    Zm = calculate_mean_axis(z, dim=1)
    offset = Zm - Zr
    return offset


def interpolate_profile_feature(Y_valid: torch.Tensor, axis_profile: torch.Tensor, *args):
    """
    Interpolate the profile features along the profile axis.
    The profiles are sorted by the mean of the Y_valid tensor.

    Parameters
    ----------
    Y_valid : torch.Tensor
        The y-coordinates of the profiles.
    axis_profile : torch.Tensor
        The axis to interpolate the features.
    args : tuple
        The features to interpolate.

    Returns
    -------
    tuple
        The interpolated features.
    """
    y_valid_mean = calculate_mean_axis(Y_valid, dim=1)
    order = torch.argsort(y_valid_mean)
    y_valid_mean_sorted = y_valid_mean[order]

    results = []
    for arg in args:
        arg_sorted = arg[order]
        result = surftopo.interpolation.interp1d(y_valid_mean_sorted, arg_sorted, axis_profile, extrapolate=True)
        result = result.squeeze()
        results.append(result)

    if len(args) == 1:
        return results[0]
    else:
        return tuple(results)

def calculate_polynomial_fit(degree: int, sub_sampling_step, resampled_x: torch.Tensor, Z_uni_fil: torch.Tensor, 
                             polynomial_device, polynomial):
    """
    Calculate the polynomial fit coefficients for the surface profiles.

    Parameters
    ----------
    degree : int
        The degree of the polynomial fit.
    sub_sampling_step : int
        The sub-sampling step for the polynomial fit.
    resampled_x : torch.Tensor
        The resampled x-coordinates.
    Z_uni_fil : torch.Tensor
        The resampled z-coordinates.
    polynomial_device : str
        The device to use for the polynomial fit.
    polynomial : str
        The polynomial to use for the fit.

    Returns
    -------
    dict
        A dictionary containing the polynomial fit coefficients.
    """
    x = resampled_x[::sub_sampling_step]
    z = Z_uni_fil[:, ::sub_sampling_step]
    coefficients = surftopo.polynomial.calculate_polynomial_fit(degree, x, z, 
                                                                polynomial_device, polynomial)
    results = {}
    for i in range(coefficients.shape[1]):
        results[f"profile_fit_coefficient_{i}"] = coefficients[:, i].squeeze()
    return results


def calculate_amplitude_wavelength(z: torch.Tensor, y: torch.Tensor, prominence_threshold):
    """
    Calculate the amplitude and wavelength of the surface fibers.

    Parameters
    ----------
    z : torch.Tensor
        The input tensor where each column is a fiber.
    y : torch.Tensor
        The position tensor (same shape as z) representing spatial positions.

    Returns
    -------
    dict
        A dictionary containing the following keys:
        - fiber_amplitudes: The amplitude of each fiber (max valley depths).
        - fiber_wavelengths: The wavelength of each fiber (corresponding wavelengths).
    """
    max_depths, wavelengths, _, _, _ = surftopo.extrema.calculate_max_fiber_valley_depth_and_wavelength(z, y, prominence_threshold)

    results = {}
    results["fiber_amplitudes"] = max_depths.squeeze()
    results["fiber_wavelengths"] = wavelengths.squeeze()
    return results
