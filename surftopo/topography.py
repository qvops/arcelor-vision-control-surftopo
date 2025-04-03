import torch
import numpy as np
from typing import Dict

from surftopo.profiler import TimeProfiler
from surftopo.options import Options
import surftopo.validation
import surftopo.segmentation
import surftopo.geometry
import surftopo.transformation
import surftopo.interpolation
import surftopo.filter
import surftopo.feature
import surftopo.polynomial
import surftopo.detrend

def analyze_surface_topography(x: np.ndarray, y: np.ndarray, z: np.ndarray, options: Options) -> Dict[str, any]:
    """
    Analyze the topography of a surface.

    Parameters
    ----------
    x : numpy.ndarray (MxN)
        The x-coordinates of the surface. Represents the width of the steel strip.
        Reference axis: the center of the roller path. 
        Negative coordinates indicate the operator side, while positive coordinates indicate the motor side.
    y : numpy.ndarray (MxN)
        The y-coordinates of the surface. Represents the length of the steel strip.
        Coordinate 0 refers to the head of the strip. Only positive coordinates are possible.
    z : numpy.ndarray (MxN)
        The z-coordinates of the surface.
        The distance from the sensor to the strip.
    Each matrix has dimensions MxN, where:
        Rows (M) represent the profiles.
        Columns (N) represent the fibers.

    options : Options
        Options for the analysis.

    Returns
    -------
    dict
        The analysis results.
    """
    surftopo.validation.validate_inputs(x, y, z)

    # List to store the execution time for each analysis step.
    profilers = []

    if options.sub_sampling_step > 1:
        step = options.sub_sampling_step
        x, y, z = x[::step, ::step], y[::step, ::step], z[::step, ::step]
    
    with TimeProfiler("gpu_transfer", registry=profilers):
        X = torch.from_numpy(x).to(options.device, dtype=options.dtype)
        Y = torch.from_numpy(y).to(options.device, dtype=options.dtype)
        Z = torch.from_numpy(z).to(options.device, dtype=options.dtype)

    with TimeProfiler("edge_detection", registry=profilers):
        threshold = options.edge_detection_threshold_mm
        left_rc, right_rc = surftopo.segmentation.detect_vertical_edge_candidates(Z, threshold=threshold)
        x_left, y_left = X[left_rc], Y[left_rc]
        x_right, y_right = X[right_rc], Y[right_rc]
        surftopo.validation.validate_left_right(x_left, x_right)

    with TimeProfiler("contour", registry=profilers):
        residual_threshold = options.line_detection_residual_mm
        _, left_direction, left_inliers = surftopo.segmentation.find_line_contour(x_left, y_left, 
                                                                                  residual_threshold=residual_threshold)
        _, right_direction, right_inliers = surftopo.segmentation.find_line_contour(x_right, y_right, 
                                                                                    residual_threshold=residual_threshold)

    with TimeProfiler("cropping", registry=profilers):
        left_rc_inliers, right_rc_inliers = left_rc[left_inliers], right_rc[right_inliers]
        x_left_inliers, y_left_inliers = X[left_rc_inliers], Y[left_rc_inliers]
        x_right_inliers, y_right_inliers = X[right_rc_inliers], Y[right_rc_inliers]

        min_col = left_rc_inliers.col.min()
        max_col = right_rc_inliers.col.max()
        min_row = torch.min(left_rc_inliers.row.min(), right_rc_inliers.row.min())
        max_row = torch.max(left_rc_inliers.row.max(), right_rc_inliers.row.max()) 

        X_cropped, Y_cropped, Z_cropped = surftopo.segmentation.crop(X, Y, Z, min_col, max_col, min_row, max_row)

        left_rc_inliers_cropped = left_rc_inliers.add(-min_row, -min_col)
        right_rc_inliers_cropped = right_rc_inliers.add(-min_row, -min_col)

    with TimeProfiler("background", registry=profilers):
        invalid_value = options.invalid_value
        Z_cropped = surftopo.segmentation.fill_outside_boundary(Z_cropped, 
                                        left_rc_inliers_cropped.row, left_rc_inliers_cropped.col,
                                        right_rc_inliers_cropped.row, right_rc_inliers_cropped.col,
                                        fill_value=invalid_value)        

    if options.outlier_detection_scale_factor > 0:
        with TimeProfiler("outliers", registry=profilers):
            scale_factor = options.outlier_detection_scale_factor
            Z_cropped = surftopo.segmentation.fill_outliers_zscore(Z_cropped, scale_factor=scale_factor,
                                                                invalid_value=invalid_value, fill_value=invalid_value)
            if options.fill_burr_neighborhood > 0:
                n = options.fill_burr_neighborhood
                Z_cropped = surftopo.segmentation.fill_burr(Z_cropped, n=n)    

            invalid_mask = (Z_cropped == invalid_value)
            X_cropped, Y_cropped = surftopo.segmentation.fill_invalid_values(X_cropped, Y_cropped, 
                                                                            invalid_mask, invalid_value)

    with TimeProfiler("rotation", registry=profilers):
        theta, parallelism = surftopo.geometry.calculate_vertical_angle_deviation(left_direction, right_direction)
        X_rotated, Y_rotated = surftopo.transformation.rotate_2d(X_cropped, Y_cropped, theta)
        Z_rotated = Z_cropped
        surftopo.validation.validate_lines(theta, parallelism, options)

    with TimeProfiler("bbox_creation", registry=profilers):
        x_left_boundary, y_left_boundary = surftopo.transformation.rotate_2d(x_left_inliers, y_left_inliers, theta)
        x_right_boundary, y_right_boundary = surftopo.transformation.rotate_2d(x_right_inliers, y_right_inliers, theta)
        bbox_left = surftopo.segmentation.tolerance_ceil((x_left_boundary).mean())
        bbox_right = x_right_boundary.mean().floor()
        bbox_top = surftopo.segmentation.tolerance_ceil(torch.min(y_left_boundary.max(), y_right_boundary.max()))
        bbox_bottom = torch.max(y_left_boundary.min(), y_right_boundary.min()).floor()

        bbox = surftopo.segmentation.reduced_bbox(bbox_left, bbox_top, bbox_right, bbox_bottom, options.bbox_inner_margin_left_mm,
                                                  options.bbox_inner_margin_top_mm, options.bbox_inner_margin_right_mm, options.bbox_inner_margin_bottom_mm)
        surftopo.validation.validate_bounding_box(bbox, options)

    with TimeProfiler("trim_edges", registry=profilers):
        residual_threshold = options.line_detection_residual_mm
        X_valid, Y_valid, Z_valid = surftopo.segmentation.remove_outside_bbox(X_rotated, Y_rotated, Z_rotated, 
                                                                            bbox, residual_threshold=residual_threshold,
                                                                            invalid_value=invalid_value)
        surftopo.validation.validate_dimensions(Z_valid, options)

    with TimeProfiler("interpolation", registry=profilers):
        resolution_mm = options.resampling_distance_mm
        X_uni, Y_uni, resampled_x, resampled_y = surftopo.segmentation.create_uniform_grid(X, Y, bbox_left, bbox_right, bbox_bottom, bbox_top, resolution_mm)

        Z_uni = surftopo.interpolation.interp2d(X_valid, Y_valid, Z_valid, resampled_x, resampled_y)
        X_uni, Y_uni, Z_uni, _ = surftopo.segmentation.remove_top_bottom_margin(X_uni, Y_uni, Z_uni, options.top_bottom_margin_removal_mm)
        axis_fiber, axis_profile = X_uni[0, :].squeeze(), Y_uni[:, 0].squeeze()
        surftopo.validation.validate_values(Z_uni)
    
    with TimeProfiler("gaussian_filter", registry=profilers):
        step_y = step_x = resolution_mm
        cutoff_x, cutoff_y = options.filter_low_pass_cutoff_x_mm, options.filter_low_pass_cutoff_y_mm
        # The kernel is created once and cached for all surfaces with the same resolution and cutoff.
        kernel_x, kernel_y = surftopo.filter.create_gaussian_kernel_2d(step_x, step_y, cutoff_x, cutoff_y, 
                                                                device=Z_uni.device, dtype=Z_uni.dtype)

        Z_uni_fil = surftopo.filter.gaussian_filter(Z_uni, kernel_x, kernel_y)
        surftopo.validation.validate_shapes(X_uni, Y_uni, Z_uni_fil)

    with TimeProfiler("features", registry=profilers):
        profile_widths = surftopo.feature.calculate_profile_widths(Z_uni_fil, dx=resolution_mm)
        profile_widths_x = surftopo.feature.calculate_profile_widths_x(X_valid)
        profile_center_deviations = surftopo.feature.calculate_profile_center_deviations(X_valid)
        profile_tilt = surftopo.feature.calculate_profile_tilt(Z_uni_fil, dx=resolution_mm)
        profile_offset = surftopo.feature.calculate_profile_offset(Z_uni_fil, Zr=options.profile_offset_zr_mm)
        fiber_flatness_i_units = surftopo.feature.calculate_flatness(Z_uni_fil, dy=resolution_mm)
        profile_widths_x, profile_center_deviations = surftopo.feature.interpolate_profile_feature(Y_valid, axis_profile, 
                                                                                                   profile_widths_x, profile_center_deviations)
        result = {
            "profile_widths": profile_widths,
            "profile_widths_x": profile_widths_x,
            "profile_center_deviations": profile_center_deviations,
            "profile_tilt": profile_tilt,
            "profile_offset": profile_offset,
            "fiber_flatness_i_units": fiber_flatness_i_units,
            "axis_fiber": axis_fiber, "axis_profile": axis_profile
            }
    
    with TimeProfiler("polynomial_fit", registry=profilers):
        degree = options.polynomial_degree
        polynomial = surftopo.polynomial.polynomials[options.polynomial]
        polynomial_device = torch.device(options.polynomial_device)
        sub_sampling_step = options.polynomial_sub_sampling_step
        profile_fit_coefficients = surftopo.feature.calculate_polynomial_fit(degree, sub_sampling_step, 
                                                                 resampled_x, Z_uni_fil, 
                                                                 polynomial_device, polynomial)
        result.update(profile_fit_coefficients)

    with TimeProfiler("detrend", registry=profilers):
        detrend_method = surftopo.detrend.detrends[options.detrend]
        Z_uni_fil_detrend = detrend_method(X_uni, Y_uni, Z_uni_fil)

    with TimeProfiler("amplitudes", registry=profilers):
        prominence_threshold_mm = options.prominence_threshold_mm
        amplitudes_wavelengths = surftopo.feature.calculate_amplitude_wavelength(Z_uni_fil_detrend, Y_uni, prominence_threshold_mm)
        result.update(amplitudes_wavelengths)

    if options.return_fiber_iso_amplitudes:
        with TimeProfiler("fiber_amplitudes_iso", registry=profilers):
            fiber_amplitudes = surftopo.feature.calculate_fiber_amplitudes(Z_uni_fil_detrend)
            result.update(fiber_amplitudes)

    if options.return_surface_iso_amplitudes:
        with TimeProfiler("surface_amplitudes_iso", registry=profilers):
            surface_amplitudes = surftopo.feature.calculate_surface_amplitudes(Z_uni_fil_detrend)
            result.update(surface_amplitudes)

    if options.return_xyz:
        result.update({"x": X_uni, "y": Y_uni, "z": Z_uni_fil, "z_detrend": Z_uni_fil_detrend})

    if options.return_as_numpy:
        with TimeProfiler("cpu_transfer", registry=profilers):
            for k, v in result.items():
                if isinstance(v, torch.Tensor):
                    result[k] = v.cpu().numpy()

    if options.return_time_measurements:
        for profiler in profilers:
            result["time_" + profiler.name] = profiler.elapsed_ms()

    return result