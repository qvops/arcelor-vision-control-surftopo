
class Options:
    """
    Options for the analysis of surface topography.
    """
    def __init__(self):
        """
        Initialize the default options.
        """
        ###########################################################################################
        # Options for the analysis

        # The device to use for the analysis: cuda, cpu
        self.device = "cpu"

        # Invalid value for the input data. It is equivalent to NaN or no data.
        self.invalid_value = 0

        # The step for sub-sampling the input data.
        # A value of 1 disables sub-sampling.
        self.sub_sampling_step = 1

        # The threshold for the edge detection algorithm.
        self.edge_detection_threshold_mm = 100

        # Maximum distance for a data point to be classified as an inlier during line detection.
        self.line_detection_residual_mm = 5

        # Cutoff distance for the low-pass filter in the x-direction. 
        # Frequencies with wavelengths shorter than this value will be removed.
        self.filter_low_pass_cutoff_x_mm = 100

        # Cutoff distance for the low-pass filter in the y-direction.
        # Frequencies with wavelengths shorter than this value will be removed.
        self.filter_low_pass_cutoff_y_mm = 100

        # The resolution of the resampled surface in the x and y direction in mm.
        self.resampling_distance_mm = 1

        # The scale factor for the outlier detection algorithm.
        # Outliers are detected based on the Z-score method using this scale factor. 
        # A value of 0 disables the filter.
        self.outlier_detection_scale_factor = 3

        # The neighborhood size for the Burr filter (removes values next to invalid values).
        # A value of 0 disables the filter.
        self.fill_burr_neighborhood = 0

        # The polynomial to use for the profile fitting: Legendre, Chebyshev, Standard, Hermite
        self.polynomial = "Legendre"

        # Degree of the polynomial used for profile fitting.
        # For a polynomial of the form a0 + a1*x + a2*x^2 + ... + an*x^n,
        # the degree is n. The coefficients are returned in the order [a0, a1, ..., an], with length n+1.
        self.polynomial_degree = 5

        # The device to use for the polynomial fitting: cuda, cpu. 
        # In some scenarios the polynomial fitting can be faster on the cpu.
        self.polynomial_device = "cuda"

        # The step for sub-sampling the profiles before polynomial fitting.
        self.polynomial_sub_sampling_step = 4

        # Margins for reducing the analysis area and controlling extrapolation beyond detected surface boundaries.
        self.bbox_inner_margin_left_mm = 0
        self.bbox_inner_margin_right_mm = 0
        self.bbox_inner_margin_top_mm = 0
        self.bbox_inner_margin_bottom_mm = 0

        # Margin removal from the top and bottom of the analysis surface.
        # This parameter is used to exclude potential outlier regions near the edges, ensuring more reliable analysis.
        self.top_bottom_margin_removal_mm = 10

        # The method used for detrend: 
        # Surface_plane: removes the plane from the data
        # Fiber_linear_regression: removes the linear regression from each fiber independently
        self.detrend = "Fiber_linear_regression"

        # Reference height offset for all profiles.
        self.profile_offset_zr = 1642.1

        # Minimum prominence value for wavelength and depth analysis
        self.prominence_threshold = 0.5

        ###########################################################################################
        # Options for the validation of intermediate results
        # An exception will be raised if the results do not meet the following criteria.

        # Minimum acceptable parallelism threshold for the left and right lines: [0, 1]
        # Lines with parallelism below this value are considered invalid.
        self.min_valid_parallelism = 0.95

        # Minimum angle deviation with respect to the vertical axis
        # Lines with more deviation than this value are considered invalid.
        self.min_valid_theta_deg = 5

        # Minimum width of the surface bounding box in mm.
        self.min_width_mm = 400

        # Minimum height of the surface bounding box in mm.
        self.min_height_mm = 400

        # Minimum number of valid rows before resampling.
        self.min_valid_rows = 100

        # Minimum number of valid columns before resampling.
        self.min_valid_cols = 100

        ###########################################################################################
        # Options for the results

        # Return the analysis results as numpy arrays if True. Otherwise, the results are returned as torch tensors.
        # Note: returning results as numpy arrays will increase execution time due to the overhead of transferring data from GPU to CPU
        self.return_as_numpy = False

        # Return the x, y, z coordinates of the resampled surface filtered.
        self.return_xyz = True

        # Return the execution time for each analysis step in milliseconds.
        # Each time measurement is prefixed with "time_".
        self.return_time_measurements = True

        # Return the amplitude parameters for the fibers according to the ISO standard (Ra, Rq, Rp, Rv, Rt).
        self.return_fiber_iso_amplitudes = False

        # Return the amplitude parameters for the surface according to the ISO standard (Sa, Sq, Sv, Sp, Sz, Ssk, Sku).
        self.return_surface_iso_amplitudes = False

