from scipy.io import loadmat
import torch
import re
import open3d as o3d
import numpy as np
import gzip
import struct
import os
from scipy.interpolate import interp1d

def load_mat(file_path: str, device="cuda"):
    """
    Load a MATLAB surface file into PyTorch tensors.

    Parameters
    ----------
    file_path : str
        The path to the MATLAB file.
    device : str
        The device to use for the tensors.

    Returns
    -------
    X : torch.Tensor
        The x-coordinates of the surface.
    Y : torch.Tensor
        The y-coordinates of the surface.
    Z : torch.Tensor
        The z-coordinates of surface
    """
    mat_data = loadmat(file_path)

    X = mat_data.get('X')
    Y = mat_data.get('Y')
    Z = mat_data.get('Z')

    X = torch.from_numpy(X).to(device)
    Y = torch.from_numpy(Y).to(device)
    Z = torch.from_numpy(Z).to(device)
    return X, Y, Z

def get_photoneo_dimensions(file_path: str):
    """
    Get the dimensions of a Photoneo PLY file.

    Parameters
    ----------
    file_path : str
        The path to the PLY file.
    
    Returns
    -------
    width : int
        The width of the point cloud.
    height : int
        The height of the point cloud.
    """
    # Read the header
    header = []
    with open(file_path, 'r', encoding="latin1") as f:
        for line in f:
            header.append(line.strip())
            if line.startswith("end_header"):
                break

    # Search for the Photoneo metadata line
    for line in header:
        if "Photoneo PLY PointCloud" in line:
            # Use regex to find Width and Height values
            width_match = re.search(r"Width = (\d+);", line)
            height_match = re.search(r"Height = (\d+)Ordered", line)
            
            if width_match and height_match:
                width = int(width_match.group(1))
                height = int(height_match.group(1))
                return width, height

    raise Exception("Photoneo metadata not found or improperly formatted.")

def load_ply_photoneo(file_path: str, device="cuda"):
    """
    Load a Photoneo PLY surface file into PyTorch tensors. 
    It assumes the file is a Photoneo PLY file with header metadata "Photoneo PLY PointCloud".

    Parameters
    ----------
    file_path : str
        The path to the PLY file.
    device : str
        The device to use for the tensors.

    Returns
    -------
    X : torch.Tensor
        The x-coordinates of the surface.
    Y : torch.Tensor
        The y-coordinates of the surface.
    Z : torch.Tensor
        The z-coordinates of the surface
    """
    width, height = get_photoneo_dimensions(file_path)
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)

    reshaped_points = points.reshape((height, width, 3))
    Y_np = reshaped_points[:, :, 0].T
    X_np = reshaped_points[:, :, 1].T
    Z_np = reshaped_points[:, :, 2].T

    X = torch.from_numpy(X_np).to(device).float()
    Y = torch.from_numpy(Y_np).to(device).float()
    Z = torch.from_numpy(Z_np).to(device).float()
    return X, Y, Z

def load_npy(data_path: str, base_filename: str, width: int, height: int, device="cuda"):
    """
    Load .npy files into PyTorch tensors.

    Parameters
    ----------
    data_path : str
        The directory containing the .npy files.
    base_filename : str
        The base filename of the .npy files.
    width : int
        The width to reshape the numpy arrays.
    height : int
        The height to reshape the numpy arrays.
    device : str
        The device to use for the tensors.

    Returns
    -------
    X : torch.Tensor
        The x-coordinates of the surface.
    Y : torch.Tensor
        The y-coordinates of the surface.
    Z : torch.Tensor
        The z-coordinates of the surface.
    """
    file_path_x = os.path.join(data_path, f'{base_filename}_x.npy')
    file_path_y = os.path.join(data_path, f'{base_filename}_y.npy')
    file_path_z = os.path.join(data_path, f'{base_filename}_z.npy')
    
    X_np = np.load(file_path_x)
    Y_np = np.load(file_path_y)
    Z_np = np.load(file_path_z)
    
    # Y_np = X_np.reshape((height, width))
    # X_np = Y_np.reshape((height, width))
    # Z_np = Z_np.reshape((height, width))

    reshaped_points = np.stack((X_np, Y_np, Z_np), axis=-1).reshape((height, width, 3))

    Y_np = reshaped_points[:, :, 0].T
    X_np = reshaped_points[:, :, 1].T
    Z_np = reshaped_points[:, :, 2].T
    
    X = torch.from_numpy(X_np).to(device).float()
    Y = torch.from_numpy(Y_np).to(device).float()
    Z = torch.from_numpy(Z_np).to(device).float()

    return X, Y, Z


def save_bin_gz(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, file_path:str = "HeightMap.bin.gz"):
    """
    Save a height map to a binary gzip file. The height map is uniformly sampled in the x and y axes.

    Parameters
    ----------
    x : torch.Tensor
        The x-coordinates of the height map.
    y : torch.Tensor
        The y-coordinates of the height map.
    z : torch.Tensor
        The z-coordinates of the heigh map.
    file_path : str
        The path to the file to save the height map.
    Each matrix has dimensions MxN, where:
        Rows (M) represent the profiles.
        Columns (N) represent the fibers.        

    Returns
    -------
    None
    """
    BIN_FILE_SIGNATURE = "HeightMapBinGzV1"

    SIGNATURE = BIN_FILE_SIGNATURE.encode("ascii")
    with gzip.open(file_path, "wb") as gf:
        gf.write(SIGNATURE)
        gf.write(struct.pack("i", x.shape[1])) 
        gf.write(struct.pack("i", y.shape[0]))

        x_coords = x[0,:].double().cpu().numpy().flatten()
        y_coords = y[:,0].double().cpu().numpy().flatten()
        z_values = z.cpu().double().numpy()

        # Convert from mm to meters
        x_coords /= 1000
        y_coords /= 1000
        z_values /= 1000

        gf.write(x_coords.tobytes())
        gf.write(y_coords.tobytes())
        gf.write(z_values.tobytes())

def generate_test_surface(nx=1000, ny=1000, device="cuda"):
    """
    Generate a test surface with a sinusoidal pattern and a lipss pattern.

    Parameters
    ----------
    nx : int
        The number of points in the x-axis.
    ny : int
        The number of points in the y-axis.
    device : str
        The device to use for the tensors.

    Returns
    -------
    X : torch.Tensor
        The x-coordinates of the surface.
    Y : torch.Tensor
        The y-coordinates of the surface.
    Z : torch.Tensor
        The z-coordinates of the surface.
    """
    np.random.seed(0)
    step = 0.1
    period = 20 / step
    period_lipss = 1 / step
    x = np.arange(nx).astype(float)
    y = np.arange(ny).astype(float)
    x, y = np.meshgrid(x, y)
    z = np.sin(x / period * 2 * np.pi)
    z += 0.5 * (np.sin((x - period / 2) / period * 2 * np.pi) + 1) * 0.3 * np.sin(
        (np.sin(y / 10) + x) / period_lipss * 2 * np.pi)
    z += np.random.normal(size=z.shape) / 5
    z *= 5
    z[:, nx//9:9*nx//10] += 100    

    X = torch.from_numpy(x).to(device).float()
    Y = torch.from_numpy(y).to(device).float()
    Z = torch.from_numpy(z).to(device).float()
    return X, Y, Z

def generate_sinusoidal_surface(N=1000, M=1000, params=None, interpolation="linear", z_offset = 1000, borders = 100, device="cuda"):
    """Generate a 2D sinusoidal surface with interpolation.

    Parameters:
        N (int): Number of rows (samples).
        M (int): Number of columns.
        params (list of tuples): List of (column index, period T, amplitude A).
        
    Returns:
        surface (torch.Tensor): Generated 2D surface.
    """
    if params is None:
        params = [
            (0, 200, 100),        # First column
            (M // 3, 300, 0.0),   # Middle column
            (M // 2, 300, 0.0),   # Middle column
            (2*M // 3, 300, 0.0),   # Middle column
            (M - 1, 400, 100.0)     # Last column
        ]

    t = torch.arange(N, dtype=torch.float32, device=device)  # Time index
    z = torch.zeros((N, M), dtype=torch.float32, device=device)

    # Assign sinusoidal values for the specified columns
    known_x = []
    known_y = []
    for col, T, A in params:
        signal = A/2 * torch.sin(2 * torch.pi * t / T)
        z[:, col] = signal
        known_x.append(col)
        known_y.append(signal.cpu().numpy())

    known_x = torch.tensor(known_x, dtype=torch.float32, device=device).cpu().numpy()
    known_y = torch.stack([torch.tensor(y) for y in known_y], dim=1).cpu().numpy()

    # Interpolate row-wise
    for i in range(N):
        interpolator = interp1d(known_x, known_y[i], kind=interpolation)
        z[i] = torch.tensor(interpolator(torch.arange(M, dtype=torch.float32))).to(device)

    # num_rows, num_cols = z.shape
    x = torch.arange(N, dtype=torch.float32, device=device)
    y = torch.arange(M, dtype=torch.float32, device=device)
    x, y = torch.meshgrid(x, y, indexing='xy')

    z += z_offset
    set_border_columns_value(borders, 0, z)

    return x, y, z

def set_border_columns_value(N, value, *matrices):
    """
    Set the first and last N columns of the input matrices to a specific value.

    Parameters
    ----------
    N : int
        The number of columns to set to the specified value.
    value : float
        The value to set the columns to.
    matrices : torch.Tensor
        The matrices to update.
    """
    updated_matrices = []
    for matrix in matrices:
        _, cols = matrix.shape

        # Ensure N is valid
        if N < 0 or 2 * N > cols:
            raise ValueError(f"N={N} is too large for a matrix with {cols} columns.")

        # Set first and last N columns to zero
        matrix[:, :N] = value
        matrix[:, -N:] = value
        updated_matrices.append(matrix)

    return updated_matrices