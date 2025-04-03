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
