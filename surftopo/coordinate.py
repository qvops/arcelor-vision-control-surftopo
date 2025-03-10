import torch
from typing import NamedTuple

class Coordinate(NamedTuple):
    """
    Coordinates in a matrix: rows and cols.
    
    This class allows you to store row and column indices as tensors,
    and it can be used to index a matrix directly:
    
        idx = Coordinate(row, col)
        matrix[idx]

    Parameters
    ----------
    row : torch.Tensor
        The row coordinates as a tensor.
    col : torch.Tensor
        The column coordinates as a tensor.
    """
    row: torch.Tensor
    col: torch.Tensor

    def __getitem__(self, idx):
        """
        Retrieve the coordinates at the specified index.

        Parameters
        ----------
        idx : int or torch.Tensor
            Index or slice to access specific elements of the coordinate tensors.

        Returns
        -------
        Coordinate
            A new Coordinate object with the corresponding elements at the given index.
        """
        return Coordinate(self.row[idx], self.col[idx])
    
    def add(self, row_offset, col_offset):
        """
        Add the specified offsets to the rows and cols.
        """
        return Coordinate(self.row + row_offset, self.col + col_offset)