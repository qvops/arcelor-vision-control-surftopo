import torch

def interp1d(x: torch.Tensor, y: torch.Tensor, xnew: torch.Tensor,
             extrapolate=False, ind=None, ret_ind=False, invalid=-torch.inf) -> torch.Tensor:
    """
    Interpolate 1-D function.

    Parameters
    ----------
    x : torch.Tensor (MxN)
        The x-coordinates of the surface.
        The x-coordinates are assumed to be sorted in ascending order in each row.
    y : torch.Tensor (MxN)
        The y-coordinates of the surface. 
    xnew : torch.Tensor (P)
        The x-coordinates where to evaluate the interpolated values.
    extrapolate : bool, optional
        If True, extrapolate the values with -inf, inf at the beginning and end, by default False
    ind : torch.Tensor, optional
        The index of the closest lower value in x, by default None
    ret_ind : bool, optional
        If True, return the index of the closest lower value in x, by default False

    Returns
    -------
    torch.Tensor
        The interpolated values.
    """
    assert x.shape == y.shape, f"x and y must have the same shape, got {x.shape} and {y.shape}"
    x, y, xnew = torch.atleast_2d(x, y, xnew)
    x = x.contiguous()
    y = y.contiguous()

    # The same xnew for all rows
    xnew = xnew.expand(x.shape[0], -1).contiguous()

    if ind is None:
        # Find the index of the closest lower value in x
        ind = torch.searchsorted(x, xnew) - 1
        invalid_mask = (ind < 0) | (ind >= x.shape[1] - 1)
        ind = torch.clamp(ind, 0, x.shape[1] - 1 - 1)

        xl = torch.gather(x, 1, ind)
        xr = torch.gather(x, 1, ind+1)
        if extrapolate:
            # # Ignore -inf, inf intervals at the beginning and end: extrapolation
            ind[torch.isinf(xl)] +=1
            ind[torch.isinf(xr)] -=1
            ind = torch.clamp(ind, 0, x.shape[1] - 1 - 1)
        else:
            invalid_mask |= torch.isinf(xl) | torch.isinf(xr)

    # Calculate the slope
    eps = torch.finfo(y.dtype).eps
    dx_slopes = x[:, 1:] - x[:, :-1] + eps
    dy_slopes = y[:, 1:] - y[:, :-1]
    slopes = dy_slopes / dx_slopes

    # Calculate the interpolated values
    dx_new = xnew - torch.gather(x, 1, ind)
    ynew = torch.gather(y, 1, ind) + torch.gather(slopes, 1, ind)*dx_new

    # Replace invalid values out of the range
    if not extrapolate:
        ynew[invalid_mask] = invalid

    if ret_ind:
        return ynew, ind
    else:
        return ynew

def interp2d(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, xnew: torch.Tensor, ynew: torch.Tensor,
             extrapolate=True, sort: bool = True) -> torch.Tensor:
    """
    Interpolate 2-D function. Fist interpolate along the x-axis and then along the y-axis.
    Roughly equivalent to scipy.interpolate.griddata.

    Parameters
    ----------
    x : torch.Tensor (MxN)
        The x-coordinates of the surface. 
        The x-coordinates are assumed to be sorted in ascending order in each row.
    y : torch.Tensor (MxN)
        The y-coordinates of the surface. 
        The y-coordinates are assumed to be sorted in ascending order in each column.
    z : torch.Tensor (MxN)
        The z-coordinates of the surface. 
    xnew : torch.Tensor (P)
        The x-coordinates where to evaluate the interpolated values.
    ynew : torch.Tensor (Q)
        The y-coordinates where to evaluate the interpolated values.
    extrapolate : bool, optional
        If True, extrapolate the values with -inf, inf at the beginning and end, by default False
    sort : bool, optional
        If True, sort the input values

    Returns
    -------
    torch.Tensor
        The interpolated values.
    """
    assert x.shape == y.shape == z.shape
    if sort:
        x, y, z = sort_by(x, y, z, order_by=x, dim=1)
    z_inter, ind = interp1d(x, z, xnew, extrapolate, ret_ind=True)
    y_inter = interp1d(x, y, xnew, extrapolate, ind=ind)
    if sort:
        _, y_inter, z_inter = sort_by(None, y_inter, z_inter, order_by=y_inter, dim=0)
    z_inter = interp1d(y_inter.T, z_inter.T, ynew, extrapolate).T

    return z_inter

    # Equivalent but first y:
    # x, y, z = sort_by(x, y, z, order_by=y, dim=0)
    # z_inter = interp1d(y.T, z.T, ynew, extrapolate).T
    # x_inter = interp1d(y.T, x.T, ynew, extrapolate).T

    # x_inter, _, z_inter = sort_by(x_inter, None, z_inter, order_by=x_inter, dim=1)
    # z_inter = interp1d(x_inter, z_inter, xnew, extrapolate)
    # return z_inter

def sort_by(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, order_by: torch.Tensor, dim) -> torch.Tensor:
    """
    Sort the input values by the order_by tensor.

    Parameters
    ----------
    x : torch.Tensor
        The input tensor x.
    y : torch.Tensor
        The input tensor y.
    z : torch.Tensor
        The input tensor z.
    order_by : torch.Tensor
        The tensor to sort by.
    dim : int
        The dimension to sort by.

    Returns
    -------
    torch.Tensor
        The sorted x tensor.
    torch.Tensor
        The sorted y tensor.
    torch.Tensor
        The sorted z tensor.
    """
    order = torch.argsort(order_by, dim=dim, stable=True)

    if x is not None:
        x = torch.gather(x, dim=dim, index=order)
    if y is not None:
        y = torch.gather(y, dim=dim, index=order)
    if z is not None:
        z = torch.gather(z, dim=dim, index=order)

    return x, y, z