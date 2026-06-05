import numpy as np
from scipy.interpolate import SmoothBivariateSpline

def interpolate_spline_grid(lon, lat, values, new_resolution=100):
    """
    Interpolation using 2D Smooth Bivariate Spline on a new grid.
    """

    # create uniform grid within lon/lat range
    lon_lin = np.linspace(min(lon), max(lon), new_resolution)
    lat_lin = np.linspace(min(lat), max(lat), new_resolution)
    lon_grid, lat_grid = np.meshgrid(lon_lin, lat_lin)

    # fit spline to data
    spline = SmoothBivariateSpline(lon, lat, values)

    # evaluate spline on new grid
    interp_values = spline.ev(lon_grid.ravel(), lat_grid.ravel())
    interp_grid = interp_values.reshape(lon_grid.shape)

    return lon_grid, lat_grid, interp_grid
