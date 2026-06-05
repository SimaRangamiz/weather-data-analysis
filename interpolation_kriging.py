import numpy as np
from pykrige.ok import OrdinaryKriging

def interpolate_kriging_grid(lon_points, lat_points, values, new_resolution=0.01, variogram_model='linear'):
    """
    Perform Ordinary Kriging interpolation on given points.
    """
    # create grid coordinates with given resolution
    lon_grid = np.arange(np.min(lon_points), np.max(lon_points), new_resolution)
    lat_grid = np.arange(np.min(lat_points), np.max(lat_points), new_resolution)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

    # initialize Ordinary Kriging object
    OK = OrdinaryKriging(
        lon_points,
        lat_points,
        values,
        variogram_model=variogram_model,
        verbose=False,
        enable_plotting=False
    )

    # execute kriging on the grid
    z, ss = OK.execute('grid', lon_grid, lat_grid)

    # return grid and interpolated values
    return lon_mesh, lat_mesh, z
