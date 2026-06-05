import numpy as np

def idw(x, y, points, values, power=2):
    # calculate Euclidean distance from (x, y) to all points
    dist = np.sqrt((points[:, 0] - x)**2 + (points[:, 1] - y)**2)

    # avoid division by zero (if point coincides with data point)
    dist[dist == 0] = 1e-10

    # compute weights as inverse of distance^power
    weights = 1 / dist**power

    # return weighted average
    return np.sum(weights * values) / np.sum(weights)


def interpolate_idw_grid(lon, lat, values, new_resolution=100):
    # create new grid with higher resolution
    lon_new = np.linspace(lon.min(), lon.max(), new_resolution)
    lat_new = np.linspace(lat.min(), lat.max(), new_resolution)
    lon_grid, lat_grid = np.meshgrid(lon_new, lat_new)

    # initialize result grid
    interp_grid = np.zeros_like(lon_grid)

    # prepare original points (lon, lat)
    points_lon, points_lat = np.meshgrid(lon, lat)
    points = np.column_stack((points_lon.flatten(), points_lat.flatten()))

    # flatten values to match points
    vals = values.flatten()

    # perform IDW interpolation for each grid point
    for i in range(lon_grid.shape[0]):
        for j in range(lon_grid.shape[1]):
            interp_grid[i, j] = idw(lon_grid[i, j], lat_grid[i, j], points, vals)

    return lon_grid, lat_grid, interp_grid
