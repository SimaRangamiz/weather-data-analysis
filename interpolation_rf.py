import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def interpolate_rf_grid(lons, lats, values, new_resolution=0.05, n_estimators=100):
    """
    Interpolation using Random Forest on a new grid.
    """

    # flatten input arrays
    lons_flat = lons.flatten()
    lats_flat = lats.flatten()
    values_flat = values.flatten()

    # remove NaN values
    mask = ~np.isnan(values_flat)
    lons_flat = lons_flat[mask]
    lats_flat = lats_flat[mask]
    values_flat = values_flat[mask]

    # create new grid within data range
    lon_min, lon_max = lons.min(), lons.max()
    lat_min, lat_max = lats.min(), lats.max()

    new_lon = np.arange(lon_min, lon_max + new_resolution, new_resolution)
    new_lat = np.arange(lat_min, lat_max + new_resolution, new_resolution)
    grid_lon, grid_lat = np.meshgrid(new_lon, new_lat)

    # prepare training data
    X_train = np.column_stack((lons_flat, lats_flat))
    y_train = values_flat

    # train Random Forest model
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)

    # prepare prediction input
    X_pred = np.column_stack((grid_lon.ravel(), grid_lat.ravel()))

    # predict values
    y_pred = rf.predict(X_pred)

    # reshape predictions to grid shape
    interp_values = y_pred.reshape(grid_lon.shape)

    return grid_lon, grid_lat, interp_values


def evaluate_rf_model(model, lon_selected, lat_selected, tp_mean):
    """
    Evaluate Random Forest model on original data.
    """
    # create mesh from selected coordinates
    lat_len, lon_len = tp_mean.shape
    lon_mesh, lat_mesh = np.meshgrid(lon_selected[:lon_len], lat_selected[:lat_len])
    X = np.column_stack((lon_mesh.flatten(), lat_mesh.flatten()))

    # true values (scaled to mm)
    y_true = tp_mean.flatten() * 1000

    # mask NaN values
    mask = ~np.isnan(y_true)
    X = X[mask]
    y_true = y_true[mask]

    # predict using trained model
    y_pred = model.predict(X)

    # calculate metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # print evaluation results
    # print("🔍 Random Forest Evaluation:")
    # print(f"📉 RMSE: {rmse:.2f} mm")
    # print(f"📈 R²: {r2:.3f}")

    return rmse, r2
