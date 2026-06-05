import matplotlib.pyplot as plt
import numpy as np

def plot_precipitation_map(lon_grid, lat_grid, interp_grid, method="", lon_selected=None, lat_selected=None, tp_mean=None):    
    # create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))

    # plot interpolated precipitation map
    c = ax.pcolormesh(lon_grid, lat_grid, interp_grid, shading='auto', cmap='Blues')
    plt.colorbar(c, label='Seasonal precipitation (mm)')
    
    # set title based on method
    if method == "IDW":
        ax.set_title("IDW Interpolated Seasonal Precipitation Map")
    elif method == "Kriging":
        ax.set_title("Kriging Interpolated Seasonal Precipitation Map")
    elif method == "Spline":
        ax.set_title("Spline Interpolated Seasonal Precipitation Map")
    elif method == "Random Forest":
        ax.set_title("Random Forest Interpolated Seasonal Precipitation Map")
    else:
        ax.set_title("Interpolated Seasonal Precipitation Map")
        
    # axis labels and grid
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # plot original data points if provided
    if lon_selected is not None and lat_selected is not None:
        ax.scatter(lon_selected, lat_selected, color='black', s=30, marker='x', label='Original Data Points')
        ax.legend()

    # interactive annotation for showing values on hover
    annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    lon_new = lon_grid[0, :]
    lat_new = lat_grid[:, 0]

    def on_move(event):
        if event.inaxes == ax:
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                # check if inside grid bounds
                if lon_new.min() <= x <= lon_new.max() and lat_new.min() <= y <= lat_new.max():
                    # find nearest grid cell
                    i = np.abs(lat_new - y).argmin()
                    j = np.abs(lon_new - x).argmin()
                    precip = interp_grid[i, j]
                    annot.xy = (x, y)
                    annot.set_text(f"Lat: {y:.2f}\nLon: {x:.2f}\nPrecip: {precip:.2f} mm")
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                else:
                    annot.set_visible(False)
            else:
                annot.set_visible(False)

    # connect hover event
    fig.canvas.mpl_connect("motion_notify_event", on_move)

    plt.tight_layout()
    plt.show(block=False)
