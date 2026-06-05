from extract_precipitation import extract_precipitation_by_time_and_location
from interpolation_IDW import interpolate_idw_grid
from interpolation_kriging import interpolate_kriging_grid
from interpolation_spline import interpolate_spline_grid
from interpolation_rf import interpolate_rf_grid
from plotting import plot_precipitation_map
from scipy.interpolate import RegularGridInterpolator
from sklearn.ensemble import RandomForestRegressor
from evaluator import plot_comparison_bar_chart
import numpy as np
import os
import pandas as pd

# Import evaluation and comparison functions from evaluator.py
from evaluator import evaluate_model, display_results_table, save_results_table, plot_evaluation

# Mapping for method selection
method_name = {
    '1': 'IDW',
    '2': 'Kriging',
    '3': 'Spline',
    '4': 'Random Forest'
}

# Function to get the desired year from user input
def get_year_input():
    while True:
        try:
            year = int(input("Please enter the desired year : "))
            if year < 1981 or year > 2025:
                print("Year must be between 1981 and 2025.")
                continue
            return year
        except ValueError:
            print("Please enter a valid integer.")

# Function to get the desired season from user input
def get_season_input():
    seasons_dict = {1: "Spring", 2: "Summer", 3: "Autumn", 4: "Winter"}
    while True:
        try:
            print("Seasons: 1=Spring, 2=Summer, 3=Autumn, 4=Winter")
            season = int(input("Please enter the number of the desired season: "))
            if season not in seasons_dict:
                print("Season number must be one of 1, 2, 3, or 4.")
                continue
            return season, seasons_dict[season]
        except ValueError:
            print("Please enter a valid integer.")
            
# Function to get the NetCDF file path based on year
def get_nc_file_by_year(year):
    if 1980 <= year <= 1988:
        return '1980_1988.nc'
    elif 1989 <= year <= 2000:
        return '1989_2000.nc'
    elif 2001 <= year <= 2012:
        return '2001_2012.nc'
    elif 2013 <= year <= 2024:
        return '2013_2024.nc'
    else:
        print("❌ Year out of range (1980-2024).")
        return None
            

# Function to load processed precipitation data from .npz file
def load_precipitation_data(npz_file):
    if not os.path.exists(npz_file):
        print(f"❌ Output file '{npz_file}' not found!")
        return None
    data = np.load(npz_file, allow_pickle=True)
    return data

# Function to save evaluation metrics to CSV
def save_metrics_to_csv(metrics, filename):
    df = pd.DataFrame(metrics if isinstance(metrics, list) else [metrics])
    df.to_csv(filename, index=False)
    print(f"✅ Evaluation results saved to '{filename}'.")

# Function to perform interpolation and plotting
def interpolate_and_plot(method, lon_selected, lat_selected, tp_mean):
    print(f"\n🔄 Performing {method_name[method]} interpolation...")

    # Create meshgrid and flatten for interpolation
    lon_mesh, lat_mesh = np.meshgrid(lon_selected, lat_selected)
    lon_points = lon_mesh.flatten()
    lat_points = lat_mesh.flatten()
    tp_points = tp_mean.flatten()
    sample_points = np.vstack((lat_points, lon_points)).T  # Note: lat/lon swapped for RegularGridInterpolator

    # IDW interpolation
    if method == '1':
        lon_grid, lat_grid, interp_grid = interpolate_idw_grid(
            lon_selected, lat_selected, tp_mean, new_resolution=100
        )
        plot_precipitation_map(lon_grid, lat_grid, interp_grid, method="IDW")
        interpolator = RegularGridInterpolator(
            (lat_grid[:, 0], lon_grid[0, :]), interp_grid,
            bounds_error=False, fill_value=None
        )
        predicted_values = interpolator(sample_points)
        return predicted_values, tp_points

    # Kriging interpolation
    elif method == '2':
        lon_grid, lat_grid, interp_grid = interpolate_kriging_grid(
            lon_points, lat_points, tp_points, new_resolution=0.01
        )
        plot_precipitation_map(lon_grid, lat_grid, interp_grid, method="Kriging")
        interpolator = RegularGridInterpolator(
            (lat_grid[:, 0], lon_grid[0, :]), interp_grid,
            bounds_error=False, fill_value=None
        )
        predicted_values = interpolator(sample_points)
        return predicted_values, tp_points

    # Spline interpolation
    elif method == '3':
        lon_grid, lat_grid, interp_grid = interpolate_spline_grid(
            lon_points, lat_points, tp_points, new_resolution=100
        )
        plot_precipitation_map(lon_grid, lat_grid, interp_grid, method="Spline")
        interpolator = RegularGridInterpolator(
            (lat_grid[:, 0], lon_grid[0, :]), interp_grid,
            bounds_error=False, fill_value=None
        )
        predicted_values = interpolator(sample_points)
        return predicted_values, tp_points

    # Random Forest interpolation
    elif method == '4':
        X_train = np.vstack((lon_points, lat_points)).T
        y_train = tp_points

        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)

        lon_grid = np.linspace(np.min(lon_selected), np.max(lon_selected), 100)
        lat_grid = np.linspace(np.min(lat_selected), np.max(lat_selected), 100)
        lon_grid_mesh, lat_grid_mesh = np.meshgrid(lon_grid, lat_grid)
        X_grid = np.vstack((lon_grid_mesh.flatten(), lat_grid_mesh.flatten())).T

        y_pred_grid = rf.predict(X_grid)
        interp_grid = y_pred_grid.reshape(lon_grid_mesh.shape)

        plot_precipitation_map(lon_grid_mesh, lat_grid_mesh, interp_grid, method="Random Forest")

        predicted_values = rf.predict(X_train)
        return predicted_values, y_train

    else:
        print("❌ Invalid interpolation method.")
        return None, None
   

# Main function
def main():
    year = get_year_input()
    file_path = get_nc_file_by_year(year)
    season_num, season_name = get_season_input()

    if file_path is None:
        return
    
    # Define region of interest (latitude and longitude bounds)
    lat_min, lat_max = 33.8, 35.3
    lon_min, lon_max = 47.5, 49.5

    # Extract precipitation data for selected year, season, and region
    extract_precipitation_by_time_and_location(
        file_path=file_path,
        year=year,
        season=season_num,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        csv_metadata_file=f'metadata_{year}_season{season_num}.csv'
    )
    
    # Load processed data
    npz_file = f'precipitation_filtered_{year}_season{season_num}.npz'
    data = load_precipitation_data(npz_file)
    if data is None:
        return

    tp_mean = data['tp_mean']
    lat_selected = data['lat_selected']
    lon_selected = data['lon_selected']

    print(f"\n📍 Minimum precipitation: {np.min(tp_mean):.2f} mm")
    print(f"📍 Maximum precipitation: {np.max(tp_mean):.2f} mm")

    if np.all(tp_mean == 0):
        print("❌ All precipitation values are zero. Interpolation is not meaningful.")
        return

    # Show available interpolation methods
    print("\nAvailable interpolation methods:")
    for key, name in method_name.items():
        print(f"{key}. {name}")

    # Loop for user to select methods
    all_results = {}
    while True:
        method = input("\nPlease select interpolation method (1-4) or 'q' to quit: ")
        if method == 'q':
            break
        if method not in method_name:
            print("❌ Invalid choice. Please enter 1-4 or 'q'.")
            continue

        # Perform interpolation and get predicted vs true values
        predicted_values, true_values = interpolate_and_plot(method, lon_selected, lat_selected, tp_mean)
        if predicted_values is None or true_values is None:
            continue

        # Evaluate model
        metrics = evaluate_model(true_values, predicted_values)
        all_results[method_name[method]] = metrics

    # Display and save evaluation results
    if all_results:
        display_results_table(all_results)
        save_metrics_to_csv(all_results, f"evaluation_results_{year}_season{season_num}.csv")
        save_results_table(all_results, f"evaluation_table_{year}_season{season_num}.txt")
    else:
        print("⚠️ No metrics were generated.")
        
    all_results_list = [
        {"Method": method, **metrics} for method, metrics in all_results.items()
    ]
    plot_comparison_bar_chart(all_results_list)    

if __name__ == "__main__":
    main()
