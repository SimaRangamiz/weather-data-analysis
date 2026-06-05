import netCDF4 as nc
import numpy as np
from netCDF4 import num2date
import pandas as pd

def extract_precipitation_by_time_and_location(
    file_path, year, season,
    lat_min, lat_max, lon_min, lon_max,
    csv_metadata_file=None
):
    ds = nc.Dataset(file_path)
    print("✅ NetCDF file opened.")

    # load variables
    lat = ds.variables['latitude'][:]
    lon = ds.variables['longitude'][:]
    tp = ds.variables['tp'][:]  # dimensions: (time, lat, lon)
    time_var = ds.variables['valid_time']

    # check units
    tp_units = getattr(ds.variables['tp'], 'units', 'unknown')
    print("tp units:", tp_units)

    # convert time to pandas.Timestamp
    time_dates = num2date(time_var[:], units=time_var.units, calendar='standard')
    time_dates = pd.to_datetime([t.strftime('%Y-%m-%d %H:%M:%S') for t in time_dates])

    # spatial indices
    lat_idx = np.where((lat >= lat_min) & (lat <= lat_max))[0]
    lon_idx = np.where((lon >= lon_min) & (lon <= lon_max))[0]

    # select months based on season
    if season == 1:
        months = [3, 4, 5]
    elif season == 2:
        months = [6, 7, 8]
    elif season == 3:
        months = [9, 10, 11]
    elif season == 4:
        months = [12, 1, 2]
    else:
        ds.close()
        raise ValueError("Season must be between 1 and 4.")

    # time indices for selected season
    time_indices = []
    for i, dt in enumerate(time_dates):
        if season == 4:  # winter spans two years
            if (dt.month == 12 and dt.year == year - 1) or (dt.month in [1, 2] and dt.year == year):
                time_indices.append(i)
        else:
            if dt.year == year and dt.month in months:
                time_indices.append(i)

    # check if data exists
    if len(time_indices) == 0:
        print(f"⚠️ No data found for year {year}, season {season}.")
        ds.close()
        return None

    print(f"🔎 Selected records for year={year}, season={season}: {len(time_indices)}")

    # filter precipitation data
    tp_filtered_all = tp[time_indices, :, :]
    tp_filtered_region = tp_filtered_all[:, lat_idx[:, None], lon_idx]

    # compute seasonal total (mm)
    if "s" in tp_units.lower():  # assume m/s → sum over seconds
        dt_seconds = np.array([
            (time_dates[i + 1] - time_dates[i]).total_seconds()
            for i in range(len(time_indices) - 1)
        ])
        dt_seconds_mean = dt_seconds.mean() if len(dt_seconds) > 0 else 0
        tp_season_mm = np.nansum(tp_filtered_region * dt_seconds_mean, axis=0) * 1000
    else:  # already cumulative (m)
        tp_season_mm = np.nansum(tp_filtered_region, axis=0) * 1000  # m → mm

    # compute stats
    min_precip = np.min(tp_season_mm)
    max_precip = np.max(tp_season_mm)

    # metadata
    metadata = {
        "year": year,
        "season": season,
        "season_name": ["Spring", "Summer", "Autumn", "Winter"][season - 1],
        "unit": "mm",
        "description": "Seasonal precipitation (filtered by region)",
        "time_range": f"{time_dates[min(time_indices)].isoformat()} to {time_dates[max(time_indices)].isoformat()}",
        "source": "ERA5",
        "region": {
            "lat_min": float(lat_min),
            "lat_max": float(lat_max),
            "lon_min": float(lon_min),
            "lon_max": float(lon_max)
        },
        "precipitation_stats": {
            "min_precipitation_mm": float(min_precip),
            "max_precipitation_mm": float(max_precip)
        }
    }

    # save filtered data
    output_file = f'precipitation_filtered_{year}_season{season}.npz'
    np.savez(output_file,
             tp_mean=tp_season_mm,
             tp=tp_filtered_region,
             lat_selected=lat[lat_idx],
             lon_selected=lon[lon_idx],
             metadata=metadata)
    print(f"✅ Data saved to '{output_file}'")

    # save metadata to CSV
    if csv_metadata_file:
        import csv
        with open(csv_metadata_file, mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Key', 'Value'])
            def write_dict(d, prefix=''):
                for k, v in d.items():
                    if isinstance(v, dict):
                        write_dict(v, prefix=f"{prefix}{k}.")
                    else:
                        writer.writerow([f"{prefix}{k}", v])
            write_dict(metadata)
        print(f"✅ Metadata saved to CSV file '{csv_metadata_file}'")

    ds.close()
    print("✅ NetCDF file closed.")
    return metadata
