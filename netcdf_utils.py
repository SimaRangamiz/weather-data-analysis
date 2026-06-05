from netCDF4 import Dataset

def open_netcdf_file(file_path):
    ds = Dataset(file_path, 'r')
    print(f"✅ NetCDF file '{file_path}' opened.")
    return ds

def print_variables(ds):
    print("Variables in the NetCDF file:")
    for var in ds.variables.keys():
        print(f" - {var}")

def print_variable_info(ds, var_name):
    if var_name in ds.variables:
        var = ds.variables[var_name]
        print(f"Information about variable '{var_name}':")
        print(var)
        if hasattr(var, 'units'):
            print(f"Units: {var.units}")
        if hasattr(var, 'long_name'):
            print(f"Description: {var.long_name}")
    else:
        print(f"Variable '{var_name}' not found in the file!")

def close_netcdf_file(ds):
    ds.close()
    print("✅ NetCDF file closed.")
