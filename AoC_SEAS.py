"""
Notebook for calculating Areas of Concern (AoC) with predictions from ECMWF SEAS5.1 .
The output is probabilistic utilizing all ensemble members.


Author: Odysseas Vlachopoulos

Copyright (C) 2025 Odysseas Vlachopoulos

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation version 3 of the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
import os
import xarray as xr
import glob
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors

ref_year_start_seas = 1993
ref_year_end_seas = 2015
forc_year_start_seas = 2017
forc_year_end_seas = 2023

""" NOW MAKE THE SAME LAST DAY TWSO BUNDLE FOR THE SEASONAL HINDCAST , FOR EACH ENSEMBLE MEMBER """
ens_numbers_h = np.arange(1, 25 + 1)
years_h = np.arange(ref_year_start_seas, ref_year_end_seas + 1, 1)
dir_name_h = 'SEAS5_HINDCAST'
seas_dir_h = f'/path/to/RNN predictions dir/PREDICTIONS_SEAS/SEAS5_HINDCAST/'

ens_numbers_f = np.arange(1, 51 + 1)
years_f = np.arange(forc_year_start_seas, forc_year_end_seas + 1, 1)
dir_name_f = 'SEAS5_FORECAST'
seas_dir_f = f'/path/to/RNN predictions dir/PREDICTIONS_SEAS/SEAS5_FORECAST/'


def combine_last_day_of_year(files, output_file, new_var_name="TWSO"):
    last_day_data = []

    for file in files:
        ds = xr.open_dataset(file)
        var_name = [var for var in ds.data_vars if var.startswith("TWSO")][0]

        last_day = ds.isel(time=-1)  # Select the last time index
        last_day = last_day.rename({var_name: new_var_name})
        last_day_data.append(last_day)

    combined = xr.concat(last_day_data, dim="time")
    combined.to_netcdf(output_file)
    print(f"Combined file saved to: {output_file}")


""" Combine last day of the years needed for the SEAS5 TWSO data"""
""" For the HINDCAST """
for ens in ens_numbers_h:
    ens_number_str = f'{ens:02d}'  # Format with leading zeros
    ens_dir = os.path.join(os.sep, seas_dir_h, f"ens_{ens_number_str}")

    seas_twso_combined_flpth = os.path.join(os.sep, ens_dir,
                                            f"{ref_year_start_seas}_{ref_year_end_seas}_ens{ens_number_str}_TWSO_finalday_SEAS.nc")
    if not os.path.exists(seas_twso_combined_flpth):
        seas_twso_files = [os.path.join(ens_dir, f"{year}_ens{ens_number_str}_TWSO_predicted_withSEAS.nc") for year in
                           range(ref_year_start_seas, ref_year_end_seas + 1)]
        combine_last_day_of_year(seas_twso_files, seas_twso_combined_flpth, 'TWSO')

""" For the FORECAST """
for ens in ens_numbers_f:
    ens_number_str = f'{ens:02d}'  # Format with leading zeros
    ens_dir = os.path.join(os.sep, seas_dir_f, f"ens_{ens_number_str}")

    seas_twso_combined_flpth = os.path.join(os.sep, ens_dir,
                                            f"{forc_year_start_seas}_{forc_year_end_seas}_ens{ens_number_str}_TWSO_finalday_SEAS.nc")
    if not os.path.exists(seas_twso_combined_flpth):
        seas_twso_files = [os.path.join(ens_dir, f"{year}_ens{ens_number_str}_TWSO_predicted_withSEAS.nc") for year in
                           range(forc_year_start_seas, forc_year_end_seas + 1)]
        combine_last_day_of_year(seas_twso_files, seas_twso_combined_flpth, 'TWSO')

""" THIS IS A CROPPING SCRIPT FOR THE METRICS OF PREDICTIONS """
""" First open a dataset with the X,Y,lat,lon in order to get the extent """
ds = xr.open_dataset("/path/to/{crop}{year}_output2_corrected_ATT_TWSO.nc")
ds = ds.load()

# 2. (Optional) Explicitly set the fill value attribute, so it's documented in ds
#    Note: xarray does not automatically mask these unless you re-decode the dataset.
ds["ATT_TWSO"].attrs["_FillValue"] = -9999

# 3. Convert all -9999 values to NaN
# ds["ATT_TWSO"] = ds["ATT_TWSO"].where(ds["ATT_TWSO"] != -9999)
ds['ATT_TWSO'] = ds['ATT_TWSO'].where(ds['ATT_TWSO'] != -9999, np.nan)
ds['ATT_TWSO'] = ds['ATT_TWSO'].where(~np.isnan(ds['lon']))

num_zeros = int((ds['ATT_TWSO'] == 0).sum().values)
print("Number of zeros:", num_zeros)

ds['ATT_TWSO'] = ds['ATT_TWSO'].where(ds['ATT_TWSO'] != 0, -1000)
num_alias = int((ds['ATT_TWSO'] == -1000).sum().values)
print("Number of -1000:", num_zeros)

# Now 'ATT_TWSO' has NaNs wherever the original data was -9999.
# Any interpolation or plotting code will treat these cells as missing.

# Select valid ATT_TWSO values (not NaN and not -1000)
valid_mask = (~np.isnan(ds["ATT_TWSO"])) & (ds["ATT_TWSO"] != -1000)

# Extract corresponding lon and lat values
valid_lons = ds["lon"].where(valid_mask)
valid_lats = ds["lat"].where(valid_mask)

# Compute min and max
min_lon = valid_lons.min().values
max_lon = valid_lons.max().values
min_lat = valid_lats.min().values
max_lat = valid_lats.max().values

print(f"Min Lon: {min_lon}, Max Lon: {max_lon}")
print(f"Min Lat: {min_lat}, Max Lat: {max_lat}")


def subset_dataset(ds2):
    # Find the closest lat/lon values in the second dataset
    min_lon_closest = ds2.lon.sel(lon=min_lon, method="nearest").values
    max_lon_closest = ds2.lon.sel(lon=max_lon, method="nearest").values
    min_lat_closest = ds2.lat.sel(lat=min_lat, method="nearest").values
    max_lat_closest = ds2.lat.sel(lat=max_lat, method="nearest").values

    # Subset the second dataset
    if ds2.lat[-1] < ds2.lat[0]:  # This is the case for SEAS , the lats are descending
        ds2_subset = ds2.sel(lon=slice(min_lon_closest, max_lon_closest),
                             lat=slice(max_lat_closest, min_lat_closest))
    else:
        ds2_subset = ds2.sel(lon=slice(min_lon_closest, max_lon_closest),
                             lat=slice(min_lat_closest, max_lat_closest))
    return ds2_subset


""" 
END OF PREPROCESSING 
START OF METRICS
COMPARE THE ERA-forced TWSO WITH THE SEAS-forced TWSO
"""


def plot7(ds, title, save_path):
    # If ds is a Dataset, extract the first data variable; otherwise assume it's a DataArray.
    if isinstance(ds, xr.Dataset):
        var_name = list(ds.data_vars.keys())[0]
        arr = ds[var_name]
    else:
        arr = ds

    years = arr["time"].dt.year.values

    # Create subplots with shared lat/lon axes (here 9 panels in a 3x3 layout)
    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 10), sharex=True, sharey=True,
                             subplot_kw={"projection": proj})
    axes = axes.flatten()

    # Extract latitude and longitude values from the coordinates.
    latitudes = arr.coords["lat"].values
    longitudes = arr.coords["lon"].values

    # Determine map extent based on latitude order.
    if latitudes[0] > latitudes[-1]:
        extent = [longitudes.min(), longitudes.max(), latitudes[-1], latitudes[0]]
    else:
        extent = [longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()]

    # Define a discrete colormap for 5 categories:
    # 0: Inconclusive, 1: Above-normal, 2: Normal to above-normal, 3: Normal, 4: Below-normal
    cmap = mcolors.ListedColormap(["gray", "blue", "lightblue", "green", "red"])
    # Boundaries are set so that integer values fall in distinct bins.
    norm = mcolors.BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], ncolors=5)

    # Loop through each time step and plot using the discrete colormap.
    for i in range(len(years)):
        ax = axes[i]
        ax.set_extent(extent, crs=proj)
        im = ax.pcolormesh(longitudes, latitudes, arr.isel(time=i).values,
                           cmap=cmap, norm=norm, transform=proj)
        ax.set_title(f"Year {years[i]}", fontsize=16)
        ax.coastlines()
        ax.add_feature(cfeature.LAND, facecolor="lightgray")
        ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
        ax.axis("off")

    # Remove any unused subplots.
    for j in range(len(years), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(title, fontsize=16)
    fig.subplots_adjust(right=0.85, bottom=0.1, left=0.12, top=0.92)

    # Add a single vertical colorbar with discrete ticks and labels.
    cbar_ax = fig.add_axes([0.87, 0.2, 0.02, 0.6])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="vertical", ticks=[0, 1, 2, 3, 4])
    cbar.ax.set_yticklabels(["Inconclusive", "Above-normal", "Normal to above-normal", "Normal", "Below-normal"],
                            fontsize=16)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {save_path}", flush=True)
    # plt.show(fig)


""" HINDCAST PREPROCESSING """
# Build a sorted list of all ensemble files using a glob pattern.
file_pattern_h = os.path.join(os.sep, seas_dir_h, 'ens_*/1993_2015_ens*_TWSO_finalday_SEAS.nc')
files_h = sorted(glob.glob(file_pattern_h))

# Read each file, add an ensemble dimension, and collect the datasets in a list.
data_list_h = []
for f in files_h:
    ds = xr.open_dataset(f)
    ds = ds.expand_dims('ensemble')  # manually add a new 'ensemble' dimension
    data_list_h.append(ds)

# Concatenate the datasets along the new 'ensemble' dimension.
ensemble_ds_h = xr.concat(data_list_h, dim='ensemble')

""" FORECAST PREPROCESSING """
# Build a sorted list of all ensemble files using a glob pattern.
file_pattern_f = os.path.join(os.sep, seas_dir_f, 'ens_*/2017_2023_ens*_TWSO_finalday_SEAS.nc')
files_f = sorted(glob.glob(file_pattern_f))

# Read each file, add an ensemble dimension, and collect the datasets in a list.
data_list_f = []
for f in files_f:
    ds = xr.open_dataset(f)
    ds = ds.expand_dims('ensemble')  # manually add a new 'ensemble' dimension
    data_list_f.append(ds)

# Concatenate the datasets along the new 'ensemble' dimension.
ensemble_ds_f = xr.concat(data_list_f, dim='ensemble')  # ensemble: 51 time: 7 lat: 48 lon: 83


def relative_anomaly_probabilistic_forecast(ref, forecast, save_dir):
    """
    Computes a probabilistic forecast for the crop yield variable TWSO based on its relative anomaly,
    considering only negative relative anomalies (yield < climatology).

    The procedure is as follows:
      1. Compute the climatology (mean over time and ensemble members) of TWSO from the reference dataset.
      2. Compute the reference relative anomaly (in percent) and keep only negative anomalies.
      3. Derive dynamic thresholds from the masked reference relative anomaly:
             lower_thresh = 33rd percentile (0.33 quantile) over time and member
             upper_thresh = 66th percentile (0.66 quantile) over time and member
      4. Compute the forecast relative anomaly (using the same climatology) and select only negative anomalies.
      5. For each ensemble member in the forecast, classify the anomaly into:
             - Above-normal: forecast anomaly > upper_thresh
             - Normal: between lower_thresh and upper_thresh
             - Below-normal: forecast anomaly < lower_thresh
      6. Compute the percentage (probability) of ensemble members in each category.
      7. Apply decision rules to assign a “most probable forecast” category per grid cell and forecast time:
             0: Inconclusive (if probability below-normal equals probability above-normal)
             1: Above-normal (if probability above-normal > both normal and below-normal)
             2: Normal to above-normal (if probability above-normal equals probability normal)
             3: Normal (if probability normal > both above-normal and below-normal)
             4: Below-normal (if probability below-normal > both above-normal and normal)

    Parameters:
      ref (xarray.Dataset): Reference dataset containing TWSO with dimensions "time" and "member".
      forecast (xarray.Dataset): Forecast dataset containing TWSO with dimensions "time", "member", and spatial dims.
      save_dir (str): Directory path where the output NetCDF files will be saved.

    Returns:
      most_probable_category (xarray.DataArray): Forecast category per time, lat, lon.
      prob_above (xarray.DataArray): Probability (in %) of above-normal.
      prob_normal (xarray.DataArray): Probability (in %) of normal.
      prob_below (xarray.DataArray): Probability (in %) of below-normal.
    """
    # Load datasets and ensure coordinate consistency
    ref = ref.load()
    forecast = forecast.load()
    forecast = forecast.assign_coords(lat=ref.lat)

    # Step 1: Compute the climatology from the reference dataset (mean over time and ensemble members)
    climatology = ref.TWSO.mean(dim=['time', 'ensemble'])

    # Step 2: Compute the reference relative anomaly (in percent) and select only negative anomalies
    ref_relative_anomaly = (ref.TWSO - climatology) / climatology * 100
    ref_relative_anomaly = ref_relative_anomaly.where(ref_relative_anomaly < 0)

    # Step 3: Derive dynamic thresholds from the reference relative anomaly (ignoring NaNs)
    lower_thresh = ref_relative_anomaly.quantile(0.33, dim=["time", "ensemble"])
    upper_thresh = ref_relative_anomaly.quantile(0.66, dim=["time", "ensemble"])

    # Step 4: Compute the forecast relative anomaly (using the same climatology) and keep only negative anomalies.
    forecast_anomaly = (forecast.TWSO - climatology) / climatology * 100
    forecast_anomaly = forecast_anomaly.where(forecast_anomaly < 0)

    # Step 5: Classify each ensemble member while preserving NaNs.
    # If forecast_anomaly is NaN, the classification remains NaN.
    is_above = xr.where(forecast_anomaly.isnull(), np.nan, forecast_anomaly > upper_thresh)
    is_normal = xr.where(forecast_anomaly.isnull(), np.nan,
                         (forecast_anomaly <= upper_thresh) & (forecast_anomaly >= lower_thresh))
    is_below = xr.where(forecast_anomaly.isnull(), np.nan, forecast_anomaly < lower_thresh)

    # Step 6: Compute the probability (percentage) of ensemble members in each category.
    # Count only valid (non-NaN) members.
    valid_count = forecast_anomaly.notnull().sum(dim="ensemble")
    valid_count = valid_count.where(valid_count > 0)  # If none are valid, result remains NaN

    # Use fillna(0) so that NaNs are not counted in the sum.
    prob_above = (is_above.fillna(0)).sum(dim="ensemble") / valid_count * 100
    prob_normal = (is_normal.fillna(0)).sum(dim="ensemble") / valid_count * 100
    prob_below = (is_below.fillna(0)).sum(dim="ensemble") / valid_count * 100

    # Step 7: Determine the "most probable forecast" category per grid cell and forecast time.
    def compute_category(prob_above, prob_normal, prob_below, tol=1e-6):
        # Create an array initialized with a placeholder (-1) for undefined cases
        category = xr.full_like(prob_above, fill_value=-1, dtype=int)

        # 0: Inconclusive if probability below-normal equals probability above-normal (within tolerance)
        category = xr.where(np.abs(prob_below - prob_above) < tol, 0, category)

        # 1: Above-normal if prob_above is strictly greater than both prob_normal and prob_below
        category = xr.where((prob_above > prob_normal) & (prob_above > prob_below) & (category == -1), 1, category)

        # 2: Normal to above-normal if prob_above equals prob_normal (within tolerance)
        category = xr.where((np.abs(prob_above - prob_normal) < tol) & (category == -1), 2, category)

        # 3: Normal if prob_normal is strictly greater than both prob_above and prob_below
        category = xr.where((prob_normal > prob_above) & (prob_normal > prob_below) & (category == -1), 3, category)

        # 4: Below-normal if prob_below is strictly greater than both prob_above and prob_normal
        category = xr.where((prob_below > prob_above) & (prob_below > prob_normal) & (category == -1), 4, category)

        # Default any remaining undefined cases to 0, but only where probabilities are valid;
        # if any of the probabilities is NaN, keep the category as NaN.
        category = xr.where((prob_above.isnull() | prob_normal.isnull() | prob_below.isnull()),
                            np.nan, xr.where(category == -1, 0, category))

        return category

    most_probable_category = compute_category(prob_above, prob_normal, prob_below)

    most_probable_category = subset_dataset(most_probable_category)
    prob_above = subset_dataset(prob_above)
    prob_normal = subset_dataset(prob_normal)
    prob_below = subset_dataset(prob_below)

    prob_above.to_netcdf(os.path.join(save_dir, "SEAS5_TWSO_AoC_prob_above.nc"))
    prob_normal.to_netcdf(os.path.join(save_dir, "SEAS5_TWSO_AoC_prob_normal.nc"))
    prob_below.to_netcdf(os.path.join(save_dir, "SEAS5_TWSO_AoC_prob_below.nc"))
    most_probable_category.to_netcdf(os.path.join(save_dir, "SEAS5_TWSO_AoC_most_probable_category.nc"))

    return most_probable_category, prob_above, prob_normal, prob_below


most_probable_category, prob_above, prob_normal, prob_below = relative_anomaly_probabilistic_forecast(ensemble_ds_h,
                                                                                                      ensemble_ds_f,
                                                                                                      seas_dir_h)

save_path = os.path.join(os.sep, seas_dir_h, f'SEAS5_TWSO_Categorical.png')

# if not os.path.exists(save_path):
title = f'SEAS5 TWSO Categorical'
plot7(most_probable_category, title, save_path)
