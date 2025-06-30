"""
Script for calculating Areas of Concern (AoC) with predictions from CMIP6 datasets.
The output is a binary classification.


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
import pandas as pd
import matplotlib.colors as mcolors
import csv
import shutil

cmip_twso_pred_dir = f'/path/to/PREDICTIONS_CMIP/'
ref_year_start_cmnip = 1993
ref_year_end_cmip = 2014
forc_year_start_cmip = 2015
forc_year_end_cmip = 2050

""" THIS IS A CROPPING SCRIPT FOR THE METRICS OF ERA PREDICTIONS vs CMIP PREDICTIONS """
""" First open a dataset with the X,Y,lat,lon in order to get the extent """
ds = xr.open_dataset("/path/to/{crop}{year}_output2_corrected_ATT_TWSO.nc")
ds = ds.load()

# Explicitly set the fill value attribute, so it's documented in ds
# Note: xarray does not automatically mask these unless you re-decode the dataset.
ds["ATT_TWSO"].attrs["_FillValue"] = -9999

# Convert all -9999 values to NaN
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
COMPARE THE ERA TWSO WITH THE CMIP TWSO
"""


def plot36(ds, title, save_path):
    var_name = list(ds.data_vars.keys())[0]
    years = ds["time"].dt.year.values

    # Create subplots with shared lat/lon axes, adjusting for 23 plots (closer to 5x5 layout)
    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(nrows=6, ncols=6, figsize=(18, 10), sharex=True, sharey=True,
                             subplot_kw={"projection": proj})

    # Ensure axes are correctly flattened
    axes = axes.flatten()

    # Extract latitude and longitude values
    latitudes = ds.coords["lat"].values
    longitudes = ds.coords["lon"].values

    # Determine the correct vertical extent based on latitude order
    if latitudes[0] > latitudes[-1]:
        # Latitudes are in descending order; set extent with lat_min as the last value
        extent = [longitudes.min(), longitudes.max(), latitudes[-1], latitudes[0]]
    else:
        extent = [longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()]

    # Loop through each time step and plot with shared Latitude and Longitude
    for i in range(len(years)):  # Only plotting 22 subplots
        ax = axes[i]
        ax.set_extent(extent, crs=proj)

        im = ax.pcolormesh(longitudes, latitudes, ds[var_name].isel(time=i).values, cmap="viridis",
                           transform=proj)
        ax.set_title(f"Year {years[i]}", fontsize=16)
        ax.coastlines()
        ax.add_feature(cfeature.LAND, facecolor="lightgray")
        ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
        ax.axis("off")  # Hide individual axes

    # Remove the last 3 empty subplots
    for j in range(len(years), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(title, fontsize=16)

    # Adjust layout for better spacing
    fig.subplots_adjust(right=0.85, bottom=0.1, left=0.12, top=0.92)  # Bring elements closer

    # Add common X and Y labels closer to figures
    # fig.text(0.5, 0.05, "Longitude", ha="center", va="center", fontsize=12)  # Closer to bottom
    # fig.text(0.08, 0.5, "Latitude", ha="center", va="center", rotation="vertical", fontsize=12)  # Closer to left

    # Add a single colorbar on the right
    cbar_ax = fig.add_axes([0.87, 0.2, 0.02, 0.6])  # Adjusted position for better spacing
    fig.colorbar(im, cax=cbar_ax, orientation="vertical")
    # plt.show()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {save_path}", flush=True)


def plot36_bin(ds, title, save_path):
    # If ds is a Dataset, extract the first data variable; if not, assume it's already a DataArray.
    if isinstance(ds, xr.Dataset):
        var_name = list(ds.data_vars.keys())[0]
        arr = ds[var_name]
    else:
        arr = ds

    years = arr["time"].dt.year.values
    proj = ccrs.PlateCarree()
    # Create a grid of subplots. Adjust nrows/ncols as needed.
    fig, axes = plt.subplots(nrows=6, ncols=6, figsize=(18, 10), sharex=True, sharey=True,
                             subplot_kw={"projection": proj})
    axes = axes.flatten()

    # Extract latitude and longitude
    latitudes = arr.coords["lat"].values
    longitudes = arr.coords["lon"].values

    # Determine the extent based on the latitude order
    if latitudes[0] > latitudes[-1]:
        extent = [longitudes.min(), longitudes.max(), latitudes[-1], latitudes[0]]
    else:
        extent = [longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()]

    # Define a discrete colormap and norm for binary output:
    # 0: "No concern" (light green), 1: "Concern" (red)
    cmap = mcolors.ListedColormap(["lightgreen", "red"])
    norm = mcolors.BoundaryNorm(boundaries=[-0.5, 0.5, 1.5], ncolors=2)

    # Loop through each time step and plot
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

    # Remove unused subplots
    for j in range(len(years), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(title, fontsize=16)
    fig.subplots_adjust(right=0.85, bottom=0.1, left=0.12, top=0.92)

    # Add a vertical colorbar with discrete ticks and binary labels.
    cbar_ax = fig.add_axes([0.87, 0.2, 0.02, 0.6])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="vertical", ticks=[0, 1])
    cbar.ax.set_yticklabels(["No concern", "Concern"], fontsize=16)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {save_path}", flush=True)


"""
The formula:  Anomaly=(Data−Climatology)/Climatology * 100, as a percentage
represents the relative anomaly or percentage deviation from the climatology. 
It quantifies how much a given data point (e.g., yearly or forecasted yield) deviates from the long-term mean (climatology) as a fraction of that mean.

A result of -10% means the data is 10% lower than the climatology.
A result of +20% means the data is 20% higher than the climatology

> Identify areas where yields are significantly higher or lower than the long-term average.
> Compare deviations across regions independently of absolute yield values.

The 5th percentile shows an absolute threshold below which only 5% of historical yields fall.
The relative anomaly tells you how much each forecast differs from the mean, allowing for continuous variation.

"""


def relative_anomaly(ref, forecast, output_dir, model, realiz, ssp):
    ref = ref.load()
    forecast = forecast.load()
    forecast = forecast.assign_coords(lat=ref.lat)

    # Compute the long-term mean (climatology) for TWSO across all years
    climatology = ref.TWSO.mean(dim='time')

    # Anomaly (forecasted - climatology)
    # Calculate anomaly (forecasted yield minus historical climatology)
    anomaly = forecast.TWSO - climatology

    # Select only negative anomalies (yield < climatology)
    negative_anomalies = anomaly.where(anomaly < 0)

    # Compute percentage reduction (anomaly relative to climatology)
    percentage_loss = (negative_anomalies / climatology) * 100

    # Select grid cells where percentage loss is ≤ -5%
    severe_loss_perc = percentage_loss.where(percentage_loss <= -5)

    severe_loss_perc_cropped = subset_dataset(severe_loss_perc)

    if isinstance(severe_loss_perc_cropped, xr.DataArray):
        severe_loss_perc_cropped = severe_loss_perc_cropped.to_dataset(name='TWSO')

    save_path = os.path.join(os.sep, output_dir, f'{model}_{ssp}_{realiz}_TWSO_relative_anomaly.png')

    if not os.path.exists(save_path):
        title = f'{model} {ssp} {realiz} TWSO relative anomaly (% loss) maps'
        plot36(severe_loss_perc_cropped, title, save_path)


def relative_anomaly_binary(ref, forecast, output_dir, model, realiz, ssp):
    ref = ref.load()
    forecast = forecast.load()
    forecast = forecast.assign_coords(lat=ref.lat)

    # Compute the long-term mean (climatology) for TWSO across all years
    climatology = ref.TWSO.mean(dim='time')

    # Calculate anomaly (forecasted yield minus historical climatology)
    anomaly = forecast.TWSO - climatology

    # Select only negative anomalies (yield < climatology)
    negative_anomalies = anomaly.where(anomaly < 0)

    # Compute percentage reduction (anomaly relative to climatology)
    percentage_loss = (negative_anomalies / climatology) * 100

    # Select grid cells where percentage loss is ≤ -5%
    severe_loss_perc = percentage_loss.where(percentage_loss <= -5)

    # Create a binary output:
    # For cells where both forecast and climatology are valid,
    # assign 1 if a severe loss (i.e. severe_loss_perc is not NaN) is present, otherwise 0.
    # For cells that are originally missing in forecast or climatology, keep as NaN.
    valid = forecast.TWSO.notnull() & climatology.notnull()
    binary_output = valid.where(valid, other=False)  # valid is a boolean mask
    binary_output = valid.where(valid, other=False)  # not strictly needed; just emphasizing
    binary_output = (
            valid * 1  # convert valid mask to numeric (1 for True, 0 for False)
    )  # this step is just to show valid cells as 1, but we overwrite below

    # Now, for cells that are valid, set to 1 if severe_loss_perc is not NaN, else 0.
    # Note: xr.where(condition, x, y) will keep NaNs where condition is False if they are already NaN.
    binary_output = valid.where(
        valid, other=None
    )  # ensure that we are only working on valid cells
    binary_output = (
        xr.where(valid, xr.where(severe_loss_perc.notnull(), 1, 0), np.nan)
    )

    # crop the dataset using the subset_dataset function.
    severe_loss_perc_cropped = subset_dataset(binary_output)

    if isinstance(severe_loss_perc_cropped, xr.DataArray):
        severe_loss_perc_cropped = severe_loss_perc_cropped.to_dataset(name='TWSO')

    save_path = os.path.join(os.sep, output_dir, f'{model}_{ssp}_{realiz}_TWSO_AoC.png')

    # if not os.path.exists(save_path):
    title = f'{model} {ssp} {realiz} TWSO Areas of Concern'
    plot36_bin(severe_loss_perc_cropped, title, save_path)

    save_nc_flpth = save_path.split('.png')[0] + '.nc'
    severe_loss_perc_cropped.to_netcdf(save_nc_flpth)


"""
Selecting grid cells where the yield is reduced by at least 5% (relative to climatology) is not the same as working with the 5th percentile. 
These are two different statistical concepts:

Percentage Loss (Relative Anomaly Method):
It selects grid points where the forecasted yield is at most 95% of the climatological average (i.e., ≤ 5% reduction).
This is a threshold-based approach and does not depend on the statistical distribution of values.

5th Percentile (Statistical Distribution Method):

Instead of a fixed percentage reduction, the 5th percentile represents the value below which only 5% of the data falls (statistical threshold).
This method requires computing the empirical or climatological distribution and identifying the value corresponding to the 5th percentile.
This is useful when assessing extremes relative to the dataset's variability.
"""


def percentile_thres(ref, forecast, output_dir, model, realiz, ssp):
    """
    Map of the 5th Percentile (Statistical Yield Threshold) : extreme loss
    This results in a dataset where only the extreme loss regions (forecasted yield below the 5th percentile) are shown.
    :param ref:
    :param forecast:
    :param output_dir:
    :param model:
    :param realiz:
    :param ssp:
    :return:
    """
    ref = ref.load()
    forecast = forecast.load()
    forecast = forecast.assign_coords(lat=ref.lat)

    # Compute the 5th percentile of the climatology at each grid cell
    percentile_5th = ref.quantile(0.05, dim="time")  # 5th percentile over 31 years
    percentile_5th = percentile_5th.load()

    # Identify grid cells where the forecast is below the 5th percentile
    below_5th_percentile = forecast.where(forecast <= percentile_5th)
    # The result contains values where the forecasted yield is in the bottom 5%

    below_5th_percentile_cropped = subset_dataset(below_5th_percentile)

    save_path = os.path.join(os.sep, output_dir, f'{model}_{ssp}_{realiz}_TWSO_5thPerc.png')

    if not os.path.exists(save_path):
        title = f'{model} {realiz} {ssp} TWSO 5th percentile extreme loss maps'
        plot36(below_5th_percentile_cropped, title, save_path)


# Define the base directory and the specific model you want to process.
base_dir = f"/path/to/PREDICTIONS_CMIP"

models = ['cnrm-cm6-1-HR',
          'gfdl-esm4',
          'ec-earth3',
          'mpi-esm1-2-hr',
          'noresm2-mm',
          'hadgem3-gc31-mm']

for model in models:
    model_dir = os.path.join(base_dir, model)

    # Name of the CSV file to be created in the model directory.
    output_csv = os.path.join(model_dir, f"{model}_AoC_list.csv")

    # Prepare a list to hold each row's data.
    rows = []

    # List all subdirectories in the model directory; these are your realization folders.
    realization_dirs = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
    for realization in realization_dirs:
        realization_path = os.path.join(model_dir, realization)

        # Look for the historical netCDF file (must contain "1993-2014").
        hist_dir = os.path.join(realization_path, "YEARLY_historical")
        hist_files = glob.glob(os.path.join(hist_dir, "*1993-2014*.nc"))
        hist_file = hist_files[0] if hist_files else ""

        # For ssp, check directories starting with "YEARLY_ssp"
        ssp_file = ""
        ssp_dirs = [d for d in os.listdir(realization_path) if d.startswith("YEARLY_ssp")]
        for ssp_sub in ssp_dirs:
            ssp_dir = os.path.join(realization_path, ssp_sub)
            files = glob.glob(os.path.join(ssp_dir, "*2015-2050*.nc"))
            if files:
                ssp_file = files[0]
                # Extract the ssp value from the directory name.
                ssp_value = ssp_sub.replace("YEARLY_", "")
                rows.append([model, realization, ssp_value, hist_file, ssp_file])

    # Write the results to a CSV file.
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "realization", "ssp", "historical_filepath", "ssp_filepath"])
        writer.writerows(rows)

    print(f"CSV file created at: {output_csv}")

    """ Then read the csv and run the AoC functions """
    csv_dir = os.path.dirname(output_csv)
    output_dir = os.path.join(csv_dir, "AoC")
    os.makedirs(output_dir, exist_ok=True)

    # Read the CSV into a DataFrame
    df = pd.read_csv(output_csv)

    # Iterate over each row in the CSV file
    for index, row in df.iterrows():
        model = row['model']
        realiz = row['realization']
        ssp = row['ssp']
        hist_file = row['historical_filepath']
        ssp_file = row['ssp_filepath']

        # Open the datasets using xarray
        ref = xr.open_dataset(hist_file)
        forec = xr.open_dataset(ssp_file)

        # Call your functions with the data and parameters from the CSV row
        print(f"Processing model: {model}, realization: {realiz}, ssp: {ssp}")
        # relative_anomaly(ref, forec, output_dir, model, realiz, ssp)
        relative_anomaly_binary(ref, forec, output_dir, model, realiz, ssp)
        # percentile_thres(ref, forec, output_dir, model, realiz, ssp)

""" PRINT THE AoC in 10y averages """


def plot_decadal_binary_bc_only(bc, save_path, start_year=2015, end_year=2050):
    """
    Plots bias-corrected binary AoC maps for each decade in a 2x2 layout with reduced spacing.

    :param bc: xarray.DataArray or Dataset; Binary (0/1, with NaNs) AoC maps indexed by a 'time' (or 'decade') coord.
    :param save_path: str; Where to write the PNG.
    :param start_year: int; The real‐world start year, used to label each 10‑year bin.
    :param end_year: int; The real‐world end year, used to label each 10‑year bin.
    :return:
    """
    # Extract DataArray if they passed in a Dataset
    if isinstance(bc, xr.Dataset):
        bc = bc[list(bc.data_vars)[0]]

    # Number of decade‑bins
    n = bc.sizes['decade']

    # Calculate rows and columns for 2x2 layout
    ncols = 2
    nrows = int(np.ceil(n / ncols))

    # Create figure + axes with tighter vertical spacing
    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(18, 4 * nrows),  # Adjusted height for tighter spacing
        subplot_kw={"projection": proj},
        gridspec_kw={"hspace": 0.1, "wspace": 0.05},  # Reduced hspace
    )

    axes = axes.flatten()

    # Get extent from your lat/lon
    lats = bc.lat.values
    lons = bc.lon.values
    if lats[0] > lats[-1]:
        extent = [lons.min(), lons.max(), lats[-1], lats[0]]
    else:
        extent = [lons.min(), lons.max(), lats.min(), lats.max()]

    # Define a 2‑color map for 0/1
    cmap = mcolors.ListedColormap(["lightgreen", "red"])
    norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5], 2)

    fig.suptitle("Decadal AoC", fontsize=18)

    # Loop over each decade‐slice
    for i, t in enumerate(bc.decade.values):
        # Compute the human‐readable label for this bin
        start_decade = start_year + 10 * i
        end_decade = min(start_decade + 9, end_year)
        dec_label = f"{start_decade}–{end_decade}"

        # BC plot
        ax = axes[i]
        ax.set_extent(extent, crs=proj)
        im = ax.pcolormesh(lons, lats,
                           bc.sel(decade=t).values,
                           cmap=cmap, norm=norm,
                           transform=proj)
        ax.set_title(f"{dec_label}", fontsize=16)
        ax.coastlines()
        ax.add_feature(cfeature.LAND, facecolor="lightgray")
        ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
        ax.axis("off")

    # Turn off unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # Tighten the outer margins
    fig.subplots_adjust(top=0.93, bottom=0.05, left=0.05, right=0.9)

    # Add a small, vertical colorbar on the right
    cbar_ax = fig.add_axes([0.92, 0.3, 0.015, 0.4])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="vertical", ticks=[0, 1])
    cbar.ax.set_yticklabels(["No concern", "Concern"], fontsize=12)

    # Save and close
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    # plt.show()

    print(f"Figure saved to {save_path}")


base_dir = f'/path/to/PREDICTIONS_CMIP'
pattern = os.path.join(base_dir, '**', 'AoC', '*.nc')
files = glob.glob(pattern, recursive=True)

for aoc_nc_flpth in files:
    print(f"Processing {aoc_nc_flpth}")

    aoc_bc = xr.open_dataset(aoc_nc_flpth)
    # number of years
    n = aoc_bc.dims['time']

    # create decade‐index 0,0,…,1,1,…,2,2,… etc.
    idx = np.repeat(np.arange(n // 10 + 1), 10)[:n]

    # wrap as DataArray with the same time coordinate
    decade = xr.DataArray(
        idx,
        coords={'time': aoc_bc.time},
        dims=['time'],
        name='decade'
    )

    # now groupby that decade label
    decadal_avg_bc = aoc_bc['TWSO'].groupby(decade).mean(dim='time')

    decadal_binary_bc = xr.where(
        decadal_avg_bc.isnull(),
        np.nan,
        xr.where(decadal_avg_bc >= 0.5, 1, 0)
    )

    output_png = aoc_nc_flpth.replace('.nc', '_decadal.png')

    plot_decadal_binary_bc_only(
        bc=decadal_binary_bc,
        save_path=output_png,
        start_year=2015,
        end_year=2050
    )


def copy_matching_files(directory_x):
    """
    COPY ALL PNG FILES TO AoC DIR
    :param directory_x: the AoC dir to hold all AoC datasets
    :return:
    """
    # Define the destination directory
    destination_folder = os.path.join(directory_x, 'AoC_all')

    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # List to store the paths of the matching files
    matching_files = []

    # Walk through all folders and files in directory_x
    for root, dirs, files in os.walk(directory_x):
        # Skip the AoC folder to avoid re-processing already copied files
        if os.path.abspath(root) == os.path.abspath(destination_folder):
            continue
        for file in files:
            if '_AoC_decadal.png' in file:  # or '_TWSO_5thPerc.png' in file:
                full_path = os.path.join(root, file)
                matching_files.append(full_path)
                # Copy the file to the destination folder
                shutil.copy2(full_path, destination_folder)
                print(f"Copied: {full_path}")

    # Print out all matching file paths
    print("\nList of matching file paths:")
    for path in matching_files:
        print(path)


if __name__ == '__main__':
    directory_x = f'/path/to/PREDICTIONS_CMIP/'
    copy_matching_files(directory_x)
