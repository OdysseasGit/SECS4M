"""
Script to feed the SEAS5.1 data to SECS

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

import xarray as xr
import os
import numpy as np
import pandas as pd
import tensorflow as tf


lsm_dir = '/path/to/land-sea mask directory'
rnn_dir = '/path/to/parentRNNdir'

month_start = 5
days_to_emulate = 240

hindcast = False  # False if Forecast years and data

if hindcast:
    ens_numbers = np.arange(1, 25 + 1)
    years = np.arange(1993, 2015 + 1, 1)
    dir_name = 'SEAS5_HINDCAST'
    seas_dir = '/path/to/SEAS5/SEAS5_HINDCAST/'

else:
    ens_numbers = np.arange(1, 51 + 1)
    years = np.arange(2017, 2023 + 1, 1)  # 2016 IS NOT THERE DUE TO DAKI MISSING IT
    dir_name = 'SEAS5_FORECAST'
    seas_dir = '/path/to/SEAS5/SEAS5_FORECAST/'

for year in years:
    for ens_number in ens_numbers:
        ens_number_str = f'{ens_number:02d}'  # Format with leading zeros

        predictions_dir = os.path.join(os.sep, rnn_dir, 'PREDICTIONS_SEAS', dir_name, f'ens_{ens_number_str}')
        os.makedirs(predictions_dir, exist_ok=True)

        predictions_with_seas_flpth = os.path.join(os.sep, predictions_dir,
                                                   f'{year}_ens{ens_number_str}_TWSO_predicted_withSEAS.nc')

        ens_dir = os.path.join(seas_dir, f'ens_{ens_number_str}')

        seas_tasmin_flpth = os.path.join(os.sep, ens_dir, f'ecmwf_{ens_number_str}_tn_{year}_00.nc')
        seas_tasmax_flpth = os.path.join(os.sep, ens_dir, f'ecmwf_{ens_number_str}_tx_{year}_00.nc')
        seas_pr_flpth = os.path.join(os.sep, ens_dir, f'ecmwf_{ens_number_str}_pr_{year}_00.nc')

        if not os.path.exists(predictions_with_seas_flpth):
            print('Working on predictions with SEAS5: ', predictions_with_seas_flpth)

            seas_tasmin = xr.open_dataset(seas_tasmin_flpth)
            seas_tasmax = xr.open_dataset(seas_tasmax_flpth)
            seas_pr = xr.open_dataset(seas_pr_flpth)

            # addfeatures_flag = True # add or remove features GDD, LAT, soil

            ''' CONSTRUCT THE BINARY LAND SEA MASK '''

            lsm_flpth = os.path.join(os.sep, lsm_dir, 'lsm_file.nc')
            lsm_binary_flpth = lsm_flpth.split('.nc')[0] + '_binary.nc'

            # the original lsm is not binary, it has proportions of land and sea with values between 0 and 1. Thus I convert it to binary with a threshold of 0.5
            if not os.path.exists(lsm_binary_flpth):
                lsm_ds = xr.open_dataset(lsm_flpth)
                lsm = lsm_ds['lsm']
                lsm_binary = xr.where(lsm >= 0.5, 1, 0)
                lsm_binary.to_netcdf(lsm_binary_flpth)

            import cdo

            lsm_binary_remaped_flpth = os.path.join(os.sep, lsm_dir, 'lsm_binary_remaped_seas.nc')
            cdo = cdo.Cdo()

            if not os.path.exists(lsm_binary_remaped_flpth):
                cdo.remapnn(seas_tasmin_flpth, input=lsm_binary_flpth, output=lsm_binary_remaped_flpth)

            ''' END OF CONSTRUCT THE BINARY LAND SEA MASK '''

            ''' Construct the full weather dataset '''
            seas_full = xr.merge([seas_tasmax, seas_tasmin, seas_pr])
            # I will remove the bnds dimension because it is too muc of a problem with DOY, as it inherits it, and after all it is really somewhat useless.
            if 'bnds' in seas_full.dims:
                seas_full = seas_full.drop_dims('bnds')

            
            seas_full['pr'] = seas_full['pr'] / 10
            seas_full['pr'] = seas_full['pr'].cumsum(dim='time')  # make it rain_total

            ''' Calculate the lag features '''


            # Define function to add lags
            def add_lags(dataset, lag_days=5):
                ds_lagged = dataset.copy()
                for var in dataset.data_vars:
                    for lag in range(1, lag_days + 1):
                        lag_name = f"{var}_lag_{lag}"
                        ds_lagged[lag_name] = dataset[var].shift(time=lag)
                return ds_lagged


            seas_full_lagged = add_lags(seas_full)

            """ THIS IS THE NEW Verify lagged features SANITY CHECK """


            def verify_lags(original_ds, lagged_ds, day_index=1, lag_days=5):
                """
                Sanity-check that lagged_ds[var_lag_k] at time (day_index + k)
                has the same data values as original_ds[var] at time day_index.
                Ignores any coordinate mismatches.
                """
                n = original_ds.sizes['time']
                base_date = original_ds.time[day_index].values
                print(f"Verifying lags for base date: {base_date!r} (index {day_index})\n")

                for var in original_ds.data_vars:
                    # extract the “true” data values at the base date
                    true_vals = original_ds[var].sel(time=base_date).values

                    for k in range(1, lag_days + 1):
                        idx = day_index + k
                        lag_name = f"{var}_lag_{k}"

                        if idx < n:
                            test_date = original_ds.time[idx].values
                            # extract the lagged data values at that later date
                            lagged_vals = lagged_ds[lag_name].sel(time=test_date).values

                            try:
                                # compare raw NumPy arrays (ignores coords entirely)
                                np.testing.assert_allclose(
                                    lagged_vals,
                                    true_vals,
                                    err_msg=(
                                        f"Mismatch in {lag_name} on {test_date} "
                                        f"(vs {var} on {base_date})"
                                    )
                                )
                                print(f"✔  {lag_name} at {test_date!r} matches data values of {var} at {base_date!r}")
                            except AssertionError as e:
                                raise AssertionError(f"❌  {e}")
                        else:
                            print(f"–   No data for {lag_name}: index {idx} out of range (n={n})")


            verify_lags(seas_full, seas_full_lagged, day_index=0, lag_days=5)

            ''' Cut the dates to the proper range: len(time)-240: len(time) '''
            time = seas_full_lagged['time']

            # Find the year for the current dataset (assuming the data spans a single year)
            year = pd.Timestamp(time[-1].values).year  # Get the year from the last time step
            # Define June 1st (DOY 152) as the start date
            start_date = pd.Timestamp(year=year, month=month_start, day=1)
            # Define the end date as 240 days after June 1st
            end_date = start_date + pd.Timedelta(days=days_to_emulate)
            # Slice the dataset using the calculated start and end date
            seas_full_lagged_240d = seas_full_lagged.sel(time=slice(start_date, end_date))

            print(seas_full_lagged_240d)

            ''' Calculate the DOY for each time point and create a DataArray '''
            doy = pd.to_datetime(seas_full_lagged_240d['time'].values).dayofyear
            doy_1d = xr.DataArray(doy, coords=[seas_full_lagged_240d['time']], dims=['time'], name='doy')

            # Now expand the DOY DataArray to include lat and lon dimensions
            # Use the broadcasting mechanism within xarray
            doy_broadcasted = doy_1d.broadcast_like(seas_full_lagged_240d)
            doy_broadcasted = doy_broadcasted.transpose('time', 'lat', 'lon')
            doy_broadcasted = doy_broadcasted - np.min(doy_broadcasted) + 1  # start it from day1 to 240

            ''' Add the DOY variable to the dataset '''
            seas_full_lagged_240d['doy'] = doy_broadcasted

            ''' Transpose the featue space as is for the training of the model
            TEMP_MAX, TEMP_MIN, RAINT, TEMP_MAX_lag1, TEMP_MIN_lag1, RAINT_lag1, TEMP_MAX_lag2, TEMP_MIN_lag2, RAINT_lag2, ..., DOY
            '''
            var_order = ['tx', 'tn', 'pr',
                         'tx_lag_1', 'tn_lag_1', 'pr_lag_1',
                         'tx_lag_2', 'tn_lag_2', 'pr_lag_2',
                         'tx_lag_3', 'tn_lag_3', 'pr_lag_3',
                         'tx_lag_4', 'tn_lag_4', 'pr_lag_4',
                         'tx_lag_5', 'tn_lag_5', 'pr_lag_5', 'doy']

            seas_full_lagged_240d_reordered = seas_full_lagged_240d[var_order]

            ''' Convert to numpy array '''
            # Convert the Dataset to a dictionary of numpy arrays
            dict_of_arrays = {}
            for var in seas_full_lagged_240d_reordered.data_vars:
                # print(f"Processing variable: {var}")  # Print the variable name for checking order
                dict_of_arrays[var] = seas_full_lagged_240d_reordered[var].values
            # stack the variables along a new axis
            np_reordered = np.stack([dict_of_arrays[var] for var in seas_full_lagged_240d_reordered.data_vars], axis=-1)
            print('Combined, reordered array shape: ', np_reordered.shape)


            def reshape_to_batches(array, batch_size):
                """
                Reshape the array into batches along the time dimension.

                Parameters:
                    array (np.ndarray): The input array of shape (time, lat, lon, vars).
                    batch_size (int): The size of each batch along the time dimension.

                Returns:
                    reshaped_array (np.ndarray): The reshaped array.
                """
                original_shape = array.shape
                num_batches = original_shape[0] // batch_size
                reshaped_array = array.reshape(num_batches, batch_size, *original_shape[1:])

                return reshaped_array


            time_shape = np_reordered.shape[0]
            lat_shape = np_reordered.shape[1]
            lon_shape = np_reordered.shape[2]
            vars_shape = np_reordered.shape[3]
            print('Time:', time_shape, ', Lats:', lat_shape, ', Lons:', lon_shape, ', Vars:', vars_shape)

            batch_size = 6

            np_reordered_batched = reshape_to_batches(np_reordered, batch_size)
            print(f"Batched array np_reordered_batched shape: {np_reordered_batched.shape}")

            ''' SANITY CHECKS: Verify batches 
            This will print True if the reshaping has been done correctly with consecutive 6-day batches.'''
            # Check the first batch (first 6 days)
            print(np.array_equal(np_reordered[0:6], np_reordered_batched[0]))
            # Check the second batch (next 6 days)
            print(np.array_equal(np_reordered[6:12], np_reordered_batched[1]))

            ''' then reshape it back the way it was and confirm it is ok '''
            reshaped_back_array = np_reordered_batched.reshape(np_reordered.shape)
            print(
                f"Is the reshaped-back array equal to the original array? {np.array_equal(np_reordered, reshaped_back_array)}")


            def flatten_lat_lon(array):
                """
                Flatten the lat and lon dimensions of the array.

                Parameters:
                    array (np.ndarray): The input array with shape (any, any, lat, lon, vars).

                Returns:
                    flattened_array (np.ndarray): The flattened array.
                    original_shape (tuple): The original shape of the input array.
                """
                original_shape = array.shape
                new_shape = (
                original_shape[0], original_shape[1], original_shape[2] * original_shape[3], original_shape[4])
                flattened_array = array.reshape(new_shape)

                return flattened_array


            np_reordered_batched_flat = flatten_lat_lon(np_reordered_batched)
            print('np_reordered_batched_flat BEFORE changing axes: ', np_reordered_batched_flat.shape)
            np_reordered_batched_flat = np.moveaxis(np_reordered_batched_flat, 2, 0)

            print(f"Flattened array shape: {np_reordered_batched_flat.shape}")

            # Reshape back to original order
            np_reordered_batched_flat_back = np.moveaxis(np_reordered_batched_flat, 0, 2)
            print('np_reordered_batched_flat AFTER changing back axes: ', np_reordered_batched_flat_back.shape)

            reshaped_back_array = np_reordered_batched_flat_back.reshape(np_reordered_batched.shape)
            print(f"Reshaped back to original shape: {reshaped_back_array.shape}")

            # Check if the reshaped-back array is the same as the original
            print(
                f"Is the reshaped-back array equal to the original array? {np.array_equal(np_reordered_batched, reshaped_back_array)}")

            ''' Load the RNN model '''
            def find_rnn_folder(directory):
                # List all items in the directory
                folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]

                # Find the folder that contains 'RNN'
                for folder in folders:
                    if 'RNN' in folder:
                        return os.path.join(directory, folder)

                return None


            model = tf.keras.models.load_model(find_rnn_folder(rnn_dir))
            predictions = model.predict(np_reordered_batched_flat)

            print('Original predictions shape:', predictions.shape)
            print('Max value of the predictions for sanity checking: ', np.max(predictions))

            ''' The predicted array of shape (60760, 35, 1) needs to be transformed to (60760, 35, 6, 1) 
            by copying the values as if they were 6 day batches of the same value '''


            def expand_to_batches(array, new_dim_size):
                """
                Expand the array to include a new dimension and replicate the values.

                Parameters:
                    array (np.ndarray): The input array with shape (samples, time, 1).
                    new_dim_size (int): The size of the new dimension to expand into.

                Returns:
                    expanded_array (np.ndarray): The expanded array with shape (samples, time, new_dim_size, 1).
                """
                # Get the original shape
                original_shape = array.shape
                # Replicate the values along the new dimension
                expanded_array = np.repeat(array[:, :, np.newaxis], new_dim_size, axis=2)
                return expanded_array


            # Expand to (60760, 35, 6, 1)
            batch_size = 6
            predictions_batched = expand_to_batches(predictions, batch_size)

            print(f"Original shape of predictions: {predictions.shape}")
            print(f"Expanded predictions array shape: {predictions_batched.shape}")


            def sanity_check(array, new_dim_size):
                """
                Perform a sanity check on the transformation.

                Parameters:
                    array (np.ndarray): The original array with shape (samples, time, 1).
                    new_dim_size (int): The size of the new dimension.

                Returns:
                    bool: True if the sanity check passes, False otherwise.
                """
                expanded_array = expand_to_batches(array, new_dim_size)

                # Check the shape
                if expanded_array.shape != (array.shape[0], array.shape[1], new_dim_size, 1):
                    return False

                # Check if all elements in the new dimension are the same
                for i in range(expanded_array.shape[0]):
                    for j in range(expanded_array.shape[1]):
                        if not np.all(expanded_array[i, j, :, 0] == array[i, j, 0]):
                            return False

                return True


            # Set the new dimension size (6-day batches)
            new_dim_size = 6

            
            # Then a sanity check for the actual predictions
            pred_sanity_check_passed = sanity_check(predictions, new_dim_size)
            print(
                f"Sanity check if the batches are generated correctly with copied data for the actual predictions: {pred_sanity_check_passed}")

            '''
            np_reordered_batched_flat AFTER changing back axes:  (35, 6, 60760, 19)
            Reshaped back to original shape: (35, 6, 196, 310, 19)
            '''
            predictions_batched_latlon_shape = (
            np_reordered_batched.shape[0], np_reordered_batched.shape[1], np_reordered_batched.shape[2],
            np_reordered_batched.shape[3], 1)
            print(predictions_batched_latlon_shape)

            predictions_batched_shape = predictions_batched.shape
            print('Current shape of the data: ', predictions_batched_shape)
            # first I need to change the predictions_batched shape which is (60760, 35, 6, 1) to the shape (35, 6, 60760, 19)
            predictions_batched_reordered = np.moveaxis(predictions_batched, 0, 2)
            print('Reordered shape of the data: ', predictions_batched_reordered.shape)
            # Then I need to transform it into the following shape:
            predictions_batched_latlon_shape = (
            np_reordered_batched.shape[0], np_reordered_batched.shape[1], np_reordered_batched.shape[2],
            np_reordered_batched.shape[3], 1)
            print('Target shape for the data: ', predictions_batched_latlon_shape)

            predictions_batched_latlon = predictions_batched_reordered.reshape(predictions_batched_latlon_shape)
            print('Lat/Lon predictions shape:', predictions_batched_latlon.shape)

            # Then I need to reshape the data to the final (240, 196, 310, 1) shape
            predictions_final_shape = (np_reordered.shape[0], np_reordered.shape[1], np_reordered.shape[2], 1)
            print('Final target shape: ', predictions_final_shape)

            predictions_final = predictions_batched_latlon.reshape(predictions_final_shape)
            print('Final predictions shape:', predictions_final.shape)

            pred_xr = xr.DataArray(
                predictions_final.squeeze(),  # Remove the singleton dimension
                dims=['time', 'lat', 'lon'],
                coords={
                    'time': seas_full_lagged_240d.coords['time'],
                    'lat': seas_full_lagged_240d.coords['lat'],
                    'lon': seas_full_lagged_240d.coords['lon']
                },
                name=str(year) + 'TWSO_predicted'
            )

            ''' Mask out the sea '''
            lsm_binary_remaped = xr.open_dataset(lsm_binary_remaped_flpth)

            if 'bnds' in lsm_binary_remaped.dims:
                lsm_binary_remaped = lsm_binary_remaped.drop_dims('bnds')

            if 'time' in lsm_binary_remaped.dims:
                lsm_binary_remaped = lsm_binary_remaped.isel(time=0).drop('time')

            # Ensure the mask has the same lat/lon dimensions as the dataset
            assert np.all(np.equal(lsm_binary_remaped.lat, pred_xr.lat))
            assert np.all(np.equal(lsm_binary_remaped.lon, pred_xr.lon))

            # Expand the mask to match the dataset's time dimension
            lsm_binary_remaped_br = lsm_binary_remaped.expand_dims(time=pred_xr.time)
            assert np.all(np.equal(lsm_binary_remaped_br.time, pred_xr.time))

            lsm_binary_remaped_br = lsm_binary_remaped_br.transpose('time', 'lat', 'lon')

            pred_xr_masked = pred_xr.where(lsm_binary_remaped_br == 1)
            print('Sanity check for maximum value of masked predictions: ', np.max(pred_xr_masked.lsm))

            # If pred_xr_masked is a Dataset, set the correct variable name
            if isinstance(pred_xr_masked, xr.Dataset):
                pred_xr_masked = pred_xr_masked.rename({list(pred_xr_masked.data_vars)[0]: f'TWSO_{year}'})
            elif isinstance(pred_xr_masked, xr.DataArray):
                # If it's a DataArray, rename it directly
                pred_xr_masked = pred_xr_masked.rename(f'TWSO_{year}')
            pred_xr_masked.to_netcdf(predictions_with_seas_flpth, engine='netcdf4')
            print('Finished predictions with SEAS5: ', predictions_with_seas_flpth)

        else:
            print('Found existing predictions with SEAS5: ', predictions_with_seas_flpth)




