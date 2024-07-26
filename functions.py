#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import warnings
import xarray as xr
from global_land_mask import globe
import proplot as pplt

# Plotting utility functions


def detrend_dim(da, dim, deg=1):
    """
    Linearly detrend data along a specified dimension.

    Parameters:
    da (xarray.DataArray): The input data array to be detrended.
    dim (str): The dimension along which to detrend the data.
    deg (int): The degree of the polynomial fit. Default is 1 (linear detrending).

    Returns:
    xarray.DataArray: The detrended data array.

    Adapted from: https://gist.github.com/rabernat/1ea82bb067c3273a6166d1b1f77d490f
    """
    # Perform polynomial fit along the specified dimension
    p = da.polyfit(dim=dim, deg=deg)

    # Evaluate the polynomial fit along the dimension to get the trend
    fit = xr.polyval(da[dim], p.polyfit_coefficients)

    # Subtract the trend from the original data to obtain the detrended data
    detrended_da = da - fit

    # Maintain the same attributes and variable name
    detrended_da.attrs = da.attrs
    detrended_da.name = da.name

    # Add the 'methodology' attribute
    detrended_da.attrs['methodology: detrend'] = f"{deg} degree detrended performed as a function of grid cell"

    return detrended_da


def remove_seasonal_cycle(data):
    """
    Remove the seasonal cycle from a dataset to create monthly anomalies.

    Parameters:
    data (xarray.DataArray or xarray.Dataset): The input data with a 'time' dimension.

    Returns:
    xarray.DataArray or xarray.Dataset: The data with the seasonal cycle removed, resulting in monthly anomalies.
    """
    # Compute the monthly mean climatology
    monthly_climatology = data.groupby('time.month').mean()

    # Compute the monthly anomalies by subtracting the monthly climatology
    data_anom = data.groupby('time.month') - monthly_climatology

    # Maintain the same attributes and variable name
    data_anom.attrs = data.attrs
    data_anom.name = data.name

    # Add the 'methodology' attribute
    data_anom.attrs['methodology: anomalies'] = "Seasonal cycle removed to create monthly anomalies by using xr.DataArray.groupby('time.month')"

    return data_anom


def generate_subplot(fig, loc, region, latlabels='l', lonlabels='b', proj='aeqd'):
    """
    Generates a subplot with specified projection and region boundaries.

    Parameters:
    loc (tuple): Location of the subplot in the figure (e.g., (1, 1, 1) for a single subplot).
    region (dict): Dictionary specifying the region boundaries with keys 'lonw', 'lone', 'lats', 'latn'.
    latlabels (str, optional): Position of latitude labels ('l' for left, 'r' for right, 'b' for both). Default is 'l'.
    lonlabels (str, optional): Position of longitude labels ('t' for top, 'b' for bottom, 'both' for both). Default is 'b'.
    proj (str, optional): Projection type for the plot. Default is 'aeqd' (azimuthal equidistant).

    Returns:
    ax: Axis object for the subplot.
    """
    ax = fig.subplot(loc, proj=proj, coast=True, labels=True, lonlabels=lonlabels, latlabels=latlabels,
                     land=True, landcolor='white', abc=True, abcloc='upper left',
                     latlines=10, lonlines=20, gridminor=True, latminorlines=5, lonminorlines=10,
                     lonlim=(region['lonw'] + 5, region['lone'] - 5), latlim=(region['lats'] + 5, region['latn'] - 5),
                     proj_kw={'central_longitude': (region['lonw'] + region['lone']) / 2},
                     )
    return ax


def stand(da):
    """
    Standardizes the input data array by removing the mean and dividing by the standard deviation along the time dimension.

    Parameters:
    da (xarray.DataArray): The data array to be standardized.

    Returns:
    xarray.DataArray: Standardized data array.
    """
    return (da - da.mean('time')) / da.std('time')


def lag_linregress_3D(x, y):
    """
    Performs linear regression between two data arrays along the time dimension.

    Parameters:
    x (xarray.DataArray): Independent variable with the first dimension being time. Can be multi-dimensional.
    y (xarray.DataArray): Dependent variable with the first dimension being time. Can be multi-dimensional.

    Returns:
    tuple: Covariance, correlation, regression slope, and intercept arrays.

    Adapted from: http://hrishichandanpurkar.blogspot.com/2017/09/vectorized-functions-for-correlation.html
    """
    # Suppress warnings that may arise from operations involving NaNs or other runtime issues
    # This ensures that warnings do not interrupt the function execution and clutter the output
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        # Compute data length, mean, and standard deviation along the time axis for further use
        n = np.shape(x)[0]
        xmean = np.nanmean(x, axis=0)
        ymean = np.nanmean(y, axis=0)
        xstd = np.nanstd(x, axis=0)
        ystd = np.nanstd(y, axis=0)

        # Compute covariance along the time axis
        cov = np.nansum((x - xmean) * (y - ymean), axis=0) / n

        # Compute correlation along the time axis
        cor = cov / (xstd * ystd)

        # Compute regression slope and intercept
        slope = cov / (xstd ** 2)
        intercept = ymean - xmean * slope

    return cov, cor, slope, intercept


def land_mask(da):
    """
    Creates a land mask for the given data array based on ocean and land grid cells.
    Assumes that the longitude coordinates are on a -180 to 180 system (not 0 to 360).

    Parameters:
    da (xarray.DataArray): Data array for which the land mask is to be generated with the first dimension being time.

    Returns:
    numpy.array: Boolean array where True indicates ocean and False indicates land.
    """
    data = da[0].copy(deep=True)
    data.coords['lon'] = (data.coords['lon'] + 180) % 360 - 180
    lon_grid, lat_grid = np.meshgrid(data['lon'], data['lat'])
    return globe.is_ocean(lat_grid, lon_grid)


def high_pass_filter(data, window_size, output_path=None, output_filename=None, trackProgress=False):
    """
    Apply a high-pass spatial filter to an atmospheric variable.

    Parameters:
    data (xarray.DataArray): The input atmospheric data.
    window_size (int): The size of the moving window for the high-pass filter.
    output_path (str, optional): The path to save the output NetCDF file. If None, the data is not saved.
    output_filename (str, optional): The name of the output NetCDF file. Required if output_path is provided.
    trackProgress (str, optional): Pass True to this variable to output the current time step while iterating to track progress.

    Returns:
    xarray.DataArray: The high-pass filtered data.
    """
    # Ensure the window size is an odd integer for proper window centering
    if window_size % 2 == 0:
        window_size += 1

    # Initialize the output array
    HPfiltWeight = data.copy(deep=True)
    weighted_avg = np.zeros_like(data)

    # Apply the spatial filter to each time step
    for k, timeVal in enumerate(data['time'].values):

        if trackProgress == True:  # Optionally track progress by outputting the time step
            print(timeVal)

        # Temporarily store the spatial values for each time step, as well as the latitude weights and array dimensions
        data_values = data[k].values
        latitudes = data.lat.values
        weights = np.cos(np.deg2rad(latitudes))
        lat_size, lon_size = data_values.shape
        half_window = window_size // 2

        # Iterate over each grid point, avoiding edges
        for i, ival in enumerate(range(half_window, lat_size - half_window)):
            for j, jval in enumerate(range(lon_size)):
                # Determine the window bounds
                lat_min = ival - half_window
                lat_max = ival + half_window + 1
                lon_min = jval - half_window
                lon_max = jval + half_window + 1

                # Select the data in the window and shape the weights to the correct size
                # Handle wrapping around for longitude
                if lon_min < 0:
                    lon_min_wrap = lon_min + lon_size
                    window_lon = np.concatenate((data_values[lat_min:lat_max, lon_min_wrap:], data_values[lat_min:lat_max, :lon_max]), axis=1)
                elif lon_max > lon_size:
                    lon_max_wrap = lon_max - lon_size
                    window_lon = np.concatenate((data_values[lat_min:lat_max, lon_min:], data_values[lat_min:lat_max, :lon_max_wrap]), axis=1)
                else:
                    window_lon = data_values[lat_min:lat_max, lon_min:lon_max]
                weights_window = np.repeat(weights[lat_min:lat_max, np.newaxis], window_size, axis=1)

                # Compute the weighted average
                weighted_avg[k, ival, jval] = np.average(window_lon, weights=weights_window)

    # Subtract low-pass filtered response from data to get high-pass filtered data
    HPfiltWeight = data - weighted_avg

    # Set all boundary latitude values to zero as the filter does not reach them
    HPfiltWeight[:, :half_window+1, :] = 0
    HPfiltWeight[:, -half_window:, :] = 0

    # Update attributes, importing original attributes and modifying as needed
    HPfiltWeight.attrs.update(data.attrs)
    HPfiltWeight.attrs['methodology: spatial filter'] = ("1000km high-pass spatial filter applied using a moving average as a function of grid cell.")
    if output_path and output_filename:
        # Save the result to a NetCDF file
        output_filepath = output_path + output_filename
        print('saving: ' + output_filepath)
        HPfiltWeight.to_netcdf(output_filepath)
        print('save complete')

    return HPfiltWeight

