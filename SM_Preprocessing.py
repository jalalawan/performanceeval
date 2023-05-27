# -*- coding: utf-8 -*-
"""

@author: jawan
"""
import sensortoolkit
import matplotlib as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce
import plotly.express as px
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pytz
from tqdm import tqdm

#REPORTING TIMESERIES PLOTS 

import matplotlib.dates as mdates
import numpy as np
from datetime import timedelta

# Ingest data with specified start_date and end_date
start_date = pd.to_datetime('2019-10-01').tz_localize('UTC')
end_date = pd.to_datetime('2020-10-01').tz_localize('UTC')

dfs = []
for i in range(1, 27):
    filename = f"C:\\Users\\jawan\\Dissertation Data\\Final Datasets\\SM\\LCS_SM_2018_2019\\Sensor{i}_SM.csv"
    try:
        df = pd.read_csv(filename)
        df['created_at'] = pd.to_datetime(df['created_at']).dt.tz_convert('UTC')
        df = df[(df['created_at'] >= start_date) & (df['created_at'] <= end_date)]
        df['Sensor'] = i
        dfs.append(df)
    except FileNotFoundError:
        print(f"File not found: {filename}. Skipping and moving on...")

# Concatenate data from all sensors
data = pd.concat(dfs)

# Group data by sensor and resample daily
data['day'] = data['created_at'].dt.to_period('D')
grouped_data = data.groupby(['Sensor', 'day'])['PM2.5_ATM_ug/m3'].agg(lambda x: (x.notnull().sum() > 0))

# Unstack the grouped data and fill NaN values with False
unstacked_data = grouped_data.unstack(level=0).fillna(False)

# Create bar plot (Gantt chart style)
fig, ax = plt.subplots(figsize=(15, 7), dpi=100)

# Define colors for each sensor (508 compliant color)
colors = plt.cm.get_cmap('viridis', 26)

for i, sensor in enumerate(unstacked_data.columns):
    sensor_data = unstacked_data[sensor].astype(int)
    for j, reporting in enumerate(sensor_data):
        if reporting:
            ax.barh(i, timedelta(days=1), left=unstacked_data.index[j].to_timestamp(), color=colors(i), align='center', height=0.6)

# Set chart labels and title
start_date_str = start_date.strftime('%B %d, %Y')
end_date_str = end_date.strftime('%B %d, %Y')
ax.set_xlabel('Time Reporting')
ax.set_ylabel('Sensors')
ax.set_title(f'Sensor Reporting Data ({start_date_str} to {end_date_str})')

# Set y-axis with sensor names
sensor_names = [f'Sensor {sensor}' for sensor in unstacked_data.columns]
ax.set_yticks(range(len(sensor_names)))
ax.set_yticklabels(sensor_names, fontsize=8)

# Set x-axis with actual dates
date_range = pd.date_range(start_date.to_pydatetime(), end_date.to_pydatetime(), freq='MS')
ax.set_xticks(date_range)
ax.set_xticklabels([date.strftime('%B %Y') for date in date_range], fontsize=8, rotation=45, ha='left')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%B %Y'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=120))
ax.tick_params(axis='x', which='minor', length=5)

# Find the maximum date in the data
max_date = unstacked_data.index.max().to_timestamp()

# Calculate and display the percentage of time reporting next to each sensor bar
total_duration = (end_date - start_date).days
for i, sensor in enumerate(unstacked_data.columns):
    percent_reporting = (unstacked_data[sensor].sum() / total_duration) * 100
    percent_reporting = min(percent_reporting, 100)  # Ensure percent reporting is not greater than 100%
    ax.annotate(f'{percent_reporting:.1f}%', xy=(end_date.to_pydatetime() + timedelta(days=7), i), fontsize=8, va='center')

# Adjust the x-axis limit to include the end date
ax.set_xlim(start_date.to_pydatetime(), end_date.to_pydatetime() + timedelta(days=30))

plt.tight_layout()
plt.show()



#WITH OUTLIERS (Quarter 0)

start_date = pd.to_datetime('2018-10-01').tz_localize('UTC')
end_date = pd.to_datetime('2019-01-01').tz_localize('UTC')

dfs = []
for i in range(1, 27):
    filename = f"C:\\Users\\jawan\\Dissertation Data\\Final Datasets\\SM\\LCS_SM_2018_2019\\Quarter 0\\Sensor{i}_SM.csv"
    try:
        df = pd.read_csv(filename)
        df['created_at'] = pd.to_datetime(df['created_at']).dt.tz_convert('UTC')
        df = df[(df['created_at'] >= start_date) & (df['created_at'] <= end_date)]
        dfs.append(df)
    except FileNotFoundError:
        print(f"File not found: {filename}. Skipping and moving on...")

# Define sensor names
sensor_names = ['Sensor 4', 'Sensor 6', 'Sensor 7', 'Sensor 8', 'Sensor 10', 'Sensor 16', 'Sensor 20', 'Sensor 21']

#Timeseries Plot With Outliers


# Create a plot
fig, axs = plt.subplots(len(sensor_names), 1, figsize=(15, 30), sharex=True)
fig.suptitle('Timeseries PM$_{2.5}$ Readings (Channel A and B) with Outliers', fontsize=20, y=0.9)

for i, sensor_name in enumerate(sensor_names):
    df = dfs[i]
    axs[i].plot(df['created_at'], df['PM2.5_ATM_ug/m3'], label='PM2.5_ATM_ug/m3')
    axs[i].plot(df['created_at'], df['PM2.5_ATM_ug/m3.1'], label='PM2.5_ATM_ug/m3.1')
    axs[i].set_title(sensor_name)
    axs[i].legend()
    axs[i].set_ylabel('PM$_{2.5}$ (µg/m³)')

plt.xlabel('Time')
plt.xticks(rotation=45)
plt.show()




# Create plot
fig, axs = plt.subplots(2, 4, figsize=(15, 15))
fig.suptitle('Intra-Sensor Precision - PM$_{2.5}$ Channel A vs. Channel B', fontsize=20)

# Process each sensor
for i in tqdm(range(min(len(dfs), 24)), desc='Processing sensors'):
    df = dfs[i]
    df.dropna(inplace=True)
    x = df['PM2.5_ATM_ug/m3']
    y = df['PM2.5_ATM_ug/m3.1']

    # Check that x and y have at least one non-missing value
    if len(x) > 0 and len(y) > 0:
        # Fit linear regression model
        model = LinearRegression()
        model.fit(x.values.reshape(-1, 1), y)

        # Generate predictions
        y_pred = model.predict(x.values.reshape(-1, 1))

        # Calculate R-squared and RMSE
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))

        # Plot data and regression line
        row = (i-1) // 4
        col = (i-1) % 4
        ax = axs[row, col]
        ax.scatter(x, y, color='blue', alpha=0.5)
        ax.plot(x, y_pred, color='black', linestyle='dotted')

        # Label subplot with sensor name and statistics
        ax.set_title(sensor_names[i-1], fontsize=12, loc='left')
        sign = "+" if model.intercept_ >= 0 else "-"
        abs_intercept = abs(model.intercept_)
        ax.text(0.05, 0.95, f"y = {model.coef_[0]:.2f}x {sign} {abs_intercept:.2f}\nR-squared = {r2:.2f}\nRMSE = {rmse:.2f}\nN={len(df)}", transform=ax.transAxes, fontsize=10, ha='left', va='top')
        # Shade 95% confidence interval around regression plot
        ci = 1.96 * np.std(y - y_pred) / np.sqrt(len(y))
        ax.fill_between(x, y_pred - ci, y_pred + ci, color='gray', alpha=.6)
        # Set x and y axis limits to be the same
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_xlim(min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))
        ax.set_ylim(min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))
        

# Label overall plot axis
fig.text(0.5, -0.03, 'Particulate Matter (PM$_{2.5}$) Channel A', ha='center', fontsize=16)
fig.text(-0.03, 0.5, 'Particulate Matter (PM$_{2.5}$) Channel B', va='center', rotation='vertical', fontsize=16)

# Show plot
plt.tight_layout()
plt.show()
    
    
     
# Without Outliers (Quarter 0)


#REMOVING OUTLIERS


start_date = pd.to_datetime('2018-10-01').tz_localize('UTC')
end_date = pd.to_datetime('2019-01-01').tz_localize('UTC')

dfs = []
for i in range(1, 27):
    filename = f"C:\\Users\\jawan\\Dissertation Data\\Final Datasets\\SM\\LCS_SM_2018_2019\\Quarter 0\\Sensor{i}_SM.csv"
    try:
        df = pd.read_csv(filename)
        df['created_at'] = pd.to_datetime(df['created_at']).dt.tz_convert('UTC')
        df = df[(df['created_at'] >= start_date) & (df['created_at'] <= end_date)]
        dfs.append(df)
    except FileNotFoundError:
        print(f"File not found: {filename}. Skipping and moving on...")
        
        
# Define sensor names
sensor_names = ['Sensor 4', 'Sensor 6', 'Sensor 7', 'Sensor 8', 'Sensor 10', 'Sensor 16', 'Sensor 20', 'Sensor 21']


#Timeseries Plot Without Outliers


# Create a plot
fig, axs = plt.subplots(len(sensor_names), 1, figsize=(15, 30), sharex=True)
fig.suptitle('Timeseries PM$_{2.5}$ Readings (Channel A and B) without Outliers', fontsize=20, y=0.9)

for i, sensor_name in enumerate(sensor_names):
    df = dfs[i]
    axs[i].plot(df['created_at'], df['PM2.5_ATM_ug/m3'], label='PM2.5_ATM_ug/m3')
    axs[i].plot(df['created_at'], df['PM2.5_ATM_ug/m3.1'], label='PM2.5_ATM_ug/m3.1')
    axs[i].set_title(sensor_name)
    axs[i].legend()
    axs[i].set_ylabel('PM$_{2.5}$ (µg/m³)')

plt.xlabel('Time')
plt.xticks(rotation=45)
plt.show()


# Create plot
fig, axs = plt.subplots(2, 4, figsize=(15, 15))
fig.suptitle('Intra-Sensor Precision (after removing outliers) - PM$_{2.5}$ Channel A vs. Channel B', fontsize=20)


# Process each sensor
for i in tqdm(range(min(len(dfs), 24)), desc='Processing sensors'):
    df = dfs[i]
    df.dropna(inplace=True)
    x = df['PM2.5_ATM_ug/m3']
    y = df['PM2.5_ATM_ug/m3.1']

    # Check that x and y have at least one non-missing value
    if len(x) > 0 and len(y) > 0:
        
        # Identify outliers
        sd_error = np.abs(y - x) / np.std(y - x)
        is_outlier = (np.abs(x - y) >= 5) & (sd_error >= 2)

        # Calculate percentage of outliers
        perc_outliers = np.round(np.sum(is_outlier) / len(df) * 100, 2)

        # Fit linear regression model
        model = LinearRegression()
        model.fit(x[~is_outlier].values.reshape(-1, 1), y[~is_outlier])

        # Generate predictions
        y_pred = model.predict(x.values.reshape(-1, 1))

        # Calculate R-squared and RMSE
        r2 = r2_score(y[~is_outlier], y_pred[~is_outlier])
        rmse = np.sqrt(mean_squared_error(y[~is_outlier], y_pred[~is_outlier]))

        # Plot data and regression line
        row = (i-1) // 4
        col = (i-1) % 4
        ax = axs[row, col]
        ax.scatter(x[is_outlier], y[is_outlier], color='red')
        ax.scatter(x[~is_outlier], y[~is_outlier], color='blue')
        ax.plot(x, y_pred, color='black')
        
        
        # Label subplot with sensor name and statistics
        ax.set_title(sensor_names[i], fontsize=12, loc='left')
        sign = "+" if model.intercept_ >= 0 else "-"
        abs_intercept = abs(model.intercept_)
        ax.text(0.05, 0.95, f"y = {model.coef_[0]:.2f}x {sign} {abs_intercept:.2f}\nR-squared = {r2:.2f}\nRMSE = {rmse:.2f}\nN={len(df)}\nOutliers: {perc_outliers}%", transform=ax.transAxes, fontsize=10, ha='left', va='top')
        # Shade 95% confidence interval around regression plot
        ci = 1.96 * np.std(y[~is_outlier] - y_pred[~is_outlier]) / np.sqrt(len(y[~is_outlier]))
        ax.fill_between(x, y_pred - ci, y_pred + ci, color='gray', alpha=.6)
        # Set x and y axis limits to be the same
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_xlim(min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))
        ax.set_ylim(min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))
        

# Label overall plot axis
fig.text(0.5, -0.03, 'Particulate Matter (PM$_{2.5}$) Channel A', ha='center', fontsize=16)
fig.text(-0.03, 0.5, 'Particulate Matter (PM$_{2.5}$) Channel B', va='center', rotation='vertical', fontsize=16)

# Show plot
plt.tight_layout()
plt.show()


    

# Datasets Final Quarter 0 (Outliers Removed)

import os

start_date = pd.to_datetime('2018-10-01').tz_localize('UTC')
end_date = pd.to_datetime('2019-01-01').tz_localize('UTC')

sensor_indices = [4, 6, 7, 8, 10, 16, 20, 21]
sensor_names = ['Sensor 4', 'Sensor 6', 'Sensor 7', 'Sensor 8', 'Sensor 10', 'Sensor 16', 'Sensor 20', 'Sensor 21']

dfs = []
for i in sensor_indices:
    filename = f"C:\\Users\\jawan\\Dissertation Data\\Final Datasets\\SM\\LCS_SM_2018_2019\\Quarter 0\\Sensor{i}_SM.csv"
    try:
        df = pd.read_csv(filename)
        df['created_at'] = pd.to_datetime(df['created_at']).dt.tz_convert('UTC')
        df = df[(df['created_at'] >= start_date) & (df['created_at'] <= end_date)]
        dfs.append(df)
    except FileNotFoundError:
        print(f"File not found: {filename}. Skipping and moving on...")

dfs_no_outliers = []
for i, sensor_name in zip(range(len(sensor_indices)), sensor_names):
    df = dfs[i].copy()
    df.dropna(inplace=True)
    x = df['PM2.5_ATM_ug/m3']
    y = df['PM2.5_ATM_ug/m3.1']

    if len(x) > 0 and len(y) > 0:
        sd_error = np.abs(y - x) / np.std(y - x)
        is_outlier = (np.abs(x - y) >= 5) & (sd_error >= 2)
        
        df_no_outliers = df[~is_outlier]
        dfs_no_outliers.append(df_no_outliers)

        # Save the new dataframe without outliers as a CSV file
        output_filename = f"C:\\Users\\jawan\\Dissertation Data\\Final Datasets\\SM\\LCS_SM_2018_2019\\Quarter 0\\No_Outliers\\{sensor_name}_SM_no_outliers.csv"
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        df_no_outliers.to_csv(output_filename, index=False)

#1-hr Averaged Datasets

sensor_names = ['Sensor 4', 'Sensor 6', 'Sensor 7', 'Sensor 8', 'Sensor 10', 'Sensor 16', 'Sensor 20', 'Sensor 21']

hourly_dfs = []
# Add a dictionary to store the percentage of non-valid readings by sensor
non_valid_percentages = {}

for sensor_name in sensor_names:
    input_filename = f"C:\\Users\\jawan\\Dissertation Data\\Final Datasets\\SM\\LCS_SM_2018_2019\\Quarter 0\\No_Outliers\\{sensor_name}_SM_no_outliers.csv"
    
    # Read the CSV file
    df = pd.read_csv(input_filename)
    df['created_at'] = pd.to_datetime(df['created_at']).dt.tz_convert('UTC')
    
    # Ensure there are at least four valid readings per hour
    df['valid_reading_PM25_ATM_A'] = df['PM2.5_ATM_ug/m3'].notna().astype(int)
    df['valid_reading_PM25_ATM_B'] = df['PM2.5_ATM_ug/m3.1'].notna().astype(int)
    
    # Resample data on an hourly basis
    df.set_index('created_at', inplace=True)
    hourly_df = df.resample('1H').agg({
        'PM2.5_ATM_ug/m3': 'mean',
        'PM2.5_ATM_ug/m3.1': 'mean',
        'valid_reading_PM25_ATM_A': 'sum',
        'valid_reading_PM25_ATM_B': 'sum'
    })
    
    # Filter out hours with less than four valid readings
    hourly_df = hourly_df[(hourly_df['valid_reading_PM25_ATM_A'] >= 4) & (hourly_df['valid_reading_PM25_ATM_B'] >= 4)]
    hourly_df.drop(['valid_reading_PM25_ATM_A', 'valid_reading_PM25_ATM_B'], axis=1, inplace=True)
    hourly_df.reset_index(inplace=True)
    
    hourly_dfs.append(hourly_df)
    
    # Save the hourly averaged dataframe as a CSV file
    output_filename = f"C:\\Users\\jawan\\Dissertation Data\\Final Datasets\\SM\\LCS_SM_2018_2019\\Quarter 0\\Hourly_Averages1\\{sensor_name}_SM_hourly_averages.csv"
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    hourly_df.to_csv(output_filename, index=False)

 # Calculate the percentage of non-valid readings
    total_readings = len(df)
    valid_readings_A = df['valid_reading_PM25_ATM_A'].sum()
    valid_readings_B = df['valid_reading_PM25_ATM_B'].sum()
    non_valid_readings = total_readings - (valid_readings_A + valid_readings_B) / 2
    non_valid_percentages[sensor_name] = (non_valid_readings / total_readings) * 100

    # Print the percentage of non-valid readings by sensor
    print("Percentage of non-valid readings by sensor:")
    for sensor_name, percentage in non_valid_percentages.items():
        print(f"{sensor_name}: {percentage:.2f}%")


# Create a plot
fig, axs = plt.subplots(len(sensor_names), 1, figsize=(15, 30), sharex=True)
fig.suptitle('Hourly Averaged PM$_{2.5}$ Readings (Channel A and B)', fontsize=20, y=0.9)

for i, hourly_df in enumerate(hourly_dfs):
    sensor_name = sensor_names[i]
    axs[i].plot(hourly_df['created_at'], hourly_df['PM2.5_ATM_ug/m3'], label='PM2.5_ATM_ug/m3')
    axs[i].plot(hourly_df['created_at'], hourly_df['PM2.5_ATM_ug/m3.1'], label='PM2.5_ATM_ug/m3.1')
    axs[i].set_title(sensor_name)
    axs[i].legend()
    axs[i].set_ylabel('PM$_{2.5}$ (µg/m³)')

plt.xlabel('Time')
plt.xticks(rotation=45)
plt.show()



#REPORTING TIMESERIES PLOTS (Quarter 1)



# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 01:40:15 2023

@author: jawan
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 17:50:21 2023

@author: jawan
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 12:03:13 2023

@author: jawan
"""
import sensortoolkit
import matplotlib as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce
import plotly.express as px
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pytz
from tqdm import tqdm
import matplotlib.dates as mdates
import numpy as np
from datetime import timedelta

# Ingest data with specified start_date and end_date
start_date = pd.to_datetime('2019-01-01').tz_localize('UTC')
end_date = pd.to_datetime('2019-04-01').tz_localize('UTC')

dfs = []
for i in range(1, 27):
    filename = f"C:\\Users\\jawan\\Dissertation Data\\Final Datasets\\SM\\LCS_SM_2018_2019\\Sensor{i}_SM.csv"
    try:
        df = pd.read_csv(filename)
        df['created_at'] = pd.to_datetime(df['created_at']).dt.tz_convert('UTC')
        df = df[(df['created_at'] >= start_date) & (df['created_at'] <= end_date)]
        df['Sensor'] = i
        dfs.append(df)
    except FileNotFoundError:
        print(f"File not found: {filename}. Skipping and moving on...")

# Concatenate data from all sensors
data = pd.concat(dfs)

# Group data by sensor and resample daily
data['day'] = data['created_at'].dt.to_period('D')
grouped_data = data.groupby(['Sensor', 'day'])['PM2.5_ATM_ug/m3'].agg(lambda x: (x.notnull().sum() > 0))

# Unstack the grouped data and fill NaN values with False
unstacked_data = grouped_data.unstack(level=0).fillna(False)

# Create bar plot (Gantt chart style)
fig, ax = plt.subplots(figsize=(15, 7), dpi=100)

# Define colors for each sensor (508 compliant color)
colors = plt.cm.get_cmap('viridis', 26)

for i, sensor in enumerate(unstacked_data.columns):
    sensor_data = unstacked_data[sensor].astype(int)
    for j, reporting in enumerate(sensor_data):
        if reporting:
            ax.barh(i, timedelta(days=1), left=unstacked_data.index[j].to_timestamp(), color=colors(i), align='center', height=0.6)

# Set chart labels and title
start_date_str = start_date.strftime('%B %d, %Y')
end_date_str = end_date.strftime('%B %d, %Y')
ax.set_xlabel('Time Reporting')
ax.set_ylabel('Sensors')
ax.set_title(f'Sensor Reporting Data ({start_date_str} to {end_date_str})')

# Set y-axis with sensor names
sensor_names = [f'Sensor {sensor}' for sensor in unstacked_data.columns]
ax.set_yticks(range(len(sensor_names)))
ax.set_yticklabels(sensor_names, fontsize=8)

# Set x-axis with actual dates - re-think this!
date_range = pd.date_range(start_date.to_pydatetime(), end_date.to_pydatetime(), freq='MS')
ax.set_xticks(date_range)
ax.set_xticklabels([date.strftime('%B %Y') for date in date_range], fontsize=8, rotation=0, ha='left')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%B %Y'))
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=120))
ax.tick_params(axis='x', which='minor', length=5)

# Find the maximum date in the data
max_date = unstacked_data.index.max().to_timestamp()

# Calculate and display the percentage of time reporting next to each sensor bar
total_duration = (end_date - start_date).days
for i, sensor in enumerate(unstacked_data.columns):
    percent_reporting = (unstacked_data[sensor].sum() / total_duration) * 100
    percent_reporting = min(percent_reporting, 100)  # Ensure percent reporting is not greater than 100%
    ax.annotate(f'{percent_reporting:.1f}%', xy=(end_date.to_pydatetime() + timedelta(days=7), i), fontsize=8, va='center')

# Adjust the x-axis limit to include the end date
ax.set_xlim(start_date.to_pydatetime(), end_date.to_pydatetime() + timedelta(days=30))

plt.tight_layout()
plt.show()



#WITH OUTLIERS (Quarter 0)

start_date = pd.to_datetime('2019-01-01').tz_localize('UTC')
end_date = pd.to_datetime('2019-04-01').tz_localize('UTC')


dfs = []
for i in range(1, 27):
    filename = f"C:\\Users\\jawan\\Dissertation Data\\Final Datasets\\SM\\LCS_SM_2018_2019\\Quarter 0\\Sensor{i}_SM.csv"
    try:
        df = pd.read_csv(filename)
        df['created_at'] = pd.to_datetime(df['created_at']).dt.tz_convert('UTC')
        df = df[(df['created_at'] >= start_date) & (df['created_at'] <= end_date)]
        dfs.append(df)
    except FileNotFoundError:
        print(f"File not found: {filename}. Skipping and moving on...")

# Define sensor names
sensor_names = ['Sensor 1', 'Sensor 4', 'Sensor 6', 'Sensor 7', 'Sensor 8', 'Sensor 10', 'Sensor 13', 'Sensor 20']

#Timeseries Plot With Outliers


# Create a plot
fig, axs = plt.subplots(len(sensor_names), 1, figsize=(15, 30), sharex=True)
fig.suptitle('Timeseries PM$_{2.5}$ Readings (Channel A and B) with Outliers', fontsize=20, y=0.9)

for i, sensor_name in enumerate(sensor_names):
    df = dfs[i]
    axs[i].plot(df['created_at'], df['PM2.5_ATM_ug/m3'], label='PM2.5_ATM_ug/m3')
    axs[i].plot(df['created_at'], df['PM2.5_ATM_ug/m3.1'], label='PM2.5_ATM_ug/m3.1')
    axs[i].set_title(sensor_name)
    axs[i].legend()
    axs[i].set_ylabel('PM$_{2.5}$ (µg/m³)')

plt.xlabel('Time')
plt.xticks(rotation=0)
plt.show()




# Create plot
fig, axs = plt.subplots(2, 4, figsize=(15, 15))
fig.suptitle('Intra-Sensor Precision - PM$_{2.5}$ Channel A vs. Channel B', fontsize=20)

# Process each sensor
for i in tqdm(range(min(len(dfs), 24)), desc='Processing sensors'):
    df = dfs[i]
    df.dropna(inplace=True)
    x = df['PM2.5_ATM_ug/m3']
    y = df['PM2.5_ATM_ug/m3.1']

    # Check that x and y have at least one non-missing value
    if len(x) > 0 and len(y) > 0:
        # Fit linear regression model
        model = LinearRegression()
        model.fit(x.values.reshape(-1, 1), y)

        # Generate predictions
        y_pred = model.predict(x.values.reshape(-1, 1))

        # Calculate R-squared and RMSE
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))

        # Plot data and regression line
        row = (i-1) // 4
        col = (i-1) % 4
        ax = axs[row, col]
        ax.scatter(x, y, color='blue', alpha=0.5)
        ax.plot(x, y_pred, color='black', linestyle='dotted')

        # Label subplot with sensor name and statistics
        ax.set_title(sensor_names[i-1], fontsize=12, loc='left')
        sign = "+" if model.intercept_ >= 0 else "-"
        abs_intercept = abs(model.intercept_)
        ax.text(0.05, 0.95, f"y = {model.coef_[0]:.2f}x {sign} {abs_intercept:.2f}\nR-squared = {r2:.2f}\nRMSE = {rmse:.2f}\nN={len(df)}", transform=ax.transAxes, fontsize=10, ha='left', va='top')
        # Shade 95% confidence interval around regression plot
        ci = 1.96 * np.std(y - y_pred) / np.sqrt(len(y))
        ax.fill_between(x, y_pred - ci, y_pred + ci, color='gray', alpha=.6)
        # Set x and y axis limits to be the same
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_xlim(min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))
        ax.set_ylim(min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))
        

# Label overall plot axis
fig.text(0.5, -0.03, 'Particulate Matter (PM$_{2.5}$) Channel A', ha='center', fontsize=16)
fig.text(-0.03, 0.5, 'Particulate Matter (PM$_{2.5}$) Channel B', va='center', rotation='vertical', fontsize=16)

# Show plot
plt.tight_layout()
plt.show()
    
    
     
# Without Outliers (Quarter 0)


#REMOVING OUTLIERS

start_date = pd.to_datetime('2019-01-01').tz_localize('UTC')
end_date = pd.to_datetime('2019-04-01').tz_localize('UTC')

dfs = []
for i in range(1, 27):
    filename = f"C:\\Users\\jawan\\Dissertation Data\\Final Datasets\\SM\\LCS_SM_2018_2019\\Quarter 0\\Sensor{i}_SM.csv"
    try:
        df = pd.read_csv(filename)
        df['created_at'] = pd.to_datetime(df['created_at']).dt.tz_convert('UTC')
        df = df[(df['created_at'] >= start_date) & (df['created_at'] <= end_date)]
        dfs.append(df)
    except FileNotFoundError:
        print(f"File not found: {filename}. Skipping and moving on...")
        
        
# Define sensor names
sensor_names = ['Sensor 1', 'Sensor 4', 'Sensor 6', 'Sensor 7', 'Sensor 8', 'Sensor 10', 'Sensor 13', 'Sensor 20']


#Timeseries Plot Without Outliers


# Create a plot
fig, axs = plt.subplots(len(sensor_names), 1, figsize=(15, 30), sharex=True)
fig.suptitle('Timeseries PM$_{2.5}$ Readings (Channel A and B) without Outliers', fontsize=20, y=0.9)

for i, sensor_name in enumerate(sensor_names):
    df = dfs[i]
    axs[i].plot(df['created_at'], df['PM2.5_ATM_ug/m3'], label='PM2.5_ATM_ug/m3')
    axs[i].plot(df['created_at'], df['PM2.5_ATM_ug/m3.1'], label='PM2.5_ATM_ug/m3.1')
    axs[i].set_title(sensor_name)
    axs[i].legend()
    axs[i].set_ylabel('PM$_{2.5}$ (µg/m³)')

plt.xlabel('Time')
plt.xticks(rotation=0)
plt.show()


# Create plot
fig, axs = plt.subplots(2, 4, figsize=(15, 15))
fig.suptitle('Intra-Sensor Precision (after removing outliers) - PM$_{2.5}$ Channel A vs. Channel B', fontsize=20)


# Process each sensor
for i in tqdm(range(min(len(dfs), 24)), desc='Processing sensors'):
    df = dfs[i]
    df.dropna(inplace=True)
    x = df['PM2.5_ATM_ug/m3']
    y = df['PM2.5_ATM_ug/m3.1']

    # Check that x and y have at least one non-missing value
    if len(x) > 0 and len(y) > 0:
        
        # Identify outliers
        sd_error = np.abs(y - x) / np.std(y - x)
        is_outlier = (np.abs(x - y) >= 5) & (sd_error >= 2)

        # Calculate percentage of outliers
        perc_outliers = np.round(np.sum(is_outlier) / len(df) * 100, 2)

        # Fit linear regression model
        model = LinearRegression()
        model.fit(x[~is_outlier].values.reshape(-1, 1), y[~is_outlier])

        # Generate predictions
        y_pred = model.predict(x.values.reshape(-1, 1))

        # Calculate R-squared and RMSE
        r2 = r2_score(y[~is_outlier], y_pred[~is_outlier])
        rmse = np.sqrt(mean_squared_error(y[~is_outlier], y_pred[~is_outlier]))

        # Plot data and regression line
        row = (i-1) // 4
        col = (i-1) % 4
        ax = axs[row, col]
        ax.scatter(x[is_outlier], y[is_outlier], color='red')
        ax.scatter(x[~is_outlier], y[~is_outlier], color='blue')
        ax.plot(x, y_pred, color='black')
        
        
        # Label subplot with sensor name and statistics
        ax.set_title(sensor_names[i], fontsize=12, loc='left')
        sign = "+" if model.intercept_ >= 0 else "-"
        abs_intercept = abs(model.intercept_)
        ax.text(0.05, 0.95, f"y = {model.coef_[0]:.2f}x {sign} {abs_intercept:.2f}\nR-squared = {r2:.2f}\nRMSE = {rmse:.2f}\nN={len(df)}\nOutliers: {perc_outliers}%", transform=ax.transAxes, fontsize=10, ha='left', va='top')
        # Shade 95% confidence interval around regression plot
        ci = 1.96 * np.std(y[~is_outlier] - y_pred[~is_outlier]) / np.sqrt(len(y[~is_outlier]))
        ax.fill_between(x, y_pred - ci, y_pred + ci, color='gray', alpha=.6)
        # Set x and y axis limits to be the same
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_xlim(min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))
        ax.set_ylim(min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))
        

# Label overall plot axis
fig.text(0.5, -0.03, 'Particulate Matter (PM$_{2.5}$) Channel A', ha='center', fontsize=16)
fig.text(-0.03, 0.5, 'Particulate Matter (PM$_{2.5}$) Channel B', va='center', rotation='vertical', fontsize=16)

# Show plot
plt.tight_layout()
plt.show()


    

# Datasets Final Quarter 0 (Outliers Removed)

import os

start_date = pd.to_datetime('2018-10-01').tz_localize('UTC')
end_date = pd.to_datetime('2019-01-01').tz_localize('UTC')

sensor_indices = [4, 6, 7, 8, 10, 16, 20, 21]
sensor_names = ['Sensor 4', 'Sensor 6', 'Sensor 7', 'Sensor 8', 'Sensor 10', 'Sensor 16', 'Sensor 20', 'Sensor 21']

dfs = []
for i in sensor_indices:
    filename = f"C:\\Users\\jawan\\Dissertation Data\\Final Datasets\\SM\\LCS_SM_2018_2019\\Quarter 0\\Sensor{i}_SM.csv"
    try:
        df = pd.read_csv(filename)
        df['created_at'] = pd.to_datetime(df['created_at']).dt.tz_convert('UTC')
        df = df[(df['created_at'] >= start_date) & (df['created_at'] <= end_date)]
        dfs.append(df)
    except FileNotFoundError:
        print(f"File not found: {filename}. Skipping and moving on...")

dfs_no_outliers = []
for i, sensor_name in zip(range(len(sensor_indices)), sensor_names):
    df = dfs[i].copy()
    df.dropna(inplace=True)
    x = df['PM2.5_ATM_ug/m3']
    y = df['PM2.5_ATM_ug/m3.1']

    if len(x) > 0 and len(y) > 0:
        sd_error = np.abs(y - x) / np.std(y - x)
        is_outlier = (np.abs(x - y) >= 5) & (sd_error >= 2)
        
        df_no_outliers = df[~is_outlier]
        dfs_no_outliers.append(df_no_outliers)

        # Save the new dataframe without outliers as a CSV file
        output_filename = f"C:\\Users\\jawan\\Dissertation Data\\Final Datasets\\SM\\LCS_SM_2018_2019\\Quarter 0\\No_Outliers\\{sensor_name}_SM_no_outliers.csv"
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        df_no_outliers.to_csv(output_filename, index=False)

#1-hr Averaged Datasets

sensor_names = ['Sensor 4', 'Sensor 6', 'Sensor 7', 'Sensor 8', 'Sensor 10', 'Sensor 16', 'Sensor 20', 'Sensor 21']

hourly_dfs = []
# Add a dictionary to store the percentage of non-valid readings by sensor
non_valid_percentages = {}

for sensor_name in sensor_names:
    input_filename = f"C:\\Users\\jawan\\Dissertation Data\\Final Datasets\\SM\\LCS_SM_2018_2019\\Quarter 0\\No_Outliers\\{sensor_name}_SM_no_outliers.csv"
    
    # Read the CSV file
    df = pd.read_csv(input_filename)
    df['created_at'] = pd.to_datetime(df['created_at']).dt.tz_convert('UTC')
    
    # Ensure there are at least four valid readings per hour
    df['valid_reading_PM25_ATM_A'] = df['PM2.5_ATM_ug/m3'].notna().astype(int)
    df['valid_reading_PM25_ATM_B'] = df['PM2.5_ATM_ug/m3.1'].notna().astype(int)
    
    # Resample data on an hourly basis
    df.set_index('created_at', inplace=True)
    hourly_df = df.resample('1H').agg({
        'PM2.5_ATM_ug/m3': 'mean',
        'PM2.5_ATM_ug/m3.1': 'mean',
        'valid_reading_PM25_ATM_A': 'sum',
        'valid_reading_PM25_ATM_B': 'sum'
    })
    
    # Filter out hours with less than four valid readings
    hourly_df = hourly_df[(hourly_df['valid_reading_PM25_ATM_A'] >= 4) & (hourly_df['valid_reading_PM25_ATM_B'] >= 4)]
    hourly_df.drop(['valid_reading_PM25_ATM_A', 'valid_reading_PM25_ATM_B'], axis=1, inplace=True)
    hourly_df.reset_index(inplace=True)
    
    hourly_dfs.append(hourly_df)
    
    # Save the hourly averaged dataframe as a CSV file
    output_filename = f"C:\\Users\\jawan\\Dissertation Data\\Final Datasets\\SM\\LCS_SM_2018_2019\\Quarter 0\\Hourly_Averages1\\{sensor_name}_SM_hourly_averages.csv"
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    hourly_df.to_csv(output_filename, index=False)

 # Calculate the percentage of non-valid readings
    total_readings = len(df)
    valid_readings_A = df['valid_reading_PM25_ATM_A'].sum()
    valid_readings_B = df['valid_reading_PM25_ATM_B'].sum()
    non_valid_readings = total_readings - (valid_readings_A + valid_readings_B) / 2
    non_valid_percentages[sensor_name] = (non_valid_readings / total_readings) * 100

    # Print the percentage of non-valid readings by sensor
    print("Percentage of non-valid readings by sensor:")
    for sensor_name, percentage in non_valid_percentages.items():
        print(f"{sensor_name}: {percentage:.2f}%")


# Create a plot
fig, axs = plt.subplots(len(sensor_names), 1, figsize=(15, 30), sharex=True)
fig.suptitle('Hourly Averaged PM$_{2.5}$ Readings (Channel A and B)', fontsize=20, y=0.9)

for i, hourly_df in enumerate(hourly_dfs):
    sensor_name = sensor_names[i]
    axs[i].plot(hourly_df['created_at'], hourly_df['PM2.5_ATM_ug/m3'], label='PM2.5_ATM_ug/m3')
    axs[i].plot(hourly_df['created_at'], hourly_df['PM2.5_ATM_ug/m3.1'], label='PM2.5_ATM_ug/m3.1')
    axs[i].set_title(sensor_name)
    axs[i].legend()
    axs[i].set_ylabel('PM$_{2.5}$ (µg/m³)')

plt.xlabel('Time')
plt.xticks(rotation=45)
plt.show()











