# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 21:24:49 2023

@author: jawan
"""

"""
Created on Tue Mar 28 15:39:00 2023

@author: jawan
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 00:34:45 2023

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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Ingest data with specified start_date and end_date
start_date = pd.to_datetime('2018-10-01').tz_localize('UTC')
end_date = pd.to_datetime('2019-01-01').tz_localize('UTC')

dfs = []
sensor_names = ['Sensor4', 'Sensor6', 'Sensor7', 'Sensor8','Sensor10', 'Sensor16', 'Sensor20', 'Sensor21']

for sensor_name in tqdm(sensor_names, desc="Processing sensors"):
    filename = f"C:\\Users\\jawan\\Dissertation Data\\Final Datasets\\SM\\LCS_SM_2018_2019\\{sensor_name}_SM.csv"
    try:
        df = pd.read_csv(filename)
        df['created_at'] = pd.to_datetime(df['created_at']).dt.tz_convert('UTC')
        df = df[(df['created_at'] >= start_date) & (df['created_at'] <= end_date)]
        dfs.append(df)
    except FileNotFoundError:
        print(f"File not found: {filename}. Skipping and moving on...")

# WITH OUTLIERS (Quarter 0_SM)

# Create plot
fig, axs = plt.subplots(2, 4, figsize=(15, 15))
fig.suptitle('Intra-Sensor Precision - PM$_{2.5}$ Channel A vs. Channel B with outliers', fontsize=20)

# Process each sensor
counter = 0

for df in tqdm(dfs, desc='Processing sensors'):
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
        row = counter // 4
        col = counter % 4
        ax = axs[row, col]
        ax.scatter(x, y, color='blue', alpha=0.5)
        ax.plot(x, y_pred, color='black', linestyle='dotted')
        # Label subplot with sensor name and statistics
        ax.set_title(sensor_names[counter], fontsize=12, loc='left')
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

        counter += 1

# Label overall plot axis
fig.text(0.5, -0.03, 'Particulate Matter (PM$_{2.5}$) Channel A', ha='center', fontsize=16)
fig.text(-0.03, 0.5, 'Particulate Matter (PM$_{2.5}$) Channel B', va='center', rotation='vertical', fontsize=16)

# Show plot
plt.tight_layout()
plt.show()
    

# Without Outliers (Quarter 0_corrected)

# Create plot
fig, axs = plt.subplots(2, 4, figsize=(15, 15))
fig.suptitle('Intra-Sensor Precision - PM$_{2.5}$ Channel A vs. Channel B without outliers', fontsize=20)

# Process each sensor
counter = 0

for df in tqdm(dfs, desc='Processing sensors'):
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
        row = counter // 4
        col = counter % 4
        ax = axs[row, col]
        ax.scatter(x[is_outlier], y[is_outlier], color='red')
        ax.scatter(x[~is_outlier], y[~is_outlier], color='blue')
        ax.plot(x, y_pred, color='black')

        # Label subplot with sensor name and statistics
        ax.set_title(sensor_names[counter], fontsize=12, loc='left')
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
    
    counter += 1

# Label overall plot axis
fig.text(0.5, -0.03, 'Particulate Matter (PM$_{2.5}$) Channel A', ha='center', fontsize=16)
fig.text(-0.03, 0.5, 'Particulate Matter (PM$_{2.5}$) Channel B', va='center', rotation='vertical', fontsize=16)

# Show plot
plt.tight_layout()
plt.show()

#Timeseries Plot Without Outliers


fig, axs = plt.subplots(len(sensor_names), 1, figsize=(15, 30), sharex=True)
fig.suptitle('Timeseries PM$_{2.5}$ Readings (Channel A and B) without Outliers', fontsize=20, y=0.9)

for i, sensor_name in enumerate(sensor_names):
    df = dfs[i]
    axs[i].plot(df['created_at'], df['PM2.5_ATM_ug/m3'], color='blue', label='PM2.5_ATM_ug/m3')
    axs[i].plot(df['created_at'], df['PM2.5_ATM_ug/m3.1'], color='orange', label='PM2.5_ATM_ug/m3.1')
    axs[i].set_title(sensor_name)
    axs[i].legend()
    axs[i].set_ylabel('PM$_{2.5}$ (µg/m³)')

plt.xlabel('Time')
plt.xticks(rotation=0)
plt.show()


    

# Datasets Final Quarter 0 (Outliers Removed - conditions imposed <=10% outliers OR >=0.9 R^2)

import os


# Ingest data with specified start_date and end_date
start_date = pd.to_datetime('2018-10-01').tz_localize('UTC')
end_date = pd.to_datetime('2019-01-01').tz_localize('UTC')

# Define list of sensors to include
sensor_names = ['Sensor4', 'Sensor6', 'Sensor7', 'Sensor8','Sensor10', 'Sensor16', 'Sensor20']

# Define folder to save CSV files
save_folder = "C:\\Users\\jawan\\Dissertation Data\\Final Datasets\\SM\\LCS_SM_2018_2019\\Quarter 0"

# Create folder if it doesn't exist
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Ingest data and remove outliers for specified sensors
dfs = []
for sensor_name in tqdm(sensor_names, desc="Processing sensors"):
    filename = f"C:\\Users\\jawan\\Dissertation Data\\Final Datasets\\SM\\LCS_SM_2018_2019\\{sensor_name}_SM.csv"
    try:
        df = pd.read_csv(filename)
        df['created_at'] = pd.to_datetime(df['created_at']).dt.tz_convert('UTC')
        df = df[(df['created_at'] >= start_date) & (df['created_at'] <= end_date)]
        
        # Identify outliers
        x = df['PM2.5_ATM_ug/m3']
        y = df['PM2.5_ATM_ug/m3.1']
        sd_error = np.abs(y - x) / np.std(y - x)
        is_outlier = (np.abs(x - y) >= 5) & (sd_error >= 2)
        df = df[~is_outlier]
        
        # Add processed data to list
        dfs.append(df)
        
        # Save processed data to CSV file
        save_path = os.path.join(save_folder, f"{sensor_name}_SM_no_outliers.csv")
        df.to_csv(save_path, index=False)
        
    except FileNotFoundError:
        print(f"File not found: {filename}. Skipping and moving on...")   
        
             

    

 # PM columns averaged i.e. 'PM2.5_ATM_ug/m3'& 'PM2.5_ATM_ug/m3.1 columns averaged
 
folder_location = "C:\\Users\\jawan\\Dissertation Data\\Final Datasets\\SM\\LCS_SM_2018_2019\\Quarter 0"
 
 # Specify the list of sensors to include
sensor_names = ['Sensor4', 'Sensor6', 'Sensor7', 'Sensor8','Sensor10', 'Sensor16', 'Sensor20']

# Loop through the list of sensors and read the CSV files
for sensor_name in sensor_names:
    file_path = os.path.join(folder_location, f'{sensor_name}_SM_no_outliers.csv')
    df = pd.read_csv(file_path)

    # Calculate the average of the two PM2.5 columns
    pm25_avg = (df['PM2.5_ATM_ug/m3'] + df['PM2.5_ATM_ug/m3.1']) / 2

    # Add a new column to the dataframe with the averaged values
    df['PM25_avg'] = pm25_avg

    # Save the dataframe as a new CSV file with '_averaged' suffix
    save_path = os.path.join(folder_location, f'{sensor_name}_no_outliers_averaged.csv')
    df.to_csv(save_path, index=False)




