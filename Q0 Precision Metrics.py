# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 17:58:02 2023

@author: jawan
"""
import pandas as pd
import os
import sensortoolkit
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
import os


import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy import stats

def epa_sd(sensor_data):
    N = len(sensor_data.index.unique())
    M = len(sensor_data['Name'].unique())

    sd_sum = 0
    for i in sensor_data.index.unique():
        sensor_data_i = sensor_data.loc[i]
        x_bar_i = sensor_data_i['PM25_avg'].mean()
        sum_squares = ((sensor_data_i['PM25_avg'] - x_bar_i) ** 2).sum()
        sd_sum += sum_squares

    sd = np.sqrt((1 / (N * M - 1)) * sd_sum)
    return sd

reg_folder_path = r"C:\Users\jawan\Dissertation Data\Final Datasets\SM\Regulatory_SM_2018_2019"
reg_file_names = ['Reg_SM_2018.csv']

reg_dfs = []

for file_name in reg_file_names:
    file_path = os.path.join(reg_folder_path, file_name)
    df = pd.read_csv(file_path)
    reg_dfs.append(df)

reg_df = pd.concat(reg_dfs)

reg_df['datetime'] = pd.to_datetime(reg_df['datetime'], format='%Y-%m-%d %H:%M').dt.tz_localize('UTC')

reg_df.set_index('datetime', drop=False, inplace=True)

start_date = pd.to_datetime('2018-10-01', format='%Y-%m-%d').tz_localize('UTC')
end_date = pd.to_datetime('2018-12-31', format='%Y-%m-%d').tz_localize('UTC')

filtered_reg_df = reg_df[(reg_df['datetime'] >= start_date) & (reg_df['datetime'] <= end_date)]

sensor_folder_path = 'C:\\Users\\jawan\\Dissertation Data\\Final Datasets\\SM\\LCS_SM_2018_2019\\Quarter 0'
sensor_file_names = os.listdir(sensor_folder_path)

sensor_dfs = []
sensor_names = ['Sensor4', 'Sensor6', 'Sensor7', 'Sensor8', 'Sensor10', 'Sensor16', 'Sensor20']

for sensor_name in tqdm(sensor_names, desc="Processing sensors"):
    filename = os.path.join(sensor_folder_path, f"{sensor_name}_no_outliers_averaged.csv")
    df = pd.read_csv(filename)
    df['Name'] = sensor_name
    sensor_dfs.append(df)

all_sensors_df = pd.concat(sensor_dfs, keys=sensor_names, names=['Name'])

all_sensors_df['created_at'] = pd.to_datetime(all_sensors_df['created_at'], format='%Y-%m-%d %H:%M')
all_sensors_df.set_index('created_at', inplace=True)

cv_list = []
sd_list = []

for sensor_name in sensor_names:
    sensor_data = all_sensors_df[all_sensors_df['Name'] == sensor_name]
    
    sd = epa_sd(sensor_data)
    cv = (sd / sensor_data['PM25_avg'].mean()) * 100
    
    cv_list.append(cv)
    sd_list.append(sd)

sensor_stats = np.zeros((len(sensor_names) + 1, 5), dtype=object)

for i, sensor_name in enumerate(sensor_names):
    sensor_stats[i, 0] = sensor_name
    sensor_stats[i, 1] = cv_list[i]
    sensor_stats[i, 2] = sd_list[i]

mean_values = ['Mean', np.mean(cv_list), np.mean(sd_list)]

sensor_stats[len(sensor_names)] = mean_values

# Convert the matrix to a DataFrame
stats_df = pd.DataFrame(sensor_stats, columns=['Sensor Name', 'CV', 'SD'])

# Save the DataFrame as a CSV file
stats_df.to_csv('sensor_stats_epa_performance_targets.csv', index=False)

# Print the DataFrame
print(stats_df)


#Sensor-Sensor Precision Metrics (Q0_uncorrected)


reg_folder_path = r"C:\Users\jawan\Dissertation Data\Final Datasets\SM\Regulatory_SM_2018_2019"
reg_file_names = ['Reg_SM_2018.csv']

# Create an empty list to store the reference DataFrames
reg_dfs = []

# Loop through the files and read them into DataFrames
for file_name in reg_file_names:
    file_path = os.path.join(reg_folder_path, file_name)
    df = pd.read_csv(file_path)
    reg_dfs.append(df)

# Concatenate all reference DataFrames into a single DataFrame
reg_df = pd.concat(reg_dfs)

# Convert the datetime column to timezone-aware timestamp objects
reg_df['datetime'] = pd.to_datetime(reg_df['datetime'], format='%Y-%m-%d %H:%M').dt.tz_localize('UTC')

# Index reg_df by datetime column without dropping the column
reg_df.set_index('datetime', drop=False, inplace=True)



# Define start and end date in UTC
start_date = pd.to_datetime('2018-10-01', format='%Y-%m-%d').tz_localize('UTC')
end_date = pd.to_datetime('2018-12-31', format='%Y-%m-%d').tz_localize('UTC')

# Filter reg_df for a specific date range
filtered_reg_df = reg_df[(reg_df['datetime'] >= start_date) & (reg_df['datetime'] <= end_date)]

# Read sensor data
sensor_folder_path = 'C:\\Users\\jawan\\Dissertation Data\\Final Datasets\\SM\\LCS_SM_2018_2019\\Quarter 0'
sensor_file_names = os.listdir(sensor_folder_path)

# Create an empty list to store the sensor DataFrames
sensor_dfs = []
sensor_names = ['Sensor4', 'Sensor6', 'Sensor7', 'Sensor8', 'Sensor10', 'Sensor16', 'Sensor20']

# Loop through the sensor names and read the corresponding data files into DataFrames
for sensor_name in tqdm(sensor_names, desc="Processing sensors"):
    filename = os.path.join(sensor_folder_path, f"{sensor_name}_no_outliers_averaged.csv")
    df = pd.read_csv(filename)
    df['Name'] = sensor_name  # Add a 'Name' column to the DataFrame with the sensor name
    sensor_dfs.append(df)

# Concatenate all the DataFrames in sensor_dfs into a single DataFrame and add 'Name' column
all_sensors_df = pd.concat(sensor_dfs, keys=sensor_names, names=['Name'])

# Convert the 'created_at' column to a datetime object and set it as the index
all_sensors_df['created_at'] = pd.to_datetime(all_sensors_df['created_at'], format='%Y-%m-%d %H:%M')
all_sensors_df.set_index('created_at', inplace=True)

# Calculate hourly and 24-hourly averages
hourly_df = all_sensors_df.resample('H').mean()
daily_df = all_sensors_df.resample('24H').mean()

hourly_df['Name'] = all_sensors_df['Name'].resample('H').first()  # Add 'Name' column to hourly_df
daily_df['Name'] = all_sensors_df['Name'].resample('24H').first()  # Add 'Name' column to hourly_df


# Primarily need all_sensors_df for CV / SD, doing this for fun


# Calculate the CV and standard deviation for each sensor (hourly and daily)
cv_hourly_list = []
sd_hourly_list = []
cv_daily_list = []
sd_daily_list = []

# Initialize lists for t-statistic and p-values
t_stat_cv_hourly_list = []
p_val_cv_hourly_list = []
t_stat_sd_hourly_list = []
p_val_sd_hourly_list = []
t_stat_cv_daily_list = []
p_val_cv_daily_list = []
t_stat_sd_daily_list = []
p_val_sd_daily_list = []

for sensor_name in sensor_names:
    hourly_sensor_data = hourly_df[hourly_df['Name'] == sensor_name]['PM25_avg']
    daily_sensor_data = daily_df[daily_df['Name'] == sensor_name]['PM25_avg']
    
    cv_hourly = hourly_sensor_data.std() / hourly_sensor_data.mean()
    sd_hourly = hourly_sensor_data.std()
    cv_daily = daily_sensor_data.std() / daily_sensor_data.mean()
    sd_daily = daily_sensor_data.std()
    
    cv_hourly_list.append(cv_hourly)
    sd_hourly_list.append(sd_hourly)
    cv_daily_list.append(cv_daily)
    sd_daily_list.append(sd_daily)
    
    # Calculate the t-statistic and p-value for CV_hourly
    t_stat_cv_hourly, p_val_cv_hourly = stats.ttest_1samp(hourly_sensor_data, 1)
    t_stat_cv_hourly_list.append(t_stat_cv_hourly)
    p_val_cv_hourly_list.append(p_val_cv_hourly)
    
    # Calculate the t-statistic and p-value for SD_hourly
    t_stat_sd_hourly, p_val_sd_hourly = stats.ttest_1samp(hourly_sensor_data, 1)
    t_stat_sd_hourly_list.append(t_stat_sd_hourly)
    p_val_sd_hourly_list.append(p_val_sd_hourly)
    
    # Calculate the t-statistic and p-value for CV_daily
    t_stat_cv_daily, p_val_cv_daily = stats.ttest_1samp(daily_sensor_data, 1)
    t_stat_cv_daily_list.append(t_stat_cv_daily)
    p_val_cv_daily_list.append(p_val_cv_daily)
    
    # Calculate the t-statistic and p-value for SD_daily
    t_stat_sd_daily, p_val_sd_daily = stats.ttest_1samp(daily_sensor_data, 1)
    t_stat_sd_daily_list.append(t_stat_sd_daily)
    p_val_sd_daily_list.append(p_val_sd_daily)

# Create a matrix to store the stats (sensor name, CV_hourly, t_stat_CV_hourly, p_val_CV_hourly, SD_hourly, t_stat_SD_hourly, p_val_SD_hourly, CV_daily, t_stat_CV_daily, p_val_CV_daily, SD_daily, t_stat_SD_daily, p_val_SD_daily)
sensor_stats = np.zeros((len(sensor_names) + 1, 13), dtype=object)

# Fill the matrix with the stats
for i, sensor_name in enumerate(sensor_names):
    sensor_stats[i, 0] = sensor_name
    sensor_stats[i, 1] = cv_hourly_list[i]
    sensor_stats[i, 2] = t_stat_cv_hourly_list[i]
    sensor_stats[i, 3] = p_val_cv_hourly_list[i]
    sensor_stats[i, 4] = sd_hourly_list[i]
    sensor_stats[i, 5] = t_stat_sd_hourly_list[i]
    sensor_stats[i, 6] = p_val_sd_hourly_list[i]
    sensor_stats[i, 7] = cv_daily_list[i]
    sensor_stats[i, 8] = t_stat_cv_daily_list[i]
    sensor_stats[i, 9] = p_val_cv_daily_list[i]
    sensor_stats[i, 10] = sd_daily_list[i]
    sensor_stats[i, 11] = t_stat_sd_daily_list[i]
    sensor_stats[i, 12] = p_val_sd_daily_list[i]

# Add a row with mean values of CV_hourly, t_stat_CV_hourly, p_val_CV_hourly, SD_hourly, t_stat_SD_hourly, p_val_SD_hourly, CV_daily, t_stat_CV_daily, p_val_CV_daily, SD_daily, t_stat_SD_daily, p_val_SD_daily
mean_values = ['Mean', np.mean(cv_hourly_list), np.mean(t_stat_cv_hourly_list), np.mean(p_val_cv_hourly_list), np.mean(sd_hourly_list), np.mean(t_stat_sd_hourly_list), np.mean(p_val_sd_hourly_list), np.mean(cv_daily_list), np.mean(t_stat_cv_daily_list), np.mean(p_val_cv_daily_list), np.mean(sd_daily_list), np.mean(t_stat_sd_daily_list), np.mean(p_val_sd_daily_list)]
sensor_stats[len(sensor_names)] = mean_values

# Convert the matrix to a DataFrame
stats_df = pd.DataFrame(sensor_stats, columns=['Sensor Name', 'CV_hourly', 't_stat_CV_hourly', 'p_val_CV_hourly', 'SD_hourly', 't_stat_SD_hourly', 'p_val_SD_hourly', 'CV_daily', 't_stat_CV_daily', 'p_val_CV_daily', 'SD_daily', 't_stat_SD_daily', 'p_val_SD_daily'])

# Save the DataFrame as a CSV file
stats_df.to_csv('sensor_stats_hourly_daily_Q0_uncorrected.csv', index=False)

# Print the DataFrame
print(stats_df)




#PLOTTING

# Group the concatenated DataFrame by the 'created_at' column and compute the mean
grouped_df = all_sensors_df.groupby('created_at').mean()

# Remove rows with missing values (NaN) from the result
sensor_dfs_all_avgs = grouped_df.dropna()

# Reset index to have 'created_at' as a regular column
sensor_dfs_all_avgs.reset_index(inplace=True)


# Convert the 'created_at' column to a datetime object
sensor_dfs_all_avgs['created_at'] = pd.to_datetime(sensor_dfs_all_avgs['created_at'])

# Set the 'created_at' column as the index
sensor_dfs_all_avgs.set_index('created_at', inplace=True)

# Resample the DataFrame to hourly frequency and compute the mean
hourly_mean_df = sensor_dfs_all_avgs.resample('1H').mean()

# Reset the index after resampling
hourly_mean_df.reset_index(inplace=True)

# Set the 'datetime' column as the index for both DataFrames without dropping the column
filtered_reg_df.set_index('datetime', drop=False, inplace=True)
hourly_mean_df.set_index('created_at', drop=False, inplace=True)


merged_df = filtered_reg_df.join(hourly_mean_df, how='outer', lsuffix='_reg', rsuffix='_hourly')



# Set plot style
sns.set_style('white')

# Create a figure with one subplot
fig, ax = plt.subplots(figsize=(8, 6))

# Create a histogram plot for PM25_ref column
sns.histplot(merged_df['Sample Measurement_x'], ax=ax, bins=30, alpha=0.6, label='Reference monitor', color='black')

# Create a histogram plot for PM25_avg column
sns.histplot(merged_df['PM25_avg'], ax=ax, bins=30, alpha=0.4, label='Sensor mean', color='green')

# Add labels and legend to the plot
ax.set_xlabel('PM25 values ($\mu$g/m$^3$)')
ax.set_title('Histogram of Sensor and Reference $PM_{2.5}$ values ($\mu$g/m$^3$)')
ax.set_ylabel('Frequency')
ax.legend(loc='upper right')

# Remove plot border and ticks
sns.despine()

# Show the plot
plt.show()




#Sensor-Sensor Precision Metrics (Q0_corrected)


reg_folder_path = r"C:\Users\jawan\Dissertation Data\Final Datasets\SM\Regulatory_SM_2018_2019"
reg_file_names = ['Reg_SM_2018.csv']

# Create an empty list to store the reference DataFrames
reg_dfs = []

# Loop through the files and read them into DataFrames
for file_name in reg_file_names:
    file_path = os.path.join(reg_folder_path, file_name)
    df = pd.read_csv(file_path)
    reg_dfs.append(df)

# Concatenate all reference DataFrames into a single DataFrame
reg_df = pd.concat(reg_dfs)

# Convert the datetime column to timezone-aware timestamp objects
reg_df['datetime'] = pd.to_datetime(reg_df['datetime'], format='%Y-%m-%d %H:%M').dt.tz_localize('UTC')

# Index reg_df by datetime column without dropping the column
reg_df.set_index('datetime', drop=False, inplace=True)



# Define start and end date in UTC
start_date = pd.to_datetime('2018-10-01', format='%Y-%m-%d').tz_localize('UTC')
end_date = pd.to_datetime('2018-12-31', format='%Y-%m-%d').tz_localize('UTC')

# Filter reg_df for a specific date range
filtered_reg_df = reg_df[(reg_df['datetime'] >= start_date) & (reg_df['datetime'] <= end_date)]

# Read sensor data
sensor_folder_path = 'C:\\Users\\jawan\\Dissertation Data\\Final Datasets\\SM\\LCS_SM_2018_2019\\Quarter 0\\Q0_corrected eqs'
sensor_file_names = os.listdir(sensor_folder_path)

# Create an empty list to store the sensor DataFrames
sensor_dfs = []
sensor_names = ['Sensor4', 'Sensor6', 'Sensor7', 'Sensor8', 'Sensor10', 'Sensor16', 'Sensor20']

# Loop through the sensor names and read the corresponding data files into DataFrames
for sensor_name in tqdm(sensor_names, desc="Processing sensors"):
    filename = os.path.join(sensor_folder_path, f"{sensor_name}_corrected.csv")
    df = pd.read_csv(filename)
    df['Name'] = sensor_name  # Add a 'Name' column to the DataFrame with the sensor name
    sensor_dfs.append(df)

# Concatenate all the DataFrames in sensor_dfs into a single DataFrame and add 'Name' column
all_sensors_df = pd.concat(sensor_dfs, keys=sensor_names, names=['Name'])

# Convert the 'created_at' column to a datetime object and set it as the index
all_sensors_df['created_at'] = pd.to_datetime(all_sensors_df['created_at'], format='%Y-%m-%d %H:%M')
all_sensors_df.set_index('created_at', inplace=True)

# Calculate hourly and 24-hourly averages
hourly_df = all_sensors_df.resample('H').mean()
daily_df = all_sensors_df.resample('24H').mean()

hourly_df['Name'] = all_sensors_df['Name'].resample('H').first()  # Add 'Name' column to hourly_df
daily_df['Name'] = all_sensors_df['Name'].resample('24H').first()  # Add 'Name' column to hourly_df


# Primarily need all_sensors_df for CV / SD, doing this for fun


# Calculate the CV and standard deviation for each sensor (hourly and daily)
cv_hourly_list = []
sd_hourly_list = []
cv_daily_list = []
sd_daily_list = []

# Initialize lists for t-statistic and p-values
t_stat_cv_hourly_list = []
p_val_cv_hourly_list = []
t_stat_sd_hourly_list = []
p_val_sd_hourly_list = []
t_stat_cv_daily_list = []
p_val_cv_daily_list = []
t_stat_sd_daily_list = []
p_val_sd_daily_list = []

for sensor_name in sensor_names:
    hourly_sensor_data = hourly_df[hourly_df['Name'] == sensor_name]['PM25_correctedA']
    daily_sensor_data = daily_df[daily_df['Name'] == sensor_name]['PM25_correctedA']
    
    cv_hourly = hourly_sensor_data.std() / hourly_sensor_data.mean()
    sd_hourly = hourly_sensor_data.std()
    cv_daily = daily_sensor_data.std() / daily_sensor_data.mean()
    sd_daily = daily_sensor_data.std()
    
    cv_hourly_list.append(cv_hourly)
    sd_hourly_list.append(sd_hourly)
    cv_daily_list.append(cv_daily)
    sd_daily_list.append(sd_daily)
    
    # Calculate the t-statistic and p-value for CV_hourly
    t_stat_cv_hourly, p_val_cv_hourly = stats.ttest_1samp(hourly_sensor_data, 1)
    t_stat_cv_hourly_list.append(t_stat_cv_hourly)
    p_val_cv_hourly_list.append(p_val_cv_hourly)
    
    # Calculate the t-statistic and p-value for SD_hourly
    t_stat_sd_hourly, p_val_sd_hourly = stats.ttest_1samp(hourly_sensor_data, 1)
    t_stat_sd_hourly_list.append(t_stat_sd_hourly)
    p_val_sd_hourly_list.append(p_val_sd_hourly)
    
    # Calculate the t-statistic and p-value for CV_daily
    t_stat_cv_daily, p_val_cv_daily = stats.ttest_1samp(daily_sensor_data, 1)
    t_stat_cv_daily_list.append(t_stat_cv_daily)
    p_val_cv_daily_list.append(p_val_cv_daily)
    
    # Calculate the t-statistic and p-value for SD_daily
    t_stat_sd_daily, p_val_sd_daily = stats.ttest_1samp(daily_sensor_data, 1)
    t_stat_sd_daily_list.append(t_stat_sd_daily)
    p_val_sd_daily_list.append(p_val_sd_daily)

# Create a matrix to store the stats (sensor name, CV_hourly, t_stat_CV_hourly, p_val_CV_hourly, SD_hourly, t_stat_SD_hourly, p_val_SD_hourly, CV_daily, t_stat_CV_daily, p_val_CV_daily, SD_daily, t_stat_SD_daily, p_val_SD_daily)
sensor_stats = np.zeros((len(sensor_names) + 1, 13), dtype=object)

# Fill the matrix with the stats
for i, sensor_name in enumerate(sensor_names):
    sensor_stats[i, 0] = sensor_name
    sensor_stats[i, 1] = cv_hourly_list[i]
    sensor_stats[i, 2] = t_stat_cv_hourly_list[i]
    sensor_stats[i, 3] = p_val_cv_hourly_list[i]
    sensor_stats[i, 4] = sd_hourly_list[i]
    sensor_stats[i, 5] = t_stat_sd_hourly_list[i]
    sensor_stats[i, 6] = p_val_sd_hourly_list[i]
    sensor_stats[i, 7] = cv_daily_list[i]
    sensor_stats[i, 8] = t_stat_cv_daily_list[i]
    sensor_stats[i, 9] = p_val_cv_daily_list[i]
    sensor_stats[i, 10] = sd_daily_list[i]
    sensor_stats[i, 11] = t_stat_sd_daily_list[i]
    sensor_stats[i, 12] = p_val_sd_daily_list[i]

# Add a row with mean values of CV_hourly, t_stat_CV_hourly, p_val_CV_hourly, SD_hourly, t_stat_SD_hourly, p_val_SD_hourly, CV_daily, t_stat_CV_daily, p_val_CV_daily, SD_daily, t_stat_SD_daily, p_val_SD_daily
mean_values = ['Mean', np.mean(cv_hourly_list), np.mean(t_stat_cv_hourly_list), np.mean(p_val_cv_hourly_list), np.mean(sd_hourly_list), np.mean(t_stat_sd_hourly_list), np.mean(p_val_sd_hourly_list), np.mean(cv_daily_list), np.mean(t_stat_cv_daily_list), np.mean(p_val_cv_daily_list), np.mean(sd_daily_list), np.mean(t_stat_sd_daily_list), np.mean(p_val_sd_daily_list)]
sensor_stats[len(sensor_names)] = mean_values

# Convert the matrix to a DataFrame
stats_df = pd.DataFrame(sensor_stats, columns=['Sensor Name', 'CV_hourly', 't_stat_CV_hourly', 'p_val_CV_hourly', 'SD_hourly', 't_stat_SD_hourly', 'p_val_SD_hourly', 'CV_daily', 't_stat_CV_daily', 'p_val_CV_daily', 'SD_daily', 't_stat_SD_daily', 'p_val_SD_daily'])

# Save the DataFrame as a CSV file
stats_df.to_csv('sensor_stats_hourly_daily_Q0corrected.csv', index=False)

# Print the DataFrame
print(stats_df)




# Calculate the CV and standard deviation for each sensor (sub-hourly)
cv_list = []
sd_list = []

for sensor_name in sensor_names:
    sensor_data1 = all_sensors_df[all_sensors_df['Name'] == sensor_name]['PM25_avg']
    cv = sensor_data1.std() / sensor_data1.mean()
    sd = sensor_data1.std()
    cv_list.append(cv)
    sd_list.append(sd)
    
    
# Create a matrix to store the stats (sensor name, CV, t-stat CV, p-val CV, SD, t-stat SD, p-val SD)
sensor_stats1 = np.zeros((len(sensor_names) + 1, 7), dtype=object)
   
# Fill the matrix with the stats
for i, sensor_name in enumerate(sensor_names):
    sensor_data1 = all_sensors_df[all_sensors_df['Name'] == sensor_name]['PM25_avg']
    cv = sensor_data1.std() / sensor_data1.mean()
    sd = sensor_data1.std()
    
    # Calculate the t-statistic and p-value for CV
    t_stat_cv, p_val_cv = stats.ttest_1samp(sensor_data1, 1)
    
    # Calculate the t-statistic and p-value for SD
    t_stat_sd, p_val_sd = stats.ttest_1samp(sensor_data1, 1)
    
    sensor_stats1[i, 0] = sensor_name
    sensor_stats1[i, 1] = cv
    sensor_stats1[i, 2] = t_stat_cv
    sensor_stats1[i, 3] = p_val_cv
    sensor_stats1[i, 4] = sd
    sensor_stats1[i, 5] = t_stat_sd
    sensor_stats1[i, 6] = p_val_sd

# Add a row with mean values of CV and SD
mean_values = ['Mean', np.mean(cv_list), np.nan, np.nan, np.mean(sd_list), np.nan, np.nan]
sensor_stats1[len(sensor_names)] = mean_values

# Convert the matrix to a DataFrame
stats_df1 = pd.DataFrame(sensor_stats1, columns=['Sensor Name', 'CV', 't-stat CV', 'p-val CV', 'SD', 't-stat SD', 'p-val SD'])

# Save the DataFrame as a CSV file
stats_df1.to_csv('sensor_stats_subhourly_Q0_uncorrected.csv', index=False)

# Print the DataFrame
print(stats_df1)


# Calculate the CV and standard deviation for each sensor (sub-hourly)
cv_list = []
sd_list = []

for sensor_name in sensor_names:
    sensor_data1 = all_sensors_df[all_sensors_df['Name'] == sensor_name]['PM25_correctedA']
    cv = sensor_data1.std() / sensor_data1.mean()
    sd = sensor_data1.std()
    cv_list.append(cv)
    sd_list.append(sd)
    
    
# Create a matrix to store the stats (sensor name, CV, t-stat CV, p-val CV, SD, t-stat SD, p-val SD)
sensor_stats1 = np.zeros((len(sensor_names) + 1, 7), dtype=object)
   
# Fill the matrix with the stats
for i, sensor_name in enumerate(sensor_names):
    sensor_data1 = all_sensors_df[all_sensors_df['Name'] == sensor_name]['PM25_correctedA']
    cv = sensor_data1.std() / sensor_data1.mean()
    sd = sensor_data1.std()
    
    # Calculate the t-statistic and p-value for CV
    t_stat_cv, p_val_cv = stats.ttest_1samp(sensor_data1, 1)
    
    # Calculate the t-statistic and p-value for SD
    t_stat_sd, p_val_sd = stats.ttest_1samp(sensor_data1, 1)
    
    sensor_stats1[i, 0] = sensor_name
    sensor_stats1[i, 1] = cv
    sensor_stats1[i, 2] = t_stat_cv
    sensor_stats1[i, 3] = p_val_cv
    sensor_stats1[i, 4] = sd
    sensor_stats1[i, 5] = t_stat_sd
    sensor_stats1[i, 6] = p_val_sd

# Add a row with mean values of CV and SD
mean_values = ['Mean', np.mean(cv_list), np.nan, np.nan, np.mean(sd_list), np.nan, np.nan]
sensor_stats1[len(sensor_names)] = mean_values

# Convert the matrix to a DataFrame
stats_df1 = pd.DataFrame(sensor_stats1, columns=['Sensor Name', 'CV', 't-stat CV', 'p-val CV', 'SD', 't-stat SD', 'p-val SD'])

# Save the DataFrame as a CSV file
stats_df1.to_csv('sensor_stats_subhourly.csv', index=False)

# Print the DataFrame
print(stats_df1)





#PLOTTING AGAIN for corrected hourly

# Group the concatenated DataFrame by the 'created_at' column and compute the mean
grouped_df = all_sensors_df.groupby('created_at').mean()

# Remove rows with missing values (NaN) from the result
sensor_dfs_all_avgs = grouped_df.dropna()

# Reset index to have 'created_at' as a regular column
sensor_dfs_all_avgs.reset_index(inplace=True)


# Convert the 'created_at' column to a datetime object
sensor_dfs_all_avgs['created_at'] = pd.to_datetime(sensor_dfs_all_avgs['created_at'])

# Set the 'created_at' column as the index
sensor_dfs_all_avgs.set_index('created_at', inplace=True)

# Resample the DataFrame to hourly frequency and compute the mean
hourly_mean_df = sensor_dfs_all_avgs.resample('1H').mean()

# Reset the index after resampling
hourly_mean_df.reset_index(inplace=True)

# Set the 'datetime' column as the index for both DataFrames without dropping the column
filtered_reg_df.set_index('datetime', drop=False, inplace=True)
hourly_mean_df.set_index('created_at', drop=False, inplace=True)


merged_df = filtered_reg_df.join(hourly_mean_df, how='outer', lsuffix='_reg', rsuffix='_hourly')



# Set plot style
sns.set_style('white')

# Create a figure with one subplot
fig, ax = plt.subplots(figsize=(8, 6))

# Create a histogram plot for PM25_ref column
sns.histplot(merged_df['Sample Measurement_x'], ax=ax, bins=30, alpha=0.6, label='Reference monitor', color='black')

# Create a histogram plot for PM25_correctedA column
sns.histplot(merged_df['PM25_correctedA'], ax=ax, bins=30, alpha=0.4, label='Sensor mean', color='green')

# Add labels and legend to the plot
ax.set_xlabel('PM25 values ($\mu$g/m$^3$)')
ax.set_title('Histogram of Sensor (corrected) and Reference $PM_{2.5}$ values ($\mu$g/m$^3$)')
ax.set_ylabel('Frequency')
ax.legend(loc='upper right')

# Remove plot border and ticks
sns.despine()

# Show the plot
plt.show()









