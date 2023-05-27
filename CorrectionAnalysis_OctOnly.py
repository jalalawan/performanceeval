# -*- coding: utf-8 -*-
"""

'''
@author: jawan
'''

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

import os
import pandas as pd

reg_folder_path = r"C:\Users\<Enter your local directory for Regulatory Data>"
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
end_date = pd.to_datetime('2018-10-31', format='%Y-%m-%d').tz_localize('UTC')

# Filter reg_df for a specific date range
filtered_reg_df = reg_df[(reg_df['datetime'] >= start_date) & (reg_df['datetime'] <= end_date)]


# Read sensor data
sensor_folder_path = 'C:\\Users\\jawan\\Dissertation Data\\Final Datasets\\SM\\LCS_SM_2018_2019\\Quarter 0\\Q0_corrected eqs'
sensor_file_names = os.listdir(sensor_folder_path)



# Create an empty list to store the sensor DataFrames
sensor_dfs = []

sensor_names = ['Sensor4', 'Sensor6', 'Sensor7', 'Sensor8','Sensor10', 'Sensor16', 'Sensor20']

# Loop through the sensor names and read the corresponding data files into DataFrames
for sensor_name in tqdm(sensor_names, desc="Processing sensors"):
    filename = os.path.join(sensor_folder_path, f"{sensor_name}_corrected.csv")
    df = pd.read_csv(filename)
    sensor_dfs.append(df) # append the DataFrame to the list


# Concatenate all the DataFrames in sensor_dfs into a single DataFrame
all_sensors_df = pd.concat(sensor_dfs)

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


# Model    Fit    Equation


# Model    Fit    Equation
# A    U.S. correction    PM2.5 = 0.524 × PM25_avg − 0.0862 × Humidity_% + 5.75
# B    Quadratic    PM2.5 = a × PM25_avg^2 + b × PM25_avg + c
# C    Cubic    PM2.5 = a × PM25_avg^3 + b × PM25_avg^2 + c × PM25_avg +d
# D    Quadratic + Humidity_%    PM2.5 = a × PM25_avg^2 + b × PM25_avg + c + d × Humidity_%
# E    Quadratic PM * Humidity_%    PM2.5 = a × PM25_avg^2 + b × PM25_avg +c + d × Humidity_% + e × PM25_avg × Humidity_%



import os

# Define correction functions
def PM25correction_A(pm25_avg, humidity_pct):
    return 0.524 * pm25_avg - 0.0862 * humidity_pct + 5.75

import statsmodels.formula.api as smf

# Add a column for PM25_avg^2 to the merged_df DataFrame
merged_df['PM25_avg_squared'] = merged_df['PM25_avg'] ** 2

# Perform the regression using the actual equation
model = smf.ols(formula='Q("Sample Measurement_x") ~ Q("PM25_avg") + Q("PM25_avg_squared")', data=merged_df).fit()

# Print the coefficients a, b, and c
print("a:", model.params['Q("PM25_avg_squared")'])
print("b:", model.params['Q("PM25_avg")'])
print("c:", model.params['Intercept'])



# Add required columns
merged_df['PM25_avg_squared'] = merged_df['PM25_avg'] ** 2
merged_df['PM25_avg_cubed'] = merged_df['PM25_avg'] ** 3
merged_df['PM25_avg_times_humidity'] = merged_df['PM25_avg'] * merged_df['Humidity_%']

# Model A
merged_df['PM_correctedA2'] = PM25correction_A(merged_df['PM25_avg'], merged_df['Humidity_%'])

# Model B
model_B = smf.ols(formula='Q("Sample Measurement_x") ~ Q("PM25_avg") + Q("PM25_avg_squared")', data=merged_df).fit()
merged_df['PM_correctedB'] = model_B.params['Q("PM25_avg_squared")'] * merged_df['PM25_avg_squared'] + model_B.params['Q("PM25_avg")'] * merged_df['PM25_avg'] + model_B.params['Intercept']

# Model C
model_C = smf.ols(formula='Q("Sample Measurement_x") ~ Q("PM25_avg_cubed") + Q("PM25_avg_squared") + Q("PM25_avg")', data=merged_df).fit()
merged_df['PM_correctedC'] = model_C.params['Q("PM25_avg_cubed")'] * merged_df['PM25_avg_cubed'] + model_C.params['Q("PM25_avg_squared")'] * merged_df['PM25_avg_squared'] + model_C.params['Q("PM25_avg")'] * merged_df['PM25_avg'] + model_C.params['Intercept']

# Model D
model_D = smf.ols(formula='Q("Sample Measurement_x") ~ Q("PM25_avg_squared") + Q("PM25_avg") + Q("Humidity_%")', data=merged_df).fit()
merged_df['PM_correctedD'] = model_D.params['Q("PM25_avg_squared")'] * merged_df['PM25_avg_squared'] + model_D.params['Q("PM25_avg")'] * merged_df['PM25_avg'] + model_D.params['Intercept'] + model_D.params['Q("Humidity_%")'] * merged_df['Humidity_%']

# Model E
model_E = smf.ols(formula='Q("Sample Measurement_x") ~ Q("PM25_avg_squared") + Q("PM25_avg") + Q("Humidity_%") + Q("PM25_avg_times_humidity")', data=merged_df).fit()
merged_df['PM_correctedE'] = model_E.params['Q("PM25_avg_squared")'] * merged_df['PM25_avg_squared'] + model_E.params['Q("PM25_avg")'] * merged_df['PM25_avg'] + model_E.params['Intercept'] + model_E.params['Q("Humidity_%")'] * merged_df['Humidity_%'] + model_E.params['Q("PM25_avg_times_humidity")'] * merged_df['PM25_avg_times_humidity']


# Generate the table with coefficients
coeff_table = pd.DataFrame({'Model': ['A', 'B', 'C', 'D', 'E'],
'a': [0.524, model_B.params['Q("PM25_avg_squared")'], model_C.params['Q("PM25_avg_cubed")'], model_D.params['Q("PM25_avg_squared")'], model_E.params['Q("PM25_avg_squared")']],
'b': [-0.0862, model_B.params['Q("PM25_avg")'], model_C.params['Q("PM25_avg_squared")'], model_D.params['Q("PM25_avg")'], model_E.params['Q("PM25_avg")']],
'c': [5.75, model_B.params['Intercept'], model_C.params['Q("PM25_avg")'], model_D.params['Intercept'], model_E.params['Intercept']],
'd': ['-', '-', model_C.params['Intercept'], model_D.params['Q("Humidity_%")'], model_E.params['Q("Humidity_%")']],
'e': ['-', '-', '-', '-', model_E.params['Q("PM25_avg_times_humidity")']]})

coeff_table.set_index('Model', inplace=True)
print(coeff_table)


# Save the coefficient table to a CSV file
coeff_table.to_csv('coefficients_table_oct.csv')

# Save merged_df to a CSV file
merged_df.to_csv('merged_df_oct.csv')

#

 r'PM2.5 = 0.524 × PM25_avg - 0.0862 × Humidity_% + 5.75',
        r'PM2.5 = a × PM25_avg² + b × PM25_avg + c',
        r'PM2.5 = a × PM25_avg³ + b × PM25_avg² + c × PM25_avg + d',
        r'PM2.5 = a × PM25_avg² + b × PM25_avg + c + d × Humidity_%',
        r'PM2.5 = a × PM25_avg² + b × PM25_avg + c + d × Humidity_% + e × PM25_avg × Humidity_%'
        
        #
        
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# User-specified start and end dates
start_date = '2018-10-01'
end_date = '2018-10-31'

# Filter the data to show only the specified date range
filtered_df = merged_df.loc[(merged_df.index >= start_date) & (merged_df.index <= end_date)]

# Set up the plot
fig, ax = plt.subplots(figsize=(12, 6))

# Plot the data
ax.plot(filtered_df.index, filtered_df['PM25_avg'], label='$PM_{2.5}$_average_all')
ax.plot(filtered_df.index, filtered_df['PM_correctedA2'], label='$PM_{2.5}$_correctedA')
ax.plot(filtered_df.index, filtered_df['PM_correctedB'], label='$PM_{2.5}$_correctedB')
ax.plot(filtered_df.index, filtered_df['PM_correctedC'], label='$PM_{2.5}$_correctedC')
ax.plot(filtered_df.index, filtered_df['PM_correctedD'], label='$PM_{2.5}$_correctedD')
ax.plot(filtered_df.index, filtered_df['PM_correctedE'], label='$PM_{2.5}$_correctedE')
ax.plot(filtered_df.index, filtered_df['Sample Measurement_x'], label='FRM / FEM Measurement')

# Format the x-axis
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %Y'))
ax.xaxis.set_minor_locator(mdates.DayLocator())

# Set axis labels and title
ax.set_xlabel('Time')
ax.set_ylabel('$PM_{2.5}$ Concentration')
ax.set_title('$PM_{2.5}$ Concentrations and Model Corrections')

# Add a legend
ax.legend()

# Grid and tight layout
ax.grid(True)
fig.tight_layout()

# Save the figure as a publication-quality image
plt.savefig('timeseries_chart_filtered_custom_date_range.png', dpi=300)

# Show the plot
plt.show()










