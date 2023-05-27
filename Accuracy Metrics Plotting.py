# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 16:33:36 2023

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
import os
import pandas as pd
import re
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



plt.rcParams.update({'font.size': 12})

# Replace file_path with the actual path to the csv file
file_path = r'C:\Users\jawan\Dissertation Data\Final Analysis\SM\Results\Precision Metrics\Accuracy Metrics\Accuracy_rev.csv'

# Read the csv file into a pandas DataFrame
df = pd.read_csv(file_path)

df['Sensor Name'] = df['Sensor Name'].replace("PurpleAir_PurpleAir-II_SM_Q0", "PurpleAir_PurpleAir-II_SM_Q0_uncorrected")
df['Sensor Name'] = df['Sensor Name'].replace("PurpleAir_PurpleAir-II_SM_Q1", "PurpleAir_PurpleAir-II_SM_Q1_uncorrected")
df['Sensor Name'] = df['Sensor Name'].replace("PurpleAir_PurpleAir-II_SM_Q2", "PurpleAir_PurpleAir-II_SM_Q2_uncorrected")
df['Sensor Name'] = df['Sensor Name'].replace("PurpleAir_PurpleAir-II_SM_Q3", "PurpleAir_PurpleAir-II_SM_Q3_uncorrected")
df['Sensor Name'] = df['Sensor Name'].replace("PurpleAir_PurpleAir-II_SM_Q4", "PurpleAir_PurpleAir-II_SM_Q4_uncorrected")

# Extract the Quarter and correction information from Sensor Name column
df['Quarter'] = df['Sensor Name'].str.split('_').str[-2] + '_' + df['Sensor Name'].str.split('_').str[-1]
df = df.fillna('')

# Define the metrics to plot
metrics = ['R$^2$', 'Slope', 'Intercept', 'Sensor RMSE']

# Define the target and goal ranges for each metric
metric_bounds_goal = {
    'R$^2$': {'column': 'R$^2$', 'bounds': (0.7, 1.0), 'goal': 1.0},
    'Slope': {'column': 'Slope', 'bounds': (0.65, 1.35), 'goal': 1.0},
    'Intercept': {'column': 'Intercept', 'bounds': (-5.0, 5.0), 'goal': 0.0},
    'Sensor RMSE': {'column': 'Sensor RMSE', 'bounds': (0.0, 7.0), 'goal': 0.0},
}

# Create a dictionary to map the metric name to its position in the data
metric_position = {'R$^2$': 0, 'Slope': 1, 'Intercept': 2, 'Sensor RMSE': 3}

# Set the figure size and font scale
sns.set(rc={'figure.figsize':(20, 10)})
sns.set(font_scale=1.5)

averaging_intervals = ['1-hour', '24-hour']

# Loop over each averaging interval and create a subplot for each metric
for interval in averaging_intervals:
    fig, axs = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(20, 10))
    plt.subplots_adjust(hspace=0.4)

    for i, metric_name in enumerate(metrics):
        column_name = metric_bounds_goal[metric_name]['column']
        goal = metric_bounds_goal[metric_name]['goal']
        bounds = metric_bounds_goal[metric_name]['bounds']
        position = metric_position[metric_name]

        # Create the boxplot
        box_plot = sns.boxplot(x='Quarter', y=column_name, hue='Sensor Name', data=df[df['Averaging Interval'] == interval], ax=axs[i], width=0.6)

        # Remove the legend
        axs[i].get_legend().remove()

        # Add horizontal lines for the target, goal, and bounds
        axs[i].axhline(y=goal, linestyle='--', color='gray', linewidth=2)
        axs[i].axhspan(bounds[0], bounds[1], alpha=0.1, color='gray')

        # Adjust y-axis limits to show all data
        axs[i].set_ylim(box_plot.get_ylim())

        # Set the axis labels
        axs[i].set_xlabel('')
        axs[i].set_ylabel(metric_name)
        axs[i].tick_params(axis='x', labelrotation=90)

        # Add median values next to the boxes and color them based on the shaded areas
        for j, artist in enumerate(axs[i].artists):
            # Get the median value for this group
            median_group = df.loc[(df['Averaging Interval'] == interval) & (df['Quarter'] == df['Quarter'].unique()[j // 2])][column_name].median()

            # Calculate the x and y coordinates for the text
            x = artist.get_x() + artist.get_width() / 2
            y = artist.get_y() + artist.get_height()

            # Add the median value next to the box
            axs[i].text(x, y, f'{median_group:.2f}', fontsize='small', ha='center')

            # Color the box plots based on the shaded areas
            data = df[(df['Averaging Interval'] == interval) & (df['Quarter'] == df['Quarter'].unique()[j // 2])][column_name]
            if (data >= bounds[0]).all() and (data <= bounds[1]).all():
                artist.set_facecolor('green')
            else:
                artist.set_facecolor('red')

    # Add title and save the figure to a file
    plt.suptitle(f'{interval} Accuracy Metrics')
    plt.savefig(f'{interval}_metrics.png', bbox_inches='tight')









        
        
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({'font.size': 12})

# Replace file_path with the actual path to the csv file
file_path = r'C:\Users\jawan\Dissertation Data\Final Analysis\SM\Results\Precision Metrics\Accuracy Metrics\Accuracy_rev.csv'

# Read the csv file into a pandas DataFrame
df = pd.read_csv(file_path)

df['Sensor Name'] = df['Sensor Name'].replace("PurpleAir_PurpleAir-II_SM_Q0", "PurpleAir_PurpleAir-II_SM_Q0_uncorrected")
df['Sensor Name'] = df['Sensor Name'].replace("PurpleAir_PurpleAir-II_SM_Q1", "PurpleAir_PurpleAir-II_SM_Q1_uncorrected")
df['Sensor Name'] = df['Sensor Name'].replace("PurpleAir_PurpleAir-II_SM_Q2", "PurpleAir_PurpleAir-II_SM_Q2_uncorrected")
df['Sensor Name'] = df['Sensor Name'].replace("PurpleAir_PurpleAir-II_SM_Q3", "PurpleAir_PurpleAir-II_SM_Q3_uncorrected")
df['Sensor Name'] = df['Sensor Name'].replace("PurpleAir_PurpleAir-II_SM_Q4", "PurpleAir_PurpleAir-II_SM_Q4_uncorrected")

# Extract the Quarter and correction information from Sensor Name column
df['Quarter'] = df['Sensor Name'].str.split('_').str[-2] + '_' + df['Sensor Name'].str.split('_').str[-1]
df = df.fillna('')

# Define the metrics to plot
metrics = ['R$^2$', 'Slope', 'Intercept', 'Sensor RMSE']

# Define the target and goal ranges for each metric
metric_bounds_goal = {
    'R$^2$': {'column': 'R$^2$', 'bounds': (0.7, 1.0), 'goal': 1.0},
    'Slope': {'column': 'Slope', 'bounds': (0.65, 1.35), 'goal': 1.0},
    'Intercept': {'column': 'Intercept', 'bounds': (-5.0, 5.0), 'goal': 0.0},
    'Sensor RMSE': {'column': 'Sensor RMSE', 'bounds': (0.0, 7.0), 'goal': 0.0},
}

# Set the figure size and font scale
sns.set(rc={'figure.figsize':(20, 10)})
sns.set(font_scale=1.5)

averaging_intervals = ['1-hour', '24-hour']

# Loop over each metric and create a side-by-side subplot for each averaging interval
for metric_name in metrics:
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))
    plt.subplots_adjust(wspace=0.3)

    for i, interval in enumerate(averaging_intervals):
        column_name = metric_bounds_goal[metric_name]['column']
        goal = metric_bounds_goal[metric_name]['goal']
        bounds = metric_bounds_goal[metric_name]['bounds']

        # Create the boxplot
        box_plot = sns.boxplot(x='Quarter', y=column_name, hue='Sensor Name', data=df[df['Averaging Interval'] == interval], ax=axs[i], width=0.6)

        # Remove the legend
        axs[i].get_legend().remove()

        # Add horizontal lines for the target, goal, and bounds
        axs[i].axhline(y=goal, linestyle='--', color='gray', linewidth=2)
        axs[i].axhspan(bounds[0], bounds[1], alpha=0.1, color='gray')

        # Adjust y-axis limits to show all data
        axs[i].set_ylim(box_plot.get_ylim())

        # Set the axis labels
        axs[i].set_xlabel('Quarter')
        axs[i].set_ylabel(metric_name)
        axs[i].tick_params(axis='x', labelrotation=45)

        # Set the subplot title
        axs[i].set_title(f'{metric_name} Metrics for {interval}')

        # Add median values next to the boxes
        for j, artist in enumerate(axs[i].artists):
            # Get the median value for this group
            median_group = df.loc[(df['Averaging Interval'] == interval) & (df['Quarter'] == df['Quarter'].unique()[j // 2])][column_name].median()

            # Calculate the x and y coordinates for the text
            x = artist.get_x() + artist.get_width() / 2
            y = artist.get_y() + artist.get_height()

            # Add the median value next to the box
            axs[i].text(x, y, f'{median_group:.2f}', fontsize='small', ha='center')

    # Save the figure to a file
    plt.savefig(f'{metric_name}_metrics.png', bbox_inches='tight')



