# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 09:27:32 2023

@author: jawan
"""
import pandas as pd
import matplotlib.pyplot as plt

# Replace this with the path to your CSV file
csv_path = "C:/Users/jawan/Dissertation Data/Final Analysis/SM/Results/merged_df.csv"

# Read the CSV file into a Pandas DataFrame
data = pd.read_csv(csv_path)
# Convert the 'datetime' column to datetime objects with UTC timezone
data['datetime'] = pd.to_datetime(data['datetime'], utc=True)

# Sort the DataFrame by the 'datetime' column
data.sort_values(by='datetime', inplace=True)

# Set the 'datetime' column as the index of the DataFrame
data.set_index('datetime', inplace=True)

# Ask the user for the start and end dates
start_date = input("Enter start date (YYYY-MM-DD HH:MM:SS): ")
end_date = input("Enter end date (YYYY-MM-DD HH:MM:SS): ")

# Convert the input dates to datetime objects with UTC timezone
start_datetime = pd.to_datetime(start_date, utc=True)
end_datetime = pd.to_datetime(end_date, utc=True)

# Filter the DataFrame for the desired time interval
filtered_data = data.truncate(before=start_datetime, after=end_datetime)

# Plot the time series
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(filtered_data.index, filtered_data['Sample Measurement_x'], label='Reference $PM_{2.5}$')
ax.plot(filtered_data.index, filtered_data['PM25_avg'], label='LCS Average $PM_{2.5}$')

ax.set_xlabel('Time')
ax.set_ylabel('$PM_{2.5}$ values')
ax.set_title('Time Series for $PM_{2.5}$ values FRM vs. LCS')
ax.legend()

# Add dotted lines at 12ug and 35ug values on the y-axis
ax.axhline(y=12, color='gray', linestyle='--')
ax.axhline(y=35, color='gray', linestyle='--')

plt.show()

#Temp (F) plots
import pandas as pd
import matplotlib.pyplot as plt

# Replace this with the path to your CSV file
csv_path = "C:/Users/jawan/Dissertation Data/Final Analysis/SM/Results/merged_df.csv"

# Read the CSV file into a Pandas DataFrame
data = pd.read_csv(csv_path)
# Convert the 'datetime' column to datetime objects with UTC timezone
data['datetime'] = pd.to_datetime(data['datetime'], utc=True)

# Sort the DataFrame by the 'datetime' column
data.sort_values(by='datetime', inplace=True)

# Set the 'datetime' column as the index of the DataFrame
data.set_index('datetime', inplace=True)

# Define a function to convert temperature from Celsius to Fahrenheit
def celsius_to_fahrenheit(celsius):
    fahrenheit = (celsius * 9/5) + 32
    return fahrenheit

# Apply the conversion function to the "Temp_C" column and add a new "Temp F" column
data["Temp F"] = data["Temp_C"].apply(celsius_to_fahrenheit)

# Prompt the user to enter the start and end time range
start_time = input("Enter start time (YYYY-MM-DD HH:MM:SS): ")
end_time = input("Enter end time (YYYY-MM-DD HH:MM:SS): ")

# Convert the input times to datetime objects with UTC timezone
start_datetime = pd.to_datetime(start_time, utc=True)
end_datetime = pd.to_datetime(end_time, utc=True)

# Filter the DataFrame for the desired time interval
filtered_data = data.loc[start_datetime:end_datetime]

# Filter the DataFrame to keep only values between 30 and 100
filtered_data = filtered_data[(filtered_data['Temp F'] >= 30) & (filtered_data['Temp F'] <= 100)]

# Plot the time series
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(filtered_data.index, filtered_data['Sample Measurement_y'], label='Reference Temperature (F)')
ax.plot(filtered_data.index, filtered_data['Temp F'], label='LCS Average Temperature (F)')

ax.set_xlabel('Time')
ax.set_ylabel('Temperature (F)')
ax.set_title('Time Series for Temperature (F) FRM vs. LCS')
ax.legend()


plt.show()

#R.H plots
import pandas as pd
import matplotlib.pyplot as plt

# Replace this with the path to your CSV file
csv_path = "C:/Users/jawan/Dissertation Data/Final Analysis/SM/Results/merged_df.csv"

# Read the CSV file into a Pandas DataFrame
data = pd.read_csv(csv_path)
# Convert the 'datetime' column to datetime objects with UTC timezone
data['datetime'] = pd.to_datetime(data['datetime'], utc=True)

# Sort the DataFrame by the 'datetime' column
data.sort_values(by='datetime', inplace=True)

# Set the 'datetime' column as the index of the DataFrame
data.set_index('datetime', inplace=True)


# Prompt the user to enter the start and end time range
start_time = input("Enter start time (YYYY-MM-DD HH:MM:SS): ")
end_time = input("Enter end time (YYYY-MM-DD HH:MM:SS): ")

# Convert the input times to datetime objects with UTC timezone
start_datetime = pd.to_datetime(start_time, utc=True)
end_datetime = pd.to_datetime(end_time, utc=True)

# Filter the DataFrame for the desired time interval
filtered_data = data.loc[start_datetime:end_datetime]

# Filter the DataFrame to keep only values between 30 and 100
#filtered_data = filtered_data[(filtered_data['Temp F'] >= 30) & (filtered_data['Temp F'] <= 100)]

# Plot the time series
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(filtered_data.index, filtered_data['Sample Measurement'], label='Reference R.H (%)')
ax.plot(filtered_data.index, filtered_data['Humidity_%'], label='LCS Average R.H (%)')

ax.set_xlabel('Time')
ax.set_ylabel('R.H (%)')
ax.set_title('Time Series for R.H (%) FRM vs. LCS')
ax.legend()


plt.show()



# Timeseries Plot
import folium
from folium.plugins import HeatMapWithTime

# Convert the 'datetime' column to string format (required by the HeatMapWithTime plugin)
data['datetime'] = data['datetime'].astype(str)

# Group the data by datetime
grouped_data = data.groupby('datetime')

# Create a list of data for each time step
heat_data = []
for time, group in grouped_data:
    heat_data.append(group[['Latitude_x', 'Longitude_x', 'Sample Measurement_x']].values.tolist())

# Create a base map
m = folium.Map(location=[data['Latitude_x'].mean(), data['Longitude_x'].mean()], zoom_start=10)

# Add HeatMapWithTime to the map
heatmap = HeatMapWithTime(heat_data, index=list(grouped_data.groups.keys()), auto_play=True, max_opacity=0.6)
heatmap.add_to(m)

# Add markers with popups for each sensor location and PM2.5 value
for index, row in data.iterrows():
    folium.Marker(
        location=[row['Latitude_x'], row['Longitude_x']],
        popup=f"PM2.5: {row['Sample Measurement_x']}",
        icon=None
    ).add_to(m)

# Save the map to an HTML file
m.save('heat_map_with_time_markers.html')






import folium
import numpy as np
from folium.plugins import HeatMapWithTime
from tqdm import tqdm
import branca.colormap as cm
import pandas as pd
import matplotlib.pyplot as plt

# Replace this with the path to your CSV file
csv_path = "C:/Users/jawan/Dissertation Data/Final Analysis/SM/Results/merged_df.csv"

# Read the CSV file into a Pandas DataFrame
data = pd.read_csv(csv_path)



# Grid boundaries for Los Angeles region
la_min_lat, la_max_lat = 33.7, 34.3
la_min_lon, la_max_lon = -118.7, -118

# Create a grid for IDW interpolation (reduced resolution to 50 x 50)
grid_x, grid_y = np.mgrid[la_min_lat:la_max_lat:50j,
                          la_min_lon:la_max_lon:50j]


def idw_interpolation(data, grid_x, grid_y, power=2):
    coords = np.array([[data['Latitude_x'].iloc[0], data['Longitude_x'].iloc[0]]])
    grid_points = np.vstack([grid_x.flatten(), grid_y.flatten()]).T

    weights = np.zeros((len(data), grid_points.shape[0]))
    for i, value in enumerate(data['Sample Measurement_x'].values):
        weights[i] = 1 / (np.sum((coords - grid_points)**2, axis=-1) + 1e-8)**power * value

    weighted_sum = np.sum(weights, axis=0)
    weights_sum = np.sum(weights != 0, axis=0)

    # Handle division when weights_sum contains zeros
    return np.where(weights_sum != 0, weighted_sum / weights_sum, 0)

# Remove rows with NaN values in the 'Sample Measurement_x' column
data = data.dropna(subset=['Sample Measurement_x'])

# Group the data by datetime
grouped_data = data.groupby('datetime')

# Create a grid for IDW interpolation (reduced resolution to 50 x 50)
grid_x, grid_y = np.mgrid[data['Latitude_x'].min() - 0.1:data['Latitude_x'].max() + 0.1:50j,
                          data['Longitude_x'].min() - 0.1:data['Longitude_x'].max() + 0.1:50j]

# Create a list of data for each time step
heat_data = []
for time, group in tqdm(grouped_data, desc="Interpolating data"):
    interpolated_data = idw_interpolation(group, grid_x, grid_y)
    heat_data.append(np.vstack([grid_x.flatten(), grid_y.flatten(), interpolated_data.flatten()]).T.tolist())

# Create a base map
m = folium.Map(location=[data['Latitude_x'].mean(), data['Longitude_x'].mean()], zoom_start=10)

# Add HeatMapWithTime to the map
heatmap = HeatMapWithTime(heat_data, index=list(grouped_data.groups.keys()), auto_play=True, max_opacity=0.6)
heatmap.add_to(m)

# Define a color scale based on the PM2.5 values range
color_scale = cm.LinearColormap(['green', 'yellow', 'orange', 'red', 'purple'], vmin=0, vmax=500)
color_scale.caption = "PM2.5 Values"
color_scale.add_to(m)

# Save the map to an HTML file
m.save('idw_interpolation_heatmap_time.html')









import folium
import pandas as pd
from folium.plugins import TimestampedGeoJson
from folium.map import Popup
from branca.colormap import LinearColormap

def create_feature(time, lat, lon, value):
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
        "properties": {
            "value": value,
            "time": time,
            "style": {"fillColor": color_scale(value), "color": color_scale(value), "fillOpacity": 0.5},
            "icon": "circle",
            "iconstyle": {"radius": 15},
        },
    }

# Remove rows with NaN values in the 'Sample Measurement_x' column
data = data.dropna(subset=['Sample Measurement_x'])

# Group the data by datetime
grouped_data = data.groupby('datetime')

# Define a color scale based on the PM2.5 values range
color_scale = LinearColormap(['green', 'yellow', 'orange', 'red', 'purple'], vmin=0, vmax=500)

# Create a list of GeoJSON features for each time step
features = []
for time, group in grouped_data:
    for _, row in group.iterrows():
        features.append(create_feature(time, row['Latitude_x'], row['Longitude_x'], row['Sample Measurement_x']))

# Create a GeoJSON feature collection
feature_collection = {"type": "FeatureCollection", "features": features}

# Create a base map
m = folium.Map(location=[data['Latitude_x'].mean(), data['Longitude_x'].mean()], zoom_start=10)

# Add TimestampedGeoJson to the map
timestamped_geojson = TimestampedGeoJson(
    feature_collection,
    period="PT1H",
    add_last_point=True,
    auto_play=True,
    loop=False,
    max_speed=10,
    loop_button=True,
    date_options="YYYY-MM-DD HH:mm",
    time_slider_drag_update=True,
)
timestamped_geojson.add_to(m)

# Add CircleMarkers with popups
for _, row in data.iterrows():
    folium.CircleMarker(
        location=[row['Latitude_x'], row['Longitude_x']],
        radius=15,
        color=color_scale(row['Sample Measurement_x']),
        fill=True,
        fill_opacity=0.5,
        popup=Popup(
            '<strong>PM2.5:</strong> {:.2f}'.format(row['Sample Measurement_x']),
            show=True,
            sticky=True,
            max_width="100%",
        ),
    ).add_to(m)

# Add the color scale to the map
color_scale.caption = "PM2.5 Values"
color_scale.add_to(m)

# Save the map to an HTML file
m.save('circle_interpolation_heatmap_time_with_values.html')