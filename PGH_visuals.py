# -*- coding: utf-8 -*-
"""


@author: jawan
"""
import folium
import os
import pandas as pd
from tqdm import tqdm


start_date = pd.to_datetime('2019-03-01').tz_localize('UTC')
end_date = pd.to_datetime('2019-07-01').tz_localize('UTC')

start_date = pd.to_datetime('2019-07-01').tz_localize('UTC')
end_date = pd.to_datetime('2019-10-01').tz_localize('UTC')

start_date = pd.to_datetime('2019-10-01').tz_localize('UTC')
end_date = pd.to_datetime('2019-12-31').tz_localize('UTC')

start_date = pd.to_datetime('2020-01-01').tz_localize('UTC')
end_date = pd.to_datetime('2020-03-31').tz_localize('UTC')

start_date = pd.to_datetime('2020-03-31').tz_localize('UTC')
end_date = pd.to_datetime('2020-07-01').tz_localize('UTC')






start_date = pd.to_datetime('2019-07-01').tz_localize('UTC')
end_date = pd.to_datetime('2019-10-01').tz_localize('UTC')

dfs = []
sensor_names = ['Sensor'+str(i) for i in range(1, 31)]

for sensor_name in tqdm(sensor_names, desc="Processing sensors"):
    filename = f"r"C:\Users\<Enter your local directory for Data>""
    try:
        df = pd.read_csv(filename)
        # Extract 'CAZIER-PA-XXX' format from file name and add 'sensor_name' column
        cazier_name = f"CAZIER-PA-{int(sensor_name[6:]):03d}"
        df['sensor_name'] = cazier_name
        df['created_at'] = pd.to_datetime(df['created_at']).dt.tz_convert('UTC')
        df = df[(df['created_at'] >= start_date) & (df['created_at'] <= end_date)]
        if not df.empty:
            dfs.append(df)
        else:
            print(f"Empty DataFrame for {sensor_name}. Skipping and moving on...")
    except FileNotFoundError:
        print(f"File not found: {filename}. Skipping and moving on...")

        
        
filename = r"C:\Users\<Enter your local directory for Regulatory Data>"
df_reg = pd.read_csv(filename)  



import folium
import requests
from folium.features import DivIcon

# Creating map
map_pittsburgh = folium.Map(location=[40.4406, -79.9959], zoom_start=13)


# Use a simplified map style
folium.TileLayer(
    tiles="https://{s}.basemaps.cartocdn.com/rastertiles/voyager_labels_under/{z}/{x}/{y}{r}.png",
    attr="&copy; <a href=https://www.openstreetmap.org/copyright>OpenStreetMap</a> contributors &copy; <a href=https://carto.com/attributions>CartoDB</a>",
    name="simplified",
    control=False,
).add_to(map_pittsburgh)

import numpy as np

def get_centroid(coordinates):
    x_coords = [c[0] for c in coordinates]
    y_coords = [c[1] for c in coordinates]
    centroid_x = np.mean(x_coords)
    centroid_y = np.mean(y_coords)
    return [centroid_y, centroid_x]




sensors = [
    ("CAZIER-PA-001", 40.461101, -79.936179),
    ("CAZIER-PA-002", 40.427875, -79.920128),
    ("CAZIER-PA-004", 40.44493, -79.932105),
    ("CAZIER-PA-005", 40.4244, -79.930194),
    ("CAZIER-PA-006", 40.44914, -79.908846),
    ("CAZIER-PA-007", 40.439452, -79.933105),
    ("CAZIER-PA-008", 40.452064, -79.905733),
    ("CAZIER-PA-009", 40.471307, -79.916907),
    ("CAZIER-PA-010", 40.474471, -79.936631),
    ("CAZIER-PA-011", 40.425227, -79.924223),
    ("CAZIER-PA-013", 40.446356, -79.899321),
    ("CAZIER-PA-014", 40.432356, -79.898454),
    ("CAZIER-PA-015", 40.434165, -79.920852),
    ("CAZIER-PA-016", 40.440692, -79.92176),
    ("CAZIER-PA-017", 40.447093, -79.912005),
    ("CAZIER-PA-019", 40.453291, -79.922502),
    ("CAZIER-PA-020", 40.444325, -79.933592),
    ("CAZIER-PA-021", 40.477681, -79.915938),
    ("CAZIER-PA-022", 40.416788, -79.880289),
    ("CAZIER-PA-023", 40.436147, -79.916992),
    ("CAZIER-PA-024", 40.439492, -79.921827),
    ("CAZIER-PA-026", 40.447085, -79.91206)
]

sensor_zip_codes = {
    "CAZIER-PA-014": "15218",
    "CAZIER-PA-022": "15218",
    "CAZIER-PA-010": "15201",
    "CAZIER-PA-009": "15206",
    "CAZIER-PA-021": "15206",
    "CAZIER-PA-019": "15232",
    "CAZIER-PA-001": "15232",
    "CAZIER-PA-026": "15208",
    "CAZIER-PA-017": "15208",
    "CAZIER-PA-006": "15208",
    "CAZIER-PA-008": "15208",
    "CAZIER-PA-013": "15208",
    "CAZIER-PA-002": "15217",
    "CAZIER-PA-004": "15217",
    "CAZIER-PA-005": "15217",
    "CAZIER-PA-007": "15217",
    "CAZIER-PA-011": "15217",
    "CAZIER-PA-015": "15217",
    "CAZIER-PA-016": "15217",
    "CAZIER-PA-020": "15217",
    "CAZIER-PA-023": "15217",
    "CAZIER-PA-024": "15217"
}

# Add sensor markers to the map as circles and display the name next to each sensor
for sensor_id, lat, lng in sensors:
    folium.Circle(
        location=[lat, lng],
        radius=20,
        color="#3186cc",
        fill=True,
        fill_color="#3186cc",
    ).add_to(map_pittsburgh)
    folium.Marker(
        [lat, lng],
        icon=DivIcon(
            icon_size=(150, 36),
            icon_anchor=(10, 0),
            html=f'<div style="font-size: 8pt; color : #3186cc; font-weight: bold">{sensor_id}</div>',
        ),
    ).add_to(map_pittsburgh)

# Download the GeoJSON file containing the zip codes for Pittsburgh, PA
url = "https://raw.githubusercontent.com/OpenDataDE/State-zip-code-GeoJSON/master/pa_pennsylvania_zip_codes_geo.min.json"
response = requests.get(url, verify=False)
zip_geojson = response.json()

# Add the zip codes as a layer on the map with black boundaries
def style_function(feature):
    zip_code = feature["properties"]["ZCTA5CE10"]
    if zip_code in sensor_zip_codes.values():
        return {
            "fillOpacity": 0,
            "weight": 2,
            "color": "black",
        }
    else:
        return {
            "fillOpacity": 0,
            "weight": 0,
            "color": "none",
        }

zip_layer = folium.GeoJson(
    zip_geojson,
    style_function=style_function,
    name="Zip Codes"
)

for _, feature in enumerate(zip_layer.data["features"]):
    zip_code = feature["properties"]["ZCTA5CE10"]
    if zip_code in sensor_zip_codes.values():
        coordinates = feature["geometry"]["coordinates"][0]
        centroid = get_centroid(coordinates)
        folium.Marker(
            centroid,
            icon=DivIcon(
                icon_size=(150, 36),
                icon_anchor=(10, 0),
                html=f'<div style="font-size: 16pt; color : black; font-weight: bold">{zip_code}</div>',
            ),
        ).add_to(map_pittsburgh)

zip_layer.add_to(map_pittsburgh)
map_pittsburgh.save("pittsburgh_sensors421.html")









import folium
import requests
import numpy as np
import pandas as pd
import branca.colormap as cm
from folium.features import DivIcon
from tqdm import tqdm


dfs = []
sensor_names = ['Sensor'+str(i) for i in range(1, 31)]

for sensor_name in tqdm(sensor_names, desc="Processing sensors"):
    filename = r"C:\Users\<Enter your local directory for Regulatory Data>"
    try:
        df = pd.read_csv(filename)
        # Extract 'CAZIER-PA-XXX' format from file name and add 'sensor_name' column
        cazier_name = f"CAZIER-PA-{int(sensor_name[6:]):03d}"
        df['sensor_name'] = cazier_name
        df['created_at'] = pd.to_datetime(df['created_at']).dt.tz_convert('UTC')
        df = df[(df['created_at'] >= start_date) & (df['created_at'] <= end_date)]
        if not df.empty:
            dfs.append(df)
        else:
            print(f"Empty DataFrame for {sensor_name}. Skipping and moving on...")
    except FileNotFoundError:
        print(f"File not found: {filename}. Skipping and moving on...")


df_reg['DateTime'] = pd.to_datetime(df_reg['DateTime'], format='%Y-%m-%d %H:%M:%S').dt.tz_localize('UTC')


# Filter df_reg to only include rows within the desired date range
df_reg = df_reg[(df_reg['DateTime'] >= start_date) & (df_reg['DateTime'] <= end_date)]

# Then compute the mean
reg_avg_pm25 = df_reg['Sample Measurement_PM'].mean()



# Compute overall average PM2.5 value from regulatory monitor data
reg_avg_pm25 = df_reg['Sample Measurement_PM'].mean()

# Calculate PM2.5 average values for each sensor by zip code
sensor_pm25 = pd.concat(dfs)
sensor_pm25['zip_code'] = sensor_pm25['sensor_name'].map(sensor_zip_codes)
zip_pm25 = sensor_pm25.groupby('zip_code')['PM25_avg'].mean()

# Create a colormap for coloring zip codes based on their average PM2.5 values
min_pm25 = zip_pm25.min()
max_pm25 = zip_pm25.max()
colormap = cm.LinearColormap(
    colors=['green', 'yellow', 'red'],
    index=[min_pm25, reg_avg_pm25, max_pm25],
    vmin=min_pm25,
    vmax=max_pm25,
)

# Add zip code areas to the map and color them according to their average PM2.5 values
def style_function(feature):
    zip_code = feature['properties']['ZCTA5CE10']
    if zip_code in zip_pm25.index:
        return {
            'fillOpacity': 0.7,
            'weight': 2,
            'color': 'black',
            'fillColor': colormap(zip_pm25[zip_code]),
        }
    else:
        return {
            'fillOpacity': 0,
            'weight': 0,
            'color': 'none',
        }

zip_layer = folium.GeoJson(
    zip_geojson,
    style_function=style_function,
    name='Zip Codes'
)

for _, feature in enumerate(zip_layer.data['features']):
    zip_code = feature['properties']['ZCTA5CE10']
    if zip_code in zip_pm25.index:
        coordinates = feature['geometry']['coordinates'][0]
        centroid = get_centroid(coordinates)
        folium.Marker(
            centroid,
            icon=DivIcon(
                icon_size=(150, 36),
                icon_anchor=(10, 0),
                html=f'<div style="font-size: 16pt; color : black; font-weight: bold">{zip_code}</div>',
            ),
        ).add_to(map_pittsburgh)

zip_layer.add_to(map_pittsburgh)

# Add a colorbar legend to the map
colormap.caption = 'PM 2.5 Average Concentration'
colormap.add_to(map_pittsburgh)

# Save the map as an HTML file
map_pittsburgh.save("pittsburgh_sensors_pm25_88_Quart 2.html")



# Create a DataFrame from the zip_pm25 series
df_pm25 = zip_pm25.reset_index()
df_pm25.columns = ['ZipCode', 'PM2.5_avg_LCS']  # LCS - Low-cost sensors

# Add a column for the regulatory average
df_pm25['PM2.5_avg_Reg'] = reg_avg_pm25

# Calculate the percent difference
df_pm25['% Difference'] = ((df_pm25['PM2.5_avg_LCS'] - df_pm25['PM2.5_avg_Reg']) / df_pm25['PM2.5_avg_Reg']) * 100

# Print the DataFrame
print(df_pm25)



# Round the values in the DataFrame to 3 decimal places
df_pm25 = df_pm25.round({
    'PM$_{2.5}$ avg LCS': 3,
    'PM$_{2.5}$ avg Reg': 3,
    '% Difference': 3
})

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1)

# Replace the column names with LaTeX-style markup for subscripts
df_pm25.columns = ['ZipCode', 'PM$_{2.5}$ avg LCS', 'PM$_{2.5}$ avg Reg', '% Difference']

table_data = [df_pm25.columns.to_list()] + df_pm25.values.tolist()

table = ax.table(cellText=table_data, loc='center')

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)

ax.axis('off')

plt.savefig("table.png")


#Quarter 3


# -*- coding: utf-8 -*-
"""
Created on Tue May  9 09:26:31 2023

@author: jawan
"""
import folium
import os
import pandas as pd
from tqdm import tqdm


start_date = pd.to_datetime('2019-03-01').tz_localize('UTC')
end_date = pd.to_datetime('2019-07-01').tz_localize('UTC')

start_date = pd.to_datetime('2019-07-01').tz_localize('UTC')
end_date = pd.to_datetime('2019-10-01').tz_localize('UTC')

start_date = pd.to_datetime('2019-10-01').tz_localize('UTC')
end_date = pd.to_datetime('2019-12-31').tz_localize('UTC')

start_date = pd.to_datetime('2020-01-01').tz_localize('UTC')
end_date = pd.to_datetime('2020-03-31').tz_localize('UTC')

start_date = pd.to_datetime('2020-03-31').tz_localize('UTC')
end_date = pd.to_datetime('2020-07-01').tz_localize('UTC')



start_date = pd.to_datetime('2020-03-31').tz_localize('UTC')
end_date = pd.to_datetime('2020-07-01').tz_localize('UTC')
dfs = []
sensor_names = ['Sensor'+str(i) for i in range(1, 31)]

for sensor_name in tqdm(sensor_names, desc="Processing sensors"):
    filename = f"C:\\Users\\jawan\\Dissertation Data\\Final Datasets\\PGH\\LCS_PGH_2018_2019\\Quarter 4\\{sensor_name}_no_outliers_averaged.csv"
    try:
        df = pd.read_csv(filename)
        # Extract 'CAZIER-PA-XXX' format from file name and add 'sensor_name' column
        cazier_name = f"CAZIER-PA-{int(sensor_name[6:]):03d}"
        df['sensor_name'] = cazier_name
        df['created_at'] = pd.to_datetime(df['created_at']).dt.tz_convert('UTC')
        df = df[(df['created_at'] >= start_date) & (df['created_at'] <= end_date)]
        if not df.empty:
            dfs.append(df)
        else:
            print(f"Empty DataFrame for {sensor_name}. Skipping and moving on...")
    except FileNotFoundError:
        print(f"File not found: {filename}. Skipping and moving on...")

        
        
filename = r"C:\Users\jawan\Dissertation Data\Final Datasets\PGH\Regulatory_PGH_2018_2019\PGH_Regulatory.csv"
df_reg = pd.read_csv(filename)  



import folium
import requests
from folium.features import DivIcon

# Creating map
map_pittsburgh = folium.Map(location=[40.4406, -79.9959], zoom_start=13)


# Use a simplified map style
folium.TileLayer(
    tiles="https://{s}.basemaps.cartocdn.com/rastertiles/voyager_labels_under/{z}/{x}/{y}{r}.png",
    attr="&copy; <a href=https://www.openstreetmap.org/copyright>OpenStreetMap</a> contributors &copy; <a href=https://carto.com/attributions>CartoDB</a>",
    name="simplified",
    control=False,
).add_to(map_pittsburgh)

import numpy as np

def get_centroid(coordinates):
    x_coords = [c[0] for c in coordinates]
    y_coords = [c[1] for c in coordinates]
    centroid_x = np.mean(x_coords)
    centroid_y = np.mean(y_coords)
    return [centroid_y, centroid_x]




sensors = [
    ("CAZIER-PA-001", 40.461101, -79.936179),
    ("CAZIER-PA-002", 40.427875, -79.920128),
    ("CAZIER-PA-004", 40.44493, -79.932105),
    ("CAZIER-PA-005", 40.4244, -79.930194),
    ("CAZIER-PA-006", 40.44914, -79.908846),
    ("CAZIER-PA-007", 40.439452, -79.933105),
    ("CAZIER-PA-008", 40.452064, -79.905733),
    ("CAZIER-PA-009", 40.471307, -79.916907),
    ("CAZIER-PA-010", 40.474471, -79.936631),
    ("CAZIER-PA-011", 40.425227, -79.924223),
    ("CAZIER-PA-013", 40.446356, -79.899321),
    ("CAZIER-PA-014", 40.432356, -79.898454),
    ("CAZIER-PA-015", 40.434165, -79.920852),
    ("CAZIER-PA-016", 40.440692, -79.92176),
    ("CAZIER-PA-017", 40.447093, -79.912005),
    ("CAZIER-PA-019", 40.453291, -79.922502),
    ("CAZIER-PA-020", 40.444325, -79.933592),
    ("CAZIER-PA-021", 40.477681, -79.915938),
    ("CAZIER-PA-022", 40.416788, -79.880289),
    ("CAZIER-PA-023", 40.436147, -79.916992),
    ("CAZIER-PA-024", 40.439492, -79.921827),
    ("CAZIER-PA-026", 40.447085, -79.91206)
]

sensor_zip_codes = {
    "CAZIER-PA-014": "15218",
    "CAZIER-PA-022": "15218",
    "CAZIER-PA-010": "15201",
    "CAZIER-PA-009": "15206",
    "CAZIER-PA-021": "15206",
    "CAZIER-PA-019": "15232",
    "CAZIER-PA-001": "15232",
    "CAZIER-PA-026": "15208",
    "CAZIER-PA-017": "15208",
    "CAZIER-PA-006": "15208",
    "CAZIER-PA-008": "15208",
    "CAZIER-PA-013": "15208",
    "CAZIER-PA-002": "15217",
    "CAZIER-PA-004": "15217",
    "CAZIER-PA-005": "15217",
    "CAZIER-PA-007": "15217",
    "CAZIER-PA-011": "15217",
    "CAZIER-PA-015": "15217",
    "CAZIER-PA-016": "15217",
    "CAZIER-PA-020": "15217",
    "CAZIER-PA-023": "15217",
    "CAZIER-PA-024": "15217"
}

# Add sensor markers to the map as circles and display the name next to each sensor
for sensor_id, lat, lng in sensors:
    folium.Circle(
        location=[lat, lng],
        radius=20,
        color="#3186cc",
        fill=True,
        fill_color="#3186cc",
    ).add_to(map_pittsburgh)
    folium.Marker(
        [lat, lng],
        icon=DivIcon(
            icon_size=(150, 36),
            icon_anchor=(10, 0),
            html=f'<div style="font-size: 8pt; color : #3186cc; font-weight: bold">{sensor_id}</div>',
        ),
    ).add_to(map_pittsburgh)

# Download the GeoJSON file containing the zip codes for Pittsburgh, PA
url = "https://raw.githubusercontent.com/OpenDataDE/State-zip-code-GeoJSON/master/pa_pennsylvania_zip_codes_geo.min.json"
response = requests.get(url, verify=False)
zip_geojson = response.json()

# Add the zip codes as a layer on the map with black boundaries
def style_function(feature):
    zip_code = feature["properties"]["ZCTA5CE10"]
    if zip_code in sensor_zip_codes.values():
        return {
            "fillOpacity": 0,
            "weight": 2,
            "color": "black",
        }
    else:
        return {
            "fillOpacity": 0,
            "weight": 0,
            "color": "none",
        }

zip_layer = folium.GeoJson(
    zip_geojson,
    style_function=style_function,
    name="Zip Codes"
)

for _, feature in enumerate(zip_layer.data["features"]):
    zip_code = feature["properties"]["ZCTA5CE10"]
    if zip_code in sensor_zip_codes.values():
        coordinates = feature["geometry"]["coordinates"][0]
        centroid = get_centroid(coordinates)
        folium.Marker(
            centroid,
            icon=DivIcon(
                icon_size=(150, 36),
                icon_anchor=(10, 0),
                html=f'<div style="font-size: 16pt; color : black; font-weight: bold">{zip_code}</div>',
            ),
        ).add_to(map_pittsburgh)

zip_layer.add_to(map_pittsburgh)
map_pittsburgh.save("pittsburgh_sensors424.html")









import folium
import requests
import numpy as np
import pandas as pd
import branca.colormap as cm
from folium.features import DivIcon
from tqdm import tqdm


dfs = []
sensor_names = ['Sensor'+str(i) for i in range(1, 31)]

for sensor_name in tqdm(sensor_names, desc="Processing sensors"):
    filename = f"C:\\Users\\jawan\\Dissertation Data\\Final Datasets\\PGH\\LCS_PGH_2018_2019\\Quarter 4\\{sensor_name}_no_outliers_averaged.csv"
    try:
        df = pd.read_csv(filename)
        # Extract 'CAZIER-PA-XXX' format from file name and add 'sensor_name' column
        cazier_name = f"CAZIER-PA-{int(sensor_name[6:]):03d}"
        df['sensor_name'] = cazier_name
        df['created_at'] = pd.to_datetime(df['created_at']).dt.tz_convert('UTC')
        df = df[(df['created_at'] >= start_date) & (df['created_at'] <= end_date)]
        if not df.empty:
            dfs.append(df)
        else:
            print(f"Empty DataFrame for {sensor_name}. Skipping and moving on...")
    except FileNotFoundError:
        print(f"File not found: {filename}. Skipping and moving on...")


df_reg['DateTime'] = pd.to_datetime(df_reg['DateTime'], format='%Y-%m-%d %H:%M:%S').dt.tz_localize('UTC')


# Filter df_reg to only include rows within the desired date range
df_reg = df_reg[(df_reg['DateTime'] >= start_date) & (df_reg['DateTime'] <= end_date)]

# Then compute the mean
reg_avg_pm25 = df_reg['Sample Measurement_PM'].mean()



# Compute overall average PM2.5 value from regulatory monitor data
reg_avg_pm25 = df_reg['Sample Measurement_PM'].mean()

# Calculate PM2.5 average values for each sensor by zip code
sensor_pm25 = pd.concat(dfs)
sensor_pm25['zip_code'] = sensor_pm25['sensor_name'].map(sensor_zip_codes)
zip_pm25 = sensor_pm25.groupby('zip_code')['PM25_avg'].mean()

# Create a colormap for coloring zip codes based on their average PM2.5 values
min_pm25 = zip_pm25.min()
max_pm25 = zip_pm25.max()
colormap = cm.LinearColormap(
    colors=['green', 'yellow', 'red'],
    index=[min_pm25, reg_avg_pm25, max_pm25],
    vmin=min_pm25,
    vmax=max_pm25,
)

# Add zip code areas to the map and color them according to their average PM2.5 values
def style_function(feature):
    zip_code = feature['properties']['ZCTA5CE10']
    if zip_code in zip_pm25.index:
        return {
            'fillOpacity': 0.7,
            'weight': 2,
            'color': 'black',
            'fillColor': colormap(zip_pm25[zip_code]),
        }
    else:
        return {
            'fillOpacity': 0,
            'weight': 0,
            'color': 'none',
        }

zip_layer = folium.GeoJson(
    zip_geojson,
    style_function=style_function,
    name='Zip Codes'
)

for _, feature in enumerate(zip_layer.data['features']):
    zip_code = feature['properties']['ZCTA5CE10']
    if zip_code in zip_pm25.index:
        coordinates = feature['geometry']['coordinates'][0]
        centroid = get_centroid(coordinates)
        folium.Marker(
            centroid,
            icon=DivIcon(
                icon_size=(150, 36),
                icon_anchor=(10, 0),
                html=f'<div style="font-size: 16pt; color : black; font-weight: bold">{zip_code}</div>',
            ),
        ).add_to(map_pittsburgh)

zip_layer.add_to(map_pittsburgh)

# Add a colorbar legend to the map
colormap.caption = 'PM 2.5 Average Concentration'
colormap.add_to(map_pittsburgh)

# Save the map as an HTML file
map_pittsburgh.save("pittsburgh_sensors_pm25_88_Quart 4.html")



# Create a DataFrame from the zip_pm25 series
df_pm25 = zip_pm25.reset_index()
df_pm25.columns = ['ZipCode', 'PM2.5_avg_LCS']  # LCS - Low-cost sensors

# Add a column for the regulatory average
df_pm25['PM2.5_avg_Reg'] = reg_avg_pm25

# Calculate the percent difference
df_pm25['% Difference'] = ((df_pm25['PM2.5_avg_LCS'] - df_pm25['PM2.5_avg_Reg']) / df_pm25['PM2.5_avg_Reg']) * 100

# Print the DataFrame
print(df_pm25)



# Round the values in the DataFrame to 3 decimal places
df_pm25 = df_pm25.round({
    'PM$_{2.5}$ avg LCS': 3,
    'PM$_{2.5}$ avg Reg': 3,
    '% Difference': 3
})

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1)

# Replace the column names with LaTeX-style markup for subscripts
df_pm25.columns = ['ZipCode', 'PM$_{2.5}$ avg LCS($\mu$g/m$^3$)', 'PM$_{2.5}$ avg Reg($\mu$g/m$^3$)', '% Difference']

table_data = [df_pm25.columns.to_list()] + df_pm25.values.tolist()

table = ax.table(cellText=table_data, loc='center')

table.auto_set_font_size(False)
table.set_fontsize(6.5)
table.scale(1, 1.5)

ax.axis('off')

plt.savefig("table4.png")




