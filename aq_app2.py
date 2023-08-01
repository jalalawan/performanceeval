import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from collections import defaultdict

# Set the path and file names
data_path = r'C:\Users\awan\Dissertation_2023\Dissertation Data\Final Datasets\SM\\'
sensor_folder = 'LCS_SM_2018_2019'
monitor_folder = 'Regulatory_SM_2018_2019'
sensor_files = []
monitor_files = []

# Collect sensor file paths
for quarter in range(0, 5):
    folder_path = os.path.join(data_path, sensor_folder, f'Quarter {quarter}')
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv') and 'no_outliers_averaged' in file_name:
            file_path = os.path.join(folder_path, file_name)
            sensor_files.append((file_path, quarter))  # Add quarter information

# Collect monitor file paths
monitor_folder_path = os.path.join(data_path, monitor_folder)
for file_name in os.listdir(monitor_folder_path):
    if file_name.endswith('.csv'):
        file_path = os.path.join(monitor_folder_path, file_name)
        monitor_files.append(file_path)

# Read and process the monitor files
df_monitor = pd.DataFrame()
for monitor_file in monitor_files:
    df = pd.read_csv(monitor_file, parse_dates=['Date GMT_x', 'Date GMT_y'], low_memory=False)
    df['Sample Measurement_y'] = (df['Sample Measurement_y'] - 32) * (5/9)
    df.rename(columns={'Sample Measurement_x': 'Monitor PM2.5',
                       'Sample Measurement_y': 'Monitor Temp',
                       'Sample Measurement': 'Monitor Humidity'},
              inplace=True)
    df_monitor = pd.concat([df_monitor, df[['Date GMT_x', 'Monitor PM2.5', 'Monitor Temp', 'Monitor Humidity']]], ignore_index=True)

df_monitor.set_index('Date GMT_x', inplace=True)
df_monitor.index = df_monitor.index.tz_localize(None)

# Initialize figures
fig_pm = go.Figure()
fig_temp = go.Figure()
fig_humidity = go.Figure()

# Collect start date and end date from the user
start_date = pd.to_datetime(st.date_input("Select a start date", value=pd.to_datetime('2018-10-01'), min_value=pd.to_datetime('2018-10-01'), max_value=pd.to_datetime('2019-12-31')))
end_date = pd.to_datetime(st.date_input("Select an end date", value=pd.to_datetime('2018-12-31'), min_value=start_date, max_value=pd.to_datetime('2019-12-31')))

def process_and_plot_sensor_data():
    average_sensor_pm25 = defaultdict(list)
    sensor_pm25_data = defaultdict(list)
    average_sensor_temp = defaultdict(list)  # Add defaultdict for temperature
    average_sensor_humidity = defaultdict(list)  # Add defaultdict for humidity
    sensor_names = set()  # Track unique sensor names
    for sensor_file, quarter in sensor_files:  # Retrieve quarter information
        file_path = sensor_file
        if not os.path.isfile(file_path):
            print(f"File {sensor_file} not found, skipping...")
            continue

        try:
            # Read and process the sensor files
            df_sensor = pd.read_csv(file_path, parse_dates=['created_at'], low_memory=False)
            df_sensor.rename(columns={'PM25_avg': f'Sensor PM2.5 Q{quarter}',
                                      'Temp_C': f'Sensor Temp Q{quarter}',
                                      'Humidity_%': f'Sensor Humidity Q{quarter}'},
                             inplace=True)
            df_sensor = df_sensor[['created_at', f'Sensor PM2.5 Q{quarter}', f'Sensor Temp Q{quarter}', f'Sensor Humidity Q{quarter}']]
            df_sensor.set_index('created_at', inplace=True)
            df_sensor.index = df_sensor.index.tz_localize(None)
            df_sensor_daily = df_sensor.resample('D').mean()

            # Filter data based on user-selected date range
            df_sensor_daily = df_sensor_daily[(df_sensor_daily.index >= start_date) & (df_sensor_daily.index <= end_date)]

            # Plot the data
            sensor_name = os.path.basename(sensor_file).split('_')[0]
            if sensor_name in sensor_names:
                continue  # Skip if already plotted for another quarter
            sensor_names.add(sensor_name)

            # PM data
            if f'Sensor PM2.5 Q{quarter}' in df_sensor_daily.columns:
                fig_pm.add_trace(go.Scatter(x=df_sensor_daily.index, y=df_sensor_daily[f'Sensor PM2.5 Q{quarter}'], mode='lines', name=sensor_name + f' PM2.5 Q{quarter}'))
                average_sensor_pm25[sensor_name] = df_sensor_daily[f'Sensor PM2.5 Q{quarter}'].mean()
                sensor_pm25_data[sensor_name] = df_sensor_daily[f'Sensor PM2.5 Q{quarter}'].tolist()

            # Temperature data
            if f'Sensor Temp Q{quarter}' in df_sensor_daily.columns:
                fig_temp.add_trace(go.Scatter(x=df_sensor_daily.index, y=df_sensor_daily[f'Sensor Temp Q{quarter}'], mode='lines', name=sensor_name + f' Temp Q{quarter}'))
                average_sensor_temp[sensor_name] = df_sensor_daily[f'Sensor Temp Q{quarter}'].mean()

            # Humidity data
            if f'Sensor Humidity Q{quarter}' in df_sensor_daily.columns:
                fig_humidity.add_trace(go.Scatter(x=df_sensor_daily.index, y=df_sensor_daily[f'Sensor Humidity Q{quarter}'], mode='lines', name=sensor_name + f' Humidity Q{quarter}'))
                average_sensor_humidity[sensor_name] = df_sensor_daily[f'Sensor Humidity Q{quarter}'].mean()

        except ValueError as e:
            print(f"Error processing file {sensor_file}: {str(e)}")
            continue

    return average_sensor_pm25, sensor_pm25_data, average_sensor_temp, average_sensor_humidity

if st.button('Use Time Period Above for Outputs'):
    average_sensor_pm25, sensor_pm25_data, average_sensor_temp, average_sensor_humidity = process_and_plot_sensor_data()

    # Filter monitor data based on user-selected date range
    df_monitor = df_monitor[(df_monitor.index >= start_date) & (df_monitor.index <= end_date)]

    # Calculate exceedance and estimation statements
    pm25_daily_threshold = 35  # EPA daily PM2.5 threshold in µg/m³
    total_sensors = len(sensor_files)
    exceedance_days = sum(1 for values in sensor_pm25_data.values() if any(value > pm25_daily_threshold for value in values))
    exceedance_percentage = (exceedance_days / total_sensors) * 100

    estimation_table = []
    for sensor, avg_pm25 in average_sensor_pm25.items():
        if not pd.isnull(avg_pm25):
            estimation = avg_pm25 - df_monitor['Monitor PM2.5'].mean()
            estimation_percent = ((avg_pm25 - df_monitor['Monitor PM2.5'].mean()) / df_monitor['Monitor PM2.5'].mean()) * 100
            estimation_table.append({"Sensor": sensor, "Estimation": f"{'+-'[estimation < 0]}{abs(estimation):.2f} µg/m³", "Estimation %": f"{'+-'[estimation < 0]}{abs(estimation_percent):.2f}%"})

    # Monitor data
    fig_pm.add_trace(go.Scatter(x=df_monitor.index, y=df_monitor['Monitor PM2.5'], mode='lines', name='Monitor PM2.5', line=dict(color='black', width=2, dash='dash')))
    fig_temp.add_trace(go.Scatter(x=df_monitor.index, y=df_monitor['Monitor Temp'], mode='lines', name='Monitor Temp', line=dict(color='black', width=2, dash='dash')))
    fig_humidity.add_trace(go.Scatter(x=df_monitor.index, y=df_monitor['Monitor Humidity'], mode='lines', name='Monitor Humidity', line=dict(color='black', width=2, dash='dash')))

    # EPA daily exceedance limit line
    fig_pm.add_shape(type='line',
                     x0=df_monitor.index[0], x1=df_monitor.index[-1],
                     y0=pm25_daily_threshold, y1=pm25_daily_threshold,
                     line=dict(color='Red', dash='dot'))

    # Create an interactive web interface
    st.title('Santa Monica Particulate Matter Pollution Data vs. Regulatory Monitor 2018-2019')


    # PM Plot
    st.subheader('Particulate Matter over time')
    fig_pm.update_layout(yaxis_title='PM (ug/m3)')
    st.plotly_chart(fig_pm)
    st.markdown(f"The average PM2.5 level exceeded the EPA daily limit of {pm25_daily_threshold} µg/m³ on {exceedance_days} days ({exceedance_percentage:.2f}% of the days in the selected range).")

    # Overestimation/Underestimation table
    st.subheader('Sensor Estimation Compared to Regulatory Monitor')
    st.dataframe(pd.DataFrame(estimation_table))
    


    # Temperature Plot
    st.subheader('Temperature over time')
    fig_temp.update_layout(yaxis_title='Temperature (°C)')
    st.plotly_chart(fig_temp)

    # Humidity Plot
    st.subheader('Relative Humidity over time')
    fig_humidity.update_layout(yaxis_title='Relative Humidity (%)')
    st.plotly_chart(fig_humidity)