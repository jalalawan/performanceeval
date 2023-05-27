
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 18:22:17 2023

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

# ---------------------------------------------------------------------------
   # PERFORMANCE EVALUATION OF SM SENSORS (n=7)
   # ---------------------------------------------------------------------------



# Configure the project directory where data, figures, etc. will be stored
sensortoolkit.presets.set_project_path()

# Create an AirSensor instance for the sensor you'd like to evaluate
sensor = sensortoolkit.AirSensor(make='PurpleAir',
                                 model='PurpleAir-II_SM_Q0')

# Construct sensor-specific directories in the project path for data, figures, etc.
sensor.create_directories()

# Run the interative setup routine for specifying how to ingest sensor data
sensor.sensor_setup()

# Loading sensor data for the first time
sensor.load_data(load_raw_data=True,
                 write_to_file=True)

#Loading 'sensor' data object from processed data files
sensor.load_data(load_raw_data=False, write_to_file=False)

# Loading reference object for the first time
reference = sensortoolkit.ReferenceMonitor()
reference.reference_setup()



# Import reference data for parameter types measured by the air sensor, also
# import meteorological data if instruments collocated at monitoring site
reference.load_data(bdate=sensor.bdate,
                    edate=sensor.edate,
                    param_list=sensor.param_headers,
                    met_data=True)



# Fill in bracketed placeholder text with your information
# Add information about the testing organization that conducted the evaluation
sensortoolkit.presets.test_org = {
    'testing_descrip': 'n=4 sensors in SM for Quarter 1',
    'org_name': 'RAND SM',
    'org_division': 'PRGS',
    'org_type': 'N/A',
    'org_website': {'title': 'PRGS',
                    'link': 'www.pardeerand.edu'}



pollutant = sensortoolkit.Parameter('PM25')

all_metrics = pollutant.PerformanceTargets.get_all_metrics()

evaluation.plot_met_dist()

#Performance Evaluation

evaluation.add_deploy_dict_stats()

evaluation = sensortoolkit.SensorEvaluation(sensor,
                                            pollutant,
                                            reference,
                                            write_to_file=True)

report = sensortoolkit.PerformanceReport(sensor,
                                         pollutant,
                                         reference,
                                         write_to_file=True)

pollutant = sensortoolkit.Parameter('PM25')

all_metrics = pollutant.PerformanceTargets.get_all_metrics()

evaluation.plot_met_dist()

#Performance Evaluation

evaluation.add_deploy_dict_stats()

evaluation = sensortoolkit.SensorEvaluation(sensor,
                                            pollutant,
                                            reference,
                                            write_to_file=True)

report = sensortoolkit.PerformanceReport(sensor,
                                         pollutant,
                                         reference,
                                         write_to_file=True)

evaluation.plot_met_dist()

evaluation.calculate_metrics()

evaluation.plot_met_dist()

evaluation.plot_metrics()

evaluation.plot_sensor_met_scatter()

evaluation.plot_sensor_scatter()

evaluation.plot_timeseries(date_interval=10)
evaluation.plot_timeseries(date_interval=7)


# Required Table

evaluation.print_eval_metrics()
evaluation.print_eval_metrics(averaging_interval='24-hour')

evaluation.print_eval_conditions()
evaluation.print_eval_conditions(averaging_interval='24-hour')


evaluation.add_deploy_dict_stats()

evaluation.calculate_metrics()



# Create a performance evaluation report for the sensor
report = sensortoolkit.PerformanceReport(sensor,
                                         pollutant,
                                         reference,
                                         write_to_file=True)

evaluation.plot_met_influence()
evaluation.plot_metrics()

evaluation.stats_df
evaluation.avg_stats_df
evaluation.deploy_dict


# Instantiate the PerformanceReport class for the example sensor dataset
report = sensortoolkit.PerformanceReport(sensor,
                                         pollutant,
                                         reference,
                                         write_to_file=True,
                                         figure_search=True)

# Compile the report and save the file to the reports subfolder
report.CreateReport()



evaluation.plot_sensor_met_scatter(averaging_interval='24-hour', met_param='Temp')

evaluation.plot_sensor_met_scatter(averaging_interval='24-hour', met_param='Temp')
evaluation.plot_sensor_met_scatter(averaging_interval='24-hour', met_param='RH')
evaluation.plot_sensor_met_scatter(averaging_interval='1-hour', met_param='RH')
evaluation.plot_sensor_met_scatter(averaging_interval='1-hour', met_param='Temp')

evaluation.plot_sensor_scatter(averaging_interval='24-hour', plot_subset=None)
evaluation.plot_sensor_scatter(averaging_interval='1-hour', plot_subset=None)



# Generate report
report.CreateReport()



evaluation.plot_sensor_met_scatter()
evaluation.print_eval_metrics(averaging_interval='24-hour')

evaluation.plot_sensor_met_scatter(averaging_interval='24-hour', met_param='Temp')
evaluation.plot_sensor_met_scatter(averaging_interval='1-hour', met_param='Temp')

evaluation.plot_sensor_met_scatter(averaging_interval='24-hour', met_param='RH')
evaluation.plot_sensor_met_scatter(averaging_interval='1-hour', met_param='RH')

evaluation.plot_sensor_scatter(averaging_interval='24-hour', plot_subset=None)


















