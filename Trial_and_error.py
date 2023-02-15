# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 14:14:21 2023

@author: coeno
"""

from ANN_Forecasting_Model import predict_weather_ANN
from LSTM_Forecasting_Model import predict_weather_LSTM
from RMSE_Function import Determine_RMSE

# Specify parameters for forecasting model

years = [2001,2002,2003,2004,2005,2006,2007,2008,2009,2010]

Train_years = [2001,2002,2003,2004, 2005,2006]
Val_years = [2007]
Test_years = [2008,2009,2010]

input_datapoints = 8
prediction_horizon = 1
datapoints_predicted = 1

epochs = 50
nodes_per_layer = 64     #Or LSTM per layer. Number of layers needs to be adjusted manually
batch_size = 64
learning_rate = 0.001

method = 1              #Select model method (1: ANN - 2: LSTM)
configuration = 2       #Select input/output configuration (1: 1 model with 3 inputs and 3 outputs - 2: 3 models with 1 input and 1 output)

# Predict weather using selected method and configuration

if method == 1:
    predict_weather_ANN(configuration, years, Train_years, Val_years, Test_years, input_datapoints, prediction_horizon, datapoints_predicted, epochs, nodes_per_layer, batch_size, learning_rate)
if method == 2:    
    predict_weather_LSTM(configuration, years, Train_years, Val_years, Test_years, input_datapoints, prediction_horizon, datapoints_predicted, epochs, nodes_per_layer, batch_size, learning_rate)

# Predicted weather data is written to file as part of the predict_weather functions (location: \Wave)

# Calculate and print RMSE values

RMSE_values = Determine_RMSE(Test_years)

print("RMSE wave height:", RMSE_values[0].item())
print("RMSE wind speed:", RMSE_values[1].item())
print("RMSE wave period:", RMSE_values[2].item())