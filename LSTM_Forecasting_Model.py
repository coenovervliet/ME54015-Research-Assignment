# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 14:49:02 2023

@author: Coen Overvliet
"""

import numpy as np
import tensorflow as tf
from LSTM_Data_Preparation_Function import LSTM_data_preparation
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(1337)

def config_1_train_predict(years, Train_years, Val_years, Test_years, input_datapoints, prediction_horizon, datapoints_predicted, epochs, LSTM_per_layer, batch_size, learning_rate):
    data_prep_results = LSTM_data_preparation(years, Train_years, Val_years, Test_years, input_datapoints, prediction_horizon, datapoints_predicted)
    
    Train_X = data_prep_results[0]
    Train_Y = data_prep_results[1]
    Val_X = data_prep_results[2]
    Val_Y = data_prep_results[3]
    Test_X = data_prep_results[4]
    Test_Y = data_prep_results[5]
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(LSTM_per_layer, activation='relu', input_shape=(Train_X.shape[1], Train_X.shape[2]), return_sequences=True),
        tf.keras.layers.LSTM(LSTM_per_layer, activation='relu', return_sequences=True),
        #tf.keras.layers.LSTM(LSTM_per_layer, activation='relu', return_sequences=True),
        tf.keras.layers.LSTM(LSTM_per_layer, activation='relu', return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(Train_Y.shape[1]),
    ])
    
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='mean_squared_error')
    
    history = model.fit(Train_X, Train_Y, validation_data=(Val_X, Val_Y), epochs=epochs, batch_size=batch_size)
    
    # Summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('mean squared error')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='upper right')
    plt.show()
    
    prediction = model.predict(Test_X)
    
    output_scaler = data_prep_results[6]
    prediction = output_scaler.inverse_transform(prediction)
    original_data = output_scaler.inverse_transform(Test_Y)
    
    test_prediction = np.split(prediction, len(Test_years))

    return test_prediction, original_data

def config_2_train_predict(years, Train_years, Val_years, Test_years, input_datapoints, prediction_horizon, datapoints_predicted, epochs, LSTM_per_layer, batch_size, learning_rate):
    # Individual prediction models for wave height, wind speed and wave period
    
    # Data preparation and split into Training, Validation and Testing
    data_prep_results = LSTM_data_preparation(years, Train_years, Val_years, Test_years, input_datapoints, prediction_horizon, datapoints_predicted)
    
    Train_X = data_prep_results[0]
    Train_Y = data_prep_results[1]
    Val_X = data_prep_results[2]
    Val_Y = data_prep_results[3]
    Test_X = data_prep_results[4]
    Test_Y = data_prep_results[5]
    
    s_wht_columns = [0]
    wind_speed_columns = [1]
    peak_period_columns = [2]
    i = 1
    
    while i < datapoints_predicted:
        s_wht_columns.append(s_wht_columns[0]+3*i)
        wind_speed_columns.append(wind_speed_columns[0]+3*i)
        peak_period_columns.append(peak_period_columns[0]+3*i)
        i+=1
    
    s_wht_Train_X = Train_X[:, :, 0].copy().reshape((Train_X.shape[0], Train_X.shape[1], 1))
    s_wht_Train_Y = Train_Y[:, s_wht_columns].copy()
    s_wht_Val_X = Val_X[:, :, 0].copy().reshape((Val_X.shape[0], Val_X.shape[1], 1))
    s_wht_Val_Y = Val_Y[:, s_wht_columns].copy()
    s_wht_Test_X = Test_X[:, :, 0].copy().reshape((Test_X.shape[0], Test_X.shape[1], 1))
    
    wind_speed_Train_X = Train_X[:, :, 1].copy().reshape((Train_X.shape[0], Train_X.shape[1], 1))
    wind_speed_Train_Y = Train_Y[:, wind_speed_columns].copy()
    wind_speed_Val_X = Val_X[:, :, 1].copy().reshape((Val_X.shape[0], Val_X.shape[1], 1))
    wind_speed_Val_Y = Val_Y[:, wind_speed_columns].copy()
    wind_speed_Test_X = Test_X[:, :, 1].copy().reshape((Test_X.shape[0], Test_X.shape[1], 1))
    
    peak_period_Train_X = Train_X[:, :, 2].copy().reshape((Train_X.shape[0], Train_X.shape[1], 1))
    peak_period_Train_Y = Train_Y[:, peak_period_columns].copy()
    peak_period_Val_X = Val_X[:, :, 2].copy().reshape((Val_X.shape[0], Val_X.shape[1], 1))
    peak_period_Val_Y = Val_Y[:, peak_period_columns].copy()
    peak_period_Test_X = Test_X[:, :, 2].copy().reshape((Test_X.shape[0], Test_X.shape[1], 1))
    
    # Setup wave height prediction model
    s_wht_model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(LSTM_per_layer, activation='relu', input_shape=(s_wht_Train_X.shape[1], s_wht_Train_X.shape[2]), return_sequences=True),
        tf.keras.layers.LSTM(LSTM_per_layer, activation='relu', return_sequences=True),
        tf.keras.layers.LSTM(LSTM_per_layer, activation='relu', return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(s_wht_Train_Y.shape[1]),
    ])
    
    s_wht_opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    s_wht_model.compile(optimizer=s_wht_opt, loss='mean_squared_error')
    
    s_wht_history = s_wht_model.fit(s_wht_Train_X, s_wht_Train_Y, validation_data=(s_wht_Val_X, s_wht_Val_Y), epochs=epochs, batch_size=batch_size)
    
    # Summarize history for loss
    plt.plot(s_wht_history.history['loss'])
    plt.plot(s_wht_history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('mean squared error')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='upper right')
    plt.show()
    
    s_wht_prediction = s_wht_model.predict(s_wht_Test_X)
    
    # Setup wind speed prediction model
    wind_speed_model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(LSTM_per_layer, activation='relu', input_shape=(wind_speed_Train_X.shape[1], wind_speed_Train_X.shape[2]), return_sequences=True),
        tf.keras.layers.LSTM(LSTM_per_layer, activation='relu', return_sequences=True),
        tf.keras.layers.LSTM(LSTM_per_layer, activation='relu', return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(wind_speed_Train_Y.shape[1]),
    ])
    
    wind_speed_opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    wind_speed_model.compile(optimizer=wind_speed_opt, loss='mean_squared_error')
    
    wind_speed_history = wind_speed_model.fit(wind_speed_Train_X, wind_speed_Train_Y, validation_data=(wind_speed_Val_X, wind_speed_Val_Y), epochs=epochs, batch_size=batch_size)
    
    # Summarize history for loss
    plt.plot(wind_speed_history.history['loss'])
    plt.plot(wind_speed_history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('mean squared error')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='upper right')
    plt.show()
    
    wind_speed_prediction = wind_speed_model.predict(wind_speed_Test_X)
    
    # Setup wave period prediction model
    peak_period_model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(LSTM_per_layer, activation='relu', input_shape=(peak_period_Train_X.shape[1], peak_period_Train_X.shape[2]), return_sequences=True),
        tf.keras.layers.LSTM(LSTM_per_layer, activation='relu', return_sequences=True),
        tf.keras.layers.LSTM(LSTM_per_layer, activation='relu', return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(peak_period_Train_Y.shape[1]),
    ])
    
    peak_period_opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    peak_period_model.compile(optimizer=peak_period_opt, loss='mean_squared_error')
    
    peak_period_history = peak_period_model.fit(peak_period_Train_X, peak_period_Train_Y, validation_data=(peak_period_Val_X, peak_period_Val_Y), epochs=epochs, batch_size=batch_size)
    
    # Summarize history for loss
    plt.plot(peak_period_history.history['loss'])
    plt.plot(peak_period_history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('mean squared error')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='upper right')
    plt.show()
    
    peak_period_prediction = peak_period_model.predict(peak_period_Test_X)
    
    # Stitch wave height, wind speed and wave period together  
    prediction = np.concatenate((s_wht_prediction[:,[0]], wind_speed_prediction[:,[0]], peak_period_prediction[:,[0]]), 1)
    
    for i in range(1,datapoints_predicted):
        prediction = np.concatenate((prediction, s_wht_prediction[:,[i]], wind_speed_prediction[:,[i]], peak_period_prediction[:,[i]]), 1)
    
    # Use inverse scaler
    output_scaler = data_prep_results[6]
    prediction = output_scaler.inverse_transform(prediction)
    original_data = output_scaler.inverse_transform(Test_Y)
    
    test_prediction = np.split(prediction, len(Test_years))

    return test_prediction, original_data

def write_to_file_LSTM(Test_years, prediction_horizon, datapoints_predicted, test_prediction):
    
    # Transform output into correct shape and write to file
    for year in Test_years:   
        startdate = str(year)+'-05-01 00:00:00'                   #starting on the first day of summer on the selected year
        ts= pd.to_datetime(startdate)                             #creates a timestamp for the start of the year      
        start= (ts.dayofyear-1)*24+ts.hour                        #gives us the hour of the year.
        
        i = Test_years.index(year)
            
        length_predicted = int(test_prediction[i].shape[0]*(test_prediction[i].shape[1]/3))
        prediction_df = pd.DataFrame(columns =['datetime','s_wht','wind_speed','peak_period'], index=range(length_predicted))
        
        split_year = test_prediction[i]
        
        s_wht_column_nrs = [0]
        wind_speed_column_nrs = [1]
        peak_period_column_nrs = [2]
        i = 1
    
        while i < datapoints_predicted:
            s_wht_column_nrs.append(s_wht_column_nrs[0]+3*i)
            wind_speed_column_nrs.append(wind_speed_column_nrs[0]+3*i)
            peak_period_column_nrs.append(peak_period_column_nrs[0]+3*i)
            i+=1
            
        s_wht_columns = split_year[:,s_wht_column_nrs]
        s_wht_stacked = np.vstack(s_wht_columns).ravel('C')
            
        wind_speed_columns = split_year[:,wind_speed_column_nrs]
        wind_speed_stacked = np.vstack(wind_speed_columns).ravel('C')
            
        peak_period_columns = split_year[:,peak_period_column_nrs]
        peak_period_stacked = np.vstack(peak_period_columns).ravel('C')
            
        prediction_df['s_wht'] = s_wht_stacked
        prediction_df['wind_speed'] = wind_speed_stacked
        prediction_df['peak_period'] = peak_period_stacked
        
        empty_rows = pd.DataFrame(columns =['datetime','s_wht','wind_speed','peak_period'], index=range(start+prediction_horizon))
        df1 = pd.concat([empty_rows, prediction_df], ignore_index=True)
        
        df1['datetime'] = pd.DataFrame({'datetime': pd.date_range(start=str(year)+'-01-01', end=str(year)+'-12-31 23:00:00', freq='H')})
        
        writePath = "Wave/"+str(year)+"pred.txt"
        
        with open(writePath, 'w') as f:
            df_as_csv = df1.to_csv(header=True, index=False, sep='\t', na_rep='NaN')
            f.write(df_as_csv)     
            
    return

def predict_weather_LSTM(configuration, years, Train_years, Val_years, Test_years, input_datapoints, prediction_horizon, datapoints_predicted, epochs, LSTM_per_layer, batch_size, learning_rate): 
    # Select model configuration
    if configuration == 1:
        test_prediction, original_data = config_1_train_predict(years, Train_years, Val_years, Test_years, input_datapoints, prediction_horizon, datapoints_predicted, epochs, LSTM_per_layer, batch_size, learning_rate)
        #write_to_file(Test_years, prediction_horizon, datapoints_predicted, test_prediction)
        
    if configuration == 2:
        test_prediction, original_data = config_2_train_predict(years, Train_years, Val_years, Test_years, input_datapoints, prediction_horizon, datapoints_predicted, epochs, LSTM_per_layer, batch_size, learning_rate)
        
    write_to_file_LSTM(Test_years, prediction_horizon, datapoints_predicted, test_prediction)
    
    return