# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 14:49:02 2023

@author: coeno
"""

import numpy as np
import tensorflow as tf
from ANN_Data_Preparation_Function import ANN_data_preparation
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(1337)

def config_1_train_predict(years, Train_years, Val_years, Test_years, input_datapoints, prediction_horizon, datapoints_predicted, epochs):
    # One prediction model to predict wave height, wind speed and wave period     
    
    # Data preparation and split into Training, Validation and Testing
    data_prep_results = ANN_data_preparation(years, Train_years, Val_years, Test_years, input_datapoints, prediction_horizon, datapoints_predicted)
    
    Train_X = data_prep_results[0]
    Train_Y = data_prep_results[1]
    Val_X = data_prep_results[2]
    Val_Y = data_prep_results[3]
    Test_X = data_prep_results[4]
    Test_Y = data_prep_results[5]
    
    # Setup model layers
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(3*input_datapoints, 1)),
      tf.keras.layers.Dense(32, activation='tanh'),
      tf.keras.layers.Dropout(0.5),
      tf.keras.layers.Dense(32, activation='tanh'),
      tf.keras.layers.Dropout(0.5),
      tf.keras.layers.Dense(32, activation='tanh'),
      tf.keras.layers.Dropout(0.5),
      tf.keras.layers.Dense(32, activation='tanh'),
      tf.keras.layers.Dense(3*datapoints_predicted, activation='tanh'),
    ])
    
    model.compile(loss='mean_squared_error')
    
    # Fit the model
    history = model.fit(Train_X, Train_Y, validation_data=(Val_X, Val_Y), epochs=epochs, batch_size=32)

    # Summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('mean squared error')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='upper right')
    plt.show()
    
    # Predict weather data using testing data input
    prediction = model.predict(Test_X)
    
    # Transform output back to original value range
    output_scaler = data_prep_results[6]
    prediction = output_scaler.inverse_transform(prediction)
    original_data = output_scaler.inverse_transform(Test_Y)
    
    test_prediction = np.split(prediction, len(Test_years))

    return test_prediction, original_data

def config_2_train_predict(years, Train_years, Val_years, Test_years, input_datapoints, prediction_horizon, datapoints_predicted, epochs):
    # Individual prediction models for wave height, wind speed and wave period

    # Data preparation and split into Training, Validation and Testing
    data_prep_results = ANN_data_preparation(years, Train_years, Val_years, Test_years, input_datapoints, prediction_horizon, datapoints_predicted)
    
    Train_X = data_prep_results[0]
    Train_Y = data_prep_results[1]
    Val_X = data_prep_results[2]
    Val_Y = data_prep_results[3]
    Test_X = data_prep_results[4]
    Test_Y = data_prep_results[5]
    
    s_wht_Train_X = Train_X.iloc[:, 0:input_datapoints]
    s_wht_Train_Y = Train_Y.iloc[:, 0:datapoints_predicted]
    s_wht_Val_X = Val_X.iloc[:, 0:input_datapoints]
    s_wht_Val_Y = Val_Y.iloc[:, 0:datapoints_predicted]
    s_wht_Test_X = Test_X.iloc[:, 0:input_datapoints]
    
    wind_speed_Train_X = Train_X.iloc[:, input_datapoints:2*input_datapoints]
    wind_speed_Train_Y = Train_Y.iloc[:, datapoints_predicted:2*datapoints_predicted]
    wind_speed_Val_X = Val_X.iloc[:, input_datapoints:2*input_datapoints]
    wind_speed_Val_Y = Val_Y.iloc[:, datapoints_predicted:2*datapoints_predicted]
    wind_speed_Test_X = Test_X.iloc[:, input_datapoints:2*input_datapoints]
    
    mean_period_Train_X = Train_X.iloc[:, 2*input_datapoints:3*input_datapoints]
    mean_period_Train_Y = Train_Y.iloc[:, 2*datapoints_predicted:3*datapoints_predicted]
    mean_period_Val_X = Val_X.iloc[:, 2*input_datapoints:3*input_datapoints]
    mean_period_Val_Y = Val_Y.iloc[:, 2*datapoints_predicted:3*datapoints_predicted]
    mean_period_Test_X = Test_X.iloc[:, 2*input_datapoints:3*input_datapoints]
    
    # Setup wave height prediction model
    s_wht_model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(input_datapoints, 1)),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(datapoints_predicted, activation='sigmoid'),
    ])
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    s_wht_model.compile(loss='mean_squared_error', optimizer=opt)
    
    # Fit the model
    s_wht_history = s_wht_model.fit(s_wht_Train_X, s_wht_Train_Y, validation_data=(s_wht_Val_X, s_wht_Val_Y), epochs=epochs, batch_size=64)

    # Summarize history for loss
    plt.plot(s_wht_history.history['loss'])
    plt.plot(s_wht_history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('mean squared error')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='upper right')
    plt.show()
    
    # Predict wave height weather data using testing data input
    s_wht_prediction = s_wht_model.predict(s_wht_Test_X)
    
    # Setup wind speed prediction model
    wind_speed_model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(input_datapoints, 1)),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(datapoints_predicted, activation='sigmoid'),
    ])
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    wind_speed_model.compile(loss='mean_squared_error', optimizer=opt)
    
    # Fit the model
    wind_speed_history = wind_speed_model.fit(wind_speed_Train_X, wind_speed_Train_Y, validation_data=(wind_speed_Val_X, wind_speed_Val_Y), epochs=epochs, batch_size=128)

    # Summarize history for loss
    plt.plot(wind_speed_history.history['loss'])
    plt.plot(wind_speed_history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('mean squared error')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='upper right')
    plt.show()
    
    # Predict wind speed weather data using testing data input
    wind_speed_prediction = wind_speed_model.predict(wind_speed_Test_X)
    
    # Setup wave period prediction model
    mean_period_model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(input_datapoints, 1)),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(datapoints_predicted, activation='sigmoid'),
    ])
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    mean_period_model.compile(loss='mean_squared_error', optimizer=opt)
    
    # Fit the model
    mean_period_history = mean_period_model.fit(mean_period_Train_X, mean_period_Train_Y, validation_data=(mean_period_Val_X, mean_period_Val_Y), epochs=epochs, batch_size=128)

    # Summarize history for loss
    plt.plot(mean_period_history.history['loss'])
    plt.plot(mean_period_history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('mean squared error')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='upper right')
    plt.show()
    
    # Predict wave period weather data using testing data input
    mean_period_prediction = mean_period_model.predict(mean_period_Test_X)
    
    # Stitch wave height, wind speed and wave period together
    prediction = np.concatenate((s_wht_prediction, wind_speed_prediction, mean_period_prediction), 1)
    
    # Transform output back to original value range
    output_scaler = data_prep_results[6]
    prediction = output_scaler.inverse_transform(prediction)
    original_data = output_scaler.inverse_transform(Test_Y)
    
    test_prediction = np.split(prediction, len(Test_years))

    return test_prediction, original_data

def write_to_file_ANN(Test_years, prediction_horizon, datapoints_predicted, test_prediction):
    
    # Transform output into correct shape and write to file
    for year in Test_years:   
        startdate = str(year)+'-05-01 00:00:00'                   #starting on the first day of summer on the selected year
        ts= pd.to_datetime(startdate)                             #creates a timestamp for the start of the year      
        start= (ts.dayofyear-1)*24+ts.hour                        #gives us the hour of the year.
        
        i = Test_years.index(year)
            
        length_predicted = int(test_prediction[i].shape[0]*(test_prediction[i].shape[1]/3))
        prediction_df = pd.DataFrame(columns =['datetime','s_wht','wind_speed','mean_period'], index=range(length_predicted))
        
        split_year = test_prediction[i]
            
        s_wht_columns = split_year[:,:datapoints_predicted]
        s_wht_stacked = np.vstack(s_wht_columns).ravel('C')
            
        wind_speed_columns = split_year[:,datapoints_predicted:2*datapoints_predicted]
        wind_speed_stacked = np.vstack(wind_speed_columns).ravel('C')
            
        mean_period_columns = split_year[:,2*datapoints_predicted:3*datapoints_predicted]
        mean_period_stacked = np.vstack(mean_period_columns).ravel('C')
            
        prediction_df['s_wht'] = s_wht_stacked
        prediction_df['wind_speed'] = wind_speed_stacked
        prediction_df['mean_period'] = mean_period_stacked
        
        empty_rows = pd.DataFrame(columns =['datetime','s_wht','wind_speed','mean_period'], index=range(start+prediction_horizon))
        df1 = pd.concat([empty_rows, prediction_df], ignore_index=True)
        
        df1['datetime'] = pd.DataFrame({'datetime': pd.date_range(start=str(year)+'-01-01', end=str(year)+'-12-31 23:00:00', freq='H')})
        
        writePath = "E:/OneDrive/Documenten/TUDelft/Master Jaar 2/ME54015 Research Assignment/Assignment Hugo Boer/Site15Wave/Wave/"+str(year)+"pred.txt"
        
        with open(writePath, 'w') as f:
            df_as_csv = df1.to_csv(header=True, index=False, sep='\t', na_rep='NaN')
            f.write(df_as_csv)     
            
    return
    
def predict_weather_ANN(configuration, years, Train_years, Val_years, Test_years, input_datapoints, prediction_horizon, datapoints_predicted, epochs): 
    
    # Select model configuration
    if configuration == 1:
        test_prediction, original_data = config_1_train_predict(years, Train_years, Val_years, Test_years, input_datapoints, prediction_horizon, datapoints_predicted, epochs)
        
    if configuration == 2:
        test_prediction, original_data = config_2_train_predict(years, Train_years, Val_years, Test_years, input_datapoints, prediction_horizon, datapoints_predicted, epochs)
        
    write_to_file_ANN(Test_years, prediction_horizon, datapoints_predicted, test_prediction)
    
    return
    
        