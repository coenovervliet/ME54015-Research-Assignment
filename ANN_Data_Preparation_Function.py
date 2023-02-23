# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 09:51:29 2022

@author: Coen Overvliet
"""
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler

def ANN_data_split(weather_data, input_datapoints, prediction_horizon, datapoints_predicted, start):#, input_par=None, output_par=None):
    # Select and transform correct data based on number of input datapoints, prediction horizon and number of datapoints predicted
    end = len(weather_data)-1
    nr_datapoints = math.floor((end - start - prediction_horizon)/datapoints_predicted)
    
    X_order = []
    Y_order = []
    
    for i in [0,1,2]:
        index = i
        while index < (weather_data.shape[1]-1)*input_datapoints:
            X_order.extend([index])
            index+=3
            
    for i in [0,1,2]:
        index = i
        while index < (weather_data.shape[1]-1)*datapoints_predicted:
            Y_order.extend([index])
            index+=3
    
    X = []
    Y = []
    
    count = start-input_datapoints+1         

    for i in range(nr_datapoints):
        X_temp = []
        count_a = count
        
        for j in range(input_datapoints):
            inputs = weather_data.loc[count_a].values.flatten().tolist()
            inputs.pop(0)
            X_temp.extend(inputs)
            count_a+=1
            
        X_temp = [X_temp[m] for m in X_order]
        X.append(X_temp)

        Y_temp = []
        count_b = count+input_datapoints-1+prediction_horizon
        
        for k in range(datapoints_predicted):
            outputs = weather_data.loc[count_b].values.flatten().tolist()
            outputs.pop(0)
            Y_temp.extend(outputs)
            count_b+=1
        
        Y_temp = [Y_temp[n] for n in Y_order]
        Y.append(Y_temp)
        
        count += datapoints_predicted            

    return X, Y

def ANN_data_preparation(years, Train_years, Val_years, Test_years, input_datapoints, prediction_horizon, datapoints_predicted):
    print('preparing data for training, validation and testing')
    
    # Initialize empty arrays
    Train_X_temp = []
    Train_Y_temp = []
    Val_X_temp = []
    Val_Y_temp = []
    Test_X_temp = []
    Test_Y_temp = []
    
    # Add weather data of each year to correct array                                        
    for year in years:
        df1 = pd.read_table("E:/OneDrive/Documenten/TUDelft/Master Jaar 2/ME54015 Research Assignment/Assignment Hugo Boer/Site15Wave/Wave/"+str(year)+".txt") 
        df1['peak_period'] = 1/df1.peak_fr
        df1 = df1[['datetime','s_wht','wind_speed','peak_period']]
        
        startdate = str(year)+'-05-01 00:00:00'
        ts = pd.to_datetime(startdate)
        start = (ts.dayofyear-1)*24+ts.hour
        
        data_split_results = ANN_data_split(df1, input_datapoints, prediction_horizon, datapoints_predicted, start)
        
        if year in Train_years:
            Train_X_temp.extend(data_split_results[0])
            Train_Y_temp.extend(data_split_results[1])
            
        elif year in Val_years:
            Val_X_temp.extend(data_split_results[0])
            Val_Y_temp.extend(data_split_results[1])
            
        elif year in Test_years:
            Test_X_temp.extend(data_split_results[0])
            Test_Y_temp.extend(data_split_results[1])
            
        else: 
            print('Year not included.')
            print(year)
    
    # Scale all datasets to be between the range of 0 and 1
    scaler = MinMaxScaler()
    
    Train_X = pd.DataFrame(scaler.fit_transform(Train_X_temp))
    Train_Y = pd.DataFrame(scaler.fit_transform(Train_Y_temp))
    Val_X = pd.DataFrame(scaler.fit_transform(Val_X_temp))
    Val_Y = pd.DataFrame(scaler.fit_transform(Val_Y_temp))
    Test_X = pd.DataFrame(scaler.fit_transform(Test_X_temp))
    output_scaler = scaler.fit(Test_Y_temp)
    Test_Y = pd.DataFrame(scaler.transform(Test_Y_temp))
    
    print('finished data preparation')  
    return Train_X, Train_Y, Val_X, Val_Y, Test_X, Test_Y, output_scaler