# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 09:23:33 2023

@author: Coen Overvliet
"""
import pandas as pd
import numpy as np
import math

def Determine_RMSE(Test_years):
    observed_weather = pd.DataFrame()
    predicted_weather = pd.DataFrame()
    
    for year in Test_years:
        df2 = pd.read_table("E:/OneDrive/Documenten/TUDelft/Master Jaar 2/ME54015 Research Assignment/Assignment Hugo Boer/Site15Wave/Wave/"+str(year)+".txt") 
        df2['peak_period'] = 1/df2.peak_fr # hier kan ik gelijk de txt bestanden mee inlezen geen omvormen naar excel mer nodig 
        df2 = df2[['datetime','s_wht','wind_speed','peak_period']]
        
        observed_weather = pd.concat([observed_weather, df2])
        
        df1 = pd.read_table("E:/OneDrive/Documenten/TUDelft/Master Jaar 2/ME54015 Research Assignment/Assignment Hugo Boer/Site15Wave/Wave/"+str(year)+"pred.txt") 
        empty_rows = pd.DataFrame(columns =['datetime','s_wht','wind_speed','peak_period'], index=range(len(df1), len(df1)+(len(df2)-len(df1))))
    
        predicted_weather = pd.concat([predicted_weather, df1, empty_rows])
    
    parameters = ['s_wht','wind_speed','peak_period']
    
    RMSE_values = np.zeros((3,1))
    
    for par in parameters:
        actual = observed_weather[par]
        predicted = predicted_weather[par]
   
        MSE = np.square(np.subtract(actual,predicted)).mean() 
        RMSE = math.sqrt(MSE)
        
        RMSE_values[parameters.index(par)] = RMSE
        
    RMSE_values = pd.DataFrame(RMSE_values, columns = ['RMSE'], index = ['Significant wave height', 'Wind speed', 'Peak wave period'])
    
    return RMSE_values

def Count_error_direction(Test_years):
    observed_weather = pd.DataFrame()
    predicted_weather = pd.DataFrame()
    
    for year in Test_years:
        df2 = pd.read_table("E:/OneDrive/Documenten/TUDelft/Master Jaar 2/ME54015 Research Assignment/Assignment Hugo Boer/Site15Wave/Wave/"+str(year)+".txt") 
        df2['peak_period'] = 1/df2.peak_fr # hier kan ik gelijk de txt bestanden mee inlezen geen omvormen naar excel mer nodig 
        df2 = df2[['datetime','s_wht','wind_speed','peak_period']]
        
        observed_weather = pd.concat([observed_weather, df2])
        
        df1 = pd.read_table("E:/OneDrive/Documenten/TUDelft/Master Jaar 2/ME54015 Research Assignment/Assignment Hugo Boer/Site15Wave/Wave/"+str(year)+"pred.txt") 
        empty_rows = pd.DataFrame(columns =['datetime','s_wht','wind_speed','peak_period'], index=range(len(df1), len(df1)+(len(df2)-len(df1))))
    
        predicted_weather = pd.concat([predicted_weather, df1, empty_rows])
       
    parameters = ['s_wht','wind_speed','peak_period']
    
    difference = np.zeros((len(observed_weather),3))
    error_direction = np.zeros((2,3))
    
    for par in parameters:
        actual = observed_weather[par]
        predicted = predicted_weather[par]
   
        diff = predicted - actual
        
        pos_count, neg_count = 0, 0
        
        for num in diff:
     
            # checking condition
            if num >= 0:
                pos_count += 1
     
            if num < 0:
                neg_count += 1
        
        difference[:,parameters.index(par)] = diff
        error_direction[0,parameters.index(par)] = pos_count
        error_direction[1,parameters.index(par)] = neg_count
        
    error_direction = pd.DataFrame(error_direction, columns = ['Significant wave height', 'Wind speed', 'Peak wave period'], index = ['Positive count', 'Negative count'])
           
    return error_direction