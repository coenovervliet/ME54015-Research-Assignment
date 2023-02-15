# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 09:23:33 2023

@author: coeno
"""
import pandas as pd
import numpy as np
import math

def Determine_RMSE(Test_years):
    observed_weather = pd.DataFrame()
    predicted_weather = pd.DataFrame()
    
    for year in Test_years:
        df2 = pd.read_table("E:/OneDrive/Documenten/TUDelft/Master Jaar 2/ME54015 Research Assignment/Assignment Hugo Boer/Site15Wave/Wave/"+str(year)+".txt") 
        df2['mean_period'] = 1/df2.peak_fr # hier kan ik gelijk de txt bestanden mee inlezen geen omvormen naar excel mer nodig 
        df2 = df2[['datetime','s_wht','wind_speed','mean_period']]
        
        observed_weather = pd.concat([observed_weather, df2])
        
        df1 = pd.read_table("E:/OneDrive/Documenten/TUDelft/Master Jaar 2/ME54015 Research Assignment/Assignment Hugo Boer/Site15Wave/Wave/"+str(year)+"pred.txt") 
        empty_rows = pd.DataFrame(columns =['datetime','s_wht','wind_speed','mean_period'], index=range(len(df1), len(df1)+(len(df2)-len(df1))))
    
        predicted_weather = pd.concat([predicted_weather, df1, empty_rows])
    
    parameters = ['s_wht','wind_speed','mean_period']
    
    RMSE_values = np.zeros((3,1))
    
    for par in parameters:
        actual = observed_weather[par]
        predicted = predicted_weather[par]
   
        MSE = np.square(np.subtract(actual,predicted)).mean() 
        RMSE = math.sqrt(MSE)
        
        RMSE_values[parameters.index(par)] = RMSE
    
    return RMSE_values