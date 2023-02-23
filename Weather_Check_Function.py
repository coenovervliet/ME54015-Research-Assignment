# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 09:18:52 2022

@author: Coen Overvliet
"""
import math
import numpy as np

def apply_alpha_factor(s_wht_limit, s_wht_alpha_factors, wind_speed_limit, wind_speed_alpha_factors):
    # Apply Alpha Factors to significant wave height and wind speed operational limits
    if not s_wht_limit is None:
        limit_up = math.ceil(s_wht_limit)
        s_wht_alpha = s_wht_alpha_factors[limit_up]
        new_s_wht_limit = s_wht_alpha*s_wht_limit
        
    else:
        new_s_wht_limit = None
    
    if not wind_speed_limit is None:
        if wind_speed_limit <= wind_speed_alpha_factors.index[0]:
            wind_speed_alpha = wind_speed_alpha_factors.iloc[0]
        
        else:
            wind_speed_alpha = wind_speed_alpha_factors.iloc[1]
        
        new_wind_speed_limit = wind_speed_alpha*wind_speed_limit
        
    else:
        new_wind_speed_limit = None
    
    return new_s_wht_limit, new_wind_speed_limit

def weather_check(current_duration, start, operation_duration, predicted_weather, observed_weather, s_wht_alpha_factors, wind_speed_alpha_factors, s_wht_limit=None, wind_speed_limit=None, peak_period_limit=None):
    # Initialize metrics
    waiting=0
    
    s_wht_true_positives=0
    s_wht_false_positives=0
    s_wht_true_negatives=0
    s_wht_false_negatives=0
    
    wind_speed_true_positives=0
    wind_speed_false_positives=0
    wind_speed_true_negatives=0
    wind_speed_false_negatives=0
    
    peak_period_true_positives=0
    peak_period_false_positives=0
    peak_period_true_negatives=0
    peak_period_false_negatives=0
    
    overall_true_positives=0
    overall_false_positives=0
    overall_true_negatives=0
    overall_false_negatives=0
    
    # Determine length of operation
    weather_window_length = math.ceil(operation_duration)
    
    # Apply Alpha Factors to significant wave height and wind speed operational limits
    s_wht_limit_pred, wind_speed_limit_pred = apply_alpha_factor(s_wht_limit, s_wht_alpha_factors, wind_speed_limit, wind_speed_alpha_factors)
    
    # Find suitable weather window in predicted weather data and check using observed weather data
    for i in range(0,len(predicted_weather.datetime)-start-math.ceil(current_duration)):
        checks_needed=0
        prediction_checks=0
        observation_checks=0
        
        weather_window_begin = start+math.ceil(current_duration)
        weather_window_end = start+math.ceil(current_duration)+weather_window_length-1
        
        if not s_wht_limit is None:
            checks_needed+=1
                   
            if (predicted_weather.loc[weather_window_begin:weather_window_end,['s_wht']].max().item() <= s_wht_limit_pred):
                #print('s_wht_limit not exceeded in prediction')
                prediction_checks+=1
                
                if (observed_weather.loc[weather_window_begin:weather_window_end,['s_wht']].max().item() <= s_wht_limit):
                    #print('correct prediction')
                    #print('s_wht_true_positive')
                    s_wht_true_positives+=1 
                    observation_checks+=1
                else:
                    #print('incorrect prediction')
                    #print('s_wht_false_positive')
                    s_wht_false_positives+=1
                    
            else:
                #print('s_wht_limit exceeded in prediction')
                
                if (observed_weather.loc[weather_window_begin:weather_window_end,['s_wht']].max().item() <= s_wht_limit):
                    #print('incorrect prediction')
                    #print('s_wht_false_negative')
                    s_wht_false_negatives+=1
                    observation_checks+=1
                else:
                    #print('correct prediction')
                    #print('s_wht_true_negative')
                    s_wht_true_negatives+=1

        if not wind_speed_limit is None:
            checks_needed+=1
            
            if (predicted_weather.loc[weather_window_begin:weather_window_end,['wind_speed']].max().item() <= wind_speed_limit_pred):
                #print('wind_speed_limit not exceeded in prediction')
                prediction_checks+=1
                
                if (observed_weather.loc[weather_window_begin:weather_window_end,['wind_speed']].max().item() <= wind_speed_limit):
                    #print('correct prediction')
                    #print('wind_speed_true_positive')
                    wind_speed_true_positives+=1
                    observation_checks+=1
                else:
                    #print('incorrect prediction')
                    #print('wind_speed_false_positive')
                    wind_speed_false_positives+=1
                    
            else:
                #print('wind_speed_limit exceeded in prediction')
                
                if (observed_weather.loc[weather_window_begin:weather_window_end,['wind_speed']].max().item() <= wind_speed_limit):
                    #print('incorrect prediction')
                    #print('wind_speed_false_negative')
                    wind_speed_false_negatives+=1
                    observation_checks+=1
                else:
                    #print('correct prediction')
                    #print('wind_speed_true_negative')
                    wind_speed_true_negatives+=1

        if not wind_speed_limit is None:
            checks_needed+=1
            
            if (predicted_weather.loc[weather_window_begin:weather_window_end,['peak_period']].max().item() <= peak_period_limit):
                #print('peak_period_limit not exceeded in prediction')
                prediction_checks+=1
                
                if (observed_weather.loc[weather_window_begin:weather_window_end,['peak_period']].max().item() <= peak_period_limit):
                    #print('correct prediction')
                    #print('peak_period_true_positive')
                    peak_period_true_positives+=1
                    observation_checks+=1
                else:
                    #print('incorrect prediction')
                    #print('peak_period_false_positive')
                    peak_period_false_positives+=1
                    
            else:
                #print('peak_period_limit exceeded in prediction')
                
                if (observed_weather.loc[weather_window_begin:weather_window_end,['peak_period']].max().item() <= peak_period_limit):
                    #print('incorrect prediction')
                    #print('peak_period_false_negative')
                    peak_period_false_negatives+=1
                    observation_checks+=1
                else:
                    #print('correct prediction')
                    #print('peak_period_true_negative')
                    peak_period_true_positives+=1
                
        #print('checks needed:'+str(checks_needed))
        #print(predicted_weather.loc[weather_window_begin:weather_window_end])
        #print('prediction checks:'+str(prediction_checks))
        #print(observed_weather.loc[weather_window_begin:weather_window_end])
        #print('observation checks:'+str(observation_checks))
        
        if prediction_checks==checks_needed:
            #print('suitable weather window predicted')
            
            if observation_checks==checks_needed:
                #print('true positive overall')
                overall_true_positives+=1
                
                #print(predicted_weather.loc[weather_window_begin:weather_window_end])
                #print(observed_weather.loc[weather_window_begin:weather_window_end])
                
                break
                
            else:
                #print('false positive overall')
                overall_false_positives+=1
                current_duration+=1
                waiting+=1
                i+=1
                
        else:
            #print('no suitable weather window predicted')
            
            if observation_checks==checks_needed:
                #print('false negative overall')
                overall_false_negatives+=1
                
            else:
                #print('true negative overall')
                overall_true_negatives+=1
                
            current_duration+=1 #= current_duration+1
            waiting+=1
            i+=1
            
    current_duration = current_duration + operation_duration

    s_wht_metrics = np.array([s_wht_true_positives, s_wht_false_positives, s_wht_true_negatives, s_wht_false_negatives])
    wind_speed_metrics = np.array([wind_speed_true_positives, wind_speed_false_positives, wind_speed_true_negatives, wind_speed_false_negatives])
    peak_period_metrics = np.array([peak_period_true_positives, peak_period_false_positives, peak_period_true_negatives, peak_period_false_negatives])
    overall_metrics = np.array([overall_true_positives, overall_false_positives, overall_true_negatives, overall_false_negatives])
    
    #print('operation duration:'+str(operation_duration))
    #print('waiting on weather:'+str(waiting))
    #print('current duration:'+str(current_duration))
    
    return current_duration, waiting, overall_metrics, s_wht_metrics, wind_speed_metrics, peak_period_metrics