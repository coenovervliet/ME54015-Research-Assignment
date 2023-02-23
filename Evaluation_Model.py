# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 15:15:30 2021

@author: Coen Overvliet (adaptation of weather restricted model by Hugo Boer)
"""
import pandas as pd  
import math
#import matplotlib.pyplot as plt
from Weather_Check_Function import weather_check
import numpy as np
from ANN_Forecasting_Model import predict_weather_ANN
from LSTM_Forecasting_Model import predict_weather_LSTM
import matplotlib.pyplot as plt
from Error_Functions import Determine_RMSE, Count_error_direction

# %% Initialize

# Reading data for vessel and operation characteristics

loc = ("Vessel Data.xls")
loc2 = ("Farm Data.xls")
loc3 = ("Alpha Factors.xls")

Data_Vessel = pd.read_excel(loc, sheet_name = 'Vessel')	            #Vessel data sheet 
Data_Operation =pd.read_excel(loc,sheet_name ='Operation')          #Operation data sheet 
Data_Farm = pd.read_excel(loc2)                                     #Farm data sheet
Data_Config = pd.read_excel(loc, sheet_name = 'Configurations')
Data_Limits = pd.read_excel(loc, sheet_name = 'Limits')
s_wht_Alpha_Factors = pd.read_excel(loc3, sheet_name = 'Active s_wht Alpha Factors')
wind_speed_Alpha_Factors = pd.read_excel(loc3, sheet_name = 'Active wind_speed Alpha Factors')

# Specify parameters for forecasting model

years = [2001,2002,2003,2004,2005,2006,2007,2008,2009,2010]

Train_years = [2001,2002,2003,2004,2005,2006]
Val_years = [2007]
Test_years = [2008,2009,2010]

input_datapoints = 2
prediction_horizon = 1
datapoints_predicted = 3

epochs = 25
nodes_per_layer = 132       #Or LSTM per layer. Number of layers needs to be adjusted manually
batch_size = 32
learning_rate = 0.0001

method = 1                  #Select model method (1: ANN - 2: LSTM)
configuration = 2           #Select input/output configuration (1: 1 model with 3 inputs and 3 outputs - 2: 3 models with 1 input and 1 output)
save_results = True         #Write results to excel file

# Predict weather using selected method and configuration

if method == 1:
    predict_weather_ANN(configuration, years, Train_years, Val_years, Test_years, input_datapoints, prediction_horizon, datapoints_predicted, epochs, nodes_per_layer, batch_size, learning_rate)
if method == 2:    
    predict_weather_LSTM(configuration, years, Train_years, Val_years, Test_years, input_datapoints, prediction_horizon, datapoints_predicted, epochs, nodes_per_layer, batch_size, learning_rate)
# Predicted weather data is written to file as part of the predict_weather functions (location: \Wave)

# Create dataframes for results

test_years_indices = []
for year in Test_years:
    test_years_indices.append(str(year))
    
T_FDN = pd.DataFrame(columns =['Option 1', 'Option 2', 'Option 3', 'Option 4', 'Option 5', 'Option 6', 'Option 7', 'Option 8', 'Option 9'], index=test_years_indices)
WAIT_FDN = pd.DataFrame(columns =['Option 1', 'Option 2', 'Option 3', 'Option 4', 'Option 5', 'Option 6', 'Option 7', 'Option 8', 'Option 9'], index=test_years_indices)
COST_FDN = pd.DataFrame(columns =['Option 1', 'Option 2', 'Option 3', 'Option 4', 'Option 5', 'Option 6', 'Option 7', 'Option 8', 'Option 9'], index=test_years_indices)
T_WTB = pd.DataFrame(columns =['Option 1', 'Option 2', 'Option 3', 'Option 4', 'Option 5', 'Option 6', 'Option 7', 'Option 8', 'Option 9'], index=test_years_indices)
WAIT_WTB = pd.DataFrame(columns =['Option 1', 'Option 2', 'Option 3', 'Option 4', 'Option 5', 'Option 6', 'Option 7', 'Option 8', 'Option 9'], index=test_years_indices)
COST_WTB = pd.DataFrame(columns =['Option 1', 'Option 2', 'Option 3', 'Option 4', 'Option 5', 'Option 6', 'Option 7', 'Option 8', 'Option 9'], index=test_years_indices)
T_TOT = pd.DataFrame(columns =['Option 1', 'Option 2', 'Option 3', 'Option 4', 'Option 5', 'Option 6', 'Option 7', 'Option 8', 'Option 9'], index=test_years_indices)
COST_TOT = pd.DataFrame(columns =['Option 1', 'Option 2', 'Option 3', 'Option 4', 'Option 5', 'Option 6', 'Option 7', 'Option 8', 'Option 9'], index=test_years_indices)
PEN_COST = pd.DataFrame(columns =['Option 1', 'Option 2', 'Option 3', 'Option 4', 'Option 5', 'Option 6', 'Option 7', 'Option 8', 'Option 9'], index=test_years_indices)

Waitingtransfdn = pd.DataFrame(columns =['Option 1', 'Option 2', 'Option 3', 'Option 4', 'Option 5', 'Option 6', 'Option 7', 'Option 8', 'Option 9'], index=test_years_indices)
Waitingjackfdn = pd.DataFrame(columns =['Option 1', 'Option 2', 'Option 3', 'Option 4', 'Option 5', 'Option 6', 'Option 7', 'Option 8', 'Option 9'], index=test_years_indices)
Waitingmono = pd.DataFrame(columns =['Option 1', 'Option 2', 'Option 3', 'Option 4', 'Option 5', 'Option 6', 'Option 7', 'Option 8', 'Option 9'], index=test_years_indices)
Waitingtp = pd.DataFrame(columns =['Option 1', 'Option 2', 'Option 3', 'Option 4', 'Option 5', 'Option 6', 'Option 7', 'Option 8', 'Option 9'], index=test_years_indices)
Waitingposfdn = pd.DataFrame(columns =['Option 1', 'Option 2', 'Option 3', 'Option 4', 'Option 5', 'Option 6', 'Option 7', 'Option 8', 'Option 9'], index=test_years_indices)

Waitingtranswtb = pd.DataFrame(columns =['Option 1', 'Option 2', 'Option 3', 'Option 4', 'Option 5', 'Option 6', 'Option 7', 'Option 8', 'Option 9'], index=test_years_indices)
Waitingposwtb = pd.DataFrame(columns =['Option 1', 'Option 2', 'Option 3', 'Option 4', 'Option 5', 'Option 6', 'Option 7', 'Option 8', 'Option 9'], index=test_years_indices)
Waitingjackwtb = pd.DataFrame(columns =['Option 1', 'Option 2', 'Option 3', 'Option 4', 'Option 5', 'Option 6', 'Option 7', 'Option 8', 'Option 9'], index=test_years_indices)
Waitingtow = pd.DataFrame(columns =['Option 1', 'Option 2', 'Option 3', 'Option 4', 'Option 5', 'Option 6', 'Option 7', 'Option 8', 'Option 9'], index=test_years_indices)
Waitingnac = pd.DataFrame(columns =['Option 1', 'Option 2', 'Option 3', 'Option 4', 'Option 5', 'Option 6', 'Option 7', 'Option 8', 'Option 9'], index=test_years_indices)
Waitingblade = pd.DataFrame(columns =['Option 1', 'Option 2', 'Option 3', 'Option 4', 'Option 5', 'Option 6', 'Option 7', 'Option 8', 'Option 9'], index=test_years_indices)

WAIT_TOT = pd.DataFrame(columns =['Option 1', 'Option 2', 'Option 3', 'Option 4', 'Option 5', 'Option 6', 'Option 7', 'Option 8', 'Option 9'], index=test_years_indices)

Overall_metrics_TP = pd.DataFrame(columns =['Option 1', 'Option 2', 'Option 3', 'Option 4', 'Option 5', 'Option 6', 'Option 7', 'Option 8', 'Option 9'], index=test_years_indices)
Overall_metrics_FP = pd.DataFrame(columns =['Option 1', 'Option 2', 'Option 3', 'Option 4', 'Option 5', 'Option 6', 'Option 7', 'Option 8', 'Option 9'], index=test_years_indices)
Overall_metrics_TN = pd.DataFrame(columns =['Option 1', 'Option 2', 'Option 3', 'Option 4', 'Option 5', 'Option 6', 'Option 7', 'Option 8', 'Option 9'], index=test_years_indices)
Overall_metrics_FN = pd.DataFrame(columns =['Option 1', 'Option 2', 'Option 3', 'Option 4', 'Option 5', 'Option 6', 'Option 7', 'Option 8', 'Option 9'], index=test_years_indices)
Overall_accuracy = pd.DataFrame(columns =['Option 1', 'Option 2', 'Option 3', 'Option 4', 'Option 5', 'Option 6', 'Option 7', 'Option 8', 'Option 9'], index=test_years_indices)
Overall_precision = pd.DataFrame(columns =['Option 1', 'Option 2', 'Option 3', 'Option 4', 'Option 5', 'Option 6', 'Option 7', 'Option 8', 'Option 9'], index=test_years_indices)
Overall_recall = pd.DataFrame(columns =['Option 1', 'Option 2', 'Option 3', 'Option 4', 'Option 5', 'Option 6', 'Option 7', 'Option 8', 'Option 9'], index=test_years_indices)

S_wht_metrics_TP = pd.DataFrame(columns =['Option 1', 'Option 2', 'Option 3', 'Option 4', 'Option 5', 'Option 6', 'Option 7', 'Option 8', 'Option 9'], index=test_years_indices)
S_wht_metrics_FP = pd.DataFrame(columns =['Option 1', 'Option 2', 'Option 3', 'Option 4', 'Option 5', 'Option 6', 'Option 7', 'Option 8', 'Option 9'], index=test_years_indices)
S_wht_metrics_TN = pd.DataFrame(columns =['Option 1', 'Option 2', 'Option 3', 'Option 4', 'Option 5', 'Option 6', 'Option 7', 'Option 8', 'Option 9'], index=test_years_indices)
S_wht_metrics_FN = pd.DataFrame(columns =['Option 1', 'Option 2', 'Option 3', 'Option 4', 'Option 5', 'Option 6', 'Option 7', 'Option 8', 'Option 9'], index=test_years_indices)

Wind_speed_metrics_TP = pd.DataFrame(columns =['Option 1', 'Option 2', 'Option 3', 'Option 4', 'Option 5', 'Option 6', 'Option 7', 'Option 8', 'Option 9'], index=test_years_indices)
Wind_speed_metrics_FP = pd.DataFrame(columns =['Option 1', 'Option 2', 'Option 3', 'Option 4', 'Option 5', 'Option 6', 'Option 7', 'Option 8', 'Option 9'], index=test_years_indices)
Wind_speed_metrics_TN = pd.DataFrame(columns =['Option 1', 'Option 2', 'Option 3', 'Option 4', 'Option 5', 'Option 6', 'Option 7', 'Option 8', 'Option 9'], index=test_years_indices)
Wind_speed_metrics_FN = pd.DataFrame(columns =['Option 1', 'Option 2', 'Option 3', 'Option 4', 'Option 5', 'Option 6', 'Option 7', 'Option 8', 'Option 9'], index=test_years_indices)

peak_period_metrics_TP = pd.DataFrame(columns =['Option 1', 'Option 2', 'Option 3', 'Option 4', 'Option 5', 'Option 6', 'Option 7', 'Option 8', 'Option 9'], index=test_years_indices)
peak_period_metrics_FP = pd.DataFrame(columns =['Option 1', 'Option 2', 'Option 3', 'Option 4', 'Option 5', 'Option 6', 'Option 7', 'Option 8', 'Option 9'], index=test_years_indices)
peak_period_metrics_TN = pd.DataFrame(columns =['Option 1', 'Option 2', 'Option 3', 'Option 4', 'Option 5', 'Option 6', 'Option 7', 'Option 8', 'Option 9'], index=test_years_indices)
peak_period_metrics_FN = pd.DataFrame(columns =['Option 1', 'Option 2', 'Option 3', 'Option 4', 'Option 5', 'Option 6', 'Option 7', 'Option 8', 'Option 9'], index=test_years_indices)

mean_overall_metrics = pd.DataFrame(columns =['Option 1', 'Option 2', 'Option 3', 'Option 4', 'Option 5', 'Option 6', 'Option 7', 'Option 8', 'Option 9'], index=['TP', 'FP', 'TN', 'FN'])

forecasting_period = prediction_horizon+datapoints_predicted-1
forecasting_periods_s_wht = s_wht_Alpha_Factors.iloc[0, 1:]
forecasting_periods_wind_speed = wind_speed_Alpha_Factors.iloc[0, 1:]

# Select applicable alpha factors from alpha factor table
i = 0
while forecasting_period > forecasting_periods_s_wht.iloc[i]:
    i+=1

s_wht_alpha_factors = s_wht_Alpha_Factors.iloc[1:, i+1]

i = 0
while forecasting_period > forecasting_periods_wind_speed.iloc[i]:
    i+=1
    
wind_speed_alpha_factors = wind_speed_Alpha_Factors.iloc[1:3, i+1]
wind_speed_10_years_return = wind_speed_Alpha_Factors.iloc[4,1]
wind_speed_alpha_factors.index = [wind_speed_10_years_return, wind_speed_10_years_return]
           
for option in range (1,10):                     
    for year in Test_years:
        
        VT_FDN = Data_Config.iloc[option,1]                     #Vessel type used for foundation installation  0 = Crane barge, 1= HLV, 2 = Jack-up 
        VT_WTB = Data_Config.iloc[option,2]                     #Vessel type used for turbine installation, 0= Crane barge, 1=HLV, 2= Jack-up
        
        # %% Data input 
        
        #------------Farm Characteristics------ 
        
        N_FDN = Data_Farm.iloc[1,2]								#Number of foundations to be installed 
        N_WTB = Data_Farm.iloc[2,2]								#Number of turbines to be installed 
        WTB_D = Data_Farm.iloc[3,2]								#Turbine Rotor Diameter 
        INT_D = Data_Farm.iloc[4,2]								#Distance between installation locations =5*Rotor diameter 
        WD = Data_Farm.iloc[5,2]								#Water depth 
        PEN_FDN = Data_Farm.iloc[6,2]							#Penetration depth Monopile 
        N_TOW = Data_Farm.iloc[7,2]								#Number of tower pieces 
        N_BLADE = Data_Farm.iloc[8,2]							#Number of blades 
        PARK_DIS = Data_Farm.iloc[9,2]						    #Distance farm to feeding harbour 
        
        
        #--------------Vessel Characteristics----- 
        
        # Foundation vessel 
        TRANS_VEL_FDN= Data_Vessel.iloc[2,2+VT_FDN]               #Transition Velocity of selected Foundation installation Vessel 
        CAP_FDN= Data_Vessel.iloc[3,2+VT_FDN]					  #Loading capacity Foundation of selected Foundation installation Vessel
        HAM_SP = Data_Vessel.iloc[5,2+VT_FDN]	                  #Hamering Speed of selected Foundation Installation Vessel 			
        AIR_FDN = Data_Vessel.iloc[7,2+VT_FDN]					  #Air gap between vessel and sea level, only for jack-up Vessels 
        PEN_LEG_FDN = Data_Vessel.iloc[8,2+VT_FDN]				  #Penetration depth of the legs of jack-up vessel 
        JACK_SP_FDN = Data_Vessel.iloc[9,2+VT_FDN]				  #Jack-up-speed, only for jack-up vessels
        CHT_C_FDN =  Data_Vessel.iloc[12,2+VT_FDN]				  #Day-rate charter cost FDN vessel 
        JACKING_FDN =Data_Vessel.iloc[14,2+VT_FDN]                #Does the vessel require jacking 
        SUPPLY_FDN = Data_Vessel.iloc[15,2+VT_FDN]
        
        
        
        # Turbine installation vessel 
        TRANS_VEL_WTB= Data_Vessel.iloc[2,2+VT_WTB]               #Transition Velocity WTB Vessel 
        CAP_WTB = Data_Vessel.iloc[4,2+VT_WTB]					  #Loading capacity WTB vessel 
        AIR_WTB = Data_Vessel.iloc[7,2+VT_WTB]					  #Air gap between vessel and sea level, only for jack-up Vessels 
        PEN_LEG_WTB =Data_Vessel.iloc[8,2+VT_WTB]				  #Penetration depth of the legs of jack-up vessel 
        JACK_SP_WTB = Data_Vessel.iloc[9,2+VT_WTB]				  #Jack-up-speed, only for jack-up vessels
        CHT_C_WTB =  Data_Vessel.iloc[12,2+VT_WTB]				  #Day-rate charter cost selected WTB vessel 
        JACKING_WTB = Data_Vessel.iloc[14,2+VT_WTB]               #Does the vessel require jacking 
        SUPPLY_WTB =Data_Vessel.iloc[15,2+VT_WTB]
        
        #---------Operation Characteristics -----
        
        LT_FDN 		= Data_Operation.iloc[2,2+VT_FDN]			  #Loading time foundations 
        LT_WTB 		= Data_Operation.iloc[3,2+VT_WTB]			  #Loading time Turbine components (includes towers, nacelles and blades)
        POS_FDN 	= Data_Operation.iloc[4,2+VT_FDN]			  #Postitioning Time of selected Foundation Installation Vessel 
        POS_WTB 	= Data_Operation.iloc[4,2+VT_WTB]			  #Postiitioning Time of selected Wind Turbine installation vessel 
        LFT_FDN 	= Data_Operation.iloc[5,2+VT_FDN]			  #Lifting time foundation 
        LFT_TP		= Data_Operation.iloc[6,2+VT_FDN]		      #Lifting time transition piece 
        GRT_TP 		= Data_Operation.iloc[7,2+VT_FDN]			  #Grouting time transition piece 
        LFT_TOW 	= Data_Operation.iloc[8,2+VT_WTB]			  #Lifting time tower piece 
        LFT_NAC 	= Data_Operation.iloc[9,2+VT_WTB]			  #Lifting time nacelle 
        LFT_BLADE	= Data_Operation.iloc[10,2+VT_WTB]			  #lifting time blade 
        INST_BLADE 	= Data_Operation.iloc[11,2+VT_WTB]		      #installation time blade 
        
        
        #---------Operational limits ---------
        
        # Foundation installation 
        HS_TRANS_FDN = Data_Limits.iloc[2+VT_FDN,1]               #Significant wave height limit transportation Foundation Vessel 
        HS_POS_FDN = Data_Limits.iloc[2+VT_FDN,2]                 #Significant wave height limit positioning Foundation Vessel 
        HS_MONO =Data_Limits.iloc[2+VT_FDN,3]                     #Significant wave height limit monopile installation  
        TP_MONO =Data_Limits.iloc[2+VT_FDN,4]                     #peak period limit monopile installation 
        WS_MONO = Data_Limits.iloc[2+VT_FDN,5]                    #Wind speed limit monopile installation 
        HS_TP = Data_Limits.iloc[2+VT_FDN,6]                      #Significant wave height limit transition piece installation     
        TP_TP =Data_Limits.iloc[2+VT_FDN,7]                       #peak period limit transition piece installation 
        WS_TP =Data_Limits.iloc[2+VT_FDN,8]                       #Wind speed limit transtition piece installation 
       
        # Turbine installation
        HS_TRANS_WTB = Data_Limits.iloc[10+VT_WTB,1]              #Significant wave height limit transportation Wind turbine Vessel 
        HS_POS_WTB = Data_Limits.iloc[10+VT_WTB,2]                #Significant wave height limit positioning Wind turbine Vessel 
        HS_TOW = Data_Limits.iloc[10+VT_WTB,3]                    #Significant wave height limit tower installation 
        TP_TOW = Data_Limits.iloc[10+VT_WTB,4]                    #peak period limit tower installation 
        WS_TOW = Data_Limits.iloc[10+VT_WTB,5]                    #Wind speed limit tower 
        HS_NAC = Data_Limits.iloc[10+VT_WTB,6]                    #Significant wave height limit nacelle installation 
        TP_NAC = Data_Limits.iloc[10+VT_WTB,7]                    #peak period limit nacelle installation 
        WS_NAC = Data_Limits.iloc[10+VT_WTB,8]                    #Wind speed limit nacelle installation 
        HS_BLADE = Data_Limits.iloc[10+VT_WTB,9]                  #Significant wave height limit blade installation 
        TP_BLADE = Data_Limits.iloc[10+VT_WTB,10]                 #peak period limit blade installation 
        WS_BLADE = Data_Limits.iloc[10+VT_WTB,11]                 #Wind speed limit blade installation 
        HS_JACK = 1                                               #Significant wave height limit jacking       
 
        #--------- Penalties ---------
        FP_PEN = 2                                                #Penalty for False Positive predictions (possible damages)
        
        #%% Weather Data input 
        startdate = str(year)+'-05-01 00:00:00'                   #Starting on the first day of summer on the selected year
        ts = pd.to_datetime(startdate)                            #Creates a timestamp for the start of the year      
        start = (ts.dayofyear-1)*24+ts.hour+prediction_horizon    #Gives us the hour of the year, including the prediction horizon 
        
        df2 = pd.read_table("Wave/"+str(year)+".txt") 
        df2['peak_period'] = 1/df2.peak_fr                        #Take inverse of peak frequency for peak period
        df2 = df2[['datetime','s_wht','wind_speed','peak_period']]
        
        df1 = pd.read_table("Wave/"+str(year)+"pred.txt")    
        #df1 = df2.copy()                                         #uncomment for perfect forecast, comment previous line      
        
        #%% Starting parameter set up
        
        TOTAL_TIME_FDN = 0 
        stock_FDN = N_FDN 
        waiting = 0
        waitingtransfdn = 0
        waitingjackfdn = 0 
        waitingmono = 0 
        waitingtp = 0
        waitingposfdn = 0
                
        FDN_Installed = 0 
        FDN_Onboard = 0
        
        check_duration = 0                                        #Used to check if foundation and turbine installation duration does not exceed the end of the year
        
        overall_metrics = np.zeros(4, dtype=int)
        s_wht_metrics = np.zeros(4, dtype=int)
        wind_speed_metrics = np.zeros(4, dtype=int)
        peak_period_metrics = np.zeros(4, dtype=int)
        
        #%% Operation steps installing foundations without supply barges 
        
        if SUPPLY_FDN == 0:                                                 #Check if foundation vessel uses supply barges
            
            while FDN_Installed<N_FDN:										#Loop until the desired number of foundations is installed 
                
                #-------- Step 1 --------
                #Loading foundations in port
                
                if stock_FDN > CAP_FDN:										#Check if the stock at the port is sufficent to load vessel 
                    FDN_Onboard = CAP_FDN						            #Number of foundations on board is loaded untill vessel capacity is reached 
                
                else:
                    FDN_Onboard = stock_FDN									#If stock level is too low load remaining foundations 
                
                stock_FDN = stock_FDN - FDN_Onboard							#New stocklevel after loading 
                
                DUR_LOAD_FDN = FDN_Onboard*LT_FDN 							#Determine time needed to load foundations
                TOTAL_TIME_FDN = TOTAL_TIME_FDN + DUR_LOAD_FDN              #Add time spent loading foundations to total time installing foundations
                #print("Loading complete")
                #print ("stockfdn",stock_FDN)
                #print("fdn on deck after port",FDN_Onboard)
                #print("Left port")
                #print("total time after loading",TOTAL_TIME_FDN)
                
                #-------- Step 2 --------
                #Travel to site for installation of foundations
                
                DUR_TRANS_FDN = (PARK_DIS/TRANS_VEL_FDN)                                #Determine time needed for travelling to site		
                #durationtrans = math.ceil(DUR_TRANS_FDN)
                        
                weather_check_results = weather_check(TOTAL_TIME_FDN, start, 
                    DUR_TRANS_FDN, df1, df2, s_wht_alpha_factors, wind_speed_alpha_factors, s_wht_limit=HS_TRANS_FDN, 
                    wind_speed_limit=None, peak_period_limit=None)                      #Find suitable weather window for travelling to site
                
                waiting = waiting + weather_check_results[1]                            #Add time waiting on weather to total time waiting
                TOTAL_TIME_FDN = weather_check_results[0]                               #Updated total time installing foundations
                waitingtransfdn = waitingtransfdn + weather_check_results[1]            #Add time waiting on weather to time waiting for transit foundations
                
                overall_metrics = overall_metrics + weather_check_results[2]            #Add overall metrics of forecasting model
                s_wht_metrics = s_wht_metrics + weather_check_results[3]                #Add s_wht metrics of forecasting model
                wind_speed_metrics = wind_speed_metrics + weather_check_results[4]      #Add wind_speed metrics of forecasting model
                peak_period_metrics = peak_period_metrics + weather_check_results[5]    #Add peak_period metrics of forecasting model
                #print(overall_metrics)
       
                #print("total time after transit",TOTAL_TIME_FDN)                
                
                while FDN_Onboard>0:                                                    #Loop to run untill all foundation on board are installed 
            																
                    #-------- Step 3--------
                    #Position vessel for installation of foundation
                    
                    DUR_POS_FDN = POS_FDN                                               #Determine the time needed to position the vessel for installation of foundation          
                    #durationpos = math.ceil(DUR_POS_FDN)               
                    
                    weather_check_results = weather_check(TOTAL_TIME_FDN, start, 
                        DUR_POS_FDN, df1, df2, s_wht_alpha_factors, wind_speed_alpha_factors, s_wht_limit=HS_POS_FDN, 
                        wind_speed_limit=None, peak_period_limit=None)                  #Find suitable weather window for positioning of the vessel for installation of foundation
                    
                    waiting = waiting + weather_check_results[1]                        #Add time waiting on weather to total time waiting
                    TOTAL_TIME_FDN = weather_check_results[0]                           #Updated total time installing foundations
                    waitingposfdn = waitingposfdn + weather_check_results[1]            #Add time waiting on weather to time waiting for positioning the vessel
                    
                    overall_metrics = overall_metrics + weather_check_results[2]            #Add overall metrics of forecasting model
                    s_wht_metrics = s_wht_metrics + weather_check_results[3]                #Add s_wht metrics of forecasting model
                    wind_speed_metrics = wind_speed_metrics + weather_check_results[4]      #Add wind_speed metrics of forecasting model
                    peak_period_metrics = peak_period_metrics + weather_check_results[5]    #Add peak_period metrics of forecasting model
                    #print(overall_metrics)
                    
                    #print('time after positioning', TOTAL_TIME_FDN)
                    
                    #-------- Step 4 --------
                    #Jack-up vessel for installation of foundation
                    
                    if JACKING_FDN == 1:											    #Check if foundation vessel is a jack-up vessel
                    
                        DUR_JACK_FDN =(PEN_LEG_FDN + AIR_FDN + WD) /JACK_SP_FDN         #Determine time needed to jack-up the vessel for installation of foundation
                        #durationjack = DUR_JACK_FDN
                        
                        weather_check_results = weather_check(TOTAL_TIME_FDN, start, 
                            DUR_JACK_FDN, df1, df2, s_wht_alpha_factors, wind_speed_alpha_factors, s_wht_limit=HS_JACK, 
                            wind_speed_limit=None, peak_period_limit=None)                     #Find suitable weather window for jack-up of the vessel for installation of foundation
                        
                        waiting = waiting + weather_check_results[1]                            #Add time waiting on weather to total time waiting
                        TOTAL_TIME_FDN = weather_check_results[0]                               #Updated total time installing foundations
                        waitingjackfdn = waitingjackfdn + weather_check_results[1]              #Add time waiting on weather to time waiting for jacking-up/down the vessel
                        
                        overall_metrics = overall_metrics + weather_check_results[2]            #Add overall metrics of forecasting model
                        s_wht_metrics = s_wht_metrics + weather_check_results[3]                #Add s_wht metrics of forecasting model
                        wind_speed_metrics = wind_speed_metrics + weather_check_results[4]      #Add wind_speed metrics of forecasting model
                        peak_period_metrics = peak_period_metrics + weather_check_results[5]    #Add peak_period metrics of forecasting model
                        #print(overall_metrics)
                        
                        #print("Time after jacking-up vessel:",TOTAL_TIME_FDN)

                    else:
                        DUR_JACK_FDN = 0
                    
                    #-------- Step 5 and 6 --------
                    #Lift and position monopile on seabed
                    #Hammer monopile down using hydraulic jack
                    
                    DUR_HOIST_FDN = LFT_FDN 								            #Determine time needed to lift and position monopile on seabed
                    DUR_HAM = PEN_FDN/HAM_SP          						            #Determine the time needed to hammer monopile into seabed
                    DUR_MONO = DUR_HOIST_FDN + DUR_HAM
                    #durationmono= math.ceil(DUR_HOIST_FDN + DUR_HAM)
                    
                    weather_check_results = weather_check(TOTAL_TIME_FDN, start, 
                        DUR_MONO, df1, df2, s_wht_alpha_factors, wind_speed_alpha_factors, s_wht_limit=HS_MONO, 
                        wind_speed_limit=WS_MONO, peak_period_limit=TP_MONO)                    #Find suitable weather window for lifting, positioning and hammering of foundation
                    
                    waiting = waiting + weather_check_results[1]                                #Add time waiting on weather to total time waiting
                    TOTAL_TIME_FDN = weather_check_results[0]                                   #Updated total time installing foundations
                    waitingmono = waitingmono + weather_check_results[1]                        #Add time waiting on weather to time waiting for lifting, positioning and hammering of foundations
                                            
                    overall_metrics = overall_metrics + weather_check_results[2]                #Add overall metrics of forecasting model
                    s_wht_metrics = s_wht_metrics + weather_check_results[3]                    #Add s_wht metrics of forecasting model
                    wind_speed_metrics = wind_speed_metrics + weather_check_results[4]          #Add wind_speed metrics of forecasting model
                    peak_period_metrics = peak_period_metrics + weather_check_results[5]        #Add peak_period metrics of forecasting model
                    #print(overall_metrics)
                    
                    #print('time after positioning and hammering foundation:', TOTAL_TIME_FDN)
                    
                    #-------- Step 7 and 8 --------
                    #Lifting transition piece into place
                    #Installing/grouting transition piece
                    
                    DUR_HOIST_TP = LFT_TP 									            #Determine time needed to lift transition piece into place
                    DUR_INST_TP = GRT_TP									            #Determine time needed to grout the transition piece
                    DUR_TP = DUR_HOIST_TP + DUR_INST_TP
                    #durationtp = math.ceil(DUR_HOIST_TP + DUR_INST_TP)
                    
                    weather_check_results = weather_check(TOTAL_TIME_FDN, start, 
                        DUR_TP, df1, df2, s_wht_alpha_factors, wind_speed_alpha_factors, s_wht_limit=HS_TP, 
                        wind_speed_limit=WS_TP, peak_period_limit=TP_TP)                    #Find suitable weather window for lifting and grouting of transition piece
                    
                    waiting = waiting + weather_check_results[1]                            #Add time waiting on weather to total time waiting
                    TOTAL_TIME_FDN = weather_check_results[0]                               #Updated total time installing foundations
                    waitingtp = waitingtp + weather_check_results[1]                        #Add time waiting on weather to time waiting for lifting and grouting of transition piece
                                            
                    overall_metrics = overall_metrics + weather_check_results[2]            #Add overall metrics of forecasting model 
                    s_wht_metrics = s_wht_metrics + weather_check_results[3]                #Add s_wht metrics of forecasting model
                    wind_speed_metrics = wind_speed_metrics + weather_check_results[4]      #Add wind_speed metrics of forecasting model
                    peak_period_metrics = peak_period_metrics + weather_check_results[5]    #Add peak_period metrics of forecasting model
                    #print(overall_metrics)
                    
                    #print('time after lifting and grouting transition piece:', TOTAL_TIME_FDN)
                    
                    #-------- Step 9 --------
                    #Jack-down vessel after installation of foundation
                    
                    if JACKING_FDN == 1:                                                #Check if foundation vessel is a jack-up vessel
                        
                        DUR_JACK_FDN = (PEN_LEG_FDN + AIR_FDN + WD) /JACK_SP_FDN        #Determine time needed to jack-down the vessel after installation of foundation
                        #durationjack = DUR_JACK_FDN
                        
                        weather_check_results = weather_check(TOTAL_TIME_FDN, start, 
                            DUR_JACK_FDN, df1, df2, s_wht_alpha_factors, wind_speed_alpha_factors, s_wht_limit=HS_JACK, 
                            wind_speed_limit=None, peak_period_limit=None)                     #Find suitable weather window for jack-down of the vessel after installation of foundation
                    
                        waiting = waiting + weather_check_results[1]                            #Add time waiting on weather to total time waiting
                        TOTAL_TIME_FDN = weather_check_results[0]                               #Updated total time installing foundations
                        waitingjackfdn = waitingjackfdn + weather_check_results[1]              #Add time waiting on weather to time waiting for jacking-up/down the vessel
                                            
                        overall_metrics = overall_metrics + weather_check_results[2]            #Add overall metrics of forecasting model 
                        s_wht_metrics = s_wht_metrics + weather_check_results[3]                #Add s_wht metrics of forecasting model
                        wind_speed_metrics = wind_speed_metrics + weather_check_results[4]      #Add wind_speed metrics of forecasting model
                        peak_period_metrics = peak_period_metrics + weather_check_results[5]    #Add peak_period metrics of forecasting model
                        #print(overall_metrics)
                        
                        #print('time after jacking-down vessel:', TOTAL_TIME_FDN) 
 		
                    else:
                        DUR_JACK_FDN = 0

                    FDN_Installed = FDN_Installed+1                                     #Update number of foundations installed
                    FDN_Onboard = FDN_Onboard -1                                        #Update number of foundations still onboard
                    
                    #print('time after installing foundation',FDN_Installed,':',TOTAL_TIME_FDN)
                
                    #-------- Step 10 --------
                    #Reposition vessel to start installation of next foundation
                
                    DUR_REPO_FDN = (INT_D/TRANS_VEL_FDN)                                #Determine duration of repositioning
                    TOTAL_TIME_FDN = TOTAL_TIME_FDN + DUR_REPO_FDN                      #Add time spent repositioning to total time installing foundations
                    
                    #print('time after repositioning', TOTAL_TIME_FDN)
                
                #-------- Step 11 --------
                #Travel back to port to load more foundations or when finished installing all foundations
                
                else:
                    DUR_TRANS_FDN = (PARK_DIS/TRANS_VEL_FDN)				            #Determine time needed for travelling back to port
                    #durationtrans = math.ceil(DUR_TRANS_FDN)
                    
                    weather_check_results = weather_check(TOTAL_TIME_FDN, start, 
                        DUR_TRANS_FDN, df1, df2, s_wht_alpha_factors, wind_speed_alpha_factors, s_wht_limit=HS_TRANS_FDN, 
                        wind_speed_limit=None, peak_period_limit=None)                 #Find suitable weather window to travel back to port
                    
                    waiting = waiting + weather_check_results[1]                        #Add time waiting on weather to total time waiting
                    TOTAL_TIME_FDN = weather_check_results[0]                           #Updated total time installing foundations
                    waitingtransfdn = waitingtransfdn + weather_check_results[1]        #Add time waiting on weather to time waiting for transit foundations
                                            
                    overall_metrics = overall_metrics + weather_check_results[2]            #Add overall metrics of forecasting model 
                    s_wht_metrics = s_wht_metrics + weather_check_results[3]                #Add s_wht metrics of forecasting model
                    wind_speed_metrics = wind_speed_metrics + weather_check_results[4]      #Add wind_speed metrics of forecasting model
                    peak_period_metrics = peak_period_metrics + weather_check_results[5]    #Add peak_period metrics of forecasting model
                    #print(overall_metrics)
                    
                    #print('return to port')
                    
            #print('TOTAL time installation foundations', math.ceil(TOTAL_TIME_FDN), 'hours')   
     
        #%% Operation steps installing foundations with supply barges 
            
        elif SUPPLY_FDN == 1:
            
            #print("Left port")
            
            #-------- Step 2 --------
            #Travel to site for installation of foundations
            
            DUR_TRANS_FDN = (PARK_DIS/TRANS_VEL_FDN)                                    #Determine time needed for travelling to site			

            weather_check_results = weather_check(TOTAL_TIME_FDN, start, DUR_TRANS_FDN, df1, df2, s_wht_alpha_factors, wind_speed_alpha_factors, s_wht_limit=HS_TRANS_FDN)                      #Find suitable weather window to travel to site
                    
            waiting = waiting + weather_check_results[1]                                #Add time waiting on weather to total time waiting
            TOTAL_TIME_FDN = weather_check_results[0]                                   #Updated total time installing foundations
            waitingtransfdn = waitingtransfdn + weather_check_results[1]                #Add time waiting on weather to time waiting for transit foundations
                                            
            overall_metrics = overall_metrics + weather_check_results[2]                #Add overall metrics of forecasting model 
            s_wht_metrics = s_wht_metrics + weather_check_results[3]                    #Add s_wht metrics of forecasting model
            wind_speed_metrics = wind_speed_metrics + weather_check_results[4]          #Add wind_speed metrics of forecasting model
            peak_period_metrics = peak_period_metrics + weather_check_results[5]        #Add peak_period metrics of forecasting model
            #print(overall_metrics)
            
            #print("total time after transit",TOTAL_TIME_FDN)            
            
            while FDN_Installed<N_FDN:										            #Loop until all the desired number of turbines are installed 
                    
                while FDN_Onboard>0:                                                    #Loop to run untill all foundation on board are installed 
                
                    #-------- Step 3 and 4 --------
                    #Position vessel for installation of foundation
                    #Jack-up vessel for installation of foundation
                
                    DUR_POS_FDN = POS_FDN									            #Determine the time needed to position the vessel for installation of foundation
                    
                    if JACKING_FDN == 1:											    #Check if foundation vessel is a jack-up vessel
                        DUR_JACK_FDN =(PEN_LEG_FDN + AIR_FDN + WD) /JACK_SP_FDN         #Determine time needed to jack-up the vessel for installation of foundation
		
                    else:
                        DUR_JACK_FDN = 0
                    
                    DUR_POS = DUR_POS_FDN + DUR_JACK_FDN
                    #durationpos = math.ceil(DUR_POS_FDN+DUR_JACK_FDN)
                    
                    weather_check_results = weather_check(TOTAL_TIME_FDN, start, 
                        DUR_POS, df1, df2, s_wht_alpha_factors, wind_speed_alpha_factors, s_wht_limit=HS_POS_FDN, 
                        wind_speed_limit=None, peak_period_limit=None)                 #Find suitable weather window to position and jack-up the vessel
                    
                    waiting = waiting + weather_check_results[1]                        #Add time waiting on weather to total time waiting
                    TOTAL_TIME_FDN = weather_check_results[0]                           #Updated total time installing foundations
                    waitingposfdn = waitingposfdn + weather_check_results[1]            #Add time waiting on weather to time waiting for positioning the vessel
                                            
                    overall_metrics = overall_metrics + weather_check_results[2]            #Add overall metrics of forecasting model 
                    s_wht_metrics = s_wht_metrics + weather_check_results[3]                #Add s_wht metrics of forecasting model
                    wind_speed_metrics = wind_speed_metrics + weather_check_results[4]      #Add wind_speed metrics of forecasting model
                    peak_period_metrics = peak_period_metrics + weather_check_results[5]    #Add peak_period metrics of forecasting model
                    #print(overall_metrics)
                    
                    #print('time after positioning and jacking-up:', TOTAL_TIME_FDN)
                    
                    #-------- Step 5 and 6 --------
                    #Lift and position monopile on seabed
                    #Hammer monopile down using hydraulic jack
                
                    DUR_HOIST_FDN = LFT_FDN                                             #Determine time needed to lift and position monopile on seabed
                    DUR_HAM = PEN_FDN/HAM_SP          						            #Determine the time needed to hammer monopile into seabed
                    DUR_MONO = DUR_HOIST_FDN + DUR_HAM
                    #durationmono= math.ceil(DUR_HOIST_FDN + DUR_HAM)
                    
                    weather_check_results = weather_check(TOTAL_TIME_FDN, start, 
                        DUR_MONO, df1, df2, s_wht_alpha_factors, wind_speed_alpha_factors, s_wht_limit=HS_MONO, 
                        wind_speed_limit=WS_MONO, peak_period_limit=TP_MONO)                    #Find suitable weather window for lifting, positioning and hammering of foundation
                    
                    waiting = waiting + weather_check_results[1]                                #Add time waiting on weather to total time waiting
                    TOTAL_TIME_FDN = weather_check_results[0]                                   #Updated total time installing foundations
                    waitingmono = waitingmono + weather_check_results[1]                        #Add time waiting on weather to time waiting for lifting, positioning and hammering of foundations
                                            
                    overall_metrics = overall_metrics + weather_check_results[2]                #Add overall metrics of forecasting model 
                    s_wht_metrics = s_wht_metrics + weather_check_results[3]                    #Add s_wht metrics of forecasting model
                    wind_speed_metrics = wind_speed_metrics + weather_check_results[4]          #Add wind_speed metrics of forecasting model
                    peak_period_metrics = peak_period_metrics + weather_check_results[5]        #Add peak_period metrics of forecasting model
                    #print(overall_metrics)
                    
                    #print('time after lifting and hammering monopile:', TOTAL_TIME_FDN)
                    
                    #-------- Step 7 and 8 --------
                    #Lifting transition piece into place
                    #Installing/grouting transition piece
                    
                    DUR_HOIST_TP = LFT_TP                                               #Determine time needed to lift transition piece into place
                    DUR_INST_TP = GRT_TP									            #Determine time needed to grout the transition piece
                    DUR_TP = DUR_HOIST_TP + DUR_INST_TP
                    #durationtp = math.ceil(DUR_HOIST_TP + DUR_INST_TP)
                    
                    weather_check_results = weather_check(TOTAL_TIME_FDN, start, 
                        DUR_TP, df1, df2, s_wht_alpha_factors, wind_speed_alpha_factors, s_wht_limit=HS_TP, 
                        wind_speed_limit=WS_TP, peak_period_limit=TP_TP)                    #Find suitable weather window for lifting and grouting of transition piece
                    
                    waiting = waiting + weather_check_results[1]                            #Add time waiting on weather to total time waiting
                    TOTAL_TIME_FDN = weather_check_results[0]                               #Updated total time installing foundations
                    waitingtp = waitingtp + weather_check_results[1]                        #Add time waiting on weather to time waiting for lifting and grouting of transition pieces
                                            
                    overall_metrics = overall_metrics + weather_check_results[2]            #Add overall metrics of forecasting model 
                    s_wht_metrics = s_wht_metrics + weather_check_results[3]                #Add s_wht metrics of forecasting model
                    wind_speed_metrics = wind_speed_metrics + weather_check_results[4]      #Add wind_speed metrics of forecasting model
                    peak_period_metrics = peak_period_metrics + weather_check_results[5]    #Add peak_period metrics of forecasting model
                    #print(overall_metrics)
                    
                    #print('time after lifting and grouting transition piece:', TOTAL_TIME_FDN)
                    
                    #-------- Step 9 --------
                    #Jack-down vessel after installation of foundation
                    
                    if JACKING_FDN == 1:										        #Check if foundation vessel is a jack-up vessel
                        DUR_JACK_FDN =(PEN_LEG_FDN + AIR_FDN + WD) /JACK_SP_FDN         #Determine time needed to jack-up the vessel for installation of foundation
                        #durationjack = DUR_JACK_FDN
                        
                        weather_check_results = weather_check(TOTAL_TIME_FDN, start, 
                            DUR_JACK_FDN, df1, df2, s_wht_alpha_factors, wind_speed_alpha_factors, s_wht_limit=HS_JACK, 
                            wind_speed_limit=None, peak_period_limit=None)                     #Find suitable weather window for jack-down of the vessel after installation of foundation
                    
                        waiting = waiting + weather_check_results[1]                            #Add time waiting on weather to total time waiting
                        TOTAL_TIME_FDN = weather_check_results[0]                               #Updated total time installing foundations
                        waitingjackfdn = waitingjackfdn + weather_check_results[1]              #Add time waiting on weather to time waiting for jacking-up/down of vessel during installation of foundations
                                            
                        overall_metrics = overall_metrics + weather_check_results[2]            #Add overall metrics of forecasting model 
                        s_wht_metrics = s_wht_metrics + weather_check_results[3]                #Add s_wht metrics of forecasting model
                        wind_speed_metrics = wind_speed_metrics + weather_check_results[4]      #Add wind_speed metrics of forecasting model
                        peak_period_metrics = peak_period_metrics + weather_check_results[5]    #Add peak_period metrics of forecasting model
                        #print(overall_metrics)
                        	
                    else:
                        DUR_JACK_FDN = 0 
                    
                    FDN_Installed = FDN_Installed+1                                     #Update number of foundations installed
                    FDN_Onboard = FDN_Onboard -1                                        #Update number of foundations still onboard
                    
                    #print('time after installing foundation',FDN_Installed,':',math.ceil(TOTAL_TIME_FDN))
                    
                    #-------- Step 10 --------
                    #Reposition vessel to start installation of next foundation
                
                    DUR_REPO_FDN = (INT_D/TRANS_VEL_FDN)                                #Determine duration of repositioning
                    TOTAL_TIME_FDN = TOTAL_TIME_FDN + DUR_REPO_FDN                      #Add time spent repositioning to total time installing foundations
                    
                    #print('time after repositioning', TOTAL_TIME_FDN)
                    
                #-------- Step 11 --------
                #Loading foundations from supply barge at sea    
                    
                else:
                    if stock_FDN > CAP_FDN:										       #Check if the stock on the supply barges is sufficent to load vessel  
                        FDN_Onboard = CAP_FDN						                   #Number of foundations on board is loaded untill vessel capacity is reached 
                    else:
                        FDN_Onboard = stock_FDN									       #If stock level is too low load remaining foundations 
                    stock_FDN = stock_FDN - FDN_Onboard	
                    
                    DUR_LOAD_FDN = FDN_Onboard*LT_FDN 							       #Determine time needed to load foundations
                    TOTAL_TIME_FDN = TOTAL_TIME_FDN + DUR_LOAD_FDN                     #Add time spent loading foundations to total time installing foundations

                    #print("Loading complete")
                    #print ("stockfdn",stock_FDN)
                    #print("fdn on deck",FDN_Onboard)
                    
                    #print("total time after loading",math.ceil(TOTAL_TIME_FDN))
          
            #-------- Step 12 --------
            #Travel back to port when finished installing all foundations       
                    
            DUR_TRANS_FDN = (PARK_DIS/TRANS_VEL_FDN)				                    #Determine time needed for travelling back to port
            #durationtrans = math.ceil(DUR_TRANS_FDN)
            
            weather_check_results = weather_check(TOTAL_TIME_FDN, start, 
                DUR_TRANS_FDN, df1, df2, s_wht_alpha_factors, wind_speed_alpha_factors, s_wht_limit=HS_TRANS_FDN, 
                wind_speed_limit=None, peak_period_limit=None)                         #Find suitable weather window to travel back to port
                    
            waiting = waiting + weather_check_results[1]                                #Add overall metrics of forecasting model 
            TOTAL_TIME_FDN = weather_check_results[0]                                   #Updated total time installing foundations
            waitingtransfdn = waitingtransfdn + weather_check_results[1]                #Add time waiting on weather to time waiting for transit foundations
                                            
            overall_metrics = overall_metrics + weather_check_results[2]                #Add overall metrics of forecasting model
            s_wht_metrics = s_wht_metrics + weather_check_results[3]                    #Add s_wht metrics of forecasting model
            wind_speed_metrics = wind_speed_metrics + weather_check_results[4]          #Add wind_speed metrics of forecasting model
            peak_period_metrics = peak_period_metrics + weather_check_results[5]        #Add peak_period metrics of forecasting model
            #print(overall_metrics)
            
        #print('return to port')          
        #print('TOTAL time installation foundations', math.ceil(TOTAL_TIME_FDN), 'hours')
        #print('Completed foundation installation')
        
        #-------- Step 13 --------
        #Save all evaluation metrics to dataframes
        if TOTAL_TIME_FDN < len(df1)-start:
            check_duration+=1
            
            T_FDN.loc[str(year),'Option '+str(option)] = math.ceil(TOTAL_TIME_FDN)              #Save total time installing foundations
            WAIT_FDN.loc[str(year),'Option '+str(option)] = math.ceil(waiting)                  #Save total time waiting during installation of foundations
            Waitingtransfdn.loc[str(year),'Option '+str(option)] = math.ceil(waitingtransfdn)   #Save time waiting to travel to/from the site during installation of foundations
            Waitingposfdn.loc[str(year),'Option '+str(option)] = math.ceil(waitingposfdn)       #Save time waiting to position vessel during installion of foundations
            Waitingjackfdn.loc[str(year),'Option '+str(option)] = math.ceil(waitingjackfdn)     #Save time waiting to jack-up/down vessel during installion of foundations
            Waitingmono.loc[str(year),'Option '+str(option)] = math.ceil(waitingmono)           #Save time waiting to lift and hammer monopile foundations during installation of foundations
            Waitingtp.loc[str(year),'Option '+str(option)] = math.ceil(waitingtp)               #Sace time waiting to lift and grout transition pieces during installation of foundations      

            TOTAL_COST_FDN = (TOTAL_TIME_FDN) *(CHT_C_FDN/24)    #Cost of installation of Foundations. Rental cost/hour 
            COST_FDN.loc[str(year),'Option '+str(option)] = math.ceil(TOTAL_COST_FDN)
                      
        #%% Additional parameter setup

        WTB_Installed = 0 
        WTB_Onboard = 0 
        waitingwtb = 0 
        start = start + 336  #turbines start operation two weeks after foundation installation 
        
        waitingtranswtb = 0 
        waitingposwtb = 0
        waitingjackwtb = 0 
        waitingtow = 0 
        waitingnac = 0 
        waitingblade = 0
        
        TOTAL_TIME_WTB = 0
        stock_WTB = N_WTB 
        
        #%% Operation steps installing turbines without supply barges 
        
        if SUPPLY_WTB == 0:                                                 #Check if foundation vessel uses supply barges
            
            while WTB_Installed<N_WTB:										#Loop to run until the desired number of turbines is installed 
            
                #-------- Step 1 --------
                #Loading turbines in port    
            
                if stock_WTB > CAP_WTB:										#Check if the stock at the port is sufficent to load vessel 
                    WTB_Onboard = CAP_WTB					                #Number of turbines on board is loaded untill vessel capacity is reached 
                
                else:
                    WTB_Onboard = stock_WTB									#If stock level is too low load remaining turbines
                
                stock_WTB = stock_WTB - WTB_Onboard							#New stocklevel after loading 
            
                DUR_LOAD_WTB = WTB_Onboard*LT_WTB 							#Determine time needed to load turbines 
                TOTAL_TIME_WTB = TOTAL_TIME_WTB + DUR_LOAD_WTB              #Add time spent loading foundations to total time installing turbines
           
                #-------- Step 2 --------
                #Travel to site for installation of turbines
                
                DUR_TRANS_WTB = (PARK_DIS/TRANS_VEL_WTB)                                #Determine time needed for travelling to site
                #durationtranswtb = math.ceil(DUR_TRANS_WTB)                         
                
                weather_check_results = weather_check(TOTAL_TIME_WTB, start, 
                    DUR_TRANS_WTB, df1, df2, s_wht_alpha_factors, wind_speed_alpha_factors, s_wht_limit=HS_TRANS_WTB, 
                    wind_speed_limit=None, peak_period_limit=None)                      #Find suitable weather window for travelling to site
                    
                waitingwtb = waitingwtb + weather_check_results[1]                      #Add time waiting on weather to total time waiting during turbine installation
                TOTAL_TIME_WTB = weather_check_results[0]                               #Updated total time installing turbines
                waitingtranswtb = waitingtranswtb + weather_check_results[1]            #Add time waiting on weather to time waiting for transit turbines
                                            
                overall_metrics = overall_metrics + weather_check_results[2]            #Add overall metrics of forecasting model
                s_wht_metrics = s_wht_metrics + weather_check_results[3]                #Add s_wht metrics of forecasting model
                wind_speed_metrics = wind_speed_metrics + weather_check_results[4]      #Add wind_speed metrics of forecasting model
                peak_period_metrics = peak_period_metrics + weather_check_results[5]    #Add peak_period metrics of forecasting model
                #print(overall_metrics)
                
            
                while WTB_Onboard>0:                                                    #Loop to run untill all turbines on board are installed 
        						
                    #-------- Step 3--------
                    #Position vessel for installation of turbines
                    					
                    DUR_POS_WTB = POS_WTB									            #Determine the time needed to position the vessel for installation of turbine
                    #durationposwtb = math.ceil(DUR_POS_WTB)
                
                    weather_check_results = weather_check(TOTAL_TIME_WTB, start, 
                        DUR_POS_WTB, df1, df2, s_wht_alpha_factors, wind_speed_alpha_factors, s_wht_limit=HS_POS_WTB, 
                        wind_speed_limit=None, peak_period_limit=None)                  #Find suitable weather window for positioning of the vessel for installation of turbine
                    
                    waitingwtb = waitingwtb + weather_check_results[1]                  #Add time waiting on weather to total time waiting during turbine installation
                    TOTAL_TIME_WTB = weather_check_results[0]                           #Updated total time installing turbines
                    waitingposwtb = waitingposwtb + weather_check_results[1]            #Add time waiting on weather to time waiting for positioning the vessel during turbine installation
                                                
                    overall_metrics = overall_metrics + weather_check_results[2]            #Add overall metrics of forecasting model
                    s_wht_metrics = s_wht_metrics + weather_check_results[3]                #Add s_wht metrics of forecasting model
                    wind_speed_metrics = wind_speed_metrics + weather_check_results[4]      #Add wind_speed metrics of forecasting model
                    peak_period_metrics = peak_period_metrics + weather_check_results[5]    #Add peak_period metrics of forecasting model
                    #print(overall_metrics)

                    #print('time after positioning', TOTAL_TIME_WTB)
                    
                    #-------- Step 4 --------
                    #Jack-up vessel for installation of turbine
                        
                    if JACKING_WTB == 1:									            #Check if turbine vessel is a jack-up vessel
                    
                        DUR_JACK_WTB =(PEN_LEG_WTB + AIR_WTB + WD) /JACK_SP_WTB         #Determine time needed to jack-up the vessel for installation of turbines
                        #durationjackwtb = math.ceil(DUR_JACK_WTB)
                        
                        weather_check_results = weather_check(TOTAL_TIME_WTB, start, 
                            DUR_JACK_WTB, df1, df2, s_wht_alpha_factors, wind_speed_alpha_factors, s_wht_limit=HS_JACK, 
                            wind_speed_limit=None, peak_period_limit=None)                      #Find suitable weather window for jack-up of the vessel for installation of turbine
                    
                        waitingwtb = waitingwtb + weather_check_results[1]                      #Add time waiting on weather to total time waiting during turbine installation
                        TOTAL_TIME_WTB = weather_check_results[0]                               #Updated total time installing turbines
                        waitingjackwtb = waitingjackwtb + weather_check_results[1]              #Add time waiting on weather to time waiting for jacking-up/down the vessel during turbine installation
                                                    
                        overall_metrics = overall_metrics + weather_check_results[2]            #Add overall metrics of forecasting model
                        s_wht_metrics = s_wht_metrics + weather_check_results[3]                #Add s_wht metrics of forecasting model
                        wind_speed_metrics = wind_speed_metrics + weather_check_results[4]      #Add wind_speed metrics of forecasting model
                        peak_period_metrics = peak_period_metrics + weather_check_results[5]    #Add peak_period metrics of forecasting model
                        #print(overall_metrics)
                         		
                    else:
                        DUR_JACK_WTB = 0
                
                    #-------- Step 5 --------
                    #Lift and position towerpieces
                
                    DUR_HOIST_TOW = N_TOW*LFT_TOW 							            #Determine time needed to lift and position towerpieces on foundation
                    #durationtow= math.ceil(DUR_HOIST_TOW)
                
                    weather_check_results = weather_check(TOTAL_TIME_WTB, start, 
                        DUR_HOIST_TOW, df1, df2, s_wht_alpha_factors, wind_speed_alpha_factors, s_wht_limit=HS_TOW, 
                        wind_speed_limit=WS_TOW, peak_period_limit=TP_TOW)                 #Find suitable weather window for lifting and positioning of towerpieces
                    
                    waitingwtb = waitingwtb + weather_check_results[1]                      #Add time waiting on weather to total time waiting during turbine installation
                    TOTAL_TIME_WTB = weather_check_results[0]                               #Updated total time installing turbines
                    waitingtow = waitingtow + weather_check_results[1]                      #Add time waiting on weather to time waiting for lifting and positioning of towerpieces
                                                    
                    overall_metrics = overall_metrics + weather_check_results[2]            #Add overall metrics of forecasting model
                    s_wht_metrics = s_wht_metrics + weather_check_results[3]                #Add s_wht metrics of forecasting model
                    wind_speed_metrics = wind_speed_metrics + weather_check_results[4]      #Add wind_speed metrics of forecasting model
                    peak_period_metrics = peak_period_metrics + weather_check_results[5]    #Add peak_period metrics of forecasting model
                    #print(overall_metrics)
                    
                    #print('time after lift tower piece',WTB_Installed+1,':',TOTAL_TIME_WTB)
                    
                    #-------- Step 6 --------
                    #Lift and install nacelle
                
                    DUR_HOIST_NAC = LFT_NAC						                        #Determine time needed to lift and position nacelle on tower 
                    #durationnac= math.ceil(DUR_HOIST_NAC)
                    
                    weather_check_results = weather_check(TOTAL_TIME_WTB, start, 
                        DUR_HOIST_NAC, df1, df2, s_wht_alpha_factors, wind_speed_alpha_factors, s_wht_limit=HS_NAC, 
                        wind_speed_limit=WS_NAC, peak_period_limit=TP_NAC)                 #Find suitable weather window for lifting and installing of nacelle
                    
                    waitingwtb = waitingwtb + weather_check_results[1]                      #Add time waiting on weather to total time waiting during turbine installation
                    TOTAL_TIME_WTB = weather_check_results[0]                               #Updated total time installing turbines
                    waitingnac = waitingnac + weather_check_results[1]                      #Add time waiting on weather to time waiting for lifting and installing of nacelle
                                                    
                    overall_metrics = overall_metrics + weather_check_results[2]            #Add overall metrics of forecasting model
                    s_wht_metrics = s_wht_metrics + weather_check_results[3]                #Add s_wht metrics of forecasting model
                    wind_speed_metrics = wind_speed_metrics + weather_check_results[4]      #Add wind_speed metrics of forecasting model
                    peak_period_metrics = peak_period_metrics + weather_check_results[5]    #Add peak_period metrics of forecasting model
                    #print(overall_metrics)
                    
                    #print('time after lift nacelle',WTB_Installed+1,':',TOTAL_TIME_WTB)
                
                    #-------- Step 7 and 8 --------
                    #Lift and install blades
                
                    #for i in range(0,N_BLADE):
                    DUR_INST_BLADE= N_BLADE*(LFT_BLADE + INST_BLADE)            #Determine time needed to lift and install blades
                        #durationblade= math.ceil(DUR_INST_BLADE)
                            
                    weather_check_results = weather_check(TOTAL_TIME_WTB, start, 
                                DUR_INST_BLADE, df1, df2, s_wht_alpha_factors, wind_speed_alpha_factors, s_wht_limit=HS_BLADE, 
                                wind_speed_limit=WS_BLADE, peak_period_limit=TP_BLADE)                 #Find suitable weather window for lifting and installing blades
                        
                        #print('TOTAL_TIME_WTB before:'+str(TOTAL_TIME_WTB))
                        #print('waiting blade before:'+str(waitingblade))
                        #print('waiting wtb before:'+str(waitingwtb))
                        #print('operation duration:' +str(DUR_INST_BLADE))
                        
                    waitingwtb = waitingwtb + weather_check_results[1]                          #Add time waiting on weather to total time waiting during turbine installation
                    TOTAL_TIME_WTB = weather_check_results[0]                                   #Updated total time installing turbines
                    waitingblade = waitingblade + weather_check_results[1]                      #Add time waiting on weather to time waiting for lifting and installing of blades
                        
                        #print('waiting blade after:'+str(waitingblade))
                        #print('TOTAL_TIME_WTB after:'+str(TOTAL_TIME_WTB))
                        #print('waiting wtb after:'+str(waitingwtb))
                                  
                    overall_metrics = overall_metrics + weather_check_results[2]                #Add overall metrics of forecasting model
                    s_wht_metrics = s_wht_metrics + weather_check_results[3]                    #Add s_wht metrics of forecasting model
                    wind_speed_metrics = wind_speed_metrics + weather_check_results[4]          #Add wind_speed metrics of forecasting model
                    peak_period_metrics = peak_period_metrics + weather_check_results[5]        #Add peak_period metrics of forecasting model
                        #print(overall_metrics)
                        
                    #print('time after installing blades',WTB_Installed+1, ':', TOTAL_TIME_WTB)
                    
                    #-------- Step 9 --------
                    #Jack-down vessel after installation of turbine
                        
                    if JACKING_WTB == 1:											    #Check if turbine vessel is a jack-up vessel
                        
                        DUR_JACK_WTB =(PEN_LEG_WTB + AIR_WTB + WD) /JACK_SP_WTB         #Determine time needed to jack-down the vessel
                        #durationjackwtb = math.ceil(DUR_JACK_WTB)
                
                        weather_check_results = weather_check(TOTAL_TIME_WTB, start, 
                            DUR_JACK_WTB, df1, df2, s_wht_alpha_factors, wind_speed_alpha_factors, s_wht_limit=HS_JACK, 
                            wind_speed_limit=None, peak_period_limit=None)                     #Find suitable weather window for jack-down of vessel
                    
                        waitingwtb = waitingwtb + weather_check_results[1]                      #Add time waiting on weather to total time waiting during turbine installation
                        TOTAL_TIME_WTB = weather_check_results[0]                               #Updated total time installing turbines
                        waitingjackwtb = waitingjackwtb + weather_check_results[1]              #Add time waiting on weather to time waiting for jacking-up/down the vessel during turbine installation
                                                        
                        overall_metrics = overall_metrics + weather_check_results[2]            #Add overall metrics of forecasting model
                        s_wht_metrics = s_wht_metrics + weather_check_results[3]                #Add s_wht metrics of forecasting model
                        wind_speed_metrics = wind_speed_metrics + weather_check_results[4]      #Add wind_speed metrics of forecasting model
                        peak_period_metrics = peak_period_metrics + weather_check_results[5]    #Add peak_period metrics of forecasting model
                        #print(overall_metrics)
                                
                    WTB_Installed = WTB_Installed+1                                     #Update number of turbines installed
                    WTB_Onboard = WTB_Onboard -1                                        #Update number of turbines still onboard
                    
                    #print('time after installing turbine',WTB_Installed,':',TOTAL_TIME_WTB)
            
                    #-------- Step 10 --------
                    #Reposition vessel to start installation of next turbine
            
                    DUR_REPO_WTB = INT_D/TRANS_VEL_WTB                                  #Determine duration of repositioning
                    TOTAL_TIME_WTB = TOTAL_TIME_WTB +DUR_REPO_WTB                       #Add time spent repositioning to total time installing turbines
                    
                    #print('time after repositioning', TOTAL_TIME_WTB)
                
                #-------- Step 11 --------
                #Travel back to port to load more turbines or when finished installing all turbines
                
                else:
                    DUR_TRANS_WTB = PARK_DIS/TRANS_VEL_WTB                              #Determine time needed for travelling back to port
				    #durationtranswtb = math.ceil(DUR_TRANS_WTB)  

                    weather_check_results = weather_check(TOTAL_TIME_WTB, start, 
                        DUR_TRANS_WTB, df1, df2, s_wht_alpha_factors, wind_speed_alpha_factors, s_wht_limit=HS_TRANS_WTB, 
                        wind_speed_limit=None, peak_period_limit=None)                     #Find suitable weather window for travelling back to port
                    
                    waitingwtb = waitingwtb + weather_check_results[1]                      #Add time waiting on weather to total time waiting during turbine installation
                    TOTAL_TIME_WTB = weather_check_results[0]                               #Updated total time installing turbines
                    waitingtranswtb = waitingtranswtb + weather_check_results[1]            #Add time waiting on weather to time waiting for transit turbines
                                                        
                    overall_metrics = overall_metrics + weather_check_results[2]            #Add overall metrics of forecasting model
                    s_wht_metrics = s_wht_metrics + weather_check_results[3]                #Add s_wht metrics of forecasting model
                    wind_speed_metrics = wind_speed_metrics + weather_check_results[4]      #Add wind_speed metrics of forecasting model
                    peak_period_metrics = peak_period_metrics + weather_check_results[5]    #Add peak_period metrics of forecasting model
                    #print(overall_metrics)
                            
                    #print('return to port')
        
        #%% Operation steps installing foundations with supply barges 
        
        elif SUPPLY_WTB == 1:                                                #Check if foundation vessel uses supply barges
            
            TOTAL_TIME_WTB = 0 
            #print('left port')
            
            #-------- Step 2 --------
            #Travel to site for installation of turbines
            
            DUR_TRANS_WTB = (PARK_DIS/TRANS_VEL_WTB)			                        #Determine time needed for travelling to site		
            #durationtranswtb = math.ceil(DUR_TRANS_WTB)

            weather_check_results = weather_check(TOTAL_TIME_WTB, start,
                DUR_TRANS_WTB, df1, df2, s_wht_alpha_factors, wind_speed_alpha_factors, s_wht_limit=HS_TRANS_WTB, 
                wind_speed_limit=None, peak_period_limit=None)                         #Find suitable weather window for travelling to site
                    
            waitingwtb = waitingwtb + weather_check_results[1]                          #Add time waiting on weather to total time waiting during turbine installation
            TOTAL_TIME_WTB = weather_check_results[0]                                   #Updated total time installing turbines
            waitingtranswtb = waitingtranswtb + weather_check_results[1]                #Add time waiting on weather to time waiting for transit turbines
                                                        
            overall_metrics = overall_metrics + weather_check_results[2]                #Add overall metrics of forecasting model
            s_wht_metrics = s_wht_metrics + weather_check_results[3]                    #Add s_wht metrics of forecasting model
            wind_speed_metrics = wind_speed_metrics + weather_check_results[4]          #Add wind_speed metrics of forecasting model
            peak_period_metrics = peak_period_metrics + weather_check_results[5]        #Add peak_period metrics of forecasting model
            #print(overall_metrics)                         
          
            #print("total time after transit",TOTAL_TIME_WTB)
          
            while WTB_Installed<N_WTB:                                      #Loop to run until the desired number of turbines is installed 
              
            
                
                while WTB_Onboard>0:                                        #Loop to run untill all turbines on board are installed 
        																
                    #-------- Step 3--------
                    #Position vessel for installation of turbines
                    
                    DUR_POS_WTB = POS_WTB									            #Determine the time needed to position the vessel for installation of turbine
                
                    if JACKING_WTB == 1:											    #Check if turbine vessel is a jack-up vessel
                        DUR_JACK_WTB =(PEN_LEG_WTB + AIR_WTB + WD) /JACK_SP_WTB         #Determine time needed to jack-up the vessel for installation of turbines
	
                    else:
                        DUR_JACK_WTB = 0
                    
                    DUR_POS_JACK = DUR_POS_WTB + DUR_JACK_WTB
                    #durationposwtb = math.ceil(DUR_POS_WTB+DUR_JACK_WTB)
                    
                    weather_check_results = weather_check(TOTAL_TIME_WTB, start, 
                        DUR_POS_JACK, df1, df2, s_wht_alpha_factors, wind_speed_alpha_factors, s_wht_limit=HS_POS_WTB, 
                        wind_speed_limit=None, peak_period_limit=None)                 #Find suitable weather window for jack-up of the vessel for installation of turbines
                    
                    waitingwtb = waitingwtb + weather_check_results[1]                  #Add time waiting on weather to total time waiting during turbine installation
                    TOTAL_TIME_WTB = weather_check_results[0]                           #Updated total time installing turbines
                    waitingposwtb = waitingposwtb + weather_check_results[1]            #Add time waiting on weather to time waiting for jacking-up/down the vessel during turbine installation
                                                        
                    overall_metrics = overall_metrics + weather_check_results[2]            #Add overall metrics of forecasting model
                    s_wht_metrics = s_wht_metrics + weather_check_results[3]                #Add s_wht metrics of forecasting model
                    wind_speed_metrics = wind_speed_metrics + weather_check_results[4]      #Add wind_speed metrics of forecasting model
                    peak_period_metrics = peak_period_metrics + weather_check_results[5]    #Add peak_period metrics of forecasting model
                    #print(overall_metrics) 
                    
                    #print('time after positioning and jacking-up:', TOTAL_TIME_WTB)
                    
                    #-------- Step 5 --------
                    #Lift and position towerpieces
                            
                    DUR_HOIST_TOW = N_TOW*LFT_TOW 							            #Determine time needed to lift and position towerpieces on foundation
                    #durationtow= math.ceil(DUR_HOIST_TOW)
                    
                    weather_check_results = weather_check(TOTAL_TIME_WTB, start, 
                        DUR_HOIST_TOW, df1, df2, s_wht_alpha_factors, wind_speed_alpha_factors, s_wht_limit=HS_TOW, 
                        wind_speed_limit=WS_TOW, peak_period_limit=TP_TOW)                 #Find suitable weather window for lifting and positioning of towerpieces
                    
                    waitingwtb = waitingwtb + weather_check_results[1]                      #Add time waiting on weather to total time waiting during turbine installation
                    TOTAL_TIME_WTB = weather_check_results[0]                               #Updated total time installing turbines
                    waitingtow = waitingtow + weather_check_results[1]                      #Add time waiting on weather to time waiting for lifting and positioning of towerpieces
                                                        
                    overall_metrics = overall_metrics + weather_check_results[2]            #Add overall metrics of forecasting model
                    s_wht_metrics = s_wht_metrics + weather_check_results[3]                #Add s_wht metrics of forecasting model
                    wind_speed_metrics = wind_speed_metrics + weather_check_results[4]      #Add wind_speed metrics of forecasting model
                    peak_period_metrics = peak_period_metrics + weather_check_results[5]    #Add peak_period metrics of forecasting model
                    #print(overall_metrics) 
                    
                    #print('time after lift tower piece',WTB_Installed+1,':',TOTAL_TIME_WTB)
                
                    #-------- Step 6 --------
                    #Lift and install nacelle
                
                    DUR_HOIST_NAC = LFT_NAC	                                            #Determine time needed to lift and position nacelle on tower 
                    #durationnac= math.ceil(DUR_HOIST_NAC)
                    
                    weather_check_results = weather_check(TOTAL_TIME_WTB, start, 
                        DUR_HOIST_NAC, df1, df2, s_wht_alpha_factors, wind_speed_alpha_factors, s_wht_limit=HS_NAC, 
                        wind_speed_limit=WS_NAC, peak_period_limit=TP_NAC)                 #Find suitable weather window for lifting and installing of nacelle
                    
                    waitingwtb = waitingwtb + weather_check_results[1]                      #Add time waiting on weather to total time waiting during turbine installation
                    TOTAL_TIME_WTB = weather_check_results[0]                               #Updated total time installing turbines
                    waitingnac = waitingnac + weather_check_results[1]                      #Add time waiting on weather to time waiting for lifting and installing of nacelle
                                                        
                    overall_metrics = overall_metrics + weather_check_results[2]            #Add overall metrics of forecasting model
                    s_wht_metrics = s_wht_metrics + weather_check_results[3]                #Add s_wht metrics of forecasting model
                    wind_speed_metrics = wind_speed_metrics + weather_check_results[4]      #Add wind_speed metrics of forecasting model
                    peak_period_metrics = peak_period_metrics + weather_check_results[5]    #Add peak_period metrics of forecasting model
                    #print(overall_metrics) 
                    
                    #print('time after lift nacelle',WTB_Installed+1,':',TOTAL_TIME_WTB)
                    
                    #-------- Step 7 and 8 --------
                    #Lift and install blades
                    
                    #for i in range(0,N_BLADE):
                    DUR_INST_BLADE = N_BLADE*(LFT_BLADE + INST_BLADE)                  #Determine time needed to lift and install blades
                        #durationblade= math.ceil(DUR_INST_BLADE)
                            
                    weather_check_results = weather_check(TOTAL_TIME_WTB, start, 
                                DUR_INST_BLADE, df1, df2, s_wht_alpha_factors, wind_speed_alpha_factors, s_wht_limit=HS_BLADE, 
                                wind_speed_limit=WS_BLADE, peak_period_limit=TP_BLADE)                  #Find suitable weather window for lifting and installing blades
                            
                        #print('TOTAL_TIME_WTB before:'+str(TOTAL_TIME_WTB))
                        #print('waiting blade before:'+str(waitingblade))
                        #print('waiting wtb before:'+str(waitingwtb))
                        #print('operation duration:' +str(DUR_INST_BLADE))
                        
                    waitingwtb = waitingwtb + weather_check_results[1]                          #Add time waiting on weather to total time waiting during turbine installation
                    TOTAL_TIME_WTB = weather_check_results[0]                                   #Updated total time installing turbines
                    waitingblade = waitingblade + weather_check_results[1]                      #Add time waiting on weather to time waiting for lifting and installing of blades
                        
                        #print('waiting blade after:'+str(waitingblade))
                        #print('TOTAL_TIME_WTB after:'+str(TOTAL_TIME_WTB))
                        #print('waiting wtb after:'+str(waitingwtb))
                                        
                    overall_metrics = overall_metrics + weather_check_results[2]                #Add overall metrics of forecasting model
                    s_wht_metrics = s_wht_metrics + weather_check_results[3]                    #Add s_wht metrics of forecasting model
                    wind_speed_metrics = wind_speed_metrics + weather_check_results[4]          #Add wind_speed metrics of forecasting model
                    peak_period_metrics = peak_period_metrics + weather_check_results[5]        #Add peak_period metrics of forecasting model
                        #print(overall_metrics) 
                    
                    #print('time after installing blades',WTB_Installed+1, ':', TOTAL_TIME_WTB)
                    
                    #-------- Step 9 --------
                    #Jack-down vessel after installation of turbine
                    
                    if JACKING_WTB == 1:                                                #Check if turbine vessel is a jack-up vessel
                        
                        DUR_JACK_WTB =(PEN_LEG_WTB + AIR_WTB + WD) /JACK_SP_WTB         #Determine time needed to jack-down the vessel
                        #durationjackwtb = math.ceil(DUR_JACK_WTB)
                        
                        weather_check_results = weather_check(TOTAL_TIME_WTB, start, 
                            DUR_JACK_WTB, df1, df2, s_wht_alpha_factors, wind_speed_alpha_factors, s_wht_limit=HS_POS_WTB, 
                            wind_speed_limit=None, peak_period_limit=None)                     #Find suitable weather window for jack-down of vessel
                    
                        waitingwtb = waitingwtb + weather_check_results[1]                      #Add time waiting on weather to total time waiting during turbine installation
                        TOTAL_TIME_WTB = weather_check_results[0]                               #Updated total time installing turbines
                        waitingjackwtb = waitingjackwtb + weather_check_results[1]              #Add time waiting on weather to time waiting for jacking-up/down the vessel during turbine installation
                                                        
                        overall_metrics = overall_metrics + weather_check_results[2]            #Add overall metrics of forecasting model
                        s_wht_metrics = s_wht_metrics + weather_check_results[3]                #Add s_wht metrics of forecasting model
                        wind_speed_metrics = wind_speed_metrics + weather_check_results[4]      #Add wind_speed metrics of forecasting model
                        peak_period_metrics = peak_period_metrics + weather_check_results[5]    #Add peak_period metrics of forecasting model
                        #print(overall_metrics) 
                    	
                    else:
                        DUR_JACK_WTB = 0 

                    WTB_Installed = WTB_Installed+1                                     #Update number of turbines installed
                    WTB_Onboard = WTB_Onboard -1                                        #Update number of turbines still onboard
                    
                    #print('time after installing turbine',WTB_Installed,':',math.ceil(TOTAL_TIME_WTB))
                    
                    #-------- Step 10 --------
                    #Reposition vessel to start installation of next turbine
                
                    DUR_REPO_WTB = INT_D/TRANS_VEL_WTB                                  #Determine duration of repositioning
                    TOTAL_TIME_WTB = TOTAL_TIME_WTB +DUR_REPO_WTB                       #Add time spent repositioning to total time installing turbines
                    
                    #print('time after repositioning', TOTAL_TIME_WTB)
                    
                #-------- Step 11 --------
                #Loading turbines from supply barge at sea
                
                else:
                										
                    if stock_WTB > CAP_WTB:                                             #Check if the stock on the supply barges is sufficent to load vessel  
                        WTB_Onboard = CAP_WTB						                    #Number of turbines on board is loaded untill vessel capacity is reached 
                    
                    else:
                        WTB_Onboard = stock_WTB									        #If stock level is too low load remaining turbines
                    stock_WTB = stock_WTB - WTB_Onboard
                
                    DUR_LOAD_WTB = WTB_Onboard*LT_WTB 							        #Determine time needed to load turbines
                    TOTAL_TIME_WTB = TOTAL_TIME_WTB + DUR_LOAD_WTB                      #Add time spent loading turbines to total time installing turbines
                    
                    #print("Loading complete")
                    #print ("stock turbines",stock_WTB)
                    #print("turbine on deck after loading",WTB_Onboard)
                    
                    #print("total time after loading",math.ceil(TOTAL_TIME_WTB))
            
            #-------- Step 12 --------
            #Travel back to port when finished installing all turbines
        
            DUR_TRANS_WTB = PARK_DIS/TRANS_VEL_WTB                                      #Determine time needed for travelling back to port		
            #durationtranswtb = math.ceil(DUR_TRANS_WTB)

            weather_check_results = weather_check(TOTAL_TIME_WTB, start, 
                DUR_TRANS_WTB, df1, df2, s_wht_alpha_factors, wind_speed_alpha_factors, s_wht_limit=HS_TRANS_WTB, 
                wind_speed_limit=None, peak_period_limit=None)                         #Find suitable weather window to travel back to port
                    
            waitingwtb = waitingwtb + weather_check_results[1]                          #Add time waiting on weather to total time waiting during turbine installation
            TOTAL_TIME_WTB = weather_check_results[0]                                   #Updated total time installing turbines
            waitingtranswtb = waitingtranswtb + weather_check_results[1]                #Add time waiting on weather to time waiting for transit turbines
                                                        
            overall_metrics = overall_metrics + weather_check_results[2]                #Add overall metrics of forecasting model
            s_wht_metrics = s_wht_metrics + weather_check_results[3]                    #Add s_wht metrics of forecasting model
            wind_speed_metrics = wind_speed_metrics + weather_check_results[4]          #Add wind_speed metrics of forecasting model
            peak_period_metrics = peak_period_metrics + weather_check_results[5]        #Add peak_period metrics of forecasting model
            #print(overall_metrics)                          
                  
        #print('return to port')
        #print('Turbine installation completed')   
        
        #-------- Step 13 --------
        #Save all evaluation metrics to dataframes
                 
        if TOTAL_TIME_WTB < len(df1)-start:
            check_duration+=1
            
            T_WTB.loc[str(year),'Option '+str(option)] = math.ceil(TOTAL_TIME_WTB)
            WAIT_WTB.loc[str(year),'Option '+str(option)] = math.ceil(waitingwtb)
            Waitingtranswtb.loc[str(year),'Option '+str(option)] = math.ceil(waitingtranswtb)
            Waitingposwtb.loc[str(year),'Option '+str(option)] = math.ceil(waitingposwtb)
            Waitingjackwtb.loc[str(year),'Option '+str(option)] = math.ceil(waitingjackwtb)
            Waitingtow.loc[str(year),'Option '+str(option)] = math.ceil(waitingtow)
            Waitingnac.loc[str(year),'Option '+str(option)] = math.ceil(waitingnac)
            Waitingblade.loc[str(year),'Option '+str(option)] = math.ceil(waitingblade)
            
            TOTAL_COST_WTB = (TOTAL_TIME_WTB) *(CHT_C_WTB/24)   #Cost of installation of Turbines, Rental cost/hour 
            COST_WTB.loc[str(year),'Option '+str(option)] = math.ceil(TOTAL_COST_WTB)
 
        #%% Results 
        #------ results-------
        Overall_metrics_TP.loc[str(year),'Option '+str(option)] = overall_metrics[0]
        Overall_metrics_FP.loc[str(year),'Option '+str(option)] = overall_metrics[1]
        Overall_metrics_TN.loc[str(year),'Option '+str(option)] = overall_metrics[2]
        Overall_metrics_FN.loc[str(year),'Option '+str(option)] = overall_metrics[3]
        Overall_accuracy.loc[str(year),'Option '+str(option)] = (overall_metrics[0] + overall_metrics[2])/(overall_metrics.sum())
        Overall_precision.loc[str(year),'Option '+str(option)] = overall_metrics[0]/(overall_metrics[0] + overall_metrics[1])
        Overall_recall.loc[str(year),'Option '+str(option)] = overall_metrics[0]/(overall_metrics[0] + overall_metrics[3])
                
        S_wht_metrics_TP.loc[str(year),'Option '+str(option)] = s_wht_metrics[0]
        S_wht_metrics_TN.loc[str(year),'Option '+str(option)] = s_wht_metrics[2]
        S_wht_metrics_FN.loc[str(year),'Option '+str(option)] = s_wht_metrics[3]
                
        Wind_speed_metrics_TP.loc[str(year),'Option '+str(option)] = wind_speed_metrics[0]
        Wind_speed_metrics_FP.loc[str(year),'Option '+str(option)] = wind_speed_metrics[1]
        Wind_speed_metrics_TN.loc[str(year),'Option '+str(option)] = wind_speed_metrics[2]
        Wind_speed_metrics_FN.loc[str(year),'Option '+str(option)] = wind_speed_metrics[3]
                
        peak_period_metrics_TP.loc[str(year),'Option '+str(option)] = peak_period_metrics[0]
        peak_period_metrics_FP.loc[str(year),'Option '+str(option)] = peak_period_metrics[1]
        peak_period_metrics_TN.loc[str(year),'Option '+str(option)] = peak_period_metrics[2]
        peak_period_metrics_FN.loc[str(year),'Option '+str(option)] = peak_period_metrics[3]
        
        if check_duration == 2:            
            TOTAL_TIME = TOTAL_TIME_FDN + TOTAL_TIME_WTB
            T_TOT.loc[str(year),'Option '+str(option)] = math.ceil(TOTAL_TIME)
            
            penalty_cost = FP_PEN * overall_metrics[1]
            PEN_COST.loc[str(year),'Option '+str(option)] = penalty_cost
            TOTAL_COST_INST = TOTAL_COST_FDN + TOTAL_COST_WTB + penalty_cost
            COST_TOT.loc[str(year),'Option '+str(option)] = math.ceil(TOTAL_COST_INST)
            
            TOTAL_WAIT = waiting + waitingwtb 
            WAIT_TOT.loc[str(year),'Option '+str(option)] = math.ceil(TOTAL_WAIT)
            
            print('Results option '+str(option)+', '+str(year))
            
            print('   ')
            print('Total duration of foundation installation', math.ceil(TOTAL_TIME_FDN), 'hours')  
            print('Total waiting time during foundation installation', math.ceil(waiting), 'hours') 
            print('Total cost of foundation installation',math.ceil(TOTAL_COST_FDN), 'kDollars')
            
            print('   ')
            print('Total duration of turbine installation', math.ceil(TOTAL_TIME_WTB), 'hours')           
            print('Total waiting time during turbine installation', math.ceil(waitingwtb), 'hours')
            print('Total cost of turbine installation', math.ceil(TOTAL_COST_WTB), 'kDollars')
            
            print('   ') 
            print('Total duration of installation farm',math.ceil(TOTAL_TIME), 'hours')    
            print('Total cost of installation farm', math.ceil(TOTAL_COST_INST), 'kDollars') 
            
            print('   ') 
            print('Completed farm installation')
            
            print('------------------------------------------------------')
            
        else:
            print('Results option '+str(option)+', '+str(year))
            
            print('   ')
            print('Installation not finished before the end of the year')
            
            print('------------------------------------------------------')

mean_overall_metrics_TP = Overall_metrics_TP.copy().mean()
mean_overall_metrics_FP = Overall_metrics_FP.copy().mean()
mean_overall_metrics_TN = Overall_metrics_TN.copy().mean()
mean_overall_metrics_FN = Overall_metrics_FN.copy().mean()
        
mean_overall_metrics.iloc[0]=mean_overall_metrics_TP
mean_overall_metrics.iloc[1]=mean_overall_metrics_FP
mean_overall_metrics.iloc[2]=mean_overall_metrics_TN
mean_overall_metrics.iloc[3]=mean_overall_metrics_FN

# Plot mean overall metrics as bar chart

x = np.arange(9)
width = 0.20
plt.bar(x-0.3, mean_overall_metrics_TP, width)
plt.bar(x-0.1, mean_overall_metrics_FP, width)
plt.bar(x+0.1, mean_overall_metrics_TN, width)
plt.bar(x+0.3, mean_overall_metrics_FN, width)

plt.xticks(x, ['1', '2', '3', '4', '5', '6', '7', '8', '9'])
plt.xlabel("Options")
#plt.ylabel("Quantity")
plt.legend(["True Positive", "False Postive", "True Negative", "False Negative"])
plt.show()

# Calculate and print RMSE values

RMSE_values = Determine_RMSE(Test_years)
error_directions = Count_error_direction(Test_years)

print('   ')
print('Forecasting model performance') 
print("RMSE wave height:", RMSE_values.iloc[0].item())
print("RMSE wind speed:", RMSE_values.iloc[1].item())
print("RMSE wave period:", RMSE_values.iloc[2].item())

# Write results to excel file for further processing
if save_results == True:
    with pd.ExcelWriter('Results '+str(prediction_horizon)+' hours.xlsx') as writer:
        RMSE_values.to_excel(writer, sheet_name='RMSE')
        error_directions.to_excel(writer, sheet_name='Error directions')
        
        T_TOT.to_excel(writer, sheet_name='Total duration')
        COST_TOT.to_excel(writer, sheet_name='Total cost')
        T_FDN.to_excel(writer, sheet_name='Duration foundations')
        T_WTB.to_excel(writer, sheet_name='Duration turbines')
        COST_FDN.to_excel(writer, sheet_name='Cost foundations')
        COST_WTB.to_excel(writer, sheet_name='Cost turbines')
        PEN_COST.to_excel(writer, sheet_name='Penalty cost')
        WAIT_TOT.to_excel(writer, sheet_name='Total waiting time')
        
        Waitingtransfdn.to_excel(writer, sheet_name='Waiting transit FDN')
        Waitingposfdn.to_excel(writer, sheet_name='Waiting positioning FDN')
        Waitingjackfdn.to_excel(writer, sheet_name='Waiting jacking FDN')
        Waitingmono.to_excel(writer, sheet_name='Waiting mono')
        Waitingtp.to_excel(writer, sheet_name='Waiting TP')
        
        Waitingtranswtb.to_excel(writer, sheet_name='Waiting transit WTB')
        Waitingposwtb.to_excel(writer, sheet_name='Waiting positioning WTB')
        Waitingjackwtb.to_excel(writer, sheet_name='Waiting jacking WTB')
        Waitingtow.to_excel(writer, sheet_name='Waiting tower')
        Waitingnac.to_excel(writer, sheet_name='Waiting nacelle')
        Waitingblade.to_excel(writer, sheet_name='Waiting blades')
        
        Overall_metrics_TP.to_excel(writer, sheet_name='TP overall metrics')
        Overall_metrics_FP.to_excel(writer, sheet_name='FP overall metrics')
        Overall_metrics_TN.to_excel(writer, sheet_name='TN overall metrics')
        Overall_metrics_FN.to_excel(writer, sheet_name='FN overall metrics')
        Overall_accuracy.to_excel(writer, sheet_name='Overall accuracy')
        Overall_precision.to_excel(writer, sheet_name='Overall precision')
        Overall_recall.to_excel(writer, sheet_name='Overall recall')
    
    
