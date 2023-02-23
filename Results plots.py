# -*- coding: utf-8 -*-
"""
@author: Coen Overvliet
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# read results Hugo Boer
loc3 = ("Results Hugo Boer.xlsx") 
results_calm_weather = pd.read_excel(loc3, sheet_name = 'Calm weather results', index_col=0)
total_duration_WR = pd.read_excel(loc3, sheet_name = 'Weather restricted duration', index_col=0)
total_cost_WR = pd.read_excel(loc3, sheet_name = 'Weather restricted cost', index_col=0)
foundations_duration_WR = pd.read_excel(loc3, sheet_name = 'WR duration foundations', index_col=0)
turbines_duration_WR = pd.read_excel(loc3, sheet_name = 'WR duration turbines', index_col=0)
foundations_cost_WR = pd.read_excel(loc3, sheet_name = 'WR cost foundations', index_col=0)
turbines_cost_WR = pd.read_excel(loc3, sheet_name = 'WR cost turbines', index_col=0)
total_waiting_time_WR = pd.read_excel(loc3, sheet_name = 'WR total waiting time', index_col=0)

waiting_time_transit_FDN_WR = pd.read_excel(loc3, sheet_name = 'WR waiting transit FDN', index_col=0)
waiting_time_positioning_FDN_WR = pd.read_excel(loc3, sheet_name = 'WR waiting positioning FDN', index_col=0)
waiting_time_jacking_FDN_WR = pd.read_excel(loc3, sheet_name = 'WR waiting jacking FDN', index_col=0)
waiting_time_mono_WR = pd.read_excel(loc3, sheet_name = 'WR waiting mono', index_col=0)
waiting_time_TP_WR = pd.read_excel(loc3, sheet_name = 'WR waiting TP', index_col=0)

waiting_time_transit_WTB_WR = pd.read_excel(loc3, sheet_name = 'WR waiting transit WTB', index_col=0)
waiting_time_positioning_WTB_WR = pd.read_excel(loc3, sheet_name = 'WR waiting positioning WTB', index_col=0)
waiting_time_jacking_WTB_WR = pd.read_excel(loc3, sheet_name = 'WR waiting jacking WTB', index_col=0)
waiting_time_tower_WR = pd.read_excel(loc3, sheet_name = 'WR waiting tower', index_col=0)
waiting_time_nacelle_WR = pd.read_excel(loc3, sheet_name = 'WR waiting nacelle', index_col=0)
waiting_time_blades_WR = pd.read_excel(loc3, sheet_name = 'WR waiting blades', index_col=0)

# read results 1 hour prediction horizon
loc4 = ("Results 1 hours.xlsx")
RMSE_values_1 = pd.read_excel(loc4, sheet_name = 'RMSE', index_col=0)
#error_directions_1 = pd.read_excel(loc4, sheet_name = 'Error directions', index_col=0)

total_duration_1 = pd.read_excel(loc4, sheet_name = 'Total duration', index_col=0)
total_cost_1 = pd.read_excel(loc4, sheet_name = 'Total cost', index_col=0)
foundations_duration_1 = pd.read_excel(loc4, sheet_name = 'Duration foundations', index_col=0)
turbines_duration_1 = pd.read_excel(loc4, sheet_name = 'Duration turbines', index_col=0)
foundations_cost_1 = pd.read_excel(loc4, sheet_name = 'Cost foundations', index_col=0)
turbines_cost_1 = pd.read_excel(loc4, sheet_name = 'Cost turbines', index_col=0)
total_waiting_time_1 = pd.read_excel(loc4, sheet_name = 'Total waiting time', index_col=0)
total_penalty_cost_1 = pd.read_excel(loc4, sheet_name = 'Penalty cost', index_col=0)

waiting_time_transit_FDN_1 = pd.read_excel(loc4, sheet_name = 'Waiting transit FDN', index_col=0)
waiting_time_positioning_FDN_1 = pd.read_excel(loc4, sheet_name = 'Waiting positioning FDN', index_col=0)
waiting_time_jacking_FDN_1 = pd.read_excel(loc4, sheet_name = 'Waiting jacking FDN', index_col=0)
waiting_time_mono_1 = pd.read_excel(loc4, sheet_name = 'Waiting mono', index_col=0)
waiting_time_TP_1 = pd.read_excel(loc4, sheet_name = 'Waiting TP', index_col=0)

waiting_time_transit_WTB_1 = pd.read_excel(loc4, sheet_name = 'Waiting transit WTB', index_col=0)
waiting_time_positioning_WTB_1 = pd.read_excel(loc4, sheet_name = 'Waiting positioning WTB', index_col=0)
waiting_time_jacking_WTB_1 = pd.read_excel(loc4, sheet_name = 'Waiting jacking WTB', index_col=0)
waiting_time_tower_1 = pd.read_excel(loc4, sheet_name = 'Waiting tower', index_col=0)
waiting_time_nacelle_1 = pd.read_excel(loc4, sheet_name = 'Waiting nacelle', index_col=0)
waiting_time_blades_1 = pd.read_excel(loc4, sheet_name = 'Waiting blades', index_col=0)

overall_metrics_TP_1 = pd.read_excel(loc4, sheet_name = 'TP overall metrics', index_col=0)
overall_metrics_FP_1 = pd.read_excel(loc4, sheet_name = 'FP overall metrics', index_col=0)
overall_metrics_TN_1 = pd.read_excel(loc4, sheet_name = 'TN overall metrics', index_col=0)
overall_metrics_FN_1 = pd.read_excel(loc4, sheet_name = 'FN overall metrics', index_col=0)

# read results 12 hour prediction horizon
loc5 = ("Results 12 hours.xlsx")
RMSE_values_12 = pd.read_excel(loc5, sheet_name = 'RMSE', index_col=0)
#error_directions_12 = pd.read_excel(loc5, sheet_name = 'Error directions', index_col=0)

total_duration_12 = pd.read_excel(loc5, sheet_name = 'Total duration', index_col=0)
total_cost_12 = pd.read_excel(loc5, sheet_name = 'Total cost', index_col=0)
foundations_duration_12 = pd.read_excel(loc5, sheet_name = 'Duration foundations', index_col=0)
turbines_duration_12 = pd.read_excel(loc5, sheet_name = 'Duration turbines', index_col=0)
foundations_cost_12 = pd.read_excel(loc5, sheet_name = 'Cost foundations', index_col=0)
turbines_cost_12 = pd.read_excel(loc5, sheet_name = 'Cost turbines', index_col=0)
total_waiting_time_12 = pd.read_excel(loc5, sheet_name = 'Total waiting time', index_col=0)
total_penalty_cost_12 = pd.read_excel(loc5, sheet_name = 'Penalty cost', index_col=0)

waiting_time_transit_FDN_12 = pd.read_excel(loc5, sheet_name = 'Waiting transit FDN', index_col=0)
waiting_time_positioning_FDN_12 = pd.read_excel(loc5, sheet_name = 'Waiting positioning FDN', index_col=0)
waiting_time_jacking_FDN_12 = pd.read_excel(loc5, sheet_name = 'Waiting jacking FDN', index_col=0)
waiting_time_mono_12 = pd.read_excel(loc5, sheet_name = 'Waiting mono', index_col=0)
waiting_time_TP_12 = pd.read_excel(loc5, sheet_name = 'Waiting TP', index_col=0)

waiting_time_transit_WTB_12 = pd.read_excel(loc5, sheet_name = 'Waiting transit WTB', index_col=0)
waiting_time_positioning_WTB_12 = pd.read_excel(loc5, sheet_name = 'Waiting positioning WTB', index_col=0)
waiting_time_jacking_WTB_12 = pd.read_excel(loc5, sheet_name = 'Waiting jacking WTB', index_col=0)
waiting_time_tower_12 = pd.read_excel(loc5, sheet_name = 'Waiting tower', index_col=0)
waiting_time_nacelle_12 = pd.read_excel(loc5, sheet_name = 'Waiting nacelle', index_col=0)
waiting_time_blades_12 = pd.read_excel(loc5, sheet_name = 'Waiting blades', index_col=0)

overall_metrics_TP_12 = pd.read_excel(loc5, sheet_name = 'TP overall metrics', index_col=0)
overall_metrics_FP_12 = pd.read_excel(loc5, sheet_name = 'FP overall metrics', index_col=0)
overall_metrics_TN_12 = pd.read_excel(loc5, sheet_name = 'TN overall metrics', index_col=0)
overall_metrics_FN_12 = pd.read_excel(loc5, sheet_name = 'FN overall metrics', index_col=0)

# read results 24 hour prediction horizon
loc6 = ("Results 24 hours.xlsx")
RMSE_values_24 = pd.read_excel(loc6, sheet_name = 'RMSE', index_col=0)
#error_directions_24 = pd.read_excel(loc6, sheet_name = 'Error directions', index_col=0)

total_duration_24 = pd.read_excel(loc6, sheet_name = 'Total duration', index_col=0)
total_cost_24 = pd.read_excel(loc6, sheet_name = 'Total cost', index_col=0)
foundations_duration_24 = pd.read_excel(loc6, sheet_name = 'Duration foundations', index_col=0)
turbines_duration_24 = pd.read_excel(loc6, sheet_name = 'Duration turbines', index_col=0)
foundations_cost_24 = pd.read_excel(loc6, sheet_name = 'Cost foundations', index_col=0)
turbines_cost_24 = pd.read_excel(loc6, sheet_name = 'Cost turbines', index_col=0)
total_waiting_time_24 = pd.read_excel(loc6, sheet_name = 'Total waiting time', index_col=0)
total_penalty_cost_24 = pd.read_excel(loc6, sheet_name = 'Penalty cost', index_col=0)

waiting_time_transit_FDN_24 = pd.read_excel(loc6, sheet_name = 'Waiting transit FDN', index_col=0)
waiting_time_positioning_FDN_24 = pd.read_excel(loc6, sheet_name = 'Waiting positioning FDN', index_col=0)
waiting_time_jacking_FDN_24 = pd.read_excel(loc6, sheet_name = 'Waiting jacking FDN', index_col=0)
waiting_time_mono_24 = pd.read_excel(loc6, sheet_name = 'Waiting mono', index_col=0)
waiting_time_TP_24 = pd.read_excel(loc6, sheet_name = 'Waiting TP', index_col=0)

waiting_time_transit_WTB_24 = pd.read_excel(loc6, sheet_name = 'Waiting transit WTB', index_col=0)
waiting_time_positioning_WTB_24 = pd.read_excel(loc6, sheet_name = 'Waiting positioning WTB', index_col=0)
waiting_time_jacking_WTB_24 = pd.read_excel(loc6, sheet_name = 'Waiting jacking WTB', index_col=0)
waiting_time_tower_24 = pd.read_excel(loc6, sheet_name = 'Waiting tower', index_col=0)
waiting_time_nacelle_24 = pd.read_excel(loc6, sheet_name = 'Waiting nacelle', index_col=0)
waiting_time_blades_24 = pd.read_excel(loc6, sheet_name = 'Waiting blades', index_col=0)

overall_metrics_TP_24 = pd.read_excel(loc6, sheet_name = 'TP overall metrics', index_col=0)
overall_metrics_FP_24 = pd.read_excel(loc6, sheet_name = 'FP overall metrics', index_col=0)
overall_metrics_TN_24 = pd.read_excel(loc6, sheet_name = 'TN overall metrics', index_col=0)
overall_metrics_FN_24 = pd.read_excel(loc6, sheet_name = 'FN overall metrics', index_col=0)

# read results 48 hour prediction horizon
loc7 = ("Results 48 hours.xlsx")
RMSE_values_48 = pd.read_excel(loc7, sheet_name = 'RMSE', index_col=0)
#error_directions_48 = pd.read_excel(loc7, sheet_name = 'Error directions', index_col=0)

total_duration_48 = pd.read_excel(loc7, sheet_name = 'Total duration', index_col=0)
total_cost_48 = pd.read_excel(loc7, sheet_name = 'Total cost', index_col=0)
foundations_duration_48 = pd.read_excel(loc7, sheet_name = 'Duration foundations', index_col=0)
turbines_duration_48 = pd.read_excel(loc7, sheet_name = 'Duration turbines', index_col=0)
foundations_cost_48 = pd.read_excel(loc7, sheet_name = 'Cost foundations', index_col=0)
turbines_cost_48 = pd.read_excel(loc7, sheet_name = 'Cost turbines', index_col=0)
total_waiting_time_48 = pd.read_excel(loc7, sheet_name = 'Total waiting time', index_col=0)
total_penalty_cost_48 = pd.read_excel(loc7, sheet_name = 'Penalty cost', index_col=0)

waiting_time_transit_FDN_48 = pd.read_excel(loc7, sheet_name = 'Waiting transit FDN', index_col=0)
waiting_time_positioning_FDN_48 = pd.read_excel(loc7, sheet_name = 'Waiting positioning FDN', index_col=0)
waiting_time_jacking_FDN_48 = pd.read_excel(loc7, sheet_name = 'Waiting jacking FDN', index_col=0)
waiting_time_mono_48 = pd.read_excel(loc7, sheet_name = 'Waiting mono', index_col=0)
waiting_time_TP_48 = pd.read_excel(loc7, sheet_name = 'Waiting TP', index_col=0)

waiting_time_transit_WTB_48 = pd.read_excel(loc7, sheet_name = 'Waiting transit WTB', index_col=0)
waiting_time_positioning_WTB_48 = pd.read_excel(loc7, sheet_name = 'Waiting positioning WTB', index_col=0)
waiting_time_jacking_WTB_48 = pd.read_excel(loc7, sheet_name = 'Waiting jacking WTB', index_col=0)
waiting_time_tower_48 = pd.read_excel(loc7, sheet_name = 'Waiting tower', index_col=0)
waiting_time_nacelle_48 = pd.read_excel(loc7, sheet_name = 'Waiting nacelle', index_col=0)
waiting_time_blades_48 = pd.read_excel(loc7, sheet_name = 'Waiting blades', index_col=0)

overall_metrics_TP_48 = pd.read_excel(loc7, sheet_name = 'TP overall metrics', index_col=0)
overall_metrics_FP_48 = pd.read_excel(loc7, sheet_name = 'FP overall metrics', index_col=0)
overall_metrics_TN_48 = pd.read_excel(loc7, sheet_name = 'TN overall metrics', index_col=0)
overall_metrics_FN_48 = pd.read_excel(loc7, sheet_name = 'FN overall metrics', index_col=0)

# read results 72 hour prediction horizon
loc8 = ("Results 72 hours.xlsx")
RMSE_values_72 = pd.read_excel(loc8, sheet_name = 'RMSE', index_col=0)
#error_directions_72 = pd.read_excel(loc8, sheet_name = 'Error directions', index_col=0)

total_duration_72 = pd.read_excel(loc8, sheet_name = 'Total duration', index_col=0)
total_cost_72 = pd.read_excel(loc8, sheet_name = 'Total cost', index_col=0)
foundations_duration_72 = pd.read_excel(loc8, sheet_name = 'Duration foundations', index_col=0)
turbines_duration_72 = pd.read_excel(loc8, sheet_name = 'Duration turbines', index_col=0)
foundations_cost_72 = pd.read_excel(loc8, sheet_name = 'Cost foundations', index_col=0)
turbines_cost_72 = pd.read_excel(loc8, sheet_name = 'Cost turbines', index_col=0)
total_waiting_time_72 = pd.read_excel(loc8, sheet_name = 'Total waiting time', index_col=0)
total_penalty_cost_72 = pd.read_excel(loc8, sheet_name = 'Penalty cost', index_col=0)

waiting_time_transit_FDN_72 = pd.read_excel(loc8, sheet_name = 'Waiting transit FDN', index_col=0)
waiting_time_positioning_FDN_72 = pd.read_excel(loc8, sheet_name = 'Waiting positioning FDN', index_col=0)
waiting_time_jacking_FDN_72 = pd.read_excel(loc8, sheet_name = 'Waiting jacking FDN', index_col=0)
waiting_time_mono_72 = pd.read_excel(loc8, sheet_name = 'Waiting mono', index_col=0)
waiting_time_TP_72 = pd.read_excel(loc8, sheet_name = 'Waiting TP', index_col=0)

waiting_time_transit_WTB_72 = pd.read_excel(loc8, sheet_name = 'Waiting transit WTB', index_col=0)
waiting_time_positioning_WTB_72 = pd.read_excel(loc8, sheet_name = 'Waiting positioning WTB', index_col=0)
waiting_time_jacking_WTB_72 = pd.read_excel(loc8, sheet_name = 'Waiting jacking WTB', index_col=0)
waiting_time_tower_72 = pd.read_excel(loc8, sheet_name = 'Waiting tower', index_col=0)
waiting_time_nacelle_72 = pd.read_excel(loc8, sheet_name = 'Waiting nacelle', index_col=0)
waiting_time_blades_72 = pd.read_excel(loc8, sheet_name = 'Waiting blades', index_col=0)

overall_metrics_TP_72 = pd.read_excel(loc8, sheet_name = 'TP overall metrics', index_col=0)
overall_metrics_FP_72 = pd.read_excel(loc8, sheet_name = 'FP overall metrics', index_col=0)
overall_metrics_TN_72 = pd.read_excel(loc8, sheet_name = 'TN overall metrics', index_col=0)
overall_metrics_FN_72 = pd.read_excel(loc8, sheet_name = 'FN overall metrics', index_col=0)

# read results 96 hour prediction horizon
loc9 = ("Results 96 hours.xlsx")
RMSE_values_96 = pd.read_excel(loc9, sheet_name = 'RMSE', index_col=0)
#error_directions_96 = pd.read_excel(loc9, sheet_name = 'Error directions', index_col=0)

total_duration_96 = pd.read_excel(loc9, sheet_name = 'Total duration', index_col=0)
total_cost_96 = pd.read_excel(loc9, sheet_name = 'Total cost', index_col=0)
foundations_duration_96 = pd.read_excel(loc9, sheet_name = 'Duration foundations', index_col=0)
turbines_duration_96 = pd.read_excel(loc9, sheet_name = 'Duration turbines', index_col=0)
foundations_cost_96 = pd.read_excel(loc9, sheet_name = 'Cost foundations', index_col=0)
turbines_cost_96 = pd.read_excel(loc9, sheet_name = 'Cost turbines', index_col=0)
total_waiting_time_96 = pd.read_excel(loc9, sheet_name = 'Total waiting time', index_col=0)
total_penalty_cost_96 = pd.read_excel(loc9, sheet_name = 'Penalty cost', index_col=0)

waiting_time_transit_FDN_96 = pd.read_excel(loc9, sheet_name = 'Waiting transit FDN', index_col=0)
waiting_time_positioning_FDN_96 = pd.read_excel(loc9, sheet_name = 'Waiting positioning FDN', index_col=0)
waiting_time_jacking_FDN_96 = pd.read_excel(loc9, sheet_name = 'Waiting jacking FDN', index_col=0)
waiting_time_mono_96 = pd.read_excel(loc9, sheet_name = 'Waiting mono', index_col=0)
waiting_time_TP_96 = pd.read_excel(loc9, sheet_name = 'Waiting TP', index_col=0)

waiting_time_transit_WTB_96 = pd.read_excel(loc9, sheet_name = 'Waiting transit WTB', index_col=0)
waiting_time_positioning_WTB_96 = pd.read_excel(loc9, sheet_name = 'Waiting positioning WTB', index_col=0)
waiting_time_jacking_WTB_96 = pd.read_excel(loc9, sheet_name = 'Waiting jacking WTB', index_col=0)
waiting_time_tower_96 = pd.read_excel(loc9, sheet_name = 'Waiting tower', index_col=0)
waiting_time_nacelle_96 = pd.read_excel(loc9, sheet_name = 'Waiting nacelle', index_col=0)
waiting_time_blades_96 = pd.read_excel(loc9, sheet_name = 'Waiting blades', index_col=0)

overall_metrics_TP_96 = pd.read_excel(loc9, sheet_name = 'TP overall metrics', index_col=0)
overall_metrics_FP_96 = pd.read_excel(loc9, sheet_name = 'FP overall metrics', index_col=0)
overall_metrics_TN_96 = pd.read_excel(loc9, sheet_name = 'TN overall metrics', index_col=0)
overall_metrics_FN_96 = pd.read_excel(loc9, sheet_name = 'FN overall metrics', index_col=0)

options = total_duration_1.columns

#%% plot RMSE values

RMSE_values=pd.concat((RMSE_values_1, RMSE_values_12, RMSE_values_24, RMSE_values_48, RMSE_values_72, RMSE_values_96), axis=1)
RMSE_values=RMSE_values.transpose()

prediction_horizons = pd.Series([1,12,24,48,72,96])

plt.figure(figsize=(15,10))

plt.plot(prediction_horizons, RMSE_values, linestyle='--', marker='x', markersize=15)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(visible=True)
plt.ylabel('RMSE', fontsize= 20)
plt.xlabel('Prediction horizon [h]', fontsize= 20)
plt.legend(['Significant wave height', 'Wind speed', 'Mean wave period'],fontsize= 'xx-large', frameon=True)#, bbox_to_anchor=(1.0, 1.0))
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.ylim(ymin=0)
plt.xlim((0,100))

plt.savefig('RMSE values vs prediction horizons.pdf',bbox_inches='tight')


#%% plot error directions

# error_directions_1.transpose().plot(kind= 'bar', rot=45, grid=True, figsize = (12,10), fontsize=20)

# plt.xticks(rotation=45, fontsize=20)
# plt.yticks(fontsize=20)
# plt.grid(visible=True)
# plt.ylabel('Total duration farm installation [h]', fontsize= 20)
# plt.legend(fontsize= 'xx-large', frameon=True)#, bbox_to_anchor=(1.0, 1.0))
# plt.grid(b=True, which='major', color='#666666', linestyle='-')
# plt.minorticks_on()
# plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
# #plt.ylim((0,4000))

#%% plot total duration results with calm weather and weather restricted results

# results_weather_restricted_2008 = total_duration_WR.iloc[0,:]
# results_weather_restricted_2009 = total_duration_WR.iloc[1,:]
# results_weather_restricted_2010 = total_duration_WR.iloc[2,:]

# results_weather_uncertainty_2008 = total_duration_72.iloc[0,:]
# results_weather_uncertainty_2009 = total_duration_72.iloc[1,:]
# results_weather_uncertainty_2010 = total_duration_72.iloc[2,:]

# y1 = [results_weather_restricted_2008, results_weather_restricted_2009, results_weather_restricted_2010]
# y2 = [results_weather_uncertainty_2008, results_weather_uncertainty_2009, results_weather_uncertainty_2010]
# colors = ['green', 'xkcd:gold', 'red']

# plt.figure(figsize=(15,10))

# for y1e, y2e, color in zip(y1, y2, colors):
#     plt.scatter(options, y1e, label=str(y1e.name)+' perfect forecast', color=color, s=100)
#     plt.scatter(options, y2e, label=str(y2e.name)+' with uncertainty', marker='x', color=color, s=200)
       
# plt.bar(range(0,len(results_calm_weather)),results_calm_weather.T_TOT,label='Calm weather Operations')

# plt.xticks(rotation=45, fontsize=20)
# plt.yticks(fontsize=20)
# plt.grid(visible=True)
# plt.ylabel('Total duration farm installation [h]', fontsize= 20)
# plt.legend(fontsize= 'xx-large', frameon=True)#, bbox_to_anchor=(1.0, 1.0))
# plt.grid(b=True, which='major', color='#666666', linestyle='-')
# plt.minorticks_on()
# plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
# plt.ylim((0,4000))

# plt.savefig('Total duration farm installation.pdf',bbox_inches='tight')
# #plt.title("Total duration installation offshore wind farm, calm weather comparison", fontsize = 28)

#%% plot mean total duration with calm weather and mean weather restricted results

for option in options:
    if total_duration_1[option].isna().sum() > 0:
        total_duration_1.loc[:, option] = np.nan
    if total_duration_12[option].isna().sum() > 0:
        total_duration_12.loc[:, option] = np.nan
    if total_duration_24[option].isna().sum() > 0:
        total_duration_24.loc[:, option] = np.nan
    if total_duration_48[option].isna().sum() > 0:
        total_duration_48.loc[:, option] = np.nan
    if total_duration_72[option].isna().sum() > 0:
        total_duration_72.loc[:, option] = np.nan
    if total_duration_96[option].isna().sum() > 0:
        total_duration_96.loc[:, option] = np.nan
        
mean_total_duration_WR = total_duration_WR.copy().mean()
mean_total_duration_1 = total_duration_1.copy().mean()
mean_total_duration_12 = total_duration_12.copy().mean()
mean_total_duration_24 = total_duration_24.copy().mean()
mean_total_duration_48 = total_duration_48.copy().mean()
mean_total_duration_72 = total_duration_72.copy().mean()        
mean_total_duration_96 = total_duration_96.copy().mean()

plt.figure(figsize=(15,10))
plt.scatter(options, mean_total_duration_WR, label='Perfect forecast', color='black', s=100)
plt.scatter(options, mean_total_duration_1, label='1 hour horizon', marker='x', color='green', s=200)
plt.scatter(options, mean_total_duration_12, label='12 hour horizon', marker='x', color='red', s=200)
plt.scatter(options, mean_total_duration_24, label='24 hour horizon', marker='x', color='xkcd:gold', s=200)
#plt.scatter(options, mean_total_duration_48, label='48 hour horizon', marker='x', color='xkcd:ocean blue', s=200)
#plt.scatter(options, mean_total_duration_72, label='72 hour horizon', marker='x', color='xkcd:grape', s=200)
#plt.scatter(options, mean_total_duration_96, label='96 hour horizon', marker='x', color='xkcd:pumpkin', s=200)

plt.bar(range(0,len(results_calm_weather)),results_calm_weather.T_TOT,label='Calm weather')

plt.xticks(rotation=45, fontsize=20)
plt.yticks(fontsize=20)
plt.grid(visible=True)
plt.ylabel('Mean total duration farm installation [h]', fontsize= 20)
plt.legend(fontsize= 'xx-large', frameon=True)#, bbox_to_anchor=(1.0, 1.0))
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
#plt.ylim((0,4000))

plt.savefig('Mean total duration farm installation.pdf',bbox_inches='tight')

#%% plot total cost installation offshore wind farm

# results_weather_restricted_2008 = total_cost_WR.iloc[0,:]
# results_weather_restricted_2009 = total_cost_WR.iloc[1,:]
# results_weather_restricted_2010 = total_cost_WR.iloc[2,:]

# results_weather_uncertainty_2008 = COST_TOT.iloc[0,:]
# results_weather_uncertainty_2009 = COST_TOT.iloc[1,:]
# results_weather_uncertainty_2010 = COST_TOT.iloc[2,:]

# y1 = [results_weather_restricted_2008, results_weather_restricted_2009, results_weather_restricted_2010]
# y2 = [results_weather_uncertainty_2008, results_weather_uncertainty_2009, results_weather_uncertainty_2010]
# colors = ['green', 'xkcd:gold', 'red']

# plt.figure(figsize=(15,10))

# for y1e, y2e, color in zip(y1, y2, colors):
#     plt.scatter(options, y1e, label=str(y1e.name)+' perfect forecast', color=color, s=100)
#     plt.scatter(options, y2e, label=str(y2e.name)+' with uncertainty', marker='x', color=color, s=200)
       
# plt.bar(range(0,len(results_calm_weather)),results_calm_weather.COST_TOT,label='Calm weather Operations')

# plt.xticks(rotation=45, fontsize=20)
# plt.yticks(fontsize=20)
# plt.grid(visible=True)
# plt.ylabel('Total cost farm installation [h]', fontsize= 20)
# plt.legend(fontsize= 'xx-large', frameon=True)#, bbox_to_anchor=(1.0, 1.0))
# plt.grid(b=True, which='major', color='#666666', linestyle='-')
# plt.minorticks_on()
# plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
# plt.ylim((0,50000))

# plt.savefig('Total cost farm installation.pdf',bbox_inches='tight')
# #plt.title("Total duration installation offshore wind farm, calm weather comparison", fontsize = 28)

#%% plot mean total cost installation offshore wind farm

for option in options:
    if total_cost_1[option].isna().sum() > 0:
        total_cost_1.loc[:, option] = np.nan
    if total_cost_12[option].isna().sum() > 0:
        total_cost_12.loc[:, option] = np.nan
    if total_cost_24[option].isna().sum() > 0:
        total_cost_24.loc[:, option] = np.nan
    if total_cost_48[option].isna().sum() > 0:
        total_cost_48.loc[:, option] = np.nan
    if total_cost_72[option].isna().sum() > 0:
        total_cost_72.loc[:, option] = np.nan
    if total_cost_96[option].isna().sum() > 0:
        total_cost_96.loc[:, option] = np.nan
        
mean_total_cost_WR = total_cost_WR.copy().mean()
mean_total_cost_1 = total_cost_1.copy().mean()
mean_total_cost_12 = total_cost_12.copy().mean()
mean_total_cost_24 = total_cost_24.copy().mean()
mean_total_cost_48 = total_cost_48.copy().mean()
mean_total_cost_72 = total_cost_72.copy().mean()        
mean_total_cost_96 = total_cost_96.copy().mean()

plt.figure(figsize=(15,10))
plt.scatter(options, mean_total_cost_WR, label='Perfect forecast', color='black', s=100)
plt.scatter(options, mean_total_cost_1, label='1 hour horizon', marker='x', color='green', s=200)
plt.scatter(options, mean_total_cost_12, label='12 hour horizon', marker='x', color='red', s=200)
plt.scatter(options, mean_total_cost_24, label='24 hour horizon', marker='x', color='xkcd:gold', s=200)
#plt.scatter(options, mean_total_cost_48, label='48 hour horizon', marker='x', color='xkcd:ocean blue', s=200)
#plt.scatter(options, mean_total_cost_72, label='72 hour horizon', marker='x', color='xkcd:grape', s=200)
#plt.scatter(options, mean_total_cost_96, label='96 hour horizon', marker='x', color='xkcd:pumpkin', s=200)

plt.bar(range(0,len(results_calm_weather)),results_calm_weather.COST_TOT,label='Calm weather')

plt.xticks(rotation=45, fontsize=20)
plt.yticks(fontsize=20)
plt.grid(visible=True)
plt.ylabel('Mean total cost farm installation [k€]', fontsize= 20)
plt.legend(fontsize= 'xx-large', frameon=True, bbox_to_anchor=(1.0, 1.0))
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
#plt.ylim((0,50000))

plt.savefig('Mean total cost farm installation.pdf',bbox_inches='tight')

#%% plot duration foundation installation

# results_weather_restricted_2008 = foundations_duration_WR.iloc[0,:]
# results_weather_restricted_2009 = foundations_duration_WR.iloc[1,:]
# results_weather_restricted_2010 = foundations_duration_WR.iloc[2,:]

# results_weather_uncertainty_2008 = T_FDN.iloc[0,:]
# results_weather_uncertainty_2009 = T_FDN.iloc[1,:]
# results_weather_uncertainty_2010 = T_FDN.iloc[2,:]

# y1 = [results_weather_restricted_2008, results_weather_restricted_2009, results_weather_restricted_2010]
# y2 = [results_weather_uncertainty_2008, results_weather_uncertainty_2009, results_weather_uncertainty_2010]
# colors = ['green', 'xkcd:gold', 'red']

# plt.figure(figsize=(15,10))

# for y1e, y2e, color in zip(y1, y2, colors):
#     plt.scatter(options, y1e, label=str(y1e.name)+' perfect forecast', color=color, s=100)
#     plt.scatter(options, y2e, label=str(y2e.name)+' with uncertainty', marker='x', color=color, s=200)
       
# plt.bar(range(0,len(results_calm_weather)),results_calm_weather.T_FDN,label='Calm weather Operations')

# plt.xticks(rotation=45, fontsize=20)
# plt.yticks(fontsize=20)
# plt.grid(visible=True)
# plt.ylabel('Duration foundation installation [h]', fontsize= 20)
# plt.legend(fontsize= 'xx-large', frameon=True)#, bbox_to_anchor=(1.0, 1.0))
# plt.grid(b=True, which='major', color='#666666', linestyle='-')
# plt.minorticks_on()
# plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
# plt.ylim((0,1600))

# plt.savefig('foundation duration farm installation.pdf',bbox_inches='tight')
# #plt.title("Total duration installation offshore wind farm, calm weather comparison", fontsize = 28)

#%% plot mean duration foundation installation
for option in options:
    if foundations_duration_1[option].isna().sum() > 0:
        foundations_duration_1.loc[:, option] = np.nan
    if foundations_duration_12[option].isna().sum() > 0:
        foundations_duration_12.loc[:, option] = np.nan
    if foundations_duration_24[option].isna().sum() > 0:
        foundations_duration_24.loc[:, option] = np.nan
    if foundations_duration_48[option].isna().sum() > 0:
        foundations_duration_48.loc[:, option] = np.nan
    if foundations_duration_72[option].isna().sum() > 0:
        foundations_duration_72.loc[:, option] = np.nan
    if foundations_duration_96[option].isna().sum() > 0:
        foundations_duration_96.loc[:, option] = np.nan
        
mean_foundations_duration_WR = foundations_duration_WR.copy().mean()
mean_foundations_duration_1 = foundations_duration_1.copy().mean()
mean_foundations_duration_12 = foundations_duration_12.copy().mean()
mean_foundations_duration_24 = foundations_duration_24.copy().mean()
mean_foundations_duration_48 = foundations_duration_48.copy().mean()
mean_foundations_duration_72 = foundations_duration_72.copy().mean()        
mean_foundations_duration_96 = foundations_duration_96.copy().mean()

plt.figure(figsize=(12,10))
plt.scatter(options, mean_foundations_duration_WR, label='Perfect forecast', color='black', s=100)
plt.scatter(options, mean_foundations_duration_1, label='1 hour horizon', marker='x', color='green', s=200)
plt.scatter(options, mean_foundations_duration_12, label='12 hour horizon', marker='x', color='red', s=200)
plt.scatter(options, mean_foundations_duration_24, label='24 hour horizon', marker='x', color='xkcd:gold', s=200)
plt.scatter(options, mean_foundations_duration_48, label='48 hour horizon', marker='x', color='xkcd:ocean blue', s=200)
#plt.scatter(options, mean_foundations_duration_72, label='72 hour horizon', marker='x', color='xkcd:grape', s=200)
#plt.scatter(options, mean_foundations_duration_96, label='96 hour horizon', marker='x', color='xkcd:pumpkin', s=200)

plt.bar(range(0,len(results_calm_weather)),results_calm_weather.T_FDN,label='Calm weather')

plt.xticks(rotation=45, fontsize=20)
plt.yticks(fontsize=20)
plt.grid(visible=True)
plt.ylabel('Mean duration foundation installation [h]', fontsize= 20)
plt.legend(fontsize= 'xx-large', frameon=True)#, bbox_to_anchor=(1.0, 1.0))
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
#plt.ylim((0,1600))

plt.savefig('Mean foundation duration farm installation.pdf',bbox_inches='tight')

#%% plot duration turbine installation

# results_weather_restricted_2008 = turbines_duration_WR.iloc[0,:]
# results_weather_restricted_2009 = turbines_duration_WR.iloc[1,:]
# results_weather_restricted_2010 = turbines_duration_WR.iloc[2,:]

# results_weather_uncertainty_2008 = T_WTB.iloc[0,:]
# results_weather_uncertainty_2009 = T_WTB.iloc[1,:]
# results_weather_uncertainty_2010 = T_WTB.iloc[2,:]

# y1 = [results_weather_restricted_2008, results_weather_restricted_2009, results_weather_restricted_2010]
# y2 = [results_weather_uncertainty_2008, results_weather_uncertainty_2009, results_weather_uncertainty_2010]
# colors = ['green', 'xkcd:gold', 'red']

# plt.figure(figsize=(15,10))

# for y1e, y2e, color in zip(y1, y2, colors):
#     plt.scatter(options, y1e, label=str(y1e.name)+' perfect forecast', color=color, s=100)
#     plt.scatter(options, y2e, label=str(y2e.name)+' with uncertainty', marker='x', color=color, s=200)
       
# plt.bar(range(0,len(results_calm_weather)),results_calm_weather.T_WTB,label='Calm weather Operations')

# plt.xticks(rotation=45, fontsize=20)
# plt.yticks(fontsize=20)
# plt.grid(visible=True)
# plt.ylabel('Duration turbine installation [h]', fontsize= 20)
# plt.legend(fontsize= 'xx-large', frameon=True)#, bbox_to_anchor=(1.0, 1.0))
# plt.grid(b=True, which='major', color='#666666', linestyle='-')
# plt.minorticks_on()
# plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
# plt.ylim((0,2500))

# plt.savefig('turbine duration farm installation.pdf',bbox_inches='tight')
# #plt.title("Total duration installation offshore wind farm, calm weather comparison", fontsize = 28)

#%% plot mean duration turbine installation
for option in options:
    if turbines_duration_1[option].isna().sum() > 0:
        turbines_duration_1.loc[:, option] = np.nan
    if turbines_duration_12[option].isna().sum() > 0:
        turbines_duration_12.loc[:, option] = np.nan
    if turbines_duration_24[option].isna().sum() > 0:
        turbines_duration_24.loc[:, option] = np.nan
    if turbines_duration_48[option].isna().sum() > 0:
        turbines_duration_48.loc[:, option] = np.nan
    if turbines_duration_72[option].isna().sum() > 0:
        turbines_duration_72.loc[:, option] = np.nan
    if turbines_duration_96[option].isna().sum() > 0:
        turbines_duration_96.loc[:, option] = np.nan
        
mean_turbines_duration_WR = turbines_duration_WR.copy().mean()
mean_turbines_duration_1 = turbines_duration_1.copy().mean()
mean_turbines_duration_12 = turbines_duration_12.copy().mean()
mean_turbines_duration_24 = turbines_duration_24.copy().mean()
mean_turbines_duration_48 = turbines_duration_48.copy().mean()
mean_turbines_duration_72 = turbines_duration_72.copy().mean()        
mean_turbines_duration_96 = turbines_duration_96.copy().mean()

plt.figure(figsize=(12,10))
plt.scatter(options, mean_turbines_duration_WR, label='Perfect forecast', color='black', s=100)
plt.scatter(options, mean_turbines_duration_1, label='1 hour horizon', marker='x', color='green', s=200)
plt.scatter(options, mean_turbines_duration_12, label='12 hour horizon', marker='x', color='red', s=200)
plt.scatter(options, mean_turbines_duration_24, label='24 hour horizon', marker='x', color='xkcd:gold', s=200)
#plt.scatter(options, mean_turbines_duration_48, label='48 hour horizon', marker='x', color='xkcd:ocean blue', s=200)
#plt.scatter(options, mean_turbines_duration_72, label='72 hour horizon', marker='x', color='xkcd:grape', s=200)
#plt.scatter(options, mean_turbines_duration_96, label='96 hour horizon', marker='x', color='xkcd:pumpkin', s=200)

plt.bar(range(0,len(results_calm_weather)),results_calm_weather.T_WTB,label='Calm weather')

plt.xticks(rotation=45, fontsize=20)
plt.yticks(fontsize=20)
plt.grid(visible=True)
plt.ylabel('Mean duration turbine installation [h]', fontsize= 20)
plt.legend(fontsize= 'xx-large', frameon=True, bbox_to_anchor=(0.85, 1.0), loc='upper left')
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
#plt.ylim((0,2500))

plt.savefig('Mean turbine duration farm installation.pdf',bbox_inches='tight')

#%% plot cost foundation installation

# results_weather_restricted_2008 = foundations_cost_WR.iloc[0,:]
# results_weather_restricted_2009 = foundations_cost_WR.iloc[1,:]
# results_weather_restricted_2010 = foundations_cost_WR.iloc[2,:]

# results_weather_uncertainty_2008 = COST_FDN.iloc[0,:]
# results_weather_uncertainty_2009 = COST_FDN.iloc[1,:]
# results_weather_uncertainty_2010 = COST_FDN.iloc[2,:]

# y1 = [results_weather_restricted_2008, results_weather_restricted_2009, results_weather_restricted_2010]
# y2 = [results_weather_uncertainty_2008, results_weather_uncertainty_2009, results_weather_uncertainty_2010]
# colors = ['green', 'xkcd:gold', 'red']

# plt.figure(figsize=(15,10))

# for y1e, y2e, color in zip(y1, y2, colors):
#     plt.scatter(options, y1e, label=str(y1e.name)+' perfect forecast', color=color, s=100)
#     plt.scatter(options, y2e, label=str(y2e.name)+' with uncertainty', marker='x', color=color, s=200)
       
# plt.bar(range(0,len(results_calm_weather)),results_calm_weather.COST_FDN,label='Calm weather Operations')

# plt.xticks(rotation=45, fontsize=20)
# plt.yticks(fontsize=20)
# plt.grid(visible=True)
# plt.ylabel('Cost foundation installation [k€]', fontsize= 20)
# plt.legend(fontsize= 'xx-large', frameon=True)#, bbox_to_anchor=(1.0, 1.0))
# plt.grid(b=True, which='major', color='#666666', linestyle='-')
# plt.minorticks_on()
# plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
# plt.ylim((0,20000))

# plt.savefig('foundation cost farm installation.pdf',bbox_inches='tight')
# #plt.title("Total cost installation offshore wind farm, calm weather comparison", fontsize = 28)

#%% plot mean cost foundation installation
for option in options:
    if foundations_cost_1[option].isna().sum() > 0:
        foundations_cost_1.loc[:, option] = np.nan
    if foundations_cost_12[option].isna().sum() > 0:
        foundations_cost_12.loc[:, option] = np.nan
    if foundations_cost_24[option].isna().sum() > 0:
        foundations_cost_24.loc[:, option] = np.nan
    if foundations_cost_48[option].isna().sum() > 0:
        foundations_cost_48.loc[:, option] = np.nan
    if foundations_cost_72[option].isna().sum() > 0:
        foundations_cost_72.loc[:, option] = np.nan
    if foundations_cost_96[option].isna().sum() > 0:
        foundations_cost_96.loc[:, option] = np.nan
        
mean_foundations_cost_WR = foundations_cost_WR.copy().mean()
mean_foundations_cost_1 = foundations_cost_1.copy().mean()
mean_foundations_cost_12 = foundations_cost_12.copy().mean()
mean_foundations_cost_24 = foundations_cost_24.copy().mean()
mean_foundations_cost_48 = foundations_cost_48.copy().mean()
mean_foundations_cost_72 = foundations_cost_72.copy().mean()        
mean_foundations_cost_96 = foundations_cost_96.copy().mean()

plt.figure(figsize=(12,10))
plt.scatter(options, mean_foundations_cost_WR, label='Perfect forecast', color='black', s=100)
plt.scatter(options, mean_foundations_cost_1, label='1 hour horizon', marker='x', color='green', s=200)
plt.scatter(options, mean_foundations_cost_12, label='12 hour horizon', marker='x', color='red', s=200)
plt.scatter(options, mean_foundations_cost_24, label='24 hour horizon', marker='x', color='xkcd:gold', s=200)
plt.scatter(options, mean_foundations_cost_48, label='48 hour horizon', marker='x', color='xkcd:ocean blue', s=200)
#plt.scatter(options, mean_foundations_cost_72, label='72 hour horizon', marker='x', color='xkcd:grape', s=200)
#plt.scatter(options, mean_foundations_cost_96, label='96 hour horizon', marker='x', color='xkcd:pumpkin', s=200)

plt.bar(range(0,len(results_calm_weather)),results_calm_weather.COST_FDN,label='Calm weather')

plt.xticks(rotation=45, fontsize=20)
plt.yticks(fontsize=20)
plt.grid(visible=True)
plt.ylabel('Mean cost foundation installation [k€]', fontsize= 20)
plt.legend(fontsize= 'xx-large', frameon=True)#, bbox_to_anchor=(1, 1), loc='upper left')
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
#plt.ylim((0,20000))

plt.savefig('Mean foundation cost farm installation.pdf',bbox_inches='tight')

#%% plot cost turbine installation

# results_weather_restricted_2008 = turbines_cost_WR.iloc[0,:]
# results_weather_restricted_2009 = turbines_cost_WR.iloc[1,:]
# results_weather_restricted_2010 = turbines_cost_WR.iloc[2,:]

# results_weather_uncertainty_2008 = COST_WTB.iloc[0,:]
# results_weather_uncertainty_2009 = COST_WTB.iloc[1,:]
# results_weather_uncertainty_2010 = COST_WTB.iloc[2,:]

# y1 = [results_weather_restricted_2008, results_weather_restricted_2009, results_weather_restricted_2010]
# y2 = [results_weather_uncertainty_2008, results_weather_uncertainty_2009, results_weather_uncertainty_2010]
# colors = ['green', 'xkcd:gold', 'red']

# plt.figure(figsize=(15,10))

# for y1e, y2e, color in zip(y1, y2, colors):
#     plt.scatter(options, y1e, label=str(y1e.name)+' perfect forecast', color=color, s=100)
#     plt.scatter(options, y2e, label=str(y2e.name)+' with uncertainty', marker='x', color=color, s=200)
       
# plt.bar(range(0,len(results_calm_weather)),results_calm_weather.COST_WTB,label='Calm weather Operations')

# plt.xticks(rotation=45, fontsize=20)
# plt.yticks(fontsize=20)
# plt.grid(visible=True)
# plt.ylabel('Cost turbine installation [k€]', fontsize= 20)
# plt.legend(fontsize= 'xx-large', frameon=True)#, bbox_to_anchor=(1.0, 1.0))
# plt.grid(b=True, which='major', color='#666666', linestyle='-')
# plt.minorticks_on()
# plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
# plt.ylim((0,32500))

# plt.savefig('turbine cost farm installation.pdf',bbox_inches='tight')
# #plt.title("Total cost installation offshore wind farm, calm weather comparison", fontsize = 28)

#%% plot mean cost turbine installation
for option in options:
    if turbines_cost_1[option].isna().sum() > 0:
        turbines_cost_1.loc[:, option] = np.nan
    if turbines_cost_12[option].isna().sum() > 0:
        turbines_cost_12.loc[:, option] = np.nan
    if turbines_cost_24[option].isna().sum() > 0:
        turbines_cost_24.loc[:, option] = np.nan
    if turbines_cost_48[option].isna().sum() > 0:
        turbines_cost_48.loc[:, option] = np.nan
    if turbines_cost_72[option].isna().sum() > 0:
        turbines_cost_72.loc[:, option] = np.nan
    if turbines_cost_96[option].isna().sum() > 0:
        turbines_cost_96.loc[:, option] = np.nan
        
mean_turbines_cost_WR = turbines_cost_WR.copy().mean()
mean_turbines_cost_1 = turbines_cost_1.copy().mean()
mean_turbines_cost_12 = turbines_cost_12.copy().mean()
mean_turbines_cost_24 = turbines_cost_24.copy().mean()
mean_turbines_cost_48 = turbines_cost_48.copy().mean()
mean_turbines_cost_72 = turbines_cost_72.copy().mean()        
mean_turbines_cost_96 = turbines_cost_96.copy().mean()

plt.figure(figsize=(12,10))
plt.scatter(options, mean_turbines_cost_WR, label='Perfect forecast', color='black', s=100)
plt.scatter(options, mean_turbines_cost_1, label='1 hour horizon', marker='x', color='green', s=200)
plt.scatter(options, mean_turbines_cost_12, label='12 hour horizon', marker='x', color='red', s=200)
plt.scatter(options, mean_turbines_cost_24, label='24 hour horizon', marker='x', color='xkcd:gold', s=200)
#plt.scatter(options, mean_turbines_cost_48, label='48 hour horizon', marker='x', color='xkcd:ocean blue', s=200)
#plt.scatter(options, mean_turbines_cost_72, label='72 hour horizon', marker='x', color='xkcd:grape', s=200)
#plt.scatter(options, mean_turbines_cost_96, label='96 hour horizon', marker='x', color='xkcd:pumpkin', s=200)

plt.bar(range(0,len(results_calm_weather)),results_calm_weather.COST_WTB,label='Calm weather')

plt.xticks(rotation=45, fontsize=20)
plt.yticks(fontsize=20)
plt.grid(visible=True)
plt.ylabel('Mean cost turbine installation [k€]', fontsize= 20)
plt.legend(fontsize= 'xx-large', frameon=True, bbox_to_anchor=(0.85, 1.0), loc='upper left')
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
#plt.ylim((0,32500))

plt.savefig('Mean turbine cost farm installation.pdf',bbox_inches='tight')

#%% plot total waiting time

# results_weather_restricted_2008 = total_waiting_time_WR.iloc[0,:]
# results_weather_restricted_2009 = total_waiting_time_WR.iloc[1,:]
# results_weather_restricted_2010 = total_waiting_time_WR.iloc[2,:]

# results_weather_uncertainty_2008 = WAIT_TOT.iloc[0,:]
# results_weather_uncertainty_2009 = WAIT_TOT.iloc[1,:]
# results_weather_uncertainty_2010 = WAIT_TOT.iloc[2,:]

# y1 = [results_weather_restricted_2008, results_weather_restricted_2009, results_weather_restricted_2010]
# y2 = [results_weather_uncertainty_2008, results_weather_uncertainty_2009, results_weather_uncertainty_2010]
# colors = ['green', 'xkcd:gold', 'red']

# plt.figure(figsize=(15,10))

# for y1e, y2e, color in zip(y1, y2, colors):
#     plt.scatter(options, y1e, label=str(y1e.name)+' perfect forecast', color=color, s=100)
#     plt.scatter(options, y2e, label=str(y2e.name)+' with uncertainty', marker='x', color=color, s=200)

# plt.xticks(rotation=45, fontsize=20)
# plt.yticks(fontsize=20)
# plt.grid(visible=True)
# plt.ylabel('Total duration waiting on weather [h]', fontsize= 20)
# plt.legend(fontsize= 'xx-large', frameon=True)#, bbox_to_anchor=(1.0, 1.0))
# plt.grid(b=True, which='major', color='#666666', linestyle='-')
# plt.minorticks_on()
# plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
# plt.ylim((0,3000))

# plt.savefig('total waiting time farm installation.pdf',bbox_inches='tight')
# #plt.title("Total cost installation offshore wind farm, calm weather comparison", fontsize = 28)

#%% plot mean total waiting time
for option in options:
    if total_waiting_time_1[option].isna().sum() > 0:
        total_waiting_time_1.loc[:, option] = np.nan
    if total_waiting_time_12[option].isna().sum() > 0:
        total_waiting_time_12.loc[:, option] = np.nan
    if total_waiting_time_24[option].isna().sum() > 0:
        total_waiting_time_24.loc[:, option] = np.nan
    if total_waiting_time_48[option].isna().sum() > 0:
        total_waiting_time_48.loc[:, option] = np.nan
    if total_waiting_time_72[option].isna().sum() > 0:
        total_waiting_time_72.loc[:, option] = np.nan
    if total_waiting_time_96[option].isna().sum() > 0:
        total_waiting_time_96.loc[:, option] = np.nan
        
mean_total_waiting_time_WR = total_waiting_time_WR.copy().mean()
mean_total_waiting_time_1 = total_waiting_time_1.copy().mean()
mean_total_waiting_time_12 = total_waiting_time_12.copy().mean()
mean_total_waiting_time_24 = total_waiting_time_24.copy().mean()
mean_total_waiting_time_48 = total_waiting_time_48.copy().mean()
mean_total_waiting_time_72 = total_waiting_time_72.copy().mean()        
mean_total_waiting_time_96 = total_waiting_time_96.copy().mean()

plt.figure(figsize=(12,10))
plt.scatter(options, mean_total_waiting_time_WR, label='Perfect forecast', color='black', s=100)
plt.scatter(options, mean_total_waiting_time_1, label='1 hour horizon', marker='x', color='green', s=200)
plt.scatter(options, mean_total_waiting_time_12, label='12 hour horizon', marker='x', color='red', s=200)
plt.scatter(options, mean_total_waiting_time_24, label='24 hour horizon', marker='x', color='xkcd:gold', s=200)
#plt.scatter(options, mean_total_waiting_time_48, label='48 hour horizon', marker='x', color='xkcd:ocean blue', s=200)
#plt.scatter(options, mean_total_waiting_time_72, label='72 hour horizon', marker='x', color='xkcd:grape', s=200)
#plt.scatter(options, mean_total_waiting_time_96, label='96 hour horizon', marker='x', color='xkcd:pumpkin', s=200)

plt.xticks(rotation=45, fontsize=20)
plt.yticks(fontsize=20)
plt.grid(visible=True)
plt.ylabel('Mean total duration waiting on weather [h]', fontsize= 20)
plt.legend(fontsize= 'xx-large', frameon=True, bbox_to_anchor=(1.0, 1.0))
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
#plt.ylim((0,3000))

plt.savefig('Mean total waiting time farm installation.pdf',bbox_inches='tight')

#%% plot operability

mean_operability_WR = results_calm_weather.T_TOT/mean_total_duration_WR
mean_operability_1 = results_calm_weather.T_TOT/mean_total_duration_1
mean_operability_12 = results_calm_weather.T_TOT/mean_total_duration_12
mean_operability_24 = results_calm_weather.T_TOT/mean_total_duration_24
mean_operability_48 = results_calm_weather.T_TOT/mean_total_duration_48
mean_operability_72 = results_calm_weather.T_TOT/mean_total_duration_72
mean_operability_96 = results_calm_weather.T_TOT/mean_total_duration_96

#mean_operability = pd.concat((mean_operability_WR, mean_operability_1, mean_operability_12, mean_operability_24, mean_operability_48, mean_operability_72, mean_operability_96), axis=1)
#mean_operability = mean_operability.rename({0: 'Perfect forecast', 1: '1 hour horizon', 2: '12 hour horizon', 3: '24 hour horizon', 4: '48 hour horizon', 5: '72 hour horizon', 6: '96 hour horizon'}, axis='columns')
#mean_operability.plot(kind = 'bar', rot=45, grid = True, figsize =(12,10), fontsize = 20)

plt.figure(figsize=(12,10))
plt.scatter(options, mean_operability_WR, label='Perfect forecast', color='black', s=100)
plt.scatter(options, mean_operability_1, label='1 hour horizon', marker='x', color='green', s=200)
plt.scatter(options, mean_operability_12, label='12 hour horizon', marker='x', color='red', s=200)
plt.scatter(options, mean_operability_24, label='24 hour horizon', marker='x', color='xkcd:gold', s=200)
#plt.scatter(options, mean_operability_48, label='48 hour horizon', marker='x', color='xkcd:ocean blue', s=200)
#plt.scatter(options, mean_operability_72, label='72 hour horizon', marker='x', color='xkcd:grape', s=200)
#plt.scatter(options, mean_operability_96, label='96 hour horizon', marker='x', color='xkcd:pumpkin', s=200)

plt.xticks(rotation=45, fontsize=20)
plt.yticks(fontsize=20)
plt.grid(visible=True)
plt.ylabel('Mean total duration waiting on weather [h]', fontsize= 20)
plt.legend(fontsize= 'xx-large', frameon=True, bbox_to_anchor=(1.0, 1.0))
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.savefig('Operability factor.pdf', bbox_inches = 'tight')

#%% plot average increase in duration compared to results calm weather model

avg_increasepdtime_WR = 100*((mean_total_duration_WR - results_calm_weather.loc[:,'T_TOT'])/results_calm_weather.loc[:,'T_TOT'])
avg_increasepdtime_1 = 100*((mean_total_duration_1 - results_calm_weather.loc[:,'T_TOT'])/results_calm_weather.loc[:,'T_TOT'])
avg_increasepdtime_12 = 100*((mean_total_duration_12 - results_calm_weather.loc[:,'T_TOT'])/results_calm_weather.loc[:,'T_TOT'])
avg_increasepdtime_24 = 100*((mean_total_duration_24 - results_calm_weather.loc[:,'T_TOT'])/results_calm_weather.loc[:,'T_TOT'])
avg_increasepdtime_48 = 100*((mean_total_duration_48 - results_calm_weather.loc[:,'T_TOT'])/results_calm_weather.loc[:,'T_TOT'])
avg_increasepdtime_72 = 100*((mean_total_duration_72 - results_calm_weather.loc[:,'T_TOT'])/results_calm_weather.loc[:,'T_TOT'])
avg_increasepdtime_96 = 100*((mean_total_duration_96 - results_calm_weather.loc[:,'T_TOT'])/results_calm_weather.loc[:,'T_TOT'])

plt.figure(figsize=(12,10))
plt.scatter(options, avg_increasepdtime_WR, label='Perfect forecast', color='black', s=100)
plt.scatter(options, avg_increasepdtime_1, label='1 hour horizon', marker='x', color='green', s=200)
plt.scatter(options, avg_increasepdtime_12, label='12 hour horizon', marker='x', color='red', s=200)
plt.scatter(options, avg_increasepdtime_24, label='24 hour horizon', marker='x', color='xkcd:gold', s=200)
#plt.scatter(options, avg_increasepdtime_48, label='48 hour horizon', marker='x', color='xkcd:ocean blue', s=200)
#plt.scatter(options, avg_increasepdtime_72, label='72 hour horizon', marker='x', color='xkcd:grape', s=200)
#plt.scatter(options, avg_increasepdtime_96, label='96 hour horizon', marker='x', color='xkcd:pumpkin', s=200)

plt.xticks(rotation=45, fontsize=20)
plt.yticks(fontsize=20)
plt.grid(visible=True)
plt.ylabel('Average increase in duration of installation [%]', fontsize= 20)
plt.legend(fontsize= 'xx-large', frameon=True, bbox_to_anchor=(0.75, 1.0))
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.ylim(ymin=0)
plt.savefig('Average increase duration compared to calm weather.pdf', bbox_inches  = 'tight')

#%% plot average increase in cost compared to results calm weather model

avg_increasepdcost_WR = 100*((mean_total_cost_WR - results_calm_weather.loc[:,'COST_TOT'])/results_calm_weather.loc[:,'COST_TOT'])
avg_increasepdcost_1 = 100*((mean_total_cost_1 - results_calm_weather.loc[:,'COST_TOT'])/results_calm_weather.loc[:,'COST_TOT'])
avg_increasepdcost_12 = 100*((mean_total_cost_12 - results_calm_weather.loc[:,'COST_TOT'])/results_calm_weather.loc[:,'COST_TOT'])
avg_increasepdcost_24 = 100*((mean_total_cost_24 - results_calm_weather.loc[:,'COST_TOT'])/results_calm_weather.loc[:,'COST_TOT'])
avg_increasepdcost_48 = 100*((mean_total_cost_48 - results_calm_weather.loc[:,'COST_TOT'])/results_calm_weather.loc[:,'COST_TOT'])
avg_increasepdcost_72 = 100*((mean_total_cost_72 - results_calm_weather.loc[:,'COST_TOT'])/results_calm_weather.loc[:,'COST_TOT'])
avg_increasepdcost_96 = 100*((mean_total_cost_96 - results_calm_weather.loc[:,'COST_TOT'])/results_calm_weather.loc[:,'COST_TOT'])

plt.figure(figsize=(12,10))
plt.scatter(options, avg_increasepdcost_WR, label='Perfect forecast', color='black', s=100)
plt.scatter(options, avg_increasepdcost_1, label='1 hour horizon', marker='x', color='green', s=200)
plt.scatter(options, avg_increasepdcost_12, label='12 hour horizon', marker='x', color='red', s=200)
plt.scatter(options, avg_increasepdcost_24, label='24 hour horizon', marker='x', color='xkcd:gold', s=200)
#plt.scatter(options, avg_increasepdcost_48, label='48 hour horizon', marker='x', color='xkcd:ocean blue', s=200)
#plt.scatter(options, avg_increasepdcost_72, label='72 hour horizon', marker='x', color='xkcd:grape', s=200)
#plt.scatter(options, avg_increasepdcost_96, label='96 hour horizon', marker='x', color='xkcd:pumpkin', s=200)

plt.xticks(rotation=45, fontsize=20)
plt.yticks(fontsize=20)
plt.grid(visible=True)
plt.ylabel('Average increase in cost of installation [%]', fontsize= 20)
plt.legend(fontsize= 'xx-large', frameon=True, bbox_to_anchor=(0.75, 1.0))
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.ylim(ymin=0)
plt.savefig('Average increase cost compared to calm weather.pdf', bbox_inches= 'tight')

#%% plot average increase in duration compared to results weather restricted model

avg_increasepdtime_1 = 100*((mean_total_duration_1 - mean_total_duration_WR)/mean_total_duration_WR)
avg_increasepdtime_12 = 100*((mean_total_duration_12 - mean_total_duration_WR)/mean_total_duration_WR)
avg_increasepdtime_24 = 100*((mean_total_duration_24 - mean_total_duration_WR)/mean_total_duration_WR)
avg_increasepdtime_48 = 100*((mean_total_duration_48 - mean_total_duration_WR)/mean_total_duration_WR)
avg_increasepdtime_72 = 100*((mean_total_duration_72 - mean_total_duration_WR)/mean_total_duration_WR)
avg_increasepdtime_96 = 100*((mean_total_duration_96 - mean_total_duration_WR)/mean_total_duration_WR)

plt.figure(figsize=(12,10))
plt.scatter(options, avg_increasepdtime_1, label='1 hour horizon', marker='x', color='green', s=200)
plt.scatter(options, avg_increasepdtime_12, label='12 hour horizon', marker='x', color='red', s=200)
plt.scatter(options, avg_increasepdtime_24, label='24 hour horizon', marker='x', color='xkcd:gold', s=200)
#plt.scatter(options, avg_increasepdtime_48, label='48 hour horizon', marker='x', color='xkcd:ocean blue', s=200)
#plt.scatter(options, avg_increasepdtime_72, label='72 hour horizon', marker='x', color='xkcd:grape', s=200)
#plt.scatter(options, avg_increasepdtime_96, label='96 hour horizon', marker='x', color='xkcd:pumpkin', s=200)

plt.xticks(rotation=45, fontsize=20)
plt.yticks(fontsize=20)
plt.grid(visible=True)
plt.ylabel('Average increase in duration of installation [%]', fontsize= 20)
plt.legend(fontsize= 'xx-large', frameon=True, bbox_to_anchor=(1.0, 1.0))
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.ylim(ymin=0)
plt.savefig('Average increase duration compared to weather restricted.pdf', bbox_inches  = 'tight')

#%% plot average increase in cost compared to results weather restricted model

avg_increasepdcost_1 = 100*((mean_total_cost_1 - mean_total_cost_WR)/mean_total_cost_WR)
avg_increasepdcost_12 = 100*((mean_total_cost_12 - mean_total_duration_WR)/mean_total_cost_WR)
avg_increasepdcost_24 = 100*((mean_total_cost_24 - mean_total_duration_WR)/mean_total_cost_WR)
avg_increasepdcost_48 = 100*((mean_total_cost_48 - mean_total_duration_WR)/mean_total_cost_WR)
avg_increasepdcost_72 = 100*((mean_total_cost_72 - mean_total_duration_WR)/mean_total_cost_WR)
avg_increasepdcost_96 = 100*((mean_total_cost_96 - mean_total_duration_WR)/mean_total_cost_WR)

plt.figure(figsize=(12,10))
plt.scatter(options, avg_increasepdcost_1, label='1 hour horizon', marker='x', color='green', s=200)
plt.scatter(options, avg_increasepdcost_12, label='12 hour horizon', marker='x', color='red', s=200)
plt.scatter(options, avg_increasepdcost_24, label='24 hour horizon', marker='x', color='xkcd:gold', s=200)
#plt.scatter(options, avg_increasepdcost_48, label='48 hour horizon', marker='x', color='xkcd:ocean blue', s=200)
#plt.scatter(options, avg_increasepdcost_72, label='72 hour horizon', marker='x', color='xkcd:grape', s=200)
#plt.scatter(options, avg_increasepdcost_96, label='96 hour horizon', marker='x', color='xkcd:pumpkin', s=200)

plt.xticks(rotation=45, fontsize=20)
plt.yticks(fontsize=20)
plt.grid(visible=True)
plt.ylabel('Average increase in cost of installation [%]', fontsize= 20)
plt.legend(fontsize= 'xx-large', frameon=True, bbox_to_anchor=(1.0, 1.0))
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.ylim(ymin=0)
plt.savefig('Average increase cost compared to weather restricted.pdf', bbox_inches= 'tight')

#%% Plot average total duration installation per OWT

average_duration_per_OWT_calm_weather = results_calm_weather.T_TOT/720
average_duration_per_OWT_WR = mean_total_duration_WR/720
average_duration_per_OWT_1 = mean_total_duration_1/720
average_duration_per_OWT_12 = mean_total_duration_12/720
average_duration_per_OWT_24 = mean_total_duration_24/720
average_duration_per_OWT_48 = mean_total_duration_48/720
average_duration_per_OWT_72 = mean_total_duration_72/720
average_duration_per_OWT_96 = mean_total_duration_96/720

plt.figure(figsize=(12,10))
plt.scatter(options, average_duration_per_OWT_WR, label='Perfect forecast', color='black', s=100)
plt.scatter(options, average_duration_per_OWT_1, label='1 hour horizon', marker='x', color='green', s=200)
plt.scatter(options, average_duration_per_OWT_12, label='12 hour horizon', marker='x', color='red', s=200)
plt.scatter(options, average_duration_per_OWT_24, label='24 hour horizon', marker='x', color='xkcd:gold', s=200)
#plt.scatter(options, average_duration_per_OWT_48, label='48 hour horizon', marker='x', color='xkcd:ocean blue', s=200)
#plt.scatter(options, average_duration_per_OWT_72, label='72 hour horizon', marker='x', color='xkcd:grape', s=200)
#plt.scatter(options, average_duration_per_OWT_96, label='96 hour horizon', marker='x', color='xkcd:pumpkin', s=200)

average_duration_per_OWT_calm_weather.plot(kind= 'bar', fontsize = 20 , grid = True, rot = 45, label='Calm weather')

plt.xticks(rotation=45, fontsize=20)
plt.yticks(fontsize=20)
plt.grid(visible=True)
plt.ylabel('Average total duration installation [days/assemby]', fontsize= 20)
plt.xlabel('')
plt.legend(fontsize= 'xx-large', frameon=True)#, bbox_to_anchor=(1.0, 1.0))
plt.grid(b=True, which='major', color='#666666', linestyle='-') 
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.savefig('Average duration of installation per assembly.pdf', bbox_inches= 'tight')

#%% Plot average total cost installation per MW

average_cost_per_MW_calm_weather = results_calm_weather.COST_TOT/300
average_cost_per_MW_WR = mean_total_cost_WR/300
average_cost_per_MW_weather_1 = mean_total_cost_1/300
average_cost_per_MW_weather_12 = mean_total_cost_12/300
average_cost_per_MW_weather_24 = mean_total_cost_24/300
average_cost_per_MW_weather_48 = mean_total_cost_48/300
average_cost_per_MW_weather_72 = mean_total_cost_72/300
average_cost_per_MW_weather_96 = mean_total_cost_96/300

plt.figure(figsize=(12,10))
plt.scatter(options, average_cost_per_MW_WR, label='Perfect forecast', color='black', s=100)
plt.scatter(options, average_cost_per_MW_weather_1, label='1 hour horizon', marker='x', color='green', s=200)
plt.scatter(options, average_cost_per_MW_weather_12, label='12 hour horizon', marker='x', color='red', s=200)
plt.scatter(options, average_cost_per_MW_weather_24, label='24 hour horizon', marker='x', color='xkcd:gold', s=200)
#plt.scatter(options, average_cost_per_MW_weather_48, label='48 hour horizon', marker='x', color='xkcd:ocean blue', s=200)
#plt.scatter(options, average_cost_per_MW_weather_72, label='72 hour horizon', marker='x', color='xkcd:grape', s=200)
#plt.scatter(options, average_cost_per_MW_weather_96, label='96 hour horizon', marker='x', color='xkcd:pumpkin', s=200)

average_cost_per_MW_calm_weather.plot(kind= 'bar', fontsize = 20 , grid = True, rot = 45, label='Calm weather')

plt.xticks(rotation=45, fontsize=20)
plt.yticks(fontsize=20)
plt.grid(visible=True)
plt.ylabel('Average Cost wind turbine install [k€/MW]', fontsize= 20)
plt.xlabel('')
plt.legend(fontsize= 'xx-large', frameon=True, bbox_to_anchor=(1.0, 1.0))
plt.grid(b=True, which='major', color='#666666', linestyle='-') 
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.savefig('Average cost of installation per MW.pdf', bbox_inches ='tight')


#%% Plot waiting times individual operations during foundation installation

# Plot time waiting on weather for transit during foundation installation
for option in options:
    if waiting_time_transit_FDN_1[option].isna().sum() > 0:
        waiting_time_transit_FDN_1.loc[:, option] = np.nan
    if waiting_time_transit_FDN_12[option].isna().sum() > 0:
        waiting_time_transit_FDN_12.loc[:, option] = np.nan
    if waiting_time_transit_FDN_24[option].isna().sum() > 0:
        waiting_time_transit_FDN_24.loc[:, option] = np.nan
    if waiting_time_transit_FDN_48[option].isna().sum() > 0:
        waiting_time_transit_FDN_48.loc[:, option] = np.nan
    if waiting_time_transit_FDN_72[option].isna().sum() > 0:
        waiting_time_transit_FDN_72.loc[:, option] = np.nan
    if waiting_time_transit_FDN_96[option].isna().sum() > 0:
        waiting_time_transit_FDN_96.loc[:, option] = np.nan

waiting_transit_FDN_WR = waiting_time_transit_FDN_WR.mean()
waiting_transit_FDN_1 = waiting_time_transit_FDN_1.mean()
waiting_transit_FDN_12 = waiting_time_transit_FDN_12.mean()
waiting_transit_FDN_24 = waiting_time_transit_FDN_24.mean()
waiting_transit_FDN_48 = waiting_time_transit_FDN_48.mean()
waiting_transit_FDN_72 = waiting_time_transit_FDN_72.mean()
waiting_transit_FDN_96 = waiting_time_transit_FDN_96.mean()

plt.figure(figsize=(12,10))
plt.scatter(options, waiting_transit_FDN_WR, label='Perfect forecast', color='black', s=100)
plt.scatter(options, waiting_transit_FDN_1, label='1 hour horizon', marker='x', color='green', s=200)
plt.scatter(options, waiting_transit_FDN_12, label='12 hour horizon', marker='x', color='red', s=200)
plt.scatter(options, waiting_transit_FDN_24, label='24 hour horizon', marker='x', color='xkcd:gold', s=200)
plt.scatter(options, waiting_transit_FDN_48, label='48 hour horizon', marker='x', color='xkcd:ocean blue', s=200)
#plt.scatter(options, waiting_transit_FDN_72, label='72 hour horizon', marker='x', color='xkcd:grape', s=200)
#plt.scatter(options, waiting_transit_FDN_96, label='96 hour horizon', marker='x', color='xkcd:pumpkin', s=200)

plt.xticks(rotation=45, fontsize=20)
plt.yticks(fontsize=20)
plt.grid(visible=True)
plt.ylabel('Waiting on weather [h]', fontsize= 20)
plt.legend(fontsize= 'xx-large', frameon=True)#, bbox_to_anchor=(1.0, 1.0), loc='upper left')
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.ylim(ymin=0)
plt.savefig('Average total waiting on transit fdn.pdf',bbox_inches='tight')

# Plot time waiting on weather for positioning during foudation installation
for option in options:
    if waiting_time_positioning_FDN_1[option].isna().sum() > 0:
        waiting_time_positioning_FDN_1.loc[:, option] = np.nan
    if waiting_time_positioning_FDN_12[option].isna().sum() > 0:
        waiting_time_positioning_FDN_12.loc[:, option] = np.nan
    if waiting_time_positioning_FDN_24[option].isna().sum() > 0:
        waiting_time_positioning_FDN_24.loc[:, option] = np.nan
    if waiting_time_positioning_FDN_48[option].isna().sum() > 0:
        waiting_time_positioning_FDN_48.loc[:, option] = np.nan
    if waiting_time_positioning_FDN_72[option].isna().sum() > 0:
        waiting_time_positioning_FDN_72.loc[:, option] = np.nan
    if waiting_time_positioning_FDN_96[option].isna().sum() > 0:
        waiting_time_positioning_FDN_96.loc[:, option] = np.nan
        
waiting_positioning_FDN_WR = waiting_time_positioning_FDN_WR.mean()
waiting_positioning_FDN_1 = waiting_time_positioning_FDN_1.mean()
waiting_positioning_FDN_12 = waiting_time_positioning_FDN_12.mean()
waiting_positioning_FDN_24 = waiting_time_positioning_FDN_24.mean()
waiting_positioning_FDN_48 = waiting_time_positioning_FDN_48.mean()
waiting_positioning_FDN_72 = waiting_time_positioning_FDN_72.mean()
waiting_positioning_FDN_96 = waiting_time_positioning_FDN_96.mean()

plt.figure(figsize=(12,10))
plt.scatter(options, waiting_positioning_FDN_WR, label='Perfect forecast', color='black', s=100)
plt.scatter(options, waiting_positioning_FDN_1, label='1 hour horizon', marker='x', color='green', s=200)
plt.scatter(options, waiting_positioning_FDN_12, label='12 hour horizon', marker='x', color='red', s=200)
plt.scatter(options, waiting_positioning_FDN_24, label='24 hour horizon', marker='x', color='xkcd:gold', s=200)
plt.scatter(options, waiting_positioning_FDN_48, label='48 hour horizon', marker='x', color='xkcd:ocean blue', s=200)
#plt.scatter(options, waiting_positioning_FDN_72, label='72 hour horizon', marker='x', color='xkcd:grape', s=200)
#plt.scatter(options, waiting_positioning_FDN_96, label='96 hour horizon', marker='x', color='xkcd:pumpkin', s=200)

plt.xticks(rotation=45, fontsize=20)
plt.yticks(fontsize=20)
plt.grid(visible=True)
plt.ylabel('Waiting on weather [h]', fontsize= 20)
plt.legend(fontsize= 'xx-large', frameon=True)#, bbox_to_anchor=(1.0, 1.0), loc='upper left')
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.ylim(ymin=0)
plt.savefig('Average total waiting on positioning fdn.pdf',bbox_inches='tight')

# Plot time waiting on weather for jacking during foundation installation
for option in options:
    if waiting_time_jacking_FDN_1[option].isna().sum() > 0:
        waiting_time_jacking_FDN_1.loc[:, option] = np.nan
    if waiting_time_jacking_FDN_12[option].isna().sum() > 0:
        waiting_time_jacking_FDN_12.loc[:, option] = np.nan
    if waiting_time_jacking_FDN_24[option].isna().sum() > 0:
        waiting_time_jacking_FDN_24.loc[:, option] = np.nan
    if waiting_time_jacking_FDN_48[option].isna().sum() > 0:
        waiting_time_jacking_FDN_48.loc[:, option] = np.nan
    if waiting_time_jacking_FDN_72[option].isna().sum() > 0:
        waiting_time_jacking_FDN_72.loc[:, option] = np.nan
    if waiting_time_jacking_FDN_96[option].isna().sum() > 0:
        waiting_time_jacking_FDN_96.loc[:, option] = np.nan

waiting_jacking_FDN_WR = waiting_time_jacking_FDN_WR.mean()
waiting_jacking_FDN_1 = waiting_time_jacking_FDN_1.mean()
waiting_jacking_FDN_12 = waiting_time_jacking_FDN_12.mean()
waiting_jacking_FDN_24 = waiting_time_jacking_FDN_24.mean()
waiting_jacking_FDN_48 = waiting_time_jacking_FDN_48.mean()
waiting_jacking_FDN_72 = waiting_time_jacking_FDN_72.mean()
waiting_jacking_FDN_96 = waiting_time_jacking_FDN_96.mean()

plt.figure(figsize=(12,10))
plt.scatter(options, waiting_jacking_FDN_WR, label='Perfect forecast', color='black', s=100)
plt.scatter(options, waiting_jacking_FDN_1, label='1 hour horizon', marker='x', color='green', s=200)
plt.scatter(options, waiting_jacking_FDN_12, label='12 hour horizon', marker='x', color='red', s=200)
plt.scatter(options, waiting_jacking_FDN_24, label='24 hour horizon', marker='x', color='xkcd:gold', s=200)
plt.scatter(options, waiting_jacking_FDN_48, label='48 hour horizon', marker='x', color='xkcd:ocean blue', s=200)
#plt.scatter(options, waiting_jacking_FDN_72, label='72 hour horizon', marker='x', color='xkcd:grape', s=200)
#plt.scatter(options, waiting_jacking_FDN_96, label='96 hour horizon', marker='x', color='xkcd:pumpkin', s=200)

plt.xticks(rotation=45, fontsize=20)
plt.yticks(fontsize=20)
plt.grid(visible=True)
plt.ylabel('Waiting on weather [h]', fontsize= 20)
plt.legend(fontsize= 'xx-large', frameon=True)#, bbox_to_anchor=(1.0, 1.0), loc='upper left')
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.ylim(ymin=0)
plt.savefig('Average total waiting on jacking fdn.pdf',bbox_inches='tight')

# Plot time waiting on weather for monopile installation during foundation installation
for option in options:
    if waiting_time_mono_1[option].isna().sum() > 0:
        waiting_time_mono_1.loc[:, option] = np.nan
    if waiting_time_mono_12[option].isna().sum() > 0:
        waiting_time_mono_12.loc[:, option] = np.nan
    if waiting_time_mono_24[option].isna().sum() > 0:
        waiting_time_mono_24.loc[:, option] = np.nan
    if waiting_time_mono_48[option].isna().sum() > 0:
        waiting_time_mono_48.loc[:, option] = np.nan
    if waiting_time_mono_72[option].isna().sum() > 0:
        waiting_time_mono_72.loc[:, option] = np.nan
    if waiting_time_mono_96[option].isna().sum() > 0:
        waiting_time_mono_96.loc[:, option] = np.nan

waiting_mono_WR = waiting_time_mono_WR.mean()
waiting_mono_1 = waiting_time_mono_1.mean()
waiting_mono_12 = waiting_time_mono_12.mean()
waiting_mono_24 = waiting_time_mono_24.mean()
waiting_mono_48 = waiting_time_mono_48.mean()
waiting_mono_72 = waiting_time_mono_72.mean()
waiting_mono_96 = waiting_time_mono_96.mean()

plt.figure(figsize=(12,10))
plt.scatter(options, waiting_mono_WR, label='Perfect forecast', color='black', s=100)
plt.scatter(options, waiting_mono_1, label='1 hour horizon', marker='x', color='green', s=200)
plt.scatter(options, waiting_mono_12, label='12 hour horizon', marker='x', color='red', s=200)
plt.scatter(options, waiting_mono_24, label='24 hour horizon', marker='x', color='xkcd:gold', s=200)
plt.scatter(options, waiting_mono_48, label='48 hour horizon', marker='x', color='xkcd:ocean blue', s=200)
#plt.scatter(options, waiting_mono_72, label='72 hour horizon', marker='x', color='xkcd:grape', s=200)
#plt.scatter(options, waiting_mono_96, label='96 hour horizon', marker='x', color='xkcd:pumpkin', s=200)

plt.xticks(rotation=45, fontsize=20)
plt.yticks(fontsize=20)
plt.grid(visible=True)
plt.ylabel('Waiting on weather [h]', fontsize= 20)
plt.legend(fontsize= 'xx-large', frameon=True)#, bbox_to_anchor=(1.0, 1.0), loc='upper left')
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.ylim(ymin=0)
plt.savefig('Average total waiting on monopile.pdf',bbox_inches='tight')

# Plot time waiting on weather for transition piece installation during foundation installation
for option in options:
    if waiting_time_TP_1[option].isna().sum() > 0:
        waiting_time_TP_1.loc[:, option] = np.nan
    if waiting_time_TP_12[option].isna().sum() > 0:
        waiting_time_TP_12.loc[:, option] = np.nan
    if waiting_time_TP_24[option].isna().sum() > 0:
        waiting_time_TP_24.loc[:, option] = np.nan
    if waiting_time_TP_48[option].isna().sum() > 0:
        waiting_time_TP_48.loc[:, option] = np.nan
    if waiting_time_TP_72[option].isna().sum() > 0:
        waiting_time_TP_72.loc[:, option] = np.nan
    if waiting_time_TP_96[option].isna().sum() > 0:
        waiting_time_TP_96.loc[:, option] = np.nan

waiting_TP_WR = waiting_time_TP_WR.mean()
waiting_TP_1 = waiting_time_TP_1.mean()
waiting_TP_12 = waiting_time_TP_12.mean()
waiting_TP_24 = waiting_time_TP_24.mean()
waiting_TP_48 = waiting_time_TP_48.mean()
waiting_TP_72 = waiting_time_TP_72.mean()
waiting_TP_96 = waiting_time_TP_96.mean()

plt.figure(figsize=(12,10))
plt.scatter(options, waiting_TP_WR, label='Perfect forecast', color='black', s=100)
plt.scatter(options, waiting_TP_1, label='1 hour horizon', marker='x', color='green', s=200)
plt.scatter(options, waiting_TP_12, label='12 hour horizon', marker='x', color='red', s=200)
plt.scatter(options, waiting_TP_24, label='24 hour horizon', marker='x', color='xkcd:gold', s=200)
plt.scatter(options, waiting_TP_48, label='48 hour horizon', marker='x', color='xkcd:ocean blue', s=200)
#plt.scatter(options, waiting_TP_72, label='72 hour horizon', marker='x', color='xkcd:grape', s=200)
#plt.scatter(options, waiting_TP_96, label='96 hour horizon', marker='x', color='xkcd:pumpkin', s=200)

plt.xticks(rotation=45, fontsize=20)
plt.yticks(fontsize=20)
plt.grid(visible=True)
plt.ylabel('Waiting on weather [h]', fontsize= 20)
plt.legend(fontsize= 'xx-large', frameon=True)#, bbox_to_anchor=(1.0, 1.0), loc='upper left')
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.ylim(ymin=0)
plt.savefig('Average total waiting on transition piece.pdf',bbox_inches='tight')

#%% Plot waiting times individual operations during turbine installation

# Plot time waiting on weather for transit during turbine installation
for option in options:
    if waiting_time_transit_WTB_1[option].isna().sum() > 0:
        waiting_time_transit_WTB_1.loc[:, option] = np.nan
    if waiting_time_transit_WTB_12[option].isna().sum() > 0:
        waiting_time_transit_WTB_12.loc[:, option] = np.nan
    if waiting_time_transit_WTB_24[option].isna().sum() > 0:
        waiting_time_transit_WTB_24.loc[:, option] = np.nan
    if waiting_time_transit_WTB_48[option].isna().sum() > 0:
        waiting_time_transit_WTB_48.loc[:, option] = np.nan
    if waiting_time_transit_WTB_72[option].isna().sum() > 0:
        waiting_time_transit_WTB_72.loc[:, option] = np.nan
    if waiting_time_transit_WTB_96[option].isna().sum() > 0:
        waiting_time_transit_WTB_96.loc[:, option] = np.nan

waiting_transit_WTB_WR = waiting_time_transit_WTB_WR.mean()
waiting_transit_WTB_1 = waiting_time_transit_WTB_1.mean()
waiting_transit_WTB_12 = waiting_time_transit_WTB_12.mean()
waiting_transit_WTB_24 = waiting_time_transit_WTB_24.mean()
waiting_transit_WTB_48 = waiting_time_transit_WTB_48.mean()
waiting_transit_WTB_72 = waiting_time_transit_WTB_72.mean()
waiting_transit_WTB_96 = waiting_time_transit_WTB_96.mean()

plt.figure(figsize=(12,10))
plt.scatter(options, waiting_transit_WTB_WR, label='Perfect forecast', color='black', s=100)
plt.scatter(options, waiting_transit_WTB_1, label='1 hour horizon', marker='x', color='green', s=200)
plt.scatter(options, waiting_transit_WTB_12, label='12 hour horizon', marker='x', color='red', s=200)
plt.scatter(options, waiting_transit_WTB_24, label='24 hour horizon', marker='x', color='xkcd:gold', s=200)
#plt.scatter(options, waiting_transit_WTB_48, label='48 hour horizon', marker='x', color='xkcd:ocean blue', s=200)
#plt.scatter(options, waiting_transit_WTB_72, label='72 hour horizon', marker='x', color='xkcd:grape', s=200)
#plt.scatter(options, waiting_transit_WTB_96, label='96 hour horizon', marker='x', color='xkcd:pumpkin', s=200)

plt.xticks(rotation=45, fontsize=20)
plt.yticks(fontsize=20)
plt.grid(visible=True)
plt.ylabel('Waiting on weather [h]', fontsize= 20)
plt.legend(fontsize= 'xx-large', frameon=True)#, bbox_to_anchor=(1.0, 1.0), loc='upper left')
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.ylim(ymin=0)
plt.savefig('Average total waiting on transit wtb.pdf',bbox_inches='tight')

# Plot time waiting on weather for positioning during turbine installation
for option in options:
    if waiting_time_positioning_WTB_1[option].isna().sum() > 0:
        waiting_time_positioning_WTB_1.loc[:, option] = np.nan
    if waiting_time_positioning_WTB_12[option].isna().sum() > 0:
        waiting_time_positioning_WTB_12.loc[:, option] = np.nan
    if waiting_time_positioning_WTB_24[option].isna().sum() > 0:
        waiting_time_positioning_WTB_24.loc[:, option] = np.nan
    if waiting_time_positioning_WTB_48[option].isna().sum() > 0:
        waiting_time_positioning_WTB_48.loc[:, option] = np.nan
    if waiting_time_positioning_WTB_72[option].isna().sum() > 0:
        waiting_time_positioning_WTB_72.loc[:, option] = np.nan
    if waiting_time_positioning_WTB_96[option].isna().sum() > 0:
        waiting_time_positioning_WTB_96.loc[:, option] = np.nan
        
waiting_positioning_WTB_WR = waiting_time_positioning_WTB_WR.mean()
waiting_positioning_WTB_1 = waiting_time_positioning_WTB_1.mean()
waiting_positioning_WTB_12 = waiting_time_positioning_WTB_12.mean()
waiting_positioning_WTB_24 = waiting_time_positioning_WTB_24.mean()
waiting_positioning_WTB_48 = waiting_time_positioning_WTB_48.mean()
waiting_positioning_WTB_72 = waiting_time_positioning_WTB_72.mean()
waiting_positioning_WTB_96 = waiting_time_positioning_WTB_96.mean()

plt.figure(figsize=(12,10))
plt.scatter(options, waiting_positioning_WTB_WR, label='Perfect forecast', color='black', s=100)
plt.scatter(options, waiting_positioning_WTB_1, label='1 hour horizon', marker='x', color='green', s=200)
plt.scatter(options, waiting_positioning_WTB_12, label='12 hour horizon', marker='x', color='red', s=200)
plt.scatter(options, waiting_positioning_WTB_24, label='24 hour horizon', marker='x', color='xkcd:gold', s=200)
#plt.scatter(options, waiting_positioning_WTB_48, label='48 hour horizon', marker='x', color='xkcd:ocean blue', s=200)
#plt.scatter(options, waiting_positioning_WTB_72, label='72 hour horizon', marker='x', color='xkcd:grape', s=200)
#plt.scatter(options, waiting_positioning_WTB_96, label='96 hour horizon', marker='x', color='xkcd:pumpkin', s=200)

plt.xticks(rotation=45, fontsize=20)
plt.yticks(fontsize=20)
plt.grid(visible=True)
plt.ylabel('Waiting on weather [h]', fontsize= 20)
plt.legend(fontsize= 'xx-large', frameon=True)#, bbox_to_anchor=(1.0, 1.0), loc='upper left')
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.ylim(ymin=0)
plt.savefig('Average total waiting on positioning wtb.pdf',bbox_inches='tight')

# Plot time waiting on weather for jacking during turbine installation
for option in options:
    if waiting_time_jacking_WTB_1[option].isna().sum() > 0:
        waiting_time_jacking_WTB_1.loc[:, option] = np.nan
    if waiting_time_jacking_WTB_12[option].isna().sum() > 0:
        waiting_time_jacking_WTB_12.loc[:, option] = np.nan
    if waiting_time_jacking_WTB_24[option].isna().sum() > 0:
        waiting_time_jacking_WTB_24.loc[:, option] = np.nan
    if waiting_time_jacking_WTB_48[option].isna().sum() > 0:
        waiting_time_jacking_WTB_48.loc[:, option] = np.nan
    if waiting_time_jacking_WTB_72[option].isna().sum() > 0:
        waiting_time_jacking_WTB_72.loc[:, option] = np.nan
    if waiting_time_jacking_WTB_96[option].isna().sum() > 0:
        waiting_time_jacking_WTB_96.loc[:, option] = np.nan
        
waiting_jacking_WTB_WR = waiting_time_jacking_WTB_WR.mean()
waiting_jacking_WTB_1 = waiting_time_jacking_WTB_1.mean()
waiting_jacking_WTB_12 = waiting_time_jacking_WTB_12.mean()
waiting_jacking_WTB_24 = waiting_time_jacking_WTB_24.mean()
waiting_jacking_WTB_48 = waiting_time_jacking_WTB_48.mean()
waiting_jacking_WTB_72 = waiting_time_jacking_WTB_72.mean()
waiting_jacking_WTB_96 = waiting_time_jacking_WTB_96.mean()

plt.figure(figsize=(12,10))
plt.scatter(options, waiting_jacking_WTB_WR, label='Perfect forecast', color='black', s=100)
plt.scatter(options, waiting_jacking_WTB_1, label='1 hour horizon', marker='x', color='green', s=200)
plt.scatter(options, waiting_jacking_WTB_12, label='12 hour horizon', marker='x', color='red', s=200)
plt.scatter(options, waiting_jacking_WTB_24, label='24 hour horizon', marker='x', color='xkcd:gold', s=200)
#plt.scatter(options, waiting_jacking_WTB_48, label='48 hour horizon', marker='x', color='xkcd:ocean blue', s=200)
#plt.scatter(options, waiting_jacking_WTB_72, label='72 hour horizon', marker='x', color='xkcd:grape', s=200)
#plt.scatter(options, waiting_jacking_WTB_96, label='96 hour horizon', marker='x', color='xkcd:pumpkin', s=200)

plt.xticks(rotation=45, fontsize=20)
plt.yticks(fontsize=20)
plt.grid(visible=True)
plt.ylabel('Waiting on weather [h]', fontsize= 20)
plt.legend(fontsize= 'xx-large', frameon=True)#, bbox_to_anchor=(1.0, 1.0), loc='upper left')
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.ylim(ymin=0)
plt.savefig('Average total waiting on jacking wtb.pdf',bbox_inches='tight')

# Plot time waiting on weather for tower installation during turbine installation
for option in options:
    if waiting_time_tower_1[option].isna().sum() > 0:
        waiting_time_tower_1.loc[:, option] = np.nan
    if waiting_time_tower_12[option].isna().sum() > 0:
        waiting_time_tower_12.loc[:, option] = np.nan
    if waiting_time_tower_24[option].isna().sum() > 0:
        waiting_time_tower_24.loc[:, option] = np.nan
    if waiting_time_tower_48[option].isna().sum() > 0:
        waiting_time_tower_48.loc[:, option] = np.nan
    if waiting_time_tower_72[option].isna().sum() > 0:
        waiting_time_tower_72.loc[:, option] = np.nan
    if waiting_time_tower_96[option].isna().sum() > 0:
        waiting_time_tower_96.loc[:, option] = np.nan
        
waiting_tower_WR = waiting_time_tower_WR.mean()
waiting_tower_1 = waiting_time_tower_1.mean()
waiting_tower_12 = waiting_time_tower_12.mean()
waiting_tower_24 = waiting_time_tower_24.mean()
waiting_tower_48 = waiting_time_tower_48.mean()
waiting_tower_72 = waiting_time_tower_72.mean()
waiting_tower_96 = waiting_time_tower_96.mean()

plt.figure(figsize=(12,10))
plt.scatter(options, waiting_tower_WR, label='Perfect forecast', color='black', s=100)
plt.scatter(options, waiting_tower_1, label='1 hour horizon', marker='x', color='green', s=200)
plt.scatter(options, waiting_tower_12, label='12 hour horizon', marker='x', color='red', s=200)
plt.scatter(options, waiting_tower_24, label='24 hour horizon', marker='x', color='xkcd:gold', s=200)
#plt.scatter(options, waiting_tower_48, label='48 hour horizon', marker='x', color='xkcd:ocean blue', s=200)
#plt.scatter(options, waiting_tower_72, label='72 hour horizon', marker='x', color='xkcd:grape', s=200)
#plt.scatter(options, waiting_tower_96, label='96 hour horizon', marker='x', color='xkcd:pumpkin', s=200)

plt.xticks(rotation=45, fontsize=20)
plt.yticks(fontsize=20)
plt.grid(visible=True)
plt.ylabel('Waiting on weather [h]', fontsize= 20)
plt.legend(fontsize= 'xx-large', frameon=True)#, bbox_to_anchor=(1.0, 1.0), loc='upper left')
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.ylim(ymin=0)
plt.savefig('Average total waiting on tower installation wtb.pdf',bbox_inches='tight')

# Plot time waiting on weather for nacelle installation during turbine installation
for option in options:
    if waiting_time_nacelle_1[option].isna().sum() > 0:
        waiting_time_nacelle_1.loc[:, option] = np.nan
    if waiting_time_nacelle_12[option].isna().sum() > 0:
        waiting_time_nacelle_12.loc[:, option] = np.nan
    if waiting_time_nacelle_24[option].isna().sum() > 0:
        waiting_time_nacelle_24.loc[:, option] = np.nan
    if waiting_time_nacelle_48[option].isna().sum() > 0:
        waiting_time_nacelle_48.loc[:, option] = np.nan
    if waiting_time_nacelle_72[option].isna().sum() > 0:
        waiting_time_nacelle_72.loc[:, option] = np.nan
    if waiting_time_nacelle_96[option].isna().sum() > 0:
        waiting_time_nacelle_96.loc[:, option] = np.nan
        
waiting_nacelle_WR = waiting_time_nacelle_WR.mean()
waiting_nacelle_1 = waiting_time_nacelle_1.mean()
waiting_nacelle_12 = waiting_time_nacelle_12.mean()
waiting_nacelle_24 = waiting_time_nacelle_24.mean()
waiting_nacelle_48 = waiting_time_nacelle_48.mean()
waiting_nacelle_72 = waiting_time_nacelle_72.mean()
waiting_nacelle_96 = waiting_time_nacelle_96.mean()

plt.figure(figsize=(12,10))
plt.scatter(options, waiting_nacelle_WR, label='Perfect forecast', color='black', s=100)
plt.scatter(options, waiting_nacelle_1, label='1 hour horizon', marker='x', color='green', s=200)
plt.scatter(options, waiting_nacelle_12, label='12 hour horizon', marker='x', color='red', s=200)
plt.scatter(options, waiting_nacelle_24, label='24 hour horizon', marker='x', color='xkcd:gold', s=200)
#plt.scatter(options, waiting_nacelle_48, label='48 hour horizon', marker='x', color='xkcd:ocean blue', s=200)
#plt.scatter(options, waiting_nacelle_72, label='72 hour horizon', marker='x', color='xkcd:grape', s=200)
#plt.scatter(options, waiting_nacelle_96, label='96 hour horizon', marker='x', color='xkcd:pumpkin', s=200)

plt.xticks(rotation=45, fontsize=20)
plt.yticks(fontsize=20)
plt.grid(visible=True)
plt.ylabel('Waiting on weather [h]', fontsize= 20)
plt.legend(fontsize= 'xx-large', frameon=True, bbox_to_anchor=(0, 0.8), loc='upper left')
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.ylim(ymin=0)
plt.savefig('Average total waiting on nacelle installation wtb.pdf',bbox_inches='tight')

# Plot time waiting on weather for blade installation during turbine installation
for option in options:
    if waiting_time_blades_1[option].isna().sum() > 0:
        waiting_time_blades_1.loc[:, option] = np.nan
    if waiting_time_blades_12[option].isna().sum() > 0:
        waiting_time_blades_12.loc[:, option] = np.nan
    if waiting_time_blades_24[option].isna().sum() > 0:
        waiting_time_blades_24.loc[:, option] = np.nan
    if waiting_time_blades_48[option].isna().sum() > 0:
        waiting_time_blades_48.loc[:, option] = np.nan
    if waiting_time_blades_72[option].isna().sum() > 0:
        waiting_time_blades_72.loc[:, option] = np.nan
    if waiting_time_blades_96[option].isna().sum() > 0:
        waiting_time_blades_96.loc[:, option] = np.nan
        
waiting_blades_WR = waiting_time_blades_WR.mean()
waiting_blades_1 = waiting_time_blades_1.mean()
waiting_blades_12 = waiting_time_blades_12.mean()
waiting_blades_24 = waiting_time_blades_24.mean()
waiting_blades_48 = waiting_time_blades_48.mean()
waiting_blades_72 = waiting_time_blades_72.mean()
waiting_blades_96 = waiting_time_blades_96.mean()

plt.figure(figsize=(12,10))
plt.scatter(options, waiting_blades_WR, label='Perfect forecast', color='black', s=100)
plt.scatter(options, waiting_blades_1, label='1 hour horizon', marker='x', color='green', s=200)
plt.scatter(options, waiting_blades_12, label='12 hour horizon', marker='x', color='red', s=200)
plt.scatter(options, waiting_blades_24, label='24 hour horizon', marker='x', color='xkcd:gold', s=200)
#plt.scatter(options, waiting_blades_48, label='48 hour horizon', marker='x', color='xkcd:ocean blue', s=200)
#plt.scatter(options, waiting_blades_72, label='72 hour horizon', marker='x', color='xkcd:grape', s=200)
#plt.scatter(options, waiting_blades_96, label='96 hour horizon', marker='x', color='xkcd:pumpkin', s=200)

plt.xticks(rotation=45, fontsize=20)
plt.yticks(fontsize=20)
plt.grid(visible=True)
plt.ylabel('Waiting on weather [h]', fontsize= 20)
plt.legend(fontsize= 'xx-large', frameon=True)#, bbox_to_anchor=(1.0, 1.0), loc='upper left')
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.ylim(ymin=0)
plt.savefig('Average total waiting on blade installation wtb.pdf',bbox_inches='tight')

#%% plot average duration foundation installation per OWT

for option in options:
    if foundations_duration_1[option].isna().sum() > 0:
        foundations_duration_1.loc[:, option] = np.nan
    if foundations_duration_12[option].isna().sum() > 0:
        foundations_duration_12.loc[:, option] = np.nan
    if foundations_duration_24[option].isna().sum() > 0:
        foundations_duration_24.loc[:, option] = np.nan
    if foundations_duration_48[option].isna().sum() > 0:
        foundations_duration_48.loc[:, option] = np.nan
    if foundations_duration_72[option].isna().sum() > 0:
        foundations_duration_72.loc[:, option] = np.nan
    if foundations_duration_96[option].isna().sum() > 0:
        foundations_duration_96.loc[:, option] = np.nan

average_duration_per_OWT_FDN_calm_weather = results_calm_weather.T_FDN/720
average_duration_per_OWT_FDN_WR = foundations_duration_WR.mean()/720
average_duration_per_OWT_FDN_1 = foundations_duration_1.mean()/720
average_duration_per_OWT_FDN_12 = foundations_duration_12.mean()/720
average_duration_per_OWT_FDN_24 = foundations_duration_24.mean()/720
average_duration_per_OWT_FDN_48 = foundations_duration_48.mean()/720
average_duration_per_OWT_FDN_72 = foundations_duration_72.mean()/720
average_duration_per_OWT_FDN_96 = foundations_duration_96.mean()/720

plt.figure(figsize=(12,10))
plt.scatter(options, average_duration_per_OWT_FDN_WR, label='Perfect forecast', color='black', s=100)
plt.scatter(options, average_duration_per_OWT_FDN_1, label='1 hour horizon', marker='x', color='green', s=200)
plt.scatter(options, average_duration_per_OWT_FDN_12, label='12 hour horizon', marker='x', color='red', s=200)
plt.scatter(options, average_duration_per_OWT_FDN_24, label='24 hour horizon', marker='x', color='xkcd:gold', s=200)
plt.scatter(options, average_duration_per_OWT_FDN_48, label='48 hour horizon', marker='x', color='xkcd:ocean blue', s=200)
#plt.scatter(options, average_duration_per_OWT_FDN_72, label='72 hour horizon', marker='x', color='xkcd:grape', s=200)
#plt.scatter(options, average_duration_per_OWT_FDN_96, label='96 hour horizon', marker='x', color='xkcd:pumpkin', s=200)

average_duration_per_OWT_FDN_calm_weather.plot(kind= 'bar', fontsize = 20 , grid = True, rot = 45, label='Calm weather')

plt.xticks(rotation=45, fontsize=20)
plt.yticks(fontsize=20)
plt.grid(visible=True)
plt.ylabel('Average duration foundation installation [days/assemby]', fontsize= 20)
plt.legend(fontsize= 'xx-large', frameon=True)#, bbox_to_anchor=(1.0, 1.0), loc='upper left')
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
#plt.ylim((0,4000))
plt.savefig('Average duration of foundation installation per OWT.pdf', bbox_inches= 'tight')

#%% plot average duration turbine installation per OWT

for option in options:
    if turbines_duration_1[option].isna().sum() > 0:
        turbines_duration_1.loc[:, option] = np.nan
    if turbines_duration_12[option].isna().sum() > 0:
        turbines_duration_12.loc[:, option] = np.nan
    if turbines_duration_24[option].isna().sum() > 0:
        turbines_duration_24.loc[:, option] = np.nan
    if turbines_duration_48[option].isna().sum() > 0:
        turbines_duration_48.loc[:, option] = np.nan
    if turbines_duration_72[option].isna().sum() > 0:
        turbines_duration_72.loc[:, option] = np.nan
    if turbines_duration_96[option].isna().sum() > 0:
        turbines_duration_96.loc[:, option] = np.nan

average_duration_per_OWT_WTB_calm_weather = results_calm_weather.T_WTB/720
average_duration_per_OWT_WTB_WR = turbines_duration_WR.mean()/720
average_duration_per_OWT_WTB_1 = turbines_duration_1.mean()/720
average_duration_per_OWT_WTB_12 = turbines_duration_12.mean()/720
average_duration_per_OWT_WTB_24 = turbines_duration_24.mean()/720
average_duration_per_OWT_WTB_48 = turbines_duration_48.mean()/720
average_duration_per_OWT_WTB_72 = turbines_duration_72.mean()/720
average_duration_per_OWT_WTB_96 = turbines_duration_96.mean()/720

plt.figure(figsize=(12,10))
plt.scatter(options, average_duration_per_OWT_WTB_WR, label='Perfect forecast', color='black', s=100)
plt.scatter(options, average_duration_per_OWT_WTB_1, label='1 hour horizon', marker='x', color='green', s=200)
plt.scatter(options, average_duration_per_OWT_WTB_12, label='12 hour horizon', marker='x', color='red', s=200)
plt.scatter(options, average_duration_per_OWT_WTB_24, label='24 hour horizon', marker='x', color='xkcd:gold', s=200)
#plt.scatter(options, average_duration_per_OWT_WTB_48, label='48 hour horizon', marker='x', color='xkcd:ocean blue', s=200)
#plt.scatter(options, average_duration_per_OWT_WTB_72, label='72 hour horizon', marker='x', color='xkcd:grape', s=200)
#plt.scatter(options, average_duration_per_OWT_WTB_96, label='96 hour horizon', marker='x', color='xkcd:pumpkin', s=200)

average_duration_per_OWT_WTB_calm_weather.plot(kind= 'bar', fontsize = 20 , grid = True, rot = 45, label='Calm weather')

plt.xticks(rotation=45, fontsize=20)
plt.yticks(fontsize=20)
plt.grid(visible=True)
plt.ylabel('Average duration turbine installation [days/assemby]', fontsize= 20)
plt.legend(fontsize= 'xx-large', frameon=True, bbox_to_anchor=(1.0, 1.0), loc='upper left')
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
#plt.ylim((0,4000))
plt.savefig('Average duration of turbine installation per OWT.pdf', bbox_inches= 'tight')

#%% plot average evaluation metrics

# plot average True Positive predictions
mean_overall_metrics_TP_1 = overall_metrics_TP_1.mean()
mean_overall_metrics_TP_12 = overall_metrics_TP_12.mean()
mean_overall_metrics_TP_24 = overall_metrics_TP_24.mean()
mean_overall_metrics_TP_48 = overall_metrics_TP_48.mean()
mean_overall_metrics_TP_72 = overall_metrics_TP_72.mean()
mean_overall_metrics_TP_96 = overall_metrics_TP_96.mean()
TP_when_finished = mean_overall_metrics_TP_1

plt.figure(figsize=(15,10))
plt.scatter(options, mean_overall_metrics_TP_1, label='1 hour horizon', marker='x', color='green', s=200)
plt.scatter(options, mean_overall_metrics_TP_12, label='12 hour horizon', marker='x', color='red', s=200)
plt.scatter(options, mean_overall_metrics_TP_24, label='24 hour horizon', marker='x', color='xkcd:gold', s=200)
plt.scatter(options, mean_overall_metrics_TP_48, label='48 hour horizon', marker='x', color='xkcd:ocean blue', s=200)
plt.scatter(options, mean_overall_metrics_TP_72, label='72 hour horizon', marker='x', color='xkcd:grape', s=200)
plt.scatter(options, mean_overall_metrics_TP_96, label='96 hour horizon', marker='x', color='xkcd:pumpkin', s=200)
plt.scatter(options, TP_when_finished, label='When finished', color='black', s=100)

plt.xticks(rotation=45, fontsize=20)
plt.yticks(fontsize=20)
plt.grid(visible=True)
plt.ylabel('True Positive predictions', fontsize= 20)
plt.legend(fontsize= 'xx-large', frameon=True, bbox_to_anchor=(1.0, 1.0), loc='upper left')
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.ylim(ymin=0)
plt.savefig('Average number of True Positive prediction.pdf', bbox_inches= 'tight')

# plot average False Positive predictions
mean_overall_metrics_FP_1 = overall_metrics_FP_1.mean()
mean_overall_metrics_FP_12 = overall_metrics_FP_12.mean()
mean_overall_metrics_FP_24 = overall_metrics_FP_24.mean()
mean_overall_metrics_FP_48 = overall_metrics_FP_48.mean()
mean_overall_metrics_FP_72 = overall_metrics_FP_72.mean()
mean_overall_metrics_FP_96 = overall_metrics_FP_96.mean()

plt.figure(figsize=(15,10))
plt.scatter(options, mean_overall_metrics_FP_1, label='1 hour horizon', marker='x', color='green', s=200)
plt.scatter(options, mean_overall_metrics_FP_12, label='12 hour horizon', marker='x', color='red', s=200)
plt.scatter(options, mean_overall_metrics_FP_24, label='24 hour horizon', marker='x', color='xkcd:gold', s=200)
plt.scatter(options, mean_overall_metrics_FP_48, label='48 hour horizon', marker='x', color='xkcd:ocean blue', s=200)
plt.scatter(options, mean_overall_metrics_FP_72, label='72 hour horizon', marker='x', color='xkcd:grape', s=200)
plt.scatter(options, mean_overall_metrics_FP_96, label='96 hour horizon', marker='x', color='xkcd:pumpkin', s=200)

plt.xticks(rotation=45, fontsize=20)
plt.yticks(fontsize=20)
plt.grid(visible=True)
plt.ylabel('False Positive predictions', fontsize= 20)
plt.legend(fontsize= 'xx-large', frameon=True, bbox_to_anchor=(1.0, 1.0), loc='upper left')
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.ylim(ymin=0)
plt.savefig('Average number of False Positive prediction.pdf', bbox_inches= 'tight')

# plot average True Negative predictions
mean_overall_metrics_TN_1 = overall_metrics_TN_1.mean()
mean_overall_metrics_TN_12 = overall_metrics_TN_12.mean()
mean_overall_metrics_TN_24 = overall_metrics_TN_24.mean()
mean_overall_metrics_TN_48 = overall_metrics_TN_48.mean()
mean_overall_metrics_TN_72 = overall_metrics_TN_72.mean()
mean_overall_metrics_TN_96 = overall_metrics_TN_96.mean()

plt.figure(figsize=(15,10))
plt.scatter(options, mean_overall_metrics_TN_1, label='1 hour horizon', marker='x', color='green', s=200)
plt.scatter(options, mean_overall_metrics_TN_12, label='12 hour horizon', marker='x', color='red', s=200)
plt.scatter(options, mean_overall_metrics_TN_24, label='24 hour horizon', marker='x', color='xkcd:gold', s=200)
plt.scatter(options, mean_overall_metrics_TN_48, label='48 hour horizon', marker='x', color='xkcd:ocean blue', s=200)
plt.scatter(options, mean_overall_metrics_TN_72, label='72 hour horizon', marker='x', color='xkcd:grape', s=200)
plt.scatter(options, mean_overall_metrics_TN_96, label='96 hour horizon', marker='x', color='xkcd:pumpkin', s=200)

plt.xticks(rotation=45, fontsize=20)
plt.yticks(fontsize=20)
plt.grid(visible=True)
plt.ylabel('True Negative predictions', fontsize= 20)
plt.legend(fontsize= 'xx-large', frameon=True, bbox_to_anchor=(1.0, 1.0), loc='upper left')
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.ylim(ymin=0)
plt.savefig('Average number of True Negative prediction.pdf', bbox_inches= 'tight')

# plot average True Negative predictions
mean_overall_metrics_FN_1 = overall_metrics_FN_1.mean()
mean_overall_metrics_FN_12 = overall_metrics_FN_12.mean()
mean_overall_metrics_FN_24 = overall_metrics_FN_24.mean()
mean_overall_metrics_FN_48 = overall_metrics_FN_48.mean()
mean_overall_metrics_FN_72 = overall_metrics_FN_72.mean()
mean_overall_metrics_FN_96 = overall_metrics_FN_96.mean()

plt.figure(figsize=(15,10))
plt.scatter(options, mean_overall_metrics_FN_1, label='1 hour horizon', marker='x', color='green', s=200)
plt.scatter(options, mean_overall_metrics_FN_12, label='12 hour horizon', marker='x', color='red', s=200)
plt.scatter(options, mean_overall_metrics_FN_24, label='24 hour horizon', marker='x', color='xkcd:gold', s=200)
plt.scatter(options, mean_overall_metrics_FN_48, label='48 hour horizon', marker='x', color='xkcd:ocean blue', s=200)
plt.scatter(options, mean_overall_metrics_FN_72, label='72 hour horizon', marker='x', color='xkcd:grape', s=200)
plt.scatter(options, mean_overall_metrics_FN_96, label='96 hour horizon', marker='x', color='xkcd:pumpkin', s=200)

plt.xticks(rotation=45, fontsize=20)
plt.yticks(fontsize=20)
plt.grid(visible=True)
plt.ylabel('False Negative predictions', fontsize= 20)
plt.legend(fontsize= 'xx-large', frameon=True, bbox_to_anchor=(1.0, 1.0), loc='upper left')
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.ylim(ymin=0)
plt.savefig('Average number of False Negative prediction.pdf', bbox_inches= 'tight')

#%% plot mean total penalty cost

for option in options:
    if total_penalty_cost_1[option].isna().sum() > 0:
        total_penalty_cost_1.loc[:, option] = np.nan
    if total_penalty_cost_12[option].isna().sum() > 0:
        total_penalty_cost_12.loc[:, option] = np.nan
    if total_penalty_cost_24[option].isna().sum() > 0:
        total_penalty_cost_24.loc[:, option] = np.nan
    if total_penalty_cost_48[option].isna().sum() > 0:
        total_penalty_cost_48.loc[:, option] = np.nan
    if total_penalty_cost_72[option].isna().sum() > 0:
        total_penalty_cost_72.loc[:, option] = np.nan
    if total_penalty_cost_96[option].isna().sum() > 0:
        total_penalty_cost_96.loc[:, option] = np.nan
        
mean_total_penalty_cost_1 = total_penalty_cost_1.copy().mean()
mean_total_penalty_cost_12 = total_penalty_cost_12.copy().mean()
mean_total_penalty_cost_24 = total_penalty_cost_24.copy().mean()
mean_total_penalty_cost_48 = total_penalty_cost_48.copy().mean()
mean_total_penalty_cost_72 = total_penalty_cost_72.copy().mean()        
mean_total_penalty_cost_96 = total_penalty_cost_96.copy().mean()

plt.figure(figsize=(12,10))
plt.scatter(options, mean_total_penalty_cost_1, label='1 hour horizon', marker='x', color='green', s=200)
plt.scatter(options, mean_total_penalty_cost_12, label='12 hour horizon', marker='x', color='red', s=200)
plt.scatter(options, mean_total_penalty_cost_24, label='24 hour horizon', marker='x', color='xkcd:gold', s=200)
#plt.scatter(options, mean_total_penalty_cost_48, label='48 hour horizon', marker='x', color='xkcd:ocean blue', s=200)
#plt.scatter(options, mean_total_penalty_cost_72, label='72 hour horizon', marker='x', color='xkcd:grape', s=200)
#plt.scatter(options, mean_total_penalty_cost_96, label='96 hour horizon', marker='x', color='xkcd:pumpkin', s=200)

plt.xticks(rotation=45, fontsize=20)
plt.yticks(fontsize=20)
plt.grid(visible=True)
plt.ylabel('Mean total False Positive penalty cost [k€]', fontsize= 20)
plt.legend(fontsize= 'xx-large', frameon=True)#, bbox_to_anchor=(1, 1), loc='upper left')
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.ylim(ymin=0)

plt.savefig('Mean total False Positive penalty cost.pdf',bbox_inches='tight')

#%% plot penalty cost percentage of total cost

percentage_penalty_cost_1 = mean_total_penalty_cost_1/mean_total_cost_1*100
percentage_penalty_cost_12 = mean_total_penalty_cost_12/mean_total_cost_12*100
percentage_penalty_cost_24 = mean_total_penalty_cost_24/mean_total_cost_24*100
percentage_penalty_cost_48 = mean_total_penalty_cost_48/mean_total_cost_48*100
percentage_penalty_cost_72 = mean_total_penalty_cost_72/mean_total_cost_72*100
percentage_penalty_cost_96 = mean_total_penalty_cost_96/mean_total_cost_96*100

plt.figure(figsize=(12,10))
plt.scatter(options, percentage_penalty_cost_1, label='1 hour horizon', marker='x', color='green', s=200)
plt.scatter(options, percentage_penalty_cost_12, label='12 hour horizon', marker='x', color='red', s=200)
plt.scatter(options, percentage_penalty_cost_24, label='24 hour horizon', marker='x', color='xkcd:gold', s=200)
#plt.scatter(options, percentage_penalty_cost_48, label='48 hour horizon', marker='x', color='xkcd:ocean blue', s=200)
#plt.scatter(options, percentage_penalty_cost_72, label='72 hour horizon', marker='x', color='xkcd:grape', s=200)
#plt.scatter(options, percentage_penalty_cost_96, label='96 hour horizon', marker='x', color='xkcd:pumpkin', s=200)

plt.xticks(rotation=45, fontsize=20)
plt.yticks(fontsize=20)
plt.grid(visible=True)
plt.ylabel('Percentage False Positive penalty cost [%]', fontsize= 20)
plt.legend(fontsize= 'xx-large', frameon=True)#, bbox_to_anchor=(1, 1), loc='upper left')
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.ylim(ymin=0)

plt.savefig('Percentage False Positive penalty cost.pdf',bbox_inches='tight')
