#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun 25 April 2021

Built On Python version 3.7

# AIM4KD - Agnostic Interpretable Machine Learning for Knowledge Discovery

# Author: Dr. Nectarios Costadopoulos  

# Repository: github.com/CyberNaut-AU/AIM4KD

"""

# ===========================================================================================
# AIM4KD AGNOSTIC INTEPRETABLE MACHINE LEARNING 4 KNOWLEDGE DISCOVERY STAGE 1 - PREPROCESSING
# ===========================================================================================



# IMPORTS FROM IMPORTANT LIBRARIES

import csv

import os

import numpy as np

from numpy import genfromtxt

from scipy import stats

import time # measure time

import re  #import regular expression library



def rolling_window(a, window):
    """Documentation: function used to create a rolling window array for mean and sd calculations."""
    import numpy as np
    pad = np.ones(len(a.shape), dtype=np.int32)
    pad[-1] = window-1
    pad = list(zip(pad, np.zeros(len(a.shape), dtype=np.int32)))
    a = np.pad(a, pad,mode='reflect')
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)



# Count number of items Arrays
def get_number_of_elementsinArray(list):
    """Documentation:  get the number of elements in Array."""
    count = 0
    for element in list:
        count += 1
    return count




# DECLARATIONS
# Create a participant dictionary to store 32 participants x 40 Trials
participants = {}

# Get the directory of the current script
path= os.path.dirname(os.path.abspath(__file__)) + "/"

# KEY DIRECTORIES
raw_datasets_dir = "DEAP_Raw_P1_Sensor_Voltages/"
output_dir = "DEAP_Preprocessed_Trials_P1/"

# DEAP Dataset Participant Valence and Arousal mapped to Stress Labels
labelsCSVfile = "DEAP_Trial_Labels_P1.csv"


ExperimentName = "PROCESSED_SIGNALS_GlobalVariables"

# Define number of Trials per Participant default 40
NumTrials = 40

# Number of Participants to Processs, set to 1 for DEAP participant/subject 1
NumParticipants = 1




# IMPORT SIGNAL PROCESSING FUNCTIONS FROM biosppy and heartpy tools

# Project developed in 2021 on version biosppy-0.6 on python version 3.7

# PIP Install biosppy read more at https://pypi.org/project/biosppy/ 
# PIP heartpy  read more at https://pypi.org/project/heartpy/

# Get Processed Signals
from biosppy.signals import bvp
from biosppy.signals import resp
from biosppy.signals import eda

# Process heart signals for HRV
import heartpy as hp

# Signal Processing Functions
from scipy import signal

# Declare Globals Inside Functions, only first time, check first time ever
#global RulesArray

# Define Fresh RulesArray for storing the rules used in J48_genKDfunction
RulesArray = []

# Start TIMER
start = time.time()


# =============================================================================
# LOAD RAW DATA FROM DEAP RAW VOLTAGE AND Z Values
# =============================================================================


# Using Numpy Arrays Only

# start loop through all participants and adjust for zero
for participant in range(1,NumParticipants +1):
    
    # Create an numpy array to store 40 trials for each participant containing 8064 rows and 8 channels
    # Note created 41 first dimension for because trials start from 1 -4 , zero dimension will be kept empty
    participant_trials_array = np.zeros((41,8064,8))
    
    #Generate Participant String to dispaly double digit integer
    
    participant_str = "s" + "{0:0=2d}".format(participant)
    
    # start loop through all trials and adjust for zero
    for trial in range(1,NumTrials +1):
        
        experiment_trial = participant_str + "_Trial" + str(trial) + "_RawChan37_40_voltageALL.csv"
        print (experiment_trial)
        
        csvfile = path + raw_datasets_dir + experiment_trial
        
        # Load into my_data entire CSV file 8064x8
        my_data = genfromtxt(csvfile, delimiter=',')
        
        # load current trial into 3 dimensional array 40 trials x 8064 x 8 channels
        participant_trials_array[trial] =  my_data
        
    #Load participant array into participant dictionary 1-32
    print ("finished loading participant into array" + participant_str)    
    
    participants[participant] = participant_trials_array


# =============================================================================
# # LOAD 2 and 3 CLASS LABELS FROM TRIAL LABELS CSV
# =============================================================================

labelsCSVfilepath = path + raw_datasets_dir + labelsCSVfile


participant_trial_labels_array = genfromtxt(labelsCSVfilepath, delimiter=',', encoding='UTF-8', dtype=None)


# =============================================================================
# LOOP THROUGH ALL THE PARTICIPANTS USE GLOBAL LIMIT NumParticipants
#                       PREPROCESSING & CSV/ARFF CREATION
# =============================================================================
    
# Define Participant and Trial Variables

    
for current_participant in range(1,NumParticipants +1):    
    
    # Calculate Stats for GSR, RESP, PLETH, TEMP
    
    #create 1D single participant array to load GSR, RESP, PLETH, TEMP
    
    participant_GSR_single_trial_array = np.zeros((8064,1)) 
    
    participant_RESP_single_trial_array = np.zeros((8064,1))
    
    participant_PLETH_single_trial_array = np.zeros((8064,1))
    
    participant_TEMP_single_trial_array = np.zeros((8064,1))
    
    
    # CREATE MIXED NUMPY ARRAY for participant matrix 41 COLUMNS and 132 ROWS 
    
    participant_all_fields_matrix = np.zeros((41,132)).astype(object)
    
    # Load Headers
    participant_all_fields_matrix[0][0] = "Participant ID"
    participant_all_fields_matrix[0][1] = "Trial"
    
    # GSR Skin Conductance (mSiemens)
    participant_all_fields_matrix[0][2] = "GSR_Mean"
    participant_all_fields_matrix[0][3] = "GSR_SD"
    participant_all_fields_matrix[0][4] = "GSR_Max"
    participant_all_fields_matrix[0][5] = "GSR_Min"
    participant_all_fields_matrix[0][6] = "GSR_Med"
    participant_all_fields_matrix[0][7] = "GSR_Mod"
    
    # RESP Breaths per minute (Bpm)
    participant_all_fields_matrix[0][8] = "RESP_Mean"
    participant_all_fields_matrix[0][9] = "RESP_SD"
    participant_all_fields_matrix[0][10] = "RESP_Max"
    participant_all_fields_matrix[0][11] = "RESP_Min"
    participant_all_fields_matrix[0][12] = "RESP_Med"
    participant_all_fields_matrix[0][13] = "RESP_Mod"
    
    # PLETH ->HeartRate(bpm)
    participant_all_fields_matrix[0][14] = "HR_Mean"
    participant_all_fields_matrix[0][15] = "HR_SD"
    participant_all_fields_matrix[0][16] = "HR_Max"
    participant_all_fields_matrix[0][17] = "HR_Min"
    participant_all_fields_matrix[0][18] = "HR_Med"
    participant_all_fields_matrix[0][19] = "HR_Mod"
    
    # PLETH -> HRV_SDNN (sdnn)
    participant_all_fields_matrix[0][20] = "HRV_SDNN"
  
    
    # PLETH -> HRV_rMSSD (rMSSD)
    participant_all_fields_matrix[0][21] = "HRV_rMSSD"
 
    
    # TEMP KEEP USING Z-VALUES no sd
    participant_all_fields_matrix[0][22] = "TEMP_Mean"
    participant_all_fields_matrix[0][23] = "TEMP_Max"
    participant_all_fields_matrix[0][24] = "TEMP_Min"
    participant_all_fields_matrix[0][25] = "TEMP_Med"
    participant_all_fields_matrix[0][26] = "TEMP_Mod"
    
    # Values of Moving Standard Deviation MIN/MAX and physiological pairs
    participant_all_fields_matrix[0][27] = "GSR_MSD_Max"
    participant_all_fields_matrix[0][28] = "GSR_MSDMax_RESP"
    participant_all_fields_matrix[0][29] = "GSR_MSDMax_HR"
    participant_all_fields_matrix[0][30] = "GSR_MSDMax_TEMP"
        
    participant_all_fields_matrix[0][31] = "GSR_MSD_Min"
    participant_all_fields_matrix[0][32] = "GSR_MSDMin_RESP"
    participant_all_fields_matrix[0][33] = "GSR_MSDMin_HR"
    participant_all_fields_matrix[0][34] = "GSR_MSDMin_TEMP"
    
    participant_all_fields_matrix[0][35] = "RESP_MSD_Max"
    participant_all_fields_matrix[0][36] = "RESP_MSDMax_GSR"
    participant_all_fields_matrix[0][37] = "RESP_MSDMax_HR"
    participant_all_fields_matrix[0][38] = "RESP_MSDMax_TEMP"
    
    participant_all_fields_matrix[0][39] = "RESP_MSD_Min"
    participant_all_fields_matrix[0][40] = "RESP_MSDMin_GSR"
    participant_all_fields_matrix[0][41] = "RESP_MSDMin_HR"
    participant_all_fields_matrix[0][42] = "RESP_MSDMin_TEMP"
    
    participant_all_fields_matrix[0][43] = "HR_MSD_Max"
    participant_all_fields_matrix[0][44] = "HR_MSDMax_GSR"
    participant_all_fields_matrix[0][45] = "HR_MSDMax_RESP"
    participant_all_fields_matrix[0][46] = "HR_MSDMax_TEMP"
        
    participant_all_fields_matrix[0][47] = "HR_MSD_Min"
    participant_all_fields_matrix[0][48] = "HR_MSDMin_GSR"
    participant_all_fields_matrix[0][49] = "HR_MSDMin_RESP"
    participant_all_fields_matrix[0][50] = "HR_MSDMin_TEMP"
        
    participant_all_fields_matrix[0][51] = "TEMP_MSD_Max"
    participant_all_fields_matrix[0][52] = "TEMP_MSD_Min"
    
    # Values of Ratios Max
    participant_all_fields_matrix[0][53] = "Ratio_G_R_Max"
    participant_all_fields_matrix[0][54] = "Ratio_G_HR_Max"
    participant_all_fields_matrix[0][55] = "Ratio_G_T_Max"
    participant_all_fields_matrix[0][56] = "Ratio_R_G_Max"
    participant_all_fields_matrix[0][57] = "Ratio_R_HR_Max"
    participant_all_fields_matrix[0][58] = "Ratio_R_T_Max"
    participant_all_fields_matrix[0][59] = "Ratio_HR_G_Max"
    participant_all_fields_matrix[0][60] = "Ratio_HR_R_Max"
    participant_all_fields_matrix[0][61] = "Ratio_HR_T_Max"
    participant_all_fields_matrix[0][62] = "Ratio_T_G_Max"
    participant_all_fields_matrix[0][63] = "Ratio_T_R_Max"
    participant_all_fields_matrix[0][64] = "Ratio_T_HR_Max"
    
    # Values of Ratios Min
    participant_all_fields_matrix[0][65] = "Ratio_G_R_Min"
    participant_all_fields_matrix[0][66] = "Ratio_G_HR_Min"
    participant_all_fields_matrix[0][67] = "Ratio_G_T_Min"
    participant_all_fields_matrix[0][68] = "Ratio_R_G_Min"
    participant_all_fields_matrix[0][69] = "Ratio_R_HR_Min"
    participant_all_fields_matrix[0][70] = "Ratio_R_T_Min"
    participant_all_fields_matrix[0][71] = "Ratio_HR_G_Min"
    participant_all_fields_matrix[0][72] = "Ratio_HR_R_Min"
    participant_all_fields_matrix[0][73] = "Ratio_HR_T_Min"
    participant_all_fields_matrix[0][74] = "Ratio_T_G_Min"
    participant_all_fields_matrix[0][75] = "Ratio_T_R_Min"
    participant_all_fields_matrix[0][76] = "Ratio_T_HR_Min"
    
    # Physiological Max/Min Pairs
    participant_all_fields_matrix[0][77] = "GSRMax_Respiration"
    participant_all_fields_matrix[0][78] = "GSRMax_HR"
    participant_all_fields_matrix[0][79] = "GSRMax_Temp"
    participant_all_fields_matrix[0][80] = "GSRMin_Respiration"
    participant_all_fields_matrix[0][81] = "GSRMin_HR"
    participant_all_fields_matrix[0][82] = "GSRMin_Temp"
    
    participant_all_fields_matrix[0][83] = "RESPMax_GSR"
    participant_all_fields_matrix[0][84] = "RESPMax_HR"
    participant_all_fields_matrix[0][85] = "RESPMax_Temp"
    participant_all_fields_matrix[0][86] = "RESPMin_GSR"
    participant_all_fields_matrix[0][87] = "RESPMin_HR"
    participant_all_fields_matrix[0][88] = "RESPMin_Temp"
    
    participant_all_fields_matrix[0][89] = "HRMax_GSR"
    participant_all_fields_matrix[0][90] = "HRMax_RESP"
    participant_all_fields_matrix[0][91] = "HRMax_Temp"
    participant_all_fields_matrix[0][92] = "HRMin_GSR"
    participant_all_fields_matrix[0][93] = "HRMin_RESP"
    participant_all_fields_matrix[0][94] = "HRMin_Temp"
    
    participant_all_fields_matrix[0][95] = "TEMPMax_GSR"
    participant_all_fields_matrix[0][96] = "TEMPMax_RESP"
    participant_all_fields_matrix[0][97] = "TEMPMax_HR"
    participant_all_fields_matrix[0][98] = "TEMPMin_GSR"
    participant_all_fields_matrix[0][99] = "TEMPMin_RESP"
    participant_all_fields_matrix[0][100] = "TEMPMin_HR"
    
    # Global to Local Normative Values of Attributes
    # GSR Skin Conductance Global Reference Nominal Values (Below,Above, Equal)
    participant_all_fields_matrix[0][101] = "GSR_Mean_GlobalRef"
    participant_all_fields_matrix[0][102] = "GSR_SD_GlobalRef"
    participant_all_fields_matrix[0][103] = "GSR_Max_GlobalRef"
    participant_all_fields_matrix[0][104] = "GSR_Min_GlobalRef"
    
    # RESP Global Reference Nominal Values (Below,Above, Equal)
    participant_all_fields_matrix[0][105] = "RESP_Mean_GlobalRef"
    participant_all_fields_matrix[0][106] = "RESP_SD_GlobalRef"
    participant_all_fields_matrix[0][107] = "RESP_Max_GlobalRef"
    participant_all_fields_matrix[0][108] = "RESP_Min_GlobalRef"
    
    # PLETH Global Reference Nominal Values (Below,Above, Equal)
    participant_all_fields_matrix[0][109] = "HR_Mean_GlobalRef"
    participant_all_fields_matrix[0][110] = "HR_SD_GlobalRef"
    participant_all_fields_matrix[0][111] = "HR_Max_GlobalRef"
    participant_all_fields_matrix[0][112] = "HR_Min_GlobalRef"
    
     # TEMP Global Reference Nominal Values (Below,Above, Equal)
    participant_all_fields_matrix[0][113] = "TEMP_Mean_GlobalRef"
    participant_all_fields_matrix[0][114] = "TEMP_Max_GlobalRef"
    participant_all_fields_matrix[0][115] = "TEMP_Min_GlobalRef"

    #  Global to Local ZScore for MEAN/MAX/MIN for each 4 channel
    # GSR Skin Conductance Global Reference Zscore Values 
    participant_all_fields_matrix[0][116] = "GSR_Mean_Global_ZScore"
    participant_all_fields_matrix[0][117] = "GSR_SD_GlobalRef_ZScore"
    participant_all_fields_matrix[0][118] = "GSR_Max_GlobalRef_ZScore"
    participant_all_fields_matrix[0][119] = "GSR_Min_GlobalRef_ZScore"
    
    # RESP Global Reference Zscore Values 
    participant_all_fields_matrix[0][120] = "RESP_Mean_GlobalRef_ZScore"
    participant_all_fields_matrix[0][121] = "RESP_SD_GlobalRef_ZScore"
    participant_all_fields_matrix[0][122] = "RESP_Max_GlobalRef_ZScore"
    participant_all_fields_matrix[0][123] = "RESP_Min_GlobalRef_ZScore"
    
    # PLETH Global Reference Zscore Values 
    participant_all_fields_matrix[0][124] = "HR_Mean_GlobalRef_ZScore"
    participant_all_fields_matrix[0][125] = "HR_SD_GlobalRef_ZScore"
    participant_all_fields_matrix[0][126] = "HR_Max_GlobalRef_ZScore"
    participant_all_fields_matrix[0][127] = "HR_Min_GlobalRef_ZScore"
    
     # TEMP Global Reference Zscore Values 
    participant_all_fields_matrix[0][128] = "TEMP_Mean_GlobalRef_ZScore"
    participant_all_fields_matrix[0][129] = "TEMP_Max_GlobalRef_ZScore"
    participant_all_fields_matrix[0][130] = "TEMP_Min_GlobalRef_ZScore"
    
    participant_all_fields_matrix[0][131] = "TRIAL_2class_stress_label"
    
     
    
    
    # =============================================================================
    # LOAD TRIALS INTO CURRENT PARTICIPANT LOOP 40 TRIALS 
    # USE RAW VOLTAGE FOR GSR/PLETH/RESP and Z for TEMP
    # Loop through all trials for currrent participant
    # =============================================================================
    
    
    # Round All Statistical Values for the Matrix
    round_dec = 0
    
    for current_trial in range(1,NumTrials +1):
        # Load the data from the participants array for each trial fields PARTICIPANT TRIAL ROW COLUMN
        # Extract RAW Voltage values from column 0 GSR,2 RESP,4 PLETH
        # Extract ZZ values from column 7 TEMP
        
        # CREATE MIXED NUMPY ARRAY for Trial Physiological signal matrix 61 ROWS and 16 COLUMNS 
        # 4 columns for Column 0 - 3 signals, Z values 4-7, 8-16 ratios
        physiological_trial_signal_matrix = np.zeros((61,20)).astype(object)
        
        #Load Array Headers
        # Signals
        physiological_trial_signal_matrix[0][0] = "GSR_Signal_mSiemens"
        physiological_trial_signal_matrix[0][1] = "RESP_Signal_Bpm"
        physiological_trial_signal_matrix[0][2] = "HR_Signal_bpm"
        physiological_trial_signal_matrix[0][3] = "TEMP_Signal_Zscore"
        
         # Z-Signals
        physiological_trial_signal_matrix[0][4] = "GSR_Signal_mSiemens_Zscore"
        physiological_trial_signal_matrix[0][5] = "RESP_Signal_Bpm_Zscore"
        physiological_trial_signal_matrix[0][6] = "HR_Signal_bpm_Zscore"
        physiological_trial_signal_matrix[0][7] = "TEMP_Signal_Zscore"
        
        # Z-Signals Ratios
        physiological_trial_signal_matrix[0][8] = "Ratio_G_R"
        physiological_trial_signal_matrix[0][9] = "Ratio_G_HR"
        physiological_trial_signal_matrix[0][10] = "Ratio_G_T"
                
        physiological_trial_signal_matrix[0][11] = "Ratio_R_G"
        physiological_trial_signal_matrix[0][12] = "Ratio_R_HR"
        physiological_trial_signal_matrix[0][13] = "Ratio_R_T"
                
        physiological_trial_signal_matrix[0][14] = "Ratio_HR_G"
        physiological_trial_signal_matrix[0][15] = "Ratio_HR_R"
        physiological_trial_signal_matrix[0][16] = "Ratio_HR_T"
                
        physiological_trial_signal_matrix[0][17] = "Ratio_T_G"
        physiological_trial_signal_matrix[0][18] = "Ratio_T_R"
        physiological_trial_signal_matrix[0][19] = "Ratio_T_HR"
        
        
        
        # =============================================================================
        #        GET GSR SIGNAL USE  column 0 GSR
        # =============================================================================
        participant_GSR_single_trial_array = participants[current_participant][current_trial][0:8064][:,0]
        
        # EXTRACT GSR SIGNALS (mSiemens) FROM RAW VOLTAGE (Capture Errors)
        Participant_GSR_Error = False
        try:
            Participant_GSR_Signal = eda.eda(signal=participant_GSR_single_trial_array, sampling_rate=128.0, show=False)
             
            # Extract GSR SCR Skin Conductivity Responses from Signal Array position 4
            Participant_GSR_SCR = Participant_GSR_Signal[4]
            # Check if No Array has been passed and flag error
            if Participant_GSR_SCR.size == 0: 
                Participant_GSR_Error = True
            else:
                # Otherwise resample array to be out of 60
                Participant_GSR_SCR = signal.resample(Participant_GSR_SCR, 60)
            
        except Exception:
            # In case of Biosppy Error Zero the Participant Array
            Participant_GSR_Error = True
            pass
       
        # Calculate GSR Descriptive Stats
        if not Participant_GSR_Error:
           
            # Load Physiological Trial Signal into Array for saving
            lengthSignal  = get_number_of_elementsinArray(Participant_GSR_SCR)
            
            for current_item in range(0,lengthSignal):
                #Save Signal in first column & Start on Row 1
                matrix_row = current_item + 1
                physiological_trial_signal_matrix[matrix_row][0] = Participant_GSR_SCR[current_item]
           
            # Round All Stats
            
            GSR_Mean = np.round(np.mean(Participant_GSR_SCR),round_dec)
            GSR_SD = np.round(np.std(Participant_GSR_SCR),round_dec)
            GSR_Max = np.round(np.max(Participant_GSR_SCR),round_dec)
            GSR_Min = np.round(np.min(Participant_GSR_SCR),round_dec)
            GSR_Med = np.round(np.median(Participant_GSR_SCR),round_dec)
            
            # Gen Mod of Array
            GSR_Mod_Array = stats.mode((Participant_GSR_SCR),axis=None)
            GSR_Mod = np.round(float(re.sub("[\[\]\']", "",str(GSR_Mod_Array[0]))),round_dec)
            
            # Create Standard Deviation Moving Window Arrays 10 seconds out 60 seconds
            GSR_MSD = np.round(np.std(rolling_window((Participant_GSR_SCR), 10), axis=-1),round_dec)
            # Calculate MIN/MAX moving standard deviation
            GSR_MSD_Max = np.max(GSR_MSD)
            GSR_MSD_Min = np.min(GSR_MSD)
            
        else:
            
            # Zero All Fields
            GSR_Mean = 0
            GSR_SD = 0
            GSR_Max = 0
            GSR_Min = 0
            GSR_Med = 0
            GSR_Mod = 0
            GSR_MSD_Max = 0
            GSR_MSD_Min = 0
            
        # =============================================================================
        #        GET RESP SIGNAL USE  column 2 RESP
        # =============================================================================    
       
        participant_RESP_single_trial_array = participants[current_participant][current_trial][0:8064][:,2]
        
        # EXTRACT RESP SIGNALS (Hz) FROM RAW VOLTAGE (Capture Errors)
        Participant_RESP_Error = False
        try:
            Participant_RESP_Signal = resp.resp(signal=participant_RESP_single_trial_array, sampling_rate=128.0, show=False)
            
            # Extract Resp rate hz from 4th tupple
            Participant_RESP_ResprateHz = Participant_RESP_Signal[4]
            
            if Participant_RESP_ResprateHz.size == 0: 
                # Check if No Array has been passed and flag error
                Participant_RESP_Error = True
            else:
                # Otherwise resample array to be out of 60
                Participant_RESP_ResprateHz = signal.resample(Participant_RESP_ResprateHz, 60)
            
        except Exception:
            # In case of Biosppy Error Zero the Participant Array
            Participant_RESP_Error = True
            Participant_RESP = 0
            pass
        
        # Calculate RESP Descriptive Stats
        if not Participant_RESP_Error:
                        
            # TRANSLATE RESP HZ INTO Bpm Breaths Per Minute
            #Calculate Length of Hz Array
            lengthHz  = get_number_of_elementsinArray(Participant_RESP_ResprateHz)
            
            # Calculate Hz into Breaths per Minute multiply hz by 60 seconds
            Participant_RESP_ResprateBpm = []
            
            for current_item in range(0,lengthHz):
               # Multiple Hz by 60 and round to zero places
               hztobpm = round(Participant_RESP_ResprateHz[current_item] * 60)
               # Load new calculate in bpm array
               Participant_RESP_ResprateBpm.append(hztobpm)
        
            # Convert Resp List to Array
            Participant_RESP_ResprateBpm = np.array(Participant_RESP_ResprateBpm)
            
             # Load Physiological Trial Signal into Array for saving
            lengthSignal  = get_number_of_elementsinArray(Participant_RESP_ResprateBpm)
            
            for current_item in range(0,lengthSignal):
                #Save Signal in first column & Start on Row 1
                matrix_row = current_item + 1
                physiological_trial_signal_matrix[matrix_row][1] = Participant_RESP_ResprateBpm[current_item]
        
            # Caculate RESP Stats from Participant_RESP_ResprateBpm Array and ROUND
            RESP_Mean = np.round(np.mean(Participant_RESP_ResprateBpm),round_dec)
            RESP_SD = np.round(np.std(Participant_RESP_ResprateBpm),round_dec)
            RESP_Max = np.round(np.max(Participant_RESP_ResprateBpm),round_dec)
            RESP_Min = np.round(np.min(Participant_RESP_ResprateBpm),round_dec)
            RESP_Med = np.round(np.median(Participant_RESP_ResprateBpm),round_dec)
            
            RESP_Mod_Array = stats.mode((Participant_RESP_ResprateBpm),axis=None)
            RESP_Mod = np.round(float(re.sub("[\[\]\']", "",str(RESP_Mod_Array[0]))),round_dec) # Get rid of array brackets and convert to float
            
            # Create Standard Deviation Moving Window Arrays 10 seconds out 60 seconds
            RESP_MSD = np.round(np.std(rolling_window((Participant_RESP_ResprateBpm), 10), axis=-1),round_dec)
            # Calculate MIN/MAX moving standard deviation
            RESP_MSD_Max = np.max(RESP_MSD)
            RESP_MSD_Min = np.min(RESP_MSD)
            
        else:
            
            # Zero All Fields
            RESP_Mean = 0
            RESP_SD = 0
            RESP_Max = 0
            RESP_Min = 0
            RESP_Med = 0
            RESP_Mod = 0
            RESP_MSD_Max = 0
            RESP_MSD_Min = 0
        
        
        # =============================================================================
        #        GET PLETH SIGNAL USE  column 4 PLETH
        # =============================================================================    
        
        participant_PLETH_single_trial_array = participants[current_participant][current_trial][0:8064][:,4]
        
        # EXTRACT PLETH SIGNALS (HeartRate bpm) FROM RAW VOLTAGE (Capture Errors)
        Participant_PLETH_Error = False
        try:
            Participant_PLETH_Signal = bvp.bvp(signal=participant_PLETH_single_trial_array, sampling_rate=128.0, show=False)
            
            # Extract Heart Rate bpm position 4 from signals array
            Participant_BVP_HR = Participant_PLETH_Signal[4]
            # Check if No Array has been passed and flag error
            if Participant_BVP_HR.size == 0: 
                # Check if No Array has been passed and flag error
                Participant_PLETH_Error = True
            else: 
                # Otherwise resample array to be out of 60
                Participant_BVP_HR = signal.resample(Participant_BVP_HR, 60)
            
        except Exception:
            # In case of Biosppy Error Zero the Participant Array
            Participant_PLETH_Error = True
            pass
       
        # Calculate PLETH Descriptive Stats
        if not Participant_PLETH_Error:
        
            # Load Physiological Trial Signal into Array for saving
            lengthSignal  = get_number_of_elementsinArray(Participant_BVP_HR)
            
            for current_item in range(0,lengthSignal):
                #Save Signal in first column & Start on Row 1
                matrix_row = current_item + 1
                physiological_trial_signal_matrix[matrix_row][2] = Participant_BVP_HR[current_item]           
        
        
            # Caculate HR Stats from Participant_BVP_HR Array and Round
            HR_Mean = np.round(np.mean(Participant_BVP_HR),round_dec)
            HR_SD = np.round(np.std(Participant_BVP_HR),round_dec)
            HR_Max = np.round(np.max(Participant_BVP_HR),round_dec)
            HR_Min = np.round(np.min(Participant_BVP_HR),round_dec)
            HR_Med = np.round(np.median(Participant_BVP_HR),round_dec)
            
            # Gen Mod of Array
            HR_Mod_Array = stats.mode((Participant_BVP_HR),axis=None)
            HR_Mod = np.round(float(re.sub("[\[\]\']", "",str(HR_Mod_Array[0]))),round_dec) # Get rid of array brackets and convert to float
            
            # Create Standard Deviation Moving Window Arrays 10 seconds out 60 seconds
            HR_MSD = np.round(np.std(rolling_window((Participant_BVP_HR), 10), axis=-1),round_dec)
            # Calculate MIN/MAX moving standard deviation
            HR_MSD_Max = np.max(HR_MSD)
            HR_MSD_Min = np.min(HR_MSD)
            
            # Caculate HRV using HeartPy from participant_PLETH_single_trial_array Raw Voltage
            
            try:
                working_data, HRVmeasures = hp.process(participant_PLETH_single_trial_array, 128, report_time=False)
                # Extract HRV measures
                HRV_SDNN = np.round(HRVmeasures['sdnn'],round_dec)
                HRV_rMSSD = np.round(HRVmeasures['rmssd'],round_dec)
                
                # Capture "nan"
                
                
                   
            except Exception:
                # In case of HeartPy Error Zero the Participant Array
                HRV_SDNN = 0
                HRV_rMSSD = 0
                pass
            
            
        else:
            
            # Zero All Fields
            HR_Mean = 0
            HR_SD = 0
            HR_Max = 0
            HR_Min = 0
            HR_Med = 0
            HR_Mod = 0
            HR_MSD_Max = 0
            HR_MSD_Min = 0
            HRV_SDNN = 0
            HRV_rMSSD = 0
        
        # =============================================================================
        #        GET TEMP Z-Score USE  column 7 TEMP
        # =============================================================================    
              
        participant_TEMP_single_trial_array = participants[current_participant][current_trial][0:8064][:,7]
        
        # Load Physiological Trial Signal into Array for saving
        
        #Downsample sample Temp Signal from 8064 to 60
        resampled_TEMP_single_trial_array = signal.resample(participant_TEMP_single_trial_array, 60)
        
        lengthSignal  = get_number_of_elementsinArray(resampled_TEMP_single_trial_array)
                
        for current_item in range(0,lengthSignal):
            #Save Signal in first column & Start on Row 1
            matrix_row = current_item + 1
            physiological_trial_signal_matrix[matrix_row][3] = resampled_TEMP_single_trial_array[current_item]           
        
        
        TEMP_Mean = np.round(np.mean(participant_TEMP_single_trial_array),round_dec)
        TEMP_SD = np.round(np.std(participant_TEMP_single_trial_array),round_dec)
        TEMP_Max = np.round(np.max(participant_TEMP_single_trial_array),round_dec)
        TEMP_Min = np.round(np.min(participant_TEMP_single_trial_array),round_dec)
        TEMP_Med = np.round(np.median(participant_TEMP_single_trial_array),round_dec)
        
        # Gen Mod of Array
        TEMP_Mod_Array = stats.mode((participant_TEMP_single_trial_array),axis=None)
        TEMP_Mod = np.round(float(re.sub("[\[\]\']", "",str(TEMP_Mod_Array[0]))),round_dec) # Get rid of array brackets and convert to float
        
        # Create Standard Deviation Moving Window Arrays 1000 Frames out 8064 frames
        TEMP_MSD = np.round(np.std(rolling_window((participant_TEMP_single_trial_array), 1000), axis=-1),round_dec)
        
        TEMP_MSD_Max = np.max(TEMP_MSD)
        TEMP_MSD_Min = np.min(TEMP_MSD)
        
        
        # =============================================================================
        #     # GET Values of Moving Standard Deviation MIN/MAX and physiological pairs
        #     # Search column 0-3 in  physiological_trial_signal_matrix
        # =============================================================================
        
        #Load column positions from physiological_trial_signal_matrix
        
        GSRCOL = 0
        RESPCOL = 1
        HRCOL = 2
        TEMPCOL = 3
        
        
        # Intialise Global MSD pair variables and zero, only give values if there are matches
        GSR_MSDMax_RESP = 0
        GSR_MSDMax_HR = 0
        GSR_MSDMax_TEMP = 0
        GSR_MSDMin_RESP = 0
        GSR_MSDMin_HR = 0
        GSR_MSDMin_TEMP = 0
        RESP_MSDMax_GSR = 0
        RESP_MSDMax_HR = 0
        RESP_MSDMax_TEMP = 0
        RESP_MSDMin_GSR = 0
        RESP_MSDMin_HR = 0
        RESP_MSDMin_TEMP = 0
        HR_MSDMax_GSR = 0
        HR_MSDMax_RESP = 0
        HR_MSDMax_TEMP = 0
        HR_MSDMin_GSR = 0
        HR_MSDMin_RESP = 0
        HR_MSDMin_TEMP = 0
                
        # Check that GSR MSD has value 
        if GSR_MSD_Max != 0:
            
                GSRColumn = physiological_trial_signal_matrix[1:][:,GSRCOL].astype(int) # Load column as INT from row 1-60
                
                #Calculate Maximum/MIN GSR based on MSD
                MaxMSDGSR =  GSR_Mean + GSR_MSD_Max
                MinMSDGSR =  GSR_Mean - GSR_MSD_Min
                
                # GSR Max Pairs
                GSRMSDMAXPos_tuple = np.where(GSRColumn == MaxMSDGSR)
                GSRMSDMAXPos_array = np.array(GSRMSDMAXPos_tuple) # convert tupple into np array
                
                # Check that search returned a position 
                if GSRMSDMAXPos_array.size != 0:
                    
                        GSRMSDMAXPos = int(GSRMSDMAXPos_array[0][0]) # Get FIRST Int of pos 
                        GSRMSDMAXPos = GSRMSDMAXPos + 1 # ADD one to sync with trial signal matrix
                    
                        #Load Global Pairs for MSD
                        GSR_MSDMax_RESP = int(physiological_trial_signal_matrix[GSRMSDMAXPos][RESPCOL])
                        GSR_MSDMax_HR = int(physiological_trial_signal_matrix[GSRMSDMAXPos][HRCOL])
                        GSR_MSDMax_TEMP = int(physiological_trial_signal_matrix[GSRMSDMAXPos][TEMPCOL])
               
                # GSR Min Pairs
                GSRMSDMINPos_tuple = np.where(GSRColumn == MinMSDGSR)
                GSRMSDMINPos_array = np.array(GSRMSDMINPos_tuple) # convert tupple into np array
                
                # Check that search returned a position 
                if GSRMSDMINPos_array.size != 0:
                    
                        GSRMSDMINPos = int(GSRMSDMINPos_array[0][0]) # Get FIRST Int of pos 
                        GSRMSDMINPos = GSRMSDMINPos + 1 # ADD one to sync with trial signal matrix
                    
                        #Load Global Pairs for MSD
                        GSR_MSDMin_RESP = int(physiological_trial_signal_matrix[GSRMSDMINPos][RESPCOL])
                        GSR_MSDMin_HR = int(physiological_trial_signal_matrix[GSRMSDMINPos][HRCOL])
                        GSR_MSDMin_TEMP = int(physiological_trial_signal_matrix[GSRMSDMINPos][TEMPCOL])
                
        # Check that RESP MSD has value 
        if RESP_MSD_Max != 0:
           
                RESPColumn = physiological_trial_signal_matrix[1:][:,RESPCOL].astype(int) # Load column as INT from row 1-60
               
                #Calculate Maximum/MIN RESP based on MSD
                MaxMSDRESP =  RESP_Mean + RESP_MSD_Max
                MinMSDRESP =  RESP_Mean - RESP_MSD_Min
                
                # RESP Max Pairs
                RESPMSDMAXPos_tuple = np.where(RESPColumn == MaxMSDRESP)
                RESPMSDMAXPos_array = np.array(RESPMSDMAXPos_tuple) # convert tupple into np array
           
                # Check that search returned a position 
                if RESPMSDMAXPos_array.size != 0:
                      RESPMSDMAXPos = int(RESPMSDMAXPos_array[0][0]) # Get FIRST Int of pos 
                      RESPMSDMAXPos = RESPMSDMAXPos + 1 # ADD one to sync with trial signal matrix
                   
                      #Load Global Pairs for MSD
                      RESP_MSDMax_GSR = int(physiological_trial_signal_matrix[RESPMSDMAXPos][GSRCOL])
                      RESP_MSDMax_HR = int(physiological_trial_signal_matrix[RESPMSDMAXPos][HRCOL])
                      RESP_MSDMax_TEMP = int(physiological_trial_signal_matrix[RESPMSDMAXPos][TEMPCOL])
          
                # RESP Min Pairs
                RESPMSDMINPos_tuple = np.where(RESPColumn == MinMSDRESP)
                RESPMSDMINPos_array = np.array(RESPMSDMINPos_tuple) # convert tupple into np array
               
                # Check that search returned a position 
                if RESPMSDMINPos_array.size != 0:
                   
                        RESPMSDMINPos = int(RESPMSDMINPos_array[0][0]) # Get FIRST Int of pos 
                        RESPMSDMINPos = RESPMSDMINPos + 1 # ADD one to sync with trial signal matrix
                   
                        #Load Global Pairs for MSD
                        RESP_MSDMin_GSR = int(physiological_trial_signal_matrix[RESPMSDMINPos][GSRCOL])
                        RESP_MSDMin_HR = int(physiological_trial_signal_matrix[RESPMSDMINPos][HRCOL])
                        RESP_MSDMin_TEMP = int(physiological_trial_signal_matrix[RESPMSDMINPos][TEMPCOL])
            
        # Check that HR MSD has value 
        if HR_MSD_Max != 0:
            
                HRColumn = physiological_trial_signal_matrix[1:][:,HRCOL].astype(int) # Load column as INT from row 1-60
                
                #Calculate Maximum/MIN HR based on MSD
                MaxMSDHR =  HR_Mean + HR_MSD_Max
                MinMSDHR =  HR_Mean - HR_MSD_Min
                
                # HR Max Pairs
                HRMSDMAXPos_tuple = np.where(HRColumn == MaxMSDHR)
                HRMSDMAXPos_array = np.array(HRMSDMAXPos_tuple) # convert tupple into np array
            
                # Check that search returned a position 
                if HRMSDMAXPos_array.size != 0:
                    HRMSDMAXPos = int(HRMSDMAXPos_array[0][0]) # Get FIRST Int of pos 
                    HRMSDMAXPos = HRMSDMAXPos + 1 # ADD one to sync with trial signal matrix
                
                    #Load Global Pairs for MSD
                    HR_MSDMax_GSR = int(physiological_trial_signal_matrix[HRMSDMAXPos][GSRCOL])
                    HR_MSDMax_RESP = int(physiological_trial_signal_matrix[HRMSDMAXPos][RESPCOL])
                    HR_MSDMax_TEMP = int(physiological_trial_signal_matrix[HRMSDMAXPos][TEMPCOL])
           
                # HR Min Pairs
                HRMSDMINPos_tuple = np.where(HRColumn == MinMSDHR)
                HRMSDMINPos_array = np.array(HRMSDMINPos_tuple) # convert tupple into np array
                
                # Check that search returned a position 
                if HRMSDMINPos_array.size != 0:
                    
                   HRMSDMINPos = int(HRMSDMINPos_array[0][0]) # Get FIRST Int of pos 
                   HRMSDMINPos = HRMSDMINPos + 1 # ADD one to sync with trial signal matrix
               
                   #Load Global Pairs for MSD
                   HR_MSDMin_GSR = int(physiological_trial_signal_matrix[HRMSDMINPos][GSRCOL])
                   HR_MSDMin_RESP = int(physiological_trial_signal_matrix[HRMSDMINPos][RESPCOL])
                   HR_MSDMin_TEMP = int(physiological_trial_signal_matrix[HRMSDMINPos][TEMPCOL])
       
        
        # =============================================================================
        #     # Create Z-score 4 Decimals Column 0-2 From GSR to HR
        #     # Copy Z-value of Temp Across to Column 7
        # =============================================================================

                        
        # Column 0-3
        for columnnum in range(0,3):
        
            # Get Entire Column from starting from row 1
            currentcolumn = physiological_trial_signal_matrix[1:][:,columnnum]
            # Convert to Float
            currentcolumn = currentcolumn.astype(float)
            # Z-Convert Column at 1 decimals
            round_dec = 4
            currentcolumn = np.round(stats.zscore(currentcolumn),round_dec)
                        
            # Populate columns 5-7
            for rownum in range(1,61):
                # Calculate position of column 4 spaces from the raw signal
                z_columnum = columnnum + 4
                physiological_trial_signal_matrix[rownum][z_columnum] = currentcolumn[rownum-1]

        # Carbon copy the Z- Value Temp Columm in Column 7
        for current_item in range(0,60):
            matrix_row = current_item + 1
            physiological_trial_signal_matrix[matrix_row][7] = resampled_TEMP_single_trial_array[current_item]  
            
        # =============================================================================
        #     # Create Z-score RATIOS Columns 8-19 & Descriptive Min/Max
        # =============================================================================
        
        
        for matrix_row in range(1,61):
            
            round_dec = 2
           
            GSR_Current_Row = physiological_trial_signal_matrix[matrix_row][4]
            Respiration_Current_Row = physiological_trial_signal_matrix[matrix_row][5]
            HR_Current_Row = physiological_trial_signal_matrix[matrix_row][6]
            Temp_Current_Row = physiological_trial_signal_matrix[matrix_row][7]
           
            #GSR Ratios 8-10
            physiological_trial_signal_matrix[matrix_row][8] = np.round((GSR_Current_Row/Respiration_Current_Row),round_dec)
            physiological_trial_signal_matrix[matrix_row][9] = np.round((GSR_Current_Row/HR_Current_Row),round_dec)
            physiological_trial_signal_matrix[matrix_row][10] = np.round((GSR_Current_Row/Temp_Current_Row),round_dec)
                                                                         
            #RESP Ratios 11-13
            physiological_trial_signal_matrix[matrix_row][11] = np.round((Respiration_Current_Row/GSR_Current_Row),round_dec)
            physiological_trial_signal_matrix[matrix_row][12] = np.round((Respiration_Current_Row/HR_Current_Row),round_dec)
            physiological_trial_signal_matrix[matrix_row][13] = np.round((Respiration_Current_Row/Temp_Current_Row),round_dec)
                                                                         
            #HR Ratios 14-16
            physiological_trial_signal_matrix[matrix_row][14] = np.round((HR_Current_Row/GSR_Current_Row),round_dec)
            physiological_trial_signal_matrix[matrix_row][15] = np.round((HR_Current_Row/Respiration_Current_Row),round_dec)
            physiological_trial_signal_matrix[matrix_row][16] = np.round((HR_Current_Row/Temp_Current_Row),round_dec)                                                            
        
            #TEMP Ratios 17-19
            physiological_trial_signal_matrix[matrix_row][17] = np.round((Temp_Current_Row/GSR_Current_Row),round_dec)
            physiological_trial_signal_matrix[matrix_row][18] = np.round((Temp_Current_Row/Respiration_Current_Row),round_dec)
            physiological_trial_signal_matrix[matrix_row][19] = np.round((Temp_Current_Row/HR_Current_Row),round_dec)                                                                 
        
        # Calculate Max/Min or Ratios
        
        round_dec = 0
        
        # GSR Ratios Col 8-10
        Ratio_G_R_Max = np.round(np.max(physiological_trial_signal_matrix[1:][:,8]),round_dec)
        Ratio_G_R_Min = np.round(np.min(physiological_trial_signal_matrix[1:][:,8]),round_dec)
        Ratio_G_HR_Max = np.round(np.max(physiological_trial_signal_matrix[1:][:,9]),round_dec)
        Ratio_G_HR_Min = np.round(np.min(physiological_trial_signal_matrix[1:][:,9]),round_dec)
        Ratio_G_T_Max = np.round(np.max(physiological_trial_signal_matrix[1:][:,10]),round_dec)
        Ratio_G_T_Min = np.round(np.min(physiological_trial_signal_matrix[1:][:,10]),round_dec)
        
        # RESP Ratios Col 11-13
        Ratio_R_G_Max = np.round(np.max(physiological_trial_signal_matrix[1:][:,11]),round_dec)
        Ratio_R_G_Min = np.round(np.min(physiological_trial_signal_matrix[1:][:,11]),round_dec)
        Ratio_R_HR_Max = np.round(np.max(physiological_trial_signal_matrix[1:][:,12]),round_dec)
        Ratio_R_HR_Min = np.round(np.min(physiological_trial_signal_matrix[1:][:,12]),round_dec)
        Ratio_R_T_Max = np.round(np.max(physiological_trial_signal_matrix[1:][:,13]),round_dec)
        Ratio_R_T_Min = np.round(np.min(physiological_trial_signal_matrix[1:][:,13]),round_dec)
        
        # HR Ratios Col 14 -16
        Ratio_HR_G_Max = np.round(np.max(physiological_trial_signal_matrix[1:][:,14]),round_dec)
        Ratio_HR_G_Min = np.round(np.min(physiological_trial_signal_matrix[1:][:,14]),round_dec)
        Ratio_HR_R_Max = np.round(np.max(physiological_trial_signal_matrix[1:][:,15]),round_dec)
        Ratio_HR_R_Min = np.round(np.min(physiological_trial_signal_matrix[1:][:,15]),round_dec)
        Ratio_HR_T_Max = np.round(np.max(physiological_trial_signal_matrix[1:][:,16]),round_dec)
        Ratio_HR_T_Min = np.round(np.min(physiological_trial_signal_matrix[1:][:,16]),round_dec)
        
        # T Ratios Col 17 - 19
        Ratio_T_G_Max = np.round(np.max(physiological_trial_signal_matrix[1:][:,17]),round_dec)
        Ratio_T_G_Min = np.round(np.min(physiological_trial_signal_matrix[1:][:,17]),round_dec)
        Ratio_T_R_Max = np.round(np.max(physiological_trial_signal_matrix[1:][:,18]),round_dec)
        Ratio_T_R_Min = np.round(np.min(physiological_trial_signal_matrix[1:][:,18]),round_dec)
        Ratio_T_HR_Max = np.round(np.max(physiological_trial_signal_matrix[1:][:,19]),round_dec)
        Ratio_T_HR_Min = np.round(np.min(physiological_trial_signal_matrix[1:][:,19]),round_dec)
        
        
        # =============================================================================
        #     # Create Physiological Pair Variables  From GSR, RESP, HR & TEMP
        #     # Use position of min and max of each column i the physiological_trial_signal_matrix
        # =============================================================================
        
        
        #Load Variables
        
        GSRCOL = 0
        RESPCOL = 1
        HRCOL = 2
        TEMPCOL = 3
        
        # GSR Max Pairs
        GSRMAXPos = np.where(physiological_trial_signal_matrix[0:][:,GSRCOL] == np.max(physiological_trial_signal_matrix[1:][:,GSRCOL]))
        GSRMAXPos = int(GSRMAXPos[0][0]) # Get Int of pos
        
        GSRMax_Respiration = int(physiological_trial_signal_matrix[GSRMAXPos][RESPCOL])
        GSRMax_HR = int(physiological_trial_signal_matrix[GSRMAXPos][HRCOL])
        GSRMax_Temp = int(physiological_trial_signal_matrix[GSRMAXPos][TEMPCOL])
        
        # GSR Min Pairs
        GSRMINPos = np.where(physiological_trial_signal_matrix[0:][:,GSRCOL] == np.min(physiological_trial_signal_matrix[1:][:,GSRCOL]))
        GSRMINPos = int(GSRMINPos[0][0]) # Get Int of pos
        
        GSRMin_Respiration = int(physiological_trial_signal_matrix[GSRMINPos][RESPCOL])
        GSRMin_HR = int(physiological_trial_signal_matrix[GSRMINPos][HRCOL])
        GSRMin_Temp = int(physiological_trial_signal_matrix[GSRMINPos][TEMPCOL])
        
        # RESP Max Pairs
        RESPMAXPos = np.where(physiological_trial_signal_matrix[0:][:,RESPCOL] == np.max(physiological_trial_signal_matrix[1:][:,RESPCOL]))
        RESPMAXPos = int(RESPMAXPos[0][0]) # Get Int of pos
        
        RESPMax_GSR = int(physiological_trial_signal_matrix[RESPMAXPos][GSRCOL])
        RESPMax_HR = int(physiological_trial_signal_matrix[RESPMAXPos][HRCOL])
        RESPMax_Temp = int(physiological_trial_signal_matrix[RESPMAXPos][TEMPCOL])
        
        # RESP Min Pairs
        RESPMINPos = np.where(physiological_trial_signal_matrix[0:][:,RESPCOL] == np.min(physiological_trial_signal_matrix[1:][:,RESPCOL]))
        RESPMINPos = int(RESPMINPos[0][0]) # Get Int of pos
        
        RESPMin_GSR = int(physiological_trial_signal_matrix[RESPMINPos][GSRCOL])
        RESPMin_HR = int(physiological_trial_signal_matrix[RESPMINPos][HRCOL])
        RESPMin_Temp = int(physiological_trial_signal_matrix[RESPMINPos][TEMPCOL])
        
        # HR Max Pairs
        HRMAXPos = np.where(physiological_trial_signal_matrix[0:][:,HRCOL] == np.max(physiological_trial_signal_matrix[1:][:,HRCOL]))
        HRMAXPos = int(HRMAXPos[0][0]) # Get Int of pos
        
        HRMax_GSR = int(physiological_trial_signal_matrix[HRMAXPos][GSRCOL])
        HRMax_RESP = int(physiological_trial_signal_matrix[HRMAXPos][RESPCOL])
        HRMax_Temp = int(physiological_trial_signal_matrix[HRMAXPos][TEMPCOL])
        
        # HR Min Pairs
        HRMINPos = np.where(physiological_trial_signal_matrix[0:][:,HRCOL] == np.min(physiological_trial_signal_matrix[1:][:,HRCOL]))
        HRMINPos = int(HRMINPos[0][0]) # Get Int of pos
        
        HRMin_GSR = int(physiological_trial_signal_matrix[HRMINPos][GSRCOL])
        HRMin_RESP = int(physiological_trial_signal_matrix[HRMINPos][RESPCOL])
        HRMin_Temp = int(physiological_trial_signal_matrix[HRMINPos][TEMPCOL])
        
        # TEMP Max Pairs
        TEMPMAXPos = np.where(physiological_trial_signal_matrix[0:][:,TEMPCOL] == np.max(physiological_trial_signal_matrix[1:][:,TEMPCOL]))
        TEMPMAXPos = int(TEMPMAXPos[0][0]) # Get Int of pos
        
        TEMPMax_GSR = int(physiological_trial_signal_matrix[TEMPMAXPos][GSRCOL])
        TEMPMax_RESP = int(physiological_trial_signal_matrix[TEMPMAXPos][RESPCOL])
        TEMPMax_HR = int(physiological_trial_signal_matrix[TEMPMAXPos][HRCOL])
        
        # TEMP Min Pairs
        TEMPMINPos = np.where(physiological_trial_signal_matrix[0:][:,TEMPCOL] == np.min(physiological_trial_signal_matrix[1:][:,TEMPCOL]))
        TEMPMINPos = int(TEMPMINPos[0][0]) # Get Int of pos
        
        TEMPMin_GSR = int(physiological_trial_signal_matrix[TEMPMINPos][GSRCOL])
        TEMPMin_RESP = int(physiological_trial_signal_matrix[TEMPMINPos][RESPCOL])
        TEMPMin_HR = int(physiological_trial_signal_matrix[TEMPMINPos][HRCOL])
        
        
        # =============================================================================
        #        DECLARE ALL Global to Local Normative/Z Values of Attributes
        #        EMPTY STRINGS WHILE PROCESSING ALL TRIALS
        # =============================================================================  
        
        # GSR Skin Conductance Global Reference Nominal Values (Below,Above, Equal)
        GSR_Mean_GlobalRef = ""
        GSR_SD_GlobalRef = ""
        GSR_Max_GlobalRef = ""
        GSR_Min_GlobalRef = ""
        
        # RESP Global Reference Nominal Values (Below,Above, Equal)
        RESP_Mean_GlobalRef = ""
        RESP_SD_GlobalRef = ""
        RESP_Max_GlobalRef = ""
        RESP_Min_GlobalRef = ""
        
        # PLETH Global Reference Nominal Values (Below,Above, Equal)
        HR_Mean_GlobalRef = ""
        HR_SD_GlobalRef = ""
        HR_Max_GlobalRef = ""
        HR_Min_GlobalRef = ""
        
        # TEMP Global Reference Nominal Values (Below,Above, Equal)
        TEMP_Mean_GlobalRef = ""
        TEMP_Max_GlobalRef = ""
        TEMP_Min_GlobalRef = ""
        
        #  Global to Local ZScore for MEAN/SD/MAX/MIN for each 4 channel
        GSR_Mean_Global_ZScore = 0
        GSR_SD_Global_ZScore = 0
        GSR_Max_GlobalRef_ZScore = 0
        GSR_Min_GlobalRef_ZScore = 0
        
        RESP_Mean_GlobalRef_ZScore = 0
        RESP_SD_GlobalRef_ZScore = 0
        RESP_Max_GlobalRef_ZScore = 0
        RESP_Min_GlobalRef_ZScore = 0
        
        HR_Mean_GlobalRef_ZScore = 0
        HR_SD_GlobalRef_ZScore = 0
        HR_Max_GlobalRef_ZScore = 0
        HR_Min_GlobalRef_ZScore = 0
        
        TEMP_Mean_GlobalRef_ZScore = 0
        TEMP_Max_GlobalRef_ZScore = 0
        TEMP_Min_GlobalRef_ZScore = 0
        
        
        # =============================================================================
        #        MATCH CLASSIFICATION LABELS TO TRIALS
        # =============================================================================  
        
        # Search Deap_Trial_Labels_CSV.csv matrix for participant and trial location in the label array, returns position of on matrix, 
        # location column 0 is participant number and column 1 is trial number
        
        idxlocationlabel = np.where((participant_trial_labels_array[:,0] == str(current_participant)) 
                                     & (participant_trial_labels_array[:,1] == str(current_trial)))
        
                
        
        trial_2class_stress_label = participant_trial_labels_array[idxlocationlabel][:,4]
        trial_2class_stress_label = re.sub("[\[\]\']", "",str(trial_2class_stress_label)) # Clean up brackets


        
        # =============================================================================
        #        LOAD FIELDS INTO PARTICIPANT MATRIX FOR CURRENT TRIAL 
        # =============================================================================  
    
        # Load Headers
        participant_all_fields_matrix[current_trial][0] = str(current_participant)
        participant_all_fields_matrix[current_trial][1] = str(current_trial)
        
        # GSR Skin Conductance (mSiemens)
        participant_all_fields_matrix[current_trial][2] = str(GSR_Mean)
        participant_all_fields_matrix[current_trial][3] = str(GSR_SD)
        participant_all_fields_matrix[current_trial][4] = str(GSR_Max)
        participant_all_fields_matrix[current_trial][5] = str(GSR_Min)
        participant_all_fields_matrix[current_trial][6] = str(GSR_Med)
        participant_all_fields_matrix[current_trial][7] = str(GSR_Mod)
        
        # RESP Breaths per minute (Bpm)
        participant_all_fields_matrix[current_trial][8] = str(RESP_Mean)
        participant_all_fields_matrix[current_trial][9] = str(RESP_SD)
        participant_all_fields_matrix[current_trial][10] = str(RESP_Max)
        participant_all_fields_matrix[current_trial][11] = str(RESP_Min)
        participant_all_fields_matrix[current_trial][12] = str(RESP_Med)
        participant_all_fields_matrix[current_trial][13] = str(RESP_Mod)
        
        # PLETH ->HeartRate(bpm)
        participant_all_fields_matrix[current_trial][14] = str(HR_Mean)
        participant_all_fields_matrix[current_trial][15] = str(HR_SD)
        participant_all_fields_matrix[current_trial][16] = str(HR_Max)
        participant_all_fields_matrix[current_trial][17] = str(HR_Min)
        participant_all_fields_matrix[current_trial][18] = str(HR_Med)
        participant_all_fields_matrix[current_trial][19] = str(HR_Mod)
        
        # PLETH -> HRV_SDNN (sdnn)
        participant_all_fields_matrix[current_trial][20] = str(HRV_SDNN)
      
        
        # PLETH -> HRV_rMSSD (rMSSD)
        participant_all_fields_matrix[current_trial][21] = str(HRV_rMSSD)
     
        
        # TEMP KEEP USING Z-VALUES no sd
        participant_all_fields_matrix[current_trial][22] = str(TEMP_Mean)
        participant_all_fields_matrix[current_trial][23] = str(TEMP_Max)
        participant_all_fields_matrix[current_trial][24] = str(TEMP_Min)
        participant_all_fields_matrix[current_trial][25] = str(TEMP_Med)
        participant_all_fields_matrix[current_trial][26] = str(TEMP_Mod)
        
        # Values of Moving Standard Deviation MIN/MAX and physiological pairs
        participant_all_fields_matrix[current_trial][27] = str(GSR_MSD_Max)
        participant_all_fields_matrix[current_trial][28] = str(GSR_MSDMax_RESP)
        participant_all_fields_matrix[current_trial][29] = str(GSR_MSDMax_HR)
        participant_all_fields_matrix[current_trial][30] = str(GSR_MSDMax_TEMP)
            
        participant_all_fields_matrix[current_trial][31] = str(GSR_MSD_Min)
        participant_all_fields_matrix[current_trial][32] = str(GSR_MSDMin_RESP)
        participant_all_fields_matrix[current_trial][33] = str(GSR_MSDMin_HR)
        participant_all_fields_matrix[current_trial][34] = str(GSR_MSDMin_TEMP)
        
        participant_all_fields_matrix[current_trial][35] = str(RESP_MSD_Max)
        participant_all_fields_matrix[current_trial][36] = str(RESP_MSDMax_GSR)
        participant_all_fields_matrix[current_trial][37] = str(RESP_MSDMax_HR)
        participant_all_fields_matrix[current_trial][38] = str(RESP_MSDMax_TEMP)
        
        participant_all_fields_matrix[current_trial][39] = str(RESP_MSD_Min)
        participant_all_fields_matrix[current_trial][40] = str(RESP_MSDMin_GSR)
        participant_all_fields_matrix[current_trial][41] = str(RESP_MSDMin_HR)
        participant_all_fields_matrix[current_trial][42] = str(RESP_MSDMin_TEMP)
        
        participant_all_fields_matrix[current_trial][43] = str(HR_MSD_Max)
        participant_all_fields_matrix[current_trial][44] = str(HR_MSDMax_GSR)
        participant_all_fields_matrix[current_trial][45] = str(HR_MSDMax_RESP)
        participant_all_fields_matrix[current_trial][46] = str(HR_MSDMax_TEMP)
            
        participant_all_fields_matrix[current_trial][47] = str(HR_MSD_Min)
        participant_all_fields_matrix[current_trial][48] = str(HR_MSDMin_GSR)
        participant_all_fields_matrix[current_trial][49] = str(HR_MSDMin_RESP)
        participant_all_fields_matrix[current_trial][50] = str(HR_MSDMin_TEMP)
            
        participant_all_fields_matrix[current_trial][51] = str(TEMP_MSD_Max)
        participant_all_fields_matrix[current_trial][52] = str(TEMP_MSD_Min)
        
        # Values of Ratios Max
        participant_all_fields_matrix[current_trial][53] = str(Ratio_G_R_Max)
        participant_all_fields_matrix[current_trial][54] = str(Ratio_G_HR_Max)
        participant_all_fields_matrix[current_trial][55] = str(Ratio_G_T_Max)
        participant_all_fields_matrix[current_trial][56] = str(Ratio_R_G_Max)
        participant_all_fields_matrix[current_trial][57] = str(Ratio_R_HR_Max)
        participant_all_fields_matrix[current_trial][58] = str(Ratio_R_T_Max)
        participant_all_fields_matrix[current_trial][59] = str(Ratio_HR_G_Max)
        participant_all_fields_matrix[current_trial][60] = str(Ratio_HR_R_Max)
        participant_all_fields_matrix[current_trial][61] = str(Ratio_HR_T_Max)
        participant_all_fields_matrix[current_trial][62] = str(Ratio_T_G_Max)
        participant_all_fields_matrix[current_trial][63] = str(Ratio_T_R_Max)
        participant_all_fields_matrix[current_trial][64] = str(Ratio_T_HR_Max)
        
        # Values of Ratios Min
        participant_all_fields_matrix[current_trial][65] = str(Ratio_G_R_Min)
        participant_all_fields_matrix[current_trial][66] = str(Ratio_G_HR_Min)
        participant_all_fields_matrix[current_trial][67] = str(Ratio_G_T_Min)
        participant_all_fields_matrix[current_trial][68] = str(Ratio_R_G_Min)
        participant_all_fields_matrix[current_trial][69] = str(Ratio_R_HR_Min)
        participant_all_fields_matrix[current_trial][70] = str(Ratio_R_T_Min)
        participant_all_fields_matrix[current_trial][71] = str(Ratio_HR_G_Min)
        participant_all_fields_matrix[current_trial][72] = str(Ratio_HR_R_Min)
        participant_all_fields_matrix[current_trial][73] = str(Ratio_HR_T_Min)
        participant_all_fields_matrix[current_trial][74] = str(Ratio_T_G_Min)
        participant_all_fields_matrix[current_trial][75] = str(Ratio_T_R_Min)
        participant_all_fields_matrix[current_trial][76] = str(Ratio_T_HR_Min)
        
        # Physiological Max/Min Pairs
        participant_all_fields_matrix[current_trial][77] = str(GSRMax_Respiration)
        participant_all_fields_matrix[current_trial][78] = str(GSRMax_HR)
        participant_all_fields_matrix[current_trial][79] = str(GSRMax_Temp)
        participant_all_fields_matrix[current_trial][80] = str(GSRMin_Respiration)
        participant_all_fields_matrix[current_trial][81] = str(GSRMin_HR)
        participant_all_fields_matrix[current_trial][82] = str(GSRMin_Temp)
        
        participant_all_fields_matrix[current_trial][83] = str(RESPMax_GSR)
        participant_all_fields_matrix[current_trial][84] = str(RESPMax_HR)
        participant_all_fields_matrix[current_trial][85] = str(RESPMax_Temp)
        participant_all_fields_matrix[current_trial][86] = str(RESPMin_GSR)
        participant_all_fields_matrix[current_trial][87] = str(RESPMin_HR)
        participant_all_fields_matrix[current_trial][88] = str(RESPMin_Temp)
        
        participant_all_fields_matrix[current_trial][89] = str(HRMax_GSR)
        participant_all_fields_matrix[current_trial][90] = str(HRMax_RESP)
        participant_all_fields_matrix[current_trial][91] = str(HRMax_Temp)
        participant_all_fields_matrix[current_trial][92] = str(HRMin_GSR)
        participant_all_fields_matrix[current_trial][93] = str(HRMin_RESP)
        participant_all_fields_matrix[current_trial][94] = str(HRMin_Temp)
        
        participant_all_fields_matrix[current_trial][95] = str(TEMPMax_GSR)
        participant_all_fields_matrix[current_trial][96] = str(TEMPMax_RESP)
        participant_all_fields_matrix[current_trial][97] = str(TEMPMax_HR)
        participant_all_fields_matrix[current_trial][98] = str(TEMPMin_GSR)
        participant_all_fields_matrix[current_trial][99] = str(TEMPMin_RESP)
        participant_all_fields_matrix[current_trial][100] = str(TEMPMin_HR)
        
        # Global to Local Normative Values of Attributes
        # GSR Skin Conductance Global Reference Nominal Values (below,above, equal)
        participant_all_fields_matrix[current_trial][101] = str(GSR_Mean_GlobalRef)
        participant_all_fields_matrix[current_trial][102] = str(GSR_SD_GlobalRef)
        participant_all_fields_matrix[current_trial][103] = str(GSR_Max_GlobalRef)
        participant_all_fields_matrix[current_trial][104] = str(GSR_Min_GlobalRef)
        
        # RESP Global Reference Nominal Values (below,above, equal)
        participant_all_fields_matrix[current_trial][105] = str(RESP_Mean_GlobalRef)
        participant_all_fields_matrix[current_trial][106] = str(RESP_SD_GlobalRef)
        participant_all_fields_matrix[current_trial][107] = str(RESP_Max_GlobalRef)
        participant_all_fields_matrix[current_trial][108] = str(RESP_Min_GlobalRef)
        
        # PLETH Global Reference Nominal Values (below,above, equal)
        participant_all_fields_matrix[current_trial][109] = str(HR_Mean_GlobalRef)
        participant_all_fields_matrix[current_trial][110] = str(HR_SD_GlobalRef)
        participant_all_fields_matrix[current_trial][111] = str(HR_Max_GlobalRef)
        participant_all_fields_matrix[current_trial][112] = str(HR_Min_GlobalRef)
        
         # TEMP Global Reference Nominal Values (below,above, equal)
        participant_all_fields_matrix[current_trial][113] = str(TEMP_Mean_GlobalRef)
        participant_all_fields_matrix[current_trial][114] = str(TEMP_Max_GlobalRef)
        participant_all_fields_matrix[current_trial][115] = str(TEMP_Min_GlobalRef)
    
        #  Global to Local ZScore for MEAN/MAX/MIN for each 4 channel
        # GSR Skin Conductance Global Reference Nominal Values (below,above, equal)
        participant_all_fields_matrix[current_trial][116] = str(GSR_Mean_Global_ZScore)
        participant_all_fields_matrix[current_trial][117] = str(GSR_SD_Global_ZScore)
        participant_all_fields_matrix[current_trial][118] = str(GSR_Max_GlobalRef_ZScore)
        participant_all_fields_matrix[current_trial][119] = str(GSR_Min_GlobalRef_ZScore)
        
        # RESP Global Reference Nominal Values (below,above, equal)
        participant_all_fields_matrix[current_trial][120] = str(RESP_Mean_GlobalRef_ZScore)
        participant_all_fields_matrix[current_trial][121] = str(RESP_SD_GlobalRef_ZScore)
        participant_all_fields_matrix[current_trial][122] = str(RESP_Max_GlobalRef_ZScore)
        participant_all_fields_matrix[current_trial][123] = str(RESP_Min_GlobalRef_ZScore)
        
        # PLETH Global Reference Nominal Values (below,above, equal)
        participant_all_fields_matrix[current_trial][124] = str(HR_Mean_GlobalRef_ZScore)
        participant_all_fields_matrix[current_trial][125] = str(HR_SD_GlobalRef_ZScore)
        participant_all_fields_matrix[current_trial][126] = str(HR_Max_GlobalRef_ZScore)
        participant_all_fields_matrix[current_trial][127] = str(HR_Min_GlobalRef_ZScore)
        
         # TEMP Global Reference Nominal Values (below,above, equal)
        participant_all_fields_matrix[current_trial][128] = str(TEMP_Mean_GlobalRef_ZScore)
        participant_all_fields_matrix[current_trial][129] = str(TEMP_Max_GlobalRef_ZScore)
        participant_all_fields_matrix[current_trial][130] = str(TEMP_Min_GlobalRef_ZScore)
        
        participant_all_fields_matrix[current_trial][131] = str(trial_2class_stress_label)

        
       
    
        # =============================================================================
        #     OUTPUT CURRENT PROCESSED SIGNALS TRIAL CSV TO FILES
        # =============================================================================
    
    
        csv.register_dialect('TrialCSVFormat',
                                 delimiter = ',',
                                 quoting=csv.QUOTE_NONE,
                                 skipinitialspace=True)
        
        participant_trial_csv_name = path + output_dir \
                + str("participant_") \
                + str(current_participant) + str("_Trial_") \
                + str(current_trial) \
                + str("_all_ProcessedSignals_matrix.csv")
        
        with open(participant_trial_csv_name, 'w') as f:
            writer = csv.writer(f, dialect='TrialCSVFormat')
            for row in physiological_trial_signal_matrix:
                writer.writerow(row)
        f.close()
        
        print("Saved Trial File:" + str("participant_") + str(current_participant) + str("_Trial") + str(current_trial) + str("_all_fields_matrix.csv"))
    
            
        print("**>Processed Participant Signals for " + str(current_participant) + " Trial Number" + str(current_trial))
    
        # =============================================================================
        # 
        # END OF LOOP CURRENT PARTICIPANT LOOP 
        # 
        # =============================================================================
    
    # =============================================================================
    #     # POSTPROCESSING GLOBAL ATTRIBUTES 101-127 participant_all_fields_matrix  
    # =============================================================================
    
    # GSR Skin Conductance Global Reference Nominal Values (Below,Above, Equal)
    # GSR_Mean_GlobalRef 
    # GSR_SD_GlobalRef 
    # GSR_Max_GlobalRef 
    # GSR_Min_GlobalRef 
    
    # GSR Source Columns 2-5
    SourcecolSTART = 2
    SourcecolEND = 5
    
    # GSR Global Destination Columns 101-104
    DestGlobalcolSTART = 101
    DestGlobalcolEND = 104
    dest_column_num = DestGlobalcolSTART
    
    for source_column_num in range(SourcecolSTART,SourcecolEND+1):
    
        # Get Column from starting from row 1
        SourceColumn = participant_all_fields_matrix[1:][:,source_column_num]
        # Convert to Float
        SourceColumn = SourceColumn.astype(float)
        # Calculate Average for Source Column
        round_dec = 0
        SourceColumn_AVG = np.round(np.mean(SourceColumn),round_dec)
        
        # Loop through all rows and compare source row to the global AVG        
        for rownum in range(1,41):
            SourceColumn_ROW = float(participant_all_fields_matrix[rownum][source_column_num])
            
            if SourceColumn_ROW == SourceColumn_AVG: 
                participant_all_fields_matrix[rownum][dest_column_num] = "Equal"
            
            if SourceColumn_ROW < SourceColumn_AVG: 
                participant_all_fields_matrix[rownum][dest_column_num] = "Below"
                
            if SourceColumn_ROW > SourceColumn_AVG: 
                participant_all_fields_matrix[rownum][dest_column_num] = "Above"
            
                    
        # Increment dest_column_num as per the number of source columns
        dest_column_num = dest_column_num + 1
    
    
    # RESP Global Reference Nominal Values (Below,Above, Equal)
    # RESP_Mean_GlobalRef = ""
    # RESP_SD_GlobalRef = ""
    # RESP_Max_GlobalRef = ""
    # RESP_Min_GlobalRef = ""
    
     
    # RESP Source Columns 8-11
    SourcecolSTART = 8
    SourcecolEND = 11
    
    # RESP Destination Columns 105-108
    DestGlobalcolSTART = 105
    DestGlobalcolEND = 108
    dest_column_num = DestGlobalcolSTART
    
    for source_column_num in range(SourcecolSTART,SourcecolEND+1):
    
        # Get Column from starting from row 1
        SourceColumn = participant_all_fields_matrix[1:][:,source_column_num]
        # Convert to Float
        SourceColumn = SourceColumn.astype(float)
        # Calculate Average for Source Column
        round_dec = 0
        SourceColumn_AVG = np.round(np.mean(SourceColumn),round_dec)
        
        # Loop through all rows and compare source row to the global AVG        
        for rownum in range(1,41):
            SourceColumn_ROW = float(participant_all_fields_matrix[rownum][source_column_num])
            
            if SourceColumn_ROW == SourceColumn_AVG: 
                participant_all_fields_matrix[rownum][dest_column_num] = "Equal"
            
            if SourceColumn_ROW < SourceColumn_AVG: 
                participant_all_fields_matrix[rownum][dest_column_num] = "Below"
                
            if SourceColumn_ROW > SourceColumn_AVG: 
                participant_all_fields_matrix[rownum][dest_column_num] = "Above"
            
                    
        # Increment dest_column_num as per the number of source columns
        dest_column_num = dest_column_num + 1
    
    
    # PLETH Global Reference Nominal Values (Below,Above, Equal)
    # HR_Mean_GlobalRef = ""
    # HR_SD_GlobalRef = ""
    # HR_Max_GlobalRef = ""
    # HR_Min_GlobalRef = ""
  
    # HR Source Columns 14-17
    SourcecolSTART = 14
    SourcecolEND = 17
    
    # HR Destination Columns 109-112
    DestGlobalcolSTART = 109
    DestGlobalcolEND = 112
    dest_column_num = DestGlobalcolSTART
    
    for source_column_num in range(SourcecolSTART,SourcecolEND+1):
    
        # Get Column from starting from row 1
        SourceColumn = participant_all_fields_matrix[1:][:,source_column_num]
        # Convert to Float
        SourceColumn = SourceColumn.astype(float)
        # Calculate Average for Source Column
        round_dec = 0
        SourceColumn_AVG = np.round(np.mean(SourceColumn),round_dec)
        
        # Loop through all rows and compare source row to the global AVG        
        for rownum in range(1,41):
            SourceColumn_ROW = float(participant_all_fields_matrix[rownum][source_column_num])
            
            if SourceColumn_ROW == SourceColumn_AVG: 
                participant_all_fields_matrix[rownum][dest_column_num] = "Equal"
            
            if SourceColumn_ROW < SourceColumn_AVG: 
                participant_all_fields_matrix[rownum][dest_column_num] = "Below"
                
            if SourceColumn_ROW > SourceColumn_AVG: 
                participant_all_fields_matrix[rownum][dest_column_num] = "Above"
            
                    
        # Increment dest_column_num as per the number of source columns
        dest_column_num = dest_column_num + 1
    
    
    # TEMP Global Reference Nominal Values (Below,Above, Equal)
    # TEMP_Mean_GlobalRef = ""
    # TEMP_Max_GlobalRef = ""
    # TEMP_Min_GlobalRef = ""
      
    # TEMP Source Columns 22-24
    SourcecolSTART = 22
    SourcecolEND = 24
    
    # TEMP Destination Columns 113-115
    DestGlobalcolSTART = 113
    DestGlobalcolEND = 115
    dest_column_num = DestGlobalcolSTART
    
    for source_column_num in range(SourcecolSTART,SourcecolEND+1):
    
        # Get Column from starting from row 1
        SourceColumn = participant_all_fields_matrix[1:][:,source_column_num]
        # Convert to Float
        SourceColumn = SourceColumn.astype(float)
        # Calculate Average for Source Column
        round_dec = 0
        SourceColumn_AVG = np.round(np.mean(SourceColumn),round_dec)
        
        # Loop through all rows and compare source row to the global AVG        
        for rownum in range(1,41):
            SourceColumn_ROW = float(participant_all_fields_matrix[rownum][source_column_num])
            
            if SourceColumn_ROW == SourceColumn_AVG: 
                participant_all_fields_matrix[rownum][dest_column_num] = "Equal"
            
            if SourceColumn_ROW < SourceColumn_AVG: 
                participant_all_fields_matrix[rownum][dest_column_num] = "Below"
                
            if SourceColumn_ROW > SourceColumn_AVG: 
                participant_all_fields_matrix[rownum][dest_column_num] = "Above"
            
                    
        # Increment dest_column_num as per the number of source columns
        dest_column_num = dest_column_num + 1
    
       
    
    #  Global to Local ZScore for MEAN/MAX/MIN for each 4 channel
    # GSR_Mean_Global_ZScore = 0
    # GSR_Max_GlobalRef_ZScore = 0
    # GSR_Min_GlobalRef_ZScore = 0
    
    # GSR Source Columns 2-5
    SourcecolSTART = 2
    SourcecolEND = 5
    
    # GSR Global Destination Columns 116-119
    DestGlobalcolSTART = 116
    DestGlobalcolEND = 119
    dest_column_num = DestGlobalcolSTART
    
    for columnnum in range(SourcecolSTART,SourcecolEND+1):
    
        # Get Column from starting from row 1
        sourcecolumn = participant_all_fields_matrix[1:][:,columnnum]
        # Convert to Float
        sourcecolumn =  sourcecolumn.astype(float)
        # Z-Convert Column at 1 decimals
        round_dec = 0
        sourcecolumn = np.round(stats.zscore(sourcecolumn),round_dec)
        # Convert current column to string
        currentcolumn = currentcolumn.astype(str)
        
        for rownum in range(1,41):
            participant_all_fields_matrix[rownum][dest_column_num] = sourcecolumn[rownum-1]
            
        # Increment dest_column_num as per the number of source columns
        dest_column_num = dest_column_num + 1   
    
    # Z- Scores for RESP
    # RESP_Mean_GlobalRef_ZScore = 0
    # RESP_Max_GlobalRef_ZScore = 0
    # RESP_Min_GlobalRef_ZScore = 0
    
    # RESP Source Columns 8-11
    SourcecolSTART = 8
    SourcecolEND = 11
    
    # RESP Global Destination Columns 120-123
    DestGlobalcolSTART = 120
    DestGlobalcolEND = 123
    dest_column_num = DestGlobalcolSTART
    
    for columnnum in range(SourcecolSTART,SourcecolEND+1):
    
        # Get Column from starting from row 1
        sourcecolumn = participant_all_fields_matrix[1:][:,columnnum]
        # Convert to Float
        sourcecolumn =  sourcecolumn.astype(float)
        # Z-Convert Column at 1 decimals
        round_dec = 0
        sourcecolumn = np.round(stats.zscore(sourcecolumn),round_dec)
        # Convert current column to string
        currentcolumn = currentcolumn.astype(str)
        
        for rownum in range(1,41):
            participant_all_fields_matrix[rownum][dest_column_num] = sourcecolumn[rownum-1]
            
        # Increment dest_column_num as per the number of source columns
        dest_column_num = dest_column_num + 1   
    
    # Z-Score for HR
    # HR_Mean_GlobalRef_ZScore = 0
    # HR_Max_GlobalRef_ZScore = 0
    # HR_Min_GlobalRef_ZScore = 0
    
    
    # HR Source Columns 14-17
    SourcecolSTART = 14
    SourcecolEND = 17
    
    # HR Global Destination Columns 124-127
    DestGlobalcolSTART = 124
    DestGlobalcolEND = 127
    dest_column_num = DestGlobalcolSTART
    
    for columnnum in range(SourcecolSTART,SourcecolEND+1):
    
        # Get Column from starting from row 1
        sourcecolumn = participant_all_fields_matrix[1:][:,columnnum]
        # Convert to Float
        sourcecolumn =  sourcecolumn.astype(float)
        # Z-Convert Column at 1 decimals
        round_dec = 0
        sourcecolumn = np.round(stats.zscore(sourcecolumn),round_dec)
        # Convert current column to string
        currentcolumn = currentcolumn.astype(str)
        
        for rownum in range(1,41):
            participant_all_fields_matrix[rownum][dest_column_num] = sourcecolumn[rownum-1]
            
        # Increment dest_column_num as per the number of source columns
        dest_column_num = dest_column_num + 1   
    
    
    #Z-Score TEMP
    # TEMP_Mean_GlobalRef_ZScore = 0
    # TEMP_Max_GlobalRef_ZScore = 0
    # TEMP_Min_GlobalRef_ZScore = 0
    
    # TEMP Source Columns 22-24
    SourcecolSTART = 22
    SourcecolEND = 24
    
    # TEMP Global Destination Columns 128-130
    DestGlobalcolSTART = 128
    DestGlobalcolEND = 130
    dest_column_num = DestGlobalcolSTART
    
    for columnnum in range(SourcecolSTART,SourcecolEND+1):
    
        # Get Column from starting from row 1
        sourcecolumn = participant_all_fields_matrix[1:][:,columnnum]
        # Convert to Float
        sourcecolumn =  sourcecolumn.astype(float)
        # Z-Convert Column at 1 decimals
        round_dec = 0
        sourcecolumn = np.round(stats.zscore(sourcecolumn),round_dec)
        # Convert current column to string
        currentcolumn = currentcolumn.astype(str)
        
        for rownum in range(1,41):
            participant_all_fields_matrix[rownum][dest_column_num] = sourcecolumn[rownum-1]
            
        # Increment dest_column_num as per the number of source columns
        dest_column_num = dest_column_num + 1   
    
        
    
    # =============================================================================
    #     # OUTPUT THE ALL TRIALS PER PARTICIPANT MATRIX CSV TO FILES
    # =============================================================================
    
    
    csv.register_dialect('ParticipantCSVFormat',
                             delimiter = ',',
                             quoting=csv.QUOTE_NONE,
                             skipinitialspace=True)
    
    participant_csv_name = path + output_dir + str("participant_") + str(current_participant) + str("_all_ProcessedSignals_matrix.csv")
    
    with open(participant_csv_name, 'w') as f:
        writer = csv.writer(f, dialect='ParticipantCSVFormat')
        for row in participant_all_fields_matrix:
            writer.writerow(row)
    f.close()
    
    print("Saved Participant File:" + str("participant_") + str(current_participant) + str("_all_fields_matrix.csv"))
    
    
    
    # =============================================================================
    #       Generate Weka Files 
    # =============================================================================
    
      
    
    
    # CREATE MIXED NUMPY ARRAY for loading the experiment name for weka and label column 30 ROWS and 3 COLUMNS  
    WEKAexperimentlist = np.zeros((11,3)).astype(object)

    #Experiment 1 Combined Attributes {Desc Stats and Complex Attributes} 
    WEKAexperimentlist[1][0] = "Exp1_Combined_Attributes_2_Class_Stress"
    WEKAexperimentlist[1][1] = 131
    WEKAexperimentlist[1][2] = "{Stress,NoStress}"
    
    #Experiment 2 Only descriptive stats attributes 
    WEKAexperimentlist[2][0] = "Exp2_Desc_AttributesOnly_2_Class_Stress"
    WEKAexperimentlist[2][1] = 131
    WEKAexperimentlist[2][2] = "{Stress,NoStress}"
    
    #Experiment 3 One Channel: GSR descriptive stats attributes 
    WEKAexperimentlist[3][0] = "Exp3_One_Channel_GSR_2_Class_Stress"
    WEKAexperimentlist[3][1] = 131
    WEKAexperimentlist[3][2] = "{Stress,NoStress}"
    
    #Experiment 4  One Channel: RESP descriptive stats attributes
    WEKAexperimentlist[4][0] = "Exp4_One_Channel_RESP_2_Class_Stress"
    WEKAexperimentlist[4][1] = 131
    WEKAexperimentlist[4][2] = "{Stress,NoStress}"
    
    #Experiment 5 One Channel: HR descriptive stats attributes
    WEKAexperimentlist[5][0] = "Exp5_One_Channel_HR_2_Class_Stress"
    WEKAexperimentlist[5][1] = 131
    WEKAexperimentlist[5][2] = "{Stress,NoStress}"
    
    #Experiment 6 One Channel: TEMP descriptive stats attributes
    WEKAexperimentlist[6][0] = "Exp6_One_Channel_TEMP_2_Class_Stress"
    WEKAexperimentlist[6][1] = 131
    WEKAexperimentlist[6][2] = "{Stress,NoStress}"
    
    #Experiment 7  Combined descriptive stats for all channels and complex attributes: Ratios
    WEKAexperimentlist[7][0] = "Exp7_Combo_Desc_Stats_Ratios_2_Class_Stress"
    WEKAexperimentlist[7][1] = 131
    WEKAexperimentlist[7][2] = "{Stress,NoStress}"
    
    #Experiment 8 Combined descriptive stats for all channels and complex attributes: Physiological pairs
    WEKAexperimentlist[8][0] = "Exp8_Combo_Desc_Stats_Pairs_2_Class_Stress"
    WEKAexperimentlist[8][1] = 131
    WEKAexperimentlist[8][2] = "{Stress,NoStress}"
    
    #Experiment 9 Combined descriptive stats for all channels and complex attributes: Global Nominal values
    WEKAexperimentlist[9][0] = "Exp9_Combo_Desc_Stats_Nominals_2_Class_Stress"
    WEKAexperimentlist[9][1] = 131
    WEKAexperimentlist[9][2] = "{Stress,NoStress}"
    
    #Experiment 10 Combined descriptive stats for all channels and complex attributes: Global Z-Values for means for each 4 channel
    WEKAexperimentlist[10][0] = "Exp10_Combo_Desc_Stats_ZScore_2_Class_Stress"
    WEKAexperimentlist[10][1] = 131
    WEKAexperimentlist[10][2] = "{Stress,NoStress}"


    

    
    # Get length of Weka Experiment list
    #numrowexperiments, numcol = WEKAexperimentlist.shape
    
    # =============================================================================
    #       GENERATE EXPERIMENT 1 FILES ARFF  # WekaExperiment number 1 
    # =============================================================================
    
    WekaExpNum = 1
    
        
    wekaarray = []
    
    wekaexperimentname = WEKAexperimentlist[WekaExpNum][0]
    
    
    wekaarray.append(["@relation DEAP_Participant_" + str(current_participant)])
    wekaarray.append([""])
    
    # Loop through fields in matrix, offset by +2 to get rid of participant and trial labels
    # Match Data Columns in Array participant_all_fields_matrix[0][current_field]
    # Current Active Fields 2-100 numeric, then 101-115 normative and again 116-130 numeric
    
    # column block 1
    start_field = 2
    end_field = 100
    
    for current_field in range(start_field,end_field+1):
        wekaarray.append(["@attribute " + participant_all_fields_matrix[0][current_field] + " numeric"])
        
    # column block 2
    start_field = 101
    end_field = 115
    
    for current_field in range(start_field,end_field+1):
        wekaarray.append(["@attribute " + participant_all_fields_matrix[0][current_field] + " {Below,Above,Equal}"])
    
    # column block 3
    start_field = 116
    end_field = 130
    
    for current_field in range(start_field,end_field+1):
        wekaarray.append(["@attribute " + participant_all_fields_matrix[0][current_field] + " numeric"])
    
    # Load Labels
    wekaarray.append(["@attribute target_labels " + str(WEKAexperimentlist[WekaExpNum][2])])
    
    wekaarray.append([""])
    wekaarray.append(["@data"])
    
    wekaarray.append([""])
    
    # Loop thorugh all the trials
    
    wekanumTrials = 40
    start_field = 2
    end_field = 130
    
    for current_trial in range(1,wekanumTrials + 1):
    
        # Loop through each trial fields and place in one line temp variable
        
        field_row = ""
        
        for current_field in range(start_field,end_field+1):
            
            current_field_value = str(participant_all_fields_matrix[current_trial][current_field])
            
            if current_field_value == "inf": current_field_value = 0
            if current_field_value == "-inf": current_field_value = 0
            if current_field_value == "nan": current_field_value = 0
            
                            
            field_row = field_row + str(current_field_value) + "," 
        
        # Add Target Field 
        
        field_row = field_row + "\'" + str(participant_all_fields_matrix[current_trial][WEKAexperimentlist[WekaExpNum][1]]) + "\'"
        
        wekaarray.append([field_row])
        
    
    # =============================================================================
    #     # OUT TXT LOGFILE FILE OF ALL THE StdOutput
    # =============================================================================
        
    participant_arff_name = path + output_dir + str("participant_") + str(current_participant) + "_" + wekaexperimentname + ".arff"
                
    with open(participant_arff_name, "w") as file_handler:
            for item in wekaarray:
                row = re.sub("[\[\]\'\"\"]", "",str(item))
                file_handler.write("{}\n".format(row))
        
    print("Saved Participant File:" + str("participant_") + str(current_participant) + "_" + wekaexperimentname + ".arff")   

    

    
    # =============================================================================
    #       GENERATE EXPERIMENT 2 FILES ARFF  # WekaExperiment number 2
    # =============================================================================
    
    WekaExpNum = 2
    
    
    
        
    wekaarray = []
    
    wekaexperimentname = WEKAexperimentlist[WekaExpNum][0]
    
    
    wekaarray.append(["@relation DEAP_Participant_" + str(current_participant)])
    wekaarray.append([""])
    
    # Loop through fields in matrix, offset by +2 to get rid of participant and trial labels
    # Match Data Columns in Array participant_all_fields_matrix[0][current_field]
    # Current DESCRIPTIVE Active Fields 2-26 numeric
    
    # column block 1
    start_fieldB1 = 2
    end_fieldB1 = 26
    
    for current_field in range(start_fieldB1,end_fieldB1+1):
        wekaarray.append(["@attribute " + participant_all_fields_matrix[0][current_field] + " numeric"])
        
    
    
    # Load Labels
    wekaarray.append(["@attribute target_labels " + str(WEKAexperimentlist[WekaExpNum][2])])
    
    wekaarray.append([""])
    wekaarray.append(["@data"])
    
    wekaarray.append([""])
    
    # Loop thorugh all the trials
    
    wekanumTrials = 40
    
    
    for current_trial in range(1,wekanumTrials + 1):
    
        # Loop through each trial fields and place in one line temp variable
        
        field_row = ""
        
        # For column block 1
        for current_field in range(start_fieldB1,end_fieldB1+1):
            
            current_field_value = str(participant_all_fields_matrix[current_trial][current_field])
            
            if current_field_value == "inf": current_field_value = 0
            if current_field_value == "-inf": current_field_value = 0
            if current_field_value == "nan": current_field_value = 0
            
                            
            field_row = field_row + str(current_field_value) + "," 
        
        # Add Target Field 
        
        field_row = field_row + "\'" + str(participant_all_fields_matrix[current_trial][WEKAexperimentlist[WekaExpNum][1]]) + "\'"
        
        wekaarray.append([field_row])
        
    
    # =============================================================================
    #     # OUT TXT LOGFILE FILE OF ALL THE StdOutput
    # =============================================================================
        
    participant_arff_name = path + output_dir + str("participant_") + str(current_participant) + "_" + wekaexperimentname + ".arff"
                
    with open(participant_arff_name, "w") as file_handler:
            for item in wekaarray:
                row = re.sub("[\[\]\'\"\"]", "",str(item))
                file_handler.write("{}\n".format(row))
        
    print("Saved Participant File:" + str("participant_") + str(current_participant) + "_" + wekaexperimentname + ".arff")   

    

    # =============================================================================
    #       GENERATE EXPERIMENT 3 FILES ARFF  # WekaExperiment number 3
    #       One Channel: GSR descriptive stats attributes for Stress
    # =============================================================================
    
    WekaExpNum = 3
    
    
    
        
    wekaarray = []
    
    wekaexperimentname = WEKAexperimentlist[WekaExpNum][0]
    
    
    wekaarray.append(["@relation DEAP_Participant_" + str(current_participant)])
    wekaarray.append([""])
    
    # Loop through fields in matrix, offset by +2 to get rid of participant and trial labels
    # Match Data Columns in Array participant_all_fields_matrix[0][current_field]
    # Current DESCRIPTIVE Active GSR Fields 2-7 numeric
    
    # column block 1
    start_fieldB1 = 2
    end_fieldB1 = 7
    
    for current_field in range(start_fieldB1,end_fieldB1+1):
        wekaarray.append(["@attribute " + participant_all_fields_matrix[0][current_field] + " numeric"])
        
        
    # Load Labels
    wekaarray.append(["@attribute target_labels " + str(WEKAexperimentlist[WekaExpNum][2])])
    
    wekaarray.append([""])
    wekaarray.append(["@data"])
    
    wekaarray.append([""])
    
    # Loop thorugh all the trials
    
    wekanumTrials = 40
    
    
    for current_trial in range(1,wekanumTrials + 1):
    
        # Loop through each trial fields and place in one line temp variable
        
        field_row = ""
        
        # For column block 1
        for current_field in range(start_fieldB1,end_fieldB1+1):
            
            current_field_value = str(participant_all_fields_matrix[current_trial][current_field])
            
            if current_field_value == "inf": current_field_value = 0
            if current_field_value == "-inf": current_field_value = 0
            if current_field_value == "nan": current_field_value = 0
            
                            
            field_row = field_row + str(current_field_value) + "," 
        
        # Add Target Field 
        
        field_row = field_row + "\'" + str(participant_all_fields_matrix[current_trial][WEKAexperimentlist[WekaExpNum][1]]) + "\'"
        
        wekaarray.append([field_row])
        
    
    # =============================================================================
    #     # OUT TXT LOGFILE FILE OF ALL THE StdOutput
    # =============================================================================
        
    participant_arff_name = path + output_dir + str("participant_") + str(current_participant) + "_" + wekaexperimentname + ".arff"
                
    with open(participant_arff_name, "w") as file_handler:
            for item in wekaarray:
                row = re.sub("[\[\]\'\"\"]", "",str(item))
                file_handler.write("{}\n".format(row))
        
    print("Saved Participant File:" + str("participant_") + str(current_participant) + "_" + wekaexperimentname + ".arff")   

       


    # =============================================================================
    #       GENERATE EXPERIMENT 4 FILES ARFF  # WekaExperiment number 4
    #       One Channel: RESP descriptive stats attributes for Stress
    # =============================================================================
    
    WekaExpNum = 4
    
           
    wekaarray = []
    
    wekaexperimentname = WEKAexperimentlist[WekaExpNum][0]
    
    
    wekaarray.append(["@relation DEAP_Participant_" + str(current_participant)])
    wekaarray.append([""])
    
    # Loop through fields in matrix, offset by +2 to get rid of participant and trial labels
    # Match Data Columns in Array participant_all_fields_matrix[0][current_field]
    # Current DESCRIPTIVE Active RESO Fields 8-13 numeric
    
    # column block 1
    start_fieldB1 = 8
    end_fieldB1 = 13
    
    for current_field in range(start_fieldB1,end_fieldB1+1):
        wekaarray.append(["@attribute " + participant_all_fields_matrix[0][current_field] + " numeric"])
        
        
    # Load Labels
    wekaarray.append(["@attribute target_labels " + str(WEKAexperimentlist[WekaExpNum][2])])
    
    wekaarray.append([""])
    wekaarray.append(["@data"])
    
    wekaarray.append([""])
    
    # Loop thorugh all the trials
    
    wekanumTrials = 40
    
    
    for current_trial in range(1,wekanumTrials + 1):
    
        # Loop through each trial fields and place in one line temp variable
        
        field_row = ""
        
        # For column block 1
        for current_field in range(start_fieldB1,end_fieldB1+1):
            
            current_field_value = str(participant_all_fields_matrix[current_trial][current_field])
            
            if current_field_value == "inf": current_field_value = 0
            if current_field_value == "-inf": current_field_value = 0
            if current_field_value == "nan": current_field_value = 0
            
                            
            field_row = field_row + str(current_field_value) + "," 
        
        # Add Target Field 
        
        field_row = field_row + "\'" + str(participant_all_fields_matrix[current_trial][WEKAexperimentlist[WekaExpNum][1]]) + "\'"
        
        wekaarray.append([field_row])
        
    
    # =============================================================================
    #     # OUT TXT LOGFILE FILE OF ALL THE StdOutput
    # =============================================================================
        
    participant_arff_name = path + output_dir + str("participant_") + str(current_participant) + "_" + wekaexperimentname + ".arff"
                
    with open(participant_arff_name, "w") as file_handler:
            for item in wekaarray:
                row = re.sub("[\[\]\'\"\"]", "",str(item))
                file_handler.write("{}\n".format(row))
        
    print("Saved Participant File:" + str("participant_") + str(current_participant) + "_" + wekaexperimentname + ".arff")   

       
  

    # =============================================================================
    #       GENERATE EXPERIMENT 5 FILES ARFF  # WekaExperiment number 5
    #       One Channel: HR descriptive stats attributes for Stress
    # =============================================================================
    
    WekaExpNum = 5

            
    wekaarray = []
    
    wekaexperimentname = WEKAexperimentlist[WekaExpNum ][0]
    
    
    wekaarray.append(["@relation DEAP_Participant_" + str(current_participant)])
    wekaarray.append([""])
    
    # Loop through fields in matrix, offset by +2 to get rid of participant and trial labels
    # Match Data Columns in Array participant_all_fields_matrix[0][current_field]
    # Current DESCRIPTIVE Active RESO Fields 14-21 numeric
    
    # column block 1
    start_fieldB1 = 14
    end_fieldB1 = 21
    
    for current_field in range(start_fieldB1,end_fieldB1+1):
        wekaarray.append(["@attribute " + participant_all_fields_matrix[0][current_field] + " numeric"])
        
        
    # Load Labels
    wekaarray.append(["@attribute target_labels " + str(WEKAexperimentlist[WekaExpNum][2])])
    
    wekaarray.append([""])
    wekaarray.append(["@data"])
    
    wekaarray.append([""])
    
    # Loop thorugh all the trials
    
    wekanumTrials = 40
    
    
    for current_trial in range(1,wekanumTrials + 1):
    
        # Loop through each trial fields and place in one line temp variable
        
        field_row = ""
        
        # For column block 1
        for current_field in range(start_fieldB1,end_fieldB1+1):
            
            current_field_value = str(participant_all_fields_matrix[current_trial][current_field])
            
            if current_field_value == "inf": current_field_value = 0
            if current_field_value == "-inf": current_field_value = 0
            if current_field_value == "nan": current_field_value = 0
            
                            
            field_row = field_row + str(current_field_value) + "," 
        
        # Add Target Field 
        
        field_row = field_row + "\'" + str(participant_all_fields_matrix[current_trial][WEKAexperimentlist[WekaExpNum][1]]) + "\'"
        
        wekaarray.append([field_row])
        
    
    # =============================================================================
    #     # OUT TXT LOGFILE FILE OF ALL THE StdOutput
    # =============================================================================
        
    participant_arff_name = path + output_dir + str("participant_") + str(current_participant) + "_" + wekaexperimentname + ".arff"
                
    with open(participant_arff_name, "w") as file_handler:
            for item in wekaarray:
                row = re.sub("[\[\]\'\"\"]", "",str(item))
                file_handler.write("{}\n".format(row))
        
    print("Saved Participant File:" + str("participant_") + str(current_participant) + "_" + wekaexperimentname + ".arff")   

    

    # =============================================================================
    #       GENERATE EXPERIMENT 6 FILES ARFF  # WekaExperiment number 6
    #       One Channel: TEMP descriptive stats attributes for Stress
    # =============================================================================
    
    WekaExpNum = 6
    
            
    wekaarray = []
    
    wekaexperimentname = WEKAexperimentlist[WekaExpNum][0]
    
    
    wekaarray.append(["@relation DEAP_Participant_" + str(current_participant)])
    wekaarray.append([""])
    
    # Loop through fields in matrix, offset by +2 to get rid of participant and trial labels
    # Match Data Columns in Array participant_all_fields_matrix[0][current_field]
    # Current DESCRIPTIVE Active RESO Fields 22-26 numeric
    
    # column block 1
    start_fieldB1 = 22
    end_fieldB1 = 26
    
    for current_field in range(start_fieldB1,end_fieldB1+1):
        wekaarray.append(["@attribute " + participant_all_fields_matrix[0][current_field] + " numeric"])
        
        
    # Load Labels
    wekaarray.append(["@attribute target_labels " + str(WEKAexperimentlist[WekaExpNum][2])])
    
    wekaarray.append([""])
    wekaarray.append(["@data"])
    
    wekaarray.append([""])
    
    # Loop thorugh all the trials
    
    wekanumTrials = 40
    
    
    for current_trial in range(1,wekanumTrials + 1):
    
        # Loop through each trial fields and place in one line temp variable
        
        field_row = ""
        
        # For column block 1
        for current_field in range(start_fieldB1,end_fieldB1+1):
            
            current_field_value = str(participant_all_fields_matrix[current_trial][current_field])
            
            if current_field_value == "inf": current_field_value = 0
            if current_field_value == "-inf": current_field_value = 0
            if current_field_value == "nan": current_field_value = 0
            
                            
            field_row = field_row + str(current_field_value) + "," 
        
        # Add Target Field 
        
        field_row = field_row + "\'" + str(participant_all_fields_matrix[current_trial][WEKAexperimentlist[WekaExpNum][1]]) + "\'"
        
        wekaarray.append([field_row])
        
    
    # =============================================================================
    #     # OUT TXT LOGFILE FILE OF ALL THE StdOutput
    # =============================================================================
        
    participant_arff_name = path + output_dir + str("participant_") + str(current_participant) + "_" + wekaexperimentname + ".arff"
                
    with open(participant_arff_name, "w") as file_handler:
            for item in wekaarray:
                row = re.sub("[\[\]\'\"\"]", "",str(item))
                file_handler.write("{}\n".format(row))
        
    print("Saved Participant File:" + str("participant_") + str(current_participant) + "_" + wekaexperimentname + ".arff")   
    
       
   

    # =============================================================================
    #       GENERATE EXPERIMENT 7 FILES ARFF  # WekaExperiment number 7
    #       Combined descriptive stats for all channels and complex attributes: Ratios
    # =============================================================================
    
    WekaExpNum = 7
   
             
    wekaarray = []
    
    wekaexperimentname = WEKAexperimentlist[WekaExpNum][0]
    
    
    wekaarray.append(["@relation DEAP_Participant_" + str(current_participant)])
    wekaarray.append([""])
    
    # Loop through fields in matrix, offset by +2 to get rid of participant and trial labels
    # Match Data Columns in Array participant_all_fields_matrix[0][current_field]
    # Current DESCRIPTIVE B1 Fields 2-26 B2 Fields 53-76 
    
    # column block 1
    start_fieldB1 = 2
    end_fieldB1 = 26
    
    for current_field in range(start_fieldB1,end_fieldB1+1):
        wekaarray.append(["@attribute " + participant_all_fields_matrix[0][current_field] + " numeric"])
    
    # column block 2
    start_fieldB2 = 53
    end_fieldB2 = 76
    
    for current_field in range(start_fieldB2,end_fieldB2+1):
        wekaarray.append(["@attribute " + participant_all_fields_matrix[0][current_field] + " numeric"])
        
        
    # Load Labels
    wekaarray.append(["@attribute target_labels " + str(WEKAexperimentlist[WekaExpNum][2])])
    
    wekaarray.append([""])
    wekaarray.append(["@data"])
    
    wekaarray.append([""])
    
    # Loop thorugh all the trials
    
    wekanumTrials = 40
    
    
    for current_trial in range(1,wekanumTrials + 1):
    
        # Loop through each trial fields and place in one line temp variable
        
        field_row = ""
        
        # load column block 1
        for current_field in range(start_fieldB1,end_fieldB1+1):
            
            current_field_value = str(participant_all_fields_matrix[current_trial][current_field])
            
            if current_field_value == "inf": current_field_value = 0
            if current_field_value == "-inf": current_field_value = 0
            if current_field_value == "nan": current_field_value = 0
            
                            
            field_row = field_row + str(current_field_value) + ","
        
        # load column block 2
        for current_field in range(start_fieldB2,end_fieldB2+1):
            
            current_field_value = str(participant_all_fields_matrix[current_trial][current_field])
            
            if current_field_value == "inf": current_field_value = 0
            if current_field_value == "-inf": current_field_value = 0
            if current_field_value == "nan": current_field_value = 0
            
                            
            field_row = field_row + str(current_field_value) + "," 
        
        # Add Target Field 
        
        field_row = field_row + "\'" + str(participant_all_fields_matrix[current_trial][WEKAexperimentlist[WekaExpNum][1]]) + "\'"
        
        wekaarray.append([field_row])
        
    
    # =============================================================================
    #     # OUT TXT LOGFILE FILE OF ALL THE StdOutput
    # =============================================================================
        
    participant_arff_name = path + output_dir + str("participant_") + str(current_participant) + "_" + wekaexperimentname + ".arff"
                
    with open(participant_arff_name, "w") as file_handler:
            for item in wekaarray:
                row = re.sub("[\[\]\'\"\"]", "",str(item))
                file_handler.write("{}\n".format(row))
        
    print("Saved Participant File:" + str("participant_") + str(current_participant) + "_" + wekaexperimentname + ".arff")   

    



    # =============================================================================
    #       GENERATE EXPERIMENT 8 FILES ARFF  # WekaExperiment number 8
    #       Combined descriptive stats for all channels and complex attributes: Physiological pairs
    # =============================================================================
    
    WekaExpNum = 8
    
            
    wekaarray = []
    
    wekaexperimentname = WEKAexperimentlist[WekaExpNum][0]
    
    
    wekaarray.append(["@relation DEAP_Participant_" + str(current_participant)])
    wekaarray.append([""])
    
    # Loop through fields in matrix, offset by +2 to get rid of participant and trial labels
    # Match Data Columns in Array participant_all_fields_matrix[0][current_field]
    # Current DESCRIPTIVE B1 Fields 2-26 B2 Fields 27-52 B3 77-100
    
    # column block 1
    start_fieldB1 = 2
    end_fieldB1 = 26
    
    for current_field in range(start_fieldB1,end_fieldB1+1):
        wekaarray.append(["@attribute " + participant_all_fields_matrix[0][current_field] + " numeric"])
    
    # column block 2
    start_fieldB2 = 27
    end_fieldB2 = 52
    
    for current_field in range(start_fieldB2,end_fieldB2+1):
        wekaarray.append(["@attribute " + participant_all_fields_matrix[0][current_field] + " numeric"])
    
    # column block 3
    start_fieldB3 = 77
    end_fieldB3 = 100
    
    for current_field in range(start_fieldB3,end_fieldB3+1):
        wekaarray.append(["@attribute " + participant_all_fields_matrix[0][current_field] + " numeric"])
        
    # Load Labels
    wekaarray.append(["@attribute target_labels " + str(WEKAexperimentlist[WekaExpNum][2])])
    
    wekaarray.append([""])
    wekaarray.append(["@data"])
    
    wekaarray.append([""])
    
    # Loop thorugh all the trials
    
    wekanumTrials = 40
    
    
    for current_trial in range(1,wekanumTrials + 1):
    
        # Loop through each trial fields and place in one line temp variable
        
        field_row = ""
        
        # load column block 1
        for current_field in range(start_fieldB1,end_fieldB1+1):
            
            current_field_value = str(participant_all_fields_matrix[current_trial][current_field])
            
            if current_field_value == "inf": current_field_value = 0
            if current_field_value == "-inf": current_field_value = 0
            if current_field_value == "nan": current_field_value = 0
            
                            
            field_row = field_row + str(current_field_value) + ","
        
        # load column block 2
        for current_field in range(start_fieldB2,end_fieldB2+1):
            
            current_field_value = str(participant_all_fields_matrix[current_trial][current_field])
            
            if current_field_value == "inf": current_field_value = 0
            if current_field_value == "-inf": current_field_value = 0
            if current_field_value == "nan": current_field_value = 0
            
                            
            field_row = field_row + str(current_field_value) + "," 
            
        # load column block 3
        for current_field in range(start_fieldB3,end_fieldB3+1):
            
            current_field_value = str(participant_all_fields_matrix[current_trial][current_field])
            
            if current_field_value == "inf": current_field_value = 0
            if current_field_value == "-inf": current_field_value = 0
            if current_field_value == "nan": current_field_value = 0
            
                            
            field_row = field_row + str(current_field_value) + "," 
        
        # Add Target Field 
        
        field_row = field_row + "\'" + str(participant_all_fields_matrix[current_trial][WEKAexperimentlist[WekaExpNum][1]]) + "\'"
        
        wekaarray.append([field_row])
        
    
    # =============================================================================
    #     # OUT TXT LOGFILE FILE OF ALL THE StdOutput
    # =============================================================================
        
    participant_arff_name = path + output_dir + str("participant_") + str(current_participant) + "_" + wekaexperimentname + ".arff"
                
    with open(participant_arff_name, "w") as file_handler:
            for item in wekaarray:
                row = re.sub("[\[\]\'\"\"]", "",str(item))
                file_handler.write("{}\n".format(row))
        
    print("Saved Participant File:" + str("participant_") + str(current_participant) + "_" + wekaexperimentname + ".arff")   

    
  

    # =============================================================================
    #       GENERATE EXPERIMENT 9 FILES ARFF  # WekaExperiment number 9
    #       Combined descriptive stats for all channels and complex attributes: Global Nominal values
    # =============================================================================
    
    WekaExpNum = 9
   
                
    wekaarray = []
    
    wekaexperimentname = WEKAexperimentlist[WekaExpNum][0]
    
    
    wekaarray.append(["@relation DEAP_Participant_" + str(current_participant)])
    wekaarray.append([""])
    
    # Loop through fields in matrix, offset by +2 to get rid of participant and trial labels
    # Match Data Columns in Array participant_all_fields_matrix[0][current_field]
    # Current DESCRIPTIVE B1 Fields 2-26 B2 Fields 101-115 
    
    # column block 1
    start_fieldB1 = 2
    end_fieldB1 = 26
    
    for current_field in range(start_fieldB1,end_fieldB1+1):
        wekaarray.append(["@attribute " + participant_all_fields_matrix[0][current_field] + " numeric"])
    
    # column block 2 remember nominals !{Below,Above,Equal}
    start_fieldB2 = 101
    end_fieldB2 =  115
    
    for current_field in range(start_fieldB2,end_fieldB2+1):
        wekaarray.append(["@attribute " + participant_all_fields_matrix[0][current_field] + " {Below,Above,Equal}"])
        
        
    # Load Labels
    wekaarray.append(["@attribute target_labels " + str(WEKAexperimentlist[WekaExpNum][2])])
    
    wekaarray.append([""])
    wekaarray.append(["@data"])
    
    wekaarray.append([""])
    
    # Loop thorugh all the trials
    
    wekanumTrials = 40
    
    
    for current_trial in range(1,wekanumTrials + 1):
    
        # Loop through each trial fields and place in one line temp variable
        
        field_row = ""
        
        # load column block 1
        for current_field in range(start_fieldB1,end_fieldB1+1):
            
            current_field_value = str(participant_all_fields_matrix[current_trial][current_field])
            
            if current_field_value == "inf": current_field_value = 0
            if current_field_value == "-inf": current_field_value = 0
            if current_field_value == "nan": current_field_value = 0
            
                            
            field_row = field_row + str(current_field_value) + ","
        
        # load column block 2
        for current_field in range(start_fieldB2,end_fieldB2+1):
            
            current_field_value = str(participant_all_fields_matrix[current_trial][current_field])
            
            if current_field_value == "inf": current_field_value = 0
            if current_field_value == "-inf": current_field_value = 0
            if current_field_value == "nan": current_field_value = 0
            
                            
            field_row = field_row + str(current_field_value) + "," 
        
        # Add Target Field 
        
        field_row = field_row + "\'" + str(participant_all_fields_matrix[current_trial][WEKAexperimentlist[WekaExpNum][1]]) + "\'"
        
        wekaarray.append([field_row])
        
    
    # =============================================================================
    #     # OUT TXT LOGFILE FILE OF ALL THE StdOutput
    # =============================================================================
        
    participant_arff_name = path + output_dir + str("participant_") + str(current_participant) + "_" + wekaexperimentname + ".arff"
                
    with open(participant_arff_name, "w") as file_handler:
            for item in wekaarray:
                row = re.sub("[\[\]\'\"\"]", "",str(item))
                file_handler.write("{}\n".format(row))
        
    print("Saved Participant File:" + str("participant_") + str(current_participant) + "_" + wekaexperimentname + ".arff")   

    


    # =============================================================================
    #       GENERATE EXPERIMENT 10 FILES ARFF  # WekaExperiment number 27-29
    #       Combined descriptive stats for all channels and complex attributes: Global Z-Values for means for each 4 channel
    # =============================================================================
    
    WekaExpNum = 10
  
            
    wekaarray = []
    
    wekaexperimentname = WEKAexperimentlist[WekaExpNum][0]
    
    
    wekaarray.append(["@relation DEAP_Participant_" + str(current_participant)])
    wekaarray.append([""])
    
    # Loop through fields in matrix, offset by +2 to get rid of participant and trial labels
    # Match Data Columns in Array participant_all_fields_matrix[0][current_field]
    # Current DESCRIPTIVE B1 Fields 2-26 B2 Fields 116-130 
    
    # column block 1
    start_fieldB1 = 2
    end_fieldB1 = 26
    
    for current_field in range(start_fieldB1,end_fieldB1+1):
        wekaarray.append(["@attribute " + participant_all_fields_matrix[0][current_field] + " numeric"])
    
    # column block 2
    start_fieldB2 = 116
    end_fieldB2 =  130
    
    for current_field in range(start_fieldB2,end_fieldB2+1):
        wekaarray.append(["@attribute " + participant_all_fields_matrix[0][current_field] + " numeric"])
        
        
    # Load Labels
    wekaarray.append(["@attribute target_labels " + str(WEKAexperimentlist[WekaExpNum][2])])
    
    wekaarray.append([""])
    wekaarray.append(["@data"])
    
    wekaarray.append([""])
    
    # Loop thorugh all the trials
    
    wekanumTrials = 40
    
    
    for current_trial in range(1,wekanumTrials + 1):
    
        # Loop through each trial fields and place in one line temp variable
        
        field_row = ""
        
        # load column block 1
        for current_field in range(start_fieldB1,end_fieldB1+1):
            
            current_field_value = str(participant_all_fields_matrix[current_trial][current_field])
            
            if current_field_value == "inf": current_field_value = 0
            if current_field_value == "-inf": current_field_value = 0
            if current_field_value == "nan": current_field_value = 0
            
                            
            field_row = field_row + str(current_field_value) + ","
        
        # load column block 2
        for current_field in range(start_fieldB2,end_fieldB2+1):
            
            current_field_value = str(participant_all_fields_matrix[current_trial][current_field])
            
            if current_field_value == "inf": current_field_value = 0
            if current_field_value == "-inf": current_field_value = 0
            if current_field_value == "nan": current_field_value = 0
            
                            
            field_row = field_row + str(current_field_value) + "," 
        
        # Add Target Field 
        
        field_row = field_row + "\'" + str(participant_all_fields_matrix[current_trial][WEKAexperimentlist[WekaExpNum][1]]) + "\'"
        
        wekaarray.append([field_row])
        
    
    # =============================================================================
    #     # OUT TXT LOGFILE FILE OF ALL THE StdOutput
    # =============================================================================
        
    participant_arff_name = path + output_dir + str("participant_") + str(current_participant) + "_" + wekaexperimentname + ".arff"
                
    with open(participant_arff_name, "w") as file_handler:
            for item in wekaarray:
                row = re.sub("[\[\]\'\"\"]", "",str(item))
                file_handler.write("{}\n".format(row))
        
    print("Saved Participant File:" + str("participant_") + str(current_participant) + "_" + wekaexperimentname + ".arff")   

     


# Print END TIME
end = time.time()
print(end - start)

