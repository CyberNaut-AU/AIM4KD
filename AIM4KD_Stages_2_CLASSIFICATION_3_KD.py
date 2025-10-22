
"""
Created on Sun 25 April 2021

# Built On Python version 3.7

# AIM4KD - Agnostic Interpretable Machine Learning for Knowledge Discovery

# Author: Dr. Nectarios Costadopoulos  

# Repository: github.com/CyberNaut-AU/AIM4KD

"""

# ====================================================================================================================
# AIM4KD AGNOSTIC INTEPRETABLE MACHINE LEARNING 4 KNOWLEDGE DISCOVERY STAGE 2 & 3 CLASSIFICATION & KNOWLEDGE DISCOVERY
# ====================================================================================================================



# IMPORTS FROM IMPORTANT LIBRARIES

import csv

import os

import numpy as np

from numpy import genfromtxt

from scipy import stats

import time # measure time

import re  #import regular expression library

# Declare functions used from exernal functions file
# CURRENT FUNCTIONS IN AIM4KD_Core

#rolling_window(a, window)
from AIM4KD_Core import rolling_window

# Get the number items in array = get_number_of_elementsinArray(list)
from AIM4KD_Core import get_number_of_elementsinArray


# =============================================================================
#                      DECLARE GLOBAL CONFIG VARIABLES
#     
# =============================================================================

# Get the directory of the current script
path= os.path.dirname(os.path.abspath(__file__)) + "/"

# Define OS path to installed weka.jar
# Ensure J48, CART, RandomForest and SySfor Tree Algorithms/packages are installed on Weka
wekajarpath =  "-classpath /Applications/weka-3-8-1/weka.jar"

# Define relative directories

# Subdirectory with Weka ARFF Participant Files
raw_datasets_dir = "DEAP_Preprocessed_Trials_P1/"

# Output subdirectory for Trees and Rules
output_dir = "AIM4KD_Classification_Output/"


# Define how many Participants
NumParticipants = 1   


# =============================================================================
#      Create Experiment Array for Classification
# =============================================================================

# CREATE MIXED NUMPY ARRAY for Stress/NoStress experiments only: 10 ROWS and 3 COLUMNS

# CREATE 1-COLUMN ARRAY (10 ROWS) WITH ONLY THE EXPERIMENT NAMES FOR Stress/NoStress
WEKAexperimentlist = np.zeros((10,1)).astype(object)



WEKAexperimentlist[0][0] = "Exp1_Combined_Attributes_2_Class_Stress"
WEKAexperimentlist[1][0] = "Exp2_Desc_AttributesOnly_2_Class_Stress"
WEKAexperimentlist[2][0] = "Exp3_One_Channel_GSR_2_Class_Stress"
WEKAexperimentlist[3][0] = "Exp4_One_Channel_RESP_2_Class_Stress"
WEKAexperimentlist[4][0] = "Exp5_One_Channel_HR_2_Class_Stress"
WEKAexperimentlist[5][0] = "Exp6_One_Channel_TEMP_2_Class_Stress"
WEKAexperimentlist[6][0] = "Exp7_Combo_Desc_Stats_Ratios_2_Class_Stress"
WEKAexperimentlist[7][0] = "Exp8_Combo_Desc_Stats_Pairs_2_Class_Stress"
WEKAexperimentlist[8][0] = "Exp9_Combo_Desc_Stats_Nominals_2_Class_Stress"
WEKAexperimentlist[9][0] = "Exp10_Combo_Desc_Stats_ZScore_2_Class_Stress"




# # =============================================================================
# #                 CLASSIFICATION USING WEKA AND KNOWLEDGE DISCOVERY
# #                       PREPROCESSING & CSV/ARFF CREATION
# # =============================================================================


def J48_KD():
    
    ## IMPORTS
    #Classification
    from AIM4KD_Core import WekaJ48

    # KD Rule Discovery
    from AIM4KD_Core import J48_genKDfunction

    # Save KD Rules and J48 prrocessing file (a, window)
    from AIM4KD_Core import OutputRuleArray
    
    # Start TIMER
    start = time.time()
    
    # get classification participant numbers
    Classification_Participants_Num = NumParticipants
    
    # WEKA Classificationa and RULE DISCOVERY FOR 
    
    ExperimentName = "J48_PROCESSED_SIGNALS_KD_RULES_GlobalVariables"
     
    for current_participant in range(1,Classification_Participants_Num +1):
        
        # Get length of Weka Experiment list
        numrowexperiments, numcol = WEKAexperimentlist.shape
            
        for current_wekaexperiment in range(0,numrowexperiments):
            
            # Define which experiment arff files to read and write
            wekaexperimentname = WEKAexperimentlist[current_wekaexperiment][0]
    
            # using global path variable for current input/output directory and the output directory with the arff files
            
            dataset = str("'") + path + raw_datasets_dir + "participant_" + str(current_participant) + "_" + wekaexperimentname + ".arff" + str("'") 
        
            # Perform classification using different minimum objects increment by two
            
            # Setup Array
            J48_WekaTreeExperiment=[]
            
            # Minimum 2 max 22 Up to 20 records per leaf
            minObj = 2
            maxObj = 22
            loopStep = 2
            
            for numrecords in range(minObj, maxObj, loopStep):
                # Execute WEKA Tree Algorithm and Extract Result in Array / Calling WekaJ48 Function
                J48_WekaTreeExperiment.append(WekaJ48(wekajarpath,dataset,numrecords))
                

            # LOAD Participant Trees into an Array for Dumping in Textfile
            ParticipantTrees = []
            
            # Check how many experiments were done
            lengthofexperiments= len(J48_WekaTreeExperiment)
            
            for numexper in range(0, lengthofexperiments):
                ParticipantTrees.append(J48_WekaTreeExperiment[numexper])
                
                    
            # Save Current Participants Trees in a Text File
            
            ParticipantTreeFilename = path + output_dir + "J48_WEKATrees_" + wekaexperimentname + "_Participant_" + str(current_participant) + ".txt"
            
            with open(ParticipantTreeFilename, "w") as file_handler:
                    for item in ParticipantTrees:
                        row = re.sub("[\[\]\']", "",str(item))
                        file_handler.write("{}\n".format(row))
            
            #KD Participant Label
            KDParticipant = str(current_participant)
            
            # Perform knowledge discovery using different minimum objects looping through all trees produced
            
            for numexper in range(0, lengthofexperiments):
                # Calling J48 KD Function to loop through all trees produced by WEKA and Extract rules
                J48_genKDfunction(J48_WekaTreeExperiment[numexper],KDParticipant,wekaexperimentname)
           
            
    # =============================================================================
    # Output KD J48 Rules using global path variable for current input/output directory
    # =============================================================================
    
    
        
    # using global path variable for current input/output directory and the output directory with the arff files
    KDOutputDir = path + output_dir
        
    OutputRuleArray(KDOutputDir,ExperimentName)

       
    # Print END TIME
    end = time.time()
    print(end - start)


def CART_KD():
    
    ## IMPORTS
    #Classification
    from AIM4KD_Core import WekaCART

    # KD Rule Discovery
    from AIM4KD_Core import CART_genKDfunction

    # Save KD Rules and CART prrocessing file (a, window)
    from AIM4KD_Core import OutputRuleArray
    
    # Start TIMER
    start = time.time()
    
    # get classification participant numbers
    Classification_Participants_Num = NumParticipants
    
    # WEKA Classificationa and RULE DISCOVERY FOR 
    
    ExperimentName = "CART_PROCESSED_SIGNALS_KD_RULES_GlobalVariables"
     
    for current_participant in range(1,Classification_Participants_Num +1):
        
        # Get length of Weka Experiment list
        numrowexperiments, numcol = WEKAexperimentlist.shape
            
        for current_wekaexperiment in range(0,numrowexperiments):
            
            # Define which experiment arff files to read and write
            wekaexperimentname = WEKAexperimentlist[current_wekaexperiment][0]
    
            # using global path variable for current input/output directory and the output directory with the arff files
            
            dataset = str("'") + path + raw_datasets_dir + "participant_" + str(current_participant) + "_" + wekaexperimentname + ".arff" + str("'") 
        
            # Perform classification using different minimum objects increment by two
            
            # Setup Array
            CART_WekaTreeExperiment=[]
            
            # Minimum 2 max 22 Up to 20 records per leaf
            minObj = 2
            maxObj = 22
            loopStep = 2
            
            for numrecords in range(minObj, maxObj, loopStep):
                # Execute WEKA Tree Algorithm and Extract Result in Array / Calling WekaCART Function
                CART_WekaTreeExperiment.append(WekaCART(wekajarpath,dataset,numrecords))
                

            # LOAD Participant Trees into an Array for Dumping in Textfile
            ParticipantTrees = []
            
            # Check how many experiments were done
            lengthofexperiments= len(CART_WekaTreeExperiment)
            
            for numexper in range(0, lengthofexperiments):
                ParticipantTrees.append(CART_WekaTreeExperiment[numexper])
                
                    
            # Save Current Participants Trees in a Text File
            
            ParticipantTreeFilename = path + output_dir + "CART_WEKATrees_" + wekaexperimentname + "_Participant_" + str(current_participant) + ".txt"
            
            with open(ParticipantTreeFilename, "w") as file_handler:
                    for item in ParticipantTrees:
                        row = re.sub("[\[\]\']", "",str(item))
                        file_handler.write("{}\n".format(row))
            
            #KD Participant Label
            KDParticipant = str(current_participant)
            
            # Perform knowledge discovery using different minimum objects looping through all trees produced
            
            for numexper in range(0, lengthofexperiments):
                # Calling CART KD Function to loop through all trees produced by WEKA and Extract rules
                CART_genKDfunction(CART_WekaTreeExperiment[numexper],KDParticipant,wekaexperimentname)
           
            
    # =============================================================================
    # #Output KD CART Rules using global path variable for current input/output directory
    # =============================================================================
    
    
        
    # using global path variable for current input/output directory and the output directory with the arff files
    KDOutputDir = path + output_dir
        
    OutputRuleArray(KDOutputDir,ExperimentName)

    #clear Array
    global RulesArray
    RulesArray = []
    
    
    # Print END TIME
    end = time.time()
    print(end - start)


def RandomForest_KD():
    
    ## IMPORTS
    #Classification
    from AIM4KD_Core import WekaRandomForest

    # KD Rule Discovery
    from AIM4KD_Core import RandomForest_genKDfunction

    # Save KD Rules and RandomForest prrocessing file (a, window)
    from AIM4KD_Core import OutputRuleArray
    
    # Start TIMER
    start = time.time()
    
    # get classification participant numbers
    Classification_Participants_Num = NumParticipants
    
    # WEKA Classificationa and RULE DISCOVERY FOR 
    
    ExperimentName = "RandomForest_PROCESSED_SIGNALS_KD_RULES_GlobalVariables"
     
    for current_participant in range(1,Classification_Participants_Num +1):
        
        # Get length of Weka Experiment list
        numrowexperiments, numcol = WEKAexperimentlist.shape
            
        for current_wekaexperiment in range(0,numrowexperiments):
            
            # Define which experiment arff files to read and write
            wekaexperimentname = WEKAexperimentlist[current_wekaexperiment][0]
    
            # using global path variable for current input/output directory and the output directory with the arff files
            
            dataset = str("'") + path + raw_datasets_dir + "participant_" + str(current_participant) + "_" + wekaexperimentname + ".arff" + str("'") 
        
            # Perform classification using different minimum objects increment by two
            
            # Setup Array
            RandomForest_WekaTreeExperiment=[]
            
            # Minimum 2 max 22 Up to 20 records per leaf
            minObj = 2
            maxObj = 22
            loopStep = 2
            
            for numrecords in range(minObj, maxObj, loopStep):
                # Execute WEKA Tree Algorithm and Extract Result in Array / Calling WekaRandomForest Function
                RandomForest_WekaTreeExperiment.append(WekaRandomForest(wekajarpath,dataset,numrecords))
                

            # LOAD Participant Trees into an Array for Dumping in Textfile
            ParticipantTrees = []
            
            # Check how many experiments were done
            lengthofexperiments= len(RandomForest_WekaTreeExperiment)
            
            for numexper in range(0, lengthofexperiments):
                ParticipantTrees.append(RandomForest_WekaTreeExperiment[numexper])
                
                    
            # Save Current Participants Trees in a Text File
            
            ParticipantTreeFilename = path + output_dir + "RandomForest_WEKATrees_" + wekaexperimentname + "_Participant_" + str(current_participant) + ".txt"
            
            with open(ParticipantTreeFilename, "w") as file_handler:
                    for item in ParticipantTrees:
                        row = re.sub("[\[\]\']", "",str(item))
                        file_handler.write("{}\n".format(row))
            
            #KD Participant Label
            KDParticipant = str(current_participant)
            
            # Perform knowledge discovery using different minimum objects looping through all trees produced
            
            for numexper in range(0, lengthofexperiments):
                # Calling RandomForest KD Function to loop through all trees produced by WEKA and Extract rules
                RandomForest_genKDfunction(RandomForest_WekaTreeExperiment[numexper],KDParticipant,wekaexperimentname)
           
            
    # =============================================================================
    # #Output KD RandomForest Rules using global path variable for current input/output directory
    # =============================================================================
    
    
        
    # using global path variable for current input/output directory and the output directory with the arff files
    KDOutputDir = path + output_dir
        
    OutputRuleArray(KDOutputDir,ExperimentName)
    
    # Print END TIME
    end = time.time()
    print(end - start)


def SySFor_KD():
    
    ## IMPORTS
    #Classification
    from AIM4KD_Core import WekaSySFor
    
    # KD Rule Discovery
    from AIM4KD_Core import SySFor_genKDfunction

    # Save KD Rules and SySFor prrocessing file (a, window)
    from AIM4KD_Core import OutputRuleArray
    
    # Start TIMER
    start = time.time()
    
    # get classification participant numbers
    Classification_Participants_Num = NumParticipants
    
    # WEKA Classificationa and RULE DISCOVERY FOR 
    
    ExperimentName = "SySFor_PROCESSED_SIGNALS_KD_RULES_GlobalVariables"
     
    for current_participant in range(1,Classification_Participants_Num +1):
        
        # Get length of Weka Experiment list
        numrowexperiments, numcol = WEKAexperimentlist.shape
            
        for current_wekaexperiment in range(0,numrowexperiments):
            
            # Define which experiment arff files to read and write
            wekaexperimentname = WEKAexperimentlist[current_wekaexperiment][0]
    
            # using global path variable for current input/output directory and the output directory with the arff files
            
            dataset = str("'") + path + raw_datasets_dir + "participant_" + str(current_participant) + "_" + wekaexperimentname + ".arff" + str("'") 
        
            # Perform classification using different minimum objects increment by two
            
            # Setup Array
            SySFor_WekaTreeExperiment=[]
            
            # Minimum 2 max 22 Up to 20 records per leaf
            minObj = 2
            maxObj = 22
            loopStep = 2
            
            for numrecords in range(minObj, maxObj, loopStep):
                # Execute WEKA Tree Algorithm and Extract Result in Array / Calling WekaSySFor Function
                SySFor_WekaTreeExperiment.append(WekaSySFor(wekajarpath,dataset,numrecords))
                
            # LOAD Participant Trees into an Array for Dumping in Textfile
            ParticipantTrees = []
            
            # Check how many experiments were done
            lengthofexperiments= len(SySFor_WekaTreeExperiment)
            
            for numexper in range(0, lengthofexperiments):
                ParticipantTrees.append(SySFor_WekaTreeExperiment[numexper])
                
                    
            # Save Current Participants Trees in a Text File
            
            ParticipantTreeFilename = path + output_dir + "SySFor_WEKATrees_" + wekaexperimentname + "_Participant_" + str(current_participant) + ".txt"
            
            with open(ParticipantTreeFilename, "w") as file_handler:
                    for item in ParticipantTrees:
                        row = re.sub("[\[\]\']", "",str(item))
                        file_handler.write("{}\n".format(row))
            
            #KD Participant Label
            KDParticipant = str(current_participant)
            
            # Perform knowledge discovery using different minimum objects looping through all trees produced
            
            for numexper in range(0, lengthofexperiments):
                # Calling SysFor KD Function to loop through all trees produced by WEKA and Extract rules
                SySFor_genKDfunction(SySFor_WekaTreeExperiment[numexper],KDParticipant,wekaexperimentname)
           
            
    # =============================================================================
    # #Output KD SySFor Rules using global path variable for current input/output directory
    # =============================================================================
    
    
        
    # using global path variable for current input/output directory and the output directory with the arff files
    KDOutputDir = path + output_dir
        
    OutputRuleArray(KDOutputDir,ExperimentName)
    
    # Print END TIME
    end = time.time()
    print(end - start)




def main():
    # =============================================================================
    #                      EXECUTE KD FUNCTIONS 
    #     
    # =============================================================================

    
    #Establish New Rules Array after each Classification Cycle
    from AIM4KD_Core import Reset_Rules_Array
    from AIM4KD_Core import Output_KD_Rule_Bank

    # Reset Global Rules Array to store rules per classification cycle 
    Reset_Rules_Array()

    #RUN J48 KD
    J48_KD()
    Reset_Rules_Array()

    #RUN CART KD
    CART_KD()
    Reset_Rules_Array()

    #RUN RF KD
    RandomForest_KD()
    Reset_Rules_Array()

    #RUN SysFor KD
    SySFor_KD()
    Reset_Rules_Array()

    # After Classification Combine All Classification Output into a KD Rule Bank
    Output_KD_Rule_Bank(output_dir)

if __name__ == "__main__":
    main()
