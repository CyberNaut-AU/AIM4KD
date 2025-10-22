#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

# Built On Python version 3.7

# AIM4KD - Agnostic Interpretable Machine Learning for Knowledge Discovery

# Author: Dr. Nectarios Costadopoulos  


1. CoreFunctions for All Algorithms
2. Included all 4 KD functions
3. Included ONE rule recording and output function to be shared by ALL algos

"""
#Universal Rolling Window Function, use np.std and np.mean to calculate 
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


# Fix Rounding up of 0.5 numbers
import math
def normal_round(n):
    if n - math.floor(n) < 0.5:
        return math.floor(n)
    return math.ceil(n)



# =============================================================================
# WEKA Tree Classifiers Functions
# =============================================================================  

def WekaJ48(wekajarpath,dataset,minrecords):
    import os
    command = "java"
    wekaclassifier = " weka.classifiers.trees.J48"
    wekaclassifer_options = " -M " + str(minrecords)
    datasetpath = " -t " + dataset
    errormsglog = " 2> /dev/null"
    command = "java " + wekajarpath + wekaclassifier + wekaclassifer_options + datasetpath + errormsglog
    print ("Starting J48 Generation" + command)
    stream = os.popen(command)
    print ("Finished J48!")
    output = stream.read()
    return output 

def WekaCART(wekajarpath,dataset,minrecords):
    import os
    minrecords = float(minrecords) # convert into float d.d
    command = "java"
    classifierjar_path = ":/$HOME/wekafiles/packages/simpleCART/simpleCART.jar"
    wekaclassifier = " weka.Run -no-load -no-scan weka.classifiers.trees.SimpleCart"
    wekaclassifer_options = " -U -M " + str(minrecords) # Activated Unprunning flag to show tree usePrune -- Use minimal cost-complexity pruning (default yes)
    datasetpath = " -t " + dataset
    errormsglog = " 2> /dev/null"
    command = "java " + wekajarpath + classifierjar_path + wekaclassifier + wekaclassifer_options + datasetpath + errormsglog
    print ("Starting CART Generation " + command)
    stream = os.popen(command)
    print ("Finished CART!")
    output = stream.read()
    return output 

def WekaRandomForest(wekajarpath,dataset,minrecords):
    import os
    minrecords = float(minrecords) # convert into float d.d
    command = "java"
    wekaclassifier = " weka.classifiers.trees.RandomForest"
    wekaclassifer_options = " -print -I 10 -M " + str(minrecords) # Print 10 classifiers
    datasetpath = " -t " + dataset
    errormsglog = " 2> /dev/null"
    command = "java " + wekajarpath + wekaclassifier + wekaclassifer_options + datasetpath + errormsglog
    print ("Starting RandomForest Generation" + command)
    stream = os.popen(command)
    print ("Finished RandomForest!")
    output = stream.read()
    return output 

def WekaSySFor(wekajarpath,dataset,minrecords):
    import os
    #all commands and options have one space, running advanced weka run option, with classifier directly referenced
    #ensure SySfor Package has been downloaded, it is not included by default 
    command = "java"
    classifierjar_path = ":/$HOME/wekafiles/packages/SysFor/SysFor.jar"
    wekaclassifier = " weka.Run -no-load -no-scan weka.classifiers.trees.SysFor"
    wekaclassifer_options = " -N 10 -L " + str(minrecords) # The minimum number of records is determined bv -L and number of trees -N
    datasetpath = " -t " + dataset
    errormsglog = " 2> /dev/null"
    command = "java " + wekajarpath + classifierjar_path + wekaclassifier + wekaclassifer_options + datasetpath + errormsglog
    print ("Starting SySFor Generation " + command)
    stream = os.popen(command)
    print ("Finished SySFor!")
    output = stream.read()
    return output 



# =============================================================================
# RECORD Rules Function
# =============================================================================   
    

def Reset_Rules_Array():
    global RulesArray
    RulesArray = [] 
       
    # DEFINE GLOBAL COUNT VARIABLES IF THEY DON'T EXIST
    global CountLocalRules
    CountLocalRules = 0

    global NumberofRulesProcessed
    NumberofRulesProcessed = 0

    # Establish Arrray Matrix with 7 columns
    # 0) Participant_Num 1) Rule 2) Rule Length 3) Support 4) Accuracy 5) Lift 6) Leaf Records 
    # 7) Leaf Misclassifed 8) Target Class 9) Number of Target Class 10) Target Ratio
    # 11) Instances 12) Attributes 13) Num_Leaves 14) Model Training Accuracy 15) Model Stratified Accuracy
    # 16) Model Scheme 17) Weka_Experiment_Name


    RulesArray=[['Participant_Num',
                    'Rule',
                    'Rule Length',
                    'Rule Support %',
                    'Rule Accuracy %',
                    'Rule Lift',
                    'Leaf Records',
                    'Leaf Misclassified',
                    'Target Class',
                    'Target Class Number',
                    'Rule Recall %',
                    'Instances',
                    'Number of Rules',
                    'Model Training Accuracy',
                    'Model Stratified Accuracy',
                    'Model Scheme',
                    'Weka_Experiment_Name',
                    'Model Algorithm',
                    'SCORE'
                    ]]




def RecordRule(Current_Rule):
    import re
    
    global RulesArray
    global CountLocalRules

    CountLocalRules = CountLocalRules + 1
            
    RulesArray.append([])

    lengthRulesArray = len(RulesArray[0])

    # Generate new Columns to match Title Columns
    for columns in range(lengthRulesArray): 
        RulesArray[CountLocalRules].append([])
        
    
    # Read the length of the rule, search for the number of <>= contained in the rules
    Loadcriteria = re.findall(r"([<>=]+)",str(Current_Rule))
    RuleLength = len(Loadcriteria)

    
    # Read Target Class from rule ic : Iris-versicolor(47.0/1.0) no space between target and bracket ()
    TargetClass = re.findall(r"[:]\s+(\w+)",str(Current_Rule))
    #CONVERT LIST TO STRING
    TargetClass = str(re.sub("[\[\]\']", "",str(TargetClass)))
    
    # Load Target Class Number from Global Node ie extracted from the confusion matrix of the tree ie PosValence
    TargetClassNumber = Nodedata[TargetClass]
    
    # Load Number of Instances in Dataset
    Total_Instances = Nodedata["Model_Instances"]
    
    # Read the number of lead records from rule (XX/YY) 
    #Flexible regex for finding Support with or without Misclassification
        
    
    LeafRecords = re.findall(r"\((\d*)[.]*\d*[/]?\d*[.]*\d*\)",str(Current_Rule))
    
    #CONVERT LIST TO STRING TO FLOAT = re.sub("[\[\]\']", "",str(Level0Node))
    LeafRecords = str(re.sub("[\[\]\']", "",str(LeafRecords)))
    LeafRecords = float(LeafRecords)
    
    LeafMisclassifed = re.findall(r"\(\d*[.]*\d*[/]?(\d*)[.]*\d*\)",str(Current_Rule))
    LeafMisclassifed = str(re.sub("[\[\]\']", "",str(LeafMisclassifed)))
    
    # Check if there is no misclassifications, and zero for the accuracy calculation
    if LeafMisclassifed  == "" : LeafMisclassifed = 0
    else: float(LeafMisclassifed) 
    
    
    # Check if there are no leaf records ie HR_Mean_GlobalRef = Equal: PosValence (0.0), dont calculate any support/accuracy/lift
    
    # Default values for support, accuracy and lift and TargetRatio
    RuleSupport = 0
    RuleAccuracy = 0
    RuleLift = 0
    RuleRecall = 0
    RuleScore = 0
    
    if LeafRecords !=0:
        # Calculate Rule Support 2 Decimals      
        RuleSupport = int(((float(LeafRecords)/float(Total_Instances))*100))
       
        # Calculate Rule Accuracy 2 Decimals
        RuleAccuracy = int((((float(LeafRecords)-float(LeafMisclassifed))/float(LeafRecords))*100))
              
        # Calculate LIFT of Rule = Rule accuracy / Probabilty of class
        RuleLift =  round((float(RuleAccuracy)/float(Nodedata["Probabilityof"+ str(TargetClass)])),1)
        
        # Calculate Rule Recall
        RuleRecall = int((float(LeafRecords)-float(LeafMisclassifed))/float(TargetClassNumber)*100)
        
        # Load Training Accuracy as INT TA and STRATFIED ACCURACY SA
        TA = int(Nodedata["Model_TrainingAccuracy"].strip('%'))
        SA = int(Nodedata["Model_StratifiedAccuracy"].strip('%'))

        # CALCULATE SCORE

        # USING SCORE EQUATION 3 (Ax2, Rx1.5, Sx0.5, TA x0.5, SA x0.5)
        RuleScore = round(((RuleAccuracy/100)*2) + ((RuleSupport/100)*0.5) + ((RuleRecall/100)*1.5) + ((TA/100)*0.5) + ((SA/100)*0.5),2)

        # Convert Accuracy and Support to String with %
        RuleSupport  = str(RuleSupport ) + "%"
        RuleAccuracy = str(RuleAccuracy) + "%"
        RuleRecall = str(RuleRecall) + "%"
        
            

    algo_flags = Nodedata.get("Model_Scheme", "")

    s = str(algo_flags).strip()
    s = s.strip("[]").replace("'", "").replace('"', "")
    tokens = s.split()

    Classification_Algo = None
    for tok in tokens:
        t = tok.lower()
        if t.startswith("-n"):
            Classification_Algo = "SySFor"
            break
        if t.startswith("-i"):
            Classification_Algo = "RandomForest"
            break
        if t.startswith("-u"):
            Classification_Algo = "CART"
            break
        if t.startswith("-m"):
            Classification_Algo = "J48"
            break
        # Classification_Algo now holds the detected algorithm or None





    # Record in rules array rule and stats
                
    RulesArray[CountLocalRules][0] = Nodedata["KDParticipant"] # global variable
    RulesArray[CountLocalRules][1] = Current_Rule # global variable
    RulesArray[CountLocalRules][2] = RuleLength
    RulesArray[CountLocalRules][3] = RuleSupport
    RulesArray[CountLocalRules][4] = RuleAccuracy
    RulesArray[CountLocalRules][5] = RuleLift
    
    RulesArray[CountLocalRules][6] = LeafRecords
    RulesArray[CountLocalRules][7] = LeafMisclassifed 
    
    RulesArray[CountLocalRules][8] = TargetClass
    RulesArray[CountLocalRules][9] = TargetClassNumber
    RulesArray[CountLocalRules][10] = RuleRecall
    
    RulesArray[CountLocalRules][11] = Nodedata["Model_Instances"] # global variable
    RulesArray[CountLocalRules][12] = Nodedata["Model_NumberofRules"] # global variable
    RulesArray[CountLocalRules][13] = Nodedata["Model_TrainingAccuracy"] # global variable
    RulesArray[CountLocalRules][14] = Nodedata["Model_StratifiedAccuracy"] # global variable
    RulesArray[CountLocalRules][15] = Nodedata["Model_Scheme"] # global variable
    RulesArray[CountLocalRules][16] = Nodedata["wekaexperimentname"] # global variable
    
    RulesArray[CountLocalRules][17] = Classification_Algo  # Classification_Algo used
    RulesArray[CountLocalRules][18] = RuleScore # Score Equation 3
    
    print ("RECORDED >>" + Current_Rule)
    
   
    # =============================================================================
    # END OF RecordRule FUNCTION
    # =============================================================================

# =============================================================================
# KD Tree/Forest Rule Extractions Function
# =============================================================================   

# =============================================================================
# J48 KD Function
# =============================================================================  

def J48_genKDfunction(J48,KDParticipant,wekaexperimentname):

    # Declare GLOBAL LogfileClassificationArray for recording events
    
    import re  #import regular expression library
    import datetime
        
    global LogfileClassificationArray
    

    # Declare LOCAL Nodedata dictionary for storing temporary branch and tree data
    global Nodedata
    Nodedata = {}
            
    
    try: LogfileClassificationArray
    except NameError:
        #Initialise LogFile First it runs
        LogfileClassificationArray = []

    now = datetime.datetime.now()
    print ("Current date and time : " + now.strftime("%Y-%m-%d %H:%M:%S"))
    LogfileClassificationArray.append(["Current date and time : " + now.strftime("%Y-%m-%d %H:%M:%S")])
    
    #Store in Nodedata Key Function Values
    Nodedata["KDParticipant"] = KDParticipant
    Nodedata["wekaexperimentname"] = wekaexperimentname 
    
    # =============================================================================
    # Capture Node Rule Info using custom filter
    # =============================================================================   
    
    
    
    def CaptureNode(NodeType,Line):
        
        global Nodeline
        
        if NodeType == "Normal":
            
            # Capture a normal node like Ch37_PGSD <= 0.10518
                       
            # This regex patern targets the attribute name and the operants such as =>< and copies all the criterion be it numeric or text
            Nodeline = re.findall(r"(\w+\s+[><=]+.*)",str(line))
            
            return Nodeline
            
            
        if NodeType == "CheckTerminal":
            print ('YES CHECK TERMINAL')
            # check if the node is terminated by (d.d/d.d)
            Nodeline = re.search(r"(\(.*\))",str(line))
            # Return boolean
            return Nodeline
        
        if NodeType == "Terminal":
            
            
            # Looks for Criterion optional [-] optional [.] and optional decimal place
                        
            # This regex patern targets the attribute name and the operants such as =>< and copies all the criterion be it numeric or text
            Nodeline = re.findall(r"(\w+\s+[><=]+.*)",str(line))
            
            return Nodeline
       
        
        
        # =============================================================================
        # END OF CAPTURE NODE FILTER FUNCTION
        # =============================================================================
        
    
    # CHECK IF THE TREE IS NOT EMPTY ie ------------------ : PosValence (40.0/20.0)
    # IF IT IS WIPE IT SO THE ALGORTHM WILL SKIP
    J48_check = re.search(r"\s([-]+\s[:])",J48)
    
    if J48_check: 
        print("No Tree has been produced - can not process")
        print(J48)
        LogfileClassificationArray.append(["No Tree has been produced - can not process"])
        LogfileClassificationArray.append([J48])
  
    # =============================================================================
    # PROCEED TO PROCESS THE TREE!
    # =============================================================================
  
    else:
        
        
        # Read Tree Attributes
        # Cut Cross Validation Results, leave only training data in J48
       
    
        
        global J48_Training
        # Capture from the begining of the tree output "Options.." to just before the "=== Stratified cross-validation ==="
        J48_Training = re.findall(r"\s(Options.*)\s=== Stratified cross-validation ===",J48,re.DOTALL)
        J48_Training = J48_Training[0] # Convert from list to string
        
              
        global CleanJ48
        CleanJ48 = "".join([s for s in J48_Training.splitlines(True) if s.strip("\r\n")])
        
        ### Extract Important Fields from Training Tree ###
        J48_Scheme = re.findall(r"Options:\s+(.*)\s",CleanJ48)
        
        Nodedata["Model_Scheme"] = J48_Scheme # Load in Global Dictionary

        LogfileClassificationArray.append([J48_Scheme])
        
        global J48_Instances
        J48_Instances = re.findall(r"Total Number of Instances\s+(\d+)",CleanJ48)
        # Clean up value and convert to int number
        J48_Instances = int(re.sub("[\[\]\']", "",str(J48_Instances)))
        
        Nodedata["Model_Instances"] = J48_Instances # Load in Global Dictionary

        LogfileClassificationArray.append([J48_Instances])
        
        global J48_TrainingCorrectInstances
        J48_TrainingCorrectInstances = re.findall(r"Correctly Classified Instances\s+(\d+)",CleanJ48)
        J48_TrainingCorrectInstances = int(re.sub("[\[\]\']", "",str(J48_TrainingCorrectInstances)))
        LogfileClassificationArray.append([J48_TrainingCorrectInstances])
        
        global J48_TrainingAccuracy
        J48_TrainingAccuracy  = int(((float(J48_TrainingCorrectInstances)/float(J48_Instances))*100))
        J48_TrainingAccuracy = str(J48_TrainingAccuracy) + "%"
        
        Nodedata["Model_TrainingAccuracy"] = J48_TrainingAccuracy # Load in Global Dictionary
        
        
         # Capture "=== Stratified cross-validation ===" section
        global J48_Stratified
        J48_Stratified = re.findall(r"(=== Stratified cross-validation.*)",J48,re.DOTALL)
        J48_Stratified = J48_Stratified[0] # Convert from list to string
        
        global J48_Stratified_Clean
        J48_Stratified_Clean = "".join([s for s in J48_Stratified.splitlines(True) if s.strip("\r\n")])
        
        global J48_StratifiedCorrectInstances
        J48_StratifiedCorrectInstances = re.findall(r"Correctly Classified Instances\s+(\d+)",J48_Stratified_Clean)
        J48_StratifiedCorrectInstances  = int(re.sub("[\[\]\']", "",str(J48_StratifiedCorrectInstances)))
        
        global J48_StratifiedAccuracy
        J48_StratifiedAccuracy  = int(((float(J48_StratifiedCorrectInstances)/float(J48_Instances))*100))
        J48_StratifiedAccuracy = str(J48_StratifiedAccuracy) + "%"
        
        Nodedata["Model_StratifiedAccuracy"] = J48_StratifiedAccuracy # Load in Global Dictionary
        
        # =============================================================================
        # Capture number of terminal nodes in tree and add each time a J48 tree is processed 
        # to ensure final rules= number are correct
        # =============================================================================
        
        J48_NumberofRules = re.findall(r"Number of Leaves  :\s+(\d+)",CleanJ48)
        # Clean up value and convert to int number
        J48_NumberofRules = int(re.sub("[\[\]\']", "",str(J48_NumberofRules)))
        
        Nodedata["Model_NumberofRules"] = J48_NumberofRules
        
        global NumberofRulesProcessed
        
        try: NumberofRulesProcessed
        except NameError:
            #Initialise first time the function runs
             NumberofRulesProcessed = 0
             
        #Add Current Tree Number of Rules to global variable
        
        NumberofRulesProcessed =  NumberofRulesProcessed + J48_NumberofRules
        
       
        # =============================================================================
        # Capture confusion matrix and calculate number of records per class
        # =============================================================================
        
        # Capture in an J48_ConfusionMatrix array each line of the confusion matrix ie 11  2  5 |  a = PosValance
        J48_ConfusionMatrix = re.findall(r"\s+(\d+\s+\d+\s*\d*)\s+[|]\s+\w\s[=]\s(\w+)",CleanJ48,re.DOTALL)
        print (J48_ConfusionMatrix)
        LogfileClassificationArray.append([J48_ConfusionMatrix])
    
        for line in range(len(J48_ConfusionMatrix)): 
          line1 =  J48_ConfusionMatrix[line][0]  
          print (line1)
          LogfileClassificationArray.append([line1])
          
          digit_line_split = line1.split()
          print (digit_line_split)
          LogfileClassificationArray.append([digit_line_split])
          
          digit_line_sum = sum([int(i) for i in digit_line_split if type(i)== int or i.isdigit()])
          line1Class = J48_ConfusionMatrix[line][1]  
          print (line1Class + "=" + str(digit_line_sum))
          LogfileClassificationArray.append([line1Class + "=" + str(digit_line_sum)])
          
          
          Nodedata[line1Class] = digit_line_sum
          Nodedata["Probabilityof"+ str(line1Class)] = round(((float(digit_line_sum)/float(J48_Instances))*100),2)
          
        
        #Capture Just TreeNodes
        J48_Treenodes = re.findall(r"[-]+\s(.*)\sNumber of Leaves",CleanJ48,re.DOTALL)
        
        # split into seperate lines
        J48_Treenodes_str = "\n".join(J48_Treenodes)  # convert list into string
        J48_Treenodes_lines = J48_Treenodes_str.splitlines()
        
        #Process each line of the tree
        
        #set the array
        linecount = 0
    
    
       
        # =============================================================================
        # START PROCESSING TREE LINE BY LINE
        # =============================================================================   
        
        Nodedata['Level0Cycle'] = 1
        
        for line in J48_Treenodes_lines:
            print (str(linecount)+"  "+str(line))
            LogfileClassificationArray.append([str(linecount)+"  "+str(line)])
               
                 
            # Load Level0Node
              
            # Check line for Node level that it doesnt contain |
            checkL0 = re.search("^[^|]*$",str(line))
            
            # Load Level0 Node and Level 0 Alt Node
            if checkL0:
                
                NodeLevel = 0
                                            
                if Nodedata['Level'+str(NodeLevel)+'Cycle']  == 1:
                                        
                    Nodedata['Level'+str(NodeLevel)+'Node'] = CaptureNode('Normal',line)
                    
                    Nodedata['Level'+str(NodeLevel)+'Node'] = re.sub("[\[\]\']", "",str(Nodedata['Level'+str(NodeLevel)+'Node']))
                    
                    # load rule part of L0 B1 
                    Nodedata['L'+str(NodeLevel)+'B1Rule'] = Nodedata['Level'+str(NodeLevel)+'Node']
                    
                    Nodedata['BranchL'+str(NodeLevel)] = 1
                     
                        
                    #Check if the node is terminated by (d.d/d.d)
                                      
                    if CaptureNode('CheckTerminal',line):
                        #record rule to rules array
                        #load full rule line with class and support and accuracy
                        
                        Nodedata['Level'+str(NodeLevel)+'Node'] = CaptureNode('Terminal',line)
                        
                                
                        #strip special characters
                        Nodedata['Level'+str(NodeLevel)+'Node'] = re.sub("[\[\]\']", "",str(Nodedata['Level'+str(NodeLevel)+'Node']))
                        
                        # load rule part of L0 B1 
                        Nodedata['L'+str(NodeLevel)+'B1Rule'] = Nodedata['Level'+str(NodeLevel)+'Node']
                        
                        # Record Current Rule Via RecordRule Function
                        Current_Rule = str(Nodedata['L'+str(NodeLevel)+'B1Rule'])
                        
                        RecordRule(Current_Rule)
                        
                        #move to BranchL0 to 2 since branch0 is terminated
                        Nodedata['BranchL'+str(NodeLevel)] = 2
                    
                if Nodedata['Level'+str(NodeLevel)+'Cycle'] == 2:
                    #Level0ALTNode = re.findall(r"(\w+\s+[><=]+\s+[-]?\d+.\d+)",str(line))
                   
                    #Nodedata['Level'+str(NodeLevel)+'ALTNode'] = re.findall(r"(\w+\s+[><=]+\s+[-]?\d+.\d+)",str(line))
                    
                    Nodedata['Level'+str(NodeLevel)+'ALTNode'] = CaptureNode('Normal',line)
                    
                     #Level0Node = re.sub("[\[\]\']", "",str(Level0Node))
                    Nodedata['Level'+str(NodeLevel)+'ALTNode'] = re.sub("[\[\]\']", "",str(Nodedata['Level'+str(NodeLevel)+'ALTNode']))
                    
                    # load rule part of L0 B2 
                    Nodedata['L'+str(NodeLevel)+'B2Rule'] = Nodedata['Level'+str(NodeLevel)+'ALTNode']
                     
                    #BranchL0 = 2
                    Nodedata['BranchL'+str(NodeLevel)] = 2
                    
                    #Check if the node is terminated by (d.d/d.d)
                    Nodedata['CheckNode'+str(NodeLevel)+'Termin'] = re.search(r"(\(.*\))",str(line))
                    
                    #if Nodedata['CheckNode'+str(NodeLevel)+'Termin']:
                    if CaptureNode('CheckTerminal',line):
                        #record rule to rules array
                                                
                        Nodedata['Level'+str(NodeLevel)+'ALTNode'] = CaptureNode('Terminal',line)
                        
                        # strip special characters
                        
                        Nodedata['Level'+str(NodeLevel)+'ALTNode'] = re.sub("[\[\]\']", "",str(Nodedata['Level'+str(NodeLevel)+'ALTNode']))
                        
                         # load rule part of L0 B2 
                        Nodedata['L'+str(NodeLevel)+'B2Rule'] = Nodedata['Level'+str(NodeLevel)+'ALTNode']
                        
                        # Record Current Rule Via RecordRule Function
                        
                        Current_Rule = str(Nodedata['L'+str(NodeLevel)+'B2Rule'])
                        
                        RecordRule(Current_Rule)
                        
                        #move to BranchL0 to 1 since branch0 is terminated
                     
                        Nodedata['BranchL'+str(NodeLevel)] = 1
                        
                        
                #if Level0Node != "":
                if Nodedata['Level'+str(NodeLevel)+'Node'] != "":
                    
                    # After L0 Branch 1 is complete move on to branch 2
                    #Level0Cycle = 2
                    Nodedata['Level'+str(NodeLevel)+'Cycle'] = 2
              
                
        # =============================================================================
        #     
        #     ## PROCESS LEVEL 1,2,3.. ONWARDS ##        
        # 
        # =============================================================================
           
        
            #Check line for Node level that it contains 1 x |
        
            checkLevel = re.findall(r"([|])",str(line))
            
            #countL1 = checkL1.count('|')
            
            NodeLevel = checkLevel.count('|')
            ParentNodeLevel = NodeLevel - 1
            
            
            # Load Level1 Node and Level1 Alt Node
            if checkLevel:
                                   
                # Load first level node for first cycle is done  
                
                cycle_key = 'Level'+str(NodeLevel)+'Cycle'
                if cycle_key in Nodedata:
                    #print("Moved to Cycle 2" + str(cycle_key))
                    #set variable for each cycle completed
                    Nodedata['Level'+str(NodeLevel)+'Cycle'] = 2
                else: 
                    #print("Declared Cycle 1" + str(cycle_key))
                    #set variable for each cycle completed
                    Nodedata['Level'+str(NodeLevel)+'Cycle'] = 1
                
                
                #if Level1Cycle == 1:
                if Nodedata['Level'+str(NodeLevel)+'Cycle'] == 1:          
                
                    # check which L0 branch you are on
                                        
                    Nodedata['Level'+str(NodeLevel)+'Node'] = CaptureNode('Normal',line)
                    
                    #BranchL1 = 1
                    Nodedata['BranchL'+str(NodeLevel)] = 1
                    
                    # Set variable for each cycle completed
                    #Level1Cycle = 1
                    Nodedata['Level'+str(NodeLevel)+'Cycle'] = 1
                    
                    
                    # Deal with the level 0 branch 1 first node level 1 branch 1
                    #if BranchL0 == 1:
                    
                    if Nodedata['BranchL'+str(ParentNodeLevel)] == 1:  
                        
                            # check Node0 is not terminated and make rule
                            
                            Nodedata['L'+str(NodeLevel)+'B1Rule'] = str( Nodedata['L'+str(ParentNodeLevel)+'B1Rule']) + " AND " + str(Nodedata['Level'+str(NodeLevel)+'Node'])
                            
                                       
                            # strip special characters
                            Nodedata['L'+str(NodeLevel)+'B1Rule'] = re.sub("[\[\]\']", "",str(Nodedata['L'+str(NodeLevel)+'B1Rule']))
                            
                            # check if the node is terminated by (d.d/d.d)
                            
                            Nodedata['CheckNode'+str(NodeLevel)+'Termin'] = re.search(r"(\(.*\))",str(line))
                        
                            if CaptureNode('CheckTerminal',line):
                        
                                # load full rule line with class and support and accuracy
                                                                
                                Nodedata['Level'+str(NodeLevel)+'Node'] = CaptureNode('Terminal',line)
                        
                                # check Node0 is not terminated and make rule
                                
                                Nodedata['L'+str(NodeLevel)+'B1Rule'] = str(Nodedata['L'+str(ParentNodeLevel)+'B1Rule']) + " AND " + str(Nodedata['Level'+str(NodeLevel)+'Node'])
                        
                                # strip special characters
                              
                                Nodedata['L'+str(NodeLevel)+'B1Rule'] = re.sub("[\[\]\']", "",str(Nodedata['L'+str(NodeLevel)+'B1Rule']))
                        
                                                            
                                # Record Current Rule Via RecordRule Function
                        
                                Current_Rule = str(Nodedata['L'+str(NodeLevel)+'B1Rule'])
                        
                                RecordRule(Current_Rule)
                                
                                #move to BranchL1 to 2
                             
                                Nodedata['BranchL'+str(NodeLevel)] = 2
                        
                        
                                                    
                            # Deal with the level 0 branch 2 first node level 1 branch 1 if BranchL0 = 2              
                            
                    if Nodedata['BranchL'+str(ParentNodeLevel)] == 2: 
                            
                            Nodedata['L'+str(NodeLevel)+'B1Rule'] = str( Nodedata['L'+str(ParentNodeLevel)+'B2Rule']) + " AND " + str(Nodedata['Level'+str(NodeLevel)+'Node'])
                        
                        
                            #BranchL1 = 1
                            Nodedata['BranchL'+str(NodeLevel)] = 1
                        
                            # strip special characters
                           
                            Nodedata['L'+str(NodeLevel)+'B1Rule'] = re.sub("[\[\]\']", "",str(Nodedata['L'+str(NodeLevel)+'B1Rule']))
                        
                            # check if the node is terminated by (d.d/d.d)
                           
                            Nodedata['CheckNode'+str(NodeLevel)+'Termin'] = re.search(r"(\(.*\))",str(line))
                        
                            #if Nodedata['CheckNode'+str(NodeLevel)+'Termin']:
                            if CaptureNode('CheckTerminal',line):
                            
                                # load full rule line with class and support and accuracy
                                                                
                                Nodedata['Level'+str(NodeLevel)+'Node'] = CaptureNode('Terminal',line)
                                
                                # check Node0 is not terminated and make rule
                                
                                Nodedata['L'+str(NodeLevel)+'B1Rule'] = str(Nodedata['L'+str(ParentNodeLevel)+'B2Rule']) + " AND " + str(Nodedata['Level'+str(NodeLevel)+'Node'])
                                
                                # strip special characters
                                
                                Nodedata['L'+str(NodeLevel)+'B1Rule'] = re.sub("[\[\]\']", "",str(Nodedata['L'+str(NodeLevel)+'B1Rule']))
                                
                                                           
                                # Record Current Rule Via RecordRule Function
                        
                                Current_Rule = str(Nodedata['L'+str(NodeLevel)+'B1Rule'])
                        
                                RecordRule(Current_Rule)
                                
                               
                                #BranchL1 = 2
                                Nodedata['BranchL'+str(NodeLevel)] = 2
                             
                                #print (L1B1Rule)
                        
                            
                        # Load ALT level node if first cycle is done  
                
                #if Level1Cycle == 2:
                if Nodedata['Level'+str(NodeLevel)+'Cycle'] == 2:   
                    
                    # Deal with the first node level 1 branch 2
                    
                   
                    Nodedata['Level'+str(NodeLevel)+'ALTNode'] = CaptureNode('Normal',line) 
                   
                    #BranchL1 = 2
                    Nodedata['BranchL'+str(NodeLevel)] = 2
                    
                    # check which L0 branch you are on, if branch = 1
                    #if BranchL0 == 1:
                    if Nodedata['BranchL'+str(ParentNodeLevel)] == 1:     
                        
                        
                        Nodedata['L'+str(NodeLevel)+'B2Rule'] = str( Nodedata['L'+str(ParentNodeLevel)+'B1Rule']) + " AND " + str(Nodedata['Level'+str(NodeLevel)+'ALTNode'])
                        
                        # strip special characters
                        
                        Nodedata['L'+str(NodeLevel)+'B2Rule'] = re.sub("[\[\]\']", "",str(Nodedata['L'+str(NodeLevel)+'B2Rule']))  
                        
                        # check if the node is terminated by (d.d/d.d)
                  
                        Nodedata['CheckNode'+str(NodeLevel)+'Termin'] = re.search(r"(\(.*\))",str(line))
                        
                        #if Nodedata['CheckNode'+str(NodeLevel)+'Termin']:
                        if CaptureNode('CheckTerminal',line):
                            
                                             
                            # load full rule line with class and support and accuracy
                                                        
                            Nodedata['Level'+str(NodeLevel)+'ALTNode'] = CaptureNode('Terminal',line)
                            
                            # check Node0 is not terminated and make rule
                                                        
                            Nodedata['L'+str(NodeLevel)+'B2Rule'] = str(Nodedata['L'+str(ParentNodeLevel)+'B1Rule']) + " AND " + str(Nodedata['Level'+str(NodeLevel)+'ALTNode'])
                            
                            # strip special characters
                            
                            Nodedata['L'+str(NodeLevel)+'B2Rule'] = re.sub("[\[\]\']", "",str(Nodedata['L'+str(NodeLevel)+'B2Rule']))  
                            
                                                   
                            # Record Current Rule Via RecordRule Function
                        
                            Current_Rule = str(Nodedata['L'+str(NodeLevel)+'B2Rule'])
                        
                            RecordRule(Current_Rule)
                            
                                               
                            #move to BranchL1 to 1
                           
                            Nodedata['BranchL'+str(NodeLevel)] = 1
                            
                            
                        
                                   
                    # check which L0 branch you are on, if branch = 2
                    
                    if Nodedata['BranchL'+str(ParentNodeLevel)] == 2:  
                        
                        #BranchL1 = 2
                        Nodedata['BranchL'+str(NodeLevel)] = 2           
                        
                        Nodedata['L'+str(NodeLevel)+'B2Rule'] = str( Nodedata['L'+str(ParentNodeLevel)+'B2Rule']) + " AND " + str(Nodedata['Level'+str(NodeLevel)+'ALTNode'])
                        
                        # strip special characters
                        
                        Nodedata['L'+str(NodeLevel)+'B2Rule'] = re.sub("[\[\]\']", "",str(Nodedata['L'+str(NodeLevel)+'B2Rule']))  
                        
                        # check if the node is terminated by (d.d/d.d)
                       
                        Nodedata['CheckNode'+str(NodeLevel)+'Termin'] = re.search(r"(\(.*\))",str(line))
                        
                        #if Nodedata['CheckNode'+str(NodeLevel)+'Termin']:
                        if CaptureNode('CheckTerminal',line):
                            
                            # load full rule line with class and support and accuracy
                                                       
                            Nodedata['Level'+str(NodeLevel)+'ALTNode'] = CaptureNode('Terminal',line) 
                           
                            # check Node0 is not terminated and make rule
                                                        
                            Nodedata['L'+str(NodeLevel)+'B2Rule'] = str(Nodedata['L'+str(ParentNodeLevel)+'B2Rule']) + " AND " + str(Nodedata['Level'+str(NodeLevel)+'ALTNode'])
                            
                            # strip special characters
                            
                            Nodedata['L'+str(NodeLevel)+'B2Rule'] = re.sub("[\[\]\']", "",str(Nodedata['L'+str(NodeLevel)+'B2Rule']))  
                            
                                                 
                            # Record Current Rule Via RecordRule Function
                        
                            Current_Rule = str(Nodedata['L'+str(NodeLevel)+'B2Rule'])
                    
                            RecordRule(Current_Rule)
                            
                                                   
                            #move to BranchL1 to 1
                           
                            Nodedata['BranchL'+str(NodeLevel)] = 1
                            
                            #print (Nodedata['L'+str(NodeLevel)+'B2Rule'])
                
            #increment array/linecount 
            linecount += 1
            
       
       
        # =============================================================================
        # Export Main Dictionary into Nodedata_Contents Array for easy reading
        # =============================================================================
        
        i = 0
       
        # Make Nodedata contents global for easy debugging
        global Nodedata_Contents
        Nodedata_Contents = []
        
        for key,value in Nodedata.items():
            
            print ('NodeData Internal Variables Generated By J48 Algorithm')
            LogfileClassificationArray.append(['NodeData Internal Variables Generated By J48 Algorithm'])
            print (str(key) + " => " + str(value))
            LogfileClassificationArray.append([str(key) + " => " + str(value)])
    
            Nodedata_Contents.insert(i,str(key) + " => " + str(value)) 
            i= i + 1 

   

    # =============================================================================
    # END OF J48 FUNCTION
    # =============================================================================
    


# =============================================================================
# SYSFOR KD Function
# =============================================================================  

# Rule Extraction from Tree
def SySFor_genKDfunction(SySFor,KDParticipant,wekaexperimentname):
    
    # Declare Global Nodedata dictionary for storing temporary branch and tree data
    
    global Nodedata
    Nodedata = {}
    
    # Declare GLOBAL LogfileClassificationArray for recording events
    
    import re  #import regular expression library
    import datetime
    
    import numpy as np
        
    global LogfileClassificationArray
    
          
    #Store in Nodedata Key Function Values
    Nodedata["KDParticipant"] = KDParticipant
    Nodedata["wekaexperimentname"] = wekaexperimentname
    
    
    try: LogfileClassificationArray
    except NameError:
        #Initialise LogFile First it runs
        LogfileClassificationArray = []

    now = datetime.datetime.now()
    print ("Current date and time : " + now.strftime("%Y-%m-%d %H:%M:%S"))
    LogfileClassificationArray.append(["Current date and time : " + now.strftime("%Y-%m-%d %H:%M:%S")])
    
        
    # =============================================================================
    # Capture Node Rule Info using custom filter
    # =============================================================================   
    
        
    def CaptureNode(NodeType,Line):
        
        global Nodeline
        
        if NodeType == "Normal":
            
            # Capture a normal node like Ch37_PGSD <= 0.10518
            #Nodeline = re.findall(r"(\w+\s+[><=]+\s+[-]*\d+[.]*\d*)",str(line))
            
            # This regex patern targets the attribute name and the operants such as =>< and copies all the criterion be it numeric or text
            Nodeline = re.findall(r"(\w+\s+[><=]+.*)",str(line))
            
            return Nodeline
            
            
        if NodeType == "CheckTerminal":
            print ('YES CHECK TERMINAL')
            # check if the node is terminated by (d.d/d.d)
            Nodeline = re.search(r"(\(.*\))",str(line))
            # Return boolean
            return Nodeline
        
        if NodeType == "Terminal":
            
            # This regex patern targets the attribute name and the operants such as =>< and copies all the criterion be it numeric or text
            #Nodeline = re.findall(r"(\w+\s+[><=]+.*)",str(line))
            
            #Typical Sysfor line HR_Max_GlobalRef = Above: PosValence {PosValence,5;NegValence,1;} (6/1)
            #Extract Field Criterion and Class Label, get rid of breakdown of misclassifications {} 
            NodePart1 = re.findall(r"(\w+\s+[><=]+\s.*[:]\s\w+)",str(line))
            
            #Extract Classified and misclasified records (d/d)
            NodePart2 = re.findall(r"(\(.*\))",str(line))
            
            # Join Parts into Nodeline
            Nodeline = str(NodePart1) + " " + str(NodePart2)
            
            return Nodeline
       

        # =============================================================================
        # END OF CAPTURE NODE FILTER FUNCTION
        # =============================================================================
    
    
    # Read Tree Attributes
    # Cut Cross Validation Results, leave only training data in SySFor
         
   
    
    global SySFor_Training
    # Capture from the begining of the tree output "Options.." to just before the "=== Stratified cross-validation ==="
    SySFor_Training = re.findall(r"\s(Options.*)\s=== Stratified cross-validation ===",SySFor,re.DOTALL)
    SySFor_Training = SySFor_Training[0] # Convert from list to string
    
          
    global CleanSySFor
    CleanSySFor = "".join([s for s in SySFor_Training.splitlines(True) if s.strip("\r\n")])
    
    ### Extract Important Fields from Training Tree ###
    global SySFor_Scheme
    SySFor_Scheme = re.findall(r"Options:\s+(.*)\s",CleanSySFor)
    
    Nodedata["Model_Scheme"] = SySFor_Scheme # Load in Global Dictionary
    LogfileClassificationArray.append([SySFor_Scheme])
    
    global SySFor_Instances
    SySFor_Instances = re.findall(r"Total Number of Instances\s+(\d+)",CleanSySFor)
    # Clean up value and convert to int number
    SySFor_Instances = int(re.sub("[\[\]\']", "",str(SySFor_Instances)))
    
    Nodedata["Model_Instances"] = SySFor_Instances # Load in Global Dictionary
    LogfileClassificationArray.append([SySFor_Instances])
    
    global SySFor_TrainingCorrectInstances
    SySFor_TrainingCorrectInstances = re.findall(r"Correctly Classified Instances\s+(\d+)",CleanSySFor)
    SySFor_TrainingCorrectInstances = int(re.sub("[\[\]\']", "",str(SySFor_TrainingCorrectInstances)))
    
    LogfileClassificationArray.append([SySFor_TrainingCorrectInstances])
    
    global SySFor_TrainingAccuracy
    SySFor_TrainingAccuracy  = int(((float(SySFor_TrainingCorrectInstances)/float(SySFor_Instances))*100))
    SySFor_TrainingAccuracy = str(SySFor_TrainingAccuracy) + "%"
    
    Nodedata["Model_TrainingAccuracy"] = SySFor_TrainingAccuracy # Load in Global Dictionary
    
    # =============================================================================
    # Capture confusion matrix and calculate number of records per class
    # =============================================================================
            
    # Capture in an SySFor_ConfusionMatrix array each line of the confusion matrix ie 11  2  5 |  a = PosValance
    SySFor_ConfusionMatrix = re.findall(r"\s+(\d+\s+\d+\s*\d*)\s+[|]\s+\w\s[=]\s(\w+)",CleanSySFor,re.DOTALL)
    print (SySFor_ConfusionMatrix)
    LogfileClassificationArray.append([SySFor_ConfusionMatrix])

    for line in range(len(SySFor_ConfusionMatrix)): 
      line1 =  SySFor_ConfusionMatrix[line][0]  
      print (line1)
      LogfileClassificationArray.append([line1])
      
      digit_line_split = line1.split()
      print (digit_line_split)
      LogfileClassificationArray.append([digit_line_split])
      
      digit_line_sum = sum([int(i) for i in digit_line_split if type(i)== int or i.isdigit()])
      line1Class = SySFor_ConfusionMatrix[line][1]  
      print (line1Class + "=" + str(digit_line_sum))
      LogfileClassificationArray.append([line1Class + "=" + str(digit_line_sum)])
      
      
      Nodedata[line1Class] = digit_line_sum
      Nodedata["Probabilityof"+ str(line1Class)] = round(((float(digit_line_sum)/float(SySFor_Instances))*100),2)

    
     # Capture "=== Stratified cross-validation ===" section
    global SySFor_Stratified
    SySFor_Stratified = re.findall(r"(=== Stratified cross-validation.*)",SySFor,re.DOTALL)
    SySFor_Stratified = SySFor_Stratified[0] # Convert from list to string
    
    global SySFor_Stratified_Clean
    SySFor_Stratified_Clean = "".join([s for s in SySFor_Stratified.splitlines(True) if s.strip("\r\n")])
    
    global SySFor_StratifiedCorrectInstances
    SySFor_StratifiedCorrectInstances = re.findall(r"Correctly Classified Instances\s+(\d+)",SySFor_Stratified_Clean)
    SySFor_StratifiedCorrectInstances  = int(re.sub("[\[\]\']", "",str(SySFor_StratifiedCorrectInstances)))
    
    global SySFor_StratifiedAccuracy
    SySFor_StratifiedAccuracy  = int(((float(SySFor_StratifiedCorrectInstances)/float(SySFor_Instances))*100))
    SySFor_StratifiedAccuracy = str(SySFor_StratifiedAccuracy) + "%"
    
    Nodedata["Model_StratifiedAccuracy"] = SySFor_StratifiedAccuracy # Load in Global Dictionary
    
    
    
    
    #Capture just the Sysfor Forest in List from options to time taken to build
    global JustForest
    #JustForest = re.findall(r"\n{3}(RandomTree.*Size\sof\sthe\stree\s:\s\d+\n{4})",SySFor_Training,re.DOTALL)
    
    JustForest = re.findall(r"Options:\s+[-]\w\s\d+\s[-]\w\s\d+(.*)Time\staken\sto\sbuild",SySFor_Training,re.DOTALL)
    
    # Capture Each Tree as a list, creates a tuple for each tree, row string [1] contains each tree
    global Trees
    Trees = re.findall(r"(Tree\s+\d+[:]\s\n((.+\n)*))",JustForest[0])
    
    # Capture the Tree Size Info as list
    global TreeSizes
    #TreeSizes = re.findall(r"Size\sof\sthe\stree\s:\s(\d+)\n",JustForest[0])
    
    # Number of Trees
    global NumberTrees
    NumberTrees = len(Trees)
    
    # Setup a Numpy Array to load all Info, increase length by one to take into account array headings
    global ForestArray
    ForestArray = np.zeros((NumberTrees+1,3)).astype(object)
    ForestArray[0][0] = 'TreeNum'
    ForestArray[0][1] = 'TreeNodes'
    ForestArray[0][2] = 'TreeSize'
    
    # Loop through all the trees, start at position 1 to write after te headings
    for treenum in range(1,NumberTrees +1):
        ForestArray[treenum][0] = treenum
        
        # Adjust by -1 row since Trees have no headings, access the tree string in row 1 of each tuple
        ForestArray[treenum][1] = Trees[treenum-1][1] 
        
        # Capture the number of nodes in the tree
        TreeSize = len(re.findall(r"[=<>]",Trees[treenum-1][1]))
        ForestArray[treenum][2] = TreeSize 
        
    
    # CHECK IF THE TREE IS NOT EMPTY ie Number of Leaf Nodes: 1
    # IF IT IS WIPE IT SO THE ALGORTHM WILL SKIP
    SySFor_check = re.search(r"Number of Leaf Nodes: 1",SySFor)
    
        
    # =============================================================================
    # FOREST LOOP - Provide each individual tree to the KD rule learner
    # =============================================================================
    
    for ForestLoop in range(1,NumberTrees +1):
    
        CurrentTreeSize = int(ForestArray[ForestLoop][2])
    
        # NEED TO REVISIT
        if CurrentTreeSize == 1 or NumberTrees == 0: 
            print("Tree number " + str(ForestLoop) + " has not produced any rules - Tree has only 1 node")
            print(SySFor)
            LogfileClassificationArray.append(["Tree number " + str(ForestLoop) + " has not produced any rules - Tree has only 1 node"])
            LogfileClassificationArray.append([SySFor])
      
        # =============================================================================
        # PROCEED TO PROCESS THE TREE!
        # =============================================================================
      
        else:
            
            # Capture tree from Forest
            CurrentTree = ForestArray[ForestLoop][1]
            
            #Cleanup any empty line
            CurrentTree = "".join([s for s in CurrentTree.splitlines(True) if s.strip("\r\n")])
            
            # =============================================================================
            # Capture number of terminal nodes in tree and add each time a SySFor tree is processed 
            # to ensure final rules= number are correct
            # =============================================================================
            
            # Capture number of terminal nodes cound the number of terminal nodes in current tree (dd/dd)
            
            SySFor_NumberofRules = len(re.findall(r"(\(.*\))",CurrentTree))
            # Clean up value and convert to int number
            SySFor_NumberofRules = int(re.sub("[\[\]\']", "",str(SySFor_NumberofRules)))
            
            Nodedata["Model_NumberofRules"] = SySFor_NumberofRules
            
            global NumberofRulesProcessed
            
            try: NumberofRulesProcessed
            except NameError:
                #Initialise first time the function runs
                 NumberofRulesProcessed = 0
                 
            #Add Current Tree Number of Rules to global variable
            
            NumberofRulesProcessed =  NumberofRulesProcessed + SySFor_NumberofRules
           
            
            # Capture between SySFor Decision Tree ....Number of Leaf Nodes: 1
            
            
            # split TREE into seperate lines
            
            SySFor_Treenodes_lines = CurrentTree.splitlines()
            
            #Process each line of the tree
            
            #set the array
            linecount = 0
        
        
           
            # =============================================================================
            # START PROCESSING TREE LINE BY LINE
            # =============================================================================   
            
            Nodedata['Level0Cycle'] = 1
            
            for line in SySFor_Treenodes_lines:
                print (str(linecount)+"  "+str(line))
                LogfileClassificationArray.append([str(linecount)+"  "+str(line)])
                   
                     
                # Load Level0Node
                  
                # Check line for Node level that it doesnt contain |
                checkL0 = re.search("^[^|]*$",str(line))
                
                # Load Level0 Node and Level 0 Alt Node
                if checkL0:
                    
                    NodeLevel = 0
                    
                                          
                    if Nodedata['Level'+str(NodeLevel)+'Cycle']  == 1:
                        
                        
                                                
                        Nodedata['Level'+str(NodeLevel)+'Node'] = CaptureNode('Normal',line)
                        
                        
                        Nodedata['Level'+str(NodeLevel)+'Node'] = re.sub("[\[\]\']", "",str(Nodedata['Level'+str(NodeLevel)+'Node']))
                        
                        # load rule part of L0 B1 
                       
                        Nodedata['L'+str(NodeLevel)+'B1Rule'] = Nodedata['Level'+str(NodeLevel)+'Node']
                        
                        Nodedata['BranchL'+str(NodeLevel)] = 1
                         
                            
                        # check if the node is terminated by (d.d/d.d)
                                                
                        if CaptureNode('CheckTerminal',line):
                            #record rule to rules array
                            # load full rule line with class and support and accuracy
                            
                            
                            Nodedata['Level'+str(NodeLevel)+'Node'] = CaptureNode('Terminal',line)
                            
                                    
                            # strip special characters
                            
                            Nodedata['Level'+str(NodeLevel)+'Node'] = re.sub("[\[\]\']", "",str(Nodedata['Level'+str(NodeLevel)+'Node']))
                            
                            # load rule part of L0 B1 
                            Nodedata['L'+str(NodeLevel)+'B1Rule'] = Nodedata['Level'+str(NodeLevel)+'Node']
                            
                            # Record Current Rule Via RecordRule Function
                            
                            Current_Rule = str(Nodedata['L'+str(NodeLevel)+'B1Rule'])
                            
                            RecordRule(Current_Rule)
                        
                            #move to BranchL0 to 2 since branch0 is terminated
                            
                            Nodedata['BranchL'+str(NodeLevel)] = 2
                        
                    if Nodedata['Level'+str(NodeLevel)+'Cycle'] == 2:
                                                
                        Nodedata['Level'+str(NodeLevel)+'ALTNode'] = CaptureNode('Normal',line)
                        
                        Nodedata['Level'+str(NodeLevel)+'ALTNode'] = re.sub("[\[\]\']", "",str(Nodedata['Level'+str(NodeLevel)+'ALTNode']))
                        
                        # load rule part of L0 B2 
                        
                        Nodedata['L'+str(NodeLevel)+'B2Rule'] = Nodedata['Level'+str(NodeLevel)+'ALTNode']
                         
                        #BranchL0 = 2
                        Nodedata['BranchL'+str(NodeLevel)] = 2
                        
                        # check if the node is terminated by (d.d/d.d)
                        
                        Nodedata['CheckNode'+str(NodeLevel)+'Termin'] = re.search(r"(\(.*\))",str(line))
                        
                        #if Nodedata['CheckNode'+str(NodeLevel)+'Termin']:
                        if CaptureNode('CheckTerminal',line):
                            #record rule to rules array
                            # load full rule line with class and support and accuracy
                                                       
                            Nodedata['Level'+str(NodeLevel)+'ALTNode'] = CaptureNode('Terminal',line)
                            
                            # strip special characters
                            
                            Nodedata['Level'+str(NodeLevel)+'ALTNode'] = re.sub("[\[\]\']", "",str(Nodedata['Level'+str(NodeLevel)+'ALTNode']))
                            
                             # load rule part of L0 B2 
                            Nodedata['L'+str(NodeLevel)+'B2Rule'] = Nodedata['Level'+str(NodeLevel)+'ALTNode']
                            
                            # Record Current Rule Via RecordRule Function
                            
                            Current_Rule = str(Nodedata['L'+str(NodeLevel)+'B2Rule'])
                            
                            RecordRule(Current_Rule)
                            
                            #move to BranchL0 to 1 since branch0 is terminated
                         
                            Nodedata['BranchL'+str(NodeLevel)] = 1
                            
                            
                    #if Level0Node != "":
                    if Nodedata['Level'+str(NodeLevel)+'Node'] != "":
                        
                        # After L0 Branch 1 is complete move on to branch 2
                        #Level0Cycle = 2
                        Nodedata['Level'+str(NodeLevel)+'Cycle'] = 2
                  
                    
            # =============================================================================
            #     
            #     ## PROCESS LEVEL 1,2,3.. ONWARDS ##        
            # 
            # =============================================================================
               
            
                #Check line for Node level that it contains 1 x |
            
                checkLevel = re.findall(r"([|])",str(line))
                
                #countL1 = checkL1.count('|')
                
                NodeLevel = checkLevel.count('|')
                ParentNodeLevel = NodeLevel - 1
                
                
                # Load Level1 Node and Level1 Alt Node
                if checkLevel:
                                           
                    # Load first level node for first cycle is done  
                    
                    cycle_key = 'Level'+str(NodeLevel)+'Cycle'
                    if cycle_key in Nodedata:
                        
                        # Set variable for each cycle completed
                        Nodedata['Level'+str(NodeLevel)+'Cycle'] = 2
                    else: 
                       
                        # Set variable for each cycle completed
                        Nodedata['Level'+str(NodeLevel)+'Cycle'] = 1
                    
                    
                    #if Level1Cycle == 1:
                    if Nodedata['Level'+str(NodeLevel)+'Cycle'] == 1:          
                    
                        
                        # check which L0 branch you are on
                                                
                        Nodedata['Level'+str(NodeLevel)+'Node'] = CaptureNode('Normal',line)
                        
                        #BranchL1 = 1
                        Nodedata['BranchL'+str(NodeLevel)] = 1
                        
                        # Set variable for each cycle completed
                        #Level1Cycle = 1
                        Nodedata['Level'+str(NodeLevel)+'Cycle'] = 1
                        
                        
                        # Deal with the level 0 branch 1 first node level 1 branch 1
                        #if BranchL0 == 1:
                        
                        if Nodedata['BranchL'+str(ParentNodeLevel)] == 1:  
                            
                                # check Node0 is not terminated and make rule
                                
                                Nodedata['L'+str(NodeLevel)+'B1Rule'] = str( Nodedata['L'+str(ParentNodeLevel)+'B1Rule']) + " AND " + str(Nodedata['Level'+str(NodeLevel)+'Node'])
                                
                                           
                                # strip special characters
                                
                                Nodedata['L'+str(NodeLevel)+'B1Rule'] = re.sub("[\[\]\']", "",str(Nodedata['L'+str(NodeLevel)+'B1Rule']))
                                
                                # check if the node is terminated by (d.d/d.d)
                                
                                Nodedata['CheckNode'+str(NodeLevel)+'Termin'] = re.search(r"(\(.*\))",str(line))
                            
                                 
                                if CaptureNode('CheckTerminal',line):
                            
                                    # load full rule line with class and support and accuracy
                                                                        
                                    Nodedata['Level'+str(NodeLevel)+'Node'] = CaptureNode('Terminal',line)
                            
                                    # check Node0 is not terminated and make rule
                                    
                                    Nodedata['L'+str(NodeLevel)+'B1Rule'] = str(Nodedata['L'+str(ParentNodeLevel)+'B1Rule']) + " AND " + str(Nodedata['Level'+str(NodeLevel)+'Node'])
                            
                                    # strip special characters
                                   
                                    Nodedata['L'+str(NodeLevel)+'B1Rule'] = re.sub("[\[\]\']", "",str(Nodedata['L'+str(NodeLevel)+'B1Rule']))
                            
                                                                
                                    # Record Current Rule Via RecordRule Function
                            
                                    Current_Rule = str(Nodedata['L'+str(NodeLevel)+'B1Rule'])
                            
                                    RecordRule(Current_Rule)
                                    
                                    #move to BranchL1 to 2
                                 
                                    Nodedata['BranchL'+str(NodeLevel)] = 2
                            
                            
                                    #print (L1B1Rule)
                        
                                # Deal with the level 0 branch 2 first node level 1 branch 1 if BranchL0 = 2              
                               
                        if Nodedata['BranchL'+str(ParentNodeLevel)] == 2: 
                                
                                Nodedata['L'+str(NodeLevel)+'B1Rule'] = str( Nodedata['L'+str(ParentNodeLevel)+'B2Rule']) + " AND " + str(Nodedata['Level'+str(NodeLevel)+'Node'])
                            
                            
                                #BranchL1 = 1
                                Nodedata['BranchL'+str(NodeLevel)] = 1
                            
                                # strip special characters
                               
                                Nodedata['L'+str(NodeLevel)+'B1Rule'] = re.sub("[\[\]\']", "",str(Nodedata['L'+str(NodeLevel)+'B1Rule']))
                            
                                # check if the node is terminated by (d.d/d.d)
                               
                                Nodedata['CheckNode'+str(NodeLevel)+'Termin'] = re.search(r"(\(.*\))",str(line))
                            
                                #if Nodedata['CheckNode'+str(NodeLevel)+'Termin']:
                                if CaptureNode('CheckTerminal',line):
                                
                                    # load full rule line with class and support and accuracy
                                                                        
                                    Nodedata['Level'+str(NodeLevel)+'Node'] = CaptureNode('Terminal',line)
                                    
                                    # check Node0 is not terminated and make rule
                                    
                                    Nodedata['L'+str(NodeLevel)+'B1Rule'] = str(Nodedata['L'+str(ParentNodeLevel)+'B2Rule']) + " AND " + str(Nodedata['Level'+str(NodeLevel)+'Node'])
                                    
                                    # strip special characters
                                    
                                    Nodedata['L'+str(NodeLevel)+'B1Rule'] = re.sub("[\[\]\']", "",str(Nodedata['L'+str(NodeLevel)+'B1Rule']))
                                    
                                                               
                                    # Record Current Rule Via RecordRule Function
                            
                                    Current_Rule = str(Nodedata['L'+str(NodeLevel)+'B1Rule'])
                            
                                    RecordRule(Current_Rule)
                                    
                                   
                                    #BranchL1 = 2
                                    Nodedata['BranchL'+str(NodeLevel)] = 2
                                 
                                    #print (L1B1Rule)
                            
                                
                            # Load ALT level node if first cycle is done  
                    
                    #if Level1Cycle == 2:
                    if Nodedata['Level'+str(NodeLevel)+'Cycle'] == 2:   
                        
                        # Deal with the first node level 1 branch 2
                                               
                        Nodedata['Level'+str(NodeLevel)+'ALTNode'] = CaptureNode('Normal',line) 
                       
                        #BranchL1 = 2
                        Nodedata['BranchL'+str(NodeLevel)] = 2
                        
                        # check which L0 branch you are on, if branch = 1
                        #if BranchL0 == 1:
                        if Nodedata['BranchL'+str(ParentNodeLevel)] == 1:     
                            
                            Nodedata['L'+str(NodeLevel)+'B2Rule'] = str( Nodedata['L'+str(ParentNodeLevel)+'B1Rule']) + " AND " + str(Nodedata['Level'+str(NodeLevel)+'ALTNode'])
                            
                            # strip special characters
                            
                            Nodedata['L'+str(NodeLevel)+'B2Rule'] = re.sub("[\[\]\']", "",str(Nodedata['L'+str(NodeLevel)+'B2Rule']))  
                            
                            # check if the node is terminated by (d.d/d.d)
                           
                            Nodedata['CheckNode'+str(NodeLevel)+'Termin'] = re.search(r"(\(.*\))",str(line))
                            
                            #if Nodedata['CheckNode'+str(NodeLevel)+'Termin']:
                            if CaptureNode('CheckTerminal',line):
                                
                                                 
                                # load full rule line with class and support and accuracy
                                                                
                                Nodedata['Level'+str(NodeLevel)+'ALTNode'] = CaptureNode('Terminal',line)
                                
                                # check Node0 is not terminated and make rule
                                                                
                                Nodedata['L'+str(NodeLevel)+'B2Rule'] = str(Nodedata['L'+str(ParentNodeLevel)+'B1Rule']) + " AND " + str(Nodedata['Level'+str(NodeLevel)+'ALTNode'])
                                
                                # strip special characters
                               
                                Nodedata['L'+str(NodeLevel)+'B2Rule'] = re.sub("[\[\]\']", "",str(Nodedata['L'+str(NodeLevel)+'B2Rule']))  
                                
                                                       
                                # Record Current Rule Via RecordRule Function
                            
                                Current_Rule = str(Nodedata['L'+str(NodeLevel)+'B2Rule'])
                            
                                RecordRule(Current_Rule)
                                
                                                   
                                #move to BranchL1 to 1
                               
                                Nodedata['BranchL'+str(NodeLevel)] = 1
                                
                                
                            
                                       
                        # check which L0 branch you are on, if branch = 2
                        
                        if Nodedata['BranchL'+str(ParentNodeLevel)] == 2:  
                            
                            #BranchL1 = 2
                            Nodedata['BranchL'+str(NodeLevel)] = 2           
                            
                            #L1B2Rule = str(Level0ALTNode) + " AND " + str(Level1ALTNode)
                            
                            Nodedata['L'+str(NodeLevel)+'B2Rule'] = str( Nodedata['L'+str(ParentNodeLevel)+'B2Rule']) + " AND " + str(Nodedata['Level'+str(NodeLevel)+'ALTNode'])
                            
                            # strip special characters
                           
                            Nodedata['L'+str(NodeLevel)+'B2Rule'] = re.sub("[\[\]\']", "",str(Nodedata['L'+str(NodeLevel)+'B2Rule']))  
                            
                            # check if the node is terminated by (d.d/d.d)
                            
                            Nodedata['CheckNode'+str(NodeLevel)+'Termin'] = re.search(r"(\(.*\))",str(line))
                            
                            
                            if CaptureNode('CheckTerminal',line):
                                
                                # load full rule line with class and support and accuracy
                                                               
                                Nodedata['Level'+str(NodeLevel)+'ALTNode'] = CaptureNode('Terminal',line) 
                               
                                # check Node0 is not terminated and make rule
                                                                
                                Nodedata['L'+str(NodeLevel)+'B2Rule'] = str(Nodedata['L'+str(ParentNodeLevel)+'B2Rule']) + " AND " + str(Nodedata['Level'+str(NodeLevel)+'ALTNode'])
                                
                                # strip special characters
                               
                                Nodedata['L'+str(NodeLevel)+'B2Rule'] = re.sub("[\[\]\']", "",str(Nodedata['L'+str(NodeLevel)+'B2Rule']))  
                                
                                                     
                                # Record Current Rule Via RecordRule Function
                            
                                Current_Rule = str(Nodedata['L'+str(NodeLevel)+'B2Rule'])
                        
                                RecordRule(Current_Rule)
                                
                                                       
                                #move to BranchL1 to 1
                               
                                Nodedata['BranchL'+str(NodeLevel)] = 1
                                
                                #print (Nodedata['L'+str(NodeLevel)+'B2Rule'])
                    
                #increment array/linecount 
                linecount += 1
    # =============================================================================
    # END FOREST LOOP - Provide each individual tree to the KD rule learner
    # =============================================================================         
     
        # =============================================================================
        # Export Main Dictionary into Nodedata_Contents Array for easy reading
        # =============================================================================
        
        i = 0
       
        # Make Nodedata contents global for easy debugging
        global Nodedata_Contents
        Nodedata_Contents = []
        
        for key,value in Nodedata.items():
            
            print ('NodeData Internal Variables Generated By SySFor Algorithm')
            LogfileClassificationArray.append(['NodeData Internal Variables Generated By SySFor Algorithm'])
            print (str(key) + " => " + str(value))
            LogfileClassificationArray.append([str(key) + " => " + str(value)])
    
            Nodedata_Contents.insert(i,str(key) + " => " + str(value)) 
            i= i + 1 

   

    # =============================================================================
    # END OF SySForKD FUNCTION
    # =============================================================================
    
# =============================================================================
# CART KD Function
# =============================================================================  

def CART_genKDfunction(CART,KDParticipant,wekaexperimentname):


    # Declare GLOBAL LogfileClassificationArray for recording events
    
    import re  #import regular expression library
    import datetime
        
    # Declare GLOBAL  Nodedata dictionary for storing temporary branch and tree data
    global Nodedata
    Nodedata = {}
            
    global LogfileClassificationArray
          
    #Store in Nodedata Key Function Values
    Nodedata["KDParticipant"] = KDParticipant
    Nodedata["wekaexperimentname"] = wekaexperimentname
    
    try: LogfileClassificationArray
    except NameError:
        #Initialise LogFile First it runs
        LogfileClassificationArray = []

    now = datetime.datetime.now()
    print ("Current date and time : " + now.strftime("%Y-%m-%d %H:%M:%S"))
    LogfileClassificationArray.append(["Current date and time : " + now.strftime("%Y-%m-%d %H:%M:%S")])
    
    
    # =============================================================================
    # Capture Node Rule Info using custom filter
    # =============================================================================   
    
    
    def CaptureNode(NodeType,Line):
        
        global Nodeline
        
        if NodeType == "Normal":
            
            # Capture a normal node like Ch37_PGSD <= 0.10518
            #Nodeline = re.findall(r"(\w+\s+[><=]+\s+[-]*\d+[.]*\d*)",str(line))
            
            # This regex patern targets the attribute name and the operants such as =>< and copies all the criterion be it numeric or text
            
            Nodeline = re.findall(r"(\w+\s+[><=]+.*)",str(line)) # Check for numerical values ie HR_SD < 3.5: NegValence(15.0/1.0)
            
            # Check For nominal paterns it could have != with no space betweem field and nominal value presented in =()
            
            if len(Nodeline) == 0: # if previous regex returned nothing for numerical
                Nodeline = re.findall(r"(\w+[!=]+.*)",str(line)) # check for nominal values ie RESP_Mean_GlobalRef=(Above)
            
            return Nodeline
            
            
        if NodeType == "CheckTerminal":
            print ('YES CHECK TERMINAL')
            # check if the node is terminated by TARGETCLASS(d.d/d.d)
            Nodeline = re.search(r"(\w+\(.*\))",str(line))
           
            # Return boolean
            return Nodeline
        
        if NodeType == "Terminal":
            
            # Ch37GSR_Min <= -1.9499: NeutralValance (4.0/1.0) basically everything after the criterion digits
            # Looks for Criterion optional [-] optional [.] and optional decimal place
            #Nodeline = re.findall(r"(\w+\s+[><=]+\s+[-]*\d+[.]*\d*.*)",str(line))
            
            # This regex patern targets the attribute name and the operants such as =>< and copies all the criterion be it numeric or text
            Nodeline = re.findall(r"(\w+\s+[><=]+.*)",str(line))
           
            if len(Nodeline) == 0: # if previous regex returned nothing for numerical
                Nodeline = re.findall(r"(\w+[!=]+.*)",str(line)) # check for nominal values ie RESP_Mean_GlobalRef=(Above)
            
            
            return Nodeline
       
        
        
        # =============================================================================
        # END OF CAPTURE NODE FILTER FUNCTION
        # =============================================================================
        
    
    # CHECK IF THE TREE IS NOT EMPTY ie Number of Leaf Nodes: 1
    # IF IT IS WIPE IT SO THE ALGORTHM WILL SKIP
    CART_check = re.search(r"Number of Leaf Nodes: 1",CART)
    
    if CART_check: 
        print("No Tree has been produced - can not process")
        print(CART)
        LogfileClassificationArray.append(["No Tree has been produced - can not process"])
        LogfileClassificationArray.append([CART])
  
    # =============================================================================
    # PROCEED TO PROCESS THE TREE!
    # =============================================================================
  
    else:
        
        
        # Read Tree Attributes
        # Cut Cross Validation Results, leave only training data in CART
       
    
        #CART = int(re.sub("[\[\]\']", "",str(CART)))
        #Clean up Tree of all empty Line
        
        # Read Tree Attributes
        # Cut Cross Validation Results, leave only training data in CART
        
        global CART_Training
        # Capture from the begining of the tree output "Options.." to just before the "=== Stratified cross-validation ==="
        CART_Training = re.findall(r"\s(Options.*)\s=== Stratified cross-validation ===",CART,re.DOTALL)
        CART_Training = CART_Training[0] # Convert from list to string
        
              
        global CleanCART
        CleanCART = "".join([s for s in CART_Training.splitlines(True) if s.strip("\r\n")])
        
        # Replace any Nominal Paterns that contain | within the criteria 
        # ie HR_Min_GlobalRef=(Below)|(Above) become HR_Min_GlobalRef=(Below) OR (Above)
        
        CleanCART = re.sub(r"\)[|]\(", ") OR (", str(CleanCART))
        
        ### Extract Important Fields from Training Tree ###
        CART_Scheme = re.findall(r"Options:\s+(.*)\s",CleanCART)
        
        Nodedata["Model_Scheme"] = CART_Scheme # Load Scheme in Global Dictionary
        LogfileClassificationArray.append([CART_Scheme])
        
        global CART_Instances
        CART_Instances = re.findall(r"Total Number of Instances\s+(\d+)",CleanCART)
        # Clean up value and convert to int number
        CART_Instances = int(re.sub("[\[\]\']", "",str(CART_Instances)))
        
        Nodedata["Model_Instances"] = CART_Instances # Load in Global Dictionary
        LogfileClassificationArray.append([CART_Instances])
        
        global CART_TrainingCorrectInstances
        CART_TrainingCorrectInstances = re.findall(r"Correctly Classified Instances\s+(\d+)",CleanCART)
        CART_TrainingCorrectInstances = int(re.sub("[\[\]\']", "",str(CART_TrainingCorrectInstances)))
        
       
        LogfileClassificationArray.append([CART_TrainingCorrectInstances])
        
        global CART_TrainingAccuracy
        CART_TrainingAccuracy  = int(((float(CART_TrainingCorrectInstances)/float(CART_Instances))*100))
        CART_TrainingAccuracy = str(CART_TrainingAccuracy) + "%"
        
        Nodedata["Model_TrainingAccuracy"] = CART_TrainingAccuracy # Load in Global Dictionary
        
         # Capture "=== Stratified cross-validation ===" section
        global CART_Stratified
        CART_Stratified = re.findall(r"(=== Stratified cross-validation.*)",CART,re.DOTALL)
        CART_Stratified = CART_Stratified[0] # Convert from list to string
        
        global CART_Stratified_Clean
        CART_Stratified_Clean = "".join([s for s in CART_Stratified.splitlines(True) if s.strip("\r\n")])
        
        global CART_StratifiedCorrectInstances
        CART_StratifiedCorrectInstances = re.findall(r"Correctly Classified Instances\s+(\d+)",CART_Stratified_Clean)
        CART_StratifiedCorrectInstances  = int(re.sub("[\[\]\']", "",str(CART_StratifiedCorrectInstances)))
        
        global CART_StratifiedAccuracy
        CART_StratifiedAccuracy  = int(((float(CART_StratifiedCorrectInstances)/float(CART_Instances))*100))
        CART_StratifiedAccuracy = str(CART_StratifiedAccuracy) + "%"
        
        Nodedata["Model_StratifiedAccuracy"] = CART_StratifiedAccuracy # Load in Global Dictionary
        
        # =============================================================================
        # Capture number of terminal nodes in tree and add each time a CART tree is processed 
        # to ensure final rules= number are correct
        # =============================================================================
        
        # Capture number of terminal nodes
        CART_NumberofRules = re.findall(r"Number of Leaf Nodes:\s+(\d+)",CleanCART)
        # Clean up value and convert to int number
        CART_NumberofRules = int(re.sub("[\[\]\']", "",str(CART_NumberofRules)))
        
        Nodedata["Model_NumberofRules"] = CART_NumberofRules
        
        global NumberofRulesProcessed
        
        try: NumberofRulesProcessed
        except NameError:
            #Initialise first time the function runs
             NumberofRulesProcessed = 0
             
        #Add Current Tree Number of Rules to global variable
        
        NumberofRulesProcessed =  NumberofRulesProcessed + CART_NumberofRules
        
    
        # =============================================================================
        # Capture confusion matrix and calculate number of records per class
        # =============================================================================
        
        # Capture in an CART_ConfusionMatrix array each line of the confusion matrix ie 11  2  5 |  a = PosValance
        CART_ConfusionMatrix = re.findall(r"\s+(\d+\s+\d+\s*\d*)\s+[|]\s+\w\s[=]\s(\w+)",CleanCART,re.DOTALL)
        print (CART_ConfusionMatrix)
        LogfileClassificationArray.append([CART_ConfusionMatrix])
    
        for line in range(len(CART_ConfusionMatrix)): 
          line1 =  CART_ConfusionMatrix[line][0]  
          print (line1)
          LogfileClassificationArray.append([line1])
          
          digit_line_split = line1.split()
          print (digit_line_split)
          LogfileClassificationArray.append([digit_line_split])
          
          digit_line_sum = sum([int(i) for i in digit_line_split if type(i)== int or i.isdigit()])
          line1Class = CART_ConfusionMatrix[line][1]  
          print (line1Class + "=" + str(digit_line_sum))
          LogfileClassificationArray.append([line1Class + "=" + str(digit_line_sum)])
          
          
          Nodedata[line1Class] = digit_line_sum
          Nodedata["Probabilityof"+ str(line1Class)] = round(((float(digit_line_sum)/float(CART_Instances))*100),2)
          
        
        # Capture between CART Decision Tree ....Number of Leaf Nodes: 1

        CART_Treenodes = re.findall(r"CART\sDecision\sTree\s(.*)\sNumber\sof\sLeaf\sNodes:\s",CleanCART,re.DOTALL)
        
        # split into seperate lines
        CART_Treenodes_str = "\n".join(CART_Treenodes)  # convert list into string
        CART_Treenodes_lines = CART_Treenodes_str.splitlines()
        
        #Process each line of the tree
        
        #set the array
        linecount = 0
    
    
       
        # =============================================================================
        # START PROCESSING TREE LINE BY LINE
        # =============================================================================   
        
        Nodedata['Level0Cycle'] = 1
        
        for line in CART_Treenodes_lines:
            print (str(linecount)+"  "+str(line))
            LogfileClassificationArray.append([str(linecount)+"  "+str(line)])
               
                 
            # Load Level0Node
              
            # Check line for Node level that it doesnt contain |
            checkL0 = re.search("^[^|]*$",str(line))
            
            # Load Level0 Node and Level 0 Alt Node
            if checkL0:
                
                NodeLevel = 0
                                
                  
                if Nodedata['Level'+str(NodeLevel)+'Cycle']  == 1:
                    
                                                            
                    Nodedata['Level'+str(NodeLevel)+'Node'] = CaptureNode('Normal',line)
                    
                    
                    Nodedata['Level'+str(NodeLevel)+'Node'] = re.sub("[\[\]\']", "",str(Nodedata['Level'+str(NodeLevel)+'Node']))
                    
                    # load rule part of L0 B1 
                    
                    Nodedata['L'+str(NodeLevel)+'B1Rule'] = Nodedata['Level'+str(NodeLevel)+'Node']
                    
                    Nodedata['BranchL'+str(NodeLevel)] = 1
                     
                        
                    # check if the node is terminated by (d.d/d.d)
                                        
                    if CaptureNode('CheckTerminal',line):
                        #record rule to rules array
                        # load full rule line with class and support and accuracy
                                                
                        Nodedata['Level'+str(NodeLevel)+'Node'] = CaptureNode('Terminal',line)
                        
                                
                        # strip special characters
                        
                        Nodedata['Level'+str(NodeLevel)+'Node'] = re.sub("[\[\]\']", "",str(Nodedata['Level'+str(NodeLevel)+'Node']))
                        
                        # load rule part of L0 B1 
                        Nodedata['L'+str(NodeLevel)+'B1Rule'] = Nodedata['Level'+str(NodeLevel)+'Node']
                        
                        # Record Current Rule Via RecordRule Function
                        
                        Current_Rule = str(Nodedata['L'+str(NodeLevel)+'B1Rule'])
                        
                        RecordRule(Current_Rule)
                        
                        #move to BranchL0 to 2 since branch0 is terminated
                        
                        Nodedata['BranchL'+str(NodeLevel)] = 2
                    
                if Nodedata['Level'+str(NodeLevel)+'Cycle'] == 2:
                                       
                                        
                    Nodedata['Level'+str(NodeLevel)+'ALTNode'] = CaptureNode('Normal',line)
                    
                     
                    Nodedata['Level'+str(NodeLevel)+'ALTNode'] = re.sub("[\[\]\']", "",str(Nodedata['Level'+str(NodeLevel)+'ALTNode']))
                    
                    # load rule part of L0 B2 
                    
                    Nodedata['L'+str(NodeLevel)+'B2Rule'] = Nodedata['Level'+str(NodeLevel)+'ALTNode']
                     
                    #BranchL0 = 2
                    Nodedata['BranchL'+str(NodeLevel)] = 2
                    
                    # check if the node is terminated by (d.d/d.d)
                   
                    Nodedata['CheckNode'+str(NodeLevel)+'Termin'] = re.search(r"(\(.*\))",str(line))
                    
                    
                    if CaptureNode('CheckTerminal',line):
                        #record rule to rules array
                        #load full rule line with class and support and accuracy
                        
                        
                        Nodedata['Level'+str(NodeLevel)+'ALTNode'] = CaptureNode('Terminal',line)
                        
                        # strip special characters
                        
                        Nodedata['Level'+str(NodeLevel)+'ALTNode'] = re.sub("[\[\]\']", "",str(Nodedata['Level'+str(NodeLevel)+'ALTNode']))
                        
                         # load rule part of L0 B2 
                        Nodedata['L'+str(NodeLevel)+'B2Rule'] = Nodedata['Level'+str(NodeLevel)+'ALTNode']
                        
                        # Record Current Rule Via RecordRule Function
                        
                        Current_Rule = str(Nodedata['L'+str(NodeLevel)+'B2Rule'])
                        
                        RecordRule(Current_Rule)
                        
                        #move to BranchL0 to 1 since branch0 is terminated
                     
                        Nodedata['BranchL'+str(NodeLevel)] = 1
                        
                        
                #if Level0Node != "":
                if Nodedata['Level'+str(NodeLevel)+'Node'] != "":
                    
                    # After L0 Branch 1 is complete move on to branch 2
                    #Level0Cycle = 2
                    Nodedata['Level'+str(NodeLevel)+'Cycle'] = 2
              
                
        # =============================================================================
        #     
        #     ## PROCESS LEVEL 1,2,3.. ONWARDS ##        
        # 
        # =============================================================================
           
        
            #Check line for Node level that it contains 1 x |
        
            checkLevel = re.findall(r"([|])",str(line))
            
            #countL1 = checkL1.count('|')
            
            NodeLevel = checkLevel.count('|')
            ParentNodeLevel = NodeLevel - 1
            
            
            # Load Level1 Node and Level1 Alt Node
            if checkLevel:
                                    
                # Load first level node for first cycle is done  
                
                cycle_key = 'Level'+str(NodeLevel)+'Cycle'
                if cycle_key in Nodedata:
                    
                    # Set variable for each cycle completed
                    Nodedata['Level'+str(NodeLevel)+'Cycle'] = 2
                else: 
                    
                    # Set variable for each cycle completed
                    Nodedata['Level'+str(NodeLevel)+'Cycle'] = 1
                
                                
                if Nodedata['Level'+str(NodeLevel)+'Cycle'] == 1:          
                
                    
                    # check which L0 branch you are on
                                       
                    Nodedata['Level'+str(NodeLevel)+'Node'] = CaptureNode('Normal',line)
                    
                    #BranchL1 = 1
                    Nodedata['BranchL'+str(NodeLevel)] = 1
                    
                    # Set variable for each cycle completed
                    #Level1Cycle = 1
                    Nodedata['Level'+str(NodeLevel)+'Cycle'] = 1
                    
                    
                    # Deal with the level 0 branch 1 first node level 1 branch 1
                                        
                    if Nodedata['BranchL'+str(ParentNodeLevel)] == 1:  
                        
                            # check Node0 is not terminated and make rule
                            
                            Nodedata['L'+str(NodeLevel)+'B1Rule'] = str( Nodedata['L'+str(ParentNodeLevel)+'B1Rule']) + " AND " + str(Nodedata['Level'+str(NodeLevel)+'Node'])
                            
                                       
                            # strip special characters
                            
                            Nodedata['L'+str(NodeLevel)+'B1Rule'] = re.sub("[\[\]\']", "",str(Nodedata['L'+str(NodeLevel)+'B1Rule']))
                            
                            # check if the node is terminated by (d.d/d.d)
                            
                            Nodedata['CheckNode'+str(NodeLevel)+'Termin'] = re.search(r"(\(.*\))",str(line))
                        
                            
                            if CaptureNode('CheckTerminal',line):
                        
                                # load full rule line with class and support and accuracy
                                                                
                                Nodedata['Level'+str(NodeLevel)+'Node'] = CaptureNode('Terminal',line)
                        
                                # check Node0 is not terminated and make rule
                                
                                Nodedata['L'+str(NodeLevel)+'B1Rule'] = str(Nodedata['L'+str(ParentNodeLevel)+'B1Rule']) + " AND " + str(Nodedata['Level'+str(NodeLevel)+'Node'])
                        
                                # strip special characters
                                
                                Nodedata['L'+str(NodeLevel)+'B1Rule'] = re.sub("[\[\]\']", "",str(Nodedata['L'+str(NodeLevel)+'B1Rule']))
                        
                                                            
                                # Record Current Rule Via RecordRule Function
                        
                                Current_Rule = str(Nodedata['L'+str(NodeLevel)+'B1Rule'])
                        
                                RecordRule(Current_Rule)
                                
                                #move to BranchL1 to 2
                             
                                Nodedata['BranchL'+str(NodeLevel)] = 2
                        
                        
                                #print (L1B1Rule)
                    
                            # Deal with the level 0 branch 2 first node level 1 branch 1 if BranchL0 = 2              
                            
                    if Nodedata['BranchL'+str(ParentNodeLevel)] == 2: 
                           
                            Nodedata['L'+str(NodeLevel)+'B1Rule'] = str( Nodedata['L'+str(ParentNodeLevel)+'B2Rule']) + " AND " + str(Nodedata['Level'+str(NodeLevel)+'Node'])
                        
                        
                            #BranchL1 = 1
                            Nodedata['BranchL'+str(NodeLevel)] = 1
                        
                            # strip special characters
                            
                            Nodedata['L'+str(NodeLevel)+'B1Rule'] = re.sub("[\[\]\']", "",str(Nodedata['L'+str(NodeLevel)+'B1Rule']))
                        
                            # check if the node is terminated by (d.d/d.d)
                            
                            Nodedata['CheckNode'+str(NodeLevel)+'Termin'] = re.search(r"(\(.*\))",str(line))
                        
                            
                            if CaptureNode('CheckTerminal',line):
                            
                                # load full rule line with class and support and accuracy
                                                                
                                Nodedata['Level'+str(NodeLevel)+'Node'] = CaptureNode('Terminal',line)
                                
                                # check Node0 is not terminated and make rule
                                
                                Nodedata['L'+str(NodeLevel)+'B1Rule'] = str(Nodedata['L'+str(ParentNodeLevel)+'B2Rule']) + " AND " + str(Nodedata['Level'+str(NodeLevel)+'Node'])
                                
                                # strip special characters
                                
                                Nodedata['L'+str(NodeLevel)+'B1Rule'] = re.sub("[\[\]\']", "",str(Nodedata['L'+str(NodeLevel)+'B1Rule']))
                                
                                                           
                                # Record Current Rule Via RecordRule Function
                        
                                Current_Rule = str(Nodedata['L'+str(NodeLevel)+'B1Rule'])
                        
                                RecordRule(Current_Rule)
                                
                               
                                #BranchL1 = 2
                                Nodedata['BranchL'+str(NodeLevel)] = 2
                             
                                #print (L1B1Rule)
                        
                            
                        # Load ALT level node if first cycle is done  
                
                #if Level1Cycle == 2:
                if Nodedata['Level'+str(NodeLevel)+'Cycle'] == 2:   
                    
                    # Deal with the first node level 1 branch 2
                                       
                    Nodedata['Level'+str(NodeLevel)+'ALTNode'] = CaptureNode('Normal',line) 
                   
                    #BranchL1 = 2
                    Nodedata['BranchL'+str(NodeLevel)] = 2
                    
                    # check which L0 branch you are on, if branch = 1
                    #if BranchL0 == 1:
                    if Nodedata['BranchL'+str(ParentNodeLevel)] == 1:     
                        
                        Nodedata['L'+str(NodeLevel)+'B2Rule'] = str( Nodedata['L'+str(ParentNodeLevel)+'B1Rule']) + " AND " + str(Nodedata['Level'+str(NodeLevel)+'ALTNode'])
                        
                        # strip special characters
                        
                        Nodedata['L'+str(NodeLevel)+'B2Rule'] = re.sub("[\[\]\']", "",str(Nodedata['L'+str(NodeLevel)+'B2Rule']))  
                        
                        #check if the node is terminated by (d.d/d.d)
                        
                        Nodedata['CheckNode'+str(NodeLevel)+'Termin'] = re.search(r"(\(.*\))",str(line))
                        
                        
                        if CaptureNode('CheckTerminal',line):
                            
                                             
                            # load full rule line with class and support and accuracy
                                                       
                            Nodedata['Level'+str(NodeLevel)+'ALTNode'] = CaptureNode('Terminal',line)
                            
                            # check Node0 is not terminated and make rule
                                                        
                            Nodedata['L'+str(NodeLevel)+'B2Rule'] = str(Nodedata['L'+str(ParentNodeLevel)+'B1Rule']) + " AND " + str(Nodedata['Level'+str(NodeLevel)+'ALTNode'])
                            
                            # strip special characters
                            
                            Nodedata['L'+str(NodeLevel)+'B2Rule'] = re.sub("[\[\]\']", "",str(Nodedata['L'+str(NodeLevel)+'B2Rule']))  
                            
                                                   
                            # Record Current Rule Via RecordRule Function
                        
                            Current_Rule = str(Nodedata['L'+str(NodeLevel)+'B2Rule'])
                        
                            RecordRule(Current_Rule)
                            
                                               
                            #move to BranchL1 to 1
                           
                            Nodedata['BranchL'+str(NodeLevel)] = 1
                            
                            #print (L1B2Rule)
                        
                                   
                    # check which L0 branch you are on, if branch = 2
                    
                    if Nodedata['BranchL'+str(ParentNodeLevel)] == 2:  
                        
                        #BranchL1 = 2
                        Nodedata['BranchL'+str(NodeLevel)] = 2           
                        
                        Nodedata['L'+str(NodeLevel)+'B2Rule'] = str( Nodedata['L'+str(ParentNodeLevel)+'B2Rule']) + " AND " + str(Nodedata['Level'+str(NodeLevel)+'ALTNode'])
                        
                        # strip special characters
                        Nodedata['L'+str(NodeLevel)+'B2Rule'] = re.sub("[\[\]\']", "",str(Nodedata['L'+str(NodeLevel)+'B2Rule']))  
                        
                        # check if the node is terminated by (d.d/d.d)
                        
                        Nodedata['CheckNode'+str(NodeLevel)+'Termin'] = re.search(r"(\(.*\))",str(line))
                        
                        #if Nodedata['CheckNode'+str(NodeLevel)+'Termin']:
                        if CaptureNode('CheckTerminal',line):
                            
                            # load full rule line with class and support and accuracy
                                                       
                            Nodedata['Level'+str(NodeLevel)+'ALTNode'] = CaptureNode('Terminal',line) 
                           
                            # check Node0 is not terminated and make rule
                                                        
                            Nodedata['L'+str(NodeLevel)+'B2Rule'] = str(Nodedata['L'+str(ParentNodeLevel)+'B2Rule']) + " AND " + str(Nodedata['Level'+str(NodeLevel)+'ALTNode'])
                            
                            # strip special characters
                           
                            Nodedata['L'+str(NodeLevel)+'B2Rule'] = re.sub("[\[\]\']", "",str(Nodedata['L'+str(NodeLevel)+'B2Rule']))  
                            
                                                 
                            # Record Current Rule Via RecordRule Function
                        
                            Current_Rule = str(Nodedata['L'+str(NodeLevel)+'B2Rule'])
                    
                            RecordRule(Current_Rule)
                            
                                                   
                            #move to BranchL1 to 1
                           
                            Nodedata['BranchL'+str(NodeLevel)] = 1
                            
                            #print (Nodedata['L'+str(NodeLevel)+'B2Rule'])
                
            #increment array/linecount 
            linecount += 1
            
       
       
        # =============================================================================
        # Export Main Dictionary into Nodedata_Contents Array for easy reading
        # =============================================================================
        
        i = 0
       
        # Make Nodedata contents global for easy debugging
        global Nodedata_Contents
        Nodedata_Contents = []
        
        for key,value in Nodedata.items():
            
            print ('NodeData Internal Variables Generated By CART Algorithm')
            LogfileClassificationArray.append(['NodeData Internal Variables Generated By CART Algorithm'])
            print (str(key) + " => " + str(value))
            LogfileClassificationArray.append([str(key) + " => " + str(value)])
    
            Nodedata_Contents.insert(i,str(key) + " => " + str(value)) 
            i= i + 1 

   

    # =============================================================================
    # END OF CART KD FUNCTION
    # =============================================================================
    

# =============================================================================
# RANDOM FOREST KD Function
# =============================================================================  

def RandomForest_genKDfunction(RandomForest,KDParticipant,wekaexperimentname):

    global Nodedata
    Nodedata = {}
    
    # Declare GLOBAL LogfileClassificationArray for recording events
    global LogfileClassificationArray
    
    import re  #import regular expression library
    import datetime
    import numpy as np
        
    #Store in Nodedata Key Function Values
    Nodedata["KDParticipant"] = KDParticipant
    Nodedata["wekaexperimentname"] = wekaexperimentname

    
    try: LogfileClassificationArray
    except NameError:
        #Initialise LogFile First it runs
        LogfileClassificationArray = []

    now = datetime.datetime.now()
    print ("Current date and time : " + now.strftime("%Y-%m-%d %H:%M:%S"))
    LogfileClassificationArray.append(["Current date and time : " + now.strftime("%Y-%m-%d %H:%M:%S")])
    
        
    # =============================================================================
    # Capture Node Rule Info using custom filter
    # =============================================================================   
    
       
    def CaptureNode(NodeType,Line):
        
        global Nodeline
        
        if NodeType == "Normal":
                                    
            # This regex patern targets the attribute name and the operants such as =>< and copies all the criterion be it numeric or text
            Nodeline = re.findall(r"(\w+\s+[><=]+.*)",str(line))
            
            return Nodeline
            
            
        if NodeType == "CheckTerminal":
            print ('YES CHECK TERMINAL')
            # check if the node is terminated by (d.d/d.d)
            Nodeline = re.search(r"(\(.*\))",str(line))
            # Return boolean
            return Nodeline
        
        if NodeType == "Terminal":
            
                        
            # This regex patern targets the attribute name and the operants such as =>< and copies all the criterion be it numeric or text
            Nodeline = re.findall(r"(\w+\s+[><=]+.*)",str(line))
            
            return Nodeline
       
        
        
        # =============================================================================
        # END OF CAPTURE NODE FILTER FUNCTION
        # =============================================================================
    
            
    # Read Tree Attributes
    # Cut Cross Validation Results, leave only training data in RandomForest
    
    global RandomForest_Training
    # Capture from the begining of the tree output "Options.." to just before the "=== Stratified cross-validation ==="
    RandomForest_Training = re.findall(r"\s(Options.*)\s=== Stratified cross-validation ===",RandomForest,re.DOTALL)
    RandomForest_Training = RandomForest_Training[0] # Convert from list to string
    
          
    global CleanRandomForest
    CleanRandomForest = "".join([s for s in RandomForest_Training.splitlines(True) if s.strip("\r\n")])
    
    ### Extract Important Fields from Training Tree ###
    global RandomForest_Scheme
    RandomForest_Scheme = re.findall(r"Options:\s+(.*)\s",CleanRandomForest)
    
    Nodedata["Model_Scheme"] = RandomForest_Scheme  # Load in Global Dictionary
    LogfileClassificationArray.append([RandomForest_Scheme])
    
    global RandomForest_Instances
    RandomForest_Instances = re.findall(r"Total Number of Instances\s+(\d+)",CleanRandomForest)
    # Clean up value and convert to int number
    RandomForest_Instances = int(re.sub("[\[\]\']", "",str(RandomForest_Instances)))
    
    Nodedata["Model_Instances"] = RandomForest_Instances # Load in Global Dictionary
    LogfileClassificationArray.append([RandomForest_Instances])
    
    global RandomForest_TrainingCorrectInstances
    RandomForest_TrainingCorrectInstances = re.findall(r"Correctly Classified Instances\s+(\d+)",CleanRandomForest)
    RandomForest_TrainingCorrectInstances = int(re.sub("[\[\]\']", "",str(RandomForest_TrainingCorrectInstances)))
    LogfileClassificationArray.append([RandomForest_TrainingCorrectInstances])
    
    global RandomForest_TrainingAccuracy
    RandomForest_TrainingAccuracy  = int(((float(RandomForest_TrainingCorrectInstances)/float(RandomForest_Instances))*100))
    RandomForest_TrainingAccuracy = str(RandomForest_TrainingAccuracy) + "%"
    
    Nodedata["Model_TrainingAccuracy"] = RandomForest_TrainingAccuracy # Load in Global Dictionary
    
    # =============================================================================
    # Capture confusion matrix and calculate number of records per class
    # =============================================================================
            
    # Capture in an RandomForest_ConfusionMatrix array each line of the confusion matrix ie 11  2  5 |  a = PosValance
    RandomForest_ConfusionMatrix = re.findall(r"\s+(\d+\s+\d+\s*\d*)\s+[|]\s+\w\s[=]\s(\w+)",CleanRandomForest,re.DOTALL)
    print (RandomForest_ConfusionMatrix)
    LogfileClassificationArray.append([RandomForest_ConfusionMatrix])

    for line in range(len(RandomForest_ConfusionMatrix)): 
      line1 =  RandomForest_ConfusionMatrix[line][0]  
      print (line1)
      LogfileClassificationArray.append([line1])
      
      digit_line_split = line1.split()
      print (digit_line_split)
      LogfileClassificationArray.append([digit_line_split])
      
      digit_line_sum = sum([int(i) for i in digit_line_split if type(i)== int or i.isdigit()])
      line1Class = RandomForest_ConfusionMatrix[line][1]  
      print (line1Class + "=" + str(digit_line_sum))
      LogfileClassificationArray.append([line1Class + "=" + str(digit_line_sum)])
      
      
      Nodedata[line1Class] = digit_line_sum
      Nodedata["Probabilityof"+ str(line1Class)] = round(((float(digit_line_sum)/float(RandomForest_Instances))*100),2)

    
     # Capture "=== Stratified cross-validation ===" section
    global RandomForest_Stratified
    RandomForest_Stratified = re.findall(r"(=== Stratified cross-validation.*)",RandomForest,re.DOTALL)
    RandomForest_Stratified = RandomForest_Stratified[0] # Convert from list to string
    
    global RandomForest_Stratified_Clean
    RandomForest_Stratified_Clean = "".join([s for s in RandomForest_Stratified.splitlines(True) if s.strip("\r\n")])
    
    global RandomForest_StratifiedCorrectInstances
    RandomForest_StratifiedCorrectInstances = re.findall(r"Correctly Classified Instances\s+(\d+)",RandomForest_Stratified_Clean)
    RandomForest_StratifiedCorrectInstances  = int(re.sub("[\[\]\']", "",str(RandomForest_StratifiedCorrectInstances)))
    
    global RandomForest_StratifiedAccuracy
    RandomForest_StratifiedAccuracy  = int(((float(RandomForest_StratifiedCorrectInstances)/float(RandomForest_Instances))*100))
    RandomForest_StratifiedAccuracy = str(RandomForest_StratifiedAccuracy) + "%"
    
    Nodedata["Model_StratifiedAccuracy"] = RandomForest_StratifiedAccuracy # Load in Global Dictionary
    
    
    #Capture just the Random Forest in List
    global JustForest
    JustForest = re.findall(r"\n{3}(RandomTree.*Size\sof\sthe\stree\s:\s\d+\n{4})",RandomForest_Training,re.DOTALL)
    
    # Capture Each Tree as a list
    global Trees
    Trees = re.findall('^(.+)(?:\n|\r\n?)((?:(?:\n|\r\n?).+)+)',JustForest[0],re.MULTILINE)
    
    # Capture the Tree Size Info as list
    global TreeSizes
    TreeSizes = re.findall(r"Size\sof\sthe\stree\s:\s(\d+)\n",JustForest[0])
    
    # Number of Trees
    global NumberTrees
    NumberTrees = len(Trees)
    
    # Setup a Numpy Array to load all Info, increase length by one to take into account array headings
    global ForestArray
    ForestArray = np.zeros((NumberTrees+1,3)).astype(object)
    ForestArray[0][0] = 'TreeNum'
    ForestArray[0][1] = 'TreeNodes'
    ForestArray[0][2] = 'TreeSize'
    
    # Loop through all the trees, start at position 1 to write after te headings
    for treenum in range(1,NumberTrees +1):
        ForestArray[treenum][0] = treenum
        ForestArray[treenum][1] = Trees[treenum-1][1] # Adjust by -1 row since Trees have no headings
        ForestArray[treenum][2] = TreeSizes[treenum-1]
        
    
    # CHECK IF THE TREE IS NOT EMPTY ie Number of Leaf Nodes: 1
    # IF IT IS WIPE IT SO THE ALGORTHM WILL SKIP
    RandomForest_check = re.search(r"Number of Leaf Nodes: 1",RandomForest)
    
        
    # =============================================================================
    # FOREST LOOP - Provide each individual tree to the KD rule learner
    # =============================================================================
    
    for ForestLoop in range(1,NumberTrees +1):
    
        CurrentTreeSize = int(ForestArray[ForestLoop][2])
    
        if CurrentTreeSize == 1: 
            print("Tree number " + str(ForestLoop) + " has not produced any rules - Tree has only 1 node")
            print(RandomForest)
            LogfileClassificationArray.append(["Tree number " + str(ForestLoop) + " has not produced any rules - Tree has only 1 node"])
            LogfileClassificationArray.append([RandomForest])
      
        # =============================================================================
        # PROCEED TO PROCESS THE TREE!
        # =============================================================================
      
        else:
            
            # Capture tree from Forest
            CurrentTree = ForestArray[ForestLoop][1]
            
            #Cleanup any empty line
            CurrentTree = "".join([s for s in CurrentTree.splitlines(True) if s.strip("\r\n")])
            
            # =============================================================================
            # Capture number of terminal nodes in tree and add each time a RandomForest tree is processed 
            # to ensure final rules= number are correct
            # =============================================================================
            
            # Capture number of terminal nodes cound the number of terminal nodes in current tree (dd/dd)
            RandomForest_NumberofRules = len(re.findall(r"\(\d\d?\/\d\d?\)",CurrentTree))
           
            
            Nodedata["Model_NumberofRules"] = RandomForest_NumberofRules 
            
            global NumberofRulesProcessed
            
            
            try: NumberofRulesProcessed
            except NameError:
                #Initialise first time the function runs
                 NumberofRulesProcessed = 0
                 
            #Add Current Tree Number of Rules to global variable
            
            NumberofRulesProcessed =  NumberofRulesProcessed + RandomForest_NumberofRules
           
            
            # Capture between RandomForest Decision Tree ....Number of Leaf Nodes: 1
                        
            # split TREE into seperate lines
            
            RandomForest_Treenodes_lines = CurrentTree.splitlines()
            
            #Process each line of the tree
            
            #set the array
            linecount = 0
        
        
           
            # =============================================================================
            # START PROCESSING TREE LINE BY LINE
            # =============================================================================   
            
            Nodedata['Level0Cycle'] = 1
            
            for line in RandomForest_Treenodes_lines:
                print (str(linecount)+"  "+str(line))
                LogfileClassificationArray.append([str(linecount)+"  "+str(line)])
                   
                     
                # Load Level0Node
                  
                # Check line for Node level that it doesnt contain |
                checkL0 = re.search("^[^|]*$",str(line))
                
                # Load Level0 Node and Level 0 Alt Node
                if checkL0:
                    
                    NodeLevel = 0
                                        
                      
                    if Nodedata['Level'+str(NodeLevel)+'Cycle']  == 1:
                                                                   
                        
                        Nodedata['Level'+str(NodeLevel)+'Node'] = CaptureNode('Normal',line)
                        
                        
                        Nodedata['Level'+str(NodeLevel)+'Node'] = re.sub("[\[\]\']", "",str(Nodedata['Level'+str(NodeLevel)+'Node']))
                        
                        # load rule part of L0 B1 
                        
                        Nodedata['L'+str(NodeLevel)+'B1Rule'] = Nodedata['Level'+str(NodeLevel)+'Node']
                        
                        Nodedata['BranchL'+str(NodeLevel)] = 1
                         
                            
                        # check if the node is terminated by (d.d/d.d)
                        
                        
                        if CaptureNode('CheckTerminal',line):
                            #record rule to rules array
                            # load full rule line with class and support and accuracy
                                                        
                            Nodedata['Level'+str(NodeLevel)+'Node'] = CaptureNode('Terminal',line)
                            
                                    
                            # strip special characters
                            
                            Nodedata['Level'+str(NodeLevel)+'Node'] = re.sub("[\[\]\']", "",str(Nodedata['Level'+str(NodeLevel)+'Node']))
                            
                            # load rule part of L0 B1 
                            Nodedata['L'+str(NodeLevel)+'B1Rule'] = Nodedata['Level'+str(NodeLevel)+'Node']
                            
                            # Record Current Rule Via RecordRule Function
                            
                            Current_Rule = str(Nodedata['L'+str(NodeLevel)+'B1Rule'])
                            
                            RecordRule(Current_Rule)
                            
                            #move to BranchL0 to 2 since branch0 is terminated
                            
                            Nodedata['BranchL'+str(NodeLevel)] = 2
                        
                    if Nodedata['Level'+str(NodeLevel)+'Cycle'] == 2:
                                                
                        Nodedata['Level'+str(NodeLevel)+'ALTNode'] = CaptureNode('Normal',line)
                        
                       
                        Nodedata['Level'+str(NodeLevel)+'ALTNode'] = re.sub("[\[\]\']", "",str(Nodedata['Level'+str(NodeLevel)+'ALTNode']))
                        
                        # load rule part of L0 B2 
                        
                        Nodedata['L'+str(NodeLevel)+'B2Rule'] = Nodedata['Level'+str(NodeLevel)+'ALTNode']
                         
                        #BranchL0 = 2
                        Nodedata['BranchL'+str(NodeLevel)] = 2
                        
                        # check if the node is terminated by (d.d/d.d)
                        
                        Nodedata['CheckNode'+str(NodeLevel)+'Termin'] = re.search(r"(\(.*\))",str(line))
                        
                        #if Nodedata['CheckNode'+str(NodeLevel)+'Termin']:
                        if CaptureNode('CheckTerminal',line):
                            #record rule to rules array
                            #load full rule line with class and support and accuracy
                                                       
                            Nodedata['Level'+str(NodeLevel)+'ALTNode'] = CaptureNode('Terminal',line)
                            
                            # strip special characters
                            
                            Nodedata['Level'+str(NodeLevel)+'ALTNode'] = re.sub("[\[\]\']", "",str(Nodedata['Level'+str(NodeLevel)+'ALTNode']))
                            
                             # load rule part of L0 B2 
                            Nodedata['L'+str(NodeLevel)+'B2Rule'] = Nodedata['Level'+str(NodeLevel)+'ALTNode']
                            
                            # Record Current Rule Via RecordRule Function
                            
                            Current_Rule = str(Nodedata['L'+str(NodeLevel)+'B2Rule'])
                            
                            RecordRule(Current_Rule)
                            
                            #move to BranchL0 to 1 since branch0 is terminated
                         
                            Nodedata['BranchL'+str(NodeLevel)] = 1
                            
                            
                    #if Level0Node != "":
                    if Nodedata['Level'+str(NodeLevel)+'Node'] != "":
                        
                        # After L0 Branch 1 is complete move on to branch 2
                        #Level0Cycle = 2
                        Nodedata['Level'+str(NodeLevel)+'Cycle'] = 2
                  
                    
            # =============================================================================
            #     
            #     ## PROCESS LEVEL 1,2,3.. ONWARDS ##        
            # 
            # =============================================================================
               
            
                #Check line for Node level that it contains 1 x |
            
                checkLevel = re.findall(r"([|])",str(line))
                
                                
                NodeLevel = checkLevel.count('|')
                ParentNodeLevel = NodeLevel - 1
                
                
                # Load Level1 Node and Level1 Alt Node
                if checkLevel:
                                            
                    # Load first level node for first cycle is done  
                    
                    cycle_key = 'Level'+str(NodeLevel)+'Cycle'
                    if cycle_key in Nodedata:
                        
                        # Set variable for each cycle completed
                        Nodedata['Level'+str(NodeLevel)+'Cycle'] = 2
                    else: 
                        
                        # Set variable for each cycle completed
                        Nodedata['Level'+str(NodeLevel)+'Cycle'] = 1
                    
                    
                    #if Level1Cycle == 1:
                    if Nodedata['Level'+str(NodeLevel)+'Cycle'] == 1:          
                    
                        
                        # check which L0 branch you are on
                                                
                        Nodedata['Level'+str(NodeLevel)+'Node'] = CaptureNode('Normal',line)
                        
                        #BranchL1 = 1
                        Nodedata['BranchL'+str(NodeLevel)] = 1
                        
                        # Set variable for each cycle completed
                        #Level1Cycle = 1
                        Nodedata['Level'+str(NodeLevel)+'Cycle'] = 1
                        
                        
                        # Deal with the level 0 branch 1 first node level 1 branch 1
                                                
                        if Nodedata['BranchL'+str(ParentNodeLevel)] == 1:  
                            
                                # check Node0 is not terminated and make rule
                                
                                Nodedata['L'+str(NodeLevel)+'B1Rule'] = str( Nodedata['L'+str(ParentNodeLevel)+'B1Rule']) + " AND " + str(Nodedata['Level'+str(NodeLevel)+'Node'])
                                
                                           
                                # strip special characters
                                
                                Nodedata['L'+str(NodeLevel)+'B1Rule'] = re.sub("[\[\]\']", "",str(Nodedata['L'+str(NodeLevel)+'B1Rule']))
                                
                                # check if the node is terminated by (d.d/d.d)
                                
                                Nodedata['CheckNode'+str(NodeLevel)+'Termin'] = re.search(r"(\(.*\))",str(line))
                                                            
                                if CaptureNode('CheckTerminal',line):
                            
                                    # load full rule line with class and support and accuracy
                                                                        
                                    Nodedata['Level'+str(NodeLevel)+'Node'] = CaptureNode('Terminal',line)
                            
                                    # check Node0 is not terminated and make rule
                                   
                                    Nodedata['L'+str(NodeLevel)+'B1Rule'] = str(Nodedata['L'+str(ParentNodeLevel)+'B1Rule']) + " AND " + str(Nodedata['Level'+str(NodeLevel)+'Node'])
                            
                                    # strip special characters
                                    
                                    Nodedata['L'+str(NodeLevel)+'B1Rule'] = re.sub("[\[\]\']", "",str(Nodedata['L'+str(NodeLevel)+'B1Rule']))
                            
                                                                
                                    # Record Current Rule Via RecordRule Function
                            
                                    Current_Rule = str(Nodedata['L'+str(NodeLevel)+'B1Rule'])
                            
                                    RecordRule(Current_Rule)
                                    
                                    #move to BranchL1 to 2
                                 
                                    Nodedata['BranchL'+str(NodeLevel)] = 2
                            
                            
                                    #print (L1B1Rule)
                        
                                # Deal with the level 0 branch 2 first node level 1 branch 1 if BranchL0 = 2              
                                
                        if Nodedata['BranchL'+str(ParentNodeLevel)] == 2: 
                                
                                Nodedata['L'+str(NodeLevel)+'B1Rule'] = str( Nodedata['L'+str(ParentNodeLevel)+'B2Rule']) + " AND " + str(Nodedata['Level'+str(NodeLevel)+'Node'])
                            
                            
                                #BranchL1 = 1
                                Nodedata['BranchL'+str(NodeLevel)] = 1
                            
                                # strip special characters
                               
                                Nodedata['L'+str(NodeLevel)+'B1Rule'] = re.sub("[\[\]\']", "",str(Nodedata['L'+str(NodeLevel)+'B1Rule']))
                            
                                # check if the node is terminated by (d.d/d.d)
                                
                                Nodedata['CheckNode'+str(NodeLevel)+'Termin'] = re.search(r"(\(.*\))",str(line))
                            
                                #if Nodedata['CheckNode'+str(NodeLevel)+'Termin']:
                                if CaptureNode('CheckTerminal',line):
                                
                                    # load full rule line with class and support and accuracy
                                                                        
                                    Nodedata['Level'+str(NodeLevel)+'Node'] = CaptureNode('Terminal',line)
                                    
                                    # check Node0 is not terminated and make rule
                                    
                                    Nodedata['L'+str(NodeLevel)+'B1Rule'] = str(Nodedata['L'+str(ParentNodeLevel)+'B2Rule']) + " AND " + str(Nodedata['Level'+str(NodeLevel)+'Node'])
                                    
                                    # strip special characters
                                   
                                    Nodedata['L'+str(NodeLevel)+'B1Rule'] = re.sub("[\[\]\']", "",str(Nodedata['L'+str(NodeLevel)+'B1Rule']))
                                    
                                                               
                                    # Record Current Rule Via RecordRule Function
                            
                                    Current_Rule = str(Nodedata['L'+str(NodeLevel)+'B1Rule'])
                            
                                    RecordRule(Current_Rule)
                                    
                                   
                                    #BranchL1 = 2
                                    Nodedata['BranchL'+str(NodeLevel)] = 2
                                 
                                    #print (L1B1Rule)
                            
                                
                            # Load ALT level node if first cycle is done  
                    
                    #if Level1Cycle == 2:
                    if Nodedata['Level'+str(NodeLevel)+'Cycle'] == 2:   
                        
                        # Deal with the first node level 1 branch 2
                                               
                        Nodedata['Level'+str(NodeLevel)+'ALTNode'] = CaptureNode('Normal',line) 
                       
                        #BranchL1 = 2
                        Nodedata['BranchL'+str(NodeLevel)] = 2
                        
                        # check which L0 branch you are on, if branch = 1
                        
                        if Nodedata['BranchL'+str(ParentNodeLevel)] == 1:     
                                                        
                            Nodedata['L'+str(NodeLevel)+'B2Rule'] = str( Nodedata['L'+str(ParentNodeLevel)+'B1Rule']) + " AND " + str(Nodedata['Level'+str(NodeLevel)+'ALTNode'])
                            
                            # strip special characters
                           
                            Nodedata['L'+str(NodeLevel)+'B2Rule'] = re.sub("[\[\]\']", "",str(Nodedata['L'+str(NodeLevel)+'B2Rule']))  
                            
                            # check if the node is terminated by (d.d/d.d)
                            
                            Nodedata['CheckNode'+str(NodeLevel)+'Termin'] = re.search(r"(\(.*\))",str(line))
                                                        
                            if CaptureNode('CheckTerminal',line):
                                
                                                 
                                # load full rule line with class and support and accuracy
                                                                
                                Nodedata['Level'+str(NodeLevel)+'ALTNode'] = CaptureNode('Terminal',line)
                                
                                # check Node0 is not terminated and make rule
                                                                
                                Nodedata['L'+str(NodeLevel)+'B2Rule'] = str(Nodedata['L'+str(ParentNodeLevel)+'B1Rule']) + " AND " + str(Nodedata['Level'+str(NodeLevel)+'ALTNode'])
                                
                                # strip special characters
                                
                                Nodedata['L'+str(NodeLevel)+'B2Rule'] = re.sub("[\[\]\']", "",str(Nodedata['L'+str(NodeLevel)+'B2Rule']))  
                                
                                                       
                                # Record Current Rule Via RecordRule Function
                            
                                Current_Rule = str(Nodedata['L'+str(NodeLevel)+'B2Rule'])
                            
                                RecordRule(Current_Rule)
                                
                                                   
                                #move to BranchL1 to 1
                               
                                Nodedata['BranchL'+str(NodeLevel)] = 1
                                
                                #print (L1B2Rule)
                            
                                       
                        # check which L0 branch you are on, if branch = 2
                        
                        if Nodedata['BranchL'+str(ParentNodeLevel)] == 2:  
                            
                            #BranchL1 = 2
                            Nodedata['BranchL'+str(NodeLevel)] = 2           
                            
                            Nodedata['L'+str(NodeLevel)+'B2Rule'] = str( Nodedata['L'+str(ParentNodeLevel)+'B2Rule']) + " AND " + str(Nodedata['Level'+str(NodeLevel)+'ALTNode'])
                            
                            # strip special characters
                            
                            Nodedata['L'+str(NodeLevel)+'B2Rule'] = re.sub("[\[\]\']", "",str(Nodedata['L'+str(NodeLevel)+'B2Rule']))  
                            
                            # check if the node is terminated by (d.d/d.d)
                            
                            Nodedata['CheckNode'+str(NodeLevel)+'Termin'] = re.search(r"(\(.*\))",str(line))
                            
                            
                            if CaptureNode('CheckTerminal',line):
                                
                                # load full rule line with class and support and accuracy
                                                               
                                Nodedata['Level'+str(NodeLevel)+'ALTNode'] = CaptureNode('Terminal',line) 
                               
                                # check Node0 is not terminated and make rule
                                                                
                                Nodedata['L'+str(NodeLevel)+'B2Rule'] = str(Nodedata['L'+str(ParentNodeLevel)+'B2Rule']) + " AND " + str(Nodedata['Level'+str(NodeLevel)+'ALTNode'])
                                
                                # strip special characters
                                
                                Nodedata['L'+str(NodeLevel)+'B2Rule'] = re.sub("[\[\]\']", "",str(Nodedata['L'+str(NodeLevel)+'B2Rule']))  
                                
                                                     
                                # Record Current Rule Via RecordRule Function
                            
                                Current_Rule = str(Nodedata['L'+str(NodeLevel)+'B2Rule'])
                        
                                RecordRule(Current_Rule)
                                
                                                       
                                #move to BranchL1 to 1
                               
                                Nodedata['BranchL'+str(NodeLevel)] = 1
                                
                                #print (Nodedata['L'+str(NodeLevel)+'B2Rule'])
                    
                #increment array/linecount 
                linecount += 1
    # =============================================================================
    # END FOREST LOOP - Provide each individual tree to the KD rule learner
    # =============================================================================         
       
       
       
       
       
        # =============================================================================
        # Export Main Dictionary into Nodedata_Contents Array for easy reading
        # =============================================================================
        
        i = 0
       
        # Make Nodedata contents global for easy debugging
        global Nodedata_Contents
        Nodedata_Contents = []
        
        for key,value in Nodedata.items():
            
            print ('NodeData Internal Variables Generated By RandomForest Algorithm')
            LogfileClassificationArray.append(['NodeData Internal Variables Generated By RandomForest Algorithm'])
            print (str(key) + " => " + str(value))
            LogfileClassificationArray.append([str(key) + " => " + str(value)])
    
            Nodedata_Contents.insert(i,str(key) + " => " + str(value)) 
            i= i + 1 

   

    # =============================================================================
    # END OF RandomForest FUNCTION
    # =============================================================================
    
    



    
# =============================================================================
# PRINT RULES AND STORE FINAL RULESARRAY INTO CSV FILE & EXPORT LOGFILE
# =============================================================================   
def OutputRuleArray(path,ExperimentName):
    
    import re  #import regular expression library  
    # =============================================================================
    #     # printing the list using loop 
    # =============================================================================
    
    # CHECK IF RULES ARRAY IS DEFINED
    CheckRulesArray = True
    
    try: RulesArray

    except NameError:  CheckRulesArray = None
    
    # Only run if Array is defined
    if CheckRulesArray:
        
        for RuleIndx in range(len(RulesArray)): 
            # Print the rule from the RulesArray position 1
            print ("Rule " + str(RuleIndx) + " " + str(RulesArray[RuleIndx][1]),)
            LogfileClassificationArray.append(["Rule " + str(RuleIndx) + " " + str(RulesArray[RuleIndx][1])])
            
            print ("\n")
            LogfileClassificationArray.append(["\n"])
        #
        # adjust for number of rules minus - heading
        NumberofRulesinList = len(RulesArray) - 1
        
        if NumberofRulesProcessed  == NumberofRulesinList:
            print ("Number of Leaves= " + str(NumberofRulesinList) + " match Number of Rules= " + str(NumberofRulesinList))
            LogfileClassificationArray.append(["Number of Leaves= " + str(NumberofRulesinList) + " match Number of Rules= " + str(NumberofRulesinList)])
            
            print ("All rules are correct")
            LogfileClassificationArray.append(["All rules are correct"])
        else:
            print ("***Mismatch between tree nodes and rules generated***")
            LogfileClassificationArray.append(["***Mismatch between tree nodes and rules generated***"])
    
        # =============================================================================
        #     # OUT CSV FILE OF ALL THE RULES IN RulesArray
        # =============================================================================
        
        import csv
    
        # Add .csv extension
        
        ExperimentKDFile = path + ExperimentName + "_RulesKD_matrix.csv"
        
        csv.register_dialect('myRulesFormat',
                             delimiter = ',',
                             quoting=csv.QUOTE_NONE,
                             skipinitialspace=True)
    
        with open(ExperimentKDFile, 'w') as f:
            writer = csv.writer(f, dialect='myRulesFormat')
            for row in RulesArray:
                writer.writerow(row)
    
        f.close()
    
        # =============================================================================
        #     # OUT TXT LOGFILE FILE OF ALL THE StdOutput
        # =============================================================================
        
        ParticipantlogFilename = path + ExperimentName + "_RulesKD_ProcessLog.txt"
           
        with open(ParticipantlogFilename, "w") as file_handler:
            for item in LogfileClassificationArray:
                row = re.sub("[\[\]\']", "",str(item))
                file_handler.write("{}\n".format(row))
    
        
        # CLEAR THE RULES ARRAY AT THE END OF OUTPUT
        #del RulesArray[:]    
        #RulesArray.clear()
       
        #Reset Rules Array
        Reset_Rules_Array()

    
    else:

        print ("Rules Array NOT DEFINED!")        
        
    # =============================================================================
    # END OutputRuleArray
    # =============================================================================
        
# =============================================================================
# PRINT RULES AND STORE FINAL RULESARRAY INTO CSV FILE & EXPORT LOGFILE
# =============================================================================   


def Output_KD_Rule_Bank(output_dir):

    import os
    import csv

    # Base directory of this script
    base_path = os.path.dirname(os.path.abspath(__file__))

    # Single input/output directory
    CSV_Directory = os.path.join(base_path, output_dir)
    os.makedirs(CSV_Directory, exist_ok=True)

    # Find all CSV files in the directory (exclude the output file if it already exists)
    # Output file path
    output_file = os.path.join(CSV_Directory, "KD_Combined_Rule_Bank.csv")

    # Collect CSV files, excluding the output file
    csv_files = []
    for f in os.listdir(CSV_Directory):
        if f.lower().endswith(".csv"):
            full_path = os.path.join(CSV_Directory, f)
            if os.path.isfile(full_path) and os.path.abspath(full_path) != os.path.abspath(output_file):
                csv_files.append(full_path)

    csv_files.sort()

    # If 0 or 1 CSV files, do nothing
    if len(csv_files) <= 1:
        if len(csv_files) == 0:
            print(f"No CSV files found in: {CSV_Directory}")
        else:
            print(f"Only one CSV found ({os.path.basename(csv_files[0])}). Skipping merge.")
    else:
        # Merge: take header from the first file, skip headers in others
        with open(output_file, "w", newline="", encoding="utf-8") as out_f:
            writer = None
            for idx, in_path in enumerate(csv_files):
                with open(in_path, "r", newline="", encoding="utf-8") as in_f:
                    reader = csv.reader(in_f)
                    try:
                        header = next(reader)  # read header/first line
                    except StopIteration:
                        # empty file; skip
                        continue

                    if writer is None:
                        writer = csv.writer(out_f)
                        writer.writerow(header)  # write header once

                    for row in reader:
                        if not row or all(cell.strip() == "" for cell in row):
                            continue
                        writer.writerow(row)

        print(f"Merged {len(csv_files)} files into: {output_file}")
