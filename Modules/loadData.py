#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append("Modules/")
# sys.path.append("../")
import os

import pandas as pd
import numpy as np

import logging
import pickle

import json

#fmt = 'logging.Formatter(''%(levelname)s_%(name)s-%(funcName)s(): - %(message)s'
fmt = '%(levelname)s_%(name)s-%(funcName)s(): - %(message)s'
logging.basicConfig(level=logging.INFO, format=fmt)
logger = logging.getLogger(__name__)

# In[1]:

wd = os.getcwd()
os.chdir(wd)

################################################################################
#### f(trialData, trIdx)

def convertIndexToMultiIndexUsingUnderscore(labelIn):

    s = labelIn.split('_')

    if len(s) is 2:

        # otherwise, its already a scalar
        (first,second) = s
        return (first,second)

    elif len(s) == 3 :

        (first,desc,second)= s
        return (first,second)


    else:
        return (labelIn,'')

def convertIndexToMultiIndexIgnoringUnderscore(labelIn):

    if( labelIn.split('_')[-1] == 'x' or  labelIn.split('_')[-1] == 'y' or labelIn.split('_')[-1] == 'z' ):

        top = '-'.join(labelIn.split('_')[:-1])
        bottom = labelIn.split('_')[-1]
        return (top,bottom)

    else:

        return (labelIn,'')
        
def processTrial(dataFolder, trialResults, gazeConfidenceThreshold, numTrials = False):

    if(numTrials):
        logger.info('Processing subject: ' + trialResults['ppid'] + ' t = ' + str(trialResults['trial_num']) + ' of ' + str(numTrials) )
    else:
        logger.info('Processing subject: ' + trialResults['ppid'] + ' t = ' + str(trialResults['trial_num']))


    ## Import ball data and rename some columns
    dataFileName = trialResults['ball_movement_filename']
    ballData = pd.read_csv( dataFolder + dataFileName)
    ballData = ballData.rename(columns={"time": "frameTime"})
    ballData.rename(columns={"pos_x": "ballPos_x", "pos_y": "ballPos_y","pos_z": "ballPos_z"},inplace=True)
    ballData.rename(columns={"rot_x": "ballRot_x", "rot_y": "ballRot_y","rot_z": "ballRot_z"},inplace=True)

    ## Import paddle data and rename some columns
    dataFileName = trialResults['paddle_movement_filename']
    paddleData = pd.read_csv( dataFolder + dataFileName)
    paddleData = paddleData.rename(columns={"time": "frameTime"})

    ## Import view data and rename some columns
    dataFileName = trialResults['camera_movement_filename']
    viewData = pd.read_csv( dataFolder + dataFileName)
    viewData = viewData.rename(columns={"time": "frameTime"})

    ## Merge view and ball data into rawTrialData
    if( len(ballData) == 0 ):
        rawTrialData = viewData.reindex(viewData.columns.union(ballData.columns), axis=1)
    else:
        rawTrialData = pd.merge(viewData, ballData, on ='frameTime',validate= 'one_to_many')

    ## Merge rawTrialData and paddle data
    if( len(paddleData) == 0 ):
        rawTrialData = rawTrialData.reindex(viewData.columns.union(paddleData.columns), axis=1)
    else:
        rawTrialData = pd.merge(rawTrialData, paddleData, on ='frameTime',validate= 'one_to_many')

    ## Import and merge pupil timestamp data (recorded within Unity)
    dataFileName = trialResults['pupil_pupilTimeStamp_filename']
    pupilTimestampData = pd.read_csv( dataFolder + dataFileName)
    pupilTimestampData = pupilTimestampData.rename(columns={"time": "frameTime"})
    rawTrialData = pd.merge( rawTrialData, pupilTimestampData, on ='frameTime',validate= 'one_to_many')

    rawTrialData['trialNumber'] = trialResults['trial_num']
    rawTrialData['blockNumber'] = trialResults['block_num']

    ## Import gaze data
    gazeDataFolderList = []
    [gazeDataFolderList.append(name) for name in os.listdir(dataFolder + 'PupilData') if name[0] is not '.']
    
    pupilSessionFolder = '/' + gazeDataFolderList[0] 
    gazeDataFolder = dataFolder + 'PupilData' + pupilSessionFolder

    try:
        pupilExportsFolder = []
        [pupilExportsFolder.append(name) for name in os.listdir(gazeDataFolder + '/Exports') if name[0] is not '.']

        # Defaults to the most recent pupil export folder (highest number)
        gazePositionsDF = pd.read_csv( gazeDataFolder + '/Exports/' + pupilExportsFolder[-1] + '/gaze_positions.csv' )
        gazePositionsDF.head()
    except:
        logger.exception('No gaze_positions.csv.  Process and export data in Pupil Player.')

    gazePositionsDF = gazePositionsDF.rename(columns={"gaze_timestamp": "pupilTimestamp"})

    # Sort, because the left/right eye data is written asynchronously, and this means timestamps may not be monotonically increasing.
    gazePositionsDF.sort_values(by='pupilTimestamp',inplace=True)

    ## Merge gaze data
    gbBlTr = rawTrialData.groupby(['blockNumber','trialNumber'])
    tr = gbBlTr.get_group( list(gbBlTr.groups.keys())[0])

    ## Gaze data is for the whole experiment while trial data is for the trial only.
    ## Find the slice of gaze data that maps onto the trial timestamps
    firstTS = tr.head(1)['pupilTimestamp']
    lastTS = tr.tail(1)['pupilTimestamp']
    firstIdx = list(map(lambda i: i> float(firstTS), gazePositionsDF['pupilTimestamp'])).index(True)
    lastIdx = list(map(lambda i: i> float(lastTS), gazePositionsDF['pupilTimestamp'])).index(True)
    rawGazeData = gazePositionsDF.loc[firstIdx:lastIdx]
    
    # Drop data below the confidence level
    filteredGazeData = rawGazeData.reset_index().drop(np.where(rawGazeData['confidence'] < gazeConfidenceThreshold )[0])
    
    # Merge gaze and trial data
    interpDF = pd.merge( rawTrialData, filteredGazeData, on ='pupilTimestamp',how='outer',sort=True)

    # Upsample trial data to the resolution of gaze data
    interpDF = interpDF.interpolate(method='linear',downcast='infer')

    # Convert to multiindex
    newColList = [convertIndexToMultiIndexUsingUnderscore(c) for c in interpDF.columns[:len(rawTrialData.columns)]]
    newGazeColList = [ convertIndexToMultiIndexIgnoringUnderscore(c) for c in interpDF.columns[(len(rawTrialData.columns)):] ]
    newColList.extend(newGazeColList)
    interpDF.columns = pd.MultiIndex.from_tuples(newColList)

    ### Some checks ###
    if( len(interpDF) > ( len(rawTrialData) + len(filteredGazeData)) ):
        logger.warning('len(interpDF) > ( len(rawTrialData) + len(filteredGazeData))')

    dictOut = {"rawUnityData": rawTrialData, "rawGazeData": rawGazeData, "interpolatedData": interpDF}

    return dictOut

################################################################################
################################################################################

def unpackSession(subNum, gazeDataConfidenceThreshold = 0.6, doNotLoad = False):

    # Get folder/filenames
    dataFolderList = []
    [dataFolderList.append(name) for name in os.listdir("Data/") if name[0] is not '.']

    for i, name in enumerate(dataFolderList):
            if i == subNum:
                print('***> ' + str(i) + ': ' + name )
            else:
                print(str(i) + ': ' + name )
    

    dataParentFolder = "Data/" + dataFolderList[subNum]
    dataSubFolderList = []
    [dataSubFolderList.append(name) for name in os.listdir(dataParentFolder) if name[0] is not '.']
    dataFolder = dataParentFolder + '/' + dataSubFolderList[0] + '/'

    logger.info('Processing session: ' + dataParentFolder)

    # Try to load pickle if doNotLoad == False
    picklePath = dataFolder + dataSubFolderList[0] + '.pickle'
    from os import path
    if( doNotLoad == False and path.exists(picklePath)):
        
        file = open(picklePath, 'rb')
        sessionData = pickle.load(file)
        file.close()
        
        logger.info('Importing session dict from pickle.')
        
        return sessionData

    logger.info('Compiling session dict from *.csv.')

    # If not loading from pickle, create and populate dataframes
    rawExpUnityDataDf = pd.DataFrame()
    rawExpGazeDataDf = pd.DataFrame()
    processedExpDataDf = pd.DataFrame()

    rawCalibUnityDataDf = pd.DataFrame()
    rawCalibGazeDataDf = pd.DataFrame()
    processedCalibDataDf = pd.DataFrame()

    trialData = pd.read_csv( dataFolder + 'trial_results.csv')

    for trIdx, trialResults in trialData.iterrows():

        trialDict = processTrial(dataFolder, trialResults, gazeDataConfidenceThreshold,len(trialData))

        def addToDF(targetDF,dfIn):
            
            if( targetDF.empty ):
                targetDF = dfIn
            else:
                targetDF = targetDF.append(dfIn)

            return targetDF

        if (trialResults['trialType'] == 'interception'):

            processedExpDataDf = addToDF(processedExpDataDf,trialDict['interpolatedData'])
            rawExpUnityDataDf = addToDF(rawExpUnityDataDf,trialDict['rawUnityData'])
            rawExpGazeDataDf = addToDF(rawExpGazeDataDf,trialDict['rawGazeData'])

        elif(trialResults['trialType'] == 'CalibrationAssessment'):

            processedCalibDataDf = addToDF(processedCalibDataDf,trialDict['interpolatedData'])
            rawCalibUnityDataDf = addToDF(rawCalibUnityDataDf,trialDict['rawUnityData'])
            rawCalibGazeDataDf = addToDF(rawCalibGazeDataDf,trialDict['rawGazeData'])


    # Rename trialdata columns and convert to multiindex
    trialData.rename(columns={"session_num":"sessionNumber","trial_num":"trialNumber",
        "block_num":"blockNumber","trial_num_in_block":"trialNumberInBlock",
        "start_time":"startTime","end_time":"endTime"},inplace=True)

    trDataFiles = [i for i in trialData.columns.to_list() if '_filename' in i] 

    newColList = [convertIndexToMultiIndexUsingUnderscore(c) for c in trialData.columns[:-len(trDataFiles)]]
    newColList.extend([convertIndexToMultiIndexIgnoringUnderscore(c) for c in trialData.columns[-len(trDataFiles):]])
    trialData.columns = pd.MultiIndex.from_tuples(newColList)
    
    expDict = json.load( open(dataFolder + 'settings.json'))

    processedExpDataDf = processedExpDataDf.reset_index(drop=True)
    processedCalibDataDf = processedCalibDataDf.reset_index(drop=True)
    
    analysisParameters = json.load( open('analysisParameters.json'))
    analysisParameters['gazeDataConfidenceThreshold'] = gazeDataConfidenceThreshold

    subID = json.load( open(dataFolder + 'participant_details.json'))['ppid']

    dictOut = {"subID": subID, "trialInfo": trialData.sort_index(axis=1),"expConfig": expDict,
        "rawExpUnity": rawExpUnityDataDf.sort_index(axis=1), "rawExpGaze": rawExpGazeDataDf.sort_index(axis=1), "processedExp": processedExpDataDf.sort_index(axis=1),
        "rawCalibUnity": rawExpUnityDataDf.sort_index(axis=1), "rawCalibGaze": rawExpGazeDataDf.sort_index(axis=1), "processedCalib": processedCalibDataDf.sort_index(axis=1),
        "analysisParameters":analysisParameters}

    with open(picklePath, 'wb') as handle:
        pickle.dump(dictOut, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return dictOut




# def compileSubData():

#     rawDataDF = False
#     calibDF = False

#     # ### Load dataframes
#     # Remember to set loadParsedData, loadProcessedData.

#     dataFolderList = []
#     [dataFolderList.append(name) for name in os.listdir("Data/") if name[0] is not '.']

#     allSessionData = []

#     for subNum, subString in enumerate(dataFolderList):

#         dataParentFolder = "Data/" + subString

#         dataSubFolderList = [];
#         [dataSubFolderList.append(name) for name in os.listdir("Data/" + dataParentFolder) if name[0] is not '.'];

#         dataFolder = dataParentFolder + '/' + dataSubFolderList[0] + '/'
#         trialData = pd.read_csv( dataFolder + 'trial_results.csv')

#         for trIdx, trialResults in trialData.iterrows():

#             trialDF = processTrial(dataFolder, trialResults)

#             if (trialResults['trialType'] == 'interception'):
#                 if( rawDataDF is False):
#                     rawDataDF = trialDF
#                 else:
#                     rawDataDF = rawDataDF.append(trialDF)

#             elif(trialResults['trialType'] == 'CalibrationAssessment'):

#                 if( calibDF is False):
#                     calibDF = trialDF
#                 else:
#                     calibDF = calibDF.append(trialDF)

#         sessionDict = {"expInfo": trialData,"rawData": rawDataDF, "calibData": calibDF}
#         allSessionData.append(sessionDict)

#     with open('../allSessionData.pickle', 'wb') as handle:
#         pickle.dump(allSessionData, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %%

if __name__ == "__main__":

    import os

    subNum = 0

    rawDataDF = False
    calibDF = False

    # ### Load dataframes
    # Remember to set loadParsedData, loadProcessedData.

    dataFolderList = []
    [dataFolderList.append(name) for name in os.listdir("Data/") if name[0] is not '.']

    for i, name in enumerate(dataFolderList):
        print(str(i) + ': ' + name )

    sessionDict = unpackSession(subNum, gazeDataConfidenceThreshold = 0.6, doNotLoad = True)


# %%
