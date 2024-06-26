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

def getSubjectSubFolders(subNum,printSessionID=False):
    ##
    # returns: folderDict
    ##


    # Get folder/filenames
    dataParentFolderList = []
    [dataParentFolderList.append(name) for name in os.listdir("Data/") if ((name[0] is not '.' and (name[0:9] == "_Pipeline")))]
    dataFolder = "Data/" + dataParentFolderList[subNum] + '/'

    if printSessionID:
        for i, name in enumerate(dataParentFolderList):
            if i == subNum:
                print('***> ' + str(i) + ': ' + name)
            else:
                print(str(i) + ': ' + name)

    # eg. dataFolder = /P_201219105516_sub2/S001/


    # I always open the lowest valued sessionfolder (e.g. S001)
    plSessionFolderList = []
    try:
        [plSessionFolderList.append(name) for name in os.listdir(dataFolder + '/PupilData') if name[0] is not '.']
    except FileNotFoundError as _:
        dataFolder += "S001/"
        [plSessionFolderList.append(name) for name in os.listdir(dataFolder + '/PupilData') if name[0] is not '.']

    # eg. pupilSessionFolder = /P_201219105516_sub2/S001/PupilData/000/
    pupilSessionFolder = dataFolder + 'PupilData/' + plSessionFolderList[0] + '/'

    # Export folder list
    pupilExportsFolderList = []
    [pupilExportsFolderList.append(name) for name in os.listdir(pupilSessionFolder + 'Exports') if name[0] is not '.']

    folderDict = {'subName': dataParentFolderList[subNum],# P_201219105516_sub2
                  #'subFolder': dataParentFolder,# /P_201219105516_sub2/
                  'sessionFolder': dataFolder, # /P_201219105516_sub2/S001/
                  'pupilSessionFolder': pupilSessionFolder, #  /P_201219105516_sub2/S001/PupilData/000/
                  'pupilExportsFolderList': pupilExportsFolderList,  # [/P_201219105516_sub2/S001/PupilData/000/Exports/000]
                  }


    return folderDict

################################################################################
#### f(trialData, trIdx)

def convertIndexToMultiIndexUsingUnderscore(labelIn):

    s = labelIn.split('_')

    if len(s) == 2:

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

def processCalibSequence(dataFolder, load_realtime_ref_data, specificExport=None):
    
    dataParentFolder = '/'.join(dataFolder.split('/')[:-2]) + '/';
    
    ## Import pupil timestamp data (recorded within Unity)
    
    realtime_calib_points_loc = dataParentFolder + 'PupilData/000/realtime_calib_points.msgpack';
    
    if os.path.exists(realtime_calib_points_loc):
        ref_data = load_realtime_ref_data(realtime_calib_points_loc)
        ref_data_df = pd.DataFrame(ref_data)
    else:
        logger.exception('No realtime pupil timestamp data.')
    ref_data_df = ref_data_df.rename(columns={"timestamp": "pupilTimestamp"})
    ref_data_df[['screen-pos_x', 'screen-pos_y', 'screen-pos_z']] = pd.DataFrame(ref_data_df['screen_pos'].tolist(), index=ref_data_df.index)
    ref_data_df.drop('screen_pos', inplace=True, axis=1)
    ref_data_df.sort_values(by='pupilTimestamp',inplace=True)
    
    prevX = None
    prevY = None
    prevZ = None
    trialIdxs = []
    currTrialIdx = 0
    for index, row in ref_data_df.iterrows():
        if prevX != row['screen-pos_x'] or prevY != row['screen-pos_y'] or prevZ != row['screen-pos_z']:
            currTrialIdx += 1
        trialIdxs.append(currTrialIdx)
        prevX = row['screen-pos_x']
        prevY = row['screen-pos_y']
        prevZ = row['screen-pos_z']
    
    ref_data_df['trialNumber'] = trialIdxs

    #dataFileName = '/'.join(trialResults['time_sync_pupilTimeStamp_location_0'].split('/')[-2:])
    #pupilTimestampData = pd.read_csv( dataParentFolder + dataFileName)
    #pupilTimestampData = pupilTimestampData.rename(columns={"time": "frameTime","timeStamp":"pupilTimestamp"})
    
    ## Import gaze direction data
    gazeDataFolderList = []
    [gazeDataFolderList.append(name) for name in os.listdir(dataParentFolder + 'PupilData') if name[0] != '.'] # is not
    
    pupilSessionFolder = '/' + gazeDataFolderList[0]   
    gazeDataFolder = dataParentFolder + 'PupilData' + pupilSessionFolder
    try:
        pupilExportsFolder = []
        [pupilExportsFolder.append(name) for name in os.listdir(gazeDataFolder + '/Exports') if name[0] != '.']
        if specificExport is None:
            # Defaults to the most recent pupil export folder (highest number)
            gazePositionsDF = pd.read_csv( gazeDataFolder + '/Exports/' + pupilExportsFolder[-1] + '/gaze_positions.csv' )
        elif specificExport not in pupilExportsFolder:
            logger.exception('Export ' + specificExport + ' not found.')
            return None
        else:
            gazePositionsDF = pd.read_csv( gazeDataFolder + '/Exports/' + specificExport + '/gaze_positions.csv' )
    except:
        logger.exception('No gaze_positions.csv.  Process and export data in Pupil Player.')
    
    gazePositionsDF = gazePositionsDF.rename(columns={"gaze_timestamp": "pupilTimestamp"})

    # Sort, because the left/right eye data is written asynchronously, and this means timestamps may not be monotonically increasing.
    gazePositionsDF.sort_values(by='pupilTimestamp',inplace=True)

    rawTrialData = ref_data_df

    firstTS = rawTrialData['pupilTimestamp'].min()
    lastTS = rawTrialData['pupilTimestamp'].max()
    
    firstIdx = list(map(lambda i: i>= float(firstTS), gazePositionsDF['pupilTimestamp'])).index(True)
    lastIdx = list(map(lambda i: i>= float(lastTS), gazePositionsDF['pupilTimestamp'])).index(True)
    rawGazeData = gazePositionsDF.loc[firstIdx:lastIdx]
    
    interpDF = pd.merge_asof( rawTrialData, rawGazeData.reset_index(), on = 'pupilTimestamp', direction='nearest')
    
    newColList = [convertIndexToMultiIndexUsingUnderscore(c) for c in interpDF.columns[:len(rawTrialData.columns)]]
    newGazeColList = [ convertIndexToMultiIndexIgnoringUnderscore(c) for c in interpDF.columns[(len(rawTrialData.columns)):] ]
    newColList.extend(newGazeColList)
    interpDF.columns = pd.MultiIndex.from_tuples(newColList)
    
    # Probably would want to add the target location here
    #   First target would be 0, 0, 1 iirc
    
    dictOut = {"rawUnityData": rawTrialData, "rawGazeData": rawGazeData, "interpolatedData": interpDF}
    
    return dictOut
    

rgd_lens = []

def processTrial(dataFolder, trialResults, numTrials = False, specificExport=None):
    '''
    This function loads in the raw gaze data for a trial and the raw unity data for a trial.
    The Unity data is upsampled to match the higher gaze sample data for the trial.
    THe return is a dictionary with:
    - Raw gaze data (at the original, high sampling rate)
    - Raw per-frame Unity data.
    - The interpolated, or "processed" data.
    '''
    exportStr = specificExport
    if exportStr is None:
        exportStr = ''
    if(numTrials):
        logger.info('Processing subject: ' + trialResults['ppid'] + '; e: ' + exportStr +'; t = ' + str(trialResults['trial_num']) + ' of ' + str(numTrials) )
    else:
        logger.info('Processing subject: ' + trialResults['ppid'] + '; e: ' + exportStr +'; t = ' + str(trialResults['trial_num']))

    dataParentFolder = '/'.join(dataFolder.split('/')[:-2]) + '/';

    ## Import view data and rename some columns
    dataFileName = '/'.join(trialResults['camera_movement_location_0'].split('/')[-2:])
    viewData = pd.read_csv( dataParentFolder + dataFileName)
    viewData = viewData.rename(columns={"time": "frameTime"})

    ## Import pupil timestamp data (recorded within Unity)
    dataFileName = '/'.join(trialResults['time_sync_pupilTimeStamp_location_0'].split('/')[-2:])
    pupilTimestampData = pd.read_csv( dataParentFolder + dataFileName)
    pupilTimestampData = pupilTimestampData.rename(columns={"time": "frameTime","timeStamp":"pupilTimestamp"})

    ## Import gaze direction data
    gazeDataFolderList = []
    [gazeDataFolderList.append(name) for name in os.listdir(dataParentFolder + 'PupilData') if name[0] != '.'] # is not
    
    pupilSessionFolder = '/' + gazeDataFolderList[0]   
    gazeDataFolder = dataParentFolder + 'PupilData' + pupilSessionFolder

    try:
        pupilExportsFolder = []
        [pupilExportsFolder.append(name) for name in os.listdir(gazeDataFolder + '/Exports') if name[0] != '.']
        
        if specificExport is None:
            # Defaults to the most recent pupil export folder (highest number)
            gazePositionsDF = pd.read_csv( gazeDataFolder + '/Exports/' + pupilExportsFolder[-1] + '/gaze_positions.csv' )
        elif specificExport not in pupilExportsFolder:
            logger.exception('Export ' + specificExport + ' not found.')
            return None
        else:
            gazePositionsDF = pd.read_csv( gazeDataFolder + '/Exports/' + specificExport + '/gaze_positions.csv' )

    except:
        logger.exception('No gaze_positions.csv.  Process and export data in Pupil Player.')

    #print(gazePositionsDF.columns)
    #exit()
    gazePositionsDF = gazePositionsDF.rename(columns={"gaze_timestamp": "pupilTimestamp"})

    # Sort, because the left/right eye data is written asynchronously, and this means timestamps may not be monotonically increasing.
    gazePositionsDF.sort_values(by='pupilTimestamp',inplace=True)

    ##################################################################################################
    ##################################################################################################
    ## Create rawTrialData and merge with view data

    if len(pupilTimestampData) == 0:
        logger.exception('No pupil timestamp data.')

    rawTrialData = pupilTimestampData
    rawTrialData['trialNumber'] = trialResults['trial_num']
    rawTrialData['blockNumber'] = trialResults['block_num']
    rawTrialData = pd.merge(rawTrialData, viewData, on ='frameTime',validate= 'one_to_many')

    ballData = []
    paddleData = []

    if(trialResults['trialType'] == 'interception'):
            
        ## Import ball data and rename some columns
        dataFileName = '/'.join(trialResults['ball_movement_location_0'].split('/')[-2:])

        ballData = pd.read_csv( dataParentFolder + dataFileName)
        ballData = ballData.rename(columns={"time": "frameTime"})
        ballData.rename(columns={"pos_x": "ballPos_x", "pos_y": "ballPos_y","pos_z": "ballPos_z"},inplace=True)
        ballData.rename(columns={"rot_x": "ballRot_x", "rot_y": "ballRot_y","rot_z": "ballRot_z"},inplace=True)

        ## Import paddle data and rename some columns
        dataFileName = '/'.join(trialResults['paddle_movement_location_0'].split('/')[-2:])
        paddleData = pd.read_csv( dataParentFolder + dataFileName)
        paddleData = paddleData.rename(columns={"time": "frameTime"})

        ## Merge view and ball data into rawTrialData
        rawTrialData = pd.merge(rawTrialData, ballData, on ='frameTime',validate= 'one_to_many')
        rawTrialData = pd.merge(rawTrialData, paddleData, on ='frameTime',validate= 'one_to_many')

    if(trialResults['trialType'] == 'CalibrationAssessment'):

        ## Import ball data and rename some columns
        dataFileName = '/'.join(trialResults['etassessment_calibrationAssessment_location_0'].split('/')[-2:])
        assessmentData = pd.read_csv(dataParentFolder + dataFileName)
        
        newKeys = ['frameTime']
        [newKeys.append(key[13:]) for key in assessmentData.keys()[1:]] # Fix a silly mistake I made when naming columns
        assessmentData.columns = newKeys

        rawTrialData = pd.merge(rawTrialData, assessmentData, on ='frameTime',validate= 'one_to_many')

    ################################################################################
    ## Pupil gaze direction data is for the whole experiment while time stamps are recorded only during an ongoing trial.
    ## Find the slices of gaze dir data that map onto the trial timestamps, and concatenate them.

    ## Group trial data by blocks and trials
    gbBlTr = rawTrialData.groupby(['blockNumber','trialNumber'])
    tr = gbBlTr.get_group( list(gbBlTr.groups.keys())[0])

    firstTS = tr.head(1)['pupilTimestamp']
    lastTS = tr.tail(1)['pupilTimestamp']
    
    firstIdx = list(map(lambda i: i> float(firstTS), gazePositionsDF['pupilTimestamp'])).index(True)
    lastIdx = list(map(lambda i: i> float(lastTS), gazePositionsDF['pupilTimestamp'])).index(True)
    rawGazeData = gazePositionsDF.loc[firstIdx:lastIdx]
    
    if len(rgd_lens) < trialResults['trial_num']:
        rgd_lens.append([])
    rgd_lens[trialResults['trial_num']-1].append((f"({float(firstTS)} - {float(lastTS)})", len(rawGazeData)))
    #print(rgd_lens[trialResults['trial_num']-1])
    #print(len(rawGazeData))
    #exit()
    # # Drop data below the confidence level
    # filteredGazeData = rawGazeData.reset_index().drop(np.where(rawGazeData['confidence'] < gazeConfidenceThreshold )[0])
    
    # Merge gaze and trial data
    interpDF = pd.merge( rawTrialData, rawGazeData.reset_index(), on ='pupilTimestamp',how='outer',sort=True)

    # Upsample trial data to the resolution of gaze data
    # logger.warning('*** UPSAMPLING NON-GAZE DATA TO MATCH EYE TRACKER SAMPLING FREQUENCY ***')
    interpDF = interpDF.interpolate(method='linear',downcast='infer')

    if(trialResults['trialType'] == 'CalibrationAssessment'):
        # There are values that should not be interpolated linearly.  
        # For example, the unity frame should be held constant as multiple pupil labs samples come in

        interpDF['frameTime'] = interpDF['frameTime'].fillna(method='ffill')
        interpDF['isHeadFixed'] = interpDF['isHeadFixed'].fillna(method='ffill')
        interpDF['currentTargetName'] = interpDF['currentTargetName'].fillna(method='ffill')

    # Drop matrixes from processed dataframe until I can figure out how to properly interpolate 
    interpDF.drop(labels=interpDF.columns[(interpDF.columns.get_level_values(0).str.contains('4x4'))],axis=1)

    # Convert to multiindex
    newColList = [convertIndexToMultiIndexUsingUnderscore(c) for c in interpDF.columns[:len(rawTrialData.columns)]]
    newGazeColList = [ convertIndexToMultiIndexIgnoringUnderscore(c) for c in interpDF.columns[(len(rawTrialData.columns)):] ]
    newColList.extend(newGazeColList)
    interpDF.columns = pd.MultiIndex.from_tuples(newColList)

    # Some sanity checks 
    if( len(interpDF) > ( len(rawTrialData) + len(rawGazeData)) ):
        logger.warning('len(interpDF) > ( len(rawTrialData) + len(rawGazeData))')

    dictOut = {"rawUnityData": rawTrialData, "rawGazeData": rawGazeData, "interpolatedData": interpDF}

    return dictOut

################################################################################
################################################################################

def unpackSession(subNum, load_realtime_ref_data, doNotLoad = False, specificExport=None):
    '''
    Exports a dictionary with the following keys:

    * subID: self explanatory
    * trialInfo: metadata for the trial
    * expConfig: metadata for the experiment

    * rawExpUnity: raw data recorded at each Unity call ot Update() - 90 Hz on the Vive.  Data is for catching experiment trials only.

    * rawExpGaze: raw data recorded at each sample of a Pupil eye camera - [two interleaved 120 hz streams, so approx 240 hz] Data is for catching experiment trials only.

    * processedExp: Formed by upsampling rawExpUnity to match the frequency of rawExpGaze, and merging. Data is for catching experiment trials only.

    * rawCalibUnity: Same as rawExpUnity but for calibraiton assessment trials only.
    * rawCalibGaze: Same as rawExpGaze but for calibraiton assessment trials only.
    * processedCalib: Same as processedExp but for calibraiton assessment trials only.


    '''

    # Get folder/filenames
    dataFolderList = []
    [dataFolderList.append(name) for name in os.listdir("Data/") if ((name[0] is not '.' and (name[0:9] == "_Pipeline")))]

    for i, name in enumerate(dataFolderList):
            if i == subNum:
                print('***> ' + str(i) + ': ' + name )
            else:
                print(str(i) + ': ' + name )
    

    dataParentFolder = "Data/" + dataFolderList[subNum]
    from os import path
    if path.isdir(dataParentFolder+"/S001"):
        dataParentFolder += "/S001"
    dataSubFolderList = []
    [dataSubFolderList.append(name) for name in os.listdir(dataParentFolder) if name[0] != '.']
    dataFolder = dataParentFolder + '/' + dataSubFolderList[0] + '/'

    logger.info('Processing session: ' + dataParentFolder)

    # Try to load pickle if doNotLoad == False
    picklePath = dataFolder + dataSubFolderList[0] + '.pickle'
    if( doNotLoad == False and path.exists(picklePath)):
        
        file = open(picklePath, 'rb')
        sessionData = pickle.load(file)
        file.close()
        
        logger.info('Importing session dict from pickle.')
        
        return sessionData

    logger.info('Compiling session dict from *.csv.')

    # If not loading from pickle, create and populate dataframes
    rawSequenceUnityDataDf = pd.DataFrame()
    rawSequenceGazeDataDf = pd.DataFrame()
    processedSequenceDataDf = pd.DataFrame()
    
    rawExpUnityDataDf = pd.DataFrame()
    rawExpGazeDataDf = pd.DataFrame()
    processedExpDataDf = pd.DataFrame()

    rawCalibUnityDataDf = pd.DataFrame()
    rawCalibGazeDataDf = pd.DataFrame()
    processedCalibDataDf = pd.DataFrame()

    trialData = pd.read_csv( dataParentFolder + '/trial_results.csv')

    def addToDF(targetDF,dfIn):

        return pd.concat([targetDF, dfIn])

        # if( targetDF.empty ):
        #     return dfIn
        # else:
        #     # targetDF = targetDF.append(dfIn)
        #     return pd.concat([targetDF, dfIn])
    
    # The Calibration Sequence
    calibSequenceDict = processCalibSequence(dataFolder, load_realtime_ref_data, specificExport=specificExport)
    
    processedSequenceDataDf = addToDF(processedSequenceDataDf,calibSequenceDict['interpolatedData'])
    rawSequenceUnityDataDf = addToDF(rawSequenceUnityDataDf,calibSequenceDict['rawUnityData'])
    rawSequenceGazeDataDf = addToDF(rawSequenceGazeDataDf,calibSequenceDict['rawGazeData'])
    
    processedSequenceDataDf = processedSequenceDataDf.reset_index(drop=True)

    # The Trials
    for trIdx, trialResults in trialData.iterrows():
        #print(trialResults)
        #exit()
        trialDict = processTrial(dataFolder, trialResults,len(trialData), specificExport=specificExport)
        if trialDict is None:
            continue  # The specified export did not exist, keep going

        if (trialResults['trialType'] == 'interception'):

            processedExpDataDf = addToDF(processedExpDataDf,trialDict['interpolatedData'])
            rawExpUnityDataDf = addToDF(rawExpUnityDataDf,trialDict['rawUnityData'])
            rawExpGazeDataDf = addToDF(rawExpGazeDataDf,trialDict['rawGazeData'])

        elif(trialResults['trialType'] == 'CalibrationAssessment'):

            processedCalibDataDf = addToDF(processedCalibDataDf,trialDict['interpolatedData'])
            rawCalibUnityDataDf = addToDF(rawCalibUnityDataDf,trialDict['rawUnityData'])
            rawCalibGazeDataDf = addToDF(rawCalibGazeDataDf,trialDict['rawGazeData'])

    # start_time 169.666
    # end_time 170.676

    # Rename trialdata columns
    trialData.rename(columns={"session_num":"sessionNumber","trial_num":"trialNumber",
        "block_num":"blockNumber","trial_num_in_block":"trialNumberInBlock",
        "start_time":"startTime","end_time":"endTime"},inplace=True)

    trDataFiles = [i for i in trialData.columns.to_list() if '_filename' in i] 

    # Convert to multiindex
    newColList = [convertIndexToMultiIndexUsingUnderscore(c) for c in trialData.columns[:-len(trDataFiles)]]
    newColList.extend([convertIndexToMultiIndexIgnoringUnderscore(c) for c in trialData.columns[-len(trDataFiles):]])
    trialData.columns = pd.MultiIndex.from_tuples(newColList)
    
    expDict = json.load( open(dataParentFolder + '/settings/' + 'settings.json'))

    processedExpDataDf = processedExpDataDf.reset_index(drop=True)
    processedCalibDataDf = processedCalibDataDf.reset_index(drop=True)
    
    analysisParameters = json.load( open('analysisParameters.json'))
    # analysisParameters['gazeDataConfidenceThreshold'] = gazeDataConfidenceThreshold

    
    #logger.warning('(**********************  SUBID FILE IS HARDCODED *******************************')
    #subID = json.load( open(dataParentFolder + '/participantdetails/participant_details.csv'))['ppid']
    subID = trialData['ppid'][1] 

    dictOut = {"subID": subID, "plExportFolder": specificExport, "trialInfo": trialData.sort_index(axis=1),"expConfig": expDict,
        "rawExpUnity": rawExpUnityDataDf.sort_index(axis=1), "rawExpGaze": rawExpGazeDataDf.sort_index(axis=1), "processedExp": processedExpDataDf.sort_index(axis=1),
        "rawCalibUnity": rawCalibUnityDataDf.sort_index(axis=1), "rawCalibGaze": rawCalibGazeDataDf.sort_index(axis=1), "processedCalib": processedCalibDataDf.sort_index(axis=1),
        "rawSequenceUnity": rawSequenceUnityDataDf, "rawSequenceGaze": rawSequenceGazeDataDf, "processedSequence": processedSequenceDataDf.sort_index(axis=1),
        "analysisParameters":analysisParameters}

    with open(picklePath, 'wb') as handle:
        pickle.dump(dictOut, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return dictOut



if __name__ == "__main__":

    import os

    subNum = 0

    rawDataDF = False
    calibDF = False

    # ### Load dataframes
    # Remember to set loadParsedData, loadProcessedData.

    dataFolderList = []
    [dataFolderList.append(name) for name in os.listdir("Data/") if ((name[0] is not '.' and (name[0:9] == "_Pipeline")))]

    for i, name in enumerate(dataFolderList):
        print(str(i) + ': ' + name )

    sessionDict = unpackSession(subNum, doNotLoad = True)


# %%
