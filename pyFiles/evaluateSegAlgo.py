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

import matplotlib.pyplot as plt
import matplotlib.cm as cm


fmt = '%(levelname)s_%(name)s-%(funcName)s(): - %(message)s'
logging.basicConfig(level=logging.INFO, format=fmt)
logger = logging.getLogger(__name__)

# In[1]:

wd = os.getcwd()
os.chdir(wd)

def calcAverageGazeDirPerCalibTrial(sessionDictIn):
    gbProcessedCalib_trial = sessionDictIn['processedSequence'].groupby(['trialNumber'])
    
    mean0ErrorCalibAz = gbProcessedCalib_trial.agg([np.nanmean])[('gaze0Spherical','az','nanmean')]
    mean0ErrorCalibEl = gbProcessedCalib_trial.agg([np.nanmean])[('gaze0Spherical','az','nanmean')]
    mean0ErrorAz = mean0ErrorCalibAz
    mean0ErrorEl = mean0ErrorCalibEl
    sessionDictIn['calibTrialInfo'][('meanGaze0_Spherical','az')] = mean0ErrorAz
    sessionDictIn['calibTrialInfo'][('meanGaze0_Spherical','el')] = mean0ErrorEl
    
    #sessionDictIn['calibTrialInfo'][('meanGaze0_Spherical','az')] = gbProcessedCalib_trial.agg([np.nanmean])[('gaze0Spherical','az','nanmean')].values
    #sessionDictIn['calibTrialInfo'][('meanGaze0_Spherical','el')] = gbProcessedCalib_trial.agg([np.nanmean])[('gaze0Spherical','el','nanmean')].values

    mean1ErrorCalibAz = gbProcessedCalib_trial.agg([np.nanmean])[('gaze1Spherical','az','nanmean')]
    mean1ErrorCalibEl = gbProcessedCalib_trial.agg([np.nanmean])[('gaze1Spherical','az','nanmean')]
    mean1ErrorAz = mean1ErrorCalibAz
    mean1ErrorEl = mean1ErrorCalibEl
    sessionDictIn['calibTrialInfo'][('meanGaze1_Spherical','az')] = mean1ErrorAz
    sessionDictIn['calibTrialInfo'][('meanGaze1_Spherical','el')] = mean1ErrorEl

    #sessionDictIn['calibTrialInfo'][('meanGaze1_Spherical','az')] = gbProcessedCalib_trial.agg([np.nanmean])[('gaze1Spherical','az','nanmean')].values
    #sessionDictIn['calibTrialInfo'][('meanGaze1_Spherical','el')] = gbProcessedCalib_trial.agg([np.nanmean])[('gaze1Spherical','el','nanmean')].values
    
    mean2ErrorCalibAz = gbProcessedCalib_trial.agg([np.nanmean])[('gaze2Spherical','az','nanmean')]
    mean2ErrorCalibEl = gbProcessedCalib_trial.agg([np.nanmean])[('gaze2Spherical','az','nanmean')]
    mean2ErrorAz = mean2ErrorCalibAz
    mean2ErrorEl = mean2ErrorCalibEl
    sessionDictIn['calibTrialInfo'][('meanGaze2_Spherical','az')] = mean2ErrorAz
    sessionDictIn['calibTrialInfo'][('meanGaze2_Spherical','el')] = mean2ErrorEl
    
    #sessionDictIn['calibTrialInfo'][('meanGaze2_Spherical','az')] = gbProcessedCalib_trial.agg([np.nanmean])[('gaze2Spherical','az','nanmean')].values
    #sessionDictIn['calibTrialInfo'][('meanGaze2_Spherical','el')] = gbProcessedCalib_trial.agg([np.nanmean])[('gaze2Spherical','el','nanmean')].values
    
    return sessionDictIn
    

def calcAverageGazeDirPerTrial(sessionDictIn, ball_catching=True):
    gbProcessedCalib_trial = sessionDictIn['processedCalib'].groupby(['trialNumber'])
    if ball_catching:
        gbProcessedExp_trial = sessionDictIn['processedExp'].groupby(['trialNumber'])
    
    mean0ErrorCalibAz = gbProcessedCalib_trial.agg([np.nanmean])[('gaze0Spherical','az','nanmean')]
    if ball_catching:
        mean0ErrorExpAz = gbProcessedExp_trial.agg([np.nanmean])[('gaze0Spherical','az','nanmean')]
    mean0ErrorCalibEl = gbProcessedCalib_trial.agg([np.nanmean])[('gaze0Spherical','az','nanmean')]
    if ball_catching:
        mean0ErrorExpEl = gbProcessedExp_trial.agg([np.nanmean])[('gaze0Spherical','az','nanmean')]
        mean0ErrorAz = pd.concat([mean0ErrorCalibAz, mean0ErrorExpAz])
        mean0ErrorEl = pd.concat([mean0ErrorCalibEl, mean0ErrorExpEl])
    else:
        mean0ErrorAz = mean0ErrorCalibAz
        mean0ErrorEl = mean0ErrorCalibEl
    sessionDictIn['trialInfo'][('meanGaze0_Spherical','az')] = mean0ErrorAz
    sessionDictIn['trialInfo'][('meanGaze0_Spherical','el')] = mean0ErrorEl
    
    #sessionDictIn['trialInfo'][('meanGaze0_Spherical','az')] = gbProcessedCalib_trial.agg([np.nanmean])[('gaze0Spherical','az','nanmean')].values
    #sessionDictIn['trialInfo'][('meanGaze0_Spherical','el')] = gbProcessedCalib_trial.agg([np.nanmean])[('gaze0Spherical','el','nanmean')].values

    mean1ErrorCalibAz = gbProcessedCalib_trial.agg([np.nanmean])[('gaze1Spherical','az','nanmean')]
    if ball_catching:
        mean1ErrorExpAz = gbProcessedExp_trial.agg([np.nanmean])[('gaze1Spherical','az','nanmean')]
    mean1ErrorCalibEl = gbProcessedCalib_trial.agg([np.nanmean])[('gaze1Spherical','az','nanmean')]
    if ball_catching:
        mean1ErrorExpEl = gbProcessedExp_trial.agg([np.nanmean])[('gaze1Spherical','az','nanmean')]
        mean1ErrorAz = pd.concat([mean1ErrorCalibAz, mean1ErrorExpAz])
        mean1ErrorEl = pd.concat([mean1ErrorCalibEl, mean1ErrorExpEl])
    else:
        mean1ErrorAz = mean1ErrorCalibAz
        mean1ErrorEl = mean1ErrorCalibEl
    sessionDictIn['trialInfo'][('meanGaze1_Spherical','az')] = mean1ErrorAz
    sessionDictIn['trialInfo'][('meanGaze1_Spherical','el')] = mean1ErrorEl

    #sessionDictIn['trialInfo'][('meanGaze1_Spherical','az')] = gbProcessedCalib_trial.agg([np.nanmean])[('gaze1Spherical','az','nanmean')].values
    #sessionDictIn['trialInfo'][('meanGaze1_Spherical','el')] = gbProcessedCalib_trial.agg([np.nanmean])[('gaze1Spherical','el','nanmean')].values
    
    mean2ErrorCalibAz = gbProcessedCalib_trial.agg([np.nanmean])[('gaze2Spherical','az','nanmean')]
    if ball_catching:
        mean2ErrorExpAz = gbProcessedExp_trial.agg([np.nanmean])[('gaze2Spherical','az','nanmean')]
    mean2ErrorCalibEl = gbProcessedCalib_trial.agg([np.nanmean])[('gaze2Spherical','az','nanmean')]
    if ball_catching:
        mean2ErrorExpEl = gbProcessedExp_trial.agg([np.nanmean])[('gaze2Spherical','az','nanmean')]
        mean2ErrorAz = pd.concat([mean2ErrorCalibAz, mean2ErrorExpAz])
        mean2ErrorEl = pd.concat([mean2ErrorCalibEl, mean2ErrorExpEl])
    else:
        mean2ErrorAz = mean2ErrorCalibAz
        mean2ErrorEl = mean2ErrorCalibEl
    sessionDictIn['trialInfo'][('meanGaze2_Spherical','az')] = mean2ErrorAz
    sessionDictIn['trialInfo'][('meanGaze2_Spherical','el')] = mean2ErrorEl
    
    #sessionDictIn['trialInfo'][('meanGaze2_Spherical','az')] = gbProcessedCalib_trial.agg([np.nanmean])[('gaze2Spherical','az','nanmean')].values
    #sessionDictIn['trialInfo'][('meanGaze2_Spherical','el')] = gbProcessedCalib_trial.agg([np.nanmean])[('gaze2Spherical','el','nanmean')].values
    
    return sessionDictIn

def calcCyclopean(sessionDictIn, sessionDictKey='processedExp'):
    xyz = (sessionDictIn[sessionDictKey]['gaze-normal0'] + sessionDictIn[sessionDictKey]['gaze-normal1']) / 2.0

    sessionDictIn[sessionDictKey][('gaze-normal2','x')] = xyz['x']
    sessionDictIn[sessionDictKey][('gaze-normal2','y')] = xyz['y']
    sessionDictIn[sessionDictKey][('gaze-normal2','z')] = xyz['z']

    # sessionDict[sessionDictKey]['gaze-normal2'] = 
    sessionDictIn[sessionDictKey]['gaze-normal2'].apply(lambda row: normalizeVector(row),axis=1)
    return sessionDictIn

def normalizeVector(xyz):
    '''
    Input be a 3 element array of containing the x,y,and z data of a 3D vector.
    Returns a normalized 3 element array
    '''

    xyz = xyz / np.linalg.norm(xyz)
    return xyz


def calcSphericalCoordinates(sessionDictIn,columnName,newColumnName,sessionDictKey='processedExp', flipY = False, override_to_2d = None):

    dataDict = sessionDictIn[sessionDictKey]
    
    if override_to_2d is not None and ('deprojected-norm-pos'+override_to_2d,'x') in dataDict.columns and ('deprojected-norm-pos'+override_to_2d,'y') in dataDict.columns and ('deprojected-norm-pos'+override_to_2d,'z') in dataDict.columns:
        try:
            dataDict[(newColumnName,'az')] = np.rad2deg(np.arctan2(dataDict[('deprojected-norm-pos'+override_to_2d,'x')],dataDict[('deprojected-norm-pos'+override_to_2d,'z')]))
            dataDict[(newColumnName,'el')] = np.rad2deg(np.arctan2(dataDict[('deprojected-norm-pos'+override_to_2d,'y')],dataDict[('deprojected-norm-pos'+override_to_2d,'z')]))
            print(f"COLUMN NAME: {columnName}")
            print(f"NEW COLUMN NAME: {newColumnName}")
            print(f"SESSION DICT KEY: {sessionDictKey}")
            print(f"DATADICT INDEXER: {('deprojected-norm-pos'+override_to_2d,'y')}")
            #if columnName == 'gaze-normal0' and sessionDictKey == 'processedCalib':
            #    for i in range(100000):
            #        print(f"LEN OF DEPROJECTED NORMPOS X: {len(dataDict[('deprojected-norm-pos'+override_to_2d,'x')])}")
        except Exception as e:
            print(e)
            print(dataDict.keys())
            exit()
    else:
        try:
            dataDict[(newColumnName,'az')] = np.rad2deg(np.arctan2(dataDict[(columnName,'x')],dataDict[(columnName,'z')]))
            dataDict[(newColumnName,'el')] = np.rad2deg(np.arctan2(dataDict[(columnName,'y')],dataDict[(columnName,'z')]))
        except Exception as e:
            print(e)
            print(dataDict.keys())
            exit()

    if flipY:
        dataDict[(newColumnName,'el')] = -    dataDict[(newColumnName,'el')]

    sessionDictIn[sessionDictKey] = dataDict
    
    logger.info('Added sessionDict[\'{0}\'][\'{1}\',\'az\']'.format(sessionDictKey, newColumnName))
    logger.info('Added sessionDict[\'{0}\'][\'{1}\',\'el\']'.format(sessionDictKey,newColumnName))
    
    return sessionDictIn

def calcTrialLevelCalibInfo(sessionIn, ball_catching=True):
    '''
    Input: Session dictionary
    Output:  Session dictionary with new column sessionDict['trialInfo']['targetType']
    '''
    
    gbProcessedCalib_trial = sessionIn['processedCalib'].groupby(['trialNumber'])
    if ball_catching:
        gbProcessedExp_trial = sessionIn['processedExp'].groupby(['trialNumber'])
    
    targetTypes = []
    gridSize_height = []
    gridSize_width = []
    targeLocalPos_az = []
    targeLocalPos_el = []

    targeLocalPosList_x = []
    targeLocalPosList_y = []
    targeLocalPosList_z = []

    print(gbProcessedCalib_trial.groups.keys())
    for trialRowIdx, trMetaData in sessionIn['trialInfo'].iterrows():
        print("Trial Number",int(trMetaData['trialNumber']))
        try:
            # This dataframe contains the per-frame processed data associated with this trial
            procDF = gbProcessedCalib_trial.get_group(int(trMetaData['trialNumber']))
        except KeyError as e:
            # This trial is in processedExp, not processedCalib.
            print("This is processedExp")
            if ball_catching:
                procDF = gbProcessedExp_trial.get_group(int(trMetaData['trialNumber']))
            else:
                continue

        targetType = []
        height = []
        width = []
        
        try:
            if ( sum(procDF['isHeadFixed'] == False) ==  len(procDF) ):

                targetType = 'VOR'
                height = np.nan
                width = np.nan

            elif( sum(procDF['isHeadFixed'] == True) == len(procDF) ):

                #print(procDF['radiusDegrees'])  # Target ring radius
                #print(procDF['targetRadiusInDegrees'])  # Radius of actual target dot
                
                try:
                    # HxW Grid
                    height = np.float(procDF['elevationHeight'].drop_duplicates())
                    width = np.float(procDF['azimuthWidth'].drop_duplicates())
                except KeyError as _:
                    # R-radius ring
                    height = np.float(procDF['radiusDegrees'].drop_duplicates())
                    width = np.float(procDF['radiusDegrees'].drop_duplicates())
                    
                # Count the number of target positions within the local space (head-centered space)
                if( len(procDF['targeLocalPos'].drop_duplicates()) == 1 ):
                    # Only one target, so it's a fixation trial
                    targetType = 'fixation'
                    
                    rounded = procDF[('targetLocalSpherical','az')]
                    rounded = np.float(rounded.drop_duplicates())
                    rounded = np.round(rounded, 1)
                    target_az = rounded
                    target_el = np.round(np.float(procDF[('targetLocalSpherical','el')].drop_duplicates()),1)

                    targeLocalPos_x = procDF['targeLocalPos','x'].drop_duplicates()
                    targeLocalPos_y = procDF['targeLocalPos','y'].drop_duplicates()
                    targeLocalPos_z = procDF['targeLocalPos','z'].drop_duplicates()

                    if( len(targeLocalPos_x)>1 or len(targeLocalPos_y)>1 or len(targeLocalPos_z)>1 ):

                        logger.warning('targeLocalPos len > 1 for a trial.  Why?')

                    else:
                        targeLocalPos_x = float(targeLocalPos_x.values[0])
                        targeLocalPos_y = float(targeLocalPos_y.values[0])
                        targeLocalPos_z = float(targeLocalPos_z.values[0])

                else:
                    # multiple targets, so it's a saccade trial
                    targetType = 'fixation+saccade'
                    
                    target_az = np.nan
                    target_el = np.nan

                    targeLocalPos_x = np.nan
                    targeLocalPos_y = np.nan
                    targeLocalPos_z = np.nan
                    

            else:
                # The trial has both head fixed and world fixed targets.  
                # We did not plan for that, so let's label it as "unknown."

                targetType = 'unknown'
                height = np.nan
                width = np.nan
                target_az = np.nan
                target_el = np.nan
        except KeyError as e:
            print('KeyError - trial setting to default')
            print(e)
            targetType = 'unknown'
            height = np.nan
            width = np.nan
            target_az = np.nan
            target_el = np.nan
            
#         print('Trial number: {tNum}, type: {tType}'.format(tNum = int(trMetaData['trialNumber']),
#                                                        tType = targetType))
        targetTypes.append(targetType)
        gridSize_height.append(height)
        gridSize_width.append(width)
        targeLocalPos_az.append(target_az)
        targeLocalPos_el.append(target_el)

        targeLocalPosList_x.append(targeLocalPos_x)
        targeLocalPosList_y.append(targeLocalPos_y)
        targeLocalPosList_z.append(targeLocalPos_z)

#         print(zip((target_az,target_el)))
        
    sessionIn['trialInfo']['targetType'] = targetTypes
    sessionIn['trialInfo']['gridSize','heightDegs'] = gridSize_height
    sessionIn['trialInfo']['gridSize','widthDegs'] = gridSize_width
    sessionIn['trialInfo']['fixTargetSpherical','az'] = targeLocalPos_az
    sessionIn['trialInfo']['fixTargetSpherical','el'] = targeLocalPos_el

    sessionIn['trialInfo']['fixTargetLocal', 'x'] = targeLocalPosList_x
    sessionIn['trialInfo']['fixTargetLocal', 'y'] = targeLocalPosList_y
    sessionIn['trialInfo']['fixTargetLocal', 'z'] = targeLocalPosList_z
    
    sessionIn['trialInfo']['fixTargetSpherical','az'] = sessionIn['trialInfo']['fixTargetSpherical','az'].round(2)
    sessionIn['trialInfo']['fixTargetSpherical','el'] = sessionIn['trialInfo']['fixTargetSpherical','el'].round(2)

    logger.info('Added sessionDict[\'trialInfo\'][\'targetType\']')
    logger.info('Added sessionDict[\'trialInfo\'][\'gridSize\']')
    logger.info('Added sessionDict[\'trialInfo\'][\'fixTargetSpherical\']')
    
    return sessionIn

def calcGazeToCalibFixError(sessionDictIn,gazeLabelIn,targetLabelIn,columnOutLabel, sessionDictKey='processedSequence'):
    # For each frame of the dataframe, calculate the distance between gaze and the target
    sessionDictIn[sessionDictKey][(columnOutLabel,'az')] = sessionDictIn[sessionDictKey].apply(lambda row: row[(gazeLabelIn,'az')] -  row[(targetLabelIn,'az')],axis=1 )
    sessionDictIn[sessionDictKey][(columnOutLabel,'el')] = sessionDictIn[sessionDictKey].apply(lambda row: row[(gazeLabelIn,'el')] -  row[(targetLabelIn,'el')],axis=1 )
    sessionDictIn[sessionDictKey][(columnOutLabel,'euclidean')] = np.sqrt(sessionDictIn[sessionDictKey][(columnOutLabel,'az')]**2 + sessionDictIn[sessionDictKey][(columnOutLabel,'el')]**2)
        
    # Group by trial, so that we can average within trial
    gbProcessedCalib_trial = sessionDictIn[sessionDictKey].groupby(['trialNumber'])
    
    meanErrorCalib = gbProcessedCalib_trial.agg(np.nanmean)[columnOutLabel]
    meanOutLabel = 'mean' + columnOutLabel[0].capitalize() + columnOutLabel[1:]

    # Add it back into the trialInfo dataframe
    # Notice that we are VERY careful of the index we are using to align data.
    # Row indices start at 0, but trial numbers start at 1!
    
    meanError = meanErrorCalib.copy(deep=True)

    meanErrorAz = meanError['az']
    meanErrorEl = meanError['el']
    
    sessionDictIn['calibTrialInfo'] = pd.DataFrame()
    
    sessionDictIn['calibTrialInfo'][(meanOutLabel,'az')] = meanErrorAz.values
    sessionDictIn['calibTrialInfo'][(meanOutLabel,'el')] = meanErrorEl.values
    sessionDictIn['calibTrialInfo'][(meanOutLabel,'euclidean')] = np.sqrt( meanErrorAz.values**2 + meanErrorEl.values**2)
    
    stdErrorCalib = gbProcessedCalib_trial.agg(np.nanstd)[columnOutLabel] 
    stdOutLabel = 'std' + columnOutLabel[0].capitalize() + columnOutLabel[1:]
    
    stdErrorAz = stdErrorCalib['az']
    stdErrorEl = stdErrorCalib['el']
    
    sessionDictIn['calibTrialInfo'][(stdOutLabel,'az')] = stdErrorAz#stdError['az'].values
    sessionDictIn['calibTrialInfo'][(stdOutLabel,'el')] = stdErrorEl#stdError['el'].values
    
    # GD: is this calculation correct?
    sessionDictIn['calibTrialInfo'][(stdOutLabel,'euclidean')] = np.sqrt( stdErrorAz.values**2 + stdErrorEl.values**2)
    
    logger.info('Added sessionDict[\'calibTrialInfo\'][(\'{0}\',\'az\']'.format(meanOutLabel))
    logger.info('Added sessionDict[\'calibTrialInfo\'][(\'{0}\',\'el\']'.format(meanOutLabel))
    logger.info('Added sessionDict[\'calibTrialInfo\'][(\'{0}\',\'euclidean\']'.format(meanOutLabel))

    logger.info('Added sessionDict[\'calibTrialInfo\'][(\'{0}\',\'az\']'.format(stdOutLabel))
    logger.info('Added sessionDict[\'calibTrialInfo\'][(\'{0}\',\'el\']'.format(stdOutLabel))
    logger.info('Added sessionDict[\'calibTrialInfo\'][(\'{0}\',\'euclidean\']'.format(stdOutLabel))
    
    return sessionDictIn

def calcGazeToTargetFixError(sessionDictIn,gazeLabelIn,targetLabelIn,columnOutLabel, sessionDictKey='processedExp'):

    # For each frame of the dataframe, calculate the distance between gaze and the target
    sessionDictIn[sessionDictKey][(columnOutLabel,'az')] = sessionDictIn[sessionDictKey].apply(lambda row: row[(gazeLabelIn,'az')] -  row[(targetLabelIn,'az')],axis=1 )
    sessionDictIn[sessionDictKey][(columnOutLabel,'el')] = sessionDictIn[sessionDictKey].apply(lambda row: row[(gazeLabelIn,'el')] -  row[(targetLabelIn,'el')],axis=1 )
    sessionDictIn[sessionDictKey][(columnOutLabel,'euclidean')] = np.sqrt(sessionDictIn[sessionDictKey][(columnOutLabel,'az')]**2 + sessionDictIn[sessionDictKey][(columnOutLabel,'el')]**2)

    print("LEN OF EUCLIDEAN: {}".format(len(sessionDictIn[sessionDictKey][(columnOutLabel,'euclidean')])))

    try:
        sessionDictIn['processedExp'][(columnOutLabel,'az')] = sessionDictIn['processedExp'].apply(lambda row: row[(gazeLabelIn,'az')] -  row[(targetLabelIn,'az')],axis=1 )
        ball_catching = True
    except ValueError:
        ball_catching = False

    if ball_catching:
        sessionDictIn['processedExp'][(columnOutLabel,'el')] = sessionDictIn['processedExp'].apply(lambda row: row[(gazeLabelIn,'el')] -  row[(targetLabelIn,'el')],axis=1 )
        sessionDictIn['processedExp'][(columnOutLabel,'euclidean')] = np.sqrt(sessionDictIn['processedExp'][(columnOutLabel,'az')]**2 + sessionDictIn['processedExp'][(columnOutLabel,'el')]**2)
    
    # Group by trial, so that we can average within trial
    gbProcessedCalib_trial = sessionDictIn[sessionDictKey].groupby(['trialNumber'])
    if ball_catching:
        gbProcessedExp_trial = sessionDictIn['processedExp'].groupby(['trialNumber'])
    
    meanErrorCalib = gbProcessedCalib_trial.agg(np.nanmean)[columnOutLabel]
    if ball_catching:
        meanErrorExp = gbProcessedExp_trial.agg(np.nanmean)[columnOutLabel]
    meanOutLabel = 'mean' + columnOutLabel[0].capitalize() + columnOutLabel[1:]

    # Add it back into the trialInfo dataframe
    # Notice that we are VERY careful of the index we are using to align data.
    # Row indices start at 0, but trial numbers start at 1!
    
    meanError = meanErrorCalib.copy(deep=True)
    azList = []
    elList = []
    #for i in range(len(meanErrorExp['az'].values)):
        #print(meanError)
        #print(meanError.keys())
    #    azList.append(meanErrorExp['az'].values[i])
    #    elList.append(meanErrorExp['el'].values[i])
        #meanError['az'].append(meanErrorExp['az'].values[i])
        #meanError['el'].append(meanErrorExp['el'].values[i])
    
    #pd.concat([meanError['az'], azList])
    #pd.concat([meanError['el'], elList])
    
    if ball_catching:
        meanErrorAz = pd.concat([meanError['az'], meanErrorExp['az']])
        meanErrorEl = pd.concat([meanError['el'], meanErrorExp['el']])
    else:
        meanErrorAz = meanError['az']
        meanErrorEl = meanError['el']
    
    #meanError['az'].concat(azList)
    #meanError['el'].concat(elList)
    
    
    sessionDictIn['trialInfo'][(meanOutLabel,'az')] = meanErrorAz.values
    sessionDictIn['trialInfo'][(meanOutLabel,'el')] = meanErrorEl.values
    sessionDictIn['trialInfo'][(meanOutLabel,'euclidean')] = np.sqrt( meanErrorAz.values**2 + meanErrorEl.values**2)
    
    stdErrorCalib = gbProcessedCalib_trial.agg(np.nanstd)[columnOutLabel] 
    if ball_catching:
        stdErrorExp = gbProcessedExp_trial.agg(np.nanstd)[columnOutLabel] 
    stdOutLabel = 'std' + columnOutLabel[0].capitalize() + columnOutLabel[1:]

    if ball_catching:
        stdErrorAz = pd.concat([stdErrorCalib['az'], stdErrorExp['az']])
        stdErrorEl = pd.concat([stdErrorCalib['el'], stdErrorExp['el']])
    else:
        stdErrorAz = stdErrorCalib['az']
        stdErrorEl = stdErrorCalib['el']

    sessionDictIn['trialInfo'][(stdOutLabel,'az')] = stdErrorAz#stdError['az'].values
    sessionDictIn['trialInfo'][(stdOutLabel,'el')] = stdErrorEl#stdError['el'].values
    
    # GD: is this calculation correct?
    sessionDictIn['trialInfo'][(stdOutLabel,'euclidean')] = np.sqrt( stdErrorAz.values**2 + stdErrorEl.values**2)
    
    logger.info('Added sessionDict[\'trialInfo\'][(\'{0}\',\'az\']'.format(meanOutLabel))
    logger.info('Added sessionDict[\'trialInfo\'][(\'{0}\',\'el\']'.format(meanOutLabel))
    logger.info('Added sessionDict[\'trialInfo\'][(\'{0}\',\'euclidean\']'.format(meanOutLabel))

    logger.info('Added sessionDict[\'trialInfo\'][(\'{0}\',\'az\']'.format(stdOutLabel))
    logger.info('Added sessionDict[\'trialInfo\'][(\'{0}\',\'el\']'.format(stdOutLabel))
    logger.info('Added sessionDict[\'trialInfo\'][(\'{0}\',\'euclidean\']'.format(stdOutLabel))
    
    return sessionDictIn


def calcFixationStatistics(sessionDictIn, confidenceThresh = False):


    if confidenceThresh == False:
        logger.info('No confidence threshold applied.')

    sessionDictIn['processedCalib']['diffPupilTime'] = sessionDictIn['processedCalib']['pupilTimestamp'].diff()

    gridSize_widthHeight_gridnum = list(sessionDictIn['trialInfo'].groupby([('gridSize', 'heightDegs'), 
                                                                     ('gridSize', 'widthDegs')]).count().index)

    gb_type_h_w = sessionDictIn['trialInfo'].groupby(['targetType',
                                                    ('gridSize', 'heightDegs'), 
                                                    ('gridSize', 'widthDegs')])



    dataFrameRows = []

    for eyeId in range(3):
        
        gazeSphericalCol ='gaze{}Spherical'.format(eyeId)
        fixErrColumn ='fixError_eye{}'.format(eyeId)
    
        for (gHeight,gWidth) in gridSize_widthHeight_gridnum:

            targetLoc_targNum_AzEl = gb_type_h_w.get_group(('fixation', gHeight,gWidth))['fixTargetSpherical'].drop_duplicates().values

            for tNum,(tX,tY) in enumerate(targetLoc_targNum_AzEl): 

                #### Mean and sstandard deviation!
                # There has got to be an easier way...

                # First, get all fixation trials
                gbTargetType = sessionDictIn['trialInfo'].groupby(['targetType'])
                fixTrialsDf = gbTargetType.get_group('fixation')

                # Now, group by grid width, height, and pull out data for the grid size that we care about
                gbFixTrials = fixTrialsDf.groupby([('gridSize', 'heightDegs'), ('gridSize', 'widthDegs')])
                fixTrialsDf = gbFixTrials.get_group((gHeight,gWidth))

                # Now, group by fixTargetSpherical az / el, and pull out data for the grid size that we care about
                gbFixTrials = fixTrialsDf.groupby([('fixTargetSpherical','az'),('fixTargetSpherical','el')])
                trialsInGroup = gbFixTrials.get_group((tX,tY)) 

                # You cannot currently get multiple groups at a time from a groupby
                # So, here's an ugly way to concatenate the data from multiple groups.
                gbTrials = sessionDictIn['processedCalib'].groupby('trialNumber')
                fixRowDataDf = gbTrials.get_group(trialsInGroup['trialNumber'].values[0])
                for x in trialsInGroup['trialNumber'][1:]:
                    fixRowDataDf = pd.concat([fixRowDataDf,gbTrials.get_group(x)])

                origDFLength = len(fixRowDataDf)
                meanConfidenceBeforeThresh = np.nanmean(fixRowDataDf['confidence'])
                stdConfidenceBeforeThresh = np.nanstd(fixRowDataDf['confidence'])

                if confidenceThresh:

                    # Confidence thresholding
                    lowConfIdx = np.where(fixRowDataDf['confidence'] < confidenceThresh)[0]

                    fixRowDataDf = fixRowDataDf[fixRowDataDf['confidence'] >= confidenceThresh]
                    numSamplesAfterConfThresh = len(fixRowDataDf)

                    numSamplesBelowConThresh = len(lowConfIdx)
                    pctSamplesBelowConThresh = 100. * (numSamplesBelowConThresh / origDFLength)

                    logger.info(
                        '{:.1f}% samples below confidence threshold for fixation analysis'.format(pctSamplesBelowConThresh))

                else:

                    numSamplesBelowConThresh = len(fixRowDataDf)
                    numSamplesBelowConThresh = np.nan
                    pctSamplesBelowConThresh = np.nan
                    numSamplesAfterConfThresh = np.nan



                ######

                #meanErr = np.nanmean(fixRowDataDf[fixErrColumn]['euclidean'])
                meanGazeAz = np.nanmean(fixRowDataDf[gazeSphericalCol]['az'])
                meanGazeEl = np.nanmean(fixRowDataDf[gazeSphericalCol]['el'])
                meanErr = np.sqrt(np.square(meanGazeAz - tX) + np.square(meanGazeEl - tY))
                
                stdGazeAz = np.nanstd(fixRowDataDf[gazeSphericalCol]['az'])
                stdGazeEl = np.nanstd(fixRowDataDf[gazeSphericalCol]['el'])

                # Cum duration of the fixation frames
                # Note that I am throwing out outliers beyond 2 STD here.
                # These large values appear because the timestamps are discontinous 
                # at the transition between fixations

                diffPupilTime = fixRowDataDf['diffPupilTime'].values
                mu = np.nanmean(np.abs(diffPupilTime))
                sigma = np.nanstd(np.abs(diffPupilTime))
                outlierIdx = np.where( np.abs(diffPupilTime > mu + (2*sigma) ))[0]
                diffPupilTime[outlierIdx] = np.nan
                fixRowDataDf['diffPupilTime'] = diffPupilTime
                totalFixTimeSecs = np.sum(fixRowDataDf['diffPupilTime'])

                fixErr = fixRowDataDf[fixErrColumn]['euclidean'].values
                fixErr[outlierIdx] = np.nan

                precision = np.nanstd(fixRowDataDf[fixErrColumn]['euclidean'])

                dataFrameRow = {
                    ('eyeId',''): eyeId,
                    ('fixTargetSpherical','az'): tX,
                    ('fixTargetSpherical','el'): tY,
                    ('gridSize', 'heightDegs'): gHeight,
                    ('gridSize', 'widthDegs'): gWidth,
                    ('meanConfidenceBeforeThresh',''): np.round(meanConfidenceBeforeThresh,2),
                    ('stdConfidenceBeforeThresh',''): np.round(stdConfidenceBeforeThresh,2),
                    ('secondsInFix',''): np.round(totalFixTimeSecs,2),
                    ('numSamplesBeforeConfThresh',''): origDFLength,
                    ('numSamplesAfterConfThresh',''): numSamplesAfterConfThresh,
                    ('confidenceThreshold',''): confidenceThresh,
                    ('accuracy','euclidean'): meanErr,
                    ('accuracy','az'): meanGazeAz,
                    ('accuracy','el'): meanGazeEl,
                    ('accuracy','el'): meanGazeEl,
                    ('accuracy','el'): meanGazeEl,
                    ('meanGazeSpherical','az'): meanGazeAz,
                    ('meanGazeSpherical','el'): meanGazeEl,
                    ('stdGazeSpherical','az'): stdGazeAz,
                    ('stdGazeSpherical','el'): stdGazeEl,
                    ('gazePrecision',''): precision,
                    ('numSamplesBelowConf',''): numSamplesBelowConThresh,
                    ('pctSamplesBelowConf',''): pctSamplesBelowConThresh,
                    }


                dataFrameRows.append(dataFrameRow)

    fixDataDF = pd.DataFrame(dataFrameRows)
    fixDataDF.columns = pd.MultiIndex.from_tuples(fixDataDF.columns)
    sessionDictIn['fixAssessmentData'] = fixDataDF
    
    logger.info('Added sessionDict[\'fixAssessmentData\']')
    
    return sessionDictIn

def plotFixAssessment(sessionDictIn, saveDir = False, confidenceThresh=False, title="", show_filtered_out=False):
    
    fixDF = sessionDictIn['fixAssessmentData']

    figWidthHeight = 30
    plt.style.use('ggplot')

    gb_h_w = fixDF.groupby([('gridSize', 'heightDegs'), ('gridSize', 'widthDegs')])

    for eyeId in range(3):
        gazeSphericalCol ='gaze{}Spherical'.format(eyeId)
        for (gHeight,gWidth) in list(gb_h_w.groups.keys()):
            fig, ax = plt.subplots()
            fig.set_size_inches(7, 7)
            ax.set_xlim([-figWidthHeight,figWidthHeight])
            ax.set_ylim([-figWidthHeight,figWidthHeight])
            if show_filtered_out:
                fig_f, ax_f = plt.subplots()
                fig_f.set_size_inches(7, 7)
                ax_f.set_xlim([-figWidthHeight,figWidthHeight])
                ax_f.set_ylim([-figWidthHeight,figWidthHeight])
            
            targetLoc_targNum_AzEl = gb_h_w.get_group((gHeight,gWidth))['fixTargetSpherical'].drop_duplicates().values
            colors = cm.Set1(np.linspace(0, 1, len(targetLoc_targNum_AzEl)))
            colorIndex = 0
            
            for (tX,tY) in list(targetLoc_targNum_AzEl):

                # cycDataDF = fixDF[(fixDF['eyeId'] == 2) &
                #                     (fixDF[('gridSize', 'heightDegs')] == gHeight) &
                #                     (fixDF[('gridSize', 'widthDegs')] == gWidth) &
                #                     (fixDF[('fixTargetSpherical', 'az')] == tX) &
                #                     (fixDF[('fixTargetSpherical', 'el')] == tY)]
                # gX = cycDataDF[('meanGazeSpherical', 'az')]
                # gY = cycDataDF[('meanGazeSpherical', 'el')]
                # acc = np.sqrt((tX - gX) ** 2.0 + (tY - gY) ** 2.0)

                fixAtTarget = fixDF[(fixDF['eyeId']== eyeId ) & 
                                      (fixDF[('gridSize', 'heightDegs')] == gHeight ) &
                                      (fixDF[('gridSize', 'widthDegs')] == gWidth ) &
                                      (fixDF[('fixTargetSpherical','az')]== tX ) &
                                      (fixDF[('fixTargetSpherical','el')]== tY )]


                ax.text(tX, tY+2, '{:.2f}\n({:.2f})\n{:.2f}%'.format(
                    float(fixAtTarget[('accuracy', 'euclidean')]),
                    float(fixAtTarget[('gazePrecision', '')]),
                    100.0-float(fixAtTarget[('pctSamplesBelowConf', '')])),ha='center',va='bottom',size=8)

                ax.plot([tX, np.float(fixAtTarget[('meanGazeSpherical', 'az')])], 
                         [tY, np.float(fixAtTarget[('meanGazeSpherical', 'el')])],c='grey')

                ax.errorbar(fixAtTarget[('meanGazeSpherical', 'az')],
                             fixAtTarget[('meanGazeSpherical', 'el')], 
                             xerr=fixAtTarget[('stdGazeSpherical', 'az')], 
                             yerr=fixAtTarget[('stdGazeSpherical', 'el')],c='r')

                tH = ax.scatter(targetLoc_targNum_AzEl[:,0],
                                 targetLoc_targNum_AzEl[:,1], c='blue')

                gH = ax.scatter(fixAtTarget[('meanGazeSpherical', 'az')], 
                                 fixAtTarget[('meanGazeSpherical', 'el')],c='r')
                
                if show_filtered_out:
                    ax_f.text(tX, tY+2, '{:.2f}\n({:.2f})\n{:.2f}%'.format(
                        float(fixAtTarget[('accuracy', 'euclidean')]),
                        float(fixAtTarget[('gazePrecision', '')]),
                        100.0-float(fixAtTarget[('pctSamplesBelowConf', '')])),ha='center',va='bottom',size=8)

                    ax_f.plot([tX, np.float(fixAtTarget[('meanGazeSpherical', 'az')])], 
                             [tY, np.float(fixAtTarget[('meanGazeSpherical', 'el')])],c='grey')

                    ax_f.errorbar(fixAtTarget[('meanGazeSpherical', 'az')],
                                 fixAtTarget[('meanGazeSpherical', 'el')], 
                                 xerr=fixAtTarget[('stdGazeSpherical', 'az')], 
                                 yerr=fixAtTarget[('stdGazeSpherical', 'el')],c='r')

                    tH_f = ax_f.scatter(targetLoc_targNum_AzEl[:,0],
                                     targetLoc_targNum_AzEl[:,1], c='blue')

                    gH_f = ax_f.scatter(fixAtTarget[('meanGazeSpherical', 'az')], 
                                     fixAtTarget[('meanGazeSpherical', 'el')],c='r')
                    
                 
                # First, get all fixation trials
                gbTargetType = sessionDictIn['trialInfo'].groupby(['targetType'])
                fixTrialsDf = gbTargetType.get_group('fixation')

                # Now, group by grid width, height, and pull out data for the grid size that we care about
                gbFixTrials = fixTrialsDf.groupby([('gridSize', 'heightDegs'), ('gridSize', 'widthDegs')])
                fixTrialsDf = gbFixTrials.get_group((gHeight,gWidth))

                # Now, group by fixTargetSpherical az / el, and pull out data for the grid size that we care about
                gbFixTrials = fixTrialsDf.groupby([('fixTargetSpherical','az'),('fixTargetSpherical','el')])
                trialsInGroup = gbFixTrials.get_group((tX,tY))
                gbTrials = sessionDictIn['processedCalib'].groupby('trialNumber')
                trialsInGroup = gbFixTrials.get_group((tX,tY))
                fixRowDataDf = gbTrials.get_group(trialsInGroup['trialNumber'].values[0])
                for x in trialsInGroup['trialNumber'][1:]:
                    fixRowDataDf = pd.concat([fixRowDataDf,gbTrials.get_group(x)])
                ## Filter by confidence
                if confidenceThresh:
                    # Confidence thresholding
                    fixRowDataDf = fixRowDataDf[fixRowDataDf['confidence'] >= confidenceThresh]
                    
                if show_filtered_out:
                    fixRowDataDf_f = fixRowDataDf
                    if 'pupil_confidence0' in fixRowDataDf_f.columns:
                        fixRowDataDf_f = fixRowDataDf_f[fixRowDataDf_f['pupil_confidence0'] != 0.75]
                        #fixRowDataDf_f = fixRowDataDf_f[fixRowDataDf_f['pupil_confidence1'] == 0.75]
                    if 'pupil_confidence1' in fixRowDataDf_f.columns:
                        fixRowDataDf_f = fixRowDataDf_f[fixRowDataDf_f['pupil_confidence1'] != 0.75]
                    fixAz_f = fixRowDataDf_f[gazeSphericalCol]['az']
                    fixEl_f = fixRowDataDf_f[gazeSphericalCol]['el']
                    
                fixAz = fixRowDataDf[gazeSphericalCol]['az']
                fixEl = fixRowDataDf[gazeSphericalCol]['el']
                color = colors[colorIndex]
                colorIndex += 1
                ax.scatter(fixAz, fixEl,color=color,s=5, alpha=0.25)
                ax.legend([gH,tH], ['gaze','target'])
                
                if show_filtered_out:
                    ax_f.scatter(fixAz_f, fixEl_f,color=color,s=5, alpha=0.25)
                    ax_f.legend([gH_f,tH_f], ['gaze','target'])

                figTitle = '{0}_eye{1}_{2}x{3}_C-{4:0.2f}_{5}'.format(sessionDictIn['subID'][-(np.min([4, len(sessionDictIn['subID'])])):], eyeId, gHeight, gWidth,
                                                                  np.float(confidenceThresh), title)
                
                ax.set_title(figTitle)
                ax.set_xlabel('degrees azimuth',fontsize=15)
                ax.set_ylabel('degrees elevation',fontsize=15)
                #ax.yticks(fontsize=15)
                #ax.xticks(fontsize=15)
                ax.tick_params(axis='both', labelsize=15)
                
                if show_filtered_out:
                    figTitle_f = '(filtered out) ' + figTitle
                    ax_f.set_title(figTitle_f)
                    ax_f.set_xlabel('degrees azimuth',fontsize=15)
                    ax_f.set_ylabel('degrees elevation',fontsize=15)
                    #ax_f.yticks(fontsize=15)
                    #ax_f.xticks(fontsize=15)
                    ax_f.tick_params(axis='both', labelsize=15)

            if saveDir :
                import os
                # saveDir =
                exportFolder = sessionDictIn['plExportFolder']
                if exportFolder is None:
                    exportFolder = ""
                s = '{}/{}_{}/'.format(saveDir,sessionDictIn['subID'], sessionDictIn['plExportFolder'])
                directory = os.path.dirname(s)
                try:
                    os.stat(directory)
                except:
                    os.mkdir(directory)

                fig.savefig(directory +  '/' + figTitle + '.png')
                plt.close(fig)

                if show_filtered_out:
                    fig_f.savefig(directory +  '/(filtered) ' + figTitle + '.png')
                    plt.close(fig_f)

            else:
                plt.show()


def plotCalibrationSequence(sessionDictIn, saveDir = False, confidenceThresh=False, title="", show_filtered_out=False):
    
    fixDF = sessionDictIn['calibrationSequenceData']

    figWidthHeight = 30
    plt.style.use('ggplot')

    for eyeId in range(3):
        gazeSphericalCol ='gaze{}Spherical'.format(eyeId)
        fig, ax = plt.subplots()
        fig.set_size_inches(7, 7)
        ax.set_xlim([-figWidthHeight,figWidthHeight])
        ax.set_ylim([-figWidthHeight,figWidthHeight])
        if show_filtered_out:
            fig_f, ax_f = plt.subplots()
            fig_f.set_size_inches(7, 7)
            ax_f.set_xlim([-figWidthHeight,figWidthHeight])
            ax_f.set_ylim([-figWidthHeight,figWidthHeight])
        
        targetLoc_targNum_AzEl = fixDF['fixTargetSpherical'].drop_duplicates().values
        colors = cm.Set1(np.linspace(0, 1, len(targetLoc_targNum_AzEl)))
        colorIndex = 0
        
        for (tX,tY) in list(targetLoc_targNum_AzEl):

            fixAtTarget = fixDF[(fixDF['eyeId']== eyeId ) &
                                  (fixDF[('fixTargetSpherical','az')]== tX ) &
                                  (fixDF[('fixTargetSpherical','el')]== tY )]


            ax.text(tX, tY+2, '{:.2f}\n({:.2f})\n{:.2f}%'.format(
                float(fixAtTarget[('accuracy', 'euclidean')]),
                float(fixAtTarget[('gazePrecision', '')]),
                100.0-float(fixAtTarget[('pctSamplesBelowConf', '')])),ha='center',va='bottom',size=8)

            ax.plot([tX, np.float(fixAtTarget[('meanGazeSpherical', 'az')])], 
                     [tY, np.float(fixAtTarget[('meanGazeSpherical', 'el')])],c='grey')

            ax.errorbar(fixAtTarget[('meanGazeSpherical', 'az')],
                         fixAtTarget[('meanGazeSpherical', 'el')], 
                         xerr=fixAtTarget[('stdGazeSpherical', 'az')], 
                         yerr=fixAtTarget[('stdGazeSpherical', 'el')],c='r')

            tH = ax.scatter(targetLoc_targNum_AzEl[:,0],
                             targetLoc_targNum_AzEl[:,1], c='blue')

            gH = ax.scatter(fixAtTarget[('meanGazeSpherical', 'az')], 
                             fixAtTarget[('meanGazeSpherical', 'el')],c='r')
            
            if show_filtered_out:
                ax_f.text(tX, tY+2, '{:.2f}\n({:.2f})\n{:.2f}%'.format(
                    float(fixAtTarget[('accuracy', 'euclidean')]),
                    float(fixAtTarget[('gazePrecision', '')]),
                    100.0-float(fixAtTarget[('pctSamplesBelowConf', '')])),ha='center',va='bottom',size=8)

                ax_f.plot([tX, np.float(fixAtTarget[('meanGazeSpherical', 'az')])], 
                         [tY, np.float(fixAtTarget[('meanGazeSpherical', 'el')])],c='grey')

                ax_f.errorbar(fixAtTarget[('meanGazeSpherical', 'az')],
                             fixAtTarget[('meanGazeSpherical', 'el')], 
                             xerr=fixAtTarget[('stdGazeSpherical', 'az')], 
                             yerr=fixAtTarget[('stdGazeSpherical', 'el')],c='r')

                tH_f = ax_f.scatter(targetLoc_targNum_AzEl[:,0],
                                 targetLoc_targNum_AzEl[:,1], c='blue')

                gH_f = ax_f.scatter(fixAtTarget[('meanGazeSpherical', 'az')], 
                                 fixAtTarget[('meanGazeSpherical', 'el')],c='r')
            
            gbFixTrials = fixDF.groupby([('fixTargetSpherical','az'), ('fixTargetSpherical','el')])
            trialsInGroup = gbFixTrials.get_group((tX,tY))
            
            gbTrials = sessionDictIn['processedSequence'].groupby([('targetLocalSpherical','az'), ('targetLocalSpherical','el')])
            fixRowDataDf = gbTrials.get_group((tX,tY))

            ## Filter by confidence
            if confidenceThresh:
                # Confidence thresholding
                fixRowDataDf = fixRowDataDf[fixRowDataDf['confidence'] >= confidenceThresh]
                
            if show_filtered_out:
                fixRowDataDf_f = fixRowDataDf
                if 'pupil_confidence0' in fixRowDataDf_f.columns:
                    fixRowDataDf_f = fixRowDataDf_f[fixRowDataDf_f['pupil_confidence0'] != 0.75]
                    #fixRowDataDf_f = fixRowDataDf_f[fixRowDataDf_f['pupil_confidence1'] == 0.75]
                if 'pupil_confidence1' in fixRowDataDf_f.columns:
                    fixRowDataDf_f = fixRowDataDf_f[fixRowDataDf_f['pupil_confidence1'] != 0.75]
                fixAz_f = fixRowDataDf_f[gazeSphericalCol]['az']
                fixEl_f = fixRowDataDf_f[gazeSphericalCol]['el']
                
            fixAz = fixRowDataDf[gazeSphericalCol]['az']
            fixEl = fixRowDataDf[gazeSphericalCol]['el']
            color = colors[colorIndex]
            colorIndex += 1
            ax.scatter(fixAz, fixEl,color=color,s=5, alpha=0.25)
            ax.legend([gH,tH], ['gaze','target'])
            
            if show_filtered_out:
                ax_f.scatter(fixAz_f, fixEl_f,color=color,s=5, alpha=0.25)
                ax_f.legend([gH_f,tH_f], ['gaze','target'])

            figTitle = 'CALIB_{0}_eye{1}_C-{2:0.2f}_{3}'.format(sessionDictIn['subID'][-(np.min([4, len(sessionDictIn['subID'])])):], eyeId,
                                                              np.float(confidenceThresh), title)
            
            ax.set_title(figTitle)
            ax.set_xlabel('degrees azimuth',fontsize=15)
            ax.set_ylabel('degrees elevation',fontsize=15)
            #ax.yticks(fontsize=15)
            #ax.xticks(fontsize=15)
            ax.tick_params(axis='both', labelsize=15)
            
            if show_filtered_out:
                figTitle_f = '(filtered out) ' + figTitle
                ax_f.set_title(figTitle_f)
                ax_f.set_xlabel('degrees azimuth',fontsize=15)
                ax_f.set_ylabel('degrees elevation',fontsize=15)
                #ax_f.yticks(fontsize=15)
                #ax_f.xticks(fontsize=15)
                ax_f.tick_params(axis='both', labelsize=15)

        if saveDir :
            import os
            # saveDir =
            exportFolder = sessionDictIn['plExportFolder']
            if exportFolder is None:
                exportFolder = ""
            s = '{}/{}_{}/'.format(saveDir,sessionDictIn['subID'], sessionDictIn['plExportFolder'])
            directory = os.path.dirname(s)
            try:
                os.stat(directory)
            except:
                os.mkdir(directory)

            fig.savefig(directory +  '/' + figTitle + '.png')
            plt.close(fig)

            if show_filtered_out:
                fig_f.savefig(directory +  '/(filtered) ' + figTitle + '.png')
                plt.close(fig_f)

        else:
            plt.show()                



def calcCalibrationSequenceStatistics(sessionDictIn, confidenceThresh = False):


    if confidenceThresh == False:
        logger.info('No confidence threshold applied.')

    sessionDictIn['processedSequence']['diffPupilTime'] = sessionDictIn['processedSequence']['pupilTimestamp'].diff()

    #gb_type_h_w = sessionDictIn['calibTrialInfo'].groupby(['targetType',
    #                                                ('gridSize', 'heightDegs'), 
    #                                                ('gridSize', 'widthDegs')])



    dataFrameRows = []

    for eyeId in range(3):
        
        gazeSphericalCol ='gaze{}Spherical'.format(eyeId)
        fixErrColumn ='fixError_eye{}'.format(eyeId)
    

        targetLoc_targNum_AzEl = sessionDictIn['processedSequence']['targetLocalSpherical'].drop_duplicates().values#gb_type_h_w.get_group(('fixation', gHeight,gWidth))['fixTargetSpherical'].drop_duplicates().values
        
        for tNum,(tX,tY) in enumerate(targetLoc_targNum_AzEl): 

            #### Mean and sstandard deviation!
            # There has got to be an easier way...

            # First, get all calibration trials
            fixTrialsDf = sessionDictIn['processedSequence']

            # Now, group by grid width, height, and pull out data for the grid size that we care about
            #gbFixTrials = fixTrialsDf.groupby([('gridSize', 'heightDegs'), ('gridSize', 'widthDegs')])
            #fixTrialsDf = gbFixTrials.get_group((gHeight,gWidth))

            # Now, group by fixTargetSpherical az / el, and pull out data for the grid size that we care about
            #gbFixTrials = fixTrialsDf.groupby([('fixTargetSpherical','az'),('fixTargetSpherical','el')])
            #trialsInGroup = gbFixTrials.get_group((tX,tY)) 

            gbFixTrials = fixTrialsDf.groupby([('targetLocalSpherical','az'), ('targetLocalSpherical','el')])
            trialsInGroup = gbFixTrials.get_group((tX,tY)) 

            # You cannot currently get multiple groups at a time from a groupby
            # So, here's an ugly way to concatenate the data from multiple groups.
            gbTrials = sessionDictIn['processedSequence'].groupby('trialNumber')
            fixRowDataDf = gbTrials.get_group(trialsInGroup['trialNumber'].values[0])
            for x in trialsInGroup['trialNumber'][1:]:
                fixRowDataDf = pd.concat([fixRowDataDf,gbTrials.get_group(x)])

            origDFLength = len(fixRowDataDf)
            meanConfidenceBeforeThresh = np.nanmean(fixRowDataDf['confidence'])
            stdConfidenceBeforeThresh = np.nanstd(fixRowDataDf['confidence'])

            if confidenceThresh:

                # Confidence thresholding
                lowConfIdx = np.where(fixRowDataDf['confidence'] < confidenceThresh)[0]

                fixRowDataDf = fixRowDataDf[fixRowDataDf['confidence'] >= confidenceThresh]
                numSamplesAfterConfThresh = len(fixRowDataDf)

                numSamplesBelowConThresh = len(lowConfIdx)
                pctSamplesBelowConThresh = 100. * (numSamplesBelowConThresh / origDFLength)

                logger.info(
                    '{:.1f}% samples below confidence threshold for fixation analysis'.format(pctSamplesBelowConThresh))

            else:

                numSamplesBelowConThresh = len(fixRowDataDf)
                numSamplesBelowConThresh = np.nan
                pctSamplesBelowConThresh = np.nan
                numSamplesAfterConfThresh = np.nan



            ######

            #meanErr = np.nanmean(fixRowDataDf[fixErrColumn]['euclidean'])
            meanGazeAz = np.nanmean(fixRowDataDf[gazeSphericalCol]['az'])
            meanGazeEl = np.nanmean(fixRowDataDf[gazeSphericalCol]['el'])
            meanErr = np.sqrt(np.square(meanGazeAz - tX) + np.square(meanGazeEl - tY))
            
            stdGazeAz = np.nanstd(fixRowDataDf[gazeSphericalCol]['az'])
            stdGazeEl = np.nanstd(fixRowDataDf[gazeSphericalCol]['el'])

            # Cum duration of the fixation frames
            # Note that I am throwing out outliers beyond 2 STD here.
            # These large values appear because the timestamps are discontinous 
            # at the transition between fixations

            diffPupilTime = fixRowDataDf['diffPupilTime'].values
            mu = np.nanmean(np.abs(diffPupilTime))
            sigma = np.nanstd(np.abs(diffPupilTime))
            outlierIdx = np.where( np.abs(diffPupilTime > mu + (2*sigma) ))[0]
            diffPupilTime[outlierIdx] = np.nan
            fixRowDataDf['diffPupilTime'] = diffPupilTime
            totalFixTimeSecs = np.sum(fixRowDataDf['diffPupilTime'])

            fixErr = fixRowDataDf[fixErrColumn]['euclidean'].values
            fixErr[outlierIdx] = np.nan

            precision = np.nanstd(fixRowDataDf[fixErrColumn]['euclidean'])

            dataFrameRow = {
                ('eyeId',''): eyeId,
                ('fixTargetSpherical','az'): tX,
                ('fixTargetSpherical','el'): tY,
                ('meanConfidenceBeforeThresh',''): np.round(meanConfidenceBeforeThresh,2),
                ('stdConfidenceBeforeThresh',''): np.round(stdConfidenceBeforeThresh,2),
                ('secondsInFix',''): np.round(totalFixTimeSecs,2),
                ('numSamplesBeforeConfThresh',''): origDFLength,
                ('numSamplesAfterConfThresh',''): numSamplesAfterConfThresh,
                ('confidenceThreshold',''): confidenceThresh,
                ('accuracy','euclidean'): meanErr,
                ('accuracy','az'): meanGazeAz,
                ('accuracy','el'): meanGazeEl,
                ('accuracy','el'): meanGazeEl,
                ('accuracy','el'): meanGazeEl,
                ('meanGazeSpherical','az'): meanGazeAz,
                ('meanGazeSpherical','el'): meanGazeEl,
                ('stdGazeSpherical','az'): stdGazeAz,
                ('stdGazeSpherical','el'): stdGazeEl,
                ('gazePrecision',''): precision,
                ('numSamplesBelowConf',''): numSamplesBelowConThresh,
                ('pctSamplesBelowConf',''): pctSamplesBelowConThresh,
                }


            dataFrameRows.append(dataFrameRow)

    fixDataDF = pd.DataFrame(dataFrameRows)
    fixDataDF.columns = pd.MultiIndex.from_tuples(fixDataDF.columns)
    sessionDictIn['calibrationSequenceData'] = fixDataDF
    
    logger.info('Added sessionDict[\'calibrationSequenceData\']')
    
    return sessionDictIn


def plotIndividualFix(sessionDictIn, confidenceThresh=False, saveDir=False, title=""):

    dataFrameRows = []
    figWidthHeight = 45

    fixDF = sessionDictIn['fixAssessmentData']

    for eyeId in range(3):

        # gazeSphericalCol ='gaze{}Spherical'.format(eyeId)
        # fixErrColumn ='fixError_eye{}'.format(eyeId)

        gridSize_widthHeight_gridnum = list(sessionDictIn['trialInfo'].groupby([('gridSize', 'heightDegs'),
                                                                                 ('gridSize', 'widthDegs')]).count().index)

        for (gHeight,gWidth) in gridSize_widthHeight_gridnum:

            gb_type_h_w = sessionDictIn['trialInfo'].groupby(['targetType',
                                                    ('gridSize', 'heightDegs'),
                                                    ('gridSize', 'widthDegs')])

            targetLoc_targNum_AzEl = gb_type_h_w.get_group(('fixation', gHeight,gWidth))['fixTargetSpherical'].drop_duplicates().values

            plt.style.use('ggplot')
            fig, ax = plt.subplots()
            fig.set_size_inches(7, 7)

            import matplotlib.cm as cm
            colors = cm.Set1(np.linspace(0, 1, len(targetLoc_targNum_AzEl)))

            for tNum,(tX,tY) in enumerate(targetLoc_targNum_AzEl):


                fixAtTarget = fixDF[(fixDF['eyeId'] == eyeId) &
                                (fixDF[('gridSize', 'heightDegs')] == gHeight) &
                                (fixDF[('gridSize', 'widthDegs')] == gWidth) &
                                (fixDF[('fixTargetSpherical', 'az')] == tX) &
                                (fixDF[('fixTargetSpherical', 'el')] == tY)]




                sessionDictIn['processedCalib']['diffPupilTime'] = sessionDictIn['processedCalib']['pupilTimestamp'].diff()

                # dataFrameRows = []

                gazeSphericalCol ='gaze{}Spherical'.format(eyeId)
                fixErrColumn ='fixError_eye{}'.format(eyeId)

                # First, get all fixation trials
                gbTargetType = sessionDictIn['trialInfo'].groupby(['targetType'])
                fixTrialsDf = gbTargetType.get_group('fixation')

                # Now, group by grid width, height, and pull out data for the grid size that we care about
                gbFixTrials = fixTrialsDf.groupby([('gridSize', 'heightDegs'), ('gridSize', 'widthDegs')])
                fixTrialsDf = gbFixTrials.get_group((gHeight,gWidth))

                # Now, group by fixTargetSpherical az / el, and pull out data for the grid size that we care about
                gbFixTrials = fixTrialsDf.groupby([('fixTargetSpherical','az'),('fixTargetSpherical','el')])
                trialsInGroup = gbFixTrials.get_group((tX,tY))

                # You cannot currently get multiple groups at a time from a groupby
                # So, here's an ugly way to concatenate the data from multiple groups.
                gbTrials = sessionDictIn['processedCalib'].groupby('trialNumber')
                fixRowDataDf = gbTrials.get_group(trialsInGroup['trialNumber'].values[0])

                for x in trialsInGroup['trialNumber'][1:]:
                    fixRowDataDf = pd.concat([fixRowDataDf,gbTrials.get_group(x)])

                ## Filter by confidence
                origDFLength = len(fixRowDataDf)
                if confidenceThresh:

                    # Confidence thresholding
                    lowConfIdx = np.where(fixRowDataDf['confidence'] < confidenceThresh)[0]

                    fixRowDataDf = fixRowDataDf[fixRowDataDf['confidence'] >= confidenceThresh]
                    numSamplesAfterConfThresh = len(fixRowDataDf)

                    numSamplesBelowConThresh = len(lowConfIdx)
                    pctSamplesBelowConThresh = 100. * (numSamplesBelowConThresh / origDFLength)
                    pctSamplesAboveConThresh = 100. - pctSamplesBelowConThresh

                    logger.info(
                            '{:.1f}% samples below confidence threshold for fixation analysis'.format(pctSamplesBelowConThresh))
                else:
                    pctSamplesAboveConThresh = 100.0

                plt.text(tX, tY + 2, '{:.2f}\n({:.2f})\n{:.2f}%'.format(
                    float(fixAtTarget[('accuracy', 'euclidean')]),
                    float(fixAtTarget[('gazePrecision', '')]),
                    float(pctSamplesAboveConThresh)), ha='center', va='bottom', size=8)


                fixAz = fixRowDataDf[gazeSphericalCol]['az']
                fixEl = fixRowDataDf[gazeSphericalCol]['el']
                ax.set_xlim([-figWidthHeight, figWidthHeight])
                ax.set_ylim([-figWidthHeight, figWidthHeight])


                color = colors[tNum]
                plt.scatter(fixAz, fixEl,color=color,s=5)
                if eyeId==2:
                    print("(",tX,",",tY,")  ", len(fixAz))
                plt.scatter(tX, tY,c='blue')

            figTitle = '{0}_eye{1}_{2}x{3}_C-{4:0.2f}_{5}'.format(sessionDictIn['subID'][-(np.min([4, len(sessionDictIn['subID'])])):], eyeId, gHeight, gWidth, np.float(confidenceThresh), title)
            ax.set_title(figTitle)
            ax.set_xlabel('degrees azimuth',fontsize=15)
            ax.set_ylabel('degrees elevation',fontsize=15)
            plt.yticks(fontsize=15)
            plt.xticks(fontsize=15)

            if saveDir :

                import os
                # saveDir = \
                exportFolder = sessionDictIn['plExportFolder']
                if exportFolder is None:
                    exportFolder = ""
                s = '{}/{}_{}/perframe/'.format(saveDir,sessionDictIn['subID'], sessionDictIn['plExportFolder'])
                directory = os.path.dirname(s)

                try:
                    os.stat(directory)
                except:
                    os.mkdir(directory)

                plt.savefig(directory +  '/' + figTitle + '.png')
                plt.close(fig)

            else:
                plt.show()

