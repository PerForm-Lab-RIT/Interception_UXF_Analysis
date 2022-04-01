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


fmt = '%(levelname)s_%(name)s-%(funcName)s(): - %(message)s'
logging.basicConfig(level=logging.INFO, format=fmt)
logger = logging.getLogger(__name__)

# In[1]:

wd = os.getcwd()
os.chdir(wd)



def calcAverageGazeDirPerTrial(sessionDictIn):


    def adjToTrialInfo(newVals, sessionDictIn):

        calibAssRowidx = sessionDictIn['trialInfo'][sessionDictIn['trialInfo']['trialType'] == 'CalibrationAssessment'].index
        newVec = np.zeros(len(sessionDictIn['trialInfo']))
        newVec[:] = np.nan
        newVec[calibAssRowidx] = newVals
        return newVec

    gbProcessedCalib_trial = sessionDictIn['processedCalib'].groupby(['trialNumber'])

    newVals = gbProcessedCalib_trial.agg([np.nanmean])[('gaze0Spherical','az','nanmean')].values
    sessionDictIn['trialInfo'][('meanGaze0_Spherical','az')] =  adjToTrialInfo(newVals, sessionDictIn)

    newVals = gbProcessedCalib_trial.agg([np.nanmean])[('gaze0Spherical','el','nanmean')].values
    sessionDictIn['trialInfo'][('meanGaze0_Spherical','el')] =  adjToTrialInfo(newVals, sessionDictIn)

    newVals = gbProcessedCalib_trial.agg([np.nanmean])[('gaze1Spherical','az','nanmean')].values
    sessionDictIn['trialInfo'][('meanGaze1_Spherical','az')] =  adjToTrialInfo(newVals, sessionDictIn)

    newVals = gbProcessedCalib_trial.agg([np.nanmean])[('gaze1Spherical','el','nanmean')].values
    sessionDictIn['trialInfo'][('meanGaze1_Spherical','el')] =  adjToTrialInfo(newVals, sessionDictIn)

    newVals = gbProcessedCalib_trial.agg([np.nanmean])[('gaze2Spherical','el','nanmean')].values
    sessionDictIn['trialInfo'][('meanGaze2_Spherical','el')] =  adjToTrialInfo(newVals, sessionDictIn)


    return sessionDictIn

def calcCyclopean(sessionDictIn):

    xyz = (sessionDictIn['processedCalib']['gaze-normal0'] + sessionDictIn['processedCalib']['gaze-normal1']) / 2.0

    sessionDictIn['processedCalib'][('gaze-normal2','x')] = xyz['x']
    sessionDictIn['processedCalib'][('gaze-normal2','y')] = xyz['y']
    sessionDictIn['processedCalib'][('gaze-normal2','z')] = xyz['z']

    # sessionDict['processedCalib']['gaze-normal2'] = 
    sessionDictIn['processedCalib']['gaze-normal2'].apply(lambda row: normalizeVector(row),axis=1)
    return sessionDictIn

def normalizeVector(xyz):
    '''
    Input be a 3 element array of containing the x,y,and z data of a 3D vector.
    Returns a normalized 3 element array
    '''

    xyz = xyz / np.linalg.norm(xyz)
    return xyz 


def calcSphericalCoordinates(sessionDictIn,columnName,newColumnName,sessionDictKey='processedExp', flipY = False):

    dataDict = sessionDictIn[sessionDictKey]
    dataDict[(newColumnName,'az')] = np.rad2deg(np.arctan2(dataDict[(columnName,'x')],dataDict[(columnName,'z')]))
    dataDict[(newColumnName,'el')] = np.rad2deg(np.arctan2(dataDict[(columnName,'y')],dataDict[(columnName,'z')]))
    
    if flipY:
        dataDict[(newColumnName,'el')] = -    dataDict[(newColumnName,'el')]

    sessionDictIn[sessionDictKey] = dataDict
    
    logger.info('Added sessionDict[\'{0}\'][\'{1}\',\'az\']'.format(sessionDictKey, newColumnName))
    logger.info('Added sessionDict[\'{0}\'][\'{1}\',\'el\']'.format(sessionDictKey,newColumnName))
    
    return sessionDictIn



def calcTrialLevelCalibInfo(sessionIn):

    '''
    Input: Session dictionary
    Output:  Session dictionary with new column sessionDict['trialInfo']['targetType']
    '''

    gbProcessedCalib_trial = sessionIn['processedCalib'].groupby(['trialNumber'])

    targetTypes = []
    gridSize_height = []
    gridSize_width = []
    targeLocalPos_az = []
    targeLocalPos_el = []


    gbProcessedCalib_trial = sessionIn['processedCalib'].groupby(['trialNumber'])

    for trialRowIdx, trMetaData in sessionIn['trialInfo'].iterrows():

        targetType = []
        height = []
        width = []

        # This dataframe contains the per-frame processed data associated with this trial

        if ( trMetaData['trialType'][0] == 'CalibrationAssessment'):

            procDF = gbProcessedCalib_trial.get_group(int(trMetaData['trialNumber']))

            if( sum(procDF['isHeadFixed'] == False) ==  len(procDF) ):

                targetType = 'VOR'
                height = np.nan
                width = np.nan

            elif( sum(procDF['isHeadFixed'] == True) == len(procDF) ):

                height = np.float64(procDF['elevationHeight'].drop_duplicates())
                width = np.float64(procDF['azimuthWidth'].drop_duplicates())

                # Count the number of target positions within the local space (head-centered space)
                if( len(procDF['targeLocalPos'].drop_duplicates()) == 1 ):
                    # Only one target, so it's a fixation trial
                    targetType = 'fixation'

                    target_az = np.round(np.float64(procDF[('targetInHead_az')].drop_duplicates()),1)
                    target_el = np.round(np.float64(procDF[('targetInHead_el')].drop_duplicates()),1)

                else:
                    # multiple targets, so it's a saccade trial
                    targetType = 'fixation+saccade'

                    target_az = np.nan
                    target_el = np.nan

        else:

            # The trial has both head fixed and world fixed targets.  
            # We did not plan for that, so let's label it as "unknown."

            targetType = 'n/a'
            height = np.nan
            width = np.nan
            target_az = np.nan
            target_el = np.nan        


    #         print('Trial number: {tNum}, type: {tType}'.format(tNum = int(trMetaData['trialNumber']),
    #                                                            tType = targetType))

        targetTypes.append(targetType)
        gridSize_height.append(height)
        gridSize_width.append(width)
        targeLocalPos_az.append(target_az)
        targeLocalPos_el.append(target_el)


    sessionIn['trialInfo']['targetType'] = targetTypes
    sessionIn['trialInfo']['gridSize','heightDegs'] = gridSize_height
    sessionIn['trialInfo']['gridSize','widthDegs'] = gridSize_width
    sessionIn['trialInfo']['fixTargetSpherical','az'] = targeLocalPos_az
    sessionIn['trialInfo']['fixTargetSpherical','el'] = targeLocalPos_el

    sessionIn['trialInfo']['fixTargetSpherical','az'] = sessionIn['trialInfo']['fixTargetSpherical','az'].round(2)
    sessionIn['trialInfo']['fixTargetSpherical','el'] = sessionIn['trialInfo']['fixTargetSpherical','el'].round(2)

    logger.info('Added sessionDict[\'trialInfo\'][\'targetType\']')
    logger.info('Added sessionDict[\'trialInfo\'][\'gridSize\']')
    logger.info('Added sessionDict[\'trialInfo\'][\'fixTargetSpherical\']')

    return sessionIn



def calcGazeToTargetFixError(sessionDictIn,gazeLabelIn,targetLabelIn,columnOutLabel ):


    # For each frame of the dataframe, calculate the distance between gaze and the target
    sessionDictIn['processedCalib'][(columnOutLabel,'az')] = sessionDictIn['processedCalib'].apply(lambda row: row[(gazeLabelIn,'az')] -  row[(targetLabelIn,'az')],axis=1 )
    sessionDictIn['processedCalib'][(columnOutLabel,'el')] = sessionDictIn['processedCalib'].apply(lambda row: row[(gazeLabelIn,'el')] -  row[(targetLabelIn,'el')],axis=1 )
    sessionDictIn['processedCalib'][(columnOutLabel,'euclidean')] = np.sqrt(sessionDictIn['processedCalib'][(columnOutLabel,'az')]**2 + sessionDictIn['processedCalib'][(columnOutLabel,'el')]**2)

    # Group by trial, so that we can average within trial
    gbProcessedCalib_trial = sessionDictIn['processedCalib'].groupby(['trialNumber'])

    meanError = gbProcessedCalib_trial.agg(np.nanmean)[columnOutLabel]    
    meanOutLabel = 'mean' + columnOutLabel[0].capitalize() + columnOutLabel[1:]

    def adjToTrialInfo(newVals, sessionDictIn):

        calibAssRowidx = sessionDictIn['trialInfo'][sessionDictIn['trialInfo']['trialType'] == 'CalibrationAssessment'].index
        newVec = np.zeros(len(sessionDictIn['trialInfo']))
        newVec[:] = np.nan
        newVec[calibAssRowidx] = newVals
        return newVec


    sessionDictIn['trialInfo'][(meanOutLabel,'az')] = adjToTrialInfo(meanError['az'], sessionDictIn)
    sessionDictIn['trialInfo'][(meanOutLabel,'el')] = adjToTrialInfo(meanError['el'], sessionDictIn)
    sessionDictIn['trialInfo'][(meanOutLabel,'euclidean')] = np.sqrt(sessionDictIn['trialInfo'][(meanOutLabel,'az')]**2 + sessionDictIn['trialInfo'][(meanOutLabel,'el')]**2)

    stdError = gbProcessedCalib_trial.agg(np.nanstd)[columnOutLabel]    
    stdOutLabel = 'std' + columnOutLabel[0].capitalize() + columnOutLabel[1:]
    sessionDictIn['trialInfo'][(stdOutLabel,'az')] = adjToTrialInfo(stdError['az'], sessionDictIn)
    sessionDictIn['trialInfo'][(stdOutLabel,'az')] = adjToTrialInfo(stdError['el'], sessionDictIn)


    eucStdDev = np.sqrt( stdError['az'].values**2 + stdError['el'].values**2)
    sessionDictIn['trialInfo'][(stdOutLabel,'euclidean')] = adjToTrialInfo(eucStdDev, sessionDictIn)

    logger.info('Added sessionDict[\'trialInfo\'][(\'{0}\',\'az\']'.format(meanOutLabel))
    logger.info('Added sessionDict[\'trialInfo\'][(\'{0}\',\'el\']'.format(meanOutLabel))
    logger.info('Added sessionDict[\'trialInfo\'][(\'{0}\',\'euclidean\']'.format(meanOutLabel))

    logger.info('Added sessionDict[\'trialInfo\'][(\'{0}\',\'az\']'.format(stdOutLabel))
    logger.info('Added sessionDict[\'trialInfo\'][(\'{0}\',\'el\']'.format(stdOutLabel))
    logger.info('Added sessionDict[\'trialInfo\'][(\'{0}\',\'euclidean\']'.format(stdOutLabel))

    return sessionDictIn



def calcFixationStatistics(sessionDictIn, confidenceThresh = False):

    sessionDictIn['processedCalib']['diffPupilTime'] = sessionDictIn['processedCalib']['pupilTimestamp'].diff()

    gridSize_widthHeight_gridnum = list(sessionDictIn['trialInfo'].groupby([('gridSize', 'heightDegs'), 
                                                                     ('gridSize', 'widthDegs')]).count().index)

    gb_type_h_w = sessionDictIn['trialInfo'].groupby(['targetType',
                                                    ('gridSize', 'heightDegs'), 
                                                    ('gridSize', 'widthDegs')])



    dataFrameRows = []

    for eyeId in range(2):
        
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

                else:

                    numSamplesBelowConThresh = len(fixRowDataDf)
                    numSamplesBelowConThresh = np.nan
                    pctSamplesBelowConThresh = np.nan
                    numSamplesAfterConfThresh = np.nan

                ######

                meanErr = np.nanmean(fixRowDataDf[fixErrColumn]['euclidean'])
                meanGazeAz = np.nanmean(fixRowDataDf[gazeSphericalCol]['az'])
                meanGazeEl = np.nanmean(fixRowDataDf[gazeSphericalCol]['el'])
                
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
                    ('targetAllPoints','az'): fixRowDataDf[gazeSphericalCol]['az'].to_numpy(),
                    ('targetAllPoints', 'el'): fixRowDataDf[gazeSphericalCol]['el'].to_numpy()
                    }


                dataFrameRows.append(dataFrameRow)

    fixDataDF = pd.DataFrame(dataFrameRows)
    fixDataDF.columns = pd.MultiIndex.from_tuples(fixDataDF.columns)
    sessionDictIn['fixAssessmentData'] = fixDataDF
    
    logger.info('Added sessionDict[\'fixAssessmentData\']')
    
    return sessionDictIn

def plotFixAssessment(sessionDictIn, saveDir = False):

    fixDF = sessionDictIn['fixAssessmentData']

    figWidthHeight = 30
    plt.style.use('ggplot')

    gb_h_w = fixDF.groupby([('gridSize', 'heightDegs'), ('gridSize', 'widthDegs')])
    
    for eyeId in range(2):
        for (gHeight,gWidth) in list(gb_h_w.groups.keys()):

            fig, ax = plt.subplots()
            fig.set_size_inches(7, 7)
            ax.set_xlim([-figWidthHeight,figWidthHeight])
            ax.set_ylim([-figWidthHeight,figWidthHeight])

            targetLoc_targNum_AzEl = gb_h_w.get_group((gHeight,gWidth))['fixTargetSpherical'].drop_duplicates().values
            
            colors = [(1.0, 0, 0), (1.0, 0.5, 0), (0.5, 0, 0.5),
                        (1.0, 1.0, 0), (0, 1.0, 0), (0, 1.0, 1.0),
                        (1.0, 0, 1.0), (0, 0.5, 1.0), (0, 0, 1.0)]
            colorInc = 0
            
            for (tX,tY) in list(targetLoc_targNum_AzEl): 
                
                fixAtTarget = fixDF[(fixDF['eyeId']== eyeId ) & 
                                      (fixDF[('gridSize', 'heightDegs')] == gHeight ) &
                                      (fixDF[('gridSize', 'widthDegs')] == gWidth ) &
                                      (fixDF[('fixTargetSpherical','az')]== tX ) &
                                      (fixDF[('fixTargetSpherical','el')]== tY )]


                plt.text(tX, tY+2, '{:.2f}\n({:.2f})'.format(float(fixAtTarget[('accuracy', 'euclidean')]),
                                          float(fixAtTarget[('gazePrecision', '')])),ha='center',va='bottom',size=14)
    
    
                #plt.scatter(fixAtTarget[('targetAllPoints','az')].to_numpy(),
                #    fixAtTarget[('targetAllPoints','el')].to_numpy().values, c='green')
                
                #print(len(fixAtTarget[('targetAllPoints','az')].values[0]))
                #print(len(fixAtTarget[('targetAllPoints','el')].values[0]))
                cloud = plt.scatter(fixAtTarget[('targetAllPoints','az')].values[0],
                             fixAtTarget[('targetAllPoints','el')].values[0],
                             c=[colors[colorInc]], alpha=0.25, marker='.')
                colorInc += 1
                
                plt.plot([tX, np.float(fixAtTarget[('meanGazeSpherical', 'az')])], 
                         [tY, np.float(fixAtTarget[('meanGazeSpherical', 'el')])],c='grey')

                plt.errorbar(fixAtTarget[('meanGazeSpherical', 'az')],
                             fixAtTarget[('meanGazeSpherical', 'el')], 
                             xerr=fixAtTarget[('stdGazeSpherical', 'az')], 
                             yerr=fixAtTarget[('stdGazeSpherical', 'el')],c='r')

                tH = plt.scatter(targetLoc_targNum_AzEl[:,0],
                                 targetLoc_targNum_AzEl[:,1], c='blue')

                gH = plt.scatter(fixAtTarget[('meanGazeSpherical', 'az')], 
                                 fixAtTarget[('meanGazeSpherical', 'el')],c='r')

                ax.legend([gH,tH], ['gaze','target'])
                
                figTitle = '{0}_eye{1}_{2}x{3}'.format(sessionDictIn['subID'],eyeId,gHeight,gWidth)
                ax.set_title(figTitle)
                ax.set_xlabel('degrees azimuth',fontsize=15)
                ax.set_ylabel('degrees elevation',fontsize=15)
                plt.yticks(fontsize=15)
                plt.xticks(fontsize=15)


            if saveDir :
                import os
                directory = os.path.dirname(saveDir)

                try:
                    os.stat(saveDir)
                except:
                    os.makedirs(saveDir)

                plt.savefig(saveDir + figTitle + '.png')

            else:
                plt.show()
