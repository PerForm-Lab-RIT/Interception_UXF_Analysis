#!/usr/bin/env python3
# coding: utf-8

# In[1]:


import sys
sys.path.append("pyFiles/")
# sys.path.append("../")
import os

import pandas as pd
import numpy as np

import logging
import pickle

import json

import matplotlib.pyplot as plt

import evaluateSegAlgo as ev
import loadData as ld

fmt = '%(levelname)s_%(name)s-%(funcName)s(): - %(message)s'
logging.basicConfig(level=logging.INFO, format=fmt)
logger = logging.getLogger(__name__)

def calibAssessment(sessionDictIn,saveDir = 'figout/', confidenceThresh = False, title="", show_filtered_out=False, override_to_2d=False):
    #if override_to_2d:
    #    ball_catching = False
    #    if ('processedExp' in sessionDictIn) and (('ballPos','x') in sessionDictIn):
    #        ball_catching = True
    #else:
    #    # !!!!! This doesn't use the binocular model -- it averages up the monocular models!
    #    sessionDictIn = ev.calcCyclopean(sessionDictIn, 'processedCalib')
    #    try:
    #        sessionDictIn = ev.calcCyclopean(sessionDictIn, 'processedExp')
    #        ball_catching = True
    #    except KeyError:
    #        ball_catching = False
    #    sessionDictIn = ev.calcCyclopean(sessionDictIn, 'processedSequence')
    sessionDictIn = ev.calcCyclopean(sessionDictIn, 'processedCalib')
    try:
        sessionDictIn = ev.calcCyclopean(sessionDictIn, 'processedExp')
        ball_catching = True
    except KeyError:
        ball_catching = False
    sessionDictIn = ev.calcCyclopean(sessionDictIn, 'processedSequence')
    
    #print(sessionDictIn['processedCalib']['targetPos'])
    #print()
    #print(sessionDictIn['processedCalib']['targeLocalPos'])
    #print(sessionDictIn['processedSequence'].keys())
    #print('x:',sessionDictIn['processedSequence']['screen-pos', 'x'])
    #print('y:',sessionDictIn['processedSequence']['screen-pos', 'y'])
    #print('z:',sessionDictIn['processedSequence']['screen-pos', 'z'])
    #exit()
    
    sessionDictIn = ev.calcSphericalCoordinates(sessionDictIn,'targetPos','targetWorldSpherical', sessionDictKey = 'processedCalib')
    if ball_catching:
        sessionDictIn = ev.calcSphericalCoordinates(sessionDictIn,'ballPos','targetWorldSpherical', sessionDictKey = 'processedExp')
    sessionDictIn = ev.calcSphericalCoordinates(sessionDictIn,'targeLocalPos','targetLocalSpherical', sessionDictKey = 'processedCalib')
    if ball_catching:
        sessionDictIn = ev.calcSphericalCoordinates(sessionDictIn,'ballPos','targetLocalSpherical', sessionDictKey = 'processedExp')  # Should be something like ballLocalPos
    sessionDictIn = ev.calcSphericalCoordinates(sessionDictIn, 'screen-pos', 'targetLocalSpherical', sessionDictKey = 'processedSequence')
    
    #print(sessionDictIn['processedCalib']['targeLocalPos'])
    #print(sessionDictIn['processedSequence'].keys())
    #print(sessionDictIn['processedSequence'].loc[0]['screen-pos', 'x'], sessionDictIn['processedSequence'].loc[0]['screen-pos', 'y'], sessionDictIn['processedSequence'].loc[0]['screen-pos', 'z'])
    #print("to")
    #print(sessionDictIn['processedSequence'].loc[0]['targetLocalSpherical', 'az'], sessionDictIn['processedSequence'].loc[0]['targetLocalSpherical', 'el'])
    #print(sessionDictIn['processedSequence'].loc[350]['targetLocalSpherical', 'az'], sessionDictIn['processedSequence'].loc[350]['targetLocalSpherical', 'el'])
    #print("vs")
    #print(sessionDictIn['processedCalib'].loc[0]['targetLocalSpherical', 'az'], sessionDictIn['processedCalib'].loc[0]['targetLocalSpherical', 'el'])
    #exit()
    
    sessionDictIn = ev.calcSphericalCoordinates(sessionDictIn,'gaze-normal0','gaze0Spherical', sessionDictKey = 'processedCalib',flipY=True,override_to_2d=('0' if override_to_2d else None))
    if ball_catching:
        sessionDictIn = ev.calcSphericalCoordinates(sessionDictIn,'gaze-normal0','gaze0Spherical', sessionDictKey = 'processedExp',flipY=True,override_to_2d=('0' if override_to_2d else None))
    sessionDictIn = ev.calcSphericalCoordinates(sessionDictIn,'gaze-normal0','gaze0Spherical', sessionDictKey = 'processedSequence',flipY=True,override_to_2d=('0' if override_to_2d else None))
    sessionDictIn = ev.calcSphericalCoordinates(sessionDictIn,'gaze-normal1','gaze1Spherical', sessionDictKey = 'processedCalib',flipY=True,override_to_2d=('1' if override_to_2d else None))
    if ball_catching:
        sessionDictIn = ev.calcSphericalCoordinates(sessionDictIn,'gaze-normal1','gaze1Spherical', sessionDictKey = 'processedExp',flipY=True,override_to_2d=('1' if override_to_2d else None))
    sessionDictIn = ev.calcSphericalCoordinates(sessionDictIn,'gaze-normal1','gaze1Spherical', sessionDictKey = 'processedSequence',flipY=True,override_to_2d=('1' if override_to_2d else None))
    sessionDictIn = ev.calcSphericalCoordinates(sessionDictIn,'gaze-normal2','gaze2Spherical', sessionDictKey = 'processedCalib',flipY=True,override_to_2d=('' if override_to_2d else None))
    if ball_catching:
        sessionDictIn = ev.calcSphericalCoordinates(sessionDictIn,'gaze-normal2','gaze2Spherical', sessionDictKey = 'processedExp',flipY=True,override_to_2d=('' if override_to_2d else None))
    sessionDictIn = ev.calcSphericalCoordinates(sessionDictIn,'gaze-normal2','gaze2Spherical', sessionDictKey = 'processedSequence',flipY=True,override_to_2d=('' if override_to_2d else None))
    
    sessionDictIn = ev.calcTrialLevelCalibInfo(sessionDictIn, ball_catching=ball_catching)
    
    sessionDictIn = ev.calcGazeToTargetFixError(sessionDictIn,'gaze0Spherical','targetLocalSpherical','fixError_eye0', sessionDictKey='processedCalib')
    #sessionDictIn = ev.calcGazeToTargetFixError(sessionDictIn,'gaze0Spherical','targetLocalSpherical','fixError_eye0', sessionDictKey='processedExp')
    sessionDictIn = ev.calcGazeToTargetFixError(sessionDictIn,'gaze1Spherical','targetLocalSpherical','fixError_eye1', sessionDictKey='processedCalib')
    #sessionDictIn = ev.calcGazeToTargetFixError(sessionDictIn,'gaze1Spherical','targetLocalSpherical','fixError_eye1', sessionDictKey='processedExp')
    sessionDictIn = ev.calcGazeToTargetFixError(sessionDictIn,'gaze2Spherical','targetLocalSpherical','fixError_eye2', sessionDictKey='processedCalib')
    #sessionDictIn = ev.calcGazeToTargetFixError(sessionDictIn,'gaze2Spherical','targetLocalSpherical','fixError_eye2', sessionDictKey='processedExp')
    
    sessionDictIn = ev.calcGazeToCalibFixError(sessionDictIn,'gaze0Spherical','targetLocalSpherical','fixError_eye0', sessionDictKey='processedSequence')
    sessionDictIn = ev.calcGazeToCalibFixError(sessionDictIn,'gaze1Spherical','targetLocalSpherical','fixError_eye1', sessionDictKey='processedSequence')
    sessionDictIn = ev.calcGazeToCalibFixError(sessionDictIn,'gaze2Spherical','targetLocalSpherical','fixError_eye2', sessionDictKey='processedSequence')
    
    # sessionDictIn['trialInfo']['fixTargetSpherical','az'] = sessionDictIn['trialInfo']['fixTargetSpherical','az'].round(2)
    # sessionDictIn['trialInfo']['fixTargetSpherical','el'] = sessionDictIn['trialInfo']['fixTargetSpherical','el'].round(2)

    sessionDictIn = ev.calcAverageGazeDirPerTrial(sessionDictIn, ball_catching=ball_catching)
    sessionDictIn = ev.calcAverageGazeDirPerCalibTrial(sessionDictIn)
    
    sessionDictIn = ev.calcCalibrationSequenceStatistics(sessionDictIn, confidenceThresh)
    sessionDictIn = ev.calcFixationStatistics(sessionDictIn, confidenceThresh)
    
    ev.plotCalibrationSequence(sessionDictIn, saveDir, confidenceThresh=confidenceThresh, title=title, show_filtered_out=show_filtered_out)

    try:
        ev.plotFixAssessment(sessionDictIn, saveDir, confidenceThresh=confidenceThresh, title=title, show_filtered_out=show_filtered_out)
    except Exception as e:
        print(sessionDictIn.keys())
    ev.plotIndividualFix(sessionDictIn, saveDir=saveDir, confidenceThresh=confidenceThresh, title=title)

    return sessionDictIn

def processAllData(confidenceThresh=False,doNotLoad = True, saveDir = 'figOut/', targets=[], show_filtered_out=False, load_realtime_ref_data=None, override_to_2d=False):

    allSessionData = []

    dataFolderList = []
    [dataFolderList.append(name) for name in os.listdir("Data/") if ((name[0] is not '.' and (name[0:9] == "_Pipeline")))]

    for subNum, subString in enumerate(dataFolderList):
        # For each export from pupil labs
        #pupilLabsExportFolderList = ld.findPupilLabsExports(subString)

        folderDict = ld.getSubjectSubFolders(subNum)

        for expNum, exportFolderString in enumerate(folderDict['pupilExportsFolderList']):

            if len(targets) == 0 or exportFolderString in targets:
                
                #sessionDict = ld.unpackSession(subNum, exportFolderString, doNotLoad=doNotLoad)
                sessionDict = ld.unpackSession(subNum, load_realtime_ref_data, doNotLoad=doNotLoad, specificExport=exportFolderString)

                # saveDir = 'figOut/{0}_{1}/{0}_{1}'.format(subString, exportFolderString)
                currSaveDir = saveDir+subString+"/"
                try:
                    os.stat(currSaveDir)
                except:
                    os.makedirs(currSaveDir, exist_ok=True)

                sessionDict = calibAssessment(sessionDict, saveDir=currSaveDir, confidenceThresh=confidenceThresh, title=subString, show_filtered_out=show_filtered_out, override_to_2d=override_to_2d)
                allSessionData.append(sessionDict)

    return allSessionData

if __name__ == "__main__":

    allSessionData = processAllData(doNotLoad=False,confidenceThresh=0.00)

    with open('allSessionData.pickle', 'wb') as handle:
        pickle.dump(allSessionData, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # makeFixAssessmentFig(0, 40,40,saveDir = 'figOut/', figWidthHeight = 30 )

