import pandas as pd
import numpy as np
import pickle
import logging
# 
import sys
import os
wd = os.getcwd()
os.chdir(wd)
print("CWD:" + os.getcwd())

import matplotlib.pyplot as plt
import matplotlib as mpl

sys.path.append("Modules/")
fmt = '%(levelname)s_%(name)s-%(funcName)s(): - %(message)s'
logging.basicConfig(level=logging.INFO, format=fmt)
logger = logging.getLogger(__name__)

from processDataFiles import unpackSession

def calcCatchingPlane(sessionDict):

    paddleDf = pd.DataFrame()
    # paddleDf.index.name = 'frameNum'

    proc = sessionDict['processedExp']

    ########################################################
    # Calc paddle-to-ball vector

    paddleToBallVec_fr_XYZ = proc['ballPos'].values - proc['paddlePos'].values

    paddleToBallVecDf = pd.DataFrame({('paddleToBallVec', 'X'): paddleToBallVec_fr_XYZ[:, 0],
                                      ('paddleToBallVec', 'Y'): paddleToBallVec_fr_XYZ[:, 1],
                                      ('paddleToBallVec', 'Z'): paddleToBallVec_fr_XYZ[:, 2]})

    paddleDf = pd.concat([paddleDf, paddleToBallVecDf], axis=1)

    ########################################################
    # Calc paddle-to-ball direction (normalized paddle-to-ball vector)

    paddleToBallDir_fr_XYZ = np.array([np.divide(XYZ, np.linalg.norm(
        XYZ)) for XYZ in paddleToBallVec_fr_XYZ], dtype=np.float)

    paddleToBallDirDf = pd.DataFrame({('paddleToBallDir', 'X'): paddleToBallDir_fr_XYZ[:, 0],
                                      ('paddleToBallDir', 'Y'): paddleToBallDir_fr_XYZ[:, 1],
                                      ('paddleToBallDir', 'Z'): paddleToBallDir_fr_XYZ[:, 2]})

    paddleDf = pd.concat([paddleDf, paddleToBallDirDf], axis=1)

    #########################################################
    # Calc paddle-to-ball direction XZ

    paddleToBallDirXZDf = pd.DataFrame(
        [np.cross([0, 1, 0], xyz) for xyz in paddleToBallDirDf.values])

    paddleToBallVecXZDf = paddleToBallDirXZDf.rename(columns={0: (
        'paddleToBallDirXZ', 'X'), 1: ('paddleToBallDirXZ', 'Y'), 2: ('paddleToBallDirXZ', 'Z')})

    paddleToBallDirXZDf = paddleToBallVecXZDf.apply(
        lambda x: np.divide(x, np.linalg.norm(x)), axis=1)

    paddleDf = pd.concat([paddleDf, paddleToBallDirXZDf],
                         axis=1, verify_integrity=True)

    sessionDict['processedExp'] = pd.merge(sessionDict['processedExp'], paddleDf.sort_index(
        axis=1), sort=True, left_index=True, right_index=True)

    logger.info('Added sessionDict[\'processedExp\'][\'paddleToBallDir\']')
    logger.info('Added sessionDict[\'processedExp\'][\'paddleToBallVec\']')
    logger.info('Added sessionDict[\'processedExp\'][\'paddleToBallDirXZ\']')

    return sessionDict


def findLastFrame(sessionDict):

    ################################################################################

    # rawDF = sessionDict['raw']
    # procDF = sessionDict['processedExp']
    trialInfoDF = sessionDict['trialInfo']

    def findFirstZeroCrossing(vecIn):
        '''
        This will return the index of the first zero crossing of the input vector
        '''
        return np.where(np.diff(np.sign(vecIn)))[0][0]

    arriveAtPaddleFr_tr = []

    # for trNum, trialData in procDF.groupby('trialNumber'):
    #     trialResults = trialInfoDF.loc[trialInfoDF['trialNumber'] == trNum]

    for trialRowIdx, trialResults in trialInfoDF.iterrows():

        trNum = int(trialResults['trialNumber'])

        # trialInfoDF.groupby(['block_num','trial_num_in_block']).get_group((blockNum, trNum))

        ballPassesOnFr = False
        endFr = False

        if str(trialResults['trialType'].to_list()[0]) != "interception":

            endFr = np.nan
            # print("Calib: " + str(trialResults["trialNumber"].to_list()[0]) + "/" + str(trialResults['trialType'].to_list()[0]) )

        else:

            # print("Int: " + str(trialResults["trialNumber"].to_list()[0]) + "/" + str(trialResults['trialType'].to_list()[0]) )
            
            trialData = sessionDict['processedExp'].groupby(
                ['trialNumber']).get_group(trNum)

            if trialResults['isCaughtQ'].bool():

                firstFrameAfterContact = list(map(lambda i: i > float(
                    trialResults['timeOfContact']), trialData['frameTime'])).index(True)

                # print( str(trialData['frameTime'].iloc[0]) + ' - ' + str(trialData['frameTime'].iloc[-1]))
                # print( 'Collision at ' + str(trialResults['timeOfContact']))

                # If it collided with paddle, use that frame
                ballPassesOnFr = False
                endFr = firstFrameAfterContact

            else:

                # Find the first frame that the ball passed the paddle
                # Where the dot product is less than 0
                ballHitPaddleOnFr = False

                def dotBallPaddle(rowIn, ballInitialPos):
                        a = np.array(
                            rowIn['paddleToBallDir'].values, dtype=np.float)
                        b = np.array(ballInitialPos - rowIn['paddlePos'])
                        c = np.dot(a, b) / np.linalg.norm(b)
                        return c

                ballInitialPos = trialResults.filter(regex="ballInitialPos")
                dott = trialData.apply(lambda row: dotBallPaddle(
                    row, ballInitialPos.values), axis=1)
                endFr = np.where(dott < 0)[0][0]-1

        # Issue - trialinfo has more rows b/c of calibration trials
        # Sol   ution:  append 0's for calibration trials.  Maybe interate on trialinfo instead of the gb trials

        arriveAtPaddleFr_tr.append(endFr)

    # These are frames within the trial (not dataframe)
    sessionDict['trialInfo']['passVertPlaneAtPaddleFr'] = np.array(
        arriveAtPaddleFr_tr)

    logger.info(
        'Added sessionDict[\'trialInfo\'][\'passVertPlaneAtPaddleFr\']')
    return sessionDict


def calcCatchingError(sessionDict):

    ################################################################################
    # Catching error: passVertPlaneAtPaddleErr X, Y, and 2D

    ballInPaddlePlaneX_fr = []
    ballInPaddlePlaneY_fr = []
    paddleToBallDist_fr = []

    for trialRowIdx, trialResults in sessionDict['trialInfo'].iterrows():

        trNum = int(trialResults['trialNumber'])

        if str(trialResults['trialType'].to_list()[0]) != "interception":
            
            ballInPaddlePlaneX_fr.append(np.nan)
            ballInPaddlePlaneY_fr.append(np.nan)
            paddleToBallDist_fr.append(np.nan)

        else:

            
            trialData = sessionDict['processedExp'].groupby(['trialNumber']).get_group(trNum)

            fr = sessionDict['trialInfo'].groupby('trialNumber').get_group(trNum)['passVertPlaneAtPaddleFr']
            row = trialData.iloc[fr]
            bXYZ = row.ballPos.values
            pXYZ = row.paddlePos.values

            #  Ball trajectory direction
            ballTrajDir_XYZ = (trialData['ballPos'].iloc[10] - trialData['ballPos'].iloc[1]) / np.linalg.norm(trialData['ballPos'].iloc[10] - trialData['ballPos'].iloc[1])
            ballTrajDir_XYZ = np.array(ballTrajDir_XYZ,dtype=np.float)
            paddleYDir_xyz = [0,1,0]
            paddleXDir_xyz = np.cross(-ballTrajDir_XYZ,paddleYDir_xyz)
            # paddleToBallVec_fr_XYZ = trialData['ballPos'].values - trialData['paddlePos'].values

            ballRelToPaddle_xyz = np.array(bXYZ-pXYZ).T
            xErr = np.float(np.dot(paddleXDir_xyz,ballRelToPaddle_xyz))
            yErr = np.float(np.dot(paddleYDir_xyz,ballRelToPaddle_xyz))

            ballInPaddlePlaneX_fr.append(xErr)
            ballInPaddlePlaneY_fr.append(yErr)
            paddleToBallDist_fr.append(np.sqrt( np.power(xErr,2) + np.power(yErr,2) ) )


    sessionDict['trialInfo'][('catchingError','X')] = ballInPaddlePlaneX_fr
    sessionDict['trialInfo'][('catchingError','Y')] = ballInPaddlePlaneY_fr
    sessionDict['trialInfo'][('catchingError','2D')] =  paddleToBallDist_fr

    return sessionDict

def gazeAnalysisWindow(sessionDict, analyzeUntilXSToArrival =  .3, stopAtXSToArrival = 0.1):

    startFr_fr = []
    endFr_fr = []

    for trialRowIdx, trialResults in sessionDict['trialInfo'].iterrows():

        trNum = int(trialResults['trialNumber'])

        if str(trialResults['trialType'].to_list()[0]) != "interception":

            startFr = np.nan
            endFr = np.nan

        else:

            trialData = sessionDict['processedExp'].groupby(['trialNumber']).get_group(trNum)

            
            endFr = int(trialResults['passVertPlaneAtPaddleFr'])
            initTTC = trialResults[('ballInitialPos','z')] / -trialResults[('ballInitialVel','z')] 

            if stopAtXSToArrival == False:
                stopAtXSToArrival = trialResults['noExpansionLastXSeconds']

            # Find the frame on which the ttc is less than noExpansionLastXSeconds, and expansion stops
            expStopsFr = list(map(lambda i: i> initTTC-stopAtXSToArrival, np.cumsum(trialData['frameTime'][1:].diff()))).index(True)
            
            endFr = np.min([expStopsFr,endFr])

            timeToArrival_fr = trialData['frameTime'] - trialData['frameTime'].iloc[endFr]
            startFr = np.where(timeToArrival_fr>-analyzeUntilXSToArrival)[0][0]

        startFr_fr.append(startFr)
        endFr_fr.append(endFr)

    sessionDict['trialInfo']['analysisStartFr'] = startFr_fr
    sessionDict['trialInfo']['analysisEndFr'] = endFr_fr

    logger.info('Added sessionDict[\'trialInfo\'][\'analysisStartFr\']')
    logger.info('Added sessionDict[\'trialInfo\'][\'analysisEndFr\']')

    return sessionDict

def calcGIW(sessionDictIn):

    def calcGIW(rowIn):
        
        a =  rowIn['gaze-point-3d'] - rowIn['cameraPos']
        cycGIWDir_XYZ = a / np.linalg.norm(a)
        return {('cycGIWDir','x'): cycGIWDir_XYZ[0],('cycGIWDir','y'): cycGIWDir_XYZ[1],('cycGIWDir','z'): cycGIWDir_XYZ[2]}

    cycGIWDf = sessionDictIn['processedExp'].apply(lambda rowIn: calcGIW(rowIn),axis=1)
    cycGIWDf = pd.DataFrame.from_records(cycGIWDf)
    cycGIWDf.columns = pd.MultiIndex.from_tuples(cycGIWDf.columns)

    sessionDictIn['processedExp'] = sessionDictIn['processedExp'].join(cycGIWDf)

    logger.info('Added sessionDict[\'processedExp\'][\'cycGIWDir\']')
    return sessionDictIn
    
def calcCycToBallVector(sessionDict):

    cycToBallVec = np.array(sessionDict['processedExp']['ballPos'] - sessionDict['processedExp']['cameraPos'],dtype=np.float )
    cycToBallDir = np.array([np.divide(XYZ,np.linalg.norm(XYZ)) for XYZ in cycToBallVec],dtype=np.float)

    sessionDict['processedExp'][('cycToBallDir','x')] = cycToBallDir[:,0]
    sessionDict['processedExp'][('cycToBallDir','y')] = cycToBallDir[:,1]
    sessionDict['processedExp'][('cycToBallDir','z')] = cycToBallDir[:,2]

    logger.info('Added sessionDict[\'processed\'][\'cycToBallDir\']')
    return sessionDict 

def calcBallAngularSize(sessionDict):

	eyeToBallDist_fr = [np.sqrt( np.sum(  np.power(bXYZ-vXYZ,2))) for bXYZ,vXYZ in zip( sessionDict['raw']['ballPos'].values,sessionDict['raw']['viewPos'].values)]
	ballRadiusM_fr = sessionDict['raw']['ballRadiusM']
	sessionDict['processed']['ballRadiusDegs'] = [np.rad2deg(np.arctan(rad/dist)) for rad, dist in zip(ballRadiusM_fr,eyeToBallDist_fr)]

	logger.info('Added sessionDict[\'processedExp\'][\'ballRadiusDegs\']')
	return sessionDict

def saveTemp(sessionDict):
    
    with open('tempOut.pickle', 'wb') as handle:
        pickle.dump(sessionDict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def loadTemp():
    
    file = open('tempOut.pickle', 'rb')
    sessionDict = pickle.load(file)
    file.close()
    return sessionDict

if __name__ == "__main__":

    subNum = 0

    rawDataDF = False
    calibDF = False

    # sessionDict = unpackSession(subNum, doNotLoad = False)
    # sessionDict = calcCatchingPlane(sessionDict)
    # sessionDict = findLastFrame(sessionDict)
    # sessionDict = calcCatchingError(sessionDict)
    # sessionDict = gazeAnalysisWindow(sessionDict)
    # sessionDict = calcGIW(sessionDict)
    
    sessionDict = loadTemp()
    sessionDict = calcCycToBallVector(sessionDict)
    sessionDict = calcBallAngularSize(sessionDict)

    # saveTemp(sessionDict)

    logger.info('***** Done! *****')


# 
# 

# # 
# # 
# # 
# # sessionDict = calcSphericalcoordinates(sessionDict)
# # sessionDict = calcTrackingError(sessionDict)
# # sessionDict = calcTTA(sessionDict)
# # sessionDict = calcSMIDeltaT(sessionDict)
# # sessionDict = filterAndDiffSignals(sessionDict,analysisParameters)
# # sessionDict = vectorMovementModel(sessionDict,analysisParameters)
# # # sessionDict = calcCalibrationQuality(sessionDict,analysisParameters)

# # sessionDict = setIpdRatioAndPassingLoc(sessionDict)
