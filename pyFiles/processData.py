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

from loadData import unpackSession

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
    #     trialInfo = trialInfoDF.loc[trialInfoDF['trialNumber'] == trNum]

    for trialRowIdx, trialInfo in trialInfoDF.iterrows():

        trNum = int(trialInfo['trialNumber'])

        # trialInfoDF.groupby(['block_num','trial_num_in_block']).get_group((blockNum, trNum))

        ballPassesOnFr = False
        endFr = False

        if str(trialInfo['trialType'].to_list()[0]) != "interception":

            endFr = np.nan
            # print("Calib: " + str(trialInfo["trialNumber"].to_list()[0]) + "/" + str(trialInfo['trialType'].to_list()[0]) )

        else:

            # print("Int: " + str(trialInfo["trialNumber"].to_list()[0]) + "/" + str(trialInfo['trialType'].to_list()[0]) )
            
            trialData = sessionDict['processedExp'].groupby(
                ['trialNumber']).get_group(trNum)

            if trialInfo['isCaughtQ'].bool():

                firstFrameAfterContact = list(map(lambda i: i > float(
                    trialInfo['timeOfContact']), trialData['frameTime'])).index(True)

                # print( str(trialData['frameTime'].iloc[0]) + ' - ' + str(trialData['frameTime'].iloc[-1]))
                # print( 'Collision at ' + str(trialInfo['timeOfContact']))

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

                ballInitialPos = trialInfo.filter(regex="ballInitialPos")
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

    for trialRowIdx, trialInfo in sessionDict['trialInfo'].iterrows():

        trNum = int(trialInfo['trialNumber'])

        if str(trialInfo['trialType'].to_list()[0]) != "interception":
            
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

    for trialRowIdx, trialInfo in sessionDict['trialInfo'].iterrows():

        trNum = int(trialInfo['trialNumber'])

        if str(trialInfo['trialType'].to_list()[0]) != "interception":

            startFr = np.nan
            endFr = np.nan

        else:

            trialData = sessionDict['processedExp'].groupby(['trialNumber']).get_group(trNum)

            
            endFr = int(trialInfo['passVertPlaneAtPaddleFr'])
            initTTC = trialInfo[('ballInitialPos','z')] / -trialInfo[('ballInitialVel','z')] 

            if stopAtXSToArrival == False:
                stopAtXSToArrival = trialInfo['noExpansionLastXSeconds']

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

# def calcBallAngularSize(sessionDict):

# 	eyeToBallDist_fr = [np.sqrt( np.sum(  np.power(bXYZ-vXYZ,2))) for bXYZ,vXYZ in zip( sessionDict['raw']['ballPos'].values,sessionDict['raw']['viewPos'].values)]
# 	ballRadiusM_fr = sessionDict['raw']['ballRadiusM']
# 	sessionDict['processed']['ballRadiusDegs'] = [np.rad2deg(np.arctan(rad/dist)) for rad, dist in zip(ballRadiusM_fr,eyeToBallDist_fr)]

# 	logger.info('Added sessionDict[\'processedExp\'][\'ballRadiusDegs\']')
# 	return sessionDict

def calcSphericalcoordinates(sessionDict):

	proc = sessionDict['processedExp']
	sessionDict['processedExp']['cycGIW_az'] = np.rad2deg(np.arctan(proc[('cycGIWDir','x')]/proc[('cycGIWDir','z')]))
	sessionDict['processedExp']['cycGIW_el']  = np.rad2deg(np.arctan(proc[('cycGIWDir','y')]/proc[('cycGIWDir','z')]))

	sessionDict['processedExp']['ball_az'] = np.rad2deg(np.arctan(proc[('cycToBallDir','x')]/proc[('cycToBallDir','z')]))
	sessionDict['processedExp']['ball_el']  = np.rad2deg(np.arctan(proc[('cycToBallDir','y')]/proc[('cycToBallDir','z')]))

	logger.info('Added sessionDict[\'processedExp\'][\'ball_az\']')
	logger.info('Added sessionDict[\'processedExp\'][\'ball_el\']')
	logger.info('Added sessionDict[\'processedExp\'][\'cycGIW_az\']')
	logger.info('Added sessionDict[\'processedExp\'][\'cycGIW_el\']')

	return sessionDict

def calcTrackingError(sessionDict):

    meanEyeToBallEdgeAz_tr = [np.nan] * len(sessionDict['trialInfo'])
    meanEyeToBallCenterAz_tr = [np.nan] * len(sessionDict['trialInfo'])

    ballChangeInRadiusDegs_tr = [np.nan] * len(sessionDict['trialInfo'])

    meanEyeToBallEdge_tr = [np.nan] * len(sessionDict['trialInfo'])
    meanEyeToBallCenter_tr = [np.nan] * len(sessionDict['trialInfo'])
    
    for trialNum, proc in sessionDict['processedExp'].groupby('trialNumber'):

        tInfoIloc = int(sessionDict['trialInfo'][sessionDict['trialInfo']['trialNumber']==trialNum].index.tolist()[0])
        trInfo = sessionDict['trialInfo'][sessionDict['trialInfo']['trialNumber']==trialNum]

        if str(trInfo['trialType'].to_list()[0]) == "interception":
                            
            def calcEyeToBallCenter(row):
                # The difference along az and el
                azimuthalDist = row['cycGIW_az'] - row['ball_az']
                elevationDist = row['cycGIW_el'] - row['ball_el']
                return (azimuthalDist,elevationDist)

            def calcEyeToBallEdge(row):
                # The difference along az and el
                azimuthalDist = row['cycGIW_az'] - row['ball_az']
                elevationDist = row['cycGIW_el'] - row['ball_el']

                ballRadiusDegs = np.float(row['ballRadiusDegs'])

                if np.float(azimuthalDist) > 0:
                    azimuthalDist -= ballRadiusDegs
                else:
                    azimuthalDist += ballRadiusDegs

                return (azimuthalDist,elevationDist)

            #(startFr, endFr) = findAnalysisWindow(trialData)
            
            startFr = int(trInfo['analysisStartFr'])
            endFr = int(trInfo['analysisEndFr'])
            endFr = endFr-1

            absAzEl_XYZ = proc.iloc[startFr:endFr].apply(lambda row: calcEyeToBallCenter(row), axis=1)
            (azDist,elDist) = zip(*absAzEl_XYZ)

            meanEyeToBallCenterAz_tr[tInfoIloc] = np.mean(azDist)
            meanEyeToBallCenter_tr[tInfoIloc] = np.nanmean([np.sqrt(np.float(fr[0]*fr[0] + fr[1]* fr[1])) for fr in absAzEl_XYZ ])

            absAzEl_XYZ = proc.iloc[startFr:endFr].apply(lambda row: calcEyeToBallEdge(row), axis=1)
            (azDist,elDist) = zip(*absAzEl_XYZ)
            meanEyeToBallEdgeAz_tr[tInfoIloc] = np.mean(azDist)
            meanEyeToBallEdge_tr[tInfoIloc] = np.nanmean([np.sqrt(np.float(fr[0]*fr[0] + fr[1]* fr[1])) for fr in absAzEl_XYZ ])

            radiusAtStart = proc['ballRadiusDegs'].iloc[startFr]
            radiusAtEnd = proc['ballRadiusDegs'].iloc[endFr]
            ballChangeInRadiusDegs_tr[tInfoIloc] = radiusAtEnd - radiusAtStart

    ### Outside trial loop
    sessionDict['trialInfo']['meanEyeToBallCenterAz'] = meanEyeToBallCenterAz_tr
    sessionDict['trialInfo']['meanEyeToBallEdgeAz'] = meanEyeToBallEdgeAz_tr

    sessionDict['trialInfo']['meanEyeToBallCenter'] = meanEyeToBallCenter_tr
    sessionDict['trialInfo']['meanEyeToBallEdge'] = meanEyeToBallEdge_tr

    sessionDict['trialInfo']['ballChangeInRadiusDegs'] = ballChangeInRadiusDegs_tr

    logger.info('Added sessionDict[\'trialInfo\'][\'meanEyeToBallEdgeAz\']')
    logger.info('Added sessionDict[\'trialInfo\'][\'meanEyeToBallCenterAz\']')

    logger.info('Added sessionDict[\'trialInfo\'][\'meanEyeToBallEdge\']')
    logger.info('Added sessionDict[\'trialInfo\'][\'meanEyeToBallCenter\']')

    logger.info('Added sessionDict[\'trialInfo\'][\'ballChangeInRadiusDegs\']')

    return sessionDict


def calcBallAngularSize(sessionDict):

	eyeToBallDist_fr = [np.sqrt( np.sum(  np.power(bXYZ-vXYZ,2))) for bXYZ,vXYZ in 
        zip( sessionDict['processedExp']['ballPos'].values,sessionDict['processedExp']['cameraPos'].values)]

	ballRadiusM_fr = sessionDict['processedExp']['ballMeshRadius']
	sessionDict['processedExp']['ballRadiusDegs'] = [np.rad2deg(np.arctan(rad/dist)) for rad, dist in zip(ballRadiusM_fr,eyeToBallDist_fr)]

	logger.info('Added sessionDict[\'processedExp\'][\'ballRadiusDegs\']')

	return sessionDict

def filterAndDiffSignals(sessionDict):

    sgWinSizeSamples = sessionDict['analysisParameters']['sgWinSizeSamples']
    sgPolyorder = sessionDict['analysisParameters']['sgPolyorder']
    medFiltSize = sessionDict['analysisParameters']['medFiltSize']
    interpResS = sessionDict['analysisParameters']['interpResS']
    sgWinSizeSamples = sessionDict['analysisParameters']['sgWinSizeSamples']

    from scipy.signal import savgol_filter

    # FIlter
    proc = sessionDict['processedExp']

    frameDur = np.float(proc['frameTime'].diff().mode()[0])


    proc['cycGIWFilt_az'] = proc['cycGIW_az'].rolling(medFiltSize).median()
    proc['cycGIWFilt_el'] = proc['cycGIW_el'].rolling(medFiltSize).median()


    proc['cycGIWFilt_az'] = proc['cycGIW_az'].fillna(0)
    proc['cycGIWFilt_el'] = proc['cycGIW_el'].fillna(0)

    proc['cycGIWFilt_az'] = savgol_filter(proc['cycGIWFilt_az'],
                                                        sgWinSizeSamples,
                                                        sgPolyorder,
                                                        deriv=0,
                                                        delta = frameDur,
                                                        axis=0,
                                                        mode='interp')

    proc['cycGIWFilt_el'] = savgol_filter(proc['cycGIWFilt_el'],
                                                        sgWinSizeSamples,
                                                        sgPolyorder,
                                                        deriv=0,
                                                        delta = frameDur,
                                                        axis=0,
                                                        mode='interp')


    # Differentiate and save gaze velocities

    gazeVelFiltAz_fr = np.diff(np.array(proc['cycGIWFilt_az'],dtype=np.float))  / frameDur
    gazeVelFiltAz_fr = np.hstack([0 ,gazeVelFiltAz_fr])
    sessionDict['processedExp']['gazeVelFiltAz'] = gazeVelFiltAz_fr

    gazeVelFiltEl_fr = np.diff(np.array(proc['cycGIWFilt_el'],dtype=np.float)) / frameDur
    gazeVelFiltEl_fr = np.hstack([0 ,gazeVelFiltEl_fr])
    proc['gazeVelFiltEl'] = gazeVelFiltEl_fr

    proc['gazeVelFilt'] = np.sqrt(np.sum(np.power([gazeVelFiltAz_fr,gazeVelFiltEl_fr],2),axis=0))

    # Differentiate and save ball / expansion velocities

    ballVel_Az = np.diff(np.array(proc['ball_az'],dtype=np.float)) / frameDur
    ballVel_El = np.diff(np.array(proc['ball_el'],dtype=np.float)) / frameDur
    ballVel_fr = np.sqrt(np.sum(np.power([ballVel_Az,ballVel_El],2),axis=0))
    ballVel_fr = np.hstack([0 ,ballVel_fr])
    proc['ballVel2D_fr'] = ballVel_fr

    ballExpansionRate_fr = np.diff(2.*np.array(proc['ballRadiusDegs'],dtype=np.float)) / frameDur
    ballExpansionRate_fr = np.hstack([0 ,ballExpansionRate_fr])
    proc['ballExpansionRate'] = ballExpansionRate_fr

    ballVelLeadingEdge_fr = ballVel_fr + ballExpansionRate_fr
    ballVelTrailingEdge_fr = ballVel_fr - ballExpansionRate_fr

    proc['ballVelLeadingEdge'] = ballVelLeadingEdge_fr
    proc['ballVelTrailingEdge'] = ballVelTrailingEdge_fr

    proc['gazeVelRelBallEdges'] = ((proc['gazeVelFilt'] - ballVelTrailingEdge_fr) / ballVelLeadingEdge_fr)

    ballVel_az = np.diff(np.array(proc['ball_az'],dtype=np.float)) / frameDur
    proc['ballVel_az'] = np.hstack([0 ,ballVel_az])

    ballVel_el = np.diff(np.array(proc['ball_el'],dtype=np.float)) / frameDur
    proc['ballVel_el'] = np.hstack([0 ,ballVel_az])

    sessionDict['processedExp'] = proc

    return sessionDict

def vectorMovementModel( sessionDict):

    analysisParameters = sessionDict['analysisParameters']
    interpResS = analysisParameters['interpResS']
    polyorder = analysisParameters['sgPolyorder']

    ballWinStart_AzEl_tr = []
    ballWinEnd_AzEl_tr = []
    ballWinEnd_AzEl_tr
    ballAtWinEndVelPred_AzEl_tr = []
    ballAtWinEndCurvilinearVelPred_AzEl_tr = []

    ballWinVel_AzEl_tr = []
    gazeWinStart_AzEl_tr = []
    gazeWinEnd_AzEl_tr = []
    winDurSeconds_tr = []

    modelToModelDist_tr = []
    normLocInWindow_tr = []
    gazeMinDistLoc_AzEl_tr = []

    gazeToVelCenterDistDegs_tr = []
    gazeToVelEdgeDistDegs_tr = []
    gazeToBallDistDegs_tr = []
    gazeToBallEdgeDistDegs_tr = []

    balllRadiusVel_tr = []
    ballRadiusWinEnd_tr = []

    for trialRowIdx, trialInfo in sessionDict['trialInfo'].iterrows():

        trNum = int(trialInfo['trialNumber'])

        ballPassesOnFr = False
        endFr = False

        if str(trialInfo['trialType'].to_list()[0]) != "interception":

            balllRadiusVel_tr.append(np.nan)
            ballRadiusWinEnd_tr.append(np.nan)

            gazeMinDistLoc_AzEl_tr.append([np.nan, np.nan])
            normLocInWindow_tr.append(np.nan)
            modelToModelDist_tr.append(np.nan)

            # sessionDict['trialInfo'] = sessionDict['trialInfo'].sort_values(['blockNumber','trialNumber'])
            ballWinStart_AzEl_tr.append([np.nan, np.nan])
            ballWinEnd_AzEl_tr.append([np.nan, np.nan])

            ballAtWinEndVelPred_AzEl_tr.append([np.nan, np.nan])

            gazeWinStart_AzEl_tr.append([np.nan, np.nan])
            gazeWinEnd_AzEl_tr.append([np.nan, np.nan])


            # sessionDict['trialInfo']['observedError']  = np.sqrt( np.sum(np.power(gazeAzEl_tr-ballAzEl_tr,2),axis=1))
            # sessionDict['trialInfo']['velPredError']  = np.sqrt( np.sum(np.power(gazeAzEl_tr-velModelAzEl_tr,2),axis=1))

            gazeToVelCenterDistDegs_tr.append(np.nan)
            gazeToVelEdgeDistDegs_tr.append(np.nan)
            gazeToBallDistDegs_tr.append(np.nan)
            gazeToBallEdgeDistDegs_tr.append(np.nan)


        else:

            # trialInfo = sessionDict['trialInfo'].groupby(['blockNumber','trialNumber']).get_group((bNum,tNum))
            tr = sessionDict['processedExp'].groupby(['trialNumber']).get_group(trNum)

            # Calculate events
            winStartTimeMs = analysisParameters['analysisWindowStart']
            winEndTimeMs = analysisParameters['analysisWindowEnd']

            trialTime_fr = np.array(tr['frameTime'],np.float) - np.array(tr['frameTime'],np.float)[0]
            interpTime_s = np.arange(0,trialTime_fr[-1],interpResS)

            # Analysis should focus on the frames before ball collision or passing
            initTTC = float(trialInfo['ballInitialPos','z']) / -float(trialInfo['ballInitialVel','z'])
            endFrameIdx = np.where( trialTime_fr > initTTC )[0][0]
            lastTrajFrame = np.min([int(endFrameIdx),
                    int(trialInfo[('passVertPlaneAtPaddleFr', '')])])

            analysisTime_fr = np.array(tr['frameTime'],np.float)[:lastTrajFrame] - np.array(tr['frameTime'],np.float)[0]

            # Interpolate
            interpBallAz_s = np.interp(interpTime_s,analysisTime_fr,np.array(tr['ball_az'][:lastTrajFrame],dtype=np.float))
            interpBallEl_s = np.interp(interpTime_s,analysisTime_fr,np.array(tr['ball_el'][:lastTrajFrame],dtype=np.float))

            interpGazeAz_s = np.interp(interpTime_s,analysisTime_fr,np.array(tr['cycGIWFilt_az'][:lastTrajFrame],dtype=np.float))
            interpGazeEl_s = np.interp(interpTime_s,analysisTime_fr,np.array(tr['cycGIWFilt_el'][:lastTrajFrame],dtype=np.float))

            cycToBallVelAz_s = np.interp(interpTime_s,analysisTime_fr,np.array(tr['ballVel_az'][:lastTrajFrame],dtype=np.float))
            cycToBallVelEl_s = np.interp(interpTime_s,analysisTime_fr,np.array(tr['ballVel_el'][:lastTrajFrame],dtype=np.float))

            ballRadiusDegs_s = np.interp(interpTime_s,analysisTime_fr,np.array(tr['ballRadiusDegs'][:lastTrajFrame],dtype=np.float))

            # gazeVelFilt_s = np.interp(interpTime_s,analysisTime_fr,tr['gazeVelFilt'][:lastTrajFrame])

            ########################################
            #### Interpolated event times

            # Win start/end relative to initial TTC
            winStartSampleIdx = np.where( interpTime_s > initTTC + winStartTimeMs/1000.0 )[0][0]
            winEndSampleIdx = np.where( interpTime_s > initTTC + winEndTimeMs/1000.0 )[0][0] - 1

            passingTime = trialTime_fr[int(trialInfo[('passVertPlaneAtPaddleFr', '')])]
            passingSampleIdx = np.where( interpTime_s > passingTime)[0][0]

            # If passing sample idx < window end, raise error
            if( passingSampleIdx < winEndSampleIdx):
                logger.warn('Ball collision occurs within analysis window!')

            # Find where the ball / gaze actually ended up
            ballWinStart_AzEl = [interpBallAz_s[winStartSampleIdx], interpBallEl_s[winStartSampleIdx]]
            ballWinEnd_AzEl = [interpBallAz_s[winEndSampleIdx], interpBallEl_s[winEndSampleIdx]]
            gazeWinStart_AzEl = [interpGazeAz_s[winStartSampleIdx], interpGazeEl_s[winStartSampleIdx]]
            gazeWinEnd_AzEl = [interpGazeAz_s[winEndSampleIdx], interpGazeEl_s[winEndSampleIdx]]


            ########################################
            #### Constant vel model

            winDurSeconds = interpTime_s[winEndSampleIdx] - interpTime_s[winStartSampleIdx]

            ballVelAtWinStart = np.sqrt(np.sum(np.power([cycToBallVelAz_s[winStartSampleIdx],
                                            cycToBallVelEl_s[winStartSampleIdx]],2)))
            ballDegsMovement = winDurSeconds * ballVelAtWinStart
            distTrav_s = np.cumsum(np.sqrt( np.diff(interpBallAz_s)**2 + np.diff(interpBallEl_s)**2 ) )
            distTrav_s = np.hstack([0, distTrav_s])
            distRelWinStart_s = distTrav_s - distTrav_s[winStartSampleIdx]
            constVelCurvSamp =  np.where(distRelWinStart_s > ballDegsMovement)[0][0]

            ballAtWinEndVelPred_AzEl = [interpBallAz_s[constVelCurvSamp],
                                        interpBallEl_s[constVelCurvSamp]]

            ########################################
            #### Model-related metrics

            ## Cast gaze onto trajectory (min gaze-to-sample distance)
            ball_samp_AzEl = np.array([interpBallAz_s, interpBallEl_s]).T
            interpGazeMin_samp = [np.sqrt(np.sum(np.power(gazeWinEnd_AzEl - azEl,2))) for azEl in ball_samp_AzEl]
            minGazeToWinSamp = np.argmin(interpGazeMin_samp)
            gazeMinDistLoc_AzEl = np.array([interpBallAz_s[minGazeToWinSamp], interpBallEl_s[minGazeToWinSamp]])

            # Calculate distances from gaze to model locations
            gazeToConstantVelDist = np.sqrt(np.sum(np.power(ballAtWinEndVelPred_AzEl - gazeMinDistLoc_AzEl,2)))

            # Normalize distances along continuum from vel to actual
            d = distRelWinStart_s
            modelToModelDist = d[winEndSampleIdx] - d[constVelCurvSamp]
            normLocInWindow = (d[minGazeToWinSamp] - d[constVelCurvSamp] ) / (modelToModelDist)

            ###  Calculate distances to the edge of ball
            gazeToVelCenterDistDegs = np.sqrt(np.sum(np.power(np.array(ballAtWinEndVelPred_AzEl) - np.array(gazeWinEnd_AzEl),2)))
            gazeToVelEdgeDistDegs = np.sqrt(np.sum(np.power(np.array(ballAtWinEndVelPred_AzEl) - np.array(gazeWinEnd_AzEl),2))) - ballRadiusDegs_s[constVelCurvSamp]

            gazeToBallCenterDistDegs = np.sqrt(np.sum(np.power(np.array(ballWinEnd_AzEl) - np.array(gazeWinEnd_AzEl),2)))
            gazeToBallEdgeDistDegs = np.sqrt(np.sum(np.power(np.array(ballWinEnd_AzEl) - np.array(gazeWinEnd_AzEl),2))) - ballRadiusDegs_s[winEndSampleIdx]

            balllRadiusVel = ballRadiusDegs_s[constVelCurvSamp]
            ballRadiusWinEnd = ballRadiusDegs_s[winEndSampleIdx]

            ###
            ballAtWinEndVelPred_AzEl_tr.append(ballAtWinEndVelPred_AzEl)

            ballWinStart_AzEl_tr.append(ballWinStart_AzEl)
            ballWinEnd_AzEl_tr.append(ballWinEnd_AzEl)
            gazeWinStart_AzEl_tr.append(gazeWinStart_AzEl)
            gazeWinEnd_AzEl_tr.append(gazeWinEnd_AzEl)

            gazeToVelCenterDistDegs_tr.append(gazeToVelCenterDistDegs)
            gazeToVelEdgeDistDegs_tr.append(gazeToVelEdgeDistDegs)
            gazeToBallDistDegs_tr.append(gazeToBallCenterDistDegs)
            gazeToBallEdgeDistDegs_tr.append(gazeToBallEdgeDistDegs)

            balllRadiusVel_tr.append(ballRadiusDegs_s[constVelCurvSamp])
            ballRadiusWinEnd_tr.append(ballRadiusDegs_s[winEndSampleIdx])
            normLocInWindow_tr.append(normLocInWindow)
            modelToModelDist_tr.append(modelToModelDist)
            gazeMinDistLoc_AzEl_tr.append(gazeMinDistLoc_AzEl)

    sessionDict['trialInfo']['balllRadiusVel'] = balllRadiusVel_tr
    sessionDict['trialInfo']['ballRadiusWinEnd'] = ballRadiusWinEnd_tr

    sessionDict['trialInfo']['gazeMinDistLoc_AzEl'] = gazeMinDistLoc_AzEl_tr
    sessionDict['trialInfo']['normLocInWindow'] = normLocInWindow_tr
    sessionDict['trialInfo']['modelToModelDist'] = modelToModelDist_tr

    # Store vector endpoints
    sessionDict['trialInfo'] = sessionDict['trialInfo'].sort_values(['blockNumber','trialNumber'])
    sessionDict['trialInfo']['ballWinStart_AzEl'] = ballWinStart_AzEl_tr
    sessionDict['trialInfo']['ballWinEnd_AzEl'] = ballWinEnd_AzEl_tr

    sessionDict['trialInfo']['ballAtWinEndVelPred_AzEl'] = ballAtWinEndVelPred_AzEl_tr

    sessionDict['trialInfo']['gazeWinStart_AzEl'] = gazeWinStart_AzEl_tr
    sessionDict['trialInfo']['gazeWinEnd_AzEl'] = gazeWinEnd_AzEl_tr

    # Calculate distance from gaze endpoint to ball endpoint, and to constant velocity ball movement model
    ballAzEl_tr = np.array([[azEl[0],azEl[1]] for azEl in sessionDict['trialInfo']['ballWinEnd_AzEl']])
    velModelAzEl_tr = np.array([[azEl[0],azEl[1]] for azEl in sessionDict['trialInfo']['ballAtWinEndVelPred_AzEl']])
    gazeAzEl_tr = np.array([[azEl[0],azEl[1]] for azEl in sessionDict['trialInfo']['gazeWinEnd_AzEl']])

    sessionDict['trialInfo']['observedError']  = np.sqrt( np.sum(np.power(gazeAzEl_tr-ballAzEl_tr,2),axis=1))
    sessionDict['trialInfo']['velPredError']  = np.sqrt( np.sum(np.power(gazeAzEl_tr-velModelAzEl_tr,2),axis=1))

    sessionDict['trialInfo']['gazeToVelCenterDistDegs'] = gazeToVelCenterDistDegs_tr
    sessionDict['trialInfo']['gazeToVelEdgeDistDegs'] = gazeToVelEdgeDistDegs_tr
    sessionDict['trialInfo']['gazeToBallDistDegs'] = gazeToBallDistDegs_tr
    sessionDict['trialInfo']['gazeToBallEdgeDistDegs'] = gazeToBallEdgeDistDegs_tr

    # Update log
    logger.info('Added sessionDict[\'trialInfo\'][\'balllRadiusVel\']')
    logger.info('Added sessionDict[\'trialInfo\'][\'ballRadiusWinEnd\']')

    logger.info('Added sessionDict[\'trialInfo\'][\'gazeMinDistLoc_AzEl\']')
    logger.info('Added sessionDict[\'trialInfo\'][\'normLocInWindow\']')
    logger.info('Added sessionDict[\'trialInfo\'][\'modelToModelDist\']')

    logger.info('Added sessionDict[\'trialInfo\'][\'ballWinStart_AzEl\']')
    logger.info('Added sessionDict[\'trialInfo\'][\'ballWinEnd_AzEl\']')
    logger.info('Added sessionDict[\'trialInfo\'][\'ballWinVel_AzEl\']')

    logger.info('Added sessionDict[\'trialInfo\'][\'gazeWinStart_AzEl\']')
    logger.info('Added sessionDict[\'trialInfo\'][\'gazeWinEnd_AzEl\']')

    logger.info('Added sessionDict[\'trialInfo\'][\'observedError\']')
    logger.info('Added sessionDict[\'trialInfo\'][\'velPredError\']')

    logger.info('Added sessionDict[\'trialInfo\'][\'gazeToVelCenterDistDegs\']')
    logger.info('Added sessionDict[\'trialInfo\'][\'gazeToVelEdgeDistDegs\']')
    logger.info('Added sessionDict[\'trialInfo\'][\'gazeToBallDistDegs\']')
    logger.info('Added sessionDict[\'trialInfo\'][\'gazeToBallEdgeDistDegs\']')

    return sessionDict


# def calcCalibrationQuality(sessionDictIn):

# 	sessionDictIn = calcCalibrationVectors(sessionDictIn)

# 	calibDf = sessionDictIn['processedCalib']
# 	gb_tIdx = calibDf.groupby('calibrationCounter')
# 	targetList = list(calibDf.groupby('calibrationCounter').groups.keys())

# 	numTargets = len(targetList)

# 	gazePos_azEl_tIdx = np.zeros([2,numTargets])
# 	targPos_azEl_tIdx  = np.zeros([2,numTargets])
# 	calibError_tIdx = np.zeros([numTargets])
# 	stdCalibError_tIdx = np.zeros([numTargets])

# 	for targetKey, data in gb_tIdx:

# 		tIdx  = [i for i, s in enumerate(targetList) if targetKey == s]

# 		gazePos_azEl_tIdx[0,tIdx] = np.nanmean(data['cycEyeInHead_az'])
# 		gazePos_azEl_tIdx[1,tIdx] = np.nanmean(data['cycEyeInHead_el'])

# 		targPos_azEl_tIdx[0,tIdx] = np.nanmean(data['targetInHead_az'])
# 		targPos_azEl_tIdx[1,tIdx] = np.nanmean(data['targetInHead_el'])

# 		calibError_tIdx[tIdx] = np.nanmean(data['calibErr'])
# 		stdCalibError_tIdx[tIdx] = np.nanstd(data['calibErr'])

# 	sessionDictIn['calibrationData'] = pd.DataFrame(dict(gazePos_az = gazePos_azEl_tIdx[0,:],
# 				  gazePos_el = gazePos_azEl_tIdx[1,:],
# 				  targetPos_az = targPos_azEl_tIdx[0,:],
# 				  targetPos_el = targPos_azEl_tIdx[1,:],
# 				  meanCalibError = calibError_tIdx))

# 	return sessionDictIn


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

    sessionDict = unpackSession(subNum, doNotLoad = True)
    
    # Move to a json file
    sessionDict = calcCatchingPlane(sessionDict)
    sessionDict = findLastFrame(sessionDict)
    sessionDict = calcCatchingError(sessionDict)
    sessionDict = gazeAnalysisWindow(sessionDict)
    sessionDict = calcGIW(sessionDict)
    sessionDict = calcCycToBallVector(sessionDict)
    sessionDict = calcBallAngularSize(sessionDict)
    sessionDict = calcSphericalcoordinates(sessionDict)
    sessionDict = calcTrackingError(sessionDict)
    sessionDict = filterAndDiffSignals(sessionDict)
    sessionDict = vectorMovementModel(sessionDict)
    # sessionDict = calcCalibrationQuality(sessionDict,analysisParameters)

    # sessionDict = loadTemp()
    # saveTemp(sessionDict)

    
    
    


    logger.info('***** Done! *****')

