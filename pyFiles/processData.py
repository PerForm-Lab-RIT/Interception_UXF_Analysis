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



def flipGazeElevation(sessionDict):
    
    sessionDict['processedExp'][('gaze-normal0','y')] = -sessionDict['processedExp'][('gaze-normal0','y')]
    sessionDict['processedCalib'][('gaze-normal0','y')] = -sessionDict['processedCalib'][('gaze-normal0','y')]
    
    sessionDict['processedExp'][('gaze-normal1','y')] = -sessionDict['processedExp'][('gaze-normal1','y')]
    sessionDict['processedCalib'][('gaze-normal1','y')] = -sessionDict['processedCalib'][('gaze-normal1','y')]

    sessionDict['processedExp'][('gaze-point-3d','y')] = -sessionDict['processedExp'][('gaze-point-3d','y')]
    sessionDict['processedCalib'][('gaze-point-3d','y')] = -sessionDict['processedCalib'][('gaze-point-3d','y')]
    
    logger.info('Mirroring sessionDict[\'processedExp\'][\'gaze-normal0_y\']')
    logger.info('Mirroring sessionDict[\'processedCalib\'][\'gaze-normal0_y\']')
    logger.info('Mirroring sessionDict[\'processedExp\'][\'gaze-normal1_y\']')
    logger.info('Mirroring sessionDict[\'processedCalib\'][\'gaze-normal1_y\']')
    logger.info('Mirroring sessionDict[\'processedExp\'][\'gaze-point-3d_y\']')
    logger.info('Mirroring sessionDict[\'processedCalib\'][\'gaze-point-3d_y\']')
    

    return sessionDict

def calcCatchingPlane(sessionDict):

    paddleDf = pd.DataFrame()
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

    paddleToBallDir_fr_XYZ = np.array([np.divide(XYZ, np.linalg.norm( XYZ)) for XYZ in paddleToBallVec_fr_XYZ], dtype=np.float64)

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
            
            trialData = sessionDict['processedExp'].groupby(['trialNumber']).get_group(trNum)

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
                        a = np.array(rowIn['paddleToBallDir'].values, dtype=np.float64)
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
            ballTrajDir_XYZ = np.array(ballTrajDir_XYZ,dtype= np.float64)
            paddleYDir_xyz = [0,1,0]
            paddleXDir_xyz = np.cross(-ballTrajDir_XYZ,paddleYDir_xyz)
            # paddleToBallVec_fr_XYZ = trialData['ballPos'].values - trialData['paddlePos'].values

            ballRelToPaddle_xyz = np.array(bXYZ-pXYZ).T
            xErr =  np.float64(np.dot(paddleXDir_xyz,ballRelToPaddle_xyz))
            yErr =  np.float64(np.dot(paddleYDir_xyz,ballRelToPaddle_xyz))

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

# gaze_normal2_xyz = np.nanmean([sessionDict['processedExp']['gaze-normal0'], sessionDict['processedExp']['gaze-normal1']],axis=0)
# np.divide(gaze_normal2_xyz,np.linalg.norm(gaze_normal2_xyz,axis=1))



def filterGazeDataByConfidence(sessionDictIn):

    confidenceThreshold = sessionDictIn['analysisParameters']['confidenceThreshold']

    def setToNan(dataFrameIn, columnLev1,idx):

        columnLev2List = list(dataFrameIn[columnLev1].columns)

        for columnLev2 in columnLev2List:

            data = np.array(dataFrameIn[(columnLev1,columnLev2)],dtype=np.float64)
            data[idx] = np.nan
            dataFrameIn[(columnLev1,columnLev2)] = data

        return dataFrameIn

    idx = np.where(sessionDictIn['processedExp']['confidence'] < confidenceThreshold)
    setToNan(sessionDictIn['processedExp'], 'gaze-normal0', idx)
    setToNan(sessionDictIn['processedExp'], 'gaze-normal1', idx)
    
    idx = np.where(sessionDictIn['processedCalib']['confidence'] < confidenceThreshold)
    setToNan(sessionDictIn['processedCalib'], 'gaze-normal0', idx)
    setToNan(sessionDictIn['processedCalib'], 'gaze-normal1', idx)
    
    logger.info('Filtered sessionDict[\'processedExp\'][\'gaze-normal0\'] by confidence threshold of ' + str(confidenceThreshold))
    logger.info('Filtered sessionDict[\'processedExp\'][\'gaze-normal1\'] by confidence threshold of ' + str(confidenceThreshold))
    logger.info('Filtered sessionDict[\'processedCalib\'][\'gaze-normal0\'] by confidence threshold of ' + str(confidenceThreshold))
    logger.info('Filtered sessionDict[\'processedCalib\'][\'gaze-normal1\'] confidence threshold of ' + str(confidenceThreshold))
    
    return sessionDictIn


def calc_gaze_Normal2(sessionDictIn):
    
    '''
    For each row, applies nanmean to gaze-normal0 and gaze-normal1 to calculate gaze-normal2.
    gaze-normal0 is normalized.
    This means that the gaze vector may switch from monocular to the binocular avg on a per-frame basis.
    '''

        
    def avgMonoGaze_ByRow(rowIn):

        def checkNans(vecIn):

            numNans = np.sum(np.isnan(np.array(vecIn,dtype=np.float64)))

            if numNans == 3:
                return True
            else: 
                return False

        if( checkNans(rowIn['gaze-normal0']) & checkNans(rowIn['gaze-normal1']) ):

            return {('gaze_normal2','x'): np.nan,('gaze_normal2','y'): np.nan,('gaze_normal2','z'): np.nan}

        else:

            xyz = np.nanmean([rowIn['gaze-normal0'], rowIn['gaze-normal1']],axis=0)
            xyz = xyz / np.linalg.norm(xyz)

            return {('gaze_normal2','x'): xyz[0],('gaze_normal2','y'): xyz[1],('gaze_normal2','z'): xyz[2]}
    
    def avgGazeForDataFrame(sessionDictIn, dataFrameKey):
        # If the column already exists, remove it and recalculate.
        if 'gaze_normal2' in sessionDictIn[dataFrameKey].columns:
            sessionDictIn[dataFrameKey].drop("gaze_normal2", axis=1, level=0,inplace=True)

        # Average per row
        dictListOut = sessionDictIn[dataFrameKey].apply(lambda row: avgMonoGaze_ByRow(row),axis=1)

        # Convert to dataframe and join with sessionDict
        dfOut = pd.DataFrame.from_records(dictListOut)
        dfOut.columns = pd.MultiIndex.from_tuples(dfOut.columns)
        sessionDictIn[dataFrameKey] = sessionDictIn[dataFrameKey].join(dfOut)
        
        return sessionDictIn
        
    
    sessionDictIn = avgGazeForDataFrame(sessionDictIn,'processedExp')
    sessionDictIn = avgGazeForDataFrame(sessionDictIn,'processedCalib')
    
    return sessionDictIn


def calcCycGIWDir(sessionDictIn):

    def calcGIW_ByRow(rowIn):
        # Grab gransformation matrix
        headTransform_4x4 = np.reshape(rowIn['camera'].values,[4,4])

        # Grab cyc EIH direction
        cycEyeInHead_XYZ = rowIn['gaze_normal2']

        # Add a 1 to convert to homogeneous coordinates
        cycEyeInHead_XYZW = np.hstack( [cycEyeInHead_XYZ,1])

        # Take the dot product!
        cycGIWVec_XYZW = np.dot( headTransform_4x4,cycEyeInHead_XYZW)

        # # Now, convert into a unit direction vector from the cyclopean eye in world coordinates
        # # Also, we can discard the w term
        cycGIWDir_XYZ = (cycGIWVec_XYZW[0:3]-rowIn["cameraPos"]) / np.linalg.norm((cycGIWVec_XYZW[0:3]-rowIn["cameraPos"]))

        # # You must return as a list or a tuple
        # #return list(cycGIWDir_XYZ)
        return {('cycGIWDir','x'): cycGIWDir_XYZ[0],('cycGIWDir','y'): cycGIWDir_XYZ[1],('cycGIWDir','z'): cycGIWDir_XYZ[2]}
    
    def calcGIWForDataframe(sessionDictIn, dataFrameKey):

        # If the column already exists, remove it and recalculate.    
        if 'cycGIWDir' in sessionDictIn[dataFrameKey].columns:
            sessionDictIn[dataFrameKey].drop("cycGIWDir", axis=1, level=0,inplace=True)

        # Average per row
        dictListOut = sessionDictIn[dataFrameKey].apply(lambda row: calcGIW_ByRow(row),axis=1)

        # Convert to dataframe and join with sessionDict
        dfOut = pd.DataFrame.from_records(dictListOut)
        dfOut.columns = pd.MultiIndex.from_tuples(dfOut.columns)
        sessionDictIn[dataFrameKey] = sessionDictIn[dataFrameKey].join(dfOut)
        return sessionDictIn
    
    sessionDictIn = calcGIWForDataframe(sessionDictIn, 'processedExp')
    sessionDictIn = calcGIWForDataframe(sessionDictIn, 'processedCalib')
    
    return sessionDictIn


def calcCycToBallVector(sessionDict):

    cycToBallVec = np.array(sessionDict['processedExp']['ballPos'] - sessionDict['processedExp']['cameraPos'],dtype= np.float64 )
    cycToBallDir = np.array([np.divide(XYZ,np.linalg.norm(XYZ)) for XYZ in cycToBallVec],dtype= np.float64)

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

    def calcTargetAzEl(row):
    
        x = row['targeLocalPos','x']
        y = row['targeLocalPos','y']
        z = row['targeLocalPos','z']
        
        az = np.rad2deg(np.arctan(np.divide(x,z)))
        el = np.rad2deg(np.arctan(np.divide(y,z)))
        
        #Note that the .apply requires you return a single data structure, 
        #so a SINGLE tuple is OK, but the seperate values for az/el are not OK.
        
        return (az,el) 
    
    # first, targetInHead_az and targetInHead_el

    proc = sessionDict['processedCalib']
    ballAzEl = proc.apply(lambda arbitraryRowName: calcTargetAzEl(arbitraryRowName),axis=1)
    proc['targetInHead_az'],proc['targetInHead_el'] = zip(*ballAzEl)

    # Now, target az and el in the world frame

    cycToTargetVec = np.array(proc['targetPos'] - proc['cameraPos'],dtype= np.float64 )
    cycToTargetDir = np.array([np.divide(XYZ,np.linalg.norm(XYZ)) for XYZ in cycToTargetVec],dtype= np.float64)

    proc['targetInWorld_az'] = np.rad2deg(np.arctan(cycToTargetDir[:,0]/cycToTargetDir[:,2]))
    proc['targetInWorld_el'] = np.rad2deg(np.arctan(cycToTargetDir[:,1]/cycToTargetDir[:,2]))

    logger.info('Added sessionDict[\'processedCalib\'][\'targetInHead_az\']')
    logger.info('Added sessionDict[\'processedCalib\'][\'targetInHead_el\']')
    logger.info('Added sessionDict[\'processedCalib\'][\'targetInWorld_az\']')
    logger.info('Added sessionDict[\'processedCalib\'][\'targetInWorld_el\']')

    sessionDict['processedCalib'] = proc

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

                ballRadiusDegs = np.float64(row['ballRadiusDegs'])

                if( np.float64(azimuthalDist) > 0):
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
            meanEyeToBallCenter_tr[tInfoIloc] = np.nanmean([np.sqrt( np.float64(fr[0]*fr[0] + fr[1]* fr[1])) for fr in absAzEl_XYZ ])

            absAzEl_XYZ = proc.iloc[startFr:endFr].apply(lambda row: calcEyeToBallEdge(row), axis=1)
            (azDist,elDist) = zip(*absAzEl_XYZ)
            meanEyeToBallEdgeAz_tr[tInfoIloc] = np.mean(azDist)
            meanEyeToBallEdge_tr[tInfoIloc] = np.nanmean([np.sqrt( np.float64(fr[0]*fr[0] + fr[1]* fr[1])) for fr in absAzEl_XYZ ])

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

    frameDur =  np.float64(proc['frameTime'].diff().mode()[0])


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

    gazeVelFiltAz_fr = np.diff(np.array(proc['cycGIWFilt_az'],dtype= np.float64)) 
    gazeVelFiltAz_fr = np.hstack([0 ,gazeVelFiltAz_fr])  / frameDur
    sessionDict['processedExp']['gazeVelFiltAz'] = gazeVelFiltAz_fr

    gazeVelFiltEl_fr = np.diff(np.array(proc['cycGIWFilt_el'],dtype= np.float64))
    gazeVelFiltEl_fr = np.hstack([0 ,gazeVelFiltEl_fr])  / frameDur
    proc['gazeVelFiltEl'] = gazeVelFiltEl_fr

    proc['gazeVelFilt'] = np.sqrt(np.sum(np.power([gazeVelFiltAz_fr,gazeVelFiltEl_fr],2),axis=0))

    # Differentiate and save ball / expansion velocities

    ballVel_Az = np.diff(np.array(proc['ball_az'],dtype= np.float64)) 
    ballVel_El = np.diff(np.array(proc['ball_el'],dtype= np.float64)) 
    ballVel_fr = np.sqrt(np.sum(np.power([ballVel_Az,ballVel_El],2),axis=0))  
    ballVel_fr = np.hstack([0 ,ballVel_fr])  / proc['frameTime'].diff()
    proc['ballVel2D_fr'] = ballVel_fr

    ballExpansionRate_fr = np.diff(2.*np.array(proc['ballRadiusDegs'],dtype= np.float64)) 
    ballExpansionRate_fr = np.hstack([0 ,ballExpansionRate_fr])  / proc['frameTime'].diff()
    proc['ballExpansionRate'] = ballExpansionRate_fr

    ballVelLeadingEdge_fr = ballVel_fr + ballExpansionRate_fr
    ballVelTrailingEdge_fr = ballVel_fr - ballExpansionRate_fr

    proc['ballVelLeadingEdge'] = ballVelLeadingEdge_fr
    proc['ballVelTrailingEdge'] = ballVelTrailingEdge_fr

    proc['gazeVelRelBallEdges'] = ((proc['gazeVelFilt'] - ballVelTrailingEdge_fr) / ballVelLeadingEdge_fr)

    ballVel_az = np.diff(np.array(proc['ball_az'],dtype= np.float64))
    proc['ballVel_az'] = np.hstack([0 ,ballVel_az])  / proc['frameTime'].diff()

    ballVel_el = np.diff(np.array(proc['ball_el'],dtype= np.float64))
    proc['ballVel_el'] = np.hstack([0 ,ballVel_az])  / proc['frameTime'].diff()

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

            trialTime_fr = np.array(tr['frameTime'], np.float64) - np.array(tr['frameTime'], np.float64)[0]
            interpTime_s = np.arange(0,trialTime_fr[-1],interpResS)

            # Analysis should focus on the frames before ball collision or passing
            initTTC = float(trialInfo['ballInitialPos','z']) / -float(trialInfo['ballInitialVel','z'])
            endFrameIdx = np.where( trialTime_fr > initTTC )[0][0]
            lastTrajFrame = np.min([int(endFrameIdx),
                    int(trialInfo[('passVertPlaneAtPaddleFr', '')])])

            analysisTime_fr = np.array(tr['frameTime'], np.float64)[:lastTrajFrame] - np.array(tr['frameTime'], np.float64)[0]

            # Interpolate
            interpBallAz_s = np.interp(interpTime_s,analysisTime_fr,np.array(tr['ball_az'][:lastTrajFrame],dtype= np.float64))
            interpBallEl_s = np.interp(interpTime_s,analysisTime_fr,np.array(tr['ball_el'][:lastTrajFrame],dtype= np.float64))

            interpGazeAz_s = np.interp(interpTime_s,analysisTime_fr,np.array(tr['cycGIWFilt_az'][:lastTrajFrame],dtype= np.float64))
            interpGazeEl_s = np.interp(interpTime_s,analysisTime_fr,np.array(tr['cycGIWFilt_el'][:lastTrajFrame],dtype= np.float64))

            cycToBallVelAz_s = np.interp(interpTime_s,analysisTime_fr,np.array(tr['ballVel_az'][:lastTrajFrame],dtype= np.float64))
            cycToBallVelEl_s = np.interp(interpTime_s,analysisTime_fr,np.array(tr['ballVel_el'][:lastTrajFrame],dtype= np.float64))

            ballRadiusDegs_s = np.interp(interpTime_s,analysisTime_fr,np.array(tr['ballRadiusDegs'][:lastTrajFrame],dtype= np.float64))

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

def plotMovementModel(tr,
                   trInfo,
                      analysisParameters,
                  halfHFOVDegs = 80,
                     figSize = [7,7]):
    
    import matplotlib.pyplot as plt
    
    p = plt.figure(figsize=(figSize))

    grid = plt.GridSpec(3, 2, wspace=0.4, hspace=0.3)
    ax = p.add_subplot(grid[:2,:2])
    ax2 = p.add_subplot(grid[2,:])
    
    ax.set(xlabel='degrees azimuth', ylabel='degrees elevation')
    ax2.set(xlabel='time (s)', ylabel='velocity (degrees)')
    
    #######
    
    # Calculate events 
    winStartTimeMs = analysisParameters['analysisWindowStart']
    winEndTimeMs = analysisParameters['analysisWindowEnd']
    interpResS = analysisParameters['interpResS']
    
    trialTime_fr = np.array(tr['frameTime'],np.float64) - np.array(tr['frameTime'],np.float64)[0]
    interpTime_s = np.arange(0,trialTime_fr[-1],interpResS)

    # Analysis should focus on the frames before ball collision or passing
    initTTC = np.float64(trInfo['ballInitialPos','z']) / -np.float64(trInfo['ballInitialVel','z'])
    endFrameIdx = np.where( trialTime_fr > initTTC )[0][0]
    lastTrajFrame = np.min([int(endFrameIdx),
               int(trInfo[('passVertPlaneAtPaddleFr', '')])])

    analysisTime_fr = np.array(tr['frameTime'],np.float64)[:lastTrajFrame] - np.array(tr['frameTime'],np.float64)[0]

    # Interpolate

    interpBallAz_s = np.interp(interpTime_s,analysisTime_fr,np.array(tr['ball_az'][:lastTrajFrame],dtype=np.float64))
    interpBallEl_s = np.interp(interpTime_s,analysisTime_fr,np.array(tr['ball_el'][:lastTrajFrame],dtype=np.float64))

    interpGazeAz_s = np.interp(interpTime_s,analysisTime_fr,np.array(tr['cycGIWFilt_az'][:lastTrajFrame],dtype=np.float64))
    interpGazeEl_s = np.interp(interpTime_s,analysisTime_fr,np.array(tr['cycGIWFilt_el'][:lastTrajFrame],dtype=np.float64))

    cycToBallVelAz_s = np.interp(interpTime_s,analysisTime_fr,np.array(tr['ballVel_az'][:lastTrajFrame],dtype=np.float64))
    cycToBallVelEl_s = np.interp(interpTime_s,analysisTime_fr,np.array(tr['ballVel_el'][:lastTrajFrame],dtype=np.float64))

    ballRadiusDegs_s = np.interp(interpTime_s,analysisTime_fr,np.array(tr['ballRadiusDegs'][:lastTrajFrame],dtype=np.float64))

    gazeVelFilt_s = np.interp(interpTime_s,analysisTime_fr,tr['gazeVelFilt'][:lastTrajFrame])

    ########################################
    #### Interpolated event times

    # Win start/end relative to initial TTC
    winStartSampleIdx = np.where( interpTime_s > initTTC + winStartTimeMs/1000.0 )[0][0]
    winEndSampleIdx = np.where( interpTime_s > initTTC + winEndTimeMs/1000.0 )[0][0] - 1

    passingTime = trialTime_fr[int(trInfo[('passVertPlaneAtPaddleFr', '')])]
    passingSampleIdx = np.where( interpTime_s > passingTime)[0][0]

    # If passing sample idx < window end, raise error
    if( passingSampleIdx < winEndSampleIdx):
        logger.warn('Ball collision occurs within analysis window!')

    # Find where the ball / gaze actually ended up
    ballWinStart_AzEl = [interpBallAz_s[winStartSampleIdx], interpBallEl_s[winStartSampleIdx]]
    ballWinEnd_AzEl = [interpBallAz_s[winEndSampleIdx], interpBallEl_s[winEndSampleIdx]]
    gazeWinStart_AzEl = [interpGazeAz_s[winStartSampleIdx], interpGazeEl_s[winStartSampleIdx]]
    gazeWinEnd_AzEl = [interpGazeAz_s[winEndSampleIdx], interpGazeEl_s[winEndSampleIdx]]
    
    
    #######
    
#     initTTC = np.float64(trInfo['ballInitialPos','Z']) / -np.float64(trInfo['ballInitialVel','Z'])
    endSampleIdx = np.where( interpTime_s > initTTC )[0][0]-1

    # Win start/end
    winStartSampleIdx = np.where( interpTime_s > initTTC + winStartTimeMs/1000.0 )[0][0]
    winEndSampleIdx = np.where( interpTime_s > initTTC + winEndTimeMs/1000.0 )[0][0] -1

    if( passingSampleIdx < winEndSampleIdx):  
        winEndSampleIdx = passingSampleIdx

    windowFr = np.arange(winStartSampleIdx,winEndSampleIdx)

    ############

    halfVFOVDegs = halfHFOVDegs / 1.77


    cList = ['r','g','b']
    lineHandles = []

    ballH = ax.plot(interpBallAz_s[:endSampleIdx],interpBallEl_s[:endSampleIdx],color='b',linewidth=3,alpha = 0.4)
    gazeH = ax.plot(interpGazeAz_s[:endSampleIdx],interpGazeEl_s[:endSampleIdx],color='r',linewidth=3,alpha = 0.4)

    from matplotlib import patches as pt

    # ax.add_patch(pt.Circle(ballAtWinEndVelPred_AzEl,radius=balllRadiusVel,
    #             fill=False,facecolor=None,ec='k',lw=3))

    ax.add_patch(pt.Circle(ballWinEnd_AzEl,radius=trInfo['ballRadiusWinEnd'],
                 fill=False,facecolor=None,ec='k',lw=3))

    ax.plot(interpBallAz_s[windowFr],interpBallEl_s[windowFr],color='b',linewidth=5, alpha = 0.6)
    ax.plot(interpGazeAz_s[windowFr],interpGazeEl_s[windowFr],color='r',linewidth=5,alpha = 0.6)

    for i in np.arange(0,len(windowFr),5):
        pf = windowFr[i]
        xs = [interpBallAz_s[pf], interpGazeAz_s[pf]]
        ys = [interpBallEl_s[pf], interpGazeEl_s[pf]]
        ax.plot(xs,ys,color='k',linewidth=1,alpha = 0.3)

    cOrM = []
    if (trInfo['isCaughtQ'].values == True):

        cOrM = ax.scatter(tr['ball_az'].iloc[lastTrajFrame-1],
                          tr['ball_el'].iloc[lastTrajFrame-1],
                          c='g',s=120,marker='8',lw=6) 
    else:

        cOrM = ax.scatter(tr['ball_az'].iloc[lastTrajFrame-1],
                          tr['ball_el'].iloc[lastTrajFrame-1],
                          c='r',s=150,marker='8',lw=6)
        
    ax.axis('equal')
    ax.axes.spines['top'].set_visible(False)
    ax.axes.spines['right'].set_visible(False)
    ax.axes.yaxis.grid(True)
    ax.axes.xaxis.grid(True)
    p.set_facecolor('w')

    plt.xlim([-30,30])
    plt.ylim([-15,35])

    observedH = ax.scatter(ballWinEnd_AzEl[0],ballWinEnd_AzEl[1],c='k',s=150,marker='8')
    constantVelH = ax.scatter(trInfo['ballAtWinEndVelPred_AzEl'].values[0][0],
                              trInfo['ballAtWinEndVelPred_AzEl'].values[0][1],c='k',s=150,marker='v')
    
    gazeLoc = ax.scatter(trInfo['gazeMinDistLoc_AzEl'].values[0][0],
                         trInfo['gazeMinDistLoc_AzEl'].values[0][1],c='m',s=150,marker='x',lw=6)

    ax.text(.01,.01,str('NormLoc: {:.2}').format(trInfo['normLocInWindow'].values[0]),transform=ax.transAxes)
#     ax.text(.01,.04,str('Expansion gain: {}').format(np.float64(trInfo['expansionGain'].values)),transform=ax.transAxes)
    
    ax.text(.01,.07,str('Bl: {} Tr: {}').format(
        int(trInfo['blockNumber']),
        int(trInfo['trialNumber'])
    ),transform=ax.transAxes)
    
    ax.legend([gazeLoc,
               constantVelH,
               observedH,
               cOrM], 

              ['point nearest to gaze',
               'constant speed model',
               'actual displacement',
              'green=catch, red=miss'])
    

    #######################################################
    #######################################################
    ## Velocity
    
    trialTime_fr = np.array(tr['frameTime'] - tr['frameTime'].iloc[0])
    initTTC = np.float64(trInfo['ballInitialPos','z']) / -np.float64(trInfo['ballInitialVel','z'])
    winStartTimeMs = analysisParameters['analysisWindowStart']
    winEndTimeMs = analysisParameters['analysisWindowEnd']
    winStartFrameIdx = np.where( trialTime_fr > initTTC + winStartTimeMs/1000.0 )[0][0]
    winEndFrameIdx = np.where( trialTime_fr > initTTC + winEndTimeMs/1000.0 )[0][0] -1


    
    ax2.set_ylim([-100,300])
    ax2.set_xlim(trialTime_fr[winStartFrameIdx],trialTime_fr[winEndFrameIdx])
    
    gaze = ax2.plot(trialTime_fr,
            tr['gazeVelFilt']
            ,color='r',linewidth=3,alpha = .5,label='gaze')

    ballCenter = ax2.plot(trialTime_fr,
            tr['ballVel2D_fr']
            ,color='b',linewidth=3,alpha = 0.5,label='ball center')

    ballLeading = ax2.plot(trialTime_fr,
            tr['ballVelLeadingEdge']
            ,color='k',linewidth=3,alpha = 0.4,label='ball leading')

    ballTrailing = ax2.plot(trialTime_fr,
            tr['ballVelTrailingEdge']
            ,color='k',linewidth=3,alpha = 0.4,label='ball trailing')

    ax3 = ax2.twinx()  # instantiate a second axes that shares the same x-axis
    ax3.set_ylim([-1,1])
    ratio = ax3.plot(trialTime_fr,
            tr['gazeVelRelBallEdges']
            ,color='c',linewidth=3,alpha = 0.4,label='ratio')
    
    ax3.axhline( .5 )
    ax3.axhline( 0 )

    ax2.legend()
    ax3.legend()
    
    
    
    return(p,ax)



def saveOutVectorMovementModel(sessionDict):
    
    analysisParameters = sessionDict['analysisParameters']

    gbData_bl_tr = sessionDict['processedExp'].groupby(['blockNumber','trialNumber']) 
    gbInfo_bl_tr = sessionDict['trialInfo'].groupby(['blockNumber','trialNumber']) 


    for idx, trDataGb in gbData_bl_tr:

        blNum = np.unique(trDataGb['blockNumber'])[0]
        trNum = np.unique(trDataGb['trialNumber'])[0]

        trInfoGb = gbInfo_bl_tr.get_group((blNum,trNum)) 

        (p,ax) = plotMovementModel(trDataGb,
                      trInfoGb,
                      sessionDict['analysisParameters'],
                      halfHFOVDegs = 80,
                      figSize = [7,7])

    #     condStr = 'g-{:.1f}_pd-{:1.1f}'.format(float(trInfo['expansionGain']),float(trInfo['passingLocX']))

        condStr = 'g-{:.1f}_pd-{:1.1f}'.format(np.float64(111),np.float64(222))
        outDir = 'Figures/Projections/' + condStr + '/' + sessionDict['subID'] + '/'

        if not os.path.exists(outDir):
            os.makedirs(outDir)

        fName = str(outDir + 'b-{}_t-{}_' + condStr + '.png').format(blNum,trNum)

        plt.savefig(fName)
        plt.close()

    
    return sessionDict


def runCalibrationAssessment(sessionDictIn):

    blockDistList = []
    accuracy_tr = []
    prec_tr = []
    pctDropouts_tr = []
    numDropouts_tr = []
    azErr_tr = []
    elErr_tr = []
    targLocAz_tr = []
    targLocEl_tr = []
    meanGazeAz_tr = []
    meanGazeEl_tr = []

    calibProc_gbBlock_Trial = sessionDictIn['processedCalib'].groupby(['blockNumber','trialNumber'])

    ############################################################
    ############################################################
    ## Iterate through assessment block

    trInfo_gbBlock = sessionDictIn['trialInfo'].groupby(['blockNumber'])
    # trInfo_gbBlock.get_group(1) # temp


    for idx, trInfoInBlock in trInfo_gbBlock:

    # trInfoInBlock = trialInfo_gbBlock.get_group(1)  # temp

        blNum = int(trInfoInBlock.iloc[0]['blockNumber'])

        if( np.unique(trInfoInBlock['trialType'])[0] != "CalibrationAssessment" ):

            # these are not calibration trials, so fill the lists with nans

            nanList_tr =  np.empty((1,len(trInfoInBlock)))[0]
            nanList_tr[:] = np.nan

            accuracy_tr.extend(nanList_tr)
            prec_tr.extend(nanList_tr)
            pctDropouts_tr.extend(nanList_tr)
            numDropouts_tr.extend(nanList_tr)
            azErr_tr.extend(nanList_tr)
            elErr_tr.extend(nanList_tr)
            targLocAz_tr.extend(nanList_tr)
            targLocEl_tr.extend(nanList_tr)
            meanGazeEl_tr.extend(nanList_tr)
            meanGazeAz_tr.extend(nanList_tr)

        else: 

            # calibTrData = gbCalib.get_group((int(assTrial['blockNumber']), int(assTrial['trialNumber'])))

            ## Check for assessment trials that were strangely far apart in time
            startTimes = trInfoInBlock['startTime']
            timeBetweenFixations = startTimes.diff().values

            if(any(timeBetweenFixations > sessionDictIn['analysisParameters']['warnIfAssessmentTrialsNSecondsApart'])):
                logger.warning('Note that assessment trials were spaced more than the duration threshold specified by sessionDict[\'analysisParameters\'][\'warnIfAssessmentTrialsNSecondsApart\'].')

            ############################################################
            ############################################################
            ## Iterate through each trial in the block
            for trIdx, thisTrInfo in trInfoInBlock.iterrows():

                trNum = thisTrInfo['trialNumber'].iloc[0]

                # Get rows of this trial from sessionDict['processedCalib']
                tr = calibProc_gbBlock_Trial.get_group((blNum, trNum))

                targetAz_fr = np.unique(tr['targetInHead_az'])
                targetEl_fr = np.unique(tr['targetInHead_el'])

                targLocAz_tr.append(targetAz_fr)
                targLocEl_tr.append(targetEl_fr)

                ## Check for multiple target locations within a single trial
                if ( len(np.unique(targetAz_fr)) > 1 or len(np.unique(targetAz_fr)) > 1 ):
                    logger.error('A single assessment trial has more than one target location.')

                # Robustness
                numDropouts = np.sum(tr.apply(lambda row: np.sum( row['gaze_normal2'].isnull()) > 0 ,axis=1)) - 1
                numDropouts_tr.append(numDropouts)

                # -1 because the first  frame is always nans
                pctDropouts = numDropouts  / len(tr)
                pctDropouts_tr.append(pctDropouts)

                # Mean gaze loc
                gazeAz_fr = np.rad2deg(np.arctan2(tr['gaze_normal2']['x'], tr['gaze_normal2']['z'] ))
                gazeEl_fr = np.rad2deg(np.arctan2(tr['gaze_normal2']['y'], tr['gaze_normal2']['z'] ))

                meanGazeAz_tr.append(np.nanmean(gazeAz_fr))
                meanGazeEl_tr.append(np.nanmean(gazeEl_fr))
                
                # Accuracy
                acc_fr = np.sqrt((gazeAz_fr - targetAz_fr)**2 + (gazeEl_fr - targetEl_fr)**2)
                accuracy_tr.append(np.nanmean(acc_fr))

                # Precision
                prec_fr = np.sqrt( (np.nanmean(gazeAz_fr) - gazeAz_fr)**2 + (np.nanmean(gazeEl_fr) - gazeEl_fr)**2)
                prec_tr.append(np.nanmean(prec_fr))

                # Err in Y and X direction
                azErr_tr.append(np.nanstd(gazeAz_fr))
                elErr_tr.append(np.nanstd(gazeEl_fr))

                ############################################################
                ## End trial loop

            blockAsessmentDict = {
                "blockNum": blNum,
                "meanAcc": np.nanmean(accuracy_tr),
                "minAcc": np.min(accuracy_tr),
                "maxAcc": np.max(accuracy_tr),
                "meanPrec": np.nanmean(prec_tr),
                "minPrec": np.min(prec_tr),
                "maxPrec": np.max(prec_tr),
                "totalDropouts": np.sum(numDropouts_tr),
                "totalNumFrames": len(sessionDictIn['processedCalib'].groupby(['blockNumber']).get_group(blNum)),
                "pctDropouts": np.nanmean(pctDropouts_tr),
                "gridWidthDegs": tr['azimuthWidth'].iloc[0],
                "gridHeightDegs": tr['elevationHeight'].iloc[0]


            }


            blockDistList.append(blockAsessmentDict)

            ############################################################
            ## End block loop


    assessmentDf = pd.DataFrame.from_records(blockDistList)
    sessionDictIn['calibrationQuality'] = assessmentDf

    sessionDictIn['trialInfo']['accuracy'] = accuracy_tr
    sessionDictIn['trialInfo']['precision'] = prec_tr
    sessionDictIn['trialInfo']['pctDropouts'] = pctDropouts_tr
    sessionDictIn['trialInfo']['numDropouts'] = numDropouts_tr
    sessionDictIn['trialInfo'][('assessmentErr','az')] = azErr_tr
    sessionDictIn['trialInfo'][('assessmentErr','el')] = elErr_tr
    sessionDictIn['trialInfo'][('targetSphericalPosInHead','az')] = targLocAz_tr
    sessionDictIn['trialInfo'][('targetSphericalPosInHead','el')] = targLocEl_tr
    sessionDictIn['trialInfo'][('gazeSphericalPosInHead','az')] = meanGazeAz_tr
    sessionDictIn['trialInfo'][('gazeSphericalPosInHead','el')] = meanGazeEl_tr


    return sessionDictIn

def plotTrackQuality(sessionDictIn):
    
    cList = ['r','g','b']

    offsets = np.linspace(-.01,.01,3)

    calibProc_gbBlock_Trial = sessionDictIn['processedCalib'].groupby(['blockNumber','trialNumber'])
    trInfo_gbBlock = sessionDictIn['trialInfo'].groupby(['blockNumber'])

    # Iterate through blocks
    for idx, trInfoInBlock in trInfo_gbBlock:

        p, ax = plt.subplots(1, 1) #sharey=True)
        p.set_size_inches(8,8)
        lineHandles = []

        blNum = int(trInfoInBlock.iloc[0]['blockNumber'])

        # Iterate through rows
        if( np.unique(trInfoInBlock['trialType'])[0] == "CalibrationAssessment" ):
            
            for trIdx, thisTrInfo in trInfoInBlock.iterrows():

                trNum = thisTrInfo['trialNumber'].iloc[0]

                # Get rows of this trial from sessionDict['processedCalib']
                tr = calibProc_gbBlock_Trial.get_group((blNum, trNum))    

                # Targets
                xx = thisTrInfo[('targetSphericalPosInHead','az')]
                yy = thisTrInfo[('targetSphericalPosInHead','el')]
                hT = ax.scatter(xx, yy,s=50,c='b')
                hT.set_label('target')

                gazeAz_fr = np.rad2deg(np.arctan2(tr['gaze_normal2']['x'], tr['gaze_normal2']['z'] ))
                gazeEl_fr = np.rad2deg(np.arctan2(tr['gaze_normal2']['y'], tr['gaze_normal2']['z'] ))

                hT = ax.scatter(gazeAz_fr,gazeEl_fr,s=5)

                meanGazeAz = thisTrInfo[('gazeSphericalPosInHead','az')] 
                meanGazeEl = thisTrInfo[('gazeSphericalPosInHead','el')] 

                stdGazeAz = thisTrInfo[('assessmentErr','az')]
                stdGazeEl = thisTrInfo[('assessmentErr','el')]

                ax.errorbar(meanGazeAz, meanGazeEl, stdGazeAz, stdGazeEl,c='k',elinewidth=2)
                ax.scatter(meanGazeAz, meanGazeEl,s=20,c='k')

                vPos = np.max([meanGazeEl,yy])
                textStr = '   %1.2f$^\circ$ (%1.2f$^\circ$)'%(thisTrInfo['accuracy'],thisTrInfo['precision'])
                hErr = ax.text(xx,vPos+1.5, textStr,horizontalalignment='center',size=10)


            ax.set_ylabel('elevation (degrees)', fontsize=12)
            ax.set_xlabel('azimuth (degrees)', fontsize=12)
            ax.tick_params(axis='both', which='major', labelsize=12)

            ax.set_ylim([-25,25])
            ax.set_xlim([-25,25])
            
            ax.axes.yaxis.grid(True)
            ax.axes.xaxis.grid(True)
            ax.axes.set_axisbelow(True)

            plt.rcParams["font.family"] = "sans-serif"
            p.set_facecolor('w')

            outDir = 'Figures/CalibQual/' + sessionDictIn['subID'] + '/'

            if not os.path.exists(outDir):
                os.makedirs(outDir)

            fName = str(outDir + 'b-{}_' + '.png').format(blNum)

        plt.savefig(fName)
        plt.close()

    return sessionDictIn



def calibAssessment(sessionDictIn,saveDir = 'figout/', confidenceThresh = False):

    import evaluateSegAlgo as ev
    
    sessionDictIn = ev.calcCyclopean(sessionDictIn)

    sessionDictIn = ev.calcSphericalCoordinates(sessionDictIn,'targetPos','targetWorldSpherical', sessionDictKey = 'processedCalib')
    sessionDictIn = ev.calcSphericalCoordinates(sessionDictIn,'targeLocalPos','targetLocalSpherical', sessionDictKey = 'processedCalib')
    
    sessionDictIn = ev.calcSphericalCoordinates(sessionDictIn,'gaze-normal0','gaze0Spherical', sessionDictKey = 'processedCalib',flipY=False)
    sessionDictIn = ev.calcSphericalCoordinates(sessionDictIn,'gaze-normal1','gaze1Spherical', sessionDictKey = 'processedCalib',flipY=False)
    sessionDictIn = ev.calcSphericalCoordinates(sessionDictIn,'gaze-normal2','gaze2Spherical', sessionDictKey = 'processedCalib',flipY=False)

    sessionDictIn = ev.calcTrialLevelCalibInfo(sessionDictIn)

    sessionDictIn = ev.calcGazeToTargetFixError(sessionDictIn,'gaze0Spherical','targetLocalSpherical','fixError_eye0')
    sessionDictIn = ev.calcGazeToTargetFixError(sessionDictIn,'gaze1Spherical','targetLocalSpherical','fixError_eye1')
    sessionDictIn = ev.calcGazeToTargetFixError(sessionDictIn,'gaze2Spherical','targetLocalSpherical','fixError_eye2')

    # sessionDictIn['trialInfo']['fixTargetSpherical','az'] = sessionDictIn['trialInfo']['fixTargetSpherical','az'].round(2)
    # sessionDictIn['trialInfo']['fixTargetSpherical','el'] = sessionDictIn['trialInfo']['fixTargetSpherical','el'].round(2)

    sessionDictIn = ev.calcAverageGazeDirPerTrial(sessionDictIn)

    sessionDictIn = ev.calcFixationStatistics(sessionDictIn, confidenceThresh)

    ev.plotFixAssessment(sessionDictIn, saveDir)

    return sessionDictIn


def saveTemp(sessionDict):
    
    with open('tempOut.pickle', 'wb') as handle:
        pickle.dump(sessionDict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def loadTemp():
    
    file = open('tempOut.pickle', 'rb')
    sessionDict = pickle.load(file)
    file.close()
    return sessionDict


def processAllSesssions(doNotLoad=False):

    dataFolderList = []
    [dataFolderList.append(name) for name in os.listdir("Data/") if name[0] != '.']

    sessionFiles = []
    for subNum in range(len(dataFolderList)):
        sessionDict = processSingleSession(subNum,doNotLoad)
        sessionDict['trialInfo']['subjectNumber'] = subNum
        sessionFiles.append(sessionDict)

    with open('sessionFiles.pickle', 'wb') as handle:        
        pickle.dump(sessionFiles, handle, protocol=pickle.HIGHEST_PROTOCOL)

    allTrialDataDf = sessionFiles[0]['trialInfo']

    for subNum in range(1,len(sessionFiles)):
        allTrialDataDf = pd.concat([allTrialDataDf, sessionFiles[subNum]['trialInfo']])

    with open('allTrialData.pickle', 'wb') as handle:
        pickle.dump(allTrialDataDf, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
    logger.info('Saved data to sessionFiles.pickle')
    logger.info('Saved data to allTrialData.pickle')

    return (sessionFiles,allTrialDataDf)

def processSingleSession(subNum, doNotLoad=False):

    # rawDataDF = False
    # calibDF = False

    sessionDict = unpackSession(subNum, doNotLoad)

    sessionDict = flipGazeElevation(sessionDict)  # Y axis is flipped in PL space
    
    sessionDict = calcCatchingPlane(sessionDict)
    sessionDict = findLastFrame(sessionDict)
    sessionDict = calcCatchingError(sessionDict)
    sessionDict = gazeAnalysisWindow(sessionDict)

    sessionDict = filterGazeDataByConfidence(sessionDict)
    
    sessionDict = calc_gaze_Normal2(sessionDict)
    sessionDict = calcCycGIWDir(sessionDict)

    sessionDict = calcCycToBallVector(sessionDict)
    sessionDict = calcSphericalcoordinates(sessionDict)  # of ball and gaze
    
    sessionDict = calcBallAngularSize(sessionDict)

    sessionDict = calcTrackingError(sessionDict)
    sessionDict = filterAndDiffSignals(sessionDict)
    sessionDict = vectorMovementModel(sessionDict)
    sessionDict = saveOutVectorMovementModel(sessionDict)
    
    # sessionDict = runCalibrationAssessment(sessionDict)
    # sessionDict = plotTrackQuality(sessionDict)

    sessionDict = calibAssessment(sessionDict,saveDir = 'figout/'+ sessionDict['subID'] +"/", confidenceThresh = False)

    # sessionDict = loadTemp()
    # saveTemp(sessionDict)

    return sessionDict
    


if __name__ == "__main__":
    
    (sessionList,allTrialData) = processAllSesssions(doNotLoad=False)
