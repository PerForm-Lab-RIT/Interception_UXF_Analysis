
from __future__ import division
import PerformParser as pp
import pandas as pd
import numpy as np
import Quaternion as qu
import performFun as pF

import bokeh.plotting as bkP
import bokeh.models as bkM
from bokeh.palettes import Spectral6


def getPaddleRadius(sessionDict):

    paddleVisObjIndex = (sessionDict['expConfig']['visObj']['visObjVarNames']).index('paddle')
    paddleSizeStr = sessionDict['expConfig']['visObj']['visObjSizesString']
    paddleSizeStr = paddleSizeStr.replace("[", "")
    paddleSizeStr = paddleSizeStr.replace("]", "").split(',')
    paddleRadius = float(paddleSizeStr[1])
    return paddleRadius

def getBallRadius(sessionDict):
    
    return float(sessionDict['expConfig']['room']['ballDiameter'])/2.0

def calcPredCatchingError(sessionDict, predictiveBuffer = 0 ):
    
    rawDF = sessionDict['raw']
    procDF = sessionDict['processed']
    trialInfoDF = sessionDict['trialInfo']
    
    ####################################################################################
    ####################################################################################
    ### Find the time the ball reappears

    gbTrials = rawDF.groupby('trialNumber')
    renderOnFr_tr = pd.DataFrame(gbTrials['eventFlag'].apply( lambda x: x.index.values[pF.findFirst(x, 'ballRenderOn')]))

    mIndex = pd.MultiIndex.from_tuples([('ballAppearsOnFrame','')])
    renderOnFr_tr.columns = mIndex

    trialInfoDF = pd.concat([renderOnFr_tr,trialInfoDF],axis=1)

    ####################################################################################
    ####################################################################################
    ### Get predictive paddle to ball direction XZ

    # We sample paddle position at ballOn, or "predictive paddle position"
    paddlePositionOnPredictiveFrame_tr_XZ = rawDF['paddlePos'].values[trialInfoDF['ballAppearsOnFrame']]
    paddlePositionOnPredictiveFrame_tr_XZ[:,1] = 0

    gbTrials = rawDF.groupby('trialNumber')

    # Now, calculate the vector from predictive paddle position to the moving ball
    #  predictive paddle position is static over time
    #  ball position changes over time
    for trNum, tr in gbTrials:

        predPaddlePosXYZ = paddlePositionOnPredictiveFrame_tr_XZ[trNum,:]
        newData = tr['ballPos'].values - predPaddlePosXYZ

        if( trNum == 0):

            predPaddlePositionToBallPositionVec_fr_XZ = newData

        else:
            predPaddlePositionToBallPositionVec_fr_XZ = np.append( predPaddlePositionToBallPositionVec_fr_XZ, newData,0 )

    # Normalize
    predPaddleToBallDirXZ = np.array([np.divide(XYZ,np.linalg.norm(XYZ)) for 
                                       XYZ in predPaddlePositionToBallPositionVec_fr_XZ],dtype=np.float)

    predPaddleToBallDirXZDf = pd.DataFrame(predPaddleToBallDirXZ)

    predPaddleToBallDirXZDf = predPaddleToBallDirXZDf.rename(columns={0: ('predPaddleToBallDirXZ','X'), 1:('predPaddleToBallDirXZ','Y'), 2: ('predPaddleToBallDirXZ','Z')})

    procDF = pd.concat([procDF,predPaddleToBallDirXZDf],axis=1,verify_integrity=True)


    ####################################################################################
    ####################################################################################
    ### Dot product of ballTrajLatDirXZ and predPaddleToBallDirXZ

    gbTrials = procDF.groupby('trialNumber')

    dotProd_idx = []

    for trNum, tr in gbTrials:

        dotProd_idx.extend( pd.Series( np.dot( tr['predPaddleToBallDirXZ'], 
               np.array( trialInfoDF['ballTrajDirXZ'].loc[ trNum ].values,dtype=float))))


    dotDf = pd.DataFrame({('dot_ballTraj_predPaddleToBallDir',''):dotProd_idx})
    dotDf.index = procDF.index

    procDF = pd.concat([dotDf,procDF],axis=1,verify_integrity=True)


    ####################################################################################
    ####################################################################################
    ### Find the zero crossing - when the ball passes the position of the predictive paddle


    # gr = gbTrials.get_group(40)
    # data = gr['dot_ballTraj_predPaddleToBallDir'].values
    # np.where(np.diff(np.sign( data )))#[0][0]

    gbTrials = procDF.groupby('trialNumber')
    arriveAtPredPaddleFr_tr = []

    ####################################################################################
    ####################################################################################
    ### Record the frame of passage for each trial

    for trNum, tr in gbTrials:

        # if BALL WAS NOT CAUGHT
        if( np.isnan(trialInfoDF['ballCaughtFr'].values[trNum]) ):

            passFr = pF.findFirstZeroCrossing(tr['dot_ballTraj_predPaddleToBallDir'].values)   # 
            arriveAtPredPaddleFr_tr.append( tr.index.values[passFr]-1 )

        else:

            passFr = np.nan;
            arriveAtPredPaddleFr_tr.append(trialInfoDF['ballCaughtFr'].values[trNum])

    trialInfoDF['passVertPlaneAtPredPaddleFr'] = np.array(arriveAtPredPaddleFr_tr,dtype=int)
    
    

    ####################################################################################
    ####################################################################################
    ### Vertical error
    
    ballPassesPredPaddleFr = trialInfoDF['passVertPlaneAtPredPaddleFr']
    ballOnFr = trialInfoDF['ballAppearsOnFrame']

    vertError_tr = rawDF['ballPos']['Y'].values[ballPassesPredPaddleFr] - rawDF['paddlePos']['Y'].values[ballOnFr]
    trialInfoDF[('predictiveCatchingError','Y')] = vertError_tr

    ####################################################################################
    ####################################################################################
    ### Horizontal error
    
    p2bDf = pd.DataFrame(rawDF['ballPos'].values[ballPassesPredPaddleFr] - rawDF['paddlePos'].values[ballOnFr])
    ballTrajLat = trialInfoDF['ballTrajLatDirXZ']

    horizError_tr = []

    for trNum in range(len(trialInfoDF)):
        
        horizError_tr.extend( pd.Series( np.dot( p2bDf.loc[trNum],ballTrajLat.loc[trNum]),dtype=float))

    trialInfoDF[('predictiveCatchingError','X')] =  horizError_tr


    ####################################################################################
    ####################################################################################
    ### 2D error

    
    XX = np.power(np.array(trialInfoDF['predictiveCatchingError']['X'],dtype=float),2)
    YY = np.power(np.array(trialInfoDF['predictiveCatchingError']['Y'],dtype=float),2)

    trialInfoDF[('predictiveCatchingError','2D')] =  np.sqrt(np.sum([XX, YY],axis=0))


    ####################################################################################
    ####################################################################################
    ### Predictive hand position

    predHandPos_tr_XYZ = rawDF['paddlePos'].loc[ballOnFr].values
    predHandPosDf = pd.DataFrame(predHandPos_tr_XYZ)

    mIndex = pd.MultiIndex.from_tuples([('predictiveHandPosition','X'),('predictiveHandPosition','Y'),('predictiveHandPosition','Z')])
    predHandPosDf.columns = mIndex

    trialInfoDF = pd.concat([trialInfoDF,predHandPosDf],axis=1)


    ####################################################################################
    ####################################################################################
    ### Flight time until arrival at predictive paddle location 

    passOrCatchFr = trialInfoDF['passVertPlaneAtPredPaddleFr'].values
    
    trialInfoDF['timeToPredPaddle'] = rawDF['frameTime'][ballPassesPredPaddleFr].values - rawDF['frameTime'][trialInfoDF['firstFrame'].values].values

    ####################################################################################
    ####################################################################################
    ### Save and return new dict

    sessionDict['raw'] = rawDF
    sessionDict['processed'] = procDF 
    sessionDict['trialInfo']=    trialInfoDF
    
    return sessionDict

def calcCatchingError(sessionDict):
    '''
    Calculate catching error, and intermediate measures.

    Appends: 
    trialInfoDF['catchingError']
    trialInfoDF['timeToPaddle']
    trialInfoDF['passVertPlaneAtPaddleFr']
    processed['dot_bDir_p2b']
    processed['ballTrajLatDirXZ']
    processed['ballTrajDirXZ']
    
    '''
    ## Get DF
    rawDF = sessionDict['raw']
    procDF = sessionDict['processed']
    trialInfoDF = sessionDict['trialInfo']

    ########################################################################
    ### ballTrajDirXZ
    ###

    # Find traj. in XZ plane for each trial
    gbTrials = rawDF.groupby('trialNumber')
    ballTrajDirXZ_XYZ = np.array(gbTrials.nth(20)['ballPos'] - gbTrials.nth(1)['ballPos'],dtype = float)

    # Throw in a dataframe
    ballTrajDirXZ_XYZ = pd.DataFrame( {('ballTrajDirXZ','X'): ballTrajDirXZ_XYZ[:,0],
                                       ('ballTrajDirXZ','Y'): ballTrajDirXZ_XYZ[:,1],
                                       ('ballTrajDirXZ','Z'): ballTrajDirXZ_XYZ[:,2]},dtype=float)

    # Set Y value to 0, normalize
    ballTrajDirXZ_XYZ[('ballTrajDirXZ','Y')] = 0
    ballTrajDirXZ_XYZ = ballTrajDirXZ_XYZ.apply(lambda x: np.divide(x,np.linalg.norm(x)),axis=1)

    trialInfoDF = pd.concat([trialInfoDF,ballTrajDirXZ_XYZ],axis=1)

    ################################################################################
    ################################################################################
    ### ballTrajLatDirXZ

    # Find dir. orthogonal to ball's trajecotry (parallel with surface of hte plane @ paddle)
    ballTrajLatDirXZ = trialInfoDF['ballTrajDirXZ'].apply(lambda x: np.cross(x,[0.0,1.0,0.0]),axis=1)
    ballTrajLatDirXZ = ballTrajLatDirXZ.apply(lambda x: np.divide(x,np.linalg.norm(x)),axis=1)

    ## Save to trialInfoDF 
    mIndex = pd.MultiIndex.from_tuples([('ballTrajLatDirXZ','X'),('ballTrajLatDirXZ','Y'),('ballTrajLatDirXZ','Z')])
    ballTrajLatDirXZ.columns = mIndex
    trialInfoDF = pd.concat([trialInfoDF,ballTrajLatDirXZ],axis=1)
    
    ################################################################################
    ################################################################################
    ### dot_bDir_p2b

    ## Take the dot product of the ball's trajectory and the ball-to-paddle dir
    gbTrials = procDF.groupby('trialNumber')

    dotProd_idx = []
    frameTime_idx = []

    for trNum, tr in gbTrials:

        dotProd_idx.extend( pd.Series( np.dot( tr['paddleToBallDirXZ'], 
               np.array( trialInfoDF['ballTrajDirXZ'].loc[ trNum-1 ].values,dtype=float))))


    dotDf = pd.DataFrame({('dot_bDir_p2b',''):dotProd_idx})
    dotDf.index = procDF.index

    procDF = pd.concat([dotDf,procDF],axis=1,verify_integrity=True)

    ################################################################################
    ################################################################################
    ### passVertPlaneAtPaddleFr

    #### Find the time the ball crosses the paddle plane
    #... which is when the dot product crosses over zero

    #gbTrials = procDF.groupby('trialNumber')
    arriveAtPaddleFr_tr = []

    gbTrials = procDF.groupby('trialNumber')

    for trNum, tr in gbTrials:
        if( trialInfoDF['ballCaughtQ'].values[trNum-1] == False ):

            passFr = pF.findFirstZeroCrossing(tr['dot_bDir_p2b'].values)       
            arriveAtPaddleFr_tr.append(tr.index.values[passFr]-1)

        else:
            arriveAtPaddleFr_tr.append(trialInfoDF['ballCaughtFr'].values[trNum-1])


    trialInfoDF['passVertPlaneAtPaddleFr'] = np.array(arriveAtPaddleFr_tr,dtype=int)


    ################################################################################
    ################################################################################
    ### Catching error: passVertPlaneAtPaddleErr X, Y, and 2D

    #### Vertical catching error

    vertError_tr = procDF['paddleToBallVec']['Y'].values[ trialInfoDF['passVertPlaneAtPaddleFr'].values]
    
    trialInfoDF[('catchingError','Y')] = vertError_tr

    #### Horizontal catching error

    gbTrials = procDF.groupby('trialNumber')

    p2bDf = procDF['paddleToBallVec'].loc[trialInfoDF['passVertPlaneAtPaddleFr'].values].values
    ballTrajLat = trialInfoDF['ballTrajLatDirXZ'].values

    horizError_tr = []

    for trNum in range(len(trialInfoDF)):

        horizError_tr.extend( pd.Series( np.dot( p2bDf[trNum],ballTrajLat[trNum]),dtype=float))

    trialInfoDF[('catchingError','X')] =  horizError_tr

    #### 2D catching error

    XX = np.power(np.array(trialInfoDF['catchingError']['X'],dtype=float),2)
    YY = np.power(np.array(trialInfoDF['catchingError']['Y'],dtype=float),2)

    trialInfoDF[('catchingError','2D')] =  np.sqrt(np.sum([XX, YY],axis=0))


    ################################################################################
    ################################################################################
    ### timeToPaddle

    #### Flight time until arrial at paddle

    passOrCatchFr = trialInfoDF['passVertPlaneAtPaddleFr'].values
    trialInfoDF['timeToPaddle'] = rawDF['frameTime'][passOrCatchFr].values - rawDF['frameTime'][trialInfoDF['firstFrame'].values].values

    ### Save to session dict and return
    sessionDict['raw'] = rawDF
    sessionDict['processed'] = procDF
    sessionDict['trialInfo'] = trialInfoDF
    
    return sessionDict

def calcPaddleBasis(sDF):

    paddleDf = pd.DataFrame()
    paddleDf.index.name = 'frameNum'

    ########################################################
    # Convert quats to rotation matrices
    
    paddleRotMat_fr_mRow_mCol = [qu.Quat(np.array(q,dtype=np.float))._quat2transform() for q in sDF.paddleQuat.values]    
    np.shape(paddleRotMat_fr_mRow_mCol)

    ########################################################
    ### Calc normal of paddle face
    
    paddleForward_fr_XYZ = np.array([[0,0,-1]] * len(paddleRotMat_fr_mRow_mCol))
       
    paddleFaceNorm_fr_XYZ = np.array([ np.dot(paddleRotMat_fr_mRow_mCol[fr],paddleForward_fr_XYZ[fr]) 
         for fr in range(len(paddleForward_fr_XYZ))])

    paddleFordwardVecDf = pd.DataFrame({('paddleFaceDir','X'):paddleFaceNorm_fr_XYZ[:,0],
                                 ('paddleFaceDir','Y'):paddleFaceNorm_fr_XYZ[:,1],
                                 ('paddleFaceDir','Z'):paddleFaceNorm_fr_XYZ[:,2]})

    paddleDf = pd.concat([paddleDf,paddleFordwardVecDf],axis=1)

    ########################################################
    ### Calc paddle UP vector

    paddleUp_fr_XYZ = np.array([[0,-1,0]] * len(paddleRotMat_fr_mRow_mCol))

    paddleUpNorm_fr_XYZ = np.array([ np.dot(paddleRotMat_fr_mRow_mCol[fr],paddleUp_fr_XYZ[fr]) 
         for fr in range(len(paddleUp_fr_XYZ))])

    paddleUpVecDf = pd.DataFrame({('paddleUpDir','X'):paddleUpNorm_fr_XYZ[:,0],
                                 ('paddleUpDir','Y'):paddleUpNorm_fr_XYZ[:,1],
                                 ('paddleUpDir','Z'):paddleUpNorm_fr_XYZ[:,2]})

    paddleDf = pd.concat([paddleDf,paddleUpVecDf],axis=1)
    
    ########################################################
    ### Calc paddle face lateral vector

    paddleFaceLatVec_fr_xyz = np.cross(paddleDf['paddleFaceDir'],paddleDf['paddleUpDir'])

    paddlFaceLatVecDf = pd.DataFrame({('paddlFaceLatDir','X'):paddleFaceLatVec_fr_xyz[:,0],
                                 ('paddlFaceLatDir','Y'):paddleFaceLatVec_fr_xyz[:,1],
                                 ('paddlFaceLatDir','Z'):paddleFaceLatVec_fr_xyz[:,2]})

    paddleDf = pd.concat([paddleDf,paddlFaceLatVecDf],axis=1)
    paddleDf[1:10]
    
    ########################################################
    ### Calc paddle-to-ball vector

    paddleToBallVec_fr_XYZ = sDF['ballPos'].values - sDF['paddlePos'].values

    paddleToBallVecDf = pd.DataFrame({('paddleToBallVec','X'):paddleToBallVec_fr_XYZ[:,0],
                                 ('paddleToBallVec','Y'):paddleToBallVec_fr_XYZ[:,1],
                                 ('paddleToBallVec','Z'):paddleToBallVec_fr_XYZ[:,2]})
                 
    paddleDf = pd.concat([paddleDf,paddleToBallVecDf],axis=1)

    ########################################################
    ### Calc paddle-to-ball direction (normalized paddle-to-ball vector)

    paddleToBallDir_fr_XYZ = np.array([np.divide(XYZ,np.linalg.norm(XYZ)) for XYZ in paddleToBallVec_fr_XYZ],dtype=np.float)

    paddleToBallDirDf = pd.DataFrame({('paddleToBallDir','X'):paddleToBallDir_fr_XYZ[:,0],
                                 ('paddleToBallDir','Y'):paddleToBallDir_fr_XYZ[:,1],
                                 ('paddleToBallDir','Z'):paddleToBallDir_fr_XYZ[:,2]})
                 
    paddleDf = pd.concat([paddleDf,paddleToBallDirDf],axis=1)

    #########################################################
    #### Calc paddle-to-ball direction XZ

    paddleToBallVec_fr = paddleToBallDirDf.values
    paddleToBallVec_fr[:,1] = 0
    paddleToBallVec_fr = pd.DataFrame(paddleToBallVec_fr)
    paddleToBallVec_fr
    
    paddleToBallVecXZDf = paddleToBallVec_fr.rename(columns={0: ('paddleToBallDirXZ','X'), 1:('paddleToBallDirXZ','Y'), 2: ('paddleToBallDirXZ','Z')})

    paddleToBallDirXZDf = paddleToBallVecXZDf.apply(lambda x: np.divide(x,np.linalg.norm(x)),axis=1)

    paddleDf = pd.concat([paddleDf,paddleToBallDirXZDf],axis=1,verify_integrity=True)

    return paddleDf

def initTrialInfo(sessionDf):
    '''
    Create a trial info dataframe with some basic settings
    
    '''

    postBlankDur_tr = []
    preBlankDur_tr = []
    blankDur_tr = []
    trialType_tr = []
    
    gbTrials = sessionDf.groupby('trialNumber')

    for trNum, tr in gbTrials:

        preBlankDur_tr.append(tr['preBlankDur'].values[0])
        blankDur_tr.append(tr['blankDur'].values[0])
        postBlankDur_tr.append(tr['postBlankDur'].values[0])
        trialType_tr.append(tr['trialType'].values[0])
        


    trialInfo = pd.DataFrame({('preBlankDur',''):preBlankDur_tr,
                       ('blankDur',''):blankDur_tr,
                       ('postBlankDur',''):postBlankDur_tr,
                       ('trialType',''):trialType_tr})
    
    trialInfo['firstFrame'] = [tr.index[0] for trNum, tr in gbTrials]
    trialInfo['lastFrame'] = [tr.index[-1] for trNum, tr in gbTrials]


    trialInfo = calcTrialOutcome(sessionDf,trialInfo)
    trialInfo.index.name = 'trialNum'


    return trialInfo



def initProcessedData(sessionDf):

    from copy import deepcopy
    procData = deepcopy(sessionDf)

    # procData =   pd.DataFrame({
    # ('trialNumber',''):sessionDf.trialNumber,
    # ('frameTime',''):sessionDf.frameTime,
    # ('eventFlag',''): sessionDf.eventFlag,
    # ('blockNumber',''): sessionDf.blockNumber})

    procData.index.name = 'frameNum'

    return procData



def calcTrialOutcome(sessionDf,sessionTrialInfo):
    '''
    Accepts: 
        session dataFrame
        trialInfo dataFrame
    Returns: 
        trialInfo dataFrame
    '''
    gbTrials = sessionDf.groupby('trialNumber')


    ballCaughtQ = []
    ballCaughtFr = []

    for trNum, tr in gbTrials:
        
        catchFr = pF.findFirst(tr['eventFlag'],'ballOnPaddle')
        
        if( catchFr ):

            ballCaughtQ.append(True)
            ballCaughtFr.append(tr.index.values[catchFr])        
            
        else:
            ballCaughtQ.append(False)
            ballCaughtFr.append(np.nan)

    df = pd.DataFrame({('ballCaughtFr',''):ballCaughtFr,('ballCaughtQ',''):ballCaughtQ})

    return pd.concat([df,sessionTrialInfo],axis=1)

