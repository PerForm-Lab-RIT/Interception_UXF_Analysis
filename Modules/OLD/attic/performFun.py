from __future__ import division
import PerformParser as pp
import pandas as pd
import numpy as np
import bokeh.plotting as bkP
import bokeh.models as bkM
from scipy import signal as sig
import catchE2Expansion as expFun
import cv2
import Quaternion as qu
import matplotlib.pyplot as plt

from configobj import ConfigObj
from configobj import flatten_errors
from validate import Validator

import classy as pC



def calcGIW(sessionDictIn):
    
    proc = sessionDictIn['processed']
    
    if( ('cycGIW' in proc.columns) is True ):
            print('cycGIW is already in the dataframe.')
            return

    def eihToGIW(rowIn):            

        # Grab gransformation matrix
        #headTransform_4x4 = np.reshape(rowIn["viewMat"],[4,4])

        mat =[]
        for i in range(16):
            mat.append(rowIn["viewMat"][str(i)])
        
        headTransform_4x4 = np.reshape(mat,[4,4])

        # Transpose
        headTransform_4x4 = headTransform_4x4.T

        # Grab cyc EIH direction
        cycEyeInHead_XYZ = rowIn['cycEyeInHead']
        # Add a 1 to convert to homogeneous coordinates
        cycEyeInHead_XYZW = np.hstack( [cycEyeInHead_XYZ,1])

        # Take the dot product!
        cycGIWVec_XYZW = np.dot( headTransform_4x4,cycEyeInHead_XYZW)

        # Now, convert into a direction from the cyclopean eye in world coordinates
        # Also, we can discard the w term
        cycGIWDir_XYZ = (cycGIWVec_XYZW[0:3]-rowIn["viewPos"]) / np.linalg.norm((cycGIWVec_XYZW[0:3]-rowIn["viewPos"]))

        # You must return as a list or a tuple
        #return list(cycGIWDir_XYZ)
        return {('cycGIWDir','X'): cycGIWDir_XYZ[0],('cycGIWDir','Y'): cycGIWDir_XYZ[1],('cycGIWDir','Z'): cycGIWDir_XYZ[2]}
    
    cycGIW = proc.apply(lambda row: eihToGIW(row),axis=1)
    cycGIWDf = pd.DataFrame.from_records(cycGIW)
    cycGIWDf.columns = pd.MultiIndex.from_tuples(cycGIWDf.columns)
    sessionDictIn['processed'] = proc.combine_first(pd.DataFrame.from_records(cycGIWDf))
    
    
    return sessionDictIn

def calcCycEIH(sessionDictIn):
    def calcEIHByRow(row):
        
        cycEIH = (np.array(row['leftEyeInHead'].values,dtype=np.float) + 
            np.array(row['rightEyeInHead'].values,dtype=np.float))/2

        cycEIH = cycEIH / np.linalg.norm(cycEIH)
        return {('cycEyeInHead','X'): cycEIH[0],('cycEyeInHead','Y'): cycEIH[1],('cycEyeInHead','Z'): cycEIH[2]}
    
    cycEIH_fr = sessionDictIn['processed'].apply(calcEIHByRow,axis=1)
    cycEIHDf = pd.DataFrame.from_records(cycEIH_fr)
    cycEIHDf.columns = pd.MultiIndex.from_tuples(cycEIHDf.columns)
    
    sessionDictIn['processed'] = sessionDictIn['processed'].combine_first(pd.DataFrame.from_records(cycEIHDf))
    
    return sessionDictIn


def calcSavitskyGolayVelocity(sessionDict, deriv = 1,polyorder = 2):

    sessionDict['processed'][('cycGIWAngles','azimuth')]   = sessionDict['processed'][('cycGIWDir','X')].apply(lambda x: np.rad2deg(np.arccos(x)))
    sessionDict['processed'][('cycGIWAngles','elevation')] = sessionDict['processed'][('cycGIWDir','Y')].apply(lambda y: np.rad2deg(np.arccos(y)))

    from scipy.signal import savgol_filter

    x = sessionDict['processed'][('cycGIWAngles')].values
    fixDurFrames =  int(np.floor(0.1 / sessionDict['analysisParameters']['fps']))
    delta = sessionDict['analysisParameters']['fps']

    #print 'Filter width: %u frames' % fixDurFrames

    vel = savgol_filter(x, fixDurFrames, polyorder, deriv=1, delta=delta, axis=0, mode='interp', cval=0.0)

    sessionDict['processed'][('cycSGVel','azimuth')] = vel[:,0]
    sessionDict['processed'][('cycSGVel','elevation')] = vel[:,1]

    vel = np.sqrt(np.sum(np.power(vel,2),1))
    sessionDict['processed'][('cycSGVel','2D')] = vel

    print 'Added sessionDict[\'processed\'][\'cycSGVel\']'
    return sessionDict

def calcAngularVel(sessionDict):
    '''
    Calculates angular velocity for cycGIW and the cyc-to-ball vectors, and the ratio of these velocities. 
    '''

    ##############################################
    ## Cyc GIW velocity 
    
    # If needed, get eye tracker time stamp
    if( columnExists( sessionDict['processed'], 'smiDeltaT') is False):
        sessionDict = calcSMIDeltaT(sessionDict)    
        
    #  If needed, calculate cyc gaze angle and velocity
    if( columnExists(sessionDict['processed'],'cycGIWDir') is False):
        sessionDict = calcGIWDirection(sessionDict)
        
    # Calc cyc angular velocity
    if( columnExists(sessionDict['processed'],'cycGIWVelocity') is False):
        sessionDict = calcAngularVelocityComponents(sessionDict, 'cycGIWDir', 'smiDeltaT', 'cycGIWVelocity')

    ##############################################
    ## Cyc-to-ball velocity 
    
    # If needed, get vizard time stamp
    if( columnExists( sessionDict['processed'], 'vizardDeltaT') is False):
        sessionDict = calcVizardDeltaT(sessionDict)
        
    # Calc cyc to ball direction
    if( columnExists(sessionDict['processed'],'cycToBallDir') is False):
        sessionDict = calcCycToBallVector(sessionDict)
        
    # Calc cyc to ball angular velocity
    if( columnExists(sessionDict['processed'],'cycToBallVelocity') is False):
        sessionDict = calcAngularVelocityComponents(sessionDict, 'cycToBallDir', 'vizardDeltaT', 'cycToBallVelocity')
    
    return sessionDict

def calcAngularVelocityComponents(sessionDict, dataColumnName, timeColumnName, outColumnName):

    #####################################################################################
    ######################################################################################################
    ### Get unsigned angular displacement and velocity

    vecDir_fr_XYZ = sessionDict['processed'][dataColumnName].values
    
    angularDisp_fr_XYZ = [ vectorAngle(vecDir_fr_XYZ[fr-1,:],vecDir_fr_XYZ[fr,:]) for fr in range(1,len(vecDir_fr_XYZ))]
    angularDisp_fr_XYZ.append(0)
    angularVel_fr_XYZ = angularDisp_fr_XYZ / sessionDict['processed'][timeColumnName]

    sessionDict['processed'][(outColumnName,'2D')] = angularVel_fr_XYZ
    print '*** calcBallAngularVelocity(): Added sessionDict[\'processed\'][(\'ballAngularVelocity\',\'2D\')] ***'

    ######################################################################################################
    ######################################################################################################
    ### Get  angular displacement and velocity along world X/Y
    
    if(columnExists(sessionDict['processed'],'worldUpDir') is False):
        sessionDict = calcWorldUpDir(sessionDict)
        
    yDir_fr_xyz = sessionDict['processed']['worldUpDir'].values
    
    azimuthulVec = np.cross(vecDir_fr_XYZ,yDir_fr_xyz)
    azimuthulDir = np.array([np.divide(XYZ,np.linalg.norm(XYZ)) for XYZ in azimuthulVec],dtype=np.float)
    
    # orthogonal to gaze vector and azimuthulDir
    elevationVec = np.cross(azimuthulDir,vecDir_fr_XYZ);
    elevationDir = np.array([np.divide(XYZ,np.linalg.norm(XYZ)) for XYZ in elevationVec],dtype=np.float)
    
    # Velocity
    vecDirOffset_fr_XYZ = np.roll(vecDir_fr_XYZ,1,axis=0)
    
    def vecDot(a,b):
            res1 = np.einsum('ij, ij->i', a, b)
            res2 = np.sum(a*b, axis=1)
            return res2
    
    # Get vector lengths when projected onto the new basis
    xDist_fr_xyz = vecDot(vecDirOffset_fr_XYZ,azimuthulDir)
    yDist_fr_xyz = vecDot(vecDirOffset_fr_XYZ,elevationDir)
    zDist_fr_xyz = vecDot(vecDirOffset_fr_XYZ,vecDir_fr_XYZ)
    
    horzError_fr = np.rad2deg( np.arctan2(xDist_fr_xyz,zDist_fr_xyz))
    vertError_fr = np.rad2deg( np.arctan2(yDist_fr_xyz,zDist_fr_xyz))

    velX_fr = horzError_fr / sessionDict['processed'][timeColumnName]
    velY_fr = vertError_fr / sessionDict['processed'][timeColumnName]
    
    sessionDict['processed'][(outColumnName,'X')] = velX_fr
    sessionDict['processed'][(outColumnName,'Y')] = velY_fr
    
    print '*** calcAngularVelocity(): Added sessionDict[\'processed\'][(\'' + outColumnName + '] ***'

    return sessionDict


def calcVizardDeltaT(sessionDict):

    #sessionDict['processed']['frameTime'] = pd.to_datetime(sessionDict['raw']['frameTime'],unit='s')
    #sessionDict['processed']['frameTime'] = sessionDict['raw']['frameTime']
    deltaTime = pd.to_datetime(sessionDict['raw']['frameTime'],unit='s').diff()
    deltaTime.loc[deltaTime.dt.microseconds==0] = pd.NaT
    deltaTime = deltaTime.fillna(method='bfill', limit=1)
    sessionDict['processed']['vizardDeltaT'] = deltaTime.dt.microseconds / 1000000

    print '*** calcVizardDeltaT(): Added sessionDict[\'processed\'][\'frameTime\'] ***'

    return sessionDict

def calcAngularCalibrationError(sessionDict):

    eyeToCalibrationPointDirDf = metricEyeOnScreenToEyeInHead(sessionDict,sessionDict['calibration']['calibPointMetricEyeOnScreen'],'eyeToCalibrationPoint')
    cycEIHDirDf = metricEyeOnScreenToEyeInHead(sessionDict,sessionDict['calibration']['cycMetricEyeOnScreen'],'cycEIH')

    zVec_fr_xyz = from1x3_to_1x4([0,0,1],eyeOffsetX=0.0,numReps = len(sessionDict['calibration']))
    zVec_fr_xyz = zVec_fr_xyz[:,0:3]

    yVec_fr_xyz = from1x3_to_1x4([0,1,0],eyeOffsetX=0.0,numReps = len(sessionDict['calibration']))
    yVec_fr_xyz = zVec_fr_xyz[:,0:3]

    cycEIHDir_fr_xyz = cycEIHDirDf.values
    eyeToCalibDir_fr_xyz = eyeToCalibrationPointDirDf.values


    def vecDot(a,b):
            res1 = np.einsum('ij, ij->i', a, b)
            res2 = np.sum(a*b, axis=1)
            return res2

    cycVertAngle_fr = np.rad2deg( np.arctan2( cycEIHDir_fr_xyz[:,1], cycEIHDir_fr_xyz[:,2]))
    cycHorizAngle_fr = np.rad2deg( np.arctan2( cycEIHDir_fr_xyz[:,0], cycEIHDir_fr_xyz[:,2]))

    calibVertAngle_fr = np.rad2deg( np.arctan2( eyeToCalibDir_fr_xyz[:,1], eyeToCalibDir_fr_xyz[:,2]))
    calibHorizAngle_fr = np.rad2deg( np.arctan2( eyeToCalibDir_fr_xyz[:,0], eyeToCalibDir_fr_xyz[:,2]))

    cycAngle_elAz = np.array([cycHorizAngle_fr,cycVertAngle_fr],dtype=np.float)
    calibAngle_elAz = np.array([calibHorizAngle_fr,calibVertAngle_fr],dtype=np.float)

    angularError = findResidualError(cycAngle_elAz,calibAngle_elAz) / sessionDict['analysisParameters']['numberOfCalibrationPoints']
    print '*****Residual angular error: *****' + str(angularError) + '*****'

    sessionDict['analysisParameters']['angularCalibrationError'] = angularError
    return sessionDict

def removeOutliers(sessionDict, columnName):
    '''
    Compatible with multiindex dataframes. 
    columnName can refer to the top level of a multiindex). eg columnName='CatchError'
    or a multiindex eg columnName=[('CatchError','X')]
    
    returns 
    '''
    # Add it to a list of variables that have been pruned of outliers
    if( 'removedOutliersFrom' in sessionDict['analysisParameters'].keys() ):
        
        # Prevent double-pruning
        if( columnName in sessionDict['analysisParameters']['removedOutliersFrom']):
            raise AssertionError, 'Column ' +  ' in trialInfo has already been pruned of outliers'
        else:
            sessionDict['analysisParameters']['removedOutliersFrom'].append(columnName)
            
    else:
        sessionDict['analysisParameters']['removedOutliersFrom'] = [columnName]
        
    outlierThresholdSD = sessionDict['analysisParameters']['outlierThresholdSD']
    
    def pruneOutliersFromSeries(series,outlierThresholdSD):
        
        data_tr = series.values
        mean = series.mean()
        std = series.std()

        outlierIdx = []

        [outlierIdx.append(i) for i,v in enumerate(data_tr) if abs(v - mean) > outlierThresholdSD * std]

        data_tr[outlierIdx] = np.NaN

        return data_tr
    
        
    if(  type(sessionDict['trialInfo'][columnName]) == pd.DataFrame ):
        
        sessionDict['trialInfo'][columnName] = sessionDict['trialInfo'][columnName].apply(
            pruneOutliersFromSeries,outlierThresholdSD=outlierThresholdSD,axis=1)

    elif( type(sessionDict['trialInfo'][columnName]) == pd.Series ):
        
        sessionDict['trialInfo'][(columnName,'')] = pruneOutliersFromSeries(sessionDict['trialInfo'][columnName],outlierThresholdSD=outlierThresholdSD)
    
    return sessionDict


def calcBallAngularSize(sessionDict):
    
    eyeToBallDist_fr = [np.sqrt( np.sum(  np.power(bXYZ-vXYZ,2))) for bXYZ,vXYZ in zip( sessionDict['raw']['ballPos'].values,sessionDict['raw']['viewPos'].values)]
    ballRadiusM_fr = sessionDict['raw']['ballRadiusM']
    sessionDict['processed']['ballRadiusDegs'] = [np.rad2deg(np.arctan(rad/dist)) for rad, dist in zip(ballRadiusM_fr,eyeToBallDist_fr)]
    return sessionDict


def calcGazeToBallError(sessionDict,eyeString,upVecString):
    
    def calcEyeToBallDir_worldCentered(row):
        
        eyeToBallDist = np.sqrt( np.sum(  np.power(row['ballPos']-row['viewPos'],2))) 

        # Calc ball position to direction within head-centered coordinates
        ballVec_XYZ = row['ballPos'] - row['viewPos']  #np.dot(mat_4x4,cVec)[0:3] 
        ballDir_XYZ = np.divide( ballVec_XYZ , np.linalg.norm(ballVec_XYZ) )

        ball_az = np.rad2deg(np.arctan(ballDir_XYZ[0]/ballDir_XYZ[2]))
        ball_el = np.rad2deg(np.arctan(ballDir_XYZ[1]/ballDir_XYZ[2]))

            
        # Calc EIH direction along az and el
        cycGIWDir_XYZ = np.array(row['cycGIWDir'])
        cycGIW_az = np.rad2deg(np.arctan(row['cycGIWDir'][0]/row['cycGIWDir'][2]))
        cycGIW_el = np.rad2deg(np.arctan(row['cycGIWDir'][1]/row['cycGIWDir'][2]))

        # The difference along az and el
        azimuthalDist = ball_az - cycGIW_az
        elevationDist = ball_el - cycGIW_el

        angDist = np.rad2deg(np.arccos(np.dot(row['cycGIWDir'] , ballDir_XYZ)))
        
        ballRadiusDegs = np.float(row['ballRadiusDegs'])
        
        # Correct for ball radius
        if azimuthalDist <= ballRadiusDegs:
            azimuthalDist = 0 
        elif azimuthalDist > 0:
            azimuthalDist -= ballRadiusDegs
        else:
            azimuthalDist += ballRadiusDegs

        if elevationDist <= ballRadiusDegs:
            elevationDist = 0
        elif elevationDist > 0:
            elevationDist -= ballRadiusDegs
        else:
            elevationDist += ballRadiusDegs

        return (angDist,azimuthalDist,elevationDist)

    tempDict = sessionDict
    tempRawDF = tempDict['raw']

    tempRawDF['ballRadiusDegs']  =  sessionDict['processed']['ballRadiusDegs']
    tempRawDF[('cycGIWDir','X')] =  sessionDict['processed'][('cycGIWDir','X')]
    tempRawDF[('cycGIWDir','Y')] =  sessionDict['processed'][('cycGIWDir','Y')]
    tempRawDF[('cycGIWDir','Z')] =  sessionDict['processed'][('cycGIWDir','Z')]

    #ballRadius = float(sessionDict['expConfig']['room']['ballDiameter'])/2.0
    
    absAzEl_XYZ = tempRawDF.apply(lambda row: calcEyeToBallDir_worldCentered(row) ,axis=1)

    (absDist,azDist,elDist) = zip(*absAzEl_XYZ)

    outVarColName = 'cycGIWtoBallAngle'
    upVecString = 'worldUp'

    tempDict['processed'][(outVarColName,'X_' + upVecString)] = azDist
    tempDict['processed'][(outVarColName,'Y_' + upVecString)] = elDist
    tempDict['processed'][(outVarColName,'2D')] = absDist

    print 'calcGazeToBallError(): Created sessionDict[\'processed\']' + '[\'' + outVarColName + '\']'

    return tempDict


def calcGazeToBallCenterError(sessionDict,eyeString,upVecString):
    
    def calcEyeToBallDir_worldCentered(row):
        
        eyeToBallDist = np.sqrt( np.sum(  np.power(row['ballPos']-row['viewPos'],2))) 

        # Calc ball position to direction within head-centered coordinates
        ballVec_XYZ = row['ballPos'] - row['viewPos']  #np.dot(mat_4x4,cVec)[0:3] 
        ballDir_XYZ = np.divide( ballVec_XYZ , np.linalg.norm(ballVec_XYZ) )

        ball_az = np.rad2deg(np.arctan(ballDir_XYZ[0]/ballDir_XYZ[2]))
        ball_el = np.rad2deg(np.arctan(ballDir_XYZ[1]/ballDir_XYZ[2]))

            
        # Calc EIH direction along az and el
        cycGIWDir_XYZ = np.array(row['cycGIWDir'])
        cycGIW_az = np.rad2deg(np.arctan(row['cycGIWDir'][0]/row['cycGIWDir'][2]))
        cycGIW_el = np.rad2deg(np.arctan(row['cycGIWDir'][1]/row['cycGIWDir'][2]))

        # The difference along az and el
        azimuthalDist = ball_az - cycGIW_az
        elevationDist = ball_el - cycGIW_el

        angDist = np.rad2deg(np.arccos(np.dot(row['cycGIWDir'] , ballDir_XYZ)))
        
        return (angDist,azimuthalDist,elevationDist)

    tempDict = sessionDict
    tempRawDF = tempDict['raw']

    #tempRawDF['ballRadiusDegs']  =  sessionDict['processed']['ballRadiusDegs']
    tempRawDF[('cycGIWDir','X')] =  sessionDict['processed'][('cycGIWDir','X')]
    tempRawDF[('cycGIWDir','Y')] =  sessionDict['processed'][('cycGIWDir','Y')]
    tempRawDF[('cycGIWDir','Z')] =  sessionDict['processed'][('cycGIWDir','Z')]

    #ballRadius = float(sessionDict['expConfig']['room']['ballDiameter'])/2.0
    
    absAzEl_XYZ = tempRawDF.apply(lambda row: calcEyeToBallDir_worldCentered(row) ,axis=1)

    (absDist,azDist,elDist) = zip(*absAzEl_XYZ)

    outVarColName = 'cycGIWtoBallCenterAngle'
    upVecString = 'worldUp'

    tempDict['processed'][(outVarColName,'X_' + upVecString)] = azDist
    tempDict['processed'][(outVarColName,'Y_' + upVecString)] = elDist
    tempDict['processed'][(outVarColName,'2D')] = absDist

    print 'calcGazeToBallCenterError(): Created sessionDict[\'processed\']' + '[\'' + outVarColName + '\']'

    return tempDict


def calcCycToBallVector(sessionDict):

    cycToBallVec = np.array(sessionDict['raw']['ballPos'] - sessionDict['raw']['viewPos'],dtype=np.float )
    cycToBallDir = np.array([np.divide(XYZ,np.linalg.norm(XYZ)) for XYZ in cycToBallVec],dtype=np.float)

    sessionDict['processed'][('cycToBallDir','X')] = cycToBallDir[:,0]
    sessionDict['processed'][('cycToBallDir','Y')] = cycToBallDir[:,1]
    sessionDict['processed'][('cycToBallDir','Z')] = cycToBallDir[:,2]

    return sessionDict


def calcBallOffOnFr(sessionDict):

    gbTrials = sessionDict['processed'].groupby('trialNumber')

    xErr_tr = []
    yErr_tr = []


    ballOffIdx_tr = []
    ballOnIdx_tr = []

    for trNum,tr in gbTrials:

        ballOffIdx_tr.append(tr.index[0] + findFirst(tr['eventFlag'],'ballRenderOff'))
        ballOnIdx_tr.append(tr.index[0] + findFirst(tr['eventFlag'],'ballRenderOn'))

    sessionDict['trialInfo'][('ballOnFr','')] = ballOnIdx_tr
    sessionDict['trialInfo'][('ballOffFr','')] = ballOffIdx_tr

    return sessionDict

def calcOrthToBallAndHeadUpDir(sessionDict):
    # Lateral vector:  orthToBallAndHeadUpDir = cross( headUpDir, cycToBall)
    
    headUpLatVec = np.cross(sessionDict['processed']['cycToBallDir'],sessionDict['processed']['headUpInWorldDir'])
    headUpLatDir = np.array([np.divide(XYZ,np.linalg.norm(XYZ)) for XYZ in headUpLatVec],dtype=np.float)

    orthToBallAndHeadUpDirDf = pd.DataFrame({('orthToBallAndHeadUpDir','X'):headUpLatDir[:,0],
                                  ('orthToBallAndHeadUpDir','Y'):headUpLatDir[:,1],
                                  ('orthToBallAndHeadUpDir','Z'):headUpLatDir[:,2]})

    sessionDict['processed'] = pd.concat([sessionDict['processed'],orthToBallAndHeadUpDirDf],axis=1)
    return sessionDict

def calcOrthToBallAndWorldUpDir(sessionDict):
    # Lateral vector:  orthToBallAndWorldUpDir = cross( worldUpDir,cycToBall)
    
    #worldUp_fr = np.array([[0,1,0]] * len(sessionDict['raw']))

    worldUpLatVec = np.cross(sessionDict['processed']['cycToBallDir'],sessionDict['processed']['worldUpDir'])
    worldUpLatDir = np.array([np.divide(XYZ,np.linalg.norm(XYZ)) for XYZ in worldUpLatVec],dtype=np.float)

    orthToBallAndWorldUpDirDf = pd.DataFrame({('orthToBallAndWorldUpDir','X'):worldUpLatDir[:,0],
                                  ('orthToBallAndWorldUpDir','Y'):worldUpLatDir[:,1],
                                  ('orthToBallAndWorldUpDir','Z'):worldUpLatDir[:,2]})

    sessionDict['processed'] = pd.concat([sessionDict['processed'],orthToBallAndWorldUpDirDf],axis=1)

    return sessionDict



def calcHeadUpDir(sessionDict):
    import Quaternion as qu

    ###################################################
    ###################################################
    ###  Calculate head up
    
    # Get rotation matrix for the head
    viewRotMat_fr_mRow_mCol = [qu.Quat(np.array(q,dtype=np.float))._quat2transform() for q in sessionDict['raw'].viewQuat.values]

    headUpInHeadVec_fr_XYZ = np.array([[0,1,0]] * len(viewRotMat_fr_mRow_mCol))

    headUpInWorldDir_fr_XYZ = np.array([ np.dot(viewRotMat_fr_mRow_mCol[fr],headUpInHeadVec_fr_XYZ[fr]) 
             for fr in range(len(headUpInHeadVec_fr_XYZ))])

    headUpInWorldDirDf = pd.DataFrame({('headUpInWorldDir','X'):headUpInWorldDir_fr_XYZ[:,0],
                                 ('headUpInWorldDir','Y'):headUpInWorldDir_fr_XYZ[:,1],
                                 ('headUpInWorldDir','Z'):headUpInWorldDir_fr_XYZ[:,2]})
    # Concat
    sessionDict['processed'] = pd.concat([sessionDict['processed'],headUpInWorldDirDf],axis=1)

    return sessionDict

def calcWorldUpDir(sessionDict):
    
    worldUp_fr_xyz = np.array(np.array([[0,1,0]] * len(sessionDict['raw'])),dtype=np.float)
    
    worldUpDf = pd.DataFrame({('worldUpDir','X'):worldUp_fr_xyz[:,0],
                                 ('worldUpDir','Y'):worldUp_fr_xyz[:,1],
                                 ('worldUpDir','Z'):worldUp_fr_xyz[:,2]})

    sessionDict['processed'] = pd.concat([sessionDict['processed'],worldUpDf],axis=1)

    return sessionDict


def calcGIWDirection(sessionDict):

	print 'calcGIWDirectionAndVelocity(): Currently only calculates GIW angle/vel for cyc eye data'

	# cycFiltEyeOnScreen -> filtCycMetricEyeOnScreen
	tempDF = eyeOnScreenToMetricEyeOnScreen(sessionDict,sessionDict['processed']['cycFiltEyeOnScreen'],'filtCycMetricEyeOnScreen')
	sessionDict['processed'] = pd.concat([sessionDict['processed'],tempDF],axis=1)

	# filtCycMetricEyeOnScreen -> filtCycEyeInHeadDirDf
	filtCycEyeInHeadDirDf = metricEyeOnScreenToEyeInHead(sessionDict,sessionDict['processed']['filtCycMetricEyeOnScreen'],'filtCycEyeInHeadDir')
	sessionDict['processed'] = pd.concat([sessionDict['processed'],filtCycEyeInHeadDirDf],axis=1)

	# filtCycEyeInHeadDir -> filtCycGazeNodeInWorld
	# filtCycGazeNodeInWorld is a point 1 meter from the head along the gaze direction 
	# This is not yet the GIW angle
	filtCycGazeNodeInWorldDF  = headToWorld(sessionDict,sessionDict['processed']['filtCycEyeInHeadDir'],'filtCycGazeNodeInWorldDF')
	sessionDict['processed'] = pd.concat([sessionDict['processed'],filtCycGazeNodeInWorldDF],axis=1)

	# filtCycGazeNodeInWorld -> filtCycGIWDir
	filtCycGIWDirDF = calcDirFromDF(sessionDict['raw']['viewPos'],sessionDict['processed']['filtCycGazeNodeInWorldDF'],'cycGIWDir')
	sessionDict['processed'] = pd.concat([sessionDict['processed'],filtCycGIWDirDF],axis=1)

	#cycGazeVelDf = GIWtoGazeVelocity(sessionDict,sessionDict['processed']['cycGIWDir'],'cycGazeVelocity')
	#sessionDict['processed'] = pd.concat([sessionDict['processed'],cycGazeVelDf],axis=1)

	return sessionDict

def calcUnsignedAngularVelocity(vector_fr,deltaTime_fr):
	'''
	Moving window takes cosine of the dot product for adjacent values.
	Appends a 0 onto the end of the vector.
	'''
	    
	angularDistance_fr = np.array(  [ np.rad2deg(np.arccos( np.vdot( vector_fr[fr,:],vector_fr[fr-1,:])))
	     for fr in range(1,len(vector_fr))]) # if range starts at 0, fr-1 wil be -1.  Keep range from 1:len(vector)

	angularDistance_fr = np.append(0, angularDistance_fr)
	angularVelocity_fr = np.divide(angularDistance_fr,deltaTime_fr)

	return angularVelocity_fr

# def GIWtoGazeVelocity(sessionDict,dfIn,columnLabelOut):
    
# 	rawDF = sessionDict['raw']
# 	procDF = sessionDict['processed']

# 	if( columnExists(procDF, 'smiDeltaT') is False):
# 	    sessionDict = calcSMIDeltaT(sessionDict)
# 	    procDF = sessionDict['processed']

# 	angularVelocity_fr = calcUnsignedAngularVelocity(dfIn.values, procDF['smiDeltaT'].values) 

# 	return pd.DataFrame({(columnLabelOut,''):angularVelocity_fr})


def calcSMIDeltaT(sessionDict):

    sessionDict['processed']['smiDateTime'] = pd.to_datetime(sessionDict['raw'].eyeTimeStamp,unit='ns')
    deltaTime = sessionDict['processed']['smiDateTime'].diff()
    deltaTime.loc[deltaTime.dt.microseconds==0] = pd.NaT
    deltaTime = deltaTime.fillna(method='bfill', limit=1)
    sessionDict['processed']['smiDeltaT'] = deltaTime.dt.microseconds / 1000000
    return sessionDict


def GIWtoGazeVelocity(sessionDict,dfIn,columnLabelOut):
    
    rawDF = sessionDict['raw']
    procDF = sessionDict['processed']

    if( columnExists(procDF, 'smiDeltaT') is False):
        sessionDict = calcSMIDeltaT(sessionDict)
        procDF = sessionDict['processed']

    angularVelocity_fr = calcUnsignedAngularVelocity(dfIn.values, procDF['smiDeltaT'].values) 
    
    return pd.DataFrame({(columnLabelOut,''):angularVelocity_fr})



def findResidualError(projectedPoints, referencePoints):

    e2 = np.zeros((projectedPoints.shape[0],2))
    
    for i in range(projectedPoints.shape[0]):
        temp = np.subtract(projectedPoints[i], referencePoints[i])
        e2[i,:] = np.power(temp[0:2], 2)
    
    return [np.sqrt(sum(sum(e2[:])))]

def calcCalibPointMetricEyeOnScreen(sessionDict):

    calibDataFrame = sessionDict['calibration']
    
    framesPerPoint = range(100)
    startFrame = 0
    endFrame = len(calibDataFrame)
    frameIndexRange = range(startFrame, endFrame)

    cameraCenterPosition = np.array([0.0,0.0,0.0])
    planeNormal = np.array([0.0,0.0,1.0])
    eyetoScreenDistance = sessionDict['analysisParameters']['averageEyetoScreenDistance'] 
    screenCenterPosition = np.array([0.0,0.0,eyetoScreenDistance])

    calibPointMetricLocOnScreen_XYZ = np.empty([1, 3], dtype = float)
    for i in range(endFrame):
        
        lineNormal = calibDataFrame['calibrationPos'][['X','Y','Z']][i:i+1].values
        
        # TODO: I kinda cheated here by {line[0]}
        tempPos = findLinePlaneIntersection( cameraCenterPosition, lineNormal[0], 
                                               planeNormal, screenCenterPosition ) 
            
        calibPointMetricLocOnScreen_XYZ = np.vstack((calibPointMetricLocOnScreen_XYZ, tempPos))

    # TODO: I hate creating an empty variable and deleting it later on there should be a better way
    calibPointMetricLocOnScreen_XYZ = np.delete(calibPointMetricLocOnScreen_XYZ, 0, 0)
    print 'Size of TruePOR array:', calibPointMetricLocOnScreen_XYZ.shape

    # Attaching the calculated Values to the CalibDataFrame

    sessionDict['calibration'][('calibPointMetricEyeOnScreen','X')]  =  calibPointMetricLocOnScreen_XYZ[:,0]
    sessionDict['calibration'][('calibPointMetricEyeOnScreen','Y')]  =  calibPointMetricLocOnScreen_XYZ[:,1]
    sessionDict['calibration'][('calibPointMetricEyeOnScreen','Z')]  =  calibPointMetricLocOnScreen_XYZ[:,2]

    return sessionDict


def calcDirFromDF(dF1,dF2,labelOut):
    vecDF = dF2-dF1
    dirDF = vecDF.apply(lambda x: np.divide(x,np.linalg.norm(x)),axis=1)
    mIndex = pd.MultiIndex.from_tuples([(labelOut,'X'),(labelOut,'Y'),(labelOut,'Z')])
    dirDF.columns = mIndex
    return dirDF

def calcEyePositions(sessionDict):

    rawDF = sessionDict['raw']
    procDF = sessionDict['processed']
    
    ## Right and left eye positions
    #(sessionDict,dataIn,eyeString,dataOutLabel):
    #rightEyePosDf = eyeToWorld(rawDF,[0,0,0],'right','rightEyePos') 
    
    rightEyePosDf = eyeToWorld(sessionDict,[0,0,0],'right','rightEyeInWorld')
    procDF = pd.concat([procDF,rightEyePosDf],axis=1,verify_integrity=True)

    leftEyePosDf = eyeToWorld(sessionDict,[0,0,0],'left','leftEyeInWorld') 
    procDF = pd.concat([procDF,leftEyePosDf],axis=1,verify_integrity=True)
    
    sessionDict['processed'] = procDF
    
    return sessionDict


def from1x3_to_1x4(dataIn_n_xyz, eyeOffsetX_fr, numReps = 1):
    
    '''
    Converts dataIn_n_xyz into an Nx4 array, with eyeOffsetX added to the [0] column.  
    DataIn may be either a 3 element list (XYZ values) or an N x XYZ array, where N >1 (and equal to the number of rows of the original raw dataframe)

    Output is an nx4 array in which IOD has been added to the [0] column
    '''

    # If needed, tile dataIn_fr_xyz to match length of dataIn_fr_xyzw
    if( numReps == 0 ):
        
        raise NameError('numReps must be >0.')
        
    elif(numReps == 1):
        
        
        dataIn_fr_xyzw = np.tile([0, 0, 0, 1.0],[len(dataIn_n_xyz),1])
        dataIn_fr_xyzw[:,0] = eyeOffsetX_fr
        #dataIn_fr_xyzw = np.tile([eyeOffsetX, 0, 0, 1.0],[len(dataIn_n_xyz),1])
        dataIn_fr_xyzw[:,:3] = dataIn_fr_xyzw[:,:3] + dataIn_n_xyz
        
    else:
        
        dataIn_fr_xyzw = np.tile([0, 0, 0, 1.0],[numReps,1])
        dataIn_fr_xyzw[:,0] = eyeOffsetX_fr
        #dataIn_fr_xyzw = np.tile([eyeOffsetX, 0, 0, 1.0],[numReps,1])
        dataIn_fr_xyzw[:,:3] = dataIn_fr_xyzw[:,:3] + np.tile(dataIn_n_xyz,[numReps,1])

    return dataIn_fr_xyzw


# def eyeToWorld(sessionDict,dataIn,eyeString,dataOutLabel):
    
#     '''
#     This function takes XYZ data in eye centered coordinates (XYZ) and transforms it into world centered coordinates.
    
#     - rawDF must be the raw dataframe containing the transform matrix for the mainview
    
#     - dataIn may be:
#         - a dataframe of XYZ data
#         - a 3 element list of XYZ data

#     - eyeString is a string indicating which FOR contains dataIn, and may be of the values ['cyc','right','left'] 
    
#     Returns:  A multiindexed dataframe with {(label,X),(label,Y),and (label,Z)}
    
#     '''
#     rawDF = sessionDict['raw'] # monkey
    
#     ######################################################################    
#     ######################################################################
        
#     # Convert viewmat data into 4x4 transformation matrix
#     viewMat_fr_4x4 = [np.reshape(mat,[4,4]).T for mat in rawDF.viewMat.values]
    
    
#     # Convert dataIn from eye centered coordinates into head centered coordinates
#     vec_fr_XYZW = eyeToHead(sessionDict,dataIn,eyeString,'dataInHead')
    
#     # Take the dot product of vec_fr_XYZW and viewMat_fr_4x4
#     dataOut_fr_XYZ = np.array([np.dot(viewMat_fr_4x4[fr],vec_fr_XYZW.values[fr])
#                               for fr in range(len(vec_fr_XYZW['dataInHead'].values))])
    
#     # Discard the 4th column
#     dataOut_fr_XYZ = dataOut_fr_XYZ[:,:3]
    
#     # Turn it into a dataframe
#     dataOutDf = pd.DataFrame(dataOut_fr_XYZ)

#     # Rename the columns
#     mIndex = pd.MultiIndex.from_tuples([(dataOutLabel,'X'),(dataOutLabel,'Y'),(dataOutLabel,'Z')])
#     dataOutDf.columns = mIndex
    
#     return dataOutDf 

# def headToWorld(sessionDict,dataIn,dataOutLabel):
#     '''
#     This function takes XYZ data in head centered coordinates (XYZ) and transforms it into world centered coordinates.

#     - rawDF must be the raw dataframe containing the transform matrix for the mainview

#     - dataIn may be:
#         - a dataframe of XYZ data
#         - a 3 element list of XYZ data

#     - eyeString is a string indicating which FOR contains dataIn, and may be of the values ['cyc','right','left'] 

#     Returns:  A multiindexed dataframe with {(label,X),(label,Y),and (label,Z)}

#     '''

#     return eyeToWorld(sessionDict,dataIn,'cyc',dataOutLabel)

# ###########################################################################
# ###########################################################################

# def eyeOnScreenToMetricEyeOnScreen(sessionDict,dFIn,dataOutColumnLabel):
#     '''
#     Convert pixel coordinates
#     to metric locations on a screen at sessionDict['analysisParameters']['averageEyeToScreenDistance']

#     0.126,0.071 = Screen size in meters according to SMI manual
#     averageEyetoScreenDistance = 0.0725
#     Note that the output is in meters, and not normalized gaze vector
#     '''
#     x_pixel = dFIn['X']
#     y_pixel = dFIn['Y']
#     z = []


#     if(  sessionDict['analysisParameters']['hmd'].upper() == 'DK2' ):
        
#         resolution_XY = sessionDict['analysisParameters']['hmdResolution']
#         pixelSize_XY = sessionDict['analysisParameters']['hmdScreenSize']

#         x = (pixelSize_XY[0]/resolution_XY[0])*np.subtract(x_pixel, resolution_XY[0]/2.0)
#         y = (pixelSize_XY[1]/resolution_XY[1])*np.subtract(resolution_XY[1]/2.0, y_pixel) # This line is diffetent than the one in Homography.py(KAMRAN)
#         z = np.zeros(len(x_pixel))
#         averageEyetoScreenDistance = sessionDict['analysisParameters']['averageEyetoScreenDistance'] 
#         z = z + averageEyetoScreenDistance
#         print '*** eyeOnScreenToMetricEyeOnScreen(): For Dk2, using [\'analysisParameters\'][\'averageEyetoScreenDistance\']***'
    
#     else:
#         raise AssertionError('Currently only works for the Oculus DK2')


#     dFOut = []
#     dFOut = pd.DataFrame({(dataOutColumnLabel, 'X'):x,
#             (dataOutColumnLabel, 'Y'):y,
#             (dataOutColumnLabel, 'Z'):z,})

#     return dFOut

# def metricEyeOnScreenToPixels(sessionDict,dFIn,dataOutColumnLabel):
#     '''
#     Convert metric locations on a screen at sessionDict['analysisParameters']['averageEyeToScreenDistance']
#     to pixel coordinates
#     '''

#     if( sessionDict['analysisParameters']['hmd'].upper() == 'DK2' is False ):
#             raise AssertionError('Currently only works for the Oculus DK2')

#     x = dFIn['X']
#     y = dFIn['Y']
        
#     if( sessionDict['analysisParameters']['hmd'].upper() == 'DK2'):
         
#         resolution_XY = sessionDict['analysisParameters']['hmdResolution']
#         pixelSize_XY = sessionDict['analysisParameters']['hmdScreenSize']
#         x_pixel = (resolution_XY[0]/pixelSize_XY[0])*np.add(x, pixelSize_XY[0]/2.0)
#         y_pixel = (resolution_XY[1]/pixelSize_XY[1])*np.add(y, pixelSize_XY[1]/2.0)
        
#     dFOut = []
#     dFOut = pd.DataFrame({(dataOutColumnLabel, 'X'):x_pixel,
#             (dataOutColumnLabel, 'Y'):y_pixel})

#     return dFOut


# def metricEyeOnScreenToEyeInHead(sessionDict,dFIn,dataOutColumnLabel):
# 	'''
# 	Really, this just converts metricEyeONScreen into a normalized eye-in-head vector.
# 	'''
# 	#tempDF = eyeOnScreenToMetricEyeOnScreen(dFIn,dataInColumnLabel,'filtMetricEyeOnScreen','DK2')

# 	# Normalize 

# 	normXYZ = np.array([np.divide(XYZ,np.linalg.norm(XYZ)) for 
# 	                                   XYZ in dFIn.values],dtype=np.float)

# 	dFOut = pd.DataFrame(normXYZ)

# 	dFOut = dFOut.rename(columns={0: (dataOutColumnLabel,'X'), 1:(dataOutColumnLabel,'Y'), 2: (dataOutColumnLabel,'Z')})


# 	return dFOut


# def filterEyeOnScreen(sessionDict,dfIn,dataOutLabel):

# 	gazeFilter = sessionDict['analysisParameters']['gazeFilter']
# 	filterParameters = sessionDict['analysisParameters']['filterParameters']

# 	rawDF = sessionDict['raw']
# 	procDF = sessionDict['processed']
# 	dFOut = []

# 	dFOut = pd.DataFrame()

# 	if( gazeFilter == 'median' ):

# 	    assert len(filterParameters) == 1, "filterParameters of length %i, expected len of 1 for median filter." % len(filterParameters)

# 	    dFOut['X'] = dfIn['X'].rolling(filterParameters[0], min_periods = 0).median()
# 	    dFOut['Y'] = dfIn['Y'].rolling(filterParameters[0], min_periods = 0).median()

# 	elif( gazeFilter == 'average' ):

# 	    assert len(filterParameters) == 1, "filterParameters of length %i, expected len of 1 for average filter." % len(filterParameters)

# 	    dFOut['X'] = dfIn['X'].rolling(filterParameters[0], min_periods = 0).mean()
# 	    dFOut['Y'] = dfIn['Y'].rolling(filterParameters[0], min_periods = 0).mean()

# 	elif( gazeFilter == 'medianAndAverage' ):
	    
# 	    'First median, then average'
	    
# 	    assert len(filterParameters) == 2, "filterParameters of length %i, expected len of 2 for median + average filter." % len(filterParameters)
	    
# 	    dFOut['X'] = dfIn['X'].rolling(filterParameters[0], min_periods = 0).median()
# 	    dFOut['Y'] = dfIn['Y'].rolling(filterParameters[0], min_periods = 0).median()

# 	    dFOut['X'] = dfIn['X'].rolling(filterParameters[0], min_periods = 0).mean()
# 	    dFOut['Y'] = dfIn['Y'].rolling(filterParameters[0], min_periods = 0).mean()

# 	else:

# 	    raise AssertionError('Invalid filter type.  Types accepted:  {\"median\",\"average\"}')

# 	dFOut.columns = pd.MultiIndex.from_tuples([(dataOutLabel,'X'),(dataOutLabel,'Y')])

# 	return dFOut



def loadSessionDict(analysisParameters,startFresh = False,loadProcessed = False):
    '''
    If startFresh is False, attempt to read in session dict from pickle.
    If pickle does not exist, or startFresh is True, 
        - read session dict from raw data file
        - create secondary dataframes
    '''

    filePath = analysisParameters['filePath']
    fileName = analysisParameters['fileName']
    expCfgName = analysisParameters['expCfgName']
    sysCfgName = analysisParameters['sysCfgName']

    # if loadProcessed, try to load the processed dataframe
    if( startFresh is False and loadProcessed is True):

        try:
            print '***loadSessionDict: Loading preprocessed data for ' + str(fileName) + ' ***'
            processedDict = pd.read_pickle(filePath + fileName + '-proc.pickle')
            return processedDict
        except:
            raise Warning, 'loadProcessedDict: Preprocessed data not available'

    # if loadProcessed failed or not true
    # ..if startFresh is True, load raw data and crunch on that
    # if startFresh is False, try to load the pickle
    if( startFresh is True):
        sessionDict = createSecondaryDataframes(filePath,fileName,expCfgName,sysCfgName)
    else:
        try:
            sessionDict = pd.read_pickle(filePath + fileName + '.pickle')
        except:
            sessionDict = createSecondaryDataframes(filePath,fileName,expCfgName,sysCfgName)
            pd.to_pickle(sessionDict, filePath + fileName + '.pickle')

    sessionDict['analysisParameters'] = analysisParameters

    pd.to_pickle(sessionDict, filePath + fileName + '.pickle')
    return sessionDict  



def createSecondaryDataframes(filePath,fileName,expCfgName,sysCfgName):
    '''
    Separates practice and calibration trials from main dataframe.
    Reads in exp and sys config.
    '''
    sessionDf = pp.readPerformDict(filePath + fileName + ".dict")

    [sessionDf, calibDf] = seperateCalib(sessionDf)

    expConfig =  createExpCfg(filePath + expCfgName)
    sysConfig =  createSysCfg(filePath + sysCfgName)
    practiceBlockIdx = [idx for idx, s in enumerate(expConfig['experiment']['blockList']) if s == 'practice']

    [sessionDf, practiceDf] =  seperatePractice(sessionDf,practiceBlockIdx)

    sessionDf = sessionDf.reset_index()
    sessionDf = sessionDf.rename(columns = {'index':'frameNumber'})

    # ### New trial numbers
    # sessionDf['originalTrialNumber'] = sessionDf.trialNumber
    
    # gbBlock = sessionDf.groupby(['blockNumber'])
    # len(np.unique(gbBlock.get_group(0).trialNumber))

    # numTrials_bl = gbBlock.apply(lambda gr: len(np.unique(gr.trialNumber)))

    # sessionDf['trialNumber'] = sessionDf.apply(lambda r: 1+ r['originalTrialNumber'] +  numTrials_bl[range(r['blockNumber'])].sum() + r['blockNumber'] ,axis=1)
    
    #####

    trialInfoDf = expFun.initTrialInfo(sessionDf)
    
    procDataDf = expFun.initProcessedData(sessionDf)

    paddleDF   = expFun.calcPaddleBasis(sessionDf)
    procDataDf = pd.concat([paddleDF,procDataDf],axis=1)

    sessionDict = {'raw': sessionDf, 'processed': procDataDf, 'calibration': calibDf, 'practice': practiceDf, 
    'trialInfo': trialInfoDf,'expConfig': expConfig,'sysCfg': sysConfig}

    return sessionDict


### Save calibration frames in a separate dataframe

def excludeTrialType(sessionDict,typeString):

    sessionDictCopy = sessionDict.copy()

    gbTrialType = sessionDictCopy['trialInfo'].groupby('trialType')
    newDf = gbTrialType.get_group(typeString)
    sessionDictCopy['trialInfo'] = sessionDictCopy['trialInfo'].drop(gbTrialType.get_group(typeString).index)
    sessionDictCopy['trialInfo'] = sessionDictCopy['trialInfo'].reset_index()
    #sessionDict['trialInfo' + dictSublabel] = newDf.reset_index()


    sessionDictCopy['processed']['trialType'] = sessionDictCopy['raw']['trialType']
    gbProc = sessionDictCopy['processed'].groupby('trialType')
    newDf = gbProc.get_group(typeString)
    sessionDictCopy['processed'] = sessionDictCopy['processed'].drop(gbProc.get_group(typeString).index)
    sessionDictCopy['processed']=sessionDictCopy['processed'].reset_index()
    #sessionDict['proc' + dictSublabel] = newDf.reset_index()

    gbRaw = sessionDictCopy['raw'].groupby('trialType')
    newDf = gbRaw.get_group(typeString)
    sessionDictCopy['raw'] = sessionDictCopy['raw'].drop(gbRaw.get_group(typeString).index)
    sessionDictCopy['raw'] = sessionDictCopy['raw'].reset_index()
    #sessionDict['raw' + dictSublabel] = newDf.reset_index()
    
    return sessionDictCopy



def seperateCalib(sessionDf):
    calibFrames = sessionDf['trialNumber']>999
    calibDf = sessionDf[calibFrames]
    sessionDf = sessionDf.drop(sessionDf[calibFrames].index)
    return sessionDf, calibDf

def seperatePractice(sessionDf,practiceBlockIdx):
    
    practiceDf = pd.DataFrame()
    
    for bIdx in practiceBlockIdx:
    	#print 'Seperating practice block: ' + str(bIdx)    
	thisPracticeBlockDF = sessionDf[sessionDf['blockNumber']==bIdx]
	practiceDf = pd.concat([practiceDf,thisPracticeBlockDF],axis=0)
	sessionDf = sessionDf.drop(thisPracticeBlockDF.index)
        
    return sessionDf, practiceDf



def createExpCfg(expCfgPathAndName):

    """
    Parses and validates a config obj
    Variables read in are stored in configObj

    """

    print "Loading experiment config file: " + expCfgPathAndName
    
    from os import path
    filePath = path.dirname(path.abspath(expCfgPathAndName))

    # This is where the parser is called.
    expCfg = ConfigObj(expCfgPathAndName, configspec=filePath + '/expCfgSpec.ini', raise_errors = True, file_error = True)

    validator = Validator()
    expCfgOK = expCfg.validate(validator)
    if expCfgOK == True:
        print "Experiment config file parsed correctly"
    else:
        print 'Experiment config file validation failed!'
        res = expCfg.validate(validator, preserve_errors=True)
        for entry in flatten_errors(expCfg, res):
        # 1each entry is a tuple
            section_list, key, error = entry
            if key is not None:
                section_list.append(key)
            else:
                section_list.append('[missing section]')
            section_string = ', '.join(section_list)
            if error == False:
                error = 'Missing value or section.'
            print section_string, ' = ', error
        sys.exit(1)
    if expCfg.has_key('_LOAD_'):
        for ld in expCfg['_LOAD_']['loadList']:
            print 'Loading: ' + ld + ' as ' + expCfg['_LOAD_'][ld]['cfgFile']
            curCfg = ConfigObj(expCfg['_LOAD_'][ld]['cfgFile'], configspec = expCfg['_LOAD_'][ld]['cfgSpec'], raise_errors = True, file_error = True)
            validator = Validator()
            expCfgOK = curCfg.validate(validator)
            if expCfgOK == True:
                print "Experiment config file parsed correctly"
            else:
                print 'Experiment config file validation failed!'
                res = curCfg.validate(validator, preserve_errors=True)
                for entry in flatten_errors(curCfg, res):
                # each entry is a tuple
                    section_list, key, error = entry
                    if key is not None:
                        section_list.append(key)
                    else:
                        section_list.append('[missing section]')
                    section_string = ', '.join(section_list)
                    if error == False:
                        error = 'Missing value or section.'
                    print section_string, ' = ', error
                sys.exit(1)
            expCfg.merge(curCfg)

    return expCfg


def createSysCfg(sysCfgPathAndName):
    """
    Set up the system config section (sysCfg)
    """

    # Get machine name
    #sysCfgName = platform.node()+".cfg"
    
    
    

    print "Loading system config file: " + sysCfgPathAndName

    # Parse system config file
    from os import path
    filePath = path.dirname(path.abspath(sysCfgPathAndName))
    
    sysCfg = ConfigObj(sysCfgPathAndName , configspec=filePath + '/sysCfgSpec.ini', raise_errors = True)

    validator = Validator()
    sysCfgOK = sysCfg.validate(validator)

    if sysCfgOK == True:
        print "System config file parsed correctly"
    else:
        print 'System config file validation failed!'
        res = sysCfg.validate(validator, preserve_errors=True)
        for entry in flatten_errors(sysCfg, res):
        # each entry is a tuple
            section_list, key, error = entry
            if key is not None:
                section_list.append(key)
            else:
                section_list.append('[missing section]')
            section_string = ', '.join(section_list)
            if error == False:
                error = 'Missing value or section.'
            print section_string, ' = ', error
        sys.exit(1)
    return sysCfg


def dotproduct( v1, v2):
    r = sum((a*b) for a, b in zip(v1, v2))
    return r

def length(v):
    return np.sqrt(dotproduct(v, v))

def vectorAngle( v1, v2):
    r = (180.0/np.pi)*np.arccos((dotproduct(v1, v2)) / (length(v1) * length(v2)))#np.arccos((np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))))#
    return r


def createLine( point0, point1 ):
    unitVector = np.subtract(point0, point1)/length(np.subtract(point0, point1))
    return unitVector

def findLinePlaneIntersection(point_0, line, planeNormal, point_1):
    s = point_1 - point_0
    numerator = dotproduct(s, planeNormal)
    denumerator = np.inner(line, planeNormal)
    if (denumerator == 0):
        print 'No Intersection'
        return None
    d = np.divide(numerator, denumerator)
    intersectionPoint = np.multiply(d, line) + point_0
    return intersectionPoint

def findResidualError(projectedPoints, referrencePoints):
    e2 = np.zeros((projectedPoints.shape[0],2))
    for i in range(projectedPoints.shape[0]):
        temp = np.subtract(projectedPoints[i], referrencePoints[i])
        e2[i,:] = np.power(temp[0:2], 2)
    return [np.sqrt(sum(sum(e2[:])))]


def quat2transform(q):
    """
    Transform a unit quaternion into its corresponding rotation matrix (to
    be applied on the right side).
    :returns: transform matrix
    :rtype: numpy array
    """
    x, y, z, w = q
    xx2 = 2 * x * x
    yy2 = 2 * y * y
    zz2 = 2 * z * z
    xy2 = 2 * x * y
    wz2 = 2 * w * z
    zx2 = 2 * z * x
    wy2 = 2 * w * y
    yz2 = 2 * y * z
    wx2 = 2 * w * x
    rmat = np.empty((3, 3), float)
    rmat[0,0] = 1. - yy2 - zz2
    rmat[0,1] = xy2 - wz2
    rmat[0,2] = zx2 + wy2
    rmat[1,0] = xy2 + wz2
    rmat[1,1] = 1. - xx2 - zz2
    rmat[1,2] = yz2 - wx2
    rmat[2,0] = zx2 - wy2
    rmat[2,1] = yz2 + wx2
    rmat[2,2] = 1. - xx2 - yy2
    return rmat

def print_source(function):
    
    """For use inside an IPython notebook: given a module and a function, print the source code."""
    from inspect import getsource,getmodule
    from pygments import highlight
    from pygments.lexers import PythonLexer
    from pygments.formatters import HtmlFormatter
    from IPython.core.display import HTML
    
    internal_module = getmodule(function)

    return HTML(highlight(getsource(function), PythonLexer(), HtmlFormatter(full=True)))



def findFirst(dataVec,targetVal):
    '''
    Reports the first occurance of targetVal in dataVec.
    If no occurances found, returns None
    '''
    return next((fr for fr, eF in enumerate(dataVec) if eF == targetVal),False)

def findColumn(dF,label):
    '''
    Searches through first level of a multi indexed dataframe
    for column labels that contain the 'label' passed in

    '''
    colIndices = []
    colNames = []

    for cIdx in range(len(dF.columns.levels[0])):

            if label in dF.columns.levels[0][cIdx]:

                colIndices.append(cIdx)
                colNames.append(dF.columns.levels[0][cIdx])
    
    return colNames

def columnExists(dF,label):
    '''
    Returns true if columns is found.
    False if not.

    '''
    colIndices = []
    colNames = []

    for cIdx in range(len(dF.columns.levels[0])):

            if label in dF.columns.levels[0][cIdx]:

                colIndices.append(cIdx)
                colNames.append(dF.columns.levels[0][cIdx])
    
    #return colIndices,colNames

    if( len(colIndices) > 0 ):
        return True
    else:
        return False

 

def findFirstZeroCrossing(vecIn):
    '''
    This will return the index of the first zero crossing of the input vector
    '''
    return np.where(np.diff(np.sign(vecIn)))[0][0]

def findFirst(dataVec,targetVal):
    '''
    Reports the first occurance of targetVal in dataVec.
    If no occurances found, returns None
    '''
    return next((fr for fr, eF in enumerate(dataVec) if eF == targetVal),False)
