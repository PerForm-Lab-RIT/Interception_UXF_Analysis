import sys
sys.path.append("../Modules/")
sys.path.append("../")
import os

import pandas as pd
import numpy as np

from utilFunctions import *
from analysisParameters import loadParameters


from configobj import ConfigObj
from configobj import flatten_errors
from validate import Validator
	
import logging
logger = logging.getLogger(__name__)

#fmt="<%(levelname)s>%(funcName)s():%(lineno)i: %(message)s "
# fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s')
# logging.basicConfig(level=logging.INFO, format=fmt)




def processData(sessionDict,analysisParameters):

	sessionDict = calcCycEIH(sessionDict)
	sessionDict = calcGIW(sessionDict)
	sessionDict = calcCycToBallVector(sessionDict)
	sessionDict = calcBallAngularSize(sessionDict)
	sessionDict = calcGazeToBallCenterError(sessionDict,'cyc','worldUp')
	sessionDict = calcGazeToBallError(sessionDict,'cyc','worldUp')
	sessionDict = calcSphericalcoordinates(sessionDict)
	sessionDict = setGainAndPassingLoc(sessionDict)
	sessionDict = calcTrackingError(sessionDict)
	sessionDict = calcTTA(sessionDict)
	sessionDict = testAccelCompensation(sessionDict)

	pd.to_pickle(sessionDict, analysisParameters['filePath'] + analysisParameters['fileName'] + '-proc.pickle')

	logger.info('Processed data saved to ' + analysisParameters['filePath'] + analysisParameters['fileName'] + '-proc.pickle')

	return sessionDict

def calcSphericalcoordinates(sessionDict):

	proc = sessionDict['processed']
	sessionDict['processed']['cycGIW_az'] = np.rad2deg(np.arctan(proc[('cycGIWDir','X')]/proc[('cycGIWDir','Z')]))
	sessionDict['processed']['cycGIW_el']  = np.rad2deg(np.arctan(proc[('cycGIWDir','Y')]/proc[('cycGIWDir','Z')]))

	sessionDict['processed']['ball_az'] = np.rad2deg(np.arctan(proc[('cycToBallDir','X')]/proc[('cycToBallDir','Z')]))
	sessionDict['processed']['ball_el']  = np.rad2deg(np.arctan(proc[('cycToBallDir','Y')]/proc[('cycToBallDir','Z')]))
	
	logger.info('Added sessionDict[\'processed\'][\'ball_az\']')
	logger.info('Added sessionDict[\'processed\'][\'ball_el\']')
	logger.info('Added sessionDict[\'processed\'][\'cycGIW_az\']')
	logger.info('Added sessionDict[\'processed\'][\'cycGIW_el\']')
	
	return sessionDict


def calcCycEIH(sessionDictIn):
	def calcEIHByRow(row):
		
		cycEIH = (np.array(row['leftEyeInHead'].values,dtype=np.float) + 
			np.array(row['rightEyeInHead'].values,dtype=np.float))/2

		cycEIH = cycEIH / np.linalg.norm(cycEIH)
		return {('cycEyeInHead','X'): cycEIH[0],('cycEyeInHead','Y'): cycEIH[1],('cycEyeInHead','Z'): cycEIH[2]}
	
	cycEIH_fr = sessionDictIn['processed'].apply(calcEIHByRow,axis=1)
	cycEIHDf = pd.DataFrame.from_records(cycEIH_fr)
	cycEIHDf.columns = pd.MultiIndex.from_tuples(cycEIHDf.columns)
	
	logger.info('Added sessionDict[\'processed\'][\'cycEyeInHead\']')
	sessionDictIn['processed'] = sessionDictIn['processed'].combine_first(pd.DataFrame.from_records(cycEIHDf))
	
	return sessionDictIn

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
	
	
	logger.info('Added sessionDict[\'processed\'][\'cycGIWDir\']')
	return sessionDictIn

def calcCycToBallVector(sessionDict):

	cycToBallVec = np.array(sessionDict['raw']['ballPos'] - sessionDict['raw']['viewPos'],dtype=np.float )
	cycToBallDir = np.array([np.divide(XYZ,np.linalg.norm(XYZ)) for XYZ in cycToBallVec],dtype=np.float)

	sessionDict['processed'][('cycToBallDir','X')] = cycToBallDir[:,0]
	sessionDict['processed'][('cycToBallDir','Y')] = cycToBallDir[:,1]
	sessionDict['processed'][('cycToBallDir','Z')] = cycToBallDir[:,2]

	
	logger.info('Added sessionDict[\'processed\'][\'cycToBallDir\']')
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
		# Positive values means gaze is leading
		azimuthalDist = cycGIW_az - ball_az
		elevationDist = cycGIW_el- ball_el 

		angDist = np.rad2deg(np.arccos(np.dot(row['cycGIWDir'] , ballDir_XYZ)))
		
		ballRadiusDegs = np.float(row['ballRadiusDegs'])
		
		if azimuthalDist > 0:
		 	azimuthalDist -= ballRadiusDegs
		else:
		 	azimuthalDist += ballRadiusDegs

		if elevationDist > 0:
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
	
	absAzEl_XYZ = tempRawDF.apply(lambda row: calcEyeToBallDir_worldCentered(row) ,raw=True,axis=1)

	(absDist,azDist,elDist) = zip(*absAzEl_XYZ)

	outVarColName = 'cycGIWtoBallEdgeAngle'
	upVecString = 'worldUp'

	tempDict['processed'][(outVarColName,'X_' + upVecString)] = azDist
	tempDict['processed'][(outVarColName,'Y_' + upVecString)] = elDist
	tempDict['processed'][(outVarColName,'2D')] = absDist

	logger.info('Added sessionDict[\'processed\'][\'cycGIWtoBallEdgeAngle\'][\'' + outVarColName + '\']')

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
		azimuthalDist = cycGIW_az - ball_az
		elevationDist = cycGIW_el - ball_el

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

	logger.info('Added sessionDict[\'processed\'][\'cycGIWtoBallCenterAngle\'][\'' + outVarColName + '\']') 

	return tempDict


def calcBallAngularSize(sessionDict):
	
	eyeToBallDist_fr = [np.sqrt( np.sum(  np.power(bXYZ-vXYZ,2))) for bXYZ,vXYZ in zip( sessionDict['raw']['ballPos'].values,sessionDict['raw']['viewPos'].values)]
	ballRadiusM_fr = sessionDict['raw']['ballRadiusM']
	sessionDict['processed']['ballRadiusDegs'] = [np.rad2deg(np.arctan(rad/dist)) for rad, dist in zip(ballRadiusM_fr,eyeToBallDist_fr)]

	logger.info('Added sessionDict[\'processed\'][\'ballRadiusDegs\']')

	return sessionDict


def calcSMIDeltaT(sessionDict):

    sessionDict['processed']['smiDateTime'] = pd.to_datetime(sessionDict['raw'].eyeTimeStamp,unit='ns')
    deltaTime = sessionDict['processed']['smiDateTime'].diff()
    deltaTime.loc[deltaTime.dt.microseconds==0] = pd.NaT
    deltaTime = deltaTime.fillna(method='bfill', limit=1)
    sessionDict['processed']['smiDeltaT'] = deltaTime.dt.microseconds / 1000000

    logger.info('Added sessionDict[\'processed\'][\'smiDeltaT\']')

    return sessionDict


def findAnalysisWindow(trialData,analyzeUntilXSToArrival =  .4, stopAtXSToArrival = 0.2):

    ballPassesOnFr = False
    ballHitPaddleOnFr = np.where(trialData['eventFlag']=='ballOnPaddle')
    endFr = False

    if len(ballHitPaddleOnFr[0]) > 0:
        ballHitPaddleOnFr = ballHitPaddleOnFr[0][0]
        ballPassesOnFr = False
        endFr = ballHitPaddleOnFr
    else:
        ballHitPaddleOnFr = False
        ballPassesOnFr = np.where(trialData[('ballPos','Z')]<0)

        if len(ballPassesOnFr[0]) > 0:
            ballPassesOnFr = ballPassesOnFr[0][0]
            endFr = ballPassesOnFr
        else:
            # Sometimes the ball seems to stop in place upon collision.  I'm not sure what's going on there.
            endFr = np.where(trialData[('ballPos','Z')].diff()==0)[0][0]

	# Choose the min:  frame on which expansion stops, or when the ball was caught
	initTTC = (trialData[('ballInitialPos','Z')].iloc[1] / -trialData[('ballInitialVel','Z')].iloc[1])
	
	if stopAtXSToArrival == False:

		stopAtXSToArrival = trialData['noExpansionForLastXSeconds'].iloc[1]
	#noExpTimeS = trialData['noExpansionForLastXSeconds'].iloc[1]

	expStopsFr = np.where( np.cumsum(trialData['frameTime'][1:].diff()) > initTTC-stopAtXSToArrival)[0][0]
	endFr = np.min([expStopsFr,endFr])

    timeToArrival_fr = trialData['frameTime'] - trialData['frameTime'].iloc[endFr]
    startFr = np.where(timeToArrival_fr>-analyzeUntilXSToArrival)[0][0]

    return startFr,endFr


def setGainAndPassingLoc(sessionDict):

    expansionGain_tr = [np.nan] * len(sessionDict['trialInfo'])
    passingLocX_tr = [np.nan] * len(sessionDict['trialInfo'])

    for (blockNum,trialNum), trialData in sessionDict['processed'].groupby(('blockNumber','trialNumber')):

        tInfoIloc = np.intersect1d(np.where(sessionDict['trialInfo']['blockNumber']==blockNum),
                               np.where(sessionDict['trialInfo']['trialNumber']==trialNum))[0]

        tt = trialData.trialType.iloc[1]
        expansionGain_tr[tInfoIloc] = sessionDict['expConfig']['trialTypes'][tt]['expansionGain']
        passingLocX_tr[tInfoIloc] = sessionDict['expConfig']['trialTypes'][tt]['passingLocNormX']

    sessionDict['trialInfo']['expansionGain'] = np.array(expansionGain_tr,dtype=np.float)
    sessionDict['trialInfo']['passingLocX'] = np.array(passingLocX_tr,dtype=np.float)

    logger.info('Added sessionDict[\'trialInfo\'][\'expansionGain\']')
    logger.info('Added sessionDict[\'trialInfo\'][\'passingLocX\']')
    
    return sessionDict

def calcTrackingError(sessionDict, analyzeUntilXSToArrival =  .4, stopAtXSToArrival = 0.2):

    meanEyeToBallEdgeAz_tr = [np.nan] * len(sessionDict['trialInfo'])
    meanEyeToBallCenterAz_tr = [np.nan] * len(sessionDict['trialInfo'])
    ballChangeInRadiusDegs_tr = [np.nan] * len(sessionDict['trialInfo'])
    
    for (blockNum,trialNum), trialData in sessionDict['processed'].groupby(('blockNumber','trialNumber')):

        tInfoIloc = np.intersect1d(np.where(sessionDict['trialInfo']['blockNumber']==blockNum),
                               np.where(sessionDict['trialInfo']['trialNumber']==trialNum))[0]

        # Azimuthal error to the edge and center of the ball
        (startFr, endFr) = findAnalysisWindow(trialData,analyzeUntilXSToArrival, stopAtXSToArrival)
        endFr = endFr-1
        meanEyeToBallEdgeAz_tr[tInfoIloc] = np.mean(trialData[('cycGIWtoBallEdgeAngle', 'X_worldUp')].iloc[startFr:endFr])
        meanEyeToBallCenterAz_tr[tInfoIloc] = np.mean(trialData[('cycGIWtoBallCenterAngle', 'X_worldUp')].iloc[startFr:endFr])
        
        radiusAtStart = sessionDict['processed']['ballRadiusDegs'].iloc[startFr]
        radiusAtEnd = sessionDict['processed']['ballRadiusDegs'].iloc[endFr]
        ballChangeInRadiusDegs_tr[tInfoIloc] = radiusAtEnd - radiusAtStart
    
    
    sessionDict['trialInfo']['ballChangeInRadiusDegs'] = ballChangeInRadiusDegs_tr
    sessionDict['trialInfo']['meanEyeToBallEdgeAz'] = meanEyeToBallEdgeAz_tr
    sessionDict['trialInfo']['meanEyeToBallCenterAz'] = meanEyeToBallCenterAz_tr

    logger.info('Added sessionDict[\'trialInfo\'][\'meanEyeToBallEdgeAz\']')
    logger.info('Added sessionDict[\'trialInfo\'][\'meanEyeToBallCenterAz\']')
    logger.info('Added sessionDict[\'trialInfo\'][\'ballChangeInRadiusDegs\']')
    
    return sessionDict

def calcTTA(sessionDict):
    def calcTTAForRow(row):
        ballDistToEnd = np.sqrt(np.sum((row['ballFinalPos'] - row['ballPos'])**2))
        ballVel = np.sqrt(np.sum(row['ballVel']**2))
        tta = ballDistToEnd / ballVel
        return tta

    sessionDict['processed']['timeToArrival'] = sessionDict['raw'].apply( lambda row: calcTTAForRow(row),axis=1)
    return sessionDict


def testAccelCompensation(sessionDict, winSize = 7, polyorder = 3):

	from scipy.signal import savgol_filter

	proc = sessionDict['processed']
	sampleTime = sessionDict['raw']['frameTime'][2:].diff().mode()
	if len(sampleTime) > 1: sampleTime=sampleTime[0]
	
	proc[('cycGIWDir','X')]
	proc[('cycToBallDir','X')]

	cycAz = savgol_filter(proc['cycGIW_az'], winSize, polyorder, deriv=1, delta= sampleTime, axis=0, mode='constant', cval=1.0)
	cycEl = savgol_filter(proc['cycGIW_el'], winSize, polyorder, deriv=1, delta= sampleTime, axis=0, mode='constant', cval=1.0)

	ballAz = savgol_filter(proc['ball_az'], winSize, polyorder, deriv=1, delta= sampleTime, axis=0, mode='constant', cval=1.0)
	ballEl = savgol_filter(proc['ball_el'], winSize, polyorder, deriv=1, delta= sampleTime, axis=0, mode='constant', cval=1.0)

	sessionDict['processed']['cycToBallVel_az'] = ballAz
	sessionDict['processed']['cycToBallVel_el'] = ballEl

	sessionDict['processed']['cycGIWVel'] = np.sqrt(np.sum(np.power([cycAz,cycEl],2),0))
	sessionDict['processed']['cycToBallVel'] = np.sqrt(np.sum(np.power([ballAz,ballEl],2),0))

	sessionDict['processed']['cycToBallAcc_az'] = sessionDict['processed']['cycToBallVel_az'].diff().shift(-1)

	##

	gbProc = sessionDict['processed'].groupby(('blockNumber','trialNumber'))

	ballAzAtWinEnd_velPred = []
	ballAzAtWinEnd_empirical = []
	gazeAzAtWinEnd = []

	for trIdx, trProc in gbProc:

	    (startFr, endFr) = findAnalysisWindow(trProc)

	    iPos = trProc['cycGIW_az'].iloc[startFr]
	    iVel = trProc['cycToBallVel_az'].iloc[startFr]
	    winDur = trProc['frameTime'].iloc[endFr] - trProc['frameTime'].iloc[startFr]

	    ballAzAtWinEnd_velPred.append( iPos + iVel*winDur )
	    ballAzAtWinEnd_empirical.append(trProc['ball_az'].iloc[endFr])
	    gazeAzAtWinEnd.append(trProc['cycGIW_az'].iloc[endFr])
	    
	    #print '%1.2f -- %1.2f' % (iPos + iVel*winDur , trProc['ball_az'].iloc[endFr])
	sessionDict['trialInfo']['ballAzAtWinEnd_velPred'] = ballAzAtWinEnd_velPred
	sessionDict['trialInfo']['ballAzAtWinEnd_empirical'] = ballAzAtWinEnd_empirical
	sessionDict['trialInfo']['giwAzAtWinEnd'] = gazeAzAtWinEnd

	trInfo = sessionDict['trialInfo']

	sessionDict['trialInfo']['empiricalError'] = trInfo['giwAzAtWinEnd'] - trInfo['ballAzAtWinEnd_empirical']
	sessionDict['trialInfo']['velPredError'] = trInfo['giwAzAtWinEnd'] - trInfo['ballAzAtWinEnd_velPred']

	logger.info('Added sessionDict[\'trialInfo\'][\'empiricalError\']')
	logger.info('Added sessionDict[\'trialInfo\'][\'velPredError\']')
	return sessionDict
