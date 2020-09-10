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

	sessionDict['trialInfo'] = sessionDict['trialInfo'].sort_values(['blockNumber','trialNumber'])
	sessionDict['trialInfo']['fileName'] = analysisParameters['fileName']

	sessionDict = calcCatchingPlane(sessionDict)
	sessionDict = findLastFrame(sessionDict)
	sessionDict = calcCatchingError(sessionDict)

	sessionDict = gazeAnalysisWindow(sessionDict)
	sessionDict = calcCycEIH(sessionDict)
	sessionDict = calcGIW(sessionDict)
	sessionDict = calcCycToBallVector(sessionDict)
	sessionDict = calcBallAngularSize(sessionDict)
	sessionDict = calcSphericalcoordinates(sessionDict)
	sessionDict = setGainAndPassingLoc(sessionDict)

	# start here
	sessionDict = calcTrackingError(sessionDict) # Done
	sessionDict = calcTTA(sessionDict)
	#sessionDict = testAccelCompensation(sessionDict)

	sessionDict = vectorMovementModel(sessionDict)


	pd.to_pickle(sessionDict, analysisParameters['filePath'] + analysisParameters['fileName'] + '-proc.pickle')

	logger.info('Processed data saved to ' + analysisParameters['filePath'] + analysisParameters['fileName'] + '-proc.pickle')

	return sessionDict


def projectTrajectories(sessionDictIn):
	
	def ballPosToDirInHead(rowIn):
		mat =[]
		for i in range(16):
			mat.append(rowIn["cycInverseMat"][str(i)])

		# rearrange and transpose
		invHeadTransform_4x4 = np.reshape(mat,[4,4]).T

		# Add a 1 to convert to homogeneous coordinates
		ballPos_XYZW = np.array(np.hstack([rowIn['ballPos'].values,1]),dtype=np.float)

		# Take the dot product!
		ballInHead_XYZW = np.dot( invHeadTransform_4x4,ballPos_XYZW)
		ballDirInHead_XYZ = ballInHead_XYZW[:3] / np.linalg.norm(ballInHead_XYZW[:3])
		return tuple(ballDirInHead_XYZ)
		
	# Eye on screen @ 1 meter
	proc = sessionDictIn['processed']
	cycEyeInHead_xyz = proc['cycEyeInHead'].values
	cycEyeInHead_xyz = np.array([cycEyeInHead_xyz[:,0],cycEyeInHead_xyz[:,1],cycEyeInHead_xyz[:,2]]).T
	eos_XYZ = pd.DataFrame(cycEyeInHead_xyz)
	eos_XYZ.columns = pd.MultiIndex.from_tuples([('eyeOnScreen','X'),('eyeOnScreen','Y'),('eyeOnScreen','Z')])
	eos_XYZ.index = proc.index
	eos_XYZ = eos_XYZ.apply(lambda row: row['eyeOnScreen'] *  (1.0/row[('eyeOnScreen', 'Z') ] ),axis=1)
	eos_XYZ.columns = pd.MultiIndex.from_tuples([('eyeOnScreen','X'),('eyeOnScreen','Y'),('eyeOnScreen','Z')])
	proc = proc.combine_first(eos_XYZ)
	
	# Ball in head
	bihDf = [ballPosToDirInHead(rowIn) for i, rowIn in proc.iterrows()]
	bihDf = np.array(bihDf)
	bihDf[bihDf[:,2]<0,:] = np.nan
	bihDf = pd.DataFrame(bihDf)
	bihDf.columns = pd.MultiIndex.from_tuples([('ballDirInHead','X'),('ballDirInHead','Y'),('ballDirInHead','Z')])
	bihDf.set_index(proc.index,inplace=True)
	proc = proc.combine_first(bihDf)
	
	# Ball on screen @ 1 meter
	bos_XYZ = proc['ballDirInHead'].values
	bos_XYZ = np.array([bos_XYZ[:,0],bos_XYZ[:,1],bos_XYZ[:,2]]).T
	bos_XYZ = pd.DataFrame(bos_XYZ)
	bos_XYZ.columns = pd.MultiIndex.from_tuples([('ballOnScreen','X'),('ballOnScreen','Y'),('ballOnScreen','Z')])
	bos_XYZ.index = proc.index
	bos_XYZ = bos_XYZ.apply(lambda row: row['ballOnScreen'] *  (1.0/row[('ballOnScreen', 'Z') ] ),axis=1)
	bos_XYZ.columns = pd.MultiIndex.from_tuples([('ballOnScreen','X'),('ballOnScreen','Y'),('ballOnScreen','Z')])
	proc = proc.combine_first(bos_XYZ)
	
	sessionDictIn['processed'] = proc
	
	logger.info('Added sessionDict[\'processed\'][\'ballOnScreen\']')
	logger.info('Added sessionDict[\'processed\'][\'ballDirInHead\']')
	logger.info('Added sessionDict[\'processed\'][\'eyeOnScreen\']')
	
	return sessionDictIn
	

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
		
		cycEIH = np.array(row['leftEyeInHead'].values,dtype=np.float) + np.array(row['rightEyeInHead'].values,dtype=np.float)

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


def calcTrackingError(sessionDict):

	meanEyeToBallEdgeAz_tr = [np.nan] * len(sessionDict['trialInfo'])
	meanEyeToBallCenterAz_tr = [np.nan] * len(sessionDict['trialInfo'])
	
	ballChangeInRadiusDegs_tr = [np.nan] * len(sessionDict['trialInfo'])

	meanEyeToBallEdge_tr = [np.nan] * len(sessionDict['trialInfo'])
	meanEyeToBallCenter_tr = [np.nan] * len(sessionDict['trialInfo'])

	for (blockNum,trialNum), trialData in sessionDict['processed'].groupby(('blockNumber','trialNumber')):

		tInfoIloc = np.intersect1d(np.where(sessionDict['trialInfo']['blockNumber']==blockNum),
							   np.where(sessionDict['trialInfo']['trialNumber']==trialNum))[0]
		
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
		trInfo = sessionDict['trialInfo'].groupby(('blockNumber','trialNumber')).get_group((blockNum,trialNum))
		startFr = int(trInfo['analysisStartFr'])
		endFr = int(trInfo['analysisEndFr'])
		endFr = endFr-1

		absAzEl_XYZ = sessionDict['processed'].iloc[startFr:endFr].apply(lambda row: calcEyeToBallCenter(row), axis=1)
		(azDist,elDist) = zip(*absAzEl_XYZ)

		meanEyeToBallCenterAz_tr[tInfoIloc] = np.mean(azDist)
		meanEyeToBallCenter_tr[tInfoIloc] = np.nanmean([np.sqrt(np.float(fr[0]*fr[0] + fr[1]* fr[1])) for fr in absAzEl_XYZ ])

		absAzEl_XYZ = sessionDict['processed'].iloc[startFr:endFr].apply(lambda row: calcEyeToBallEdge(row), axis=1)
		(azDist,elDist) = zip(*absAzEl_XYZ)
		meanEyeToBallEdgeAz_tr[tInfoIloc] = np.mean(azDist)
		meanEyeToBallEdge_tr[tInfoIloc] = np.nanmean([np.sqrt(np.float(fr[0]*fr[0] + fr[1]* fr[1])) for fr in absAzEl_XYZ ])

		radiusAtStart = sessionDict['processed']['ballRadiusDegs'].iloc[startFr]
		radiusAtEnd = sessionDict['processed']['ballRadiusDegs'].iloc[endFr]
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

def gazeAnalysisWindow(sessionDict, 
	analyzeUntilXSToArrival =  .3, 
	stopAtXSToArrival = 0.1):
	
	startFr_fr = []
	endFr_fr = []
	
	gbTrials = sessionDict['processed'].groupby(('blockNumber','trialNumber'))
	
	for (blockNum,trNum), trialData in gbTrials:
		
		trInfo = sessionDict['trialInfo'].groupby(('blockNumber','trialNumber')).get_group((blockNum,trNum))
		endFr = int(trInfo['passVertPlaneAtPaddleFr'])

		initTTC = (trialData[('ballInitialPos','Z')].iloc[0] / -trialData[('ballInitialVel','Z')].iloc[0])

		if stopAtXSToArrival == False:

			stopAtXSToArrival = trialData['noExpansionForLastXSeconds'].iloc[1]
		#noExpTimeS = trialData['noExpansionForLastXSeconds'].iloc[1]

		expStopsFr = np.where( np.cumsum(trialData['frameTime'][1:].diff()) > initTTC-stopAtXSToArrival)[0][0]
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


def calcTTA(sessionDict):
	def calcTTAForRow(row):
		ballDistToEnd = np.sqrt(np.sum((row['ballFinalPos'] - row['ballPos'])**2))
		ballVel = np.sqrt(np.sum(row['ballVel']**2))
		tta = ballDistToEnd / ballVel
		return tta

	sessionDict['processed']['timeToArrival'] = sessionDict['raw'].apply( lambda row: calcTTAForRow(row),axis=1)
	return sessionDict


# def vectorMovementModel( sessionDict, winSize = 7, polyorder = 3):

# 	from scipy.signal import savgol_filter
# 	sampleTime = sessionDict['raw']['frameTime'].diff().mode()
# 	if len(sampleTime) > 1: sampleTime=sampleTime[0]

# 	ballWinStart_AzEl_tr = []
# 	ballWinEnd_AzEl_tr = []
# 	ballWinEnd_AzEl_tr
# 	ballAtWinEndVelPred_AzEl_tr = []
# 	ballAtWinEndCurvilinearVelPred_AzEl_tr = []

# 	ballWinVel_AzEl_tr = []
# 	gazeWinStart_AzEl_tr = []
# 	gazeWinEnd_AzEl_tr = []
# 	winDurSeconds_tr = []


# 	for (bNum,tNum), tr in sessionDict['processed'].groupby(('blockNumber','trialNumber')):

# 		# Calculate velocity
# 		cycToBallVel_az= savgol_filter(tr['ball_az'], winSize, polyorder, deriv=1, 
# 											  delta= sampleTime, axis=0, mode='constant', cval=1.0)

# 		cycToBallVel_el = savgol_filter(tr['ball_el'], winSize, polyorder, deriv=1, 
# 											  delta= sampleTime, axis=0, mode='constant', cval=1.0)
	
		
# 		trInfo = sessionDict['trialInfo'].groupby(('blockNumber','trialNumber')).get_group((bNum,tNum))
# 		startIdx = int(trInfo['analysisStartFr'])
# 		endIdx = int(trInfo['analysisEndFr'])

# 		windowFr = tr.index[startIdx:endIdx]
# 		wStartFr = windowFr[0]
# 		wEndFr = windowFr[-1]
		
# 		from scipy.stats import linregress
		
# 		reg = linregress(tr['ball_az'][startIdx:endIdx],tr['ball_el'][startIdx:endIdx])

# 		x1 = tr['ball_az'][wStartFr]
# 		y1 = tr['ball_el'][wStartFr]

# 		x2 = tr['ball_az'][wEndFr]
# 		y2 = tr['ball_el'][wEndFr]

# 		ballDir_azEl = [x2-x1,y2-y1] / np.linalg.norm([x2-x1,y2-y1])
		
# 		ballWinStart_AzEl_tr.append([tr['ball_az'][wStartFr], tr['ball_el'][wStartFr]])
# 		ballWinEnd_AzEl_tr.append([tr['ball_az'][wEndFr], tr['ball_el'][wEndFr]])
		
# 		ballVel = np.sqrt(np.sum(np.power([cycToBallVel_az[startIdx], cycToBallVel_el[startIdx]],2)))
# 		winDurSeconds_tr.append(tr['frameTime'][wEndFr] - tr['frameTime'][wStartFr])
# 		ballDegsMovement = winDurSeconds_tr[-1] * ballVel

# 		ballAtWinEndVelPred_AzEl_tr.append(list(ballWinStart_AzEl_tr[-1] + np.multiply(ballDir_azEl,ballDegsMovement)))
		
# 		gazeWinStart_AzEl_tr.append([tr['cycGIW_az'][wStartFr], tr['cycGIW_el'][wStartFr]])
# 		gazeWinEnd_AzEl_tr.append([tr['cycGIW_az'][wEndFr], tr['cycGIW_el'][wEndFr]])

# 		## Constant vel along ACTUAL trajectory (not a linear fit)
# 		distTrav_fr = np.cumsum(np.sqrt( np.diff(np.array(tr['ball_az'][windowFr]))**2 + 
# 		                             np.diff(np.array(tr['ball_el'][windowFr]))**2))

# 		constVelCurvFrame =  wStartFr + np.where(distTrav_fr>ballDegsMovement)[0][0]
# 		ballAtWinEndCurvilinearVelPred_AzEl = [tr['ball_az'][constVelCurvFrame],tr['ball_el'][constVelCurvFrame]]
# 		ballAtWinEndCurvilinearVelPred_AzEl_tr.append(ballAtWinEndCurvilinearVelPred_AzEl)



				
# 	# 1 Save start of ball vector
# 	# 2 Save end of gaze vector (observed, and constant vel model)
# 	# 3 Compare gaze endpoint to observed endpoint and constant vel model endpoint

# 	# Store vector endpoints
# 	sessionDict['trialInfo'] = sessionDict['trialInfo'].sort_values(['blockNumber','trialNumber'])
# 	sessionDict['trialInfo']['ballWinStart_AzEl'] = ballWinStart_AzEl_tr
# 	sessionDict['trialInfo']['ballWinEnd_AzEl'] = ballWinEnd_AzEl_tr
	
# 	sessionDict['trialInfo']['ballAtWinEndVelPred_AzEl'] = ballAtWinEndVelPred_AzEl_tr
# 	sessionDict['trialInfo']['ballAtWinEndVelPredB_AzEl'] = ballAtWinEndCurvilinearVelPred_AzEl_tr


# 	sessionDict['trialInfo']['gazeWinStart_AzEl'] = gazeWinStart_AzEl_tr
# 	sessionDict['trialInfo']['gazeWinEnd_AzEl'] = gazeWinEnd_AzEl_tr



# 	# Calculate distance from gaze endpoint to ball endpoint, and to constant velocity ball movement model
# 	ballAzEl_tr = np.array([[azEl[0],azEl[1]] for azEl in sessionDict['trialInfo']['ballWinEnd_AzEl']])
	
# 	velModelAzEl_tr = np.array([[azEl[0],azEl[1]] for azEl in sessionDict['trialInfo']['ballAtWinEndVelPred_AzEl']])
# 	velModelBAzEl_tr = np.array([[azEl[0],azEl[1]] for azEl in sessionDict['trialInfo']['ballAtWinEndVelPredB_AzEl']])

# 	gazeAzEl_tr = np.array([[azEl[0],azEl[1]] for azEl in sessionDict['trialInfo']['gazeWinEnd_AzEl']])

# 	sessionDict['trialInfo']['observedError']  = np.sqrt( np.sum(np.power(gazeAzEl_tr-ballAzEl_tr,2),axis=1))
# 	sessionDict['trialInfo']['velPredError']  = np.sqrt( np.sum(np.power(gazeAzEl_tr-velModelAzEl_tr,2),axis=1))
# 	sessionDict['trialInfo']['velPredErrorB']  = np.sqrt( np.sum(np.power(gazeAzEl_tr-velModelBAzEl_tr,2),axis=1))
	
# 	# Update log
# 	logger.info('Added sessionDict[\'trialInfo\'][\'ballWinStart_AzEl\']')
# 	logger.info('Added sessionDict[\'trialInfo\'][\'ballWinEnd_AzEl\']')
# 	logger.info('Added sessionDict[\'trialInfo\'][\'ballWinVel_AzEl\']')
	
# 	logger.info('Added sessionDict[\'trialInfo\'][\'gazeWinStart_AzEl\']')
# 	logger.info('Added sessionDict[\'trialInfo\'][\'gazeWinEnd_AzEl\']')
	
# 	logger.info('Added sessionDict[\'trialInfo\'][\'observedError\']')
# 	logger.info('Added sessionDict[\'trialInfo\'][\'velPredError\']')
# 	logger.info('Added sessionDict[\'trialInfo\'][\'velPredErrorB\']')

# 	return sessionDict


def vectorMovementModel( sessionDict, interpResMs = 1. / 1000, winSize = 7, polyorder = 3):

	from scipy.signal import savgol_filter
	sampleTime = sessionDict['raw']['frameTime'].diff().mode()
	if len(sampleTime) > 1: sampleTime=sampleTime[0]

	ballWinStart_AzEl_tr = []
	ballWinEnd_AzEl_tr = []
	ballWinEnd_AzEl_tr
	ballAtWinEndVelPred_AzEl_tr = []
	ballAtWinEndCurvilinearVelPred_AzEl_tr = []

	ballWinVel_AzEl_tr = []
	gazeWinStart_AzEl_tr = []
	gazeWinEnd_AzEl_tr = []
	winDurSeconds_tr = []


	for (bNum,tNum), tr in sessionDict['processed'].groupby(('blockNumber','trialNumber')):
		    
		trInfo = sessionDict['trialInfo'].groupby(('blockNumber','trialNumber')).get_group((bNum,tNum))

		# Calculate velocity
		cycToBallVel_az = savgol_filter(tr['ball_az'], winSize, polyorder, deriv=1, 
		                                      delta= sampleTime, axis=0, mode='constant', cval=1.0)

		cycToBallVel_el = savgol_filter(tr['ball_el'], winSize, polyorder, deriv=1, 
		                                      delta= sampleTime, axis=0, mode='constant', cval=1.0)

		startIdx = int(trInfo['analysisStartFr'])
		endIdx = int(trInfo['analysisEndFr']) 

		windowFr = tr.index[startIdx:endIdx]
		wStartFr = windowFr[0]
		wEndFr = windowFr[-1]

		###  Interpolate
		time_fr = np.array(tr['frameTime'][windowFr]- tr['frameTime'][windowFr[0]],np.float)
		interpTime_fr = np.arange(0,time_fr[-1],interpResMs)

		distTrav_fr = np.cumsum(np.sqrt( np.diff(np.array(tr['ball_az'][windowFr]))**2 + 
		                             np.diff(np.array(tr['ball_el'][windowFr]))**2))
		 
		distTrav_fr = np.hstack([0, distTrav_fr])
		interpDist_fr = np.interp(interpTime_fr, time_fr, distTrav_fr)
		interpAz_fr = np.interp(interpTime_fr, time_fr, np.array(tr['ball_az'][windowFr]))
		interpEl_fr = np.interp(interpTime_fr, time_fr, np.array(tr['ball_el'][windowFr]))

		# Find where the ball / gaze actually ended up
		ballWinStart_AzEl_tr.append([tr['ball_az'][wStartFr], tr['ball_el'][wStartFr]])
		ballWinEnd_AzEl_tr.append([tr['ball_az'][wEndFr], tr['ball_el'][wEndFr]])
		
		gazeWinStart_AzEl_tr.append([tr['cycGIW_az'][wStartFr], tr['cycGIW_el'][wStartFr]])
		gazeWinEnd_AzEl_tr.append([tr['cycGIW_az'][wEndFr], tr['cycGIW_el'][wEndFr]])

		## Fixed path, constant speed model
		winDurSeconds = tr['frameTime'][wEndFr] - tr['frameTime'][wStartFr]
		ballVel = np.sqrt(np.sum(np.power([cycToBallVel_az[startIdx], cycToBallVel_el[startIdx]],2)))
		ballDegsMovement = winDurSeconds * ballVel
		constVelCurvFrame =  np.where(interpDist_fr > ballDegsMovement)[0][0]

		ballAtWinEndVelPred_AzEl_tr.append([interpAz_fr[constVelCurvFrame], interpEl_fr[constVelCurvFrame]])
	
	# 1 Save start of ball vector
	# 2 Save end of gaze vector (observed, and constant vel model)
	# 3 Compare gaze endpoint to observed endpoint and constant vel model endpoint

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
	
	# Update log
	logger.info('Added sessionDict[\'trialInfo\'][\'ballWinStart_AzEl\']')
	logger.info('Added sessionDict[\'trialInfo\'][\'ballWinEnd_AzEl\']')
	logger.info('Added sessionDict[\'trialInfo\'][\'ballWinVel_AzEl\']')
	
	logger.info('Added sessionDict[\'trialInfo\'][\'gazeWinStart_AzEl\']')
	logger.info('Added sessionDict[\'trialInfo\'][\'gazeWinEnd_AzEl\']')
	
	logger.info('Added sessionDict[\'trialInfo\'][\'observedError\']')
	logger.info('Added sessionDict[\'trialInfo\'][\'velPredError\']')

	return sessionDict
	
def calcCatchingPlane(sessionDict):

	paddleDf = pd.DataFrame()
	paddleDf.index.name = 'frameNum'

	proc = sessionDict['processed']

	########################################################
	### Calc paddle-to-ball vector

	paddleToBallVec_fr_XYZ = proc['ballPos'].values - proc['paddlePos'].values

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

	paddleToBallDirXZDf = pd.DataFrame([np.cross([0,1,0],xyz) for xyz in paddleToBallDirDf.values])
	paddleToBallVecXZDf = paddleToBallDirXZDf.rename(columns={0: ('paddleToBallDirXZ','X'), 1:('paddleToBallDirXZ','Y'), 2: ('paddleToBallDirXZ','Z')})
	paddleToBallDirXZDf = paddleToBallVecXZDf.apply(lambda x: np.divide(x,np.linalg.norm(x)),axis=1)
	paddleDf = pd.concat([paddleDf,paddleToBallDirXZDf],axis=1,verify_integrity=True)
	
	sessionDict['processed'] = pd.concat([proc,paddleDf],axis=1)
	
	logger.info('Added sessionDict[\'processed\'][\'paddleToBallDir\']')
	logger.info('Added sessionDict[\'processed\'][\'paddleToBallVec\']')
	logger.info('Added sessionDict[\'processed\'][\'paddleToBallDirXZ\']')

	return sessionDict


def calcCatchingError(sessionDict):

	################################################################################
	### Catching error: passVertPlaneAtPaddleErr X, Y, and 2D

	ballInPaddlePlaneX_fr = []
	ballInPaddlePlaneY_fr = []
	paddleToBallDist_fr = []

	for (blockNum,trNum), tProcData in sessionDict['processed'].groupby(('blockNumber','trialNumber')):
		
		fr = sessionDict['trialInfo'].groupby(('blockNumber','trialNumber')).get_group((blockNum,trNum))['passVertPlaneAtPaddleFr']
		row = tProcData.iloc[fr]
		bXYZ = row.ballPos.values
		pXYZ = row.paddlePos.values

		#  Ball trajectory direction
		ballTrajDir_XYZ = (tProcData['ballPos'].iloc[10] - tProcData['ballPos'].iloc[1]) / np.linalg.norm(tProcData['ballPos'].iloc[10] - tProcData['ballPos'].iloc[1])
		ballTrajDir_XYZ = np.array(ballTrajDir_XYZ,dtype=np.float)
		paddleYDir_xyz = [0,1,0]
		paddleXDir_xyz = np.cross(-ballTrajDir_XYZ,paddleYDir_xyz)
		paddleToBallVec_fr_XYZ = tProcData['ballPos'].values - tProcData['paddlePos'].values

		ballRelToPaddle_xyz = np.array(bXYZ-pXYZ).T
		xErr = np.dot(paddleXDir_xyz,ballRelToPaddle_xyz)
		yErr = np.dot(paddleYDir_xyz,ballRelToPaddle_xyz)

		ballInPaddlePlaneX_fr.append(xErr)
		ballInPaddlePlaneY_fr.append(yErr)
		paddleToBallDist_fr.append(np.sqrt(np.power(xErr,2)+np.power(yErr,2)))

	sessionDict['trialInfo'][('catchingError','X')] = ballInPaddlePlaneX_fr
	sessionDict['trialInfo'][('catchingError','Y')] = ballInPaddlePlaneY_fr
	sessionDict['trialInfo'][('catchingError','2D')] =  paddleToBallDist_fr
	
	return sessionDict



def findLastFrame(sessionDict):

	################################################################################

	rawDF = sessionDict['raw']
	procDF = sessionDict['processed']
	trialInfoDF = sessionDict['trialInfo']

	def findFirstZeroCrossing(vecIn):
		'''
		This will return the index of the first zero crossing of the input vector
		'''
		return np.where(np.diff(np.sign(vecIn)))[0][0]

	gbTrials = procDF.groupby(('blockNumber','trialNumber'))
	arriveAtPaddleFr_tr = []

	for (blockNum,trNum), trialData in gbTrials:
		
	#trialData = gbTrials.get_group((bNum,tNum)) ###
		ballPassesOnFr = False
		ballHitPaddleOnFr = np.where(trialData['eventFlag']=='ballOnPaddle')
		endFr = False

		if len(ballHitPaddleOnFr[0]) > 0:
			#print('Eventflag method')
			# If it collided with paddle, use that frame
			ballHitPaddleOnFr = ballHitPaddleOnFr[0][0]
			ballPassesOnFr = False
			endFr = ballHitPaddleOnFr
		else:

			ballHitPaddleOnFr = False
			ballPassesOnFr = np.where(trialData[('ballPos','Z')]<0)

			if len(ballPassesOnFr[0]) > 0:
				#print('Calculated method A')

				def dotBallPaddle(rowIn):
					a = np.array(rowIn['paddleToBallDir'].values,dtype=np.float)
					b = np.array(rowIn['ballInitialPos'] - rowIn['paddlePos'] )
					c = np.dot(a,b) / np.linalg.norm(b)
					return c

				dott = trialData.apply(lambda row: dotBallPaddle(row),axis=1)
				endFr = np.where(dott<0)[0][0]-1
			else:
				#print('Calculated method B')
				# Sometimes the ball seems to stop in place upon collision.  I'm not sure what's going on there.
				endFr = np.where(trialData[('ballPos','Z')].diff()==0)[0][0]
			
			
		arriveAtPaddleFr_tr.append(endFr)

	sessionDict['trialInfo']['passVertPlaneAtPaddleFr'] = np.array(arriveAtPaddleFr_tr,dtype=int)
	
	logger.info('Added sessionDict[\'trialInfo\'][\'passVertPlaneAtPaddleFr\']')
	return sessionDict



