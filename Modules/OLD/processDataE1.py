import sys
sys.path.append("../Modules/")
sys.path.append("../")
import os

import pandas as pd
import numpy as np

from utilFunctions import *

# from analysisParameters import loadParameters
# from configobj import ConfigObj
# from configobj import flatten_errors
# from validate import Validator

import matplotlib.pyplot as plt

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

	sessionDict = calcMeanGazeToBallDistDuringWindow(sessionDict)

	sessionDict = calcTrackingError(sessionDict)
	sessionDict = calcTTA(sessionDict)
	sessionDict = calcSMIDeltaT(sessionDict)
	sessionDict = filterAndDiffSignals(sessionDict,analysisParameters)
	sessionDict = vectorMovementModel(sessionDict,analysisParameters)

	# sessionDict = calcCalibrationQuality(sessionDict,analysisParameters)

	sessionDict = setIpdRatioAndPassingLoc(sessionDict)

	pd.to_pickle(sessionDict, analysisParameters['filePath'] + analysisParameters['fileName'] + '-proc.pickle')

	logger.info('Processed data saved to ' + analysisParameters['filePath'] + analysisParameters['fileName'] + '-proc.pickle')

	return sessionDict


def removeOutliers(allTrialData,columnName,stdRange=3):

    outliers = np.abs(allTrialData[columnName]-allTrialData[columnName].mean()) > (stdRange*allTrialData[columnName].std())

    outStr = 'Removed {} outliers from {}'.format(np.sum(outliers),columnName)
    logger.info(outStr)

    allTrialData[columnName] = allTrialData[columnName].mask(outliers)

    return allTrialData
	
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
	# sessionDict['processed'] = sessionDict['processed'].join(cycEIHDf)

	calibDf = sessionDictIn['calibration']
	cycEIH_fr = calibDf.apply(calcEIHByRow,axis=1)
	cycEIHDf = pd.DataFrame.from_records(cycEIH_fr)
	cycEIHDf.columns = pd.MultiIndex.from_tuples(cycEIHDf.columns)

	logger.info('Added sessionDict[\'calibration\'][\'cycEyeInHead\']')
	sessionDictIn['calibration'] = calibDf.combine_first(pd.DataFrame.from_records(cycEIHDf))

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

	# cycGIWDf.columns = pd.MultiIndex.from_tuples(cycGIWDf.columns)
	# sessionDictIn['processed'] = proc.combine_first(pd.DataFrame.from_records(cycGIWDf))

	sessionDictIn['processed'] = sessionDictIn['processed'].join(cycGIWDf)

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

	for (blockNum,trialNum), trialData in sessionDict['processed'].groupby(['blockNumber','trialNumber']):

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
		trInfo = sessionDict['trialInfo'].groupby(['blockNumber','trialNumber']).get_group((blockNum,trialNum))
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

	gbTrials = sessionDict['processed'].groupby(['blockNumber','trialNumber'])

	for (blockNum,trNum), trialData in gbTrials:

		trInfo = sessionDict['trialInfo'].groupby(['blockNumber','trialNumber']).get_group((blockNum,trNum))
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

def setIpdRatioAndPassingLoc(sessionDict):

	ipdRatio_tr = [np.nan] * len(sessionDict['trialInfo'])
	passingLocX_tr = [np.nan] * len(sessionDict['trialInfo'])

	for (blockNum,trialNum), trialData in sessionDict['processed'].groupby(['blockNumber','trialNumber']):

		tInfoIloc = np.intersect1d(np.where(sessionDict['trialInfo']['blockNumber']==blockNum),
							   np.where(sessionDict['trialInfo']['trialNumber']==trialNum))[0]

		tt = trialData.trialType.iloc[1]
		ipdRatio_tr[tInfoIloc] = sessionDict['expConfig']['trialTypes'][tt]['ipdRatio']
		passingLocX_tr[tInfoIloc] = sessionDict['expConfig']['trialTypes'][tt]['passingLocNormX']

	sessionDict['trialInfo']['ipdRatio'] = np.array(ipdRatio_tr,dtype=np.float)
	sessionDict['trialInfo']['passingLocX'] = np.array(passingLocX_tr,dtype=np.float)

	logger.info('Added sessionDict[\'trialInfo\'][\'ipdRatio\']')
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


def filterAndDiffSignals(sessionDict,analysisParameters):

	sgWinSizeSamples = analysisParameters['sgWinSizeSamples']
	sgPolyorder = analysisParameters['sgPolyorder']
	medFiltSize = analysisParameters['medFiltSize']
	interpResS = analysisParameters['interpResS']
	sgWinSizeSamples = analysisParameters['sgWinSizeSamples']

	from scipy.signal import savgol_filter

	# FIlter
	proc = sessionDict['processed']

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
	sessionDict['processed']['gazeVelFiltAz'] = gazeVelFiltAz_fr

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

	sessionDict['processed'] = proc

	return sessionDict

def vectorMovementModel( sessionDict,
						analysisParameters):

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

	for (bNum,tNum), tr in sessionDict['processed'].groupby(['blockNumber','trialNumber']):

		trInfo = sessionDict['trialInfo'].groupby(['blockNumber','trialNumber']).get_group((bNum,tNum))

		# Calculate events
		winStartTimeMs = analysisParameters['analysisWindowStart']
		winEndTimeMs = analysisParameters['analysisWindowEnd']

		trialTime_fr = np.array(tr['frameTime'],np.float) - np.array(tr['frameTime'],np.float)[0]
		interpTime_s = np.arange(0,trialTime_fr[-1],interpResS)

		# Analysis should focus on the frames before ball collision or passing
		initTTC = float(trInfo['ballInitialPos','Z']) / -float(trInfo['ballInitialVel','Z'])
		endFrameIdx = np.where( trialTime_fr > initTTC )[0][0]
		lastTrajFrame = np.min([int(endFrameIdx),
				   int(trInfo[('passVertPlaneAtPaddleFr', '')])])

		analysisTime_fr = np.array(tr['frameTime'],np.float)[:lastTrajFrame] - np.array(tr['frameTime'],np.float)[0]


		# Interpolate

		interpBallAz_s = np.interp(interpTime_s,analysisTime_fr,np.array(tr['ball_az'][:lastTrajFrame],dtype=np.float))
		interpBallEl_s = np.interp(interpTime_s,analysisTime_fr,np.array(tr['ball_el'][:lastTrajFrame],dtype=np.float))

		interpGazeAz_s = np.interp(interpTime_s,analysisTime_fr,np.array(tr['cycGIWFilt_az'][:lastTrajFrame],dtype=np.float))
		interpGazeEl_s = np.interp(interpTime_s,analysisTime_fr,np.array(tr['cycGIWFilt_el'][:lastTrajFrame],dtype=np.float))

		cycToBallVelAz_s = np.interp(interpTime_s,analysisTime_fr,np.array(tr['ballVel_az'][:lastTrajFrame],dtype=np.float))
		cycToBallVelEl_s = np.interp(interpTime_s,analysisTime_fr,np.array(tr['ballVel_el'][:lastTrajFrame],dtype=np.float))

		ballRadiusDegs_s = np.interp(interpTime_s,analysisTime_fr,np.array(tr['ballRadiusDegs'][:lastTrajFrame],dtype=np.float))

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
				  halfHFOVDegs = 80):

	import matplotlib.pyplot as plt

	p = plt.figure(figsize=(10, 15))

	plt.style.use('ggplot')

	grid = plt.GridSpec(3, 2, wspace=0.4, hspace=0.3)

	ax = p.add_subplot(grid[:2,:2])
	ax2 = p.add_subplot(grid[2,:],title='Velocity')

	ax.set(xlabel='degrees azimuth', ylabel='degrees elevation')
	ax2.set(xlabel='time (s)', ylabel='velocity (degrees/second)')

	#######

	# Calculate events
	winStartTimeMs = analysisParameters['analysisWindowStart']
	winEndTimeMs = analysisParameters['analysisWindowEnd']
	interpResS = analysisParameters['interpResS']

	trialTime_fr = np.array(tr['frameTime'],np.float) - np.array(tr['frameTime'],np.float)[0]
	interpTime_s = np.arange(0,trialTime_fr[-1],interpResS)

	# Analysis should focus on the frames before ball collision or passing
	initTTC = float(trInfo['ballInitialPos','Z']) / -float(trInfo['ballInitialVel','Z'])
	endFrameIdx = np.where( trialTime_fr > initTTC )[0][0]
	lastTrajFrame = np.min([int(endFrameIdx),
			   int(trInfo[('passVertPlaneAtPaddleFr', '')])])

	analysisTime_fr = np.array(tr['frameTime'],np.float)[:lastTrajFrame] - np.array(tr['frameTime'],np.float)[0]

	# Interpolate

	interpBallAz_s = np.interp(interpTime_s,analysisTime_fr,np.array(tr['ball_az'][:lastTrajFrame],dtype=np.float))
	interpBallEl_s = np.interp(interpTime_s,analysisTime_fr,np.array(tr['ball_el'][:lastTrajFrame],dtype=np.float))

	interpGazeAz_s = np.interp(interpTime_s,analysisTime_fr,np.array(tr['cycGIWFilt_az'][:lastTrajFrame],dtype=np.float))
	interpGazeEl_s = np.interp(interpTime_s,analysisTime_fr,np.array(tr['cycGIWFilt_el'][:lastTrajFrame],dtype=np.float))

	cycToBallVelAz_s = np.interp(interpTime_s,analysisTime_fr,np.array(tr['ballVel_az'][:lastTrajFrame],dtype=np.float))
	cycToBallVelEl_s = np.interp(interpTime_s,analysisTime_fr,np.array(tr['ballVel_el'][:lastTrajFrame],dtype=np.float))

	ballRadiusDegs_s = np.interp(interpTime_s,analysisTime_fr,np.array(tr['ballRadiusDegs'][:lastTrajFrame],dtype=np.float))

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

	initTTC = float(trInfo['ballInitialPos','Z']) / -float(trInfo['ballInitialVel','Z'])
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
	if (trInfo['ballCaughtQ'].values == True):

		cOrM = ax.scatter(tr['ball_az'].iloc[lastTrajFrame-1],
						  tr['ball_el'].iloc[lastTrajFrame-1],
						  c='g',s=120,marker='8',lw=6)
	else:

		cOrM = ax.scatter(tr['ball_az'].iloc[lastTrajFrame-1],
						  tr['ball_el'].iloc[lastTrajFrame-1],
						  c='r',s=150,marker='8',lw=6)

	ax.axis('equal')
	ax.set_aspect('equal')
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
	ax.text(.01,.04,str('IPD Ratio: {}').format(float(trInfo['ipdRatio'].values)),transform=ax.transAxes)
	ax.text(.01,.07,str('Sub: {} Bl: {} Tr: {}').format(
		int(trInfo['subjectNumber']),
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
	initTTC = float(trInfo['ballInitialPos','Z']) / -float(trInfo['ballInitialVel','Z'])

	winStartTimeMs = analysisParameters['analysisWindowStart']
	winEndTimeMs = analysisParameters['analysisWindowEnd']

	winStartFrameIdx = np.where( trialTime_fr > initTTC + winStartTimeMs/1000.0 )[0][0]
	winEndFrameIdx = np.where( trialTime_fr > initTTC + winEndTimeMs/1000.0 )[0][0] -1

	frameOfPassage = int(trInfo['passVertPlaneAtPaddleFr'])

	ax2.set_ylim([0,150])
	ax2.set_xlim(trialTime_fr[0],trialTime_fr[frameOfPassage])

	gazeVel = ax2.plot(trialTime_fr,
			tr['gazeVelFilt']
			,color='r',linewidth=3,alpha = .5,label='gazeVel')

#     gallVell = ax2.plot(trialTime_fr,
#         tr['ballVel2D_fr']
#         ,color='LightSeaGreen',linewidth=3,alpha = .5,label='ballVel')



	ballCenter = ax2.plot(trialTime_fr,
			tr['ballVel2D_fr']
			,color='b',linewidth=3,alpha = 0.5,label='ball center')

	ballLeading = ax2.plot(trialTime_fr,
			tr['ballVelLeadingEdge']
			,color='k',linewidth=3,alpha = 0.4,label='ball leading')

	ballTrailing = ax2.plot(trialTime_fr,
			tr['ballVelTrailingEdge']
			,color='k',linewidth=3,alpha = 0.4,label='ball trailing')

	ax2.axvspan(trialTime_fr[winStartFrameIdx], trialTime_fr[winEndFrameIdx], color='LightSeaGreen', alpha=0.5)
#     ax2.axvspan(.4,.6, color='red', alpha=0.5)

	ax2.legend()

	#######################

#     ax3 = ax2.twinx()  # instantiate a second axes that shares the same x-axis
#     ax3.set_ylim([-1,1])

#     ratio = ax3.plot(trialTime_fr,
#             tr['gazeVelRelBallEdges']
#             ,color='y',linewidth=3,alpha = 0.4,label='ratio')

#     ax3.axhline( 1 )
#     ax3.axhline( .5 )
#     ax3.axhline( 0 )
#     ax3.legend()

	return(p,ax)

def calcMeanGazeToBallDistDuringWindow(sessionDict):

	sessionDict['processed']['gazeToBallDist'] = np.sqrt((sessionDict['processed']['cycGIWFilt_az']-
                                                      sessionDict['processed']['ball_az'])**2 +
                                                     (sessionDict['processed']['cycGIWFilt_el']-
                                                      sessionDict['processed']['ball_el'])**2)

	gbTrialInfo = sessionDict['trialInfo'].groupby(['blockNumber','trialNumber'])

	meanGazeToBallDistDuringWindow = []

	for (bNum,tNum), proc in sessionDict['processed'].groupby(['blockNumber','trialNumber']):
		trInfo = gbTrialInfo.get_group((bNum,tNum))
		meanGazeToBallDistDuringWindow.append(np.mean(proc['gazeToBallDist'][int(trInfo['analysisStartFr']):int(trInfo['analysisEndFr'])]))

	sessionDict['trialInfo']['meanGazeToBallDistDuringWindow'] = meanGazeToBallDistDuringWindow

	logger.info('Added sessionDict[\'trialInfo\'][\'meanGazeToBallDistDuringWindow\']')

	return sessionDict

def plotGazeData(proc,tInfo,timeColumnLabel, velColumnLabels, xLim=[0,1],yLim=[0 ,120],winRange = False,width=800,height=600,inline=True,ytitle='velocity (pix/s)'):

	import plotly.graph_objs as go
	traces = []

	colors_idx = ['rgb(0,204,204)','rgb(128,128,128)','rgb(204,0,0)','rgb(102,0,204)']
	frameOfPassage = int(tInfo['passVertPlaneAtPaddleFr'])
	timeOfPassage = float(proc['frameTime'].iloc[tInfo['passVertPlaneAtPaddleFr']])

	time_fr = np.array(proc[timeColumnLabel])

	for idx, columnName in enumerate(velColumnLabels):

		scatterObj = go.Scatter(
		x=time_fr[:frameOfPassage],
		y=proc[columnName][:frameOfPassage],
		name = str(columnName),
		line = dict(color = colors_idx[idx],width=3),
		opacity = 0.8)
		traces.append(scatterObj)

#     if winRange is False:

#     winRange=[proc['frameTime'].iloc[1],proc['frameTime']int(tInfo['passVertPlaneAtPaddleFr'])],
							  #proc['frameTime'].iloc[1]+1.5],
	winRange = [proc['frameTime'].iloc[0],timeOfPassage]

	layout = dict(
		dragmode= 'pan',
		width=width,
		height=height,
		yaxis=dict(range=yLim, title=ytitle),
		xaxis=dict(
			rangeslider=dict(),
			type='linear',
			range=winRange,
			title='time'),

		shapes = [go.layout.Shape( type="rect",
				xref="x",
				yref="paper",
				x0=float(proc['frameTime'].iloc[tInfo['analysisStartFr']]),
				y0=0,
				x1=float(proc['frameTime'].iloc[tInfo['analysisEndFr']]),
				y1=1,
				fillcolor="LightSeaGreen",
				opacity=0.5,
#                 layer="below",
				line_width=2)]



	)


	fig = dict(data=traces, layout=layout)
	return fig


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

	for (blockNum,trNum), tProcData in sessionDict['processed'].groupby(['blockNumber','trialNumber']):

		fr = sessionDict['trialInfo'].groupby(['blockNumber','trialNumber']).get_group((blockNum,trNum))['passVertPlaneAtPaddleFr']
		row = tProcData.iloc[fr]
		bXYZ = row.ballPos.values
		pXYZ = row.paddlePos.values

		#  Ball trajectory direction
		ballTrajDir_XYZ = (tProcData['ballPos'].iloc[10] - tProcData['ballPos'].iloc[1]) / np.linalg.norm(tProcData['ballPos'].iloc[10] - tProcData['ballPos'].iloc[1])
		ballTrajDir_XYZ = np.array(ballTrajDir_XYZ,dtype=np.float)
		paddleYDir_xyz = [0,1,0]
		paddleXDir_xyz = np.cross(-ballTrajDir_XYZ,paddleYDir_xyz)
		# paddleToBallVec_fr_XYZ = tProcData['ballPos'].values - tProcData['paddlePos'].values

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

	gbTrials = procDF.groupby(['blockNumber','trialNumber'])
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

def calcTargetDir(sessionDictIn):

	def calcTargetDirByRow(row):
		targetInHead_xyz =  np.array(row['calibrationPos'] / np.linalg.norm(row['calibrationPos']),dtype=np.float)
		return {('targetInHead','X'): targetInHead_xyz[0],('targetInHead','Y'): targetInHead_xyz[1],('targetInHead','Z'): targetInHead_xyz[2]}

	calibDf = sessionDictIn['calibration']
	targetDir_fr = calibDf.apply(lambda row: calcTargetDirByRow(row),axis=1).values
	targetDirDf = pd.DataFrame.from_records(targetDir_fr)
	targetDirDf.columns = pd.MultiIndex.from_tuples(targetDirDf.columns)
	#subDataDf = calibDf.combine_first(targetDirDf)
	sessionDictIn['calibration'] = calibDf.combine_first(targetDirDf)

	return sessionDictIn

# def calcCycEIH(sessionDictIn):
#     def calcEIHByRow(row):
#
#         cycEIH = (np.array(row['leftEyeInHead'].values,dtype=np.float) +
#             np.array(row['rightEyeInHead'].values,dtype=np.float))/2
#
#         cycEIH = cycEIH / np.linalg.norm(cycEIH)
#         return {('cycEyeInHead','X'): cycEIH[0],('cycEyeInHead','Y'): cycEIH[1],('cycEyeInHead','Z'): cycEIH[2]}
#
#     calibDf = sessionDictIn['calibration']
#     cycEIH_fr = calibDf.apply(calcEIHByRow,axis=1)
#     cycEIHDf = pd.DataFrame.from_records(cycEIH_fr)
#     cycEIHDf.columns = pd.MultiIndex.from_tuples(cycEIHDf.columns)
#
#     logger.info('Added sessionDict[\'processed\'][\'cycEyeInHead\']')
#     sessionDictIn['calibration'] = calibDf.combine_first(pd.DataFrame.from_records(cycEIHDf))
#
#     return sessionDictIn

def calcCalibrationVectors(sessionDictIn):

	sessionDictIn = calcTargetDir(sessionDictIn)
	# sessionDictIn = calcCycEIH(sessionDictIn)
	calibDf = sessionDictIn['calibration']

	calibDf['targetInHead_az'] = calibDf.apply(lambda row: np.rad2deg(np.arctan(row[('targetInHead','X')]
															   /row[('targetInHead','Z')])),axis=1)

	calibDf['targetInHead_el'] = calibDf.apply(lambda row: np.rad2deg(np.arctan(row[('targetInHead','Y')]
															   /row[('targetInHead','Z')])),axis=1)

	calibDf['cycEyeInHead_az'] = calibDf.apply(lambda row: np.rad2deg(np.arctan(row[('cycEyeInHead','X')]
															   /row[('cycEyeInHead','Z')])),axis=1)

	calibDf['cycEyeInHead_el'] = calibDf.apply(lambda row: np.rad2deg(np.arctan(row[('cycEyeInHead','Y')]
															   /row[('cycEyeInHead','Z')])),axis=1)

	calibDf['calibErr'] = calibDf.apply(lambda row: np.rad2deg(np.arccos( np.vdot(row['cycEyeInHead'],row['targetInHead']))),
												axis=1,raw=True)

	logger.info('Added sessionDict[\'calibration\'][\'targetInHead_az\']')
	logger.info('Added sessionDict[\'calibration\'][\'targetInHead_el\']')
	logger.info('Added sessionDict[\'calibration\'][\'cycEyeInHead_az\']')
	logger.info('Added sessionDict[\'calibration\'][\'cycEyeInHead_el\']')

	sessionDictIn['calibration'] = calibDf

	return sessionDictIn

def calcCalibrationQuality(sessionDictIn):

	sessionDictIn = calcCalibrationVectors(sessionDictIn)

	calibDf = sessionDictIn['calibration']
	gb_tIdx = calibDf.groupby('calibrationCounter')
	targetList = list(calibDf.groupby('calibrationCounter').groups.keys())

	numTargets = len(targetList)

	gazePos_azEl_tIdx = np.zeros([2,numTargets])
	targPos_azEl_tIdx  = np.zeros([2,numTargets])
	calibError_tIdx = np.zeros([numTargets])
	stdCalibError_tIdx = np.zeros([numTargets])

	for targetKey, data in gb_tIdx:

		tIdx  = [i for i, s in enumerate(targetList) if targetKey == s]

		gazePos_azEl_tIdx[0,tIdx] = np.nanmean(data['cycEyeInHead_az'])
		gazePos_azEl_tIdx[1,tIdx] = np.nanmean(data['cycEyeInHead_el'])

		targPos_azEl_tIdx[0,tIdx] = np.nanmean(data['targetInHead_az'])
		targPos_azEl_tIdx[1,tIdx] = np.nanmean(data['targetInHead_el'])

		calibError_tIdx[tIdx] = np.nanmean(data['calibErr'])
		stdCalibError_tIdx[tIdx] = np.nanstd(data['calibErr'])

	sessionDictIn['calibrationData'] = pd.DataFrame(dict(gazePos_az = gazePos_azEl_tIdx[0,:],
				  gazePos_el = gazePos_azEl_tIdx[1,:],
				  targetPos_az = targPos_azEl_tIdx[0,:],
				  targetPos_el = targPos_azEl_tIdx[1,:],
				  meanCalibError = calibError_tIdx))

	return sessionDictIn
