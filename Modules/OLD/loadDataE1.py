import sys
sys.path.append("../Modules/")
sys.path.append("../")
import os

import pandas as pd
import numpy as np

from analysisParameters import loadParameters

from utilFunctions import *

from configobj import ConfigObj
from configobj import flatten_errors
from validate import Validator
  
import logging
logger = logging.getLogger(__name__)

# import logging
# fmt=" %(levelname)s-%(name)s-%(funcName)s()[%(lineno)i]: %(message)s"
# #fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s')
# #fmt = '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s'

# logging.basicConfig(level=logging.INFO, format=fmt)
# logger = logging.getLogger(__name__)



def initTrialInfo(sessionDf):
    '''
    Create a trial info dataframe with some basic settings
    
    '''
    def calcTrialOutcome(sessionDf,sessionTrialInfo):
        '''
        Accepts: 
            session dataFrame
            trialInfo dataFrame
        Returns: 
            trialInfo dataFrame
        '''

        ballCaughtQ = []
        ballCaughtFr = []

        for groupIdx, tr in sessionDf.groupby(['trialNumber','blockNumber']):

            catchFr = findFirst(tr['eventFlag'],'ballOnPaddle')

            if( catchFr ):

                ballCaughtQ.append(True)
                ballCaughtFr.append(tr.index.values[catchFr])        

            else:
                ballCaughtQ.append(False)
                ballCaughtFr.append(np.nan)

        df = pd.DataFrame({('ballCaughtFr',''):ballCaughtFr,('ballCaughtQ',''):ballCaughtQ})

        return pd.concat([df,sessionTrialInfo],axis=1)


    #gbTrials = sessionDf.groupby('trialNumber')
    listOfDicts = []
    for groupIdx, tr in sessionDf.groupby(['trialNumber','blockNumber']):
    

        listOfDicts.append({
            ('ballRadiusM','') : tr['ballRadiusM'].iloc[0],
            ('ballInitialVel','X') : tr[('ballInitialVel','X')].iloc[0],
            ('ballInitialVel','Y') : tr[('ballInitialVel','Y')].iloc[0],
            ('ballInitialVel','Z') : tr[('ballInitialVel','Z')].iloc[0],
            
            ('ballInitialPos','X') : tr[('ballInitialPos','X')].iloc[0],
            ('ballInitialPos','Y') : tr[('ballInitialPos','Y')].iloc[0],
            ('ballInitialPos','Z') : tr[('ballInitialPos','Z')].iloc[0],
            
            ('ballFinalPos','X') : tr[('ballFinalPos','X')].iloc[0],
            ('ballFinalPos','Y') : tr[('ballFinalPos','Y')].iloc[0],
            ('ballFinalPos','Z') : tr[('ballFinalPos','Z')].iloc[0],
            
            ('frameTime','') : tr['frameTime'].iloc[0],
            ('blockNumber','') : groupIdx[1],

            ('maxReach','') : tr['maxReach'].iloc[0],
            ('noExpansionForLastXSeconds','') : tr['noExpansionForLastXSeconds'].iloc[0],
            ('trialType','') : tr['trialType'].iloc[0],
            ('trialNumber','') : groupIdx[0],
            
            ('firstFrame','') : tr.index[0],
            ('lastFrame','') : tr.index[-1]},
                          )
    
    trialInfo =  pd.DataFrame(listOfDicts)
    trialInfo = calcTrialOutcome(sessionDf,trialInfo)
    #trialInfo.index.name = 'trialNum'

    return trialInfo


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

def createSecondaryDataframes(sessionDf,filePath,fileName,expCfgName,sysCfgName):
        '''
        Separates practice and calibration trials from main dataframe.
        Reads in exp and sys config.
        '''

        logger.warning('createSecondaryDataframes(): SHIFTING EVENTFLAG:TRIALEND AND REMOVING ROWS TO COMPENSATE FOR OFFSET BUG DURING DATA COLLECTION')

        eIdx = np.where([sessionDf['eventFlag']=='trialEnd'])[1]
        eF = sessionDf['eventFlag'].values
        eF[eIdx-1] = 'trialEnd'
        eF[eIdx] = 'delMe'
        sessionDf['eventFlag'] = eF
        sessionDf['eventFlag']
        sessionDf.drop(index=eIdx,inplace=True)

        
        def seperateCalib(sessionDf):
            calibFrames = sessionDf['inCalibrationQ'] == True
            calibDf = sessionDf[calibFrames]
            sessionDf = sessionDf.drop(sessionDf[calibFrames].index)
            return sessionDf, calibDf

        def seperatePractice(sessionDf,practiceBlockIdx):
            # This assumes that calibration has already been removed

            practiceDf = pd.DataFrame()

            for bIdx in practiceBlockIdx:
                thisPracticeBlockDF = sessionDf[sessionDf['blockNumber']==bIdx]
                practiceDf = pd.concat([practiceDf,thisPracticeBlockDF],axis=0)
                sessionDf = sessionDf.drop(thisPracticeBlockDF.index)

            return sessionDf, practiceDf


        [sessionDf, calibDf] = seperateCalib(sessionDf)

        expConfig =  createExpCfg(filePath + expCfgName)
        sysConfig =  createSysCfg(filePath + sysCfgName)
        practiceBlockIdx = [idx for idx, s in enumerate(expConfig['experiment']['blockList']) if s == 'practice']

        [sessionDf, practiceDf] =  seperatePractice(sessionDf,practiceBlockIdx)

        sessionDf = sessionDf.reset_index()
        sessionDf = sessionDf.rename(columns = {'index':'frameNumber'})

        trialInfoDf = initTrialInfo(sessionDf)
        
        from copy import deepcopy
        procDataDf = deepcopy(sessionDf)
        procDataDf.index.name = 'frameNum'

        sessionDict = {'fileName': fileName,
                        'raw': sessionDf, 
                        'processed': procDataDf, 
                        'calibration': calibDf, 
                        'practice': practiceDf, 
                        'trialInfo': trialInfoDf,
                        'expConfig': expConfig,
                        'sysCfg': sysConfig}

        return sessionDict


def excludeTrialType(sessionDict,typeString):

    sessionDictCopy = sessionDict.copy()

    gbTrialType = sessionDictCopy['trialInfo'].groupby('trialType')
    newDf = gbTrialType.get_group(typeString)
    sessionDictCopy['trialInfo'] = sessionDictCopy['trialInfo'].drop(gbTrialType.get_group(typeString).index)
    sessionDictCopy['trialInfo'] = sessionDictCopy['trialInfo'].reset_index()


    sessionDictCopy['processed']['trialType'] = sessionDictCopy['raw']['trialType']
    gbProc = sessionDictCopy['processed'].groupby('trialType')
    newDf = gbProc.get_group(typeString)
    sessionDictCopy['processed'] = sessionDictCopy['processed'].drop(gbProc.get_group(typeString).index)
    sessionDictCopy['processed']=sessionDictCopy['processed'].reset_index()

    gbRaw = sessionDictCopy['raw'].groupby('trialType')
    newDf = gbRaw.get_group(typeString)
    sessionDictCopy['raw'] = sessionDictCopy['raw'].drop(gbRaw.get_group(typeString).index)
    sessionDictCopy['raw'] = sessionDictCopy['raw'].reset_index()

    return sessionDictCopy


def loadSessionDict(analysisParameters,loadParsedData = True,loadProcessedData = False, doNotProcess=False):
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
    
    parseData = False
    processData = False

    if loadProcessedData == True :

        try:
            logger.info('Loading preprocessed data for ' + str(fileName) + ' ***')

            processedDict = pd.read_pickle(filePath + fileName + '-proc.pickle')
            return processedDict
        except:
            logger.warning('loadProcessedDict: Preprocessed data not available')
            processData = True
            loadParsedData = True


    if loadParsedData == False :

        parseData = True
    
    else:
        # Try and load the parsed data
        try:

            logger.info('Loading parsed data for ' + str(fileName) + ' ***')
            sessionDict = pd.read_pickle(filePath + fileName + '.pickle')
            sessionDict['analysisParameters'] = analysisParameters
            processData = True

        except:
            
            logger.warning('Attempt to load parsed data failed ' + str(fileName) + ' ***')
            parseData = True

    if (parseData == True or (loadParsedData == False and loadProcessedData == False )):

        logger.info('Parsing raw data: ' + str(fileName) + ' ***')
        import PerformParser as pp
        
        sessionDf = pp.readPerformDict(filePath + fileName + ".dict")

        logger.info('Shifting gaze data to compensate for latency')
        sessionDf = ShiftGazeData(sessionDf,analysisParameters['gazeDataShift'])


        # This takes the parsed data and turns it into the sessionDict we all know and love
        sessionDict = createSecondaryDataframes(sessionDf,filePath,fileName,expCfgName,sysCfgName)

        outloc = filePath + fileName + '.pickle'
        pd.to_pickle(sessionDict,outloc)
        logger.info('Parsed data saved to  ' + outloc)
        processData = True
    
    if processData and doNotProcess == False:
        
        logger.info('Processing parsed data: ' + str(fileName) + ' ***')
        from processDataE1 import processData
        sessionDict = processData(sessionDict,analysisParameters)

        pd.to_pickle(sessionDict, analysisParameters['filePath'] + analysisParameters['fileName'] + '-proc.pickle')
        logger.info('Processed data saved to file.')

    return sessionDict  

def ShiftGazeData(sessionDictIn,shiftBy):
    
    gazeDataKeys = [
     'cycEyeBasePoint','cycEyeNodeInHead','cycEyeNodeInWorld','cycGazeDir','cycGazeNodeInWorld',
        'cycInverseMat','cycMat','eyeTimeStamp',
        'leftEyeBasePoint','leftEyeInverseMat','leftEyeInHead','leftEyeMat','leftEyeNodeInHead',
        'leftEyeNodeInWorld','leftEyeOnScreen','leftGazeNodeInWorld',
        'rightEyeBasePoint','rightEyeInHead','rightEyeInverseMat','rightEyeMat','rightEyeNodeInHead',
        'rightEyeNodeInWorld','rightEyeOnScreen','rightGazeNodeInWorld',
    ]
    
    for key in gazeDataKeys:
        sessionDictIn[key] = sessionDictIn[key].shift(periods=shiftBy)
        
    return sessionDictIn

