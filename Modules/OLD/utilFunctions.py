import pandas as pandas
import numpy as np

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

