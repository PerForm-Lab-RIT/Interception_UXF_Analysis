
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def assessTrackQuality(sessionDictIn):
    
    def calcEIHByRow(row):

        cycEIH = (np.array(row['leftEyeInHead'].values,dtype=np.float) + 
            np.array(row['rightEyeInHead'].values,dtype=np.float))/2

        cycEIH = cycEIH / np.linalg.norm(cycEIH)

        return {('cycEyeInHead','X'): cycEIH[0],('cycEyeInHead','Y'): cycEIH[1],('cycEyeInHead','Z'): cycEIH[2]}

    def calcTargetDir(row):

        targetInHead_xyz =  np.array(row['calibrationPos'] / np.linalg.norm(row['calibrationPos']),dtype=np.float)
        return {('targetInHead','X'): targetInHead_xyz[0],('targetInHead','Y'): targetInHead_xyz[1],('targetInHead','Z'): targetInHead_xyz[2]}


    subDataDf = sessionDictIn['calibration']
    cycEIH_fr = subDataDf.apply(lambda row: calcEIHByRow(row),axis=1).values
    cycEIHDf = pd.DataFrame.from_records(cycEIH_fr)
    cycEIHDf.columns = pd.MultiIndex.from_tuples(cycEIHDf.columns)
    subDataDf = subDataDf.combine_first(cycEIHDf)

    targetDir_fr = subDataDf.apply(lambda row: calcTargetDir(row),axis=1).values
    targetDirDf = pd.DataFrame.from_records(targetDir_fr)
    targetDirDf.columns = pd.MultiIndex.from_tuples(targetDirDf.columns)
    subDataDf = subDataDf.combine_first(targetDirDf)

    subDataDf['targetInHead_az'] = subDataDf.apply(lambda row: np.rad2deg(np.arctan(row[('targetInHead','X')]
                                                               /row[('targetInHead','Z')])),axis=1)

    subDataDf['targetInHead_el'] = subDataDf.apply(lambda row: np.rad2deg(np.arctan(row[('targetInHead','Y')]
                                                               /row[('targetInHead','Z')])),axis=1)

    subDataDf['cycEyeInHead_az'] = subDataDf.apply(lambda row: np.rad2deg(np.arctan(row[('cycEyeInHead','X')]
                                                               /row[('cycEyeInHead','Z')])),axis=1)

    subDataDf['cycEyeInHead_el'] = subDataDf.apply(lambda row: np.rad2deg(np.arctan(row[('cycEyeInHead','Y')]
                                                               /row[('cycEyeInHead','Z')])),axis=1)

    subDataDf['calibErr'] = subDataDf.apply(lambda row: np.rad2deg(np.arccos( np.vdot(row['cycEyeInHead'],row['targetInHead']))),
                                                    axis=1,
                                                     raw=True)

    gb_tIdx = subDataDf.groupby('calibrationCounter')
    targetList = list(subDataDf.groupby('calibrationCounter').groups.keys())

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


def plotTrackQuality(sessionDictIn, saveFig = True):

    calibData = sessionDictIn['calibrationData']
    p, ax = plt.subplots(1, 1) #sharey=True)

    cList = ['r','g','b']
    lineHandles = []

    offsets = np.linspace(-.01,.01,3)

    # Targets
    xx = calibData['targetPos_az']
    yy = calibData['targetPos_el']
    hT = ax.scatter(xx, yy,s=100)
    hT.set_label('target')

    x = calibData['gazePos_az']
    y = calibData['gazePos_el']
    hG = ax.scatter(x, y,s=20,c='r')
    hG.set_label('gaze')

    
    for idx in range(len(calibError_tIdx)):
        xxx = targPos_azEl_tIdx[0,:]
        yyy = targPos_azEl_tIdx[1,:]+1
        textStr = '   %1.2f$^\circ$' %calibData['meanCalibError'][idx]
        hErr = ax.text(xxx[idx], yyy[idx], textStr,horizontalalignment='center',size=20)


    plt.gcf().set_size_inches(8,8)
    ax.axes.set_title('Calibration quality', fontsize=15)
    ax.set_ylabel('elevation (degrees)', fontsize=12)
    ax.set_xlabel('azimuth (degrees)', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)

    ax.set_ylim([-20,20])
    ax.set_xlim([-20,20])
    ax.axes.yaxis.grid(True)
    ax.axes.xaxis.grid(True)
    ax.axes.set_axisbelow(True)

    plt.rcParams["font.family"] = "sans-serif"
    
    fileName = sessionDictIn['calibration'].fileName

    plt.legend()
    if(saveFig == True):
        plt.savefig('../calibrationFigs/' + str(fileName) + '.png', facecolor=p.get_facecolor(), transparent=True)

    return