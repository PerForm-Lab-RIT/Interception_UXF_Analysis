
import sys
sys.path.append("../Modules/")
sys.path.append("../")
import os

import pandas as pd
import numpy as np

#from utilFunctions import *
#from analysisParameters import loadParameters

from configobj import ConfigObj
from configobj import flatten_errors
from validate import Validator
    
import matplotlib.pyplot as plt
    
import logging
logger = logging.getLogger(__name__)


# def plotMovementModel(tr,
#              trInfo,
#              halfHFOVDegs = 60):

#     halfVFOVDegs = halfHFOVDegs / 1.77


#     ####

#     from matplotlib import patches as pt

#     winStartTimeMs = analysisParameters['analysisWindowStart']
#     winEndTimeMs = analysisParameters['analysisWindowEnd']

#     trialTime_fr = np.array(tr['frameTime'],np.float) - np.array(tr['frameTime'],np.float)[0]
#     interpTime_s = np.arange(0,trialTime_fr[-1],interpResS)

#     # Analysis should focus on the frames before ball collision or passing
#     initTTC = float(trInfo['ballInitialPos','Z']) / -float(trInfo['ballInitialVel','Z'])
#     endFrameIdx = np.where( trialTime_fr > initTTC )[0][0]

#     lastTrajFrame = np.min([int(endFrameIdx),
#                int(trInfo[('passVertPlaneAtPaddleFr', '')])])

#     analysisTime_fr = np.array(tr['frameTime'],np.float)[:lastTrajFrame] - np.array(tr['frameTime'],np.float)[0]

#     winStartFrameIdx = np.where( trialTime_fr > initTTC + winStartTimeMs/1000.0 )[0][0]
#     winEndFrameIdx = np.where( trialTime_fr > initTTC + winEndTimeMs/1000.0 )[0][0] - 1
#     windowFr = np.arange(winStartFrameIdx, winEndFrameIdx)

#     ###

#     # Analysis should focus on the frames before ball collision or passing
#     initTTC = float(trInfo['ballInitialPos','Z']) / -float(trInfo['ballInitialVel','Z'])
#     endFrameIdx = np.where( trialTime_fr > initTTC )[0][0]
#     lastTrajFrame = np.min([int(endFrameIdx),
#                int(trInfo[('passVertPlaneAtPaddleFr', '')])])

#     analysisTime_fr = np.array(tr['frameTime'],np.float)[:lastTrajFrame] - np.array(tr['frameTime'],np.float)[0]

#     p, ax = plt.subplots(1, 1,figsize=(10,10)) #sharey=True)
#     cList = ['r','g','b']
#     lineHandles = []

#     ballH = ax.plot(tr['ball_az'][:endFrameIdx],tr['ball_el'][:endFrameIdx],color='b',linewidth=3,alpha = 0.4)
#     gazeH = ax.plot(tr['cycGIW_az'][:endFrameIdx],tr['cycGIW_el'][:endFrameIdx],color='r',linewidth=3,alpha = 0.4)

#     ax.add_patch(pt.Circle(trInfo['ballWinEnd_AzEl'].values[0],
#                            radius=np.float(trInfo['ballRadiusWinEnd']),
#                  fill=False,facecolor=None,ec='k',lw=3))

#     ax.plot(tr['ball_az'].iloc[windowFr],tr['ball_el'].iloc[windowFr],color='b',linewidth=5, alpha = 0.6)
#     ax.plot(tr['cycGIW_az'].iloc[windowFr],tr['cycGIW_el'].iloc[windowFr],color='r',linewidth=5, alpha = 0.6)

#     for i in np.arange(0,len(windowFr),1):
#         pf = windowFr[i]
#         xs = [tr['ball_az'].iloc[pf], tr['cycGIW_az'].iloc[pf]]
#         ys = [tr['ball_el'].iloc[pf], tr['cycGIW_el'].iloc[pf]]
#         ax.plot(xs,ys,color='k',linewidth=1,alpha = 0.3)

#     ax.axis('equal')
#     ax.axes.spines['top'].set_visible(False)
#     ax.axes.spines['right'].set_visible(False)
#     ax.axes.yaxis.grid(True)
#     ax.axes.xaxis.grid(True)
#     p.set_facecolor('w')

#     plt.xlim([-30,30])
#     plt.ylim([-15,35])

#     ########################################################################
#     ########################################################################

#     observedH = ax.scatter(trInfo['ballWinEnd_AzEl'].values[0][0],trInfo['ballWinEnd_AzEl'].values[0][1],c='k',s=150,marker='8')
#     constantVelH = ax.scatter(trInfo['ballAtWinEndVelPred_AzEl'].values[0][0],trInfo['ballAtWinEndVelPred_AzEl'].values[0][1],c='k',s=150,marker='v')
#     gazeLoc = ax.scatter(trInfo['gazeMinDistLoc_AzEl'].values[0][0],
#                          trInfo['gazeMinDistLoc_AzEl'].values[0][1],c='m',s=150,marker='x',lw=6)


#     ax.text(.01,.01,str('NormLoc: {}').format(trInfo['normLocInWindow'].values),transform=ax.transAxes)
#     ax.text(.01,.03,str('Expansion gain: {}').format(float(trInfo['expansionGain'].values)),transform=ax.transAxes)
#     ax.text(.01,.05,str('Sub: {} Bl: {} Tr: {}').format(
#         int(trInfo['subjectNumber']), 
#         int(trInfo['blockNumber']),
#         int(trInfo['trialNumber'])
#     ),transform=ax.transAxes)
    
#     ax.legend([gazeLoc,
#                constantVelH,
#                observedH], 

#               ['point nearest to gaze',
#                'constant speed model',
#                'actual displacement'])


# def plotProjectedTrajectory(tr, 
#                             halfHFOVDegs = 45,
#                             analyzeUntilXSToArrival =  .2,
#                             stopAtXSToArrival = 0.075):

#     def getLastFr(trialData):

#         ballPassesOnFr = False
#         ballHitPaddleOnFr = np.where(trialData['eventFlag']=='ballOnPaddle')
#         endFr = False

#         initTTC = (trialData[('ballInitialPos','Z')].iloc[1] / -trialData[('ballInitialVel','Z')].iloc[1])

#         if len(ballHitPaddleOnFr[0]) > 0:
#             ballHitPaddleOnFr = ballHitPaddleOnFr[0][0]
#             ballPassesOnFr = False
#             endFr = ballHitPaddleOnFr
#         else:
#             ballHitPaddleOnFr = False
#             ballPassesOnFr = np.where(trialData[('ballPos','Z')]<0)

#             if len(ballPassesOnFr[0]) > 0:
#                 ballPassesOnFr = ballPassesOnFr[0][0]
#                 endFr = ballPassesOnFr
#             else:
#                 # Sometimes the ball seems to stop in place upon collision.  I'm not sure what's going on there.
#                 endFr = np.where(trialData[('ballPos','Z')].diff()==0)[0][0]-1

#         return endFr

#     endFr = getLastFr(tr)
#     plotFr = tr.index[1:endFr]
    
#     halfVFOVDegs = halfHFOVDegs / 1.77

#     p, ax = plt.subplots(1, 1) #sharey=True)
#     cList = ['r','g','b']
#     lineHandles = []

#     ballH = ax.plot(tr['ball_az'][plotFr],tr['ball_el'][plotFr],color='b',linewidth=3,alpha = 0.4)
#     gazeH = ax.plot(tr['cycGIW_az'][plotFr],tr['cycGIW_el'][plotFr],color='r',linewidth=3,alpha = 0.4)

#     from processDataE1 import findAnalysisWindow

#     (startFr, endFr) = findAnalysisWindow(tr,
#                                          analyzeUntilXSToArrival =  analyzeUntilXSToArrival, 
#                                           stopAtXSToArrival = stopAtXSToArrival)
    
#     windowFr = tr.index[startFr:endFr]
#     ax.plot(tr['ball_az'][windowFr],tr['ball_el'][windowFr],color='b',linewidth=5, alpha = 0.6)
#     ax.plot(tr['cycGIW_az'][windowFr],tr['cycGIW_el'][windowFr],color='r',linewidth=5,alpha = 0.6)
    
#     for pf in windowFr:
#         xs = [tr['ball_az'][pf], tr['cycGIW_az'][pf]]
#         ys = [tr['ball_el'][pf], tr['cycGIW_el'][pf]]
#         ax.plot(xs,ys,color='k',linewidth=1,alpha = 0.5)
    
    
#     ax.text(-.85*halfHFOVDegs, -10,str('{0:.2f}').format(tr.frameTime.loc[plotFr[1]]))
#     ax.text(-.85*halfHFOVDegs, -12,str('{0:.2f}').format(tr.frameTime.loc[plotFr[-1]]))

#     ax.axis('equal')
#     ax.axes.spines['top'].set_visible(False)
#     ax.axes.spines['right'].set_visible(False)
#     ax.axes.yaxis.grid(True)
#     ax.axes.xaxis.grid(True)
#     p.set_facecolor('w')

#     plt.xlim([-halfHFOVDegs,halfHFOVDegs])
#     plt.ylim([-10,-10+2*halfVFOVDegs])

#     return p
    