import os
import sys
import click
import csv
import importlib

import logging
import pickle

from dotenv import load_dotenv

import numpy as np
import matplotlib.pyplot as plt

@click.command()
@click.option(
    "--allow_session_loading",
    is_flag=True
)
@click.option(
    "--skip_pupil_detection",
    is_flag=True
)
@click.option(
    "--vanilla_only",
    is_flag=True
)
@click.option(
    "--skip_vanilla",
    is_flag=True
)
@click.option(
    "--surpress_runtimewarnings",
    is_flag=True
)
@click.option(
    "--load_2d_pupils",
    is_flag=True
)
@click.option(
    "--min_calibration_confidence",
    required=False,
    type=click.FloatRange(min=0.0, max=1.0),
    default=0.0
)
@click.option(
    "--show_filtered_out",
    is_flag=True
)
@click.option(
    "--display_world_video",
    is_flag=True
)
@click.option(
    "--graphs_only",
    is_flag=True
)
@click.option(
    "--core_shared_modules_loc",
    required=False,
    type=click.Path(exists=True),
    envvar="CORE_SHARED_MODULES_LOCATION",
)
@click.option(
    "--pipeline_loc",
    required=True,
    type=click.Path(exists=True),
    envvar="PIPELINE_LOC",
)
@click.option(
    "--plugins_file",
    required=False,
    type=click.Path(exists=True),
    envvar="PLUGINS_CSV",
)
def main(allow_session_loading, skip_pupil_detection, vanilla_only, skip_vanilla, surpress_runtimewarnings, load_2d_pupils, min_calibration_confidence, show_filtered_out, display_world_video, graphs_only, core_shared_modules_loc, pipeline_loc, plugins_file):
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("numexpr").setLevel(logging.WARNING)
    logging.getLogger("OpenGL").setLevel(logging.WARNING)

    if surpress_runtimewarnings:
        import warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning) 
    
    print(pipeline_loc)
    sys.path.append(pipeline_loc)
    from core.pipeline import available_mapping_methods, patch_plugin_notify_all, calibrate_and_validate, map_pupil_data, save_gaze_data, get_first_realtime_ref_data_timestamp, get_first_ref_data_timestamp, get_last_realtime_ref_data_timestamp, get_last_ref_data_timestamp, load_realtime_ref_data
    from core.pupil_detection import perform_pupil_detection
    from processAssessmentData import processAllData
    
    if core_shared_modules_loc:
        sys.path.append(core_shared_modules_loc)
        sys.path.append(os.path.join(core_shared_modules_loc,'pupil_detector_plugins'))
    else:
        logging.warning("Core source location unknown. Imports might fail.")
    
    if skip_vanilla:
        plugins = []
    else:
        plugins = [None]
    if not vanilla_only:
        plugins_unpacked = []
        with open(plugins_file, newline='') as f:
            reader = csv.reader(filter(lambda row: row[0] != '#', f))
            plugins_unpacked = list(reader)
        for row in plugins_unpacked:
            sys.path.append(row[0])
            CurrPlugin = getattr(importlib.import_module(row[1]), row[2])
            plugins.append(CurrPlugin)
    
    if not graphs_only:
        logging.debug(f"Loaded pupil detector plugins: {plugins}")
        
        mapping_methods_by_label = available_mapping_methods()
        mapping_method_label = click.prompt(
            "Choose gaze mapping method",
            type=click.Choice(mapping_methods_by_label.keys(), case_sensitive=True),
        )
        mapping_method = mapping_methods_by_label[mapping_method_label]
        patch_plugin_notify_all(mapping_method)
    
        for name in [item for item in os.listdir("Data/") if item[0:9] == '_Pipeline']:
            subject_loc = os.path.join("Data/", name)
            rec_loc = os.path.join(subject_loc, 'S001/PupilData/000')
            logging.info(f'Proccessing {rec_loc} through pipeline...')
            
            reference_data_loc = rec_loc+'/offline_data/reference_locations.msgpack'
            if not skip_pupil_detection:
                logging.info("Performing pupil detection on eye videos. This may take a while.")
                for plugin in plugins:
                    logging.info(f"Current plugin: {plugin}")
                    # Checking to see if we can freeze the model at the first calibration point
                    realtime_calib_points_loc = rec_loc+'/realtime_calib_points.msgpack';
                    
                    if os.path.exists(realtime_calib_points_loc):
                        start_model_timestamp = get_first_realtime_ref_data_timestamp(realtime_calib_points_loc)
                        freeze_model_timestamp = get_last_realtime_ref_data_timestamp(realtime_calib_points_loc)
                    else:
                        start_model_timestamp = get_first_ref_data_timestamp(reference_data_loc)
                        freeze_model_timestamp = get_last_ref_data_timestamp(reference_data_loc)
                    
                    #start_model_timestamp = None
                    #freeze_model_timestamp = None
                    
                    perform_pupil_detection(rec_loc, plugin=plugin, pupil_params=[{
                                                    # eye 0 params
                                                    "intensity_range": 23,
                                                    "pupil_size_min": 10,
                                                    "pupil_size_max": 100
                                                }, {
                                                    # eye 1 params
                                                    "intensity_range": 23,
                                                    "pupil_size_min": 10,
                                                    "pupil_size_max": 100
                                                }],
                                                world_file="world.mp4",
                                                load_2d_pupils=load_2d_pupils,
                                                start_model_timestamp=start_model_timestamp,
                                                freeze_model_timestamp=freeze_model_timestamp,
                                                display_world_video=display_world_video
                                            )
                logging.info("Pupil detection complete.")
            
            for plugin in plugins:
                logging.info(f"Exporting gaze data for plugin {plugin}")
                if plugin is None:
                    pupil_data_loc = rec_loc + f"/offline_data/vanilla/offline_pupil.pldata"
                else:
                    pupil_data_loc = rec_loc + f"/offline_data/{plugin.__name__}/offline_pupil.pldata"
                intrinsics_loc = rec_loc + "/world.intrinsics"
                realtime_calib_points_loc = rec_loc+'/realtime_calib_points.msgpack';
                if os.path.exists(realtime_calib_points_loc):
                    print("Using exported realtime calibration points.")
                    calibrated_gazer, pupil_data = calibrate_and_validate(reference_data_loc, pupil_data_loc, intrinsics_loc, mapping_method, realtime_ref_loc=realtime_calib_points_loc)#, min_calibration_confidence=min_calibration_confidence)
                else:
                    print("Realtime calibration points have not been exported.")
                    calibrated_gazer, pupil_data = calibrate_and_validate(reference_data_loc, pupil_data_loc, intrinsics_loc, mapping_method)#, min_calibration_confidence=min_calibration_confidence)
                
                #rr_data = load_realtime_ref_data(realtime_calib_points_loc)
                #rr_data = np.array([rr_data[i]['screen_pos'] for i in range(len(rr_data))])
                    
                gaze, gaze_ts = map_pupil_data(calibrated_gazer, pupil_data)
                save_gaze_data(gaze, gaze_ts, rec_loc, plugin=plugin)
        
        logging.info('All gaze data obtained. Generating trial charts.')
        targets = []
        if vanilla_only:
            targets.append('vanilla')
            targets.append('realtime')  # Realtime is technically vanilla
            targets.append('vanilla_player')  # Allow vanilla data exported from PL Player
        
        allSessionData = processAllData(doNotLoad=not allow_session_loading, confidenceThresh=0.00, targets=targets, show_filtered_out=show_filtered_out, load_realtime_ref_data=load_realtime_ref_data)

        with open('allSessionData.pickle', 'wb') as handle:
            pickle.dump(allSessionData, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    import pandas as pd
    
    filehandler = open(b"allSessionData (thresholds enabled, frozen).pickle", "rb")#open(b"allSessionData.pickle","rb")
    allSessionData = pickle.load(filehandler)
    
    results_by_subject = {}
    results_by_resolution = {}
    
    for sessionDict in allSessionData:
        print()
        subject = int(sessionDict['subID'][0:3])
        resolution = int(sessionDict['subID'][4:7])
        run = int(sessionDict['subID'][8:])
        print(f"SUB {subject}, {resolution}x{resolution}, run {run} ({sessionDict['plExportFolder']}):")
        
        # ---------- ACCURACY ----------
        calibrationEuclideanFixErrors = sessionDict['processedSequence'][('fixError_eye2', 'euclidean')].to_numpy()
        analysisEuclideanFixErrors = sessionDict['processedCalib'][('fixError_eye2', 'euclidean')].to_numpy()
        # ------------------------------
        
        # ---------- PRECISION ----------
        calibrationPrecision = np.nanstd(sessionDict['processedSequence'][('gaze2Spherical', 'az')])
        targetLoc_targNum_AzEl = sessionDict['processedSequence']['targetLocalSpherical'].drop_duplicates().values
        calibration_precision_errors = []
        for tNum,(tX,tY) in enumerate(targetLoc_targNum_AzEl):
            gbFixTrials = sessionDict['processedSequence'].groupby([('targetLocalSpherical','az'), ('targetLocalSpherical','el')])
            trialsInGroup = gbFixTrials.get_group((tX,tY))
            gbTrials = sessionDict['processedSequence'].groupby('trialNumber')
            fixRowDataDf = gbTrials.get_group(trialsInGroup['trialNumber'].values[0])
            for x in trialsInGroup['trialNumber'][1:]:
                fixRowDataDf = pd.concat([fixRowDataDf,gbTrials.get_group(x)])
            
            meanGazeAz = np.nanmean(fixRowDataDf['gaze2Spherical']['az'])  # sigma_a
            meanGazeEl = np.nanmean(fixRowDataDf['gaze2Spherical']['el'])  # sigma_e
            err_prec = np.sum(np.sqrt(np.square(fixRowDataDf['gaze2Spherical']['az'] - meanGazeAz) + np.square(fixRowDataDf['gaze2Spherical']['el'] - meanGazeEl))) / len(fixRowDataDf['gaze2Spherical']['az'])
            calibration_precision_errors.append(err_prec)
        
        fixDF = sessionDict['fixAssessmentData']
        gb_h_w = fixDF.groupby([('gridSize', 'heightDegs'), ('gridSize', 'widthDegs')])
        for (gHeight,gWidth) in list(gb_h_w.groups.keys()):
            targetLoc_targNum_AzEl = gb_h_w.get_group((gHeight,gWidth))['fixTargetSpherical'].drop_duplicates().values
            analysis_precision_errors = []
            for tNum,(tX,tY) in enumerate(targetLoc_targNum_AzEl):
                #gbFixTrials = sessionDict['processedCalib'].groupby([('targetLocalSpherical','az'), ('targetLocalSpherical','el')])
                gbTargetType = sessionDict['trialInfo'].groupby(['targetType'])
                fixTrialsDf = gbTargetType.get_group('fixation')
                gbFixTrials = fixTrialsDf.groupby([('gridSize', 'heightDegs'), ('gridSize', 'widthDegs')])
                fixTrialsDf = gbFixTrials.get_group((gHeight,gWidth))
                gbFixTrials = fixTrialsDf.groupby([('fixTargetSpherical','az'),('fixTargetSpherical','el')])
                trialsInGroup = gbFixTrials.get_group((tX,tY))
                gbTrials = sessionDict['processedCalib'].groupby('trialNumber')
                fixRowDataDf = gbTrials.get_group(trialsInGroup['trialNumber'].values[0])
                #print(trialsInGroup['trialNumber'][1:])
                for x in trialsInGroup['trialNumber'][1:]:
                    fixRowDataDf = pd.concat([fixRowDataDf,gbTrials.get_group(x)])
                meanGazeAz = np.nanmean(fixRowDataDf['gaze2Spherical']['az'])  # sigma_a
                meanGazeEl = np.nanmean(fixRowDataDf['gaze2Spherical']['el'])  # sigma_e
                err_prec = np.sum(np.sqrt(np.square(fixRowDataDf['gaze2Spherical']['az'] - meanGazeAz) + np.square(fixRowDataDf['gaze2Spherical']['el'] - meanGazeEl))) / len(fixRowDataDf['gaze2Spherical']['az'])
                analysis_precision_errors.append(err_prec)
        # ------------------------------

        if subject not in results_by_subject:
            results_by_subject[subject] = {sessionDict['plExportFolder']: {'calibration_precision': calibration_precision_errors, 'analysis_precision': analysis_precision_errors, 'calibration': [], 'analysis': []}}
        elif sessionDict['plExportFolder'] not in results_by_subject[subject]:
            results_by_subject[subject][sessionDict['plExportFolder']] = {'calibration_precision': calibration_precision_errors, 'analysis_precision': analysis_precision_errors,  'calibration': [], 'analysis': []}
        else:
            results_by_subject[subject][sessionDict['plExportFolder']]['calibration_precision'] = np.append(results_by_subject[subject][sessionDict['plExportFolder']]['calibration_precision'], calibration_precision_errors)
            results_by_subject[subject][sessionDict['plExportFolder']]['analysis_precision'] = np.append(results_by_subject[subject][sessionDict['plExportFolder']]['calibration_precision'], analysis_precision_errors)
        if subject not in results_by_subject:
            results_by_subject[subject] = {sessionDict['plExportFolder']: {'calibration_precision': calibration_precision_errors, 'analysis_precision': analysis_precision_errors, 'calibration': calibrationEuclideanFixErrors, 'analysis': analysisEuclideanFixErrors}}
        elif sessionDict['plExportFolder'] not in results_by_subject[subject]:
            results_by_subject[subject][sessionDict['plExportFolder']] = {'calibration_precision': calibration_precision_errors, 'analysis_precision': analysis_precision_errors, 'calibration': calibrationEuclideanFixErrors, 'analysis': analysisEuclideanFixErrors}
        else:
            results_by_subject[subject][sessionDict['plExportFolder']]['calibration_precision'] = np.append(results_by_subject[subject][sessionDict['plExportFolder']]['calibration_precision'], calibration_precision_errors)
            results_by_subject[subject][sessionDict['plExportFolder']]['analysis_precision'] = np.append(results_by_subject[subject][sessionDict['plExportFolder']]['analysis_precision'], analysis_precision_errors)
            results_by_subject[subject][sessionDict['plExportFolder']]['calibration'] = np.append(results_by_subject[subject][sessionDict['plExportFolder']]['calibration'], calibrationEuclideanFixErrors)
            results_by_subject[subject][sessionDict['plExportFolder']]['analysis'] = np.append(results_by_subject[subject][sessionDict['plExportFolder']]['analysis'], analysisEuclideanFixErrors)
        
        if resolution not in results_by_resolution:
            results_by_resolution[resolution] = {subject: {sessionDict['plExportFolder']: {'calibration_precision': calibration_precision_errors, 'analysis_precision': np.array(analysis_precision_errors), 'calibration': calibrationEuclideanFixErrors, 'analysis': analysisEuclideanFixErrors}}}
        elif subject not in results_by_resolution[resolution]:
            results_by_resolution[resolution][subject] = {sessionDict['plExportFolder']: {'calibration_precision': calibration_precision_errors, 'analysis_precision': np.array(analysis_precision_errors), 'calibration': calibrationEuclideanFixErrors, 'analysis': analysisEuclideanFixErrors}}
        elif sessionDict['plExportFolder'] not in results_by_resolution[resolution][subject]:
            results_by_resolution[resolution][subject][sessionDict['plExportFolder']] = {'calibration_precision': calibration_precision_errors, 'analysis_precision': np.array(analysis_precision_errors), 'calibration': calibrationEuclideanFixErrors, 'analysis': analysisEuclideanFixErrors}
        else:
            results_by_resolution[resolution][subject][sessionDict['plExportFolder']]['calibration_precision'] = np.append(results_by_resolution[resolution][subject][sessionDict['plExportFolder']]['calibration_precision'], calibration_precision_errors)
            results_by_resolution[resolution][subject][sessionDict['plExportFolder']]['analysis_precision'] = np.append(results_by_resolution[resolution][subject][sessionDict['plExportFolder']]['analysis_precision'], np.array(analysis_precision_errors))
            results_by_resolution[resolution][subject][sessionDict['plExportFolder']]['calibration'] = np.append(results_by_resolution[resolution][subject][sessionDict['plExportFolder']]['calibration'], calibrationEuclideanFixErrors)
            results_by_resolution[resolution][subject][sessionDict['plExportFolder']]['analysis'] = np.append(results_by_resolution[resolution][subject][sessionDict['plExportFolder']]['analysis'], analysisEuclideanFixErrors)
        #print('Calibration Euclidean Fixation Errors')
        #print('min:    ',np.min(calibrationEuclideanFixErrors))
        #print('mean:   ',np.mean(calibrationEuclideanFixErrors))
        #print('max:    ',np.max(calibrationEuclideanFixErrors))
        #print()
        #print('Analysis Euclidean Fixation Errors')
        #print('min:    ',np.min(analysisEuclideanFixErrors))
        #print('mean:   ',np.mean(analysisEuclideanFixErrors))
        #print('max:    ',np.max(analysisEuclideanFixErrors))
        #print(sessionDict)
    
    vanilla_mean_calib_acc = []
    vanilla_mean_calib_prec = []
    ellseg_mean_calib_acc = []
    ellseg_mean_calib_prec = []
    
    vanilla_mean_analysis_acc = []
    vanilla_mean_analysis_prec = []
    ellseg_mean_analysis_acc = []
    ellseg_mean_analysis_prec = []

    for subject in results_by_subject.keys():#range(1, len(results_by_subject)+1):
        results = {}
        for plugin in results_by_subject[subject]:
            calibrationEuclideanFixErrors = results_by_subject[subject][plugin]['calibration']
            calibrationPrecisionFixErrors = results_by_subject[subject][plugin]['calibration_precision']
            analysisEuclideanFixErrors = results_by_subject[subject][plugin]['analysis']
            analysisPrecisionFixErrors = results_by_subject[subject][plugin]['analysis_precision']
            results[plugin] = {
                "calibrationEuclideanFixErrors": calibrationEuclideanFixErrors,
                "calibrationPrecisionFixErrors": calibrationPrecisionFixErrors,
                "analysisEuclideanFixErrors": analysisEuclideanFixErrors,
                "analysisPrecisionFixErrors": analysisPrecisionFixErrors,
                "calibration min": np.nanmin(calibrationEuclideanFixErrors),
                "calibration mean": np.nanmean(calibrationEuclideanFixErrors),
                "calibration max": np.nanmax(calibrationEuclideanFixErrors),
                "calibration precision min": np.nanmin(calibrationPrecisionFixErrors),
                "calibration precision mean": np.nanmean(calibrationPrecisionFixErrors),
                "calibration precision max": np.nanmax(calibrationPrecisionFixErrors),
                "analysis min": np.nanmin(analysisEuclideanFixErrors),
                "analysis mean": np.nanmean(analysisEuclideanFixErrors),
                "analysis max": np.nanmax(analysisEuclideanFixErrors),
                "analysis precision min": np.nanmin(analysisPrecisionFixErrors),
                "analysis precision mean": np.nanmean(analysisPrecisionFixErrors),
                "analysis precision max": np.nanmax(analysisPrecisionFixErrors),
            }

        for result in results:
            if result != 'vanilla':
                print(f"SUB {subject} ({result})")
                print('Calibration Precision Improvement (degrees)')
                print('min:    ',results['vanilla']["calibration precision min"]-results[result]["calibration precision min"])
                print('mean:   ',results['vanilla']["calibration precision mean"]-results[result]["calibration precision mean"])
                print('max:    ',results['vanilla']["calibration precision max"]-results[result]["calibration precision max"])
                print('Calibration Accuracy Improvement (degrees)')
                print('min:    ',results['vanilla']["calibration min"]-results[result]["calibration min"])
                print('mean:   ',results['vanilla']["calibration mean"]-results[result]["calibration mean"])
                print('max:    ',results['vanilla']["calibration max"]-results[result]["calibration max"])
                print()
                #print('Analysis Accuracy Improvement')
                #print('min:    ',results[result]["analysis min"]-results['vanilla']["analysis min"])
                #print('mean:   ',results[result]["analysis mean"]-results['vanilla']["analysis mean"])
                #print('max:    ',results[result]["analysis max"]-results['vanilla']["analysis max"])
                #print()
                ellseg_mean_calib_prec.append(results[result]["calibration precision mean"])
                vanilla_mean_calib_prec.append(results['vanilla']["calibration precision mean"])
                ellseg_mean_calib_acc.append(results[result]["calibration mean"])
                vanilla_mean_calib_acc.append(results['vanilla']["calibration mean"])
                
                vanilla_mean_analysis_acc.append(results['vanilla']["analysis mean"])
                vanilla_mean_analysis_prec.append(results['vanilla']["analysis precision mean"])
                ellseg_mean_analysis_acc.append(results[result]["analysis mean"])
                ellseg_mean_analysis_prec.append(results[result]["analysis precision mean"])
    
    labels = [f'{g:02d}' for g in results_by_subject.keys()]
    
    X = [results_by_resolution[400][key]['vanilla']['analysis'] for key in results_by_resolution[400].keys()]
    Y = [results_by_resolution[400][key]['Detector2DRITnetEllsegV2AllvonePlugin']['analysis'] for key in results_by_resolution[400].keys()]
    generate_box_graph(X, Y,
                        labels, ('Vanilla', 'EllSeg'), 'Accuracy Error', '400x400 Analysis Accuracy Errors by Subject (LOWER IS BETTER)',
                        '400x400 Analysis Accuracy Errors by Subject.png')
    
    X = [results_by_resolution[400][key]['vanilla']['analysis_precision'] for key in results_by_resolution[400].keys()]
    Y = [results_by_resolution[400][key]['Detector2DRITnetEllsegV2AllvonePlugin']['analysis_precision'] for key in results_by_resolution[400].keys()]
    generate_box_graph(X, Y,
                        labels, ('Vanilla', 'EllSeg'), 'Precision Error', '400x400 Analysis Precision Errors by Subject (LOWER IS BETTER)',
                        '400x400 Analysis Precision Errors by Subject.png')
    
    X = [results_by_resolution[192][key]['vanilla']['analysis'] for key in results_by_resolution[192].keys()]
    Y = [results_by_resolution[192][key]['Detector2DRITnetEllsegV2AllvonePlugin']['analysis'] for key in results_by_resolution[192].keys()]
    generate_box_graph(X, Y,
                        labels, ('Vanilla', 'EllSeg'), 'Accuracy Error', '192x192 Analysis Accuracy Errors by Subject (LOWER IS BETTER)',
                        '192x192 Analysis Accuracy Errors by Subject.png')
    
    X = [results_by_resolution[192][key]['vanilla']['analysis_precision'] for key in results_by_resolution[192].keys()]
    Y = [results_by_resolution[192][key]['Detector2DRITnetEllsegV2AllvonePlugin']['analysis_precision'] for key in results_by_resolution[192].keys()]
    generate_box_graph(X, Y,
                        labels, ('Vanilla', 'EllSeg'), 'Precision Error', '192x192 Analysis Precision Errors by Subject (LOWER IS BETTER)',
                        '192x192 Analysis Precision Errors by Subject.png')
    
    generate_errorcomp_graph(vanilla_mean_calib_acc, ellseg_mean_calib_acc, labels, ('Vanilla', 'EllSeg'),
        'Accuracy Error', 'Mean Calibration Accuracy Errors by Subject (LOWER IS BETTER)',
        'Mean Calibration Accuracy Errors by Subject.png')
    
    generate_errorcomp_graph(vanilla_mean_calib_prec, ellseg_mean_calib_prec, labels, ('Vanilla', 'EllSeg'),
        'Precision Error', 'Mean Calibration Precision Errors by Subject (LOWER IS BETTER)',
        'Mean Calibration Precision Errors by Subject.png')
    
    generate_errorcomp_graph(vanilla_mean_analysis_acc, ellseg_mean_analysis_acc, labels, ('Vanilla', 'EllSeg'),
        'Accuracy Error', 'Mean Analysis Accuracy Errors by Subject (LOWER IS BETTER)',
        'Mean Analysis Accuracy Errors by Subject.png')
    
    generate_errorcomp_graph(vanilla_mean_analysis_prec, ellseg_mean_analysis_prec, labels, ('Vanilla', 'EllSeg'),
        'Precision Error', 'Mean Analysis Precision Errors by Subject (LOWER IS BETTER)',
        'Mean Analysis Precision Errors by Subject.png')


def generate_box_graph(X, Y, labels, barlabels, ylabel, title, filename):

    from pylab import plot, show, savefig, xlim, figure, \
                ylim, legend, boxplot, setp, axes

    # function for setting the colors of the box plots pairs
    def setBoxColors(bp):
        setp(bp['boxes'][0], color='blue')
        setp(bp['caps'][0], color='blue')
        setp(bp['caps'][1], color='blue')
        setp(bp['whiskers'][0], color='blue')
        setp(bp['whiskers'][1], color='blue')
        try:
            setp(bp['fliers'][0], color='blue')
            setp(bp['fliers'][1], color='blue')
        except:
            pass
        setp(bp['medians'][0], color='blue')

        setp(bp['boxes'][1], color='red')
        setp(bp['caps'][2], color='red')
        setp(bp['caps'][3], color='red')
        setp(bp['whiskers'][2], color='red')
        setp(bp['whiskers'][3], color='red')
        #setp(bp['fliers'][2], color='red')
        #setp(bp['fliers'][3], color='red')
        setp(bp['medians'][1], color='red')

    # Some fake data to plot
    A= [[1, 2, 5,],  [7, 2]]
    B = [[5, 7, 2, 2, 5], [7, 2, 5]]
    C = [[3,2,5,7], [6, 7, 3]]

    fig = figure()
    ax = axes()
    #hold(True)
    
    for i in range(0, len(X)):
        # boxplot pair
        bp = ax.boxplot([X[i][~np.isnan(X[i])], Y[i][~np.isnan(Y[i])]], positions = [i*3+0, i*3+1], widths = 0.6, showfliers=False)
        setBoxColors(bp)

    # second boxplot pair
    #bp = boxplot(B, positions = [4, 5], widths = 0.6)
    #setBoxColors(bp)

    # thrid boxplot pair
    #bp = boxplot(C, positions = [7, 8], widths = 0.6)
    #setBoxColors(bp)

    # set axes limits and labels
    #xlim(0,9)
    #ylim(0,9)
    ax.set_xticks(np.arange(len(labels))*3 + 1)
    ax.set_xticklabels(labels)
    
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Subject Number')
    ax.set_title(title)
    
    # draw temporary red and blue lines and use them to create a legend
    hB, = ax.plot([1,1],'b-')
    hR, = ax.plot([1,1],'r-')
    legend((hB, hR),(barlabels[0], barlabels[1]))
    hB.set_visible(False)
    hR.set_visible(False)

    savefig(filename)
    """
    x = np.arange(len(labels))
    width = 0.32
        
    fig, ax = plt.subplots()
    ax.set_ylim([0, 4])
    ax.grid(axis='y')
    rects1 = ax.boxplot(X, color='blue')
    rects1 = ax.boxplot(Y, color='orange')
    
    text_x = np.arange(len(labels))
    text_y = [X[i] if X[i] > Y[i] else Y[i]+0.05 for i in range(len(X))]
    text_label = [np.around(Y[i] - X[i], 4) for i in range(len(X))]
    
    #for zx, zy, zp in zip(text_x, text_y, text_label):
    #    ax.text(zx, zy, zp, color='darkgreen' if zp<0 else 'firebrick')
    
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Subject Number')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()
    
    fig.savefig(filename)
    """

def generate_errorcomp_graph(X, Y, labels, barlabels, ylabel, title, filename):
    x = np.arange(len(labels))
    width = 0.32
        
    fig, ax = plt.subplots()
    ax.set_ylim([0, 4])
    ax.grid(axis='y')
    rects1 = ax.bar(x - width/2, X, width, label=barlabels[0], color='red')
    rects1 = ax.bar(x + width/2, Y, width, label=barlabels[1], color='green')
    
    text_x = np.arange(len(labels))
    text_y = [X[i] if X[i] > Y[i] else Y[i]+0.05 for i in range(len(X))]
    text_label = [np.around(Y[i] - X[i], 4) for i in range(len(X))]
    
    for zx, zy, zp in zip(text_x, text_y, text_label):
        ax.text(zx, zy, zp, color='darkgreen' if zp<0 else 'firebrick')
    
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Subject Number')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()
    
    fig.savefig(filename)

    
if __name__ == "__main__":
    load_dotenv()
    main()
