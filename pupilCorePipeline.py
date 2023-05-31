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
import pandas as pd

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
    "--velocity_graphs",
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
def main(allow_session_loading, skip_pupil_detection, vanilla_only, skip_vanilla, surpress_runtimewarnings, load_2d_pupils, min_calibration_confidence, show_filtered_out, display_world_video, graphs_only, velocity_graphs, core_shared_modules_loc, pipeline_loc, plugins_file):
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("numexpr").setLevel(logging.WARNING)
    logging.getLogger("OpenGL").setLevel(logging.WARNING)

    if surpress_runtimewarnings:
        import warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning) 
    
    print(pipeline_loc)
    sys.path.append(pipeline_loc)
    from core.pipeline import load_intrinsics, available_mapping_methods, patch_plugin_notify_all, calibrate_and_validate, map_pupil_data, save_gaze_data, get_first_realtime_ref_data_timestamp, get_first_ref_data_timestamp, get_last_realtime_ref_data_timestamp, get_last_ref_data_timestamp, load_realtime_ref_data
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

        skip_3d_detection = False
        if mapping_method_label == "2D":
            skip_3d_detection = True

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
                    elif os.path.exists(reference_data_loc):
                        start_model_timestamp = get_first_ref_data_timestamp(reference_data_loc)
                        freeze_model_timestamp = get_last_ref_data_timestamp(reference_data_loc)
                    else:
                        print("No reference data found, gaze prediction may fail.")
                        start_model_timestamp = None
                        freeze_model_timestamp = None
                    
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
                                                display_world_video=display_world_video,
                                                mapping_method=mapping_method,
                                                skip_3d_detection=skip_3d_detection
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
                    try:
                        calibrated_gazer, pupil_data = calibrate_and_validate(reference_data_loc, pupil_data_loc, intrinsics_loc, mapping_method, realtime_ref_loc=realtime_calib_points_loc)#, min_calibration_confidence=min_calibration_confidence)
                    except FileNotFoundError:
                        print("No calibration points found.")
                        continue
                else:
                    print("Realtime calibration points have not been exported.")
                    try:
                        calibrated_gazer, pupil_data = calibrate_and_validate(reference_data_loc, pupil_data_loc, intrinsics_loc, mapping_method)#, min_calibration_confidence=min_calibration_confidence)
                    except FileNotFoundError:
                        print("No calibration points found.")
                        continue
                
                #rr_data = load_realtime_ref_data(realtime_calib_points_loc)
                #rr_data = np.array([rr_data[i]['screen_pos'] for i in range(len(rr_data))])
                
                gaze, gaze_ts = map_pupil_data(calibrated_gazer, pupil_data, rec_loc)
                save_gaze_data(gaze, gaze_ts, rec_loc, plugin=plugin)
        
        logging.info('All gaze data obtained. Generating trial charts.')
        targets = []
        if vanilla_only:
            targets.append('vanilla')
            targets.append('realtime')  # Realtime is technically vanilla
            targets.append('vanilla_player')  # Allow vanilla data exported from PL Player
        
        allSessionData = processAllData(doNotLoad=not allow_session_loading, confidenceThresh=0.00, targets=targets, show_filtered_out=show_filtered_out, load_realtime_ref_data=load_realtime_ref_data, override_to_2d=skip_3d_detection)

        with open('allSessionData.pickle', 'wb') as handle:
            pickle.dump(allSessionData, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    from pylab import savefig

    filehandler = open(b"allSessionData.pickle","rb")
    allSessionData = pickle.load(filehandler)
    print(len(allSessionData))
    print(allSessionData[0].keys())
    print(len(allSessionData[0]['processedSequence'][('targetLocalSpherical', 'az')]))
    #exit()

    if velocity_graphs:
        # THIS SECTION CURRENTLY ASSUMES 2D GAZE DATA (deprojected_norm_pos rather than gaze_normal)
        for subjectFolder in next(os.walk('./Data'))[1]:
            print("Processing velocity data for " + subjectFolder)
            session = None
            for d in allSessionData:
                if d['subID'] == subjectFolder[-9:]:
                    session = d
                    break
            if session is None:
                # contingency for accidental _01/_02 formatting
                for d in allSessionData:
                    if d['subID'] == subjectFolder[-10:]:
                        session = d
                        break
            gazeDataFolder = './Data/'+subjectFolder+'/S001/PupilData/000/Exports/'
            
            plt.figure()
            ax = plt.subplot()
            detect_non_saccads(gazeDataFolder, 'vanilla', 'Native', session, 'blue', ax)
            detect_non_saccads(gazeDataFolder, 'Detector2DRITnetEllsegV2AllvonePlugin', 'EllSegGen', session, 'red', ax)
            plt.clf()
            plt.close('all')
    
    results_by_subject = {}
    results_by_resolution = {}
    results_by_eccentricity = {}
    
    i = 1
    for sessionDict in allSessionData:
        print()
        try:
            subject = int(sessionDict['subID'][0:3])
            resolution = int(sessionDict['subID'][4:7])
            run = int(sessionDict['subID'][8:])
        except ValueError:
            subject = sessionDict['subID'][0:-4]
            resolution = int(sessionDict['subID'][-3:])
            run = 1
        plugin = sessionDict['plExportFolder']
        if subject == 4:
            continue
        print(f"({i}/{len(allSessionData)})SUB {subject}, {resolution}x{resolution}, run {run} ({sessionDict['plExportFolder']}):")
        i += 1
        
        ecc_targetLoc_targNum_AzEl = sessionDict['processedCalib']['targetLocalSpherical'].drop_duplicates().values
        # ---------- ACCURACY ----------
        # THIS IS WHERE I ADJUST IT TO BE MORE LIKE THE PRECISION APPROACH BELOW
        #       (meaning averaging up per-target and adding those to calibrationEuclideanFixErrors and analysisEuclideanFixErrors
        #       instead of just adding every gaze point to them
        """
        OLD METHOD (every point is added)
        calibrationEuclideanFixErrors = sessionDict['processedSequence'][('fixError_eye2', 'euclidean')].to_numpy()
        analysisEuclideanFixErrors = sessionDict['processedCalib'][('fixError_eye2', 'euclidean')].to_numpy()
        """
        calibrationEuclideanFixErrors = []
        targetLoc_targNum_AzEl = sessionDict['processedSequence']['targetLocalSpherical'].drop_duplicates().values
        for tNum,(tX,tY) in enumerate(targetLoc_targNum_AzEl):
            gbFixTrials = sessionDict['processedSequence'].groupby([('targetLocalSpherical','az'), ('targetLocalSpherical','el')])
            trialsInGroup = gbFixTrials.get_group((tX,tY))
            gbTrials = sessionDict['processedSequence'].groupby('trialNumber')
            fixRowDataDf = gbTrials.get_group(trialsInGroup['trialNumber'].values[0])
            for x in trialsInGroup['trialNumber'][1:]:
                fixRowDataDf = pd.concat([fixRowDataDf,gbTrials.get_group(x)])
            err_acc = np.sum(fixRowDataDf['fixError_eye2']['euclidean'].to_numpy()) / len(fixRowDataDf['fixError_eye2']['euclidean'].to_numpy())
            calibrationEuclideanFixErrors.append(err_acc)

        eccentricities = []
        eccentricitiesAccDict = {}
        analysisEuclideanFixErrors = []
        fixDF = sessionDict['fixAssessmentData']
        gb_h_w = fixDF.groupby([('gridSize', 'heightDegs'), ('gridSize', 'widthDegs')])
        for (gHeight,gWidth) in list(gb_h_w.groups.keys()):
            targetLoc_targNum_AzEl = gb_h_w.get_group((gHeight,gWidth))['fixTargetSpherical'].drop_duplicates().values
            for tNum,(tX,tY) in enumerate(targetLoc_targNum_AzEl):
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
                err_acc = np.nansum(fixRowDataDf['fixError_eye2']['euclidean'].to_numpy()) / np.count_nonzero(~np.isnan(fixRowDataDf['fixError_eye2']['euclidean'].to_numpy()))

                eccentricity = np.round(np.sqrt(tX**2 + tY**2))
                if eccentricity not in eccentricities and not np.isnan(eccentricity):
                    eccentricities.append(eccentricity)
                if eccentricity in eccentricitiesAccDict:
                    eccentricitiesAccDict[eccentricity].append(err_acc)
                else:
                    eccentricitiesAccDict[eccentricity] = [err_acc]
                analysisEuclideanFixErrors.append(err_acc)

        sessionDict['processedCalib']['eccentricity'] = np.round(np.linalg.norm(sessionDict['processedCalib']['targetLocalSpherical'].values, axis=1))
        #eccentricities = []
        #for tNum,(tX,tY) in enumerate(ecc_targetLoc_targNum_AzEl):
        #    eccentricity = np.round(np.sqrt(tX**2 + tY**2))
        #    if eccentricity not in eccentricities and not np.isnan(eccentricity):
        #        eccentricities.append(eccentricity)
        
        #eccentricitiesAccDict = {}
        for eccentricity in eccentricities:
            #gbEccentricity = sessionDict['processedCalib'].groupby('eccentricity')
            #fixRowDataDf = gbEccentricity.get_group(eccentricity)
            #err_acc = np.nansum(fixRowDataDf['fixError_eye2']['euclidean'].to_numpy()) / np.count_nonzero(~np.isnan(fixRowDataDf['fixError_eye2']['euclidean'].to_numpy()))
            #eccentricitiesAccDict[eccentricity] = err_acc
            eccentricitiesAccDict[eccentricity] = eccentricitiesAccDict[eccentricity]#np.mean(eccentricitiesAccDict[eccentricity])

        
        # ---------- PRECISION ----------
        targetLoc_targNum_AzEl = sessionDict['processedSequence']['targetLocalSpherical'].drop_duplicates().values
        calibrationPrecision = np.nanstd(sessionDict['processedSequence'][('gaze2Spherical', 'az')])
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
        analysis_precision_errors = []  # wrong position (now right position?)
        eccentricitiesPrecDict = {}
        for (gHeight,gWidth) in list(gb_h_w.groups.keys()):
            targetLoc_targNum_AzEl = gb_h_w.get_group((gHeight,gWidth))['fixTargetSpherical'].drop_duplicates().values
            
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
                
                eccentricity = np.round(np.sqrt(tX**2 + tY**2))
                if eccentricity in eccentricitiesPrecDict:
                    eccentricitiesPrecDict[eccentricity].append(err_prec)
                else:
                    eccentricitiesPrecDict[eccentricity] = [err_prec]

        for eccentricity in eccentricities:
            #gbEccentricity = sessionDict['processedCalib'].groupby('eccentricity')
            #eccentricity_trials = gbEccentricity.get_group(eccentricity)
            #gbTLS = eccentricity_trials.groupby([('targetLocalSpherical', 'az'), ('targetLocalSpherical', 'el')])
            #for name, group in gbTLS:
            #    meanGazeAz = np.nanmean(group['gaze2Spherical']['az'])
            #    meanGazeEl = np.nanmean(group['gaze2Spherical']['el'])
            #    err_prec = np.sum(np.sqrt(np.square(group['gaze2Spherical']['az'] - meanGazeAz) + np.square(group['gaze2Spherical']['el'] - meanGazeEl))) / len(group['gaze2Spherical']['az'])
            #    if not eccentricity in eccentricitiesPrecDict:
            #        eccentricitiesPrecDict[eccentricity] = [err_prec]
            #    else:
            #        eccentricitiesPrecDict[eccentricity].append(err_prec)
            eccentricitiesPrecDict[eccentricity] = eccentricitiesPrecDict[eccentricity]
        # ------------------------------

        if subject not in results_by_subject:
            results_by_subject[subject] = {sessionDict['plExportFolder']: {'calibration_precision': np.array(calibration_precision_errors), 'analysis_precision': np.array(analysis_precision_errors), 'calibration': np.array([]), 'analysis': np.array([])}}
        elif sessionDict['plExportFolder'] not in results_by_subject[subject]:
            results_by_subject[subject][sessionDict['plExportFolder']] = {'calibration_precision': np.array(calibration_precision_errors), 'analysis_precision': np.array(analysis_precision_errors),  'calibration': np.array([]), 'analysis': np.array([])}
        else:
            results_by_subject[subject][sessionDict['plExportFolder']]['calibration_precision'] = np.append(results_by_subject[subject][sessionDict['plExportFolder']]['calibration_precision'], calibration_precision_errors)
            results_by_subject[subject][sessionDict['plExportFolder']]['analysis_precision'] = np.append(results_by_subject[subject][sessionDict['plExportFolder']]['calibration_precision'], analysis_precision_errors)
        if subject not in results_by_subject:
            results_by_subject[subject] = {sessionDict['plExportFolder']: {'calibration_precision': np.array(calibration_precision_errors), 'analysis_precision': np.array(analysis_precision_errors), 'calibration': np.array(calibrationEuclideanFixErrors), 'analysis': np.array(analysisEuclideanFixErrors)}}
        elif sessionDict['plExportFolder'] not in results_by_subject[subject]:
            results_by_subject[subject][sessionDict['plExportFolder']] = {'calibration_precision': np.array(calibration_precision_errors), 'analysis_precision': np.array(analysis_precision_errors), 'calibration': np.array(calibrationEuclideanFixErrors), 'analysis': np.array(analysisEuclideanFixErrors)}
        else:
            results_by_subject[subject][sessionDict['plExportFolder']]['calibration_precision'] = np.append(results_by_subject[subject][sessionDict['plExportFolder']]['calibration_precision'], calibration_precision_errors)
            results_by_subject[subject][sessionDict['plExportFolder']]['analysis_precision'] = np.append(results_by_subject[subject][sessionDict['plExportFolder']]['analysis_precision'], analysis_precision_errors)
            results_by_subject[subject][sessionDict['plExportFolder']]['calibration'] = np.append(results_by_subject[subject][sessionDict['plExportFolder']]['calibration'], calibrationEuclideanFixErrors)
            results_by_subject[subject][sessionDict['plExportFolder']]['analysis'] = np.append(results_by_subject[subject][sessionDict['plExportFolder']]['analysis'], analysisEuclideanFixErrors)
        
        if resolution not in results_by_resolution:
            results_by_resolution[resolution] = {subject: {sessionDict['plExportFolder']: {'calibration_precision': np.array(calibration_precision_errors), 'analysis_precision': np.array(analysis_precision_errors), 'calibration': np.array(calibrationEuclideanFixErrors), 'analysis': np.array(analysisEuclideanFixErrors)}}}
        elif subject not in results_by_resolution[resolution]:
            results_by_resolution[resolution][subject] = {sessionDict['plExportFolder']: {'calibration_precision': np.array(calibration_precision_errors), 'analysis_precision': np.array(analysis_precision_errors), 'calibration': np.array(calibrationEuclideanFixErrors), 'analysis': np.array(analysisEuclideanFixErrors)}}
        elif sessionDict['plExportFolder'] not in results_by_resolution[resolution][subject]:
            results_by_resolution[resolution][subject][sessionDict['plExportFolder']] = {'calibration_precision': np.array(calibration_precision_errors), 'analysis_precision': np.array(analysis_precision_errors), 'calibration': np.array(calibrationEuclideanFixErrors), 'analysis': np.array(analysisEuclideanFixErrors)}
        else:
            results_by_resolution[resolution][subject][sessionDict['plExportFolder']]['calibration_precision'] = np.append(results_by_resolution[resolution][subject][sessionDict['plExportFolder']]['calibration_precision'], calibration_precision_errors)
            results_by_resolution[resolution][subject][sessionDict['plExportFolder']]['analysis_precision'] = np.append(results_by_resolution[resolution][subject][sessionDict['plExportFolder']]['analysis_precision'], np.array(analysis_precision_errors))
            results_by_resolution[resolution][subject][sessionDict['plExportFolder']]['calibration'] = np.append(results_by_resolution[resolution][subject][sessionDict['plExportFolder']]['calibration'], calibrationEuclideanFixErrors)
            results_by_resolution[resolution][subject][sessionDict['plExportFolder']]['analysis'] = np.append(results_by_resolution[resolution][subject][sessionDict['plExportFolder']]['analysis'], analysisEuclideanFixErrors)
        
        for eccentricity in eccentricities:
            if eccentricity not in results_by_eccentricity:
                results_by_eccentricity[eccentricity] = {resolution: {subject: {sessionDict['plExportFolder']: {'calibration_precision': None, 'analysis_precision': np.array(eccentricitiesPrecDict[eccentricity]), 'calibration': None, 'analysis': np.array(eccentricitiesAccDict[eccentricity])}}}}
            elif resolution not in results_by_eccentricity[eccentricity]:
                results_by_eccentricity[eccentricity][resolution] = {subject: {sessionDict['plExportFolder']: {'calibration_precision': None, 'analysis_precision': np.array(eccentricitiesPrecDict[eccentricity]), 'calibration': None, 'analysis': np.array(eccentricitiesAccDict[eccentricity])}}}
            elif subject not in results_by_eccentricity[eccentricity][resolution]:
                results_by_eccentricity[eccentricity][resolution][subject] = {sessionDict['plExportFolder']: {'calibration_precision': None, 'analysis_precision': np.array(eccentricitiesPrecDict[eccentricity]), 'calibration': None, 'analysis': np.array(eccentricitiesAccDict[eccentricity])}}
            elif sessionDict['plExportFolder'] not in results_by_eccentricity[eccentricity][resolution][subject]:
                results_by_eccentricity[eccentricity][resolution][subject][sessionDict['plExportFolder']] = {'calibration_precision': None, 'analysis_precision': np.array(eccentricitiesPrecDict[eccentricity]), 'calibration': None, 'analysis': np.array(eccentricitiesAccDict[eccentricity])}
            else:
                results_by_eccentricity[eccentricity][resolution][subject][sessionDict['plExportFolder']]['analysis'] = np.append(results_by_eccentricity[eccentricity][resolution][subject][sessionDict['plExportFolder']]['analysis'], eccentricitiesAccDict[eccentricity])
                results_by_eccentricity[eccentricity][resolution][subject][sessionDict['plExportFolder']]['calibration'] = None
                results_by_eccentricity[eccentricity][resolution][subject][sessionDict['plExportFolder']]['analysis_precision'] = np.append(results_by_eccentricity[eccentricity][resolution][subject][sessionDict['plExportFolder']]['analysis_precision'], eccentricitiesPrecDict[eccentricity])
                results_by_eccentricity[eccentricity][resolution][subject][sessionDict['plExportFolder']]['calibration_precision'] = None
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

    vanpoints = np.array([])
    otherpoints = {}
    
    prec_vanpoints = np.array([])
    prec_otherpoints = {}

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

        if True:
            vanpoints = np.concatenate((vanpoints, np.array(results['vanilla']['analysisEuclideanFixErrors'])))
            for method in results.keys():
                if method != "vanilla":
                    if method not in otherpoints:
                        otherpoints[method] = [np.array(results['vanilla']['analysisEuclideanFixErrors']), np.array(results[method]['analysisEuclideanFixErrors'])]
                    else:
                        otherpoints[method][0] = np.concatenate((otherpoints[method][0], np.array(results['vanilla']['analysisEuclideanFixErrors'])))
                        otherpoints[method][1] = np.concatenate((otherpoints[method][1], np.array(results[method]['analysisEuclideanFixErrors'])))
            
            prec_vanpoints = np.concatenate((prec_vanpoints, np.array(results['vanilla']['analysisPrecisionFixErrors'])))
            for method in results.keys():
                if method != "vanilla":
                    if method not in prec_otherpoints:
                        prec_otherpoints[method] = [np.array(results['vanilla']['analysisPrecisionFixErrors']), np.array(results[method]['analysisPrecisionFixErrors'])]
                    else:
                        prec_otherpoints[method][0] = np.concatenate((prec_otherpoints[method][0], np.array(results['vanilla']['analysisPrecisionFixErrors'])))
                        prec_otherpoints[method][1] = np.concatenate((prec_otherpoints[method][1], np.array(results[method]['analysisPrecisionFixErrors'])))

        for result in results:
            #if result != 'vanilla':
            if result == 'Detector2DESFnetPlugin':#'Detector2DRITnetEllsegV2AllvonePlugin':
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
    
    if True:
        MARKERSIZE = 2
        colors = ['red', 'green', 'purple', 'black']
        
        # ----- Plot All Points' Accuracy Error Over Native Accuracy Error -----
        plt.plot([0, np.max(vanpoints)], [0, np.max(vanpoints)], '-', label="Native Accuracy Error", c='blue')
        i = 0
        for method in otherpoints.keys():
            plt.plot(otherpoints[method][0], otherpoints[method][1], '+', label=method, c=colors[i], markersize=MARKERSIZE)
            i += 1
        plt.title("NN-assisted Accuracy Error vs Native Accuracy Error")
        plt.xlabel("Native Accuracy Error")
        plt.ylabel("NN-assisted Accuracy Error")
        plt.legend()
        #plt.xlim(0, 6)
        #plt.ylim(0, 6)
        plt.savefig('./figOut/VANIL_ACCURACY_COMP.png', bbox_inches='tight')
        plt.clf()

        # ----- Plot All Points' Precision Error Over Native Precision Error -----
        plt.plot([0, np.max(prec_vanpoints)], [0, np.max(prec_vanpoints)], '-', label="Native Precision Error", c='blue')
        i = 0
        for method in prec_otherpoints.keys():
            plt.plot(prec_otherpoints[method][0], prec_otherpoints[method][1], '+', label=method, c=colors[i], markersize=MARKERSIZE)
            i += 1
        plt.title("NN-assisted Precision Error vs Native Precision Error")
        plt.xlabel("Native Precision Error")
        plt.ylabel("NN-assisted Precision Error")
        plt.legend()
        #plt.xlim(0, 6)
        #plt.ylim(0, 6)
        plt.savefig('./figOut/VANIL_PRECISION_COMP.png', bbox_inches='tight')
        plt.clf()
        
        # ----- Plot All Points' Binned&Averaged Accuracy Error Over Native Binned&Averaged Accuracy Error -----
        from decimal import Decimal
        bins = {}
        expand_bin_size_at = 12.0
        colors = ['red', 'green', 'purple', 'black']
        plt.plot([0, np.max(vanpoints)], [0, np.max(vanpoints)], '-', label="Native Accuracy Error", c='blue')
        i = 0
        for method in otherpoints.keys():
            bins[method] = []
            vanillas = otherpoints[method][0]
            percentile50 = np.percentile(vanillas, 50)
            percentile90 = np.percentile(vanillas, 90)
            percentile95 = np.percentile(vanillas, 95)
            max_vanillas = round(np.max(vanillas), 1)  # nearest tenth place
            bin_group_1_size = 0.25
            bin_group_3_size = 3.0
            bg1s_decimal = Decimal(str(bin_group_1_size))
            bg3s_decimal = Decimal(str(bin_group_3_size))
            curr = 0.0
            while curr <= max_vanillas:
                bins[method].append([])
                if curr < expand_bin_size_at:
                    curr += bin_group_1_size
                else:
                    curr += bin_group_3_size
            bins[method].append([])
            for pt_idx in range(len(otherpoints[method][0])):
                pt = otherpoints[method][0][pt_idx]
                if pt < expand_bin_size_at + (bin_group_1_size / 2):
                    if (pt % bin_group_1_size) < (bin_group_1_size / 2):
                        modfix = float(Decimal(pt) % bg1s_decimal)
                        idx = pt - modfix
                    else:
                        modfix = float(Decimal(pt) % bg1s_decimal)
                        idx = pt + (bin_group_1_size - modfix)
                    idx = idx / bin_group_1_size
                    bins[method][int(round(idx))].append(otherpoints[method][1][pt_idx])
                else:
                    if (pt % bin_group_3_size) < (bin_group_3_size / 2):
                        modfix = float(Decimal(pt) % bg3s_decimal)
                        idx = pt - modfix
                    else:
                        modfix = float(Decimal(pt) % bg3s_decimal)
                        idx = pt + (bin_group_3_size - modfix)
                    #idx = round(idx)
                    idx = (expand_bin_size_at / bin_group_1_size) + ((idx - expand_bin_size_at) / bin_group_3_size)
                    print(pt_idx, "/", len(otherpoints[method][1]))
                    print(int(round(idx)), "/", len(bins[method]))
                    bins[method][int(round(idx))].append(otherpoints[method][1][pt_idx])
            X = []
            Y = []
            for idx in range(len(bins[method])):
                if len(bins[method][idx]):
                    if idx <= (expand_bin_size_at / bin_group_1_size):
                        bin = bin_group_1_size * idx
                    else:
                        bin = expand_bin_size_at + bin_group_3_size * (idx - (expand_bin_size_at / bin_group_1_size))
                    X.append(bin)
                    Y.append(np.mean(bins[method][idx]))
            plt.plot(X, Y, '-o', markersize=5, label=method, c=colors[i])
            i += 1
        
        plt.title("NN-assisted Binned Mean Accuracy Error vs Native Accuracy Error")
        plt.xlabel("Native Accuracy Error")
        plt.ylabel("NN-assisted Accuracy Error (bin interval = {} -> {})".format(bin_group_1_size, bin_group_3_size))
        #plt.xlim(0, 12)
        #plt.ylim(0, 12)
        plt.axvline(percentile50, linestyle=":", color="blue", label="50th percentile")
        plt.axvline(percentile90, linestyle=":", color="green", label="90th percentile")
        plt.axvline(percentile95, linestyle=":", color="red", label="95th percentile")
        plt.legend()
        plt.savefig('./figOut/VANIL_ACC_BINNED_COMP.png', bbox_inches='tight')
        plt.xlim(0, percentile90)
        plt.ylim(0, 22)
        plt.savefig('./figOut/VANIL_ACC_BINNED_LIM_PERCENTILE90_COMP.png', bbox_inches='tight')
        plt.clf()
        
        
        # ----- Plot All Points' Binned&Averaged Precision Error Over Native Binned&Averaged Precision Error -----
        from decimal import Decimal
        bins = {}
        expand_bin_size_at = 2.0
        colors = ['red', 'green', 'purple', 'black']
        plt.plot([0, np.max(prec_vanpoints)], [0, np.max(prec_vanpoints)], '-', label="Native Precision Error", c='blue')
        i = 0
        for method in prec_otherpoints.keys():
            bins[method] = []
            vanillas = prec_otherpoints[method][0]
            percentile50 = np.percentile(vanillas, 50)
            percentile90 = np.percentile(vanillas, 90)
            percentile95 = np.percentile(vanillas, 95)
            max_vanillas = round(np.max(vanillas), 1)  # nearest tenth place
            bin_group_1_size = 0.1
            bin_group_3_size = 0.25
            bg1s_decimal = Decimal(str(bin_group_1_size))
            bg3s_decimal = Decimal(str(bin_group_3_size))
            curr = 0.0
            while curr <= max_vanillas:
                bins[method].append([])
                if curr < expand_bin_size_at:
                    curr += bin_group_1_size
                else:
                    curr += bin_group_3_size
            bins[method].append([])
            for pt_idx in range(len(prec_otherpoints[method][0])):
                pt = prec_otherpoints[method][0][pt_idx]
                if pt < expand_bin_size_at + (bin_group_1_size / 2):
                    if (pt % bin_group_1_size) < (bin_group_1_size / 2):
                        modfix = float(Decimal(pt) % bg1s_decimal)
                        idx = pt - modfix
                    else:
                        modfix = float(Decimal(pt) % bg1s_decimal)
                        idx = pt + (bin_group_1_size - modfix)
                    idx = idx / bin_group_1_size
                    bins[method][int(round(idx))].append(prec_otherpoints[method][1][pt_idx])
                else:
                    modfix = float(Decimal(pt) % bg3s_decimal)
                    if modfix < (bin_group_3_size / 2):
                        idx = pt - modfix
                    else:
                        idx = pt + (bin_group_3_size - modfix)
                    #idx = round(idx)
                    idx = (expand_bin_size_at / bin_group_1_size) + ((idx - expand_bin_size_at) / bin_group_3_size)
                    print(pt_idx, "/", len(prec_otherpoints[method][1]))
                    print(int(round(idx)), "/", len(bins[method]))
                    bins[method][int(round(idx))].append(prec_otherpoints[method][1][pt_idx])
            X = []
            Y = []
            for idx in range(len(bins[method])):
                if len(bins[method][idx]):
                    if idx <= (expand_bin_size_at / bin_group_1_size):
                        bin = bin_group_1_size * idx
                    else:
                        bin = expand_bin_size_at + bin_group_3_size * (idx - (expand_bin_size_at / bin_group_1_size))
                    X.append(bin)
                    Y.append(np.mean(bins[method][idx]))
            plt.plot(X, Y, '-o', markersize=5, label=method, c=colors[i])
            i += 1
        
        plt.title("NN-assisted Binned Mean Precision Error vs Native Precision Error")
        plt.xlabel("Native Precision Error")
        plt.ylabel("NN-assisted Precision Error (bin interval = {} -> {})".format(bin_group_1_size, bin_group_3_size))
        #plt.xlim(0, 12)
        #plt.ylim(0, 12)
        plt.axvline(percentile50, linestyle=":", color="blue", label="50th percentile")
        plt.axvline(percentile90, linestyle=":", color="green", label="90th percentile")
        plt.axvline(percentile95, linestyle=":", color="red", label="95th percentile")
        plt.legend()
        plt.savefig('./figOut/VANIL_PREC_BINNED_COMP.png', bbox_inches='tight')
        plt.xlim(0, percentile90)
        plt.ylim(0, 10)
        plt.savefig('./figOut/VANIL_PREC_BINNED_LIM_PERCENTILE90_COMP.png', bbox_inches='tight')
        plt.clf()

    try:
        # int name
        labels = [f'{g:02d}' for g in results_by_subject.keys()]
    except ValueError:
        # str name
        labels = [g for g in results_by_subject.keys()]

    try:
        X = [results_by_resolution[400][key]['vanilla']['analysis'] for key in results_by_resolution[400].keys()]
        Y = [results_by_resolution[400][key]['Detector2DRITnetEllsegV2AllvonePlugin']['analysis'] for key in results_by_resolution[400].keys()]
        generate_box_graph(X, Y,
                            labels, ('Vanilla', 'EllSeg'), 'Accuracy Error', '400x400 Analysis Accuracy Errors by Subject (LOWER IS BETTER)',
                            '400x400 Analysis Accuracy Errors by Subject.png',
                            ylimit=12)
    except KeyError:
        pass

    try:
        X = [results_by_resolution[400][key]['vanilla']['analysis_precision'] for key in results_by_resolution[400].keys()]
        Y = [results_by_resolution[400][key]['Detector2DRITnetEllsegV2AllvonePlugin']['analysis_precision'] for key in results_by_resolution[400].keys()]
        generate_box_graph(X, Y,
                            labels, ('Vanilla', 'EllSeg'), 'Precision Error', '400x400 Analysis Precision Errors by Subject (LOWER IS BETTER)',
                            '400x400 Analysis Precision Errors by Subject.png',
                            ylimit=2)
    except KeyError:
        pass

    try:
        X = [results_by_resolution[192][key]['vanilla']['analysis'] for key in results_by_resolution[192].keys()]
        Y = [results_by_resolution[192][key]['Detector2DRITnetEllsegV2AllvonePlugin']['analysis'] for key in results_by_resolution[192].keys()]
        generate_box_graph(X, Y,
                            labels, ('Vanilla', 'EllSeg'), 'Accuracy Error', '192x192 Analysis Accuracy Errors by Subject (LOWER IS BETTER)',
                            '192x192 Analysis Accuracy Errors by Subject.png',
                            ylimit=29)
    except KeyError:
        pass

    try:
        X = [results_by_resolution[192][key]['vanilla']['analysis_precision'] for key in results_by_resolution[192].keys()]
        Y = [results_by_resolution[192][key]['Detector2DRITnetEllsegV2AllvonePlugin']['analysis_precision'] for key in results_by_resolution[192].keys()]
        generate_box_graph(X, Y,
                            labels, ('Vanilla', 'EllSeg'), 'Precision Error', '192x192 Analysis Precision Errors by Subject (LOWER IS BETTER)',
                            '192x192 Analysis Precision Errors by Subject.png',
                            ylimit=2)
    except KeyError:
        pass

    for subject_num in range(1, 7):
        if subject_num != 4:
            try:
                X = [results_by_eccentricity[key][400][subject_num]['vanilla']['analysis'] for key in results_by_eccentricity.keys()]
                Y = [results_by_eccentricity[key][400][subject_num]['Detector2DRITnetEllsegV2AllvonePlugin']['analysis'] for key in results_by_eccentricity.keys()]
                generate_box_graph(X, Y,
                                    results_by_eccentricity.keys(), ('Vanilla', 'EllSeg'), 'Accuracy Error (degrees)', '400x400 Sub{} Analysis Accuracy Errors by Eccentricity (LOWER IS BETTER)'.format(subject_num),
                                    '400x400 Sub{} Analysis Accuracy Errors by Eccentricity.png'.format(subject_num),
                                    ylimit=None, xlabel='Eccentricity (degrees)')
            except KeyError:
                pass

            try:
                X = [results_by_eccentricity[key][400][subject_num]['vanilla']['analysis_precision'] for key in results_by_eccentricity.keys()]
                Y = [results_by_eccentricity[key][400][subject_num]['Detector2DRITnetEllsegV2AllvonePlugin']['analysis_precision'] for key in results_by_eccentricity.keys()]
                generate_box_graph(X, Y,
                                    results_by_eccentricity.keys(), ('Vanilla', 'EllSeg'), 'Precision Error (degrees)', '400x400 Sub{} Analysis Precision Errors by Eccentricity (LOWER IS BETTER)'.format(subject_num),
                                    '400x400 Sub{} Analysis Precision Errors by Eccentricity.png'.format(subject_num),
                                    ylimit=None, xlabel='Eccentricity (degrees)')
            except KeyError:
                pass

    output_grids = {}
    for eccentricity in results_by_eccentricity.keys():
        for resolution in (192, 400):
            for subject_num in range(1, 8):
                if subject_num != 4:
                    for plugin in ('vanilla', 'Detector2DRITnetEllsegV2AllvonePlugin'):
                        key = str(subject_num)+'-'+str(resolution)+'-'+str(eccentricity)+'-'+plugin
                        vanilla_key = str(subject_num)+'-'+str(resolution)+'-'+str(eccentricity)+'-vanilla'
                        try:
                            results_accuracy = results_by_eccentricity[eccentricity][resolution][subject_num][plugin]['analysis']
                            results_accuracy_vanilla = results_by_eccentricity[eccentricity][resolution][subject_num]['vanilla']['analysis']
                            print(results_accuracy)
                            print(results_accuracy_vanilla)
                            print()
                            results_precision = results_by_eccentricity[eccentricity][resolution][subject_num][plugin]['analysis_precision']
                            results_precision_vanilla = results_by_eccentricity[eccentricity][resolution][subject_num]['vanilla']['analysis_precision']
                        except KeyError:
                            continue
                        output_grids[key] = {}
                        average_accuracy = np.mean(results_accuracy[~np.isnan(results_accuracy)])
                        average_precision = np.mean(results_precision[~np.isnan(results_precision)])
                        accurate_above_vanilla = [(1 if i <= j else 0) for (i,j) in zip(results_accuracy, results_accuracy_vanilla)]
                        num_accurate_above_vanilla = np.sum(accurate_above_vanilla)
                        precise_above_vanilla = [(1 if i <= j else 0) for (i,j) in zip(results_precision, results_precision_vanilla)]
                        num_precise_above_vanilla = np.sum(precise_above_vanilla)

                        output_grids[key]['numAccurateAboveVanilla'] = num_accurate_above_vanilla
                        output_grids[key]['numPreciseAboveVanilla'] = num_precise_above_vanilla

                        output_grids[key]['averageAccuracy'] = average_accuracy
                        output_grids[key]['averagePrecision'] = average_precision
                        if plugin == 'vanilla':
                            output_grids[key]['percentAccurateAboveVanilla'] = '-'
                            output_grids[key]['percentPreciseAboveVanilla'] = '-'
                        else:
                            output_grids[key]['percentAccurateAboveVanilla'] = str(100 * round(num_accurate_above_vanilla / output_grids[vanilla_key]['numAccurateAboveVanilla'], 2)) + '%'
                            output_grids[key]['percentPreciseAboveVanilla'] = str(100 * round(num_precise_above_vanilla / output_grids[vanilla_key]['numPreciseAboveVanilla'], 2)) + '%'
    
    output_grids_df = pd.DataFrame(output_grids)
    output_grids_df.to_csv('output_csv.csv')
    #print(output_grids_df)
    #exit()
    subjects = [2, 3, 5, 6, 7, 8, 9, 10, 11]

    subject_labels = ["Sub1Vanilla", "Sub1EllSeg"]
    for subj_num in subjects:
        try:
            subject_labels.append("Sub{}Vanilla".format(subj_num))
        except KeyError:
            pass
        try:
            subject_labels.append("Sub{}EllSeg".format(subj_num))
        except KeyError:
            pass

    # GRAPH SET 1: Comparing all subjects' accuracy and precision with each other
    # (These graphs are very dense and have a lot of info in them)

    try:
        X = [results_by_eccentricity[key][400][1]['vanilla']['analysis'] for key in results_by_eccentricity.keys()]
        Y = [results_by_eccentricity[key][400][1]['Detector2DRITnetEllsegV2AllvonePlugin']['analysis'] for key in results_by_eccentricity.keys()]
        Z = []
        for subj_num in subjects:
            curr = [results_by_eccentricity[key][400][subj_num]['vanilla']['analysis'] for key in results_by_eccentricity.keys()]
            Z.append(curr)
            Z.append([results_by_eccentricity[key][400][subj_num]['Detector2DRITnetEllsegV2AllvonePlugin']['analysis'] for key in results_by_eccentricity.keys()])
        generate_box_graph(X, Y,
                            results_by_eccentricity.keys(), subject_labels, 'Accuracy Error (degrees)', '400x400 FULL analysis Accuracy Errors by Eccentricity (LOWER IS BETTER)',
                            '400x400 FULL Analysis Accuracy Errors by Eccentricity.png',
                            ylimit=None, xlabel='Eccentricity (degrees)', Z=Z, override_plotsize=True)
    except KeyError:
        pass

    plt.clf()

    try:
        X = [results_by_eccentricity[key][400][1]['vanilla']['analysis_precision'] for key in results_by_eccentricity.keys()]
        Y = [results_by_eccentricity[key][400][1]['Detector2DRITnetEllsegV2AllvonePlugin']['analysis_precision'] for key in results_by_eccentricity.keys()]
        Z = []
        for subj_num in subjects:
            curr = [results_by_eccentricity[key][400][subj_num]['vanilla']['analysis_precision'] for key in results_by_eccentricity.keys()]
            Z.append(curr)
            Z.append([results_by_eccentricity[key][400][subj_num]['Detector2DRITnetEllsegV2AllvonePlugin']['analysis_precision'] for key in results_by_eccentricity.keys()])
        generate_box_graph(X, Y,
                            results_by_eccentricity.keys(), subject_labels, 'Precision Error (degrees)', '400x400 FULL analysis Precision Errors by Eccentricity (LOWER IS BETTER)',
                            '400x400 FULL Analysis Precision Errors by Eccentricity.png',
                            ylimit=None, xlabel='Eccentricity (degrees)', Z=Z, override_plotsize=True)
    except KeyError:
        pass

    plt.clf()

    try:
        X = [results_by_eccentricity[key][192][1]['vanilla']['analysis'] for key in results_by_eccentricity.keys()]
        Y = [results_by_eccentricity[key][192][1]['Detector2DRITnetEllsegV2AllvonePlugin']['analysis'] for key in results_by_eccentricity.keys()]
        Z = []
        for subj_num in subjects:
            curr = [results_by_eccentricity[key][192][subj_num]['vanilla']['analysis'] for key in results_by_eccentricity.keys()]
            Z.append(curr)
            Z.append([results_by_eccentricity[key][192][subj_num]['Detector2DRITnetEllsegV2AllvonePlugin']['analysis'] for key in results_by_eccentricity.keys()])
        generate_box_graph(X, Y,
                            results_by_eccentricity.keys(), subject_labels, 'Accuracy Error (degrees)', '192x192 FULL analysis Accuracy Errors by Eccentricity (LOWER IS BETTER)',
                            '192x192 FULL Analysis Accuracy Errors by Eccentricity.png',
                            ylimit=None, xlabel='Eccentricity (degrees)', Z=Z, override_plotsize=True)
    except KeyError:
        pass

    plt.clf()

    try:
        X = [results_by_eccentricity[key][192][1]['vanilla']['analysis_precision'] for key in results_by_eccentricity.keys()]
        Y = [results_by_eccentricity[key][192][1]['Detector2DRITnetEllsegV2AllvonePlugin']['analysis_precision'] for key in results_by_eccentricity.keys()]
        Z = []
        for subj_num in subjects:
            try:
                Z.append([results_by_eccentricity[key][192][subj_num]['vanilla']['analysis_precision'] for key in results_by_eccentricity.keys()])
            except KeyError:
                pass
            try:
                Z.append([results_by_eccentricity[key][192][subj_num]['Detector2DRITnetEllsegV2AllvonePlugin']['analysis_precision'] for key in results_by_eccentricity.keys()])
            except KeyError:
                pass
        #generate_box_graph(X, Y,
        #                    results_by_eccentricity.keys(), ('Sub1Vanilla', 'Sub1EllSeg', 'Sub2Vanilla', 'Sub2EllSeg', 'Sub3Vanilla', 'Sub3EllSeg',
        #                        'Sub5Vanilla', 'Sub5EllSeg', 'Sub6Vanilla', 'Sub6EllSeg',), 'Precision Error (degrees)', '192x192 FULL analysis Precision Errors by Eccentricity (LOWER IS BETTER)',
        #                    '192x192 FULL Analysis Precision Errors by Eccentricity.png',
        #                    ylimit=None, xlabel='Eccentricity (degrees)', Z=Z, override_plotsize=True)
        generate_box_graph(X, Y,
                            results_by_eccentricity.keys(), subject_labels, 'Precision Error (degrees)', '192x192 FULL analysis Precision Errors by Eccentricity (LOWER IS BETTER)',
                            '192x192 FULL Analysis Precision Errors by Eccentricity.png',
                            ylimit=None, xlabel='Eccentricity (degrees)', Z=Z, override_plotsize=True)
    except KeyError:
        pass

    plt.clf()

    # Graph set 2: Distributions of all subjects' accuracy and precision error separated into categories by degree
    methods = ('vanilla', 'Detector2DRITnetEllsegV2AllvonePlugin')
    all_accuracies = {}
    all_precisions = {}
    for method in methods:
        all_accuracies[method] = {}
        all_precisions[method] = {}
    for ecc in results_by_eccentricity.keys():
        for res in results_by_eccentricity[ecc].keys():
            for subj in results_by_eccentricity[ecc][res].keys():
                if subj == 4:
                    continue
                for method in methods:
                    if res not in all_accuracies[method]:
                        all_accuracies[method][res] = results_by_eccentricity[ecc][res][subj][method]['analysis']
                    else:
                        all_accuracies[method][res] = np.concatenate((all_accuracies[method][res], results_by_eccentricity[ecc][res][subj][method]['analysis']))
                    
                    if res not in all_precisions[method]:
                        all_precisions[method][res] = results_by_eccentricity[ecc][res][subj][method]['analysis']
                    else:
                        all_precisions[method][res] = np.concatenate((all_precisions[method][res], results_by_eccentricity[ecc][res][subj][method]['analysis_precision']))

    import scipy.stats as stats

    def get_max(data1, data2):
        max1 = np.max(data1)
        max2 = np.max(data2)
        if max1 > max2:
            return int(np.ceil(max1))
        else:
            return int(np.ceil(max2))

    def plot_hist(data, filename, type, max=None):
        if max is None:
            bins = [x/4 for x in range(0, 4*int(np.ceil(np.max(data))))]
        else:
            bins = [x/4 for x in range(0, 4*max)]
        plt.hist(data, bins, density=True)
        plt.xlabel(type)
        plt.ylabel("Distribution")
        savefig(filename, bbox_inches='tight')

    def plot_overlapped_hist(data1, data2, labels, filename, type, title, binmult=4):
        DIST = stats.beta
        max = get_max(data1, data2)
        bins = [x/binmult for x in range(0, binmult*max)]

        fit_alpha1, fit_beta1, fit_loc1, fit_scale1 = DIST.fit(data1)
        x = np.linspace(0, max, max*binmult)
        y = DIST.pdf(x, a=fit_alpha1, b=fit_beta1, loc=fit_loc1, scale=fit_scale1)
        plt.plot(x, y, "-", linewidth=3, c="red")
        plt.hist(data1, bins, alpha=0.2, density=True, color="red", label=labels[0])

        fit_alpha2, fit_beta2, fit_loc2, fit_scale2 = DIST.fit(data2)
        x = np.linspace(0, max, max*binmult)
        y = DIST.pdf(x, a=fit_alpha2, b=fit_beta2, loc=fit_loc2, scale=fit_scale2)
        plt.plot(x, y, "-", linewidth=3, c="blue")
        plt.hist(data2, bins, alpha=0.2, density=True, color="blue", label=labels[1])
        plt.xlabel(type)
        plt.ylabel("Distribution")
        plt.legend(loc='upper right')
        plt.title(title)
        savefig(filename, bbox_inches='tight')

    def plot_overlapped_hist_invgauss(data1, data2, labels, filename, type, title, binmult=4):
        DIST = stats.invgauss
        max = get_max(data1, data2)
        bins = [x/binmult for x in range(0, binmult*max)]

        fit_mu1, fit_loc1, fit_scale1 = DIST.fit(data1)
        x = np.linspace(0, max, max*binmult)
        y = DIST.pdf(x, mu=fit_mu1, loc=fit_loc1, scale=fit_scale1)
        plt.plot(x, y, "-", linewidth=3, c="red")
        plt.hist(data1, bins, alpha=0.2, density=True, color="red", label=labels[0])

        fit_mu2, fit_loc2, fit_scale2 = DIST.fit(data2)
        x = np.linspace(0, max, max*binmult)
        y = DIST.pdf(x, mu=fit_mu2, loc=fit_loc2, scale=fit_scale2)
        plt.plot(x, y, "-", linewidth=3, c="blue")
        plt.hist(data2, bins, alpha=0.2, density=True, color="blue", label=labels[1])
        plt.xlabel(type)
        plt.ylabel("Distribution")
        plt.legend(loc='upper right')
        plt.title(title)
        savefig(filename, bbox_inches='tight')

    max = get_max(all_accuracies['vanilla'][192], all_accuracies['Detector2DRITnetEllsegV2AllvonePlugin'][192])
    plot_hist(all_accuracies['vanilla'][192], 'HIST_all_accuracies_vanilla_192.png', "Accuracy Error", max)
    plt.clf()
    plot_hist(all_accuracies['Detector2DRITnetEllsegV2AllvonePlugin'][192], 'HIST_all_accuracies_ellseggen_192.png', "Accuracy Error", max)
    plt.clf()

    max = get_max(all_precisions['vanilla'][192], all_precisions['Detector2DRITnetEllsegV2AllvonePlugin'][192])
    plot_hist(all_precisions['vanilla'][192], 'HIST_all_precisions_vanilla_192.png', "Precision Error", max)
    plt.clf()
    plot_hist(all_precisions['Detector2DRITnetEllsegV2AllvonePlugin'][192], 'HIST_all_precisions_ellseggen_192.png', "Precision Error", max)
    plt.clf()

    max = get_max(all_accuracies['vanilla'][400], all_accuracies['Detector2DRITnetEllsegV2AllvonePlugin'][400])
    plot_hist(all_accuracies['vanilla'][400], 'HIST_all_accuracies_vanilla_400.png', "Accuracy Error", max)
    plt.clf()
    plot_hist(all_accuracies['Detector2DRITnetEllsegV2AllvonePlugin'][400], 'HIST_all_accuracies_ellseggen_400.png', "Accuracy Error", max)
    plt.clf()

    max = get_max(all_precisions['vanilla'][400], all_precisions['Detector2DRITnetEllsegV2AllvonePlugin'][400])
    plot_hist(all_precisions['vanilla'][400], 'HIST_all_precisions_vanilla_400.png', "Precision Error", max)
    plt.clf()
    plot_hist(all_precisions['Detector2DRITnetEllsegV2AllvonePlugin'][400], 'HIST_all_precisions_ellseggen_400.png', "Precision Error", max)
    plt.clf()

    plot_overlapped_hist_invgauss(all_accuracies['vanilla'][192], all_accuracies['Detector2DRITnetEllsegV2AllvonePlugin'][192],
        ('Native', 'EllSegGen'), 'HIST_all_accuracies_192', "Accuracy Error (Degrees)", "Accuracy Error Distributions at 192x192")
    plt.clf()
    
    plot_overlapped_hist_invgauss(all_accuracies['vanilla'][400], all_accuracies['Detector2DRITnetEllsegV2AllvonePlugin'][400],
        ('Native', 'EllSegGen'), 'HIST_all_accuracies_400', "Accuracy Error (Degrees)", "Accuracy Error Distributions at 400x400")
    plt.clf()

    plot_overlapped_hist_invgauss(all_precisions['vanilla'][192], all_precisions['Detector2DRITnetEllsegV2AllvonePlugin'][192],
        ('Native', 'EllSegGen'), 'HIST_all_precisions_192', "Precision Error (Degrees)", "Precision Error Distributions at 192x192")
    plt.clf()
    
    plot_overlapped_hist_invgauss(all_precisions['vanilla'][400], all_precisions['Detector2DRITnetEllsegV2AllvonePlugin'][400],
        ('Native', 'EllSegGen'), 'HIST_all_precisions_400', "Precision Error (Degrees)", "Precision Error Distributions at 400x400")
    plt.clf()

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

def detect_non_saccads(gazeDataFolder, specificExport, label, session, color, ax):
    import pandas as pd
    from pylab import savefig
    from scipy.fft import fft

    subID = session['subID']
    calibration_timestamps = session['processedSequence']['pupilTimestamp'].to_numpy()
    calibration_worldindices = session['processedSequence']['world_index'].to_numpy()
    assessment_timestamps = session['processedCalib']['pupilTimestamp'].to_numpy()

    DPS_SCALE = 50.0

    def calcSlopes(row, df):
        currRowIdx = row.name
        prevRowIdx = currRowIdx - 1
        if prevRowIdx >= 0:
            az_1 = row['az']
            el_1 = row['el']
            prev_row = df.loc[prevRowIdx]
            az_2 = prev_row['az']
            el_2 = prev_row['el']
            
            az_2 = az_2 - az_1
            el_2 = el_2 - el_1
            
            ts_1 = row['pupilTimestamp']
            ts_2 = prev_row['pupilTimestamp']

            if ts_1 < ts_2:
                #print("TS1 < TS2 at ", currRowIdx, " ", prevRowIdx)
                return 0.0
            elif ts_1 != ts_2:
                res = (np.sqrt(np.square(az_2) + np.square(el_2))) / ((ts_1 - ts_2))
                return res
            else:
                #print("Identical timestamps encountered at ", currRowIdx, " ", prevRowIdx)
                return 0.0
        else:
            return 0.0
    
    def addSlope(df):
        slopes = df.apply(lambda i: calcSlopes(i, df), axis=1)
        df['velocity'] = slopes
    
    gazePositionsDF = pd.read_csv( gazeDataFolder + specificExport + '/gaze_positions.csv' )
    gazePositionsDF = gazePositionsDF.rename(columns={"gaze_timestamp": "pupilTimestamp"})
    gazePositionsDF.sort_values(by='pupilTimestamp',inplace=True, ignore_index=True)

    gazePositionsDF[('deprojected_norm_pos0','az')] = np.rad2deg(np.arctan2(gazePositionsDF['deprojected_norm_pos0_x'],gazePositionsDF[('deprojected_norm_pos0_z')]))
    gazePositionsDF[('deprojected_norm_pos0','el')] = np.rad2deg(np.arctan2(gazePositionsDF[('deprojected_norm_pos0_y')],gazePositionsDF[('deprojected_norm_pos0_z')]))

    gazePositionsDF[('deprojected_norm_pos1','az')] = np.rad2deg(np.arctan2(gazePositionsDF['deprojected_norm_pos1_x'],gazePositionsDF[('deprojected_norm_pos1_z')]))
    gazePositionsDF[('deprojected_norm_pos1','el')] = np.rad2deg(np.arctan2(gazePositionsDF[('deprojected_norm_pos1_y')],gazePositionsDF[('deprojected_norm_pos1_z')]))

    gazePositionsDF['az'] = np.rad2deg(np.arctan2(gazePositionsDF['deprojected_norm_pos_x'],gazePositionsDF[('deprojected_norm_pos_z')]))
    gazePositionsDF['el'] = np.rad2deg(np.arctan2(gazePositionsDF[('deprojected_norm_pos_y')],gazePositionsDF[('deprojected_norm_pos_z')]))

    addSlope(gazePositionsDF)
    #print("az: ", gazePositionsDF['az'].min(), " to ", gazePositionsDF['az'].max())
    #print("el: ", gazePositionsDF['el'].min(), " to ", gazePositionsDF['el'].max())
    #print("velocity: ", gazePositionsDF['velocity'].min(), " to ", gazePositionsDF['velocity'].max())
    #exit()

    plot_az = gazePositionsDF['az']
    plot_el = gazePositionsDF['el']
    fig = plt.figure(figsize=(12, 12), dpi=1600, facecolor='w', edgecolor='k')
    plt.xlim([-40, 40])
    plt.ylim([-40, 40])
    plt.scatter(plot_az, plot_el, s=0.5)
    #fig.savefig('./out_disp{}_dur{}_{}.png'.format(DISPERSION_THRESHOLD, DURATION_THRESHOLD, label))
    fig.savefig('./figOut/Velocity/out_{}_{}.png'.format(subID, label))
    plt.clf()
    plt.close(fig)
    return

    plTime_fr = np.array(gazePositionsDF['pupilTimestamp'] - gazePositionsDF['pupilTimestamp'].iloc[0])
    yLim = [-50, 50]
    width = 800
    height = 600

    import plotly.graph_objs as go
    colors_idx = ['rgb(0,204,204)','rgb(128,128,128)','rgb(204,0,0)','rgb(102,0,204)']
    traces = []
    
    eih_el = go.Scattergl(
        x=plTime_fr,
        y=gazePositionsDF['el'],
        name='el',
        marker_color=colors_idx[0],
        mode='markers',
        marker_size=5,
        opacity=0.8
    )
    traces.append(eih_el)
    
    eih_az = go.Scattergl(
        x=plTime_fr,
        y=gazePositionsDF['az'],
        name='az',
        marker_color=colors_idx[1],
        mode='markers',
        marker_size=5,
        opacity=0.8
    )
    traces.append(eih_az)

    eih_slp = go.Scattergl(
        x=plTime_fr,
        y=gazePositionsDF['velocity'],
        name='velocity',
        marker_color=colors_idx[2],
        mode='markers',
        marker_size=5,
        opacity=0.8
    )
    traces.append(eih_slp)
    
    #layout = dict(
    #    dragmode='pan',
    #    title='Azimuth, Elevation, & Slope',
    #    width=width,
    #    height=height,
    #    yaxis=dict(range=yLim, title='angular position (degrees)'),
    #    xaxis=dict(
    #        rangeslider=dict(visible=True),
    #        range=[0, 500]
    #        title='time elapsed (seconds)'
    #    )
    #)

    layout = go.Layout(
        title_text = 'Azimuth, Elevation, & Slope',
        dragmode='pan',
        width=width,
        height=height,
        yaxis=dict(range=yLim, title='angular position (degrees)'),
        xaxis=dict(
            rangeslider=dict(visible=True),
            range=[0, plTime_fr.max()],
            title='time elapsed (seconds)'
        )
    )
    fig = go.Figure(data=traces, layout=layout)
    fig.write_html('./figOut/HTML/out_{}_{}.html'.format(subID, label))

    DISPERSION_THRESHOLD = 1.5  # I-DT dispersion threshold (degrees)
    DURATION_THRESHOLD = 0.2  # I-DT duration threshold (seconds)
    #CONFIDENCE_THRESHOLD = 0.95  # Unused, the filter magnitude filters the low-confidence data out well enough for now.
    FILTER_MAGNITUDE = 30.0  # I-VT Velocity threshold in degrees-per-second
    
    def I_VT(gaze_data, saccade_velocity):
        return (gaze_data[gaze_data['velocity']<saccade_velocity], [], [])
    
    def I_DT(gaze_data, disp_thresh, dur_thresh):
        def dispersion(start_pt, end_pt):
            window = gaze_data.iloc[start_pt:end_pt]
            window_az = window['az']
            window_el = window['el']
            return (window_az.max() - window_az.min()) +\
                (window_el.max() - window_el.min())
        
        # Initialize the window over the first points to cover duration threshold
        start_point = 1
        end_point = 1
        while(gaze_data.iloc[end_point]['pupilTimestamp']
            - gaze_data.iloc[start_point]['pupilTimestamp']) < dur_thresh:
            end_point += 1
        
        fixation_ranges = []
        fixation_lengths = []
        fixation_starts = []
        fixation_means_az = []
        fixation_means_el = []
        
        # While there are still points
        while end_point < len(gaze_data):
            if dispersion(start_point, end_point) < disp_thresh:
                while end_point < len(gaze_data) and\
                    dispersion(start_point, end_point) < disp_thresh:
                    end_point += 1
                if end_point != len(gaze_data)-1:
                    end_point -= 1
                fixation_length_sum = 0
                for num in list(range(start_point, end_point+1)):
                    fixation_ranges.append(num)
                fixation_lengths.append(gaze_data.iloc[end_point+1]['pupilTimestamp'] - gaze_data.iloc[start_point]['pupilTimestamp'])
                fixation_starts.append(gaze_data.iloc[start_point]['pupilTimestamp'] - gaze_data.iloc[0]['pupilTimestamp'])
                fixation_means_az.append(np.sum(np.multiply(
                    gaze_data.iloc[start_point:end_point+1]['az'].to_numpy(),
                    list(range(1, end_point+2 - start_point))
                    )) / np.sum(list(range(1, end_point+2 - start_point)))
                )
                fixation_means_el.append(np.sum(np.multiply(
                    gaze_data.iloc[start_point:end_point+1]['el'].to_numpy(),
                    list(range(1, end_point+2 - start_point))
                    )) / np.sum(list(range(1, end_point+2 - start_point)))
                )
                start_point = end_point + 1
                end_point = end_point + 1
                while end_point < len(gaze_data) and (gaze_data.iloc[end_point]['pupilTimestamp']
                    - gaze_data.iloc[start_point]['pupilTimestamp']) < dur_thresh:
                    end_point += 1
            else:
                start_point += 1
                end_point += 1
        print("Mean fixation length:", np.mean(fixation_lengths))
        for f_s in fixation_starts:
            print("Fixation: ", int(f_s/60), "min", f_s%60, "sec.")
        return (gaze_data.iloc[fixation_ranges], fixation_means_az, fixation_means_el)

    # CALIBRATE I_VT FILTER MAGNITUDE
    all_calib_gaze_positions = gazePositionsDF[gazePositionsDF['world_index'].isin(calibration_worldindices)]
    print("mean: ", np.mean(all_calib_gaze_positions['velocity'].to_numpy()))
    print("median: ", np.median(all_calib_gaze_positions['velocity'].to_numpy()))
    def reject_outliers(data):
        m = 2
        u = np.mean(data)
        s = np.std(data)
        filtered = [e for e in data if (u - 2 * s < e < u + 2 * s)]
        return filtered
    newthing = reject_outliers(all_calib_gaze_positions['velocity'].to_numpy())
    print("new mean: ", np.mean(newthing))
    print("new median: ", np.median(newthing))
    CALIBRATED_FILTER_MAGNITUDE = np.mean(newthing) + np.std(newthing)
    print("adjusted threshold: ", CALIBRATED_FILTER_MAGNITUDE)

    #gaze_data_nonsaccade, fixation_means_az, fixation_means_el = I_DT(gazePositionsDF, DISPERSION_THRESHOLD, DURATION_THRESHOLD)
    gaze_data_nonsaccade, fixation_means_az, fixation_means_el = I_VT(gazePositionsDF, CALIBRATED_FILTER_MAGNITUDE)
    plot_az = gaze_data_nonsaccade['az']
    plot_el = gaze_data_nonsaccade['el']
    fig = plt.figure(figsize=(12, 12), dpi=1600, facecolor='w', edgecolor='k')
    plt.xlim([-40, 40])
    plt.ylim([-40, 40])
    plt.scatter(plot_az, plot_el, s=0.5)
    #fig.savefig('./out_disp{}_dur{}_{}.png'.format(DISPERSION_THRESHOLD, DURATION_THRESHOLD, label))
    fig.savefig('./figOut/Velocity/out_fil_{}_mean{}_{}.png'.format(subID, round(np.mean(newthing), 2), label))

    plt.clf()
    plt.close(fig)

    
    
    #exit()

    #gaze_data_nonsaccade_known_calib = gaze_data_nonsaccade[gaze_data_nonsaccade['pupilTimestamp'].isin(calibration_timestamps)]
    gaze_data_nonsaccade_known_calib = gaze_data_nonsaccade[gaze_data_nonsaccade['world_index'].isin(calibration_worldindices)]
    
    print("gaze_data_nonsaccade_known_calib:", len(gaze_data_nonsaccade_known_calib), "/", len(gazePositionsDF[gazePositionsDF['world_index'].isin(calibration_worldindices)]))
    plot_az = gaze_data_nonsaccade_known_calib['az']
    plot_el = gaze_data_nonsaccade_known_calib['el']
    fig = plt.figure(figsize=(12, 12), dpi=1600, facecolor='w', edgecolor='k')
    plt.xlim([-40, 40])
    plt.ylim([-40, 40])
    plt.scatter(plot_az, plot_el, s=0.5)
    #fig.savefig('./out_disp{}_dur{}_{}.png'.format(DISPERSION_THRESHOLD, DURATION_THRESHOLD, label))
    fig.savefig('./figOut/Velocity/out_fil_{}_{}_CalibOverlap.png'.format(subID, label))
    
    plt.clf()
    plt.close(fig)
    
    gaze_data_nonsaccade_known_assesment = gaze_data_nonsaccade[gaze_data_nonsaccade['pupilTimestamp'].isin(assessment_timestamps)]
    print("gaze_data_nonsaccade_known_assesment:", len(gaze_data_nonsaccade_known_assesment), "/", len(gazePositionsDF[gazePositionsDF['pupilTimestamp'].isin(assessment_timestamps)]))
    plot_az = gaze_data_nonsaccade_known_assesment['az']
    plot_el = gaze_data_nonsaccade_known_assesment['el']
    fig = plt.figure(figsize=(12, 12), dpi=1600, facecolor='w', edgecolor='k')
    plt.xlim([-40, 40])
    plt.ylim([-40, 40])
    plt.scatter(plot_az, plot_el, s=0.5)
    #fig.savefig('./out_disp{}_dur{}_{}.png'.format(DISPERSION_THRESHOLD, DURATION_THRESHOLD, label))
    fig.savefig('./figOut/Velocity/out_fil_{}_{}_AssessOverlap.png'.format(subID, label))
    plt.close(fig)
    
    #exit()

    #fig = dict(data=traces, layout=layout)
    #iplot(fig)

    #gazePositionsDF['gaze_normal2'] = gazePositionsDF['az'].pow(2).add(gazePositionsDF['el'].pow(2)).pow(1/2)
    #gazePositionsDF['velo_normal2'] = gazePositionsDF['gaze_normal2'].diff()
    #gazePositionsDF['veloFFT_normal2'] = fft(gazePositionsDF['velo_normal2'].fillna(0).values)

    #gazePositionsDFNOTNAN = gazePositionsDF[~gazePositionsDF['velo_normal2'].isna()]
    #gazePositionsDFNOTNAN = gazePositionsDFNOTNAN.loc[((gazePositionsDFNOTNAN['pupilTimestamp'] > 12120) & (gazePositionsDFNOTNAN['pupilTimestamp'] <= 12130))]

    #gazePositionsDFNOTNAN.plot(x='pupilTimestamp', y='velo_normal2', figsize=(32,14), label=label, color=color, ax=ax, alpha=0.5)
    #gazePositionsDFNOTNAN.plot(x='pupilTimestamp', y='veloFFT_normal2', figsize=(32,14), label='FFT'+label, ax=ax, alpha=0.5)

def generate_box_graph(X, Y, labels, barlabels, ylabel, title, filename, ylimit=None, xlabel='Subject Number', Z=[], override_plotsize=False, group_size=2):

    from pylab import plot, show, savefig, xlim, figure, \
                ylim, legend, boxplot, setp, axes
    from matplotlib.cbook import boxplot_stats
    color_list = ['blue', 'aqua', 'red', 'lightcoral', 'green', 'lime', 'black', 'darkgrey', 'gold', 'yellow', 'purple', 'violet', 'hotpink', 'pink', 'mediumaquamarine', 'aquamarine',
        'sandybrown', 'peachpuff', 'tan', 'navajowhite', 'olive', 'yellowgreen']
    # function for setting the colors of the box plots pairs
    def setBoxColors(bp, color_count=2):
        for i in range(color_count):
            setp(bp['boxes'][i], color=color_list[i])
            #setp(bp['boxes'][1], color='red')
            
            setp(bp['caps'][i*2], color=color_list[i])
            setp(bp['caps'][i*2+1], color=color_list[i])
            #setp(bp['caps'][2], color='red')
            #setp(bp['caps'][3], color='red')
            
            setp(bp['whiskers'][i*2], color=color_list[i])
            setp(bp['whiskers'][i*2+1], color=color_list[i])
            #setp(bp['whiskers'][2], color='red')
            #setp(bp['whiskers'][3], color='red')
            try:
                setp(bp['fliers'][i], color='black')
                #setp(bp['fliers'][1], color='black')
            except:
                pass
            setp(bp['medians'][i], color=color_list[i])
           # setp(bp['medians'][1], color='red')

    # Some fake data to plot
    A= [[1, 2, 5,],  [7, 2]]
    B = [[5, 7, 2, 2, 5], [7, 2, 5]]
    C = [[3,2,5,7], [6, 7, 3]]

    if override_plotsize:
        fig = figure(figsize=(20,15))
    else:
        fig = figure()
    ax = axes()
    #hold(True)
    
    
    num_boxes = 2 + len(Z)
    Xs = [[] for i in range(num_boxes)]
    boxplots = []
    mass_medians = [[] for i in range(num_boxes)]
    
    for i in range(0, len(X)):
        # boxplot pair
        bp = ax.boxplot([X[i][~np.isnan(X[i])], Y[i][~np.isnan(Y[i])], *[z[i][~np.isnan(z[i])] for z in Z]], positions = [i*(num_boxes+1)+I for I in range(num_boxes)], widths = 0.6, showfliers=False)
        boxplots.append(bp)
        setBoxColors(bp, color_count=num_boxes)
        
        mass_medians[0].append(np.median(X[i][~np.isnan(X[i])]))
        mass_medians[1].append(np.median(Y[i][~np.isnan(Y[i])]))
        for zed in range(len(Z)):
            z = Z[zed]
            mass_medians[zed+2].append(np.median(z[i][~np.isnan(z[i])]))
        
        Xs[0].append((i*num_boxes)+0+i)
        Xs[1].append((i*num_boxes)+1+i)
        for zed in range(len(Z)):
            Xs[zed+2].append((i*num_boxes)+zed+2+i)
    
    plots = []
    for i in range(len(mass_medians)):
        for j in range(len(Xs[0])):
            h, = plt.plot([Xs[i - (i % group_size)][j], Xs[i - (i % group_size) + group_size - 1][j]], [mass_medians[i][j], mass_medians[i][j]], '-', c=color_list[i])
            if (i % group_size) != 0:
                text_x = Xs[i][j]-0.35
                text_y = boxplots[j]['caps'][i*2+1].get_data()[1][0]
                text_label = np.around(mass_medians[i][j] - mass_medians[i - (i % group_size)][j], 2)
                ax.text(text_x, text_y, text_label, color='darkgreen' if (mass_medians[i][j] - mass_medians[i - (i % group_size)][j])<=0 else 'firebrick')
        
        #h, = plt.plot(Xs[i], mass_medians[i], '-', c=color_list[i])
        plots.append(h)


    # second boxplot pair
    #bp = boxplot(B, positions = [4, 5], widths = 0.6)
    #setBoxColors(bp)

    # thrid boxplot pair
    #bp = boxplot(C, positions = [7, 8], widths = 0.6)
    #setBoxColors(bp)

    # set axes limits and labels
    #xlim(0,9)
    #ylim(0,9)
    if ylimit:
        ylim(0, ylimit)
    ax.set_xticks(np.arange(len(labels))*(num_boxes+1) + int(np.floor((num_boxes+1)/2)))
    ax.set_xticklabels(labels)
    
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    
    # draw temporary red and blue lines and use them to create a legend
    #legend_lines = []
    #for i in range(len(mass_medians)):
    #    legend_lines.append(ax.plot([1,1], '-', c=color_list[i]))
    #hB, = ax.plot([1,1],'b-')
    #hR, = ax.plot([1,1],'r-')
    
    #legend((hB, hR),(barlabels[0], barlabels[1]))
    #hB.set_visible(False)
    #hR.set_visible(False)
    legend(plots, barlabels)
    if override_plotsize:
        savefig(filename, bbox_inches='tight')
    else:
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
