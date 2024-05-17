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
import seaborn as sns
import scipy

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
    "--not_uxf",
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
    "--skip_eye_tracking",
    is_flag=True
)
@click.option(
    "--load_pandas_checkpoint",
    is_flag=True
)
@click.option(
    "--skip_trial_assessment",
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
@click.option(
    "--figout_loc",
    required=False,
    type=click.Path(exists=False),
    default="./figOut/"
)
@click.option(
    "--allsessiondata_loader",
    required=False,
    type=click.Path(exists=False),
    default="allSessionData.pickle"
)
def main(allow_session_loading, skip_pupil_detection, vanilla_only, skip_vanilla, surpress_runtimewarnings,
        not_uxf, load_2d_pupils, min_calibration_confidence, show_filtered_out, display_world_video, skip_eye_tracking,
        load_pandas_checkpoint, skip_trial_assessment, velocity_graphs, core_shared_modules_loc, pipeline_loc,
        plugins_file, figout_loc, allsessiondata_loader):
    
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('matplotlib.axes').disabled = True
    plt.set_loglevel(level = 'warning')
    
    if not os.path.exists(figout_loc):
        os.makedirs(figout_loc)
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
    
    if (not skip_trial_assessment) or (not skip_eye_tracking):
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
    
    if not skip_eye_tracking:
        logging.debug(f"Loaded pupil detector plugins: {plugins}")

        total_items = len([item for item in os.listdir("Data/") if item[0:9] == '_Pipeline'])
        if total_items == 0:
            logging.error("No valid data folders found. Did you make sure to start the data folder names with '_Pipeline'?")
            return

        i = 0
        for name in [item for item in os.listdir("Data/") if item[0:9] == '_Pipeline']:
            subject_loc = os.path.join("Data/", name)
            rec_loc = os.path.join(subject_loc, '' if not_uxf else 'S001/PupilData/000')
            logging.info(f'Proccessing {rec_loc} through pipeline...')
            
            reference_data_loc = rec_loc+'/offline_data/reference_locations.msgpack'
            total_plugins = len(plugins)
            if not skip_pupil_detection:
                logging.info("Performing pupil detection on eye videos. This may take a while.")
                j = 0
                for plugin in plugins:
                    j += 1
                    logging.info(f"---------------[{i*total_plugins + j}/{total_items*total_plugins}]----------------")
                    logging.info(f"Current plugin: {plugin}")
                    # Checking to see if we can freeze the model at the first calibration point
                    realtime_calib_points_loc = rec_loc+'/realtime_calib_points.msgpack';
                    
                    # --__--__--__--WE ARE FREEZING THE 3D EYE MODELS--__--__--__--
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

                    resolution = int(name[14:17])
                    if resolution == 192:
                        eye_detector_params = [{
                            # eye 0 params
                            "intensity_range": 23,
                            "pupil_size_min": 10,
                            "pupil_size_max": 100
                        }, {
                            # eye 1 params
                            "intensity_range": 23,
                            "pupil_size_min": 10,
                            "pupil_size_max": 100
                        }]
                    elif resolution == 400:
                        eye_detector_params = [{
                            # eye 0 params
                            "intensity_range": 10,
                            "pupil_size_min": 10,
                            "pupil_size_max": 100
                        }, {
                            # eye 1 params
                            "intensity_range": 10,
                            "pupil_size_min": 10,
                            "pupil_size_max": 100
                        }]
                    else:
                        logging.error('RESOLUTION {} NOT SUPPORTED!'.format(resolution))
                        exit()
                    
                    perform_pupil_detection(rec_loc, plugin=plugin, pupil_params=eye_detector_params,
                                                world_file="world.mp4",
                                                load_2d_pupils=load_2d_pupils,
                                                start_model_timestamp=start_model_timestamp,
                                                freeze_model_timestamp=freeze_model_timestamp,
                                                display_world_video=display_world_video,
                                                mapping_method=mapping_method,
                                                skip_3d_detection=skip_3d_detection
                                            )
                logging.info("Pupil detection complete.")
            
            j = 0
            for plugin in plugins:
                j += 1
                logging.info(f"---------------[{i*total_plugins + j}/{total_items*total_plugins}]----------------")
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
                    except TypeError:
                        print("No calibration points found.")
                        continue
                else:
                    print("Realtime calibration points have not been exported.")
                    try:
                        calibrated_gazer, pupil_data = calibrate_and_validate(reference_data_loc, pupil_data_loc, intrinsics_loc, mapping_method)#, min_calibration_confidence=min_calibration_confidence)
                    except FileNotFoundError:
                        print("No calibration points found.")
                        continue
                    except TypeError:
                        print("No calibration points found.")
                        continue

                gaze, gaze_ts = map_pupil_data(calibrated_gazer, pupil_data, rec_loc)
                save_gaze_data(gaze, gaze_ts, rec_loc, plugin=plugin)
            i += 1
        
        logging.info('All gaze data obtained.')
        if not_uxf:
            return
    elif not_uxf:
        logging.error('Only UXF gaze files can have fixation data graphed.')
        return
    
    if not skip_trial_assessment:
        logging.info('Generating trial charts.')
        targets = []
        if vanilla_only:
            targets.append('vanilla')
            targets.append('realtime')  # Realtime is technically vanilla
            targets.append('vanilla_player')  # Allow vanilla data exported from PL Player
        
        allSessionData = processAllData(doNotLoad=not allow_session_loading, confidenceThresh=0.00, targets=targets, show_filtered_out=show_filtered_out, load_realtime_ref_data=load_realtime_ref_data, override_to_2d=skip_3d_detection)

        with open(allsessiondata_loader, 'wb') as handle:
            pickle.dump(allSessionData, handle, protocol=pickle.HIGHEST_PROTOCOL)

    from pylab import savefig

    if not load_pandas_checkpoint:
        filehandler = open(allsessiondata_loader, "rb")
        allSessionData = pickle.load(filehandler)
        print(f"LOADED {allsessiondata_loader}")

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
            """
            OLD METHOD (every point is added)
            calibrationEuclideanFixErrors = sessionDict['processedSequence'][('fixError_eye2', 'euclidean')].to_numpy()
            analysisEuclideanFixErrors = sessionDict['processedCalib'][('fixError_eye2', 'euclidean')].to_numpy()
            """
            pupil_0_X = np.array([])
            pupil_0_Y = np.array([])
            pupil_0_ts = np.array([])
            calibrationEuclideanFixErrors = []
            targetLoc_targNum_AzEl = sessionDict['processedSequence']['targetLocalSpherical'].drop_duplicates().values
            for tNum,(tX,tY) in enumerate(targetLoc_targNum_AzEl):
                gbFixTrials = sessionDict['processedSequence'].groupby([('targetLocalSpherical','az'), ('targetLocalSpherical','el')])
                trialsInGroup = gbFixTrials.get_group((tX,tY))
                gbTrials = sessionDict['processedSequence'].groupby('trialNumber')
                fixRowDataDf = gbTrials.get_group(trialsInGroup['trialNumber'].values[0])
                for x in trialsInGroup['trialNumber'][1:]:
                    fixRowDataDf = pd.concat([fixRowDataDf,gbTrials.get_group(x)])
                err_acc = np.nanmean(
                    fixRowDataDf['fixError_eye2']['euclidean'].to_numpy()
                )
                if (tX,tY) == (0.0, 0.0):
                    pupil_0_X = np.append(pupil_0_X, fixRowDataDf[fixRowDataDf[('pupil-centroid0', 'x')] > 0.0][('pupil-centroid0', 'x')].to_numpy())
                    pupil_0_Y = np.append(pupil_0_Y, fixRowDataDf[fixRowDataDf[('pupil-centroid0', 'x')] > 0.0][('pupil-centroid0', 'y')].to_numpy())
                    pupil_0_ts = np.append(pupil_0_ts, fixRowDataDf[fixRowDataDf[('pupil-centroid0', 'x')] > 0.0][('pupilTimestamp', '')].to_numpy())
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

                    fixRowDataDFs = []
                    for x in trialsInGroup['trialNumber']:
                        fixRowDataDFs.append((x, gbTrials.get_group(x)))
                    
                    eccentricity = np.round(np.sqrt(tX**2 + tY**2))
                    if eccentricity not in eccentricities and not np.isnan(eccentricity):
                        eccentricities.append(eccentricity)
                    for trialID, DFarr in fixRowDataDFs:
                        if (tX, tY) == (0.0, 0.0):
                            pupil_0_X = np.append(pupil_0_X, DFarr[DFarr[('pupil-centroid0', 'x')] > 0.0][('pupil-centroid0', 'x')].to_numpy())
                            pupil_0_Y = np.append(pupil_0_Y, DFarr[DFarr[('pupil-centroid0', 'x')] > 0.0][('pupil-centroid0', 'y')].to_numpy())
                            pupil_0_ts = np.append(pupil_0_ts, DFarr[DFarr[('pupil-centroid0', 'x')] > 0.0][('pupilTimestamp', '')].to_numpy())
                        nparr = DFarr['fixError_eye2']['euclidean'].to_numpy()
                        if eccentricity in eccentricitiesAccDict:
                            eccentricitiesAccDict[eccentricity].append((trialID, nparr))#(nparr[np.logical_not(np.isnan(nparr))])
                        else:
                            eccentricitiesAccDict[eccentricity] = [(trialID, nparr)]#[nparr[np.logical_not(np.isnan(nparr))]]
                        analysisEuclideanFixErrors.append(np.nanmean(nparr))

            sessionDict['processedCalib']['eccentricity'] = np.round(np.linalg.norm(sessionDict['processedCalib']['targetLocalSpherical'].values, axis=1))

            # [INTERMISSION] Plot the pupil 0 centroids of the (0.0, 0.0) targets over time
            cppp_dir = f"./{figout_loc}/central_point_pupil_positions/"
            if not os.path.exists(cppp_dir):
                os.makedirs(cppp_dir)

            fig, (ax0, ax1) = plt.subplots(2, 1)
            ax0.plot(pupil_0_ts, pupil_0_X)
            #ax0.set_ylim(bottom=0.58, top=0.63)
            ax1.plot(pupil_0_ts, pupil_0_Y)
            #ax1.set_ylim(bottom=0.44, top=0.46)
            fig.suptitle(f"Pupil Centroids (x, y) During Central Fixations {subject}_{resolution}_{run} ({plugin})")
            ax1.set_xlabel("Timestamp")
            ax0.set_ylabel("Centroid Position (X)")
            ax1.set_ylabel("Centroid Position (Y)")
            plt.savefig(f'{cppp_dir}centroid_time_{subject}_{resolution}_{run}_eye{0}.png')

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
                err_prec = np.mean(
                    np.sqrt(
                        np.square(fixRowDataDf['gaze2Spherical']['az'] - meanGazeAz) +\
                        np.square(fixRowDataDf['gaze2Spherical']['el'] - meanGazeEl)
                    )
                )
                calibration_precision_errors.append(err_prec)
            
            fixDF = sessionDict['fixAssessmentData']
            gb_h_w = fixDF.groupby([('gridSize', 'heightDegs'), ('gridSize', 'widthDegs')])
            analysis_precision_errors = []  # wrong position (now right position?)
            eccentricitiesPrecDict = {}
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
                    
                    fixRowDataDFs = []
                    for x in trialsInGroup['trialNumber']:
                        grp = gbTrials.get_group(x)
                        fixRowDataDFs.append((x, grp, np.nanmean(grp['gaze2Spherical']['az']), np.nanmean(grp['gaze2Spherical']['el'])))
  
                    # NEW WAY: Pass raw gaze precisions to figure gen
                    for trialID, DFarr, avgAz, avgEl in fixRowDataDFs:
                        nparr = np.sqrt(
                            np.square(DFarr['gaze2Spherical']['az'].to_numpy() - avgAz) +\
                            np.square(DFarr['gaze2Spherical']['el'].to_numpy() - avgEl)
                        )
                        eccentricity = np.round(np.sqrt(tX**2 + tY**2))
                        if eccentricity in eccentricitiesPrecDict:
                            eccentricitiesPrecDict[eccentricity].append((trialID, nparr))#(nparr[np.logical_not(np.isnan(nparr))])
                        else:
                            eccentricitiesPrecDict[eccentricity] = [(trialID, nparr)]#[nparr[np.logical_not(np.isnan(nparr))]]
                        analysis_precision_errors.append(np.nanmean(nparr))

            # ------------------------------

            if subject not in results_by_subject:
                results_by_subject[subject] = {sessionDict['plExportFolder']: {'calibration_precision': np.array(calibration_precision_errors), 'analysis_precision': np.array(analysis_precision_errors), 'calibration': np.array([]), 'analysis': np.array([])}}
            elif sessionDict['plExportFolder'] not in results_by_subject[subject]:
                results_by_subject[subject][sessionDict['plExportFolder']] = {'calibration_precision': np.array(calibration_precision_errors), 'analysis_precision': np.array(analysis_precision_errors),  'calibration': np.array([]), 'analysis': np.array([])}
            else:
                results_by_subject[subject][sessionDict['plExportFolder']]['calibration_precision'] = np.append(results_by_subject[subject][sessionDict['plExportFolder']]['calibration_precision'], calibration_precision_errors)
                results_by_subject[subject][sessionDict['plExportFolder']]['analysis_precision'] = np.append(results_by_subject[subject][sessionDict['plExportFolder']]['analysis_precision'], analysis_precision_errors)
            if subject not in results_by_subject:                                                                                                                           #^^^ WRONG? (Only using results by eccentricity anyways)
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
            
            def fix_np_len_bug(np_array):
                return np.array([
                    np.append(np_array[j], [
                        np.nan for _ in range(
                            [
                                np.max([
                                    len(np_array[g]) for g in range(len(np_array))
                                ]) - len(np_array[q]) for q in range(len(np_array))
                            ][j]
                        )
                    ]) for j in range(len(np_array))
                ])

            for eccentricity in eccentricities:
                if eccentricity not in results_by_eccentricity:
                    results_by_eccentricity[eccentricity] = {resolution: {subject: {sessionDict['plExportFolder']: {'calibration_precision': None, 'analysis_precision': np.array([(i, d) for i, d in eccentricitiesPrecDict[eccentricity]], dtype=object), 'calibration': None, 'analysis': np.array([(i, d) for i, d in eccentricitiesAccDict[eccentricity]], dtype=object), 'eye0_rate': len(sessionDict['rawCalibGaze']['gaze_normal0_y']), 'eye1_rate': len(sessionDict['rawCalibGaze']['gaze_normal1_y'])}}}}
                elif resolution not in results_by_eccentricity[eccentricity]:
                    results_by_eccentricity[eccentricity][resolution] = {subject: {sessionDict['plExportFolder']: {'calibration_precision': None, 'analysis_precision': np.array([(i, d) for i, d in eccentricitiesPrecDict[eccentricity]], dtype=object), 'calibration': None, 'analysis': np.array([(i, d) for i, d in eccentricitiesAccDict[eccentricity]], dtype=object), 'eye0_rate': len(sessionDict['rawCalibGaze']['gaze_normal0_y']), 'eye1_rate': len(sessionDict['rawCalibGaze']['gaze_normal1_y'])}}}
                elif subject not in results_by_eccentricity[eccentricity][resolution]:
                    results_by_eccentricity[eccentricity][resolution][subject] = {sessionDict['plExportFolder']: {'calibration_precision': None, 'analysis_precision': np.array([(i, d) for i, d in eccentricitiesPrecDict[eccentricity]], dtype=object), 'calibration': None, 'analysis': np.array([(i, d) for i, d in eccentricitiesAccDict[eccentricity]], dtype=object), 'eye0_rate': len(sessionDict['rawCalibGaze']['gaze_normal0_y']), 'eye1_rate': len(sessionDict['rawCalibGaze']['gaze_normal1_y'])}}
                elif sessionDict['plExportFolder'] not in results_by_eccentricity[eccentricity][resolution][subject]:
                    results_by_eccentricity[eccentricity][resolution][subject][sessionDict['plExportFolder']] = {'calibration_precision': None, 'analysis_precision': np.array([(i, d) for i, d in eccentricitiesPrecDict[eccentricity]], dtype=object), 'calibration': None, 'analysis': np.array([(i, d) for i, d in eccentricitiesAccDict[eccentricity]], dtype=object), 'eye0_rate': len(sessionDict['rawCalibGaze']['gaze_normal0_y']), 'eye1_rate': len(sessionDict['rawCalibGaze']['gaze_normal1_y'])}
                else:
                    results_by_eccentricity[eccentricity][resolution][subject][sessionDict['plExportFolder']]['analysis'] = np.append(results_by_eccentricity[eccentricity][resolution][subject][sessionDict['plExportFolder']]['analysis'], np.array([(i, d) for i, d in eccentricitiesAccDict[eccentricity]], dtype=object), axis=0)
                    results_by_eccentricity[eccentricity][resolution][subject][sessionDict['plExportFolder']]['calibration'] = None
                    results_by_eccentricity[eccentricity][resolution][subject][sessionDict['plExportFolder']]['analysis_precision'] = np.append(results_by_eccentricity[eccentricity][resolution][subject][sessionDict['plExportFolder']]['analysis_precision'], np.array([(i, d) for i, d in eccentricitiesPrecDict[eccentricity]], dtype=object), axis=0)
                    results_by_eccentricity[eccentricity][resolution][subject][sessionDict['plExportFolder']]['calibration_precision'] = None
                    results_by_eccentricity[eccentricity][resolution][subject][sessionDict['plExportFolder']]['eye0_rate'] = len(sessionDict['rawCalibGaze']['gaze_normal0_y'])
                    results_by_eccentricity[eccentricity][resolution][subject][sessionDict['plExportFolder']]['eye1_rate'] = len(sessionDict['rawCalibGaze']['gaze_normal1_y'])

        pd_acc_constructor = []
        pd_prec_constructor = []
        for eccentricity in results_by_eccentricity.keys():
            for res in results_by_eccentricity[eccentricity]:
                for sub in results_by_eccentricity[eccentricity][res]:
                    for plugin in results_by_eccentricity[eccentricity][res][sub]:
                        for idx, (trialID, datapoint) in enumerate(results_by_eccentricity[eccentricity][res][sub][plugin]['analysis']):
                            pd_acc_constructor.append({
                                "subject": sub,
                                "resolution": res,
                                "plugin": plugin,
                                "eccentricity": eccentricity,
                                "index": idx,
                                "trial-id": trialID,
                                "eye0-rate": results_by_eccentricity[eccentricity][res][sub][plugin]['eye0_rate'],
                                "eye1-rate": results_by_eccentricity[eccentricity][res][sub][plugin]['eye1_rate'],
                                "accuracy-error": datapoint
                            })
                        for idx, (trialID, datapoint) in enumerate(results_by_eccentricity[eccentricity][res][sub][plugin]['analysis_precision']):
                            pd_prec_constructor.append({
                                "subject": sub,
                                "resolution": res,
                                "plugin": plugin,
                                "eccentricity": eccentricity,
                                "index": idx,
                                "trial-id": trialID,
                                "eye0-rate": results_by_eccentricity[eccentricity][res][sub][plugin]['eye0_rate'],
                                "eye1-rate": results_by_eccentricity[eccentricity][res][sub][plugin]['eye1_rate'],
                                "precision-error": datapoint
                            })
        pd_analysis_acc = pd.DataFrame.from_records(
            pd_acc_constructor
        )
                        
        pd_analysis_prec = pd.DataFrame.from_records(
            pd_prec_constructor
        )
        np.set_printoptions(threshold=sys.maxsize)
        pd_analysis_acc.to_csv(f'./{figout_loc}/analysis_accuracy_pd.csv')
        pd_analysis_prec.to_csv(f'./{figout_loc}/analysis_precision_pd.csv')
    else:
        def converter(instr):
            return np.fromstring(instr[1:-1], sep=' ').astype(np.float32)
        pd_analysis_acc = pd.read_csv(f'./{figout_loc}/analysis_accuracy_pd.csv', converters={'accuracy-error': converter})
        pd_analysis_prec = pd.read_csv(f'./{figout_loc}/analysis_precision_pd.csv', converters={'precision-error': converter})

    ANOVA = False
    if ANOVA:
        import pingouin
        pd_analysis_acc = pd_analysis_acc.rename(columns={"accuracy-error": "accuracyError"})
        pd_analysis_prec = pd_analysis_prec.rename(columns={"precision-error": "precisionError"})
        print("performing anova...")
        print("---------------------ACCURACY---------------------")
        #result = pingouin.anova(data=pd_analysis_acc, dv='accuracyError', between=['subject', 'resolution', 'eccentricity', 'plugin'])
        result = pingouin.rm_anova(data=pd_analysis_acc, dv='accuracyError', within=['eccentricity', 'plugin'], subject='subject')
        print(result)
        print()
        print("---------------------PRECISION---------------------")
        #result = pingouin.anova(data=pd_analysis_prec, dv='precisionError', between=['subject', 'resolution', 'eccentricity', 'plugin'])
        result = pingouin.rm_anova(data=pd_analysis_prec, dv='precisionError', within=['eccentricity', 'plugin'], subject='subject')
        print(result)
        exit()

    COLORS = ['red', 'purple', 'pink', 'orange',  'gold', 'black']
    nn_names = [
        'Detector2DRITnetEllsegV2AllvonePlugin',
        'Detector2DRITnetEllsegV2AllvoneEmbeddedPlugin',
        'Detector2DRITnetEllsegV2AllvoneEmbeddedIrisPlugin',
        'Detector2DESFnetPlugin', 'Detector2DESFnetEmbeddedPlugin',
        'Detector2DRITnetPupilPlugin',
    ]
    xlabel_dict = {
            'vanilla': 'Native',
            'Native': 'Native',
            'Detector2DRITnetEllsegV2AllvonePlugin': 'EllSegGen',
            'Detector2DRITnetEllsegV2AllvoneEmbeddedPlugin': 'EllSegGen\n(Direct Pupil)',
            'Detector2DRITnetEllsegV2AllvoneEmbeddedIrisPlugin': 'EllSegGen\n(Direct Iris)',
            'Detector2DESFnetPlugin': 'ESFnet',
            'Detector2DESFnetEmbeddedPlugin': 'ESFnet\n(Direct Pupil)',
            'Detector2DRITnetPupilPlugin': 'RITnet Pupil',
        }
    barlabel_dict = {
        'vanilla': 'Native',
        'Native': 'Native',
        'Detector2DRITnetEllsegV2AllvonePlugin': 'EllSegGen',
        'Detector2DRITnetEllsegV2AllvoneEmbeddedPlugin': 'EllSegGen (Direct Pupil)',
        'Detector2DRITnetEllsegV2AllvoneEmbeddedIrisPlugin': 'EllSegGen (Direct Iris)',
        'Detector2DESFnetPlugin': 'ESFnet',
        'Detector2DESFnetEmbeddedPlugin': 'ESFnet (Direct Pupil)',
        'Detector2DRITnetPupilPlugin': 'RITnet Pupil',
        
    }
    nn_names_ecc = nn_names

    def flatten_np(nparray):
        return np.array([item for sublist in nparray for item in sublist])

    def mean_subarrays(nparray):
        return np.array([np.nanmean(sublist) for sublist in nparray])

    def mean_confidence_interval(data, confidence=0.95):
        a = 1.0 * np.array(data[~np.isnan(data)])
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
        return m-h, m, m+h

    if True:
        DROPOUT = 10
        ELIMINATE_DROPOUTS = True
        MARKERSIZE = 2
        colors = COLORS
        SUBJECTS = (1,2,3,5,6,7,8,9,10,11)#(1,)
        
        P = flatten_np(pd_analysis_acc['accuracy-error'].to_numpy())
        gaussian_thresh = np.nanmean(P) + 2*np.nanstd(P)
        print("Mean: {}, std: {}, 2*std: {}, mean+2*std: {}".format(
            np.nanmean(P), np.nanstd(P), 2*np.nanstd(P), np.nanmean(P) + 2*np.nanstd(P)
        ))
        
        plt.figure(figsize=(6.4, 2.8))
        plt.plot(list(range(0, 50)), [100.0 - (100*np.count_nonzero(np.where(P >= i, 1, 0)) / len(P)) for i in range(0, 50)], '-o', mfc='none')
        plt.title("Data Preserved Under Different Dropout Thresholds")
        plt.xlabel("Dropout Threshold (Degrees)")
        plt.ylabel("Percentage of Data Under Threshold")
        plt.grid()
        plt.savefig("plotted.png", bbox_inches='tight')
        #exit()

        print("Data dropped: {:.3f}% dropout, {:.3f} mean, {:.3f} std (data-driven)".format(
            100*np.count_nonzero(np.where(P >= gaussian_thresh, 1, 0)) / len(P),
            np.nanmean(P[P < gaussian_thresh]),
            np.nanstd(P[P < gaussian_thresh])
        ))
        print(" "*11+"vs {:.3f}% dropout, {:.3f} mean, {:.3f} std  (hard thresh of {})".format(
            100*np.count_nonzero(np.where(P >= DROPOUT, 1, 0)) / len(P),
            np.nanmean(P[P < DROPOUT]),
            np.nanstd(P[P < DROPOUT]),
            DROPOUT
        ))
        
        for subj in SUBJECTS:
            P = flatten_np(pd_analysis_acc.loc[pd_analysis_acc['subject'] == subj]['accuracy-error'])
            print("(SUBJECT {})".format(subj))
            print("Data dropped: {:.3f}% dropout, {:.3f} mean, {:.3f} std (data-driven)".format(
                100*np.count_nonzero(np.where(P >= gaussian_thresh, 1, 0)) / len(P),
                np.nanmean(P[P < gaussian_thresh]),
                np.nanstd(P[P < gaussian_thresh])
            ))
            print(" "*11+"vs {:.3f}% dropout, {:.3f} mean, {:.3f} std  (hard thresh of {})".format(
                100*np.count_nonzero(np.where(P >= DROPOUT, 1, 0)) / len(P),
                np.nanmean(P[P < DROPOUT]),
                np.nanstd(P[P < DROPOUT]),
                DROPOUT
            ))
        print()
        print()

        grouped_dropouts = pd_analysis_acc.groupby(['resolution', 'eccentricity', 'plugin'])['accuracy-error'].aggregate(lambda d: [100*np.sum(s >= DROPOUT)/len(s) for s in d])
        print(grouped_dropouts[192, 0.0, 'Detector2DESFnetPlugin'])
        grouped_dropouts_by_subject = pd_analysis_acc.groupby(['resolution', 'subject', 'plugin'])['accuracy-error'].aggregate(lambda d: [100*np.sum(s >= DROPOUT)/len(s) for s in d])
        print(grouped_dropouts_by_subject[192, 1, 'Detector2DESFnetPlugin'])

        np_dropouts_plugins = {}
        np_dropouts_resolution = {}
        
        for plugin in ['vanilla'] + nn_names:
            P = flatten_np(pd_analysis_acc.loc[pd_analysis_acc['plugin'] == plugin]['accuracy-error'])
            print("(PLUGIN {})".format(plugin))
            print(" "*11+"{:.3f}% dropout  (hard thresh of {}deg)".format(
                100*np.count_nonzero(np.where(P >= DROPOUT, 1, 0)) / len(P),
                DROPOUT
            ))
            np_dropouts_plugins[plugin] = 100*np.count_nonzero(np.where(P >= DROPOUT, 1, 0)) / len(P)
        print()
        print()
        for resolution in (192, 400):
            print("------(RESOLUTION {})------".format(resolution))
            temp_top_dropouts = {}
            for eccentricity in (0.0, 10.0, 15.0, 20.0):
                print("------(ECCENTRICITY {})------".format(eccentricity))
                temp_dropouts = {}
                
                for plugin in ['vanilla'] + nn_names:
                    P = flatten_np(pd_analysis_acc.loc[(pd_analysis_acc['eccentricity'] == eccentricity) & (pd_analysis_acc['plugin'] == plugin) & (pd_analysis_acc['resolution'] == resolution)]['accuracy-error'])
                    print("(PLUGIN {})".format(plugin))
                    print(" "*11+"{:.3f}% dropout  (hard thresh of {}deg)".format(
                        100*np.count_nonzero(np.where(P >= DROPOUT, 1, 0)) / len(P),
                        DROPOUT
                    ))
                    temp_dropouts[plugin] = 100*np.count_nonzero(np.where(P >= DROPOUT, 1, 0)) / len(P)
                temp_top_dropouts[eccentricity] = temp_dropouts
                print()
            np_dropouts_resolution[resolution] = temp_top_dropouts

        if ELIMINATE_DROPOUTS:
            def tempfunc(row):
                row['accuracy-error'] = row['accuracy-error'][row['accuracy-error'] < DROPOUT]
                return row
            def tempfunc2(row):
                row['precision-error'] = row['precision-error'][row['accuracy-error'] < DROPOUT]
                return row

            pd_analysis_prec = pd_analysis_acc.join(pd_analysis_prec['precision-error'], how='left').apply(tempfunc2, axis=1)
            pd_analysis_acc = pd_analysis_acc.apply(tempfunc, axis=1)

        # ----- Plot All Points' Accuracy Error Over Native Accuracy Error -----
        X = mean_subarrays(pd_analysis_acc.loc[
                (pd_analysis_acc['plugin'] == 'vanilla')
            ]['accuracy-error'].to_numpy())
        #print(X)
        #print(X.dtype)
        #print(X.shape)
        #print(X.reshape(-1))
        plt.plot([0, np.max(np.nanmean(X))], [0, np.max(np.nanmean(X))], '-', label="Native Accuracy Error", c='blue')
        i = 0
        for method in nn_names:
            Y = mean_subarrays(pd_analysis_acc.loc[
                (pd_analysis_acc['plugin'] == method)
            ]['accuracy-error'].to_numpy())
            plt.plot(X, Y, '+', label=method, c=colors[i], markersize=MARKERSIZE)
            i += 1
        plt.title("NN-assisted Accuracy Error vs Native Accuracy Error")
        plt.xlabel("Native Accuracy Error")
        plt.ylabel("NN-assisted Accuracy Error")
        plt.legend()
        #plt.xlim(0, 6)
        #plt.ylim(0, 6)
        plt.savefig(f'{figout_loc}VANIL_ACCURACY_COMP.png', bbox_inches='tight')
        plt.clf()
        
        # Specify Resolution 192
        X = mean_subarrays(pd_analysis_acc.loc[
                (pd_analysis_acc['plugin'] == 'vanilla') &\
                (pd_analysis_acc['resolution'] == 192)
            ]['accuracy-error'].to_numpy())
        
        plt.plot([0, np.max(X)], [0, np.max(X)], '-', label="Native Accuracy Error", c='blue')
        i = 0
        for method in nn_names:
            Y = mean_subarrays(pd_analysis_acc.loc[
                (pd_analysis_acc['plugin'] == method) &\
                (pd_analysis_acc['resolution'] == 192)
            ]['accuracy-error'].to_numpy())
            plt.plot(X, Y, '+', label=method, c=colors[i], markersize=MARKERSIZE)
            i += 1
        plt.title("NN-assisted Accuracy Error vs Native Accuracy Error (192x192)")
        plt.xlabel("Native Accuracy Error")
        plt.ylabel("NN-assisted Accuracy Error")
        plt.legend()
        #plt.xlim(0, 6)
        #plt.ylim(0, 6)
        plt.savefig(f'{figout_loc}VANIL_ACCURACY_192_COMP.png', bbox_inches='tight')
        plt.clf()
        
        # Specify Resolution 400
        X = mean_subarrays(pd_analysis_acc.loc[
                (pd_analysis_acc['plugin'] == 'vanilla') &\
                (pd_analysis_acc['resolution'] == 400)
            ]['accuracy-error'].to_numpy())
        
        plt.plot([0, np.max(X)], [0, np.max(X)], '-', label="Native Accuracy Error", c='blue')
        i = 0
        for method in nn_names:
            Y = mean_subarrays(pd_analysis_acc.loc[
                (pd_analysis_acc['plugin'] == method) &\
                (pd_analysis_acc['resolution'] == 400)
            ]['accuracy-error'].to_numpy())
            plt.plot(X, Y, '+', label=method, c=colors[i], markersize=MARKERSIZE)
            i += 1
        plt.title("NN-assisted Accuracy Error vs Native Accuracy Error (400x400)")
        plt.xlabel("Native Accuracy Error")
        plt.ylabel("NN-assisted Accuracy Error")
        plt.legend()
        #plt.xlim(0, 6)
        #plt.ylim(0, 6)
        plt.savefig(f'{figout_loc}VANIL_ACCURACY_400_COMP.png', bbox_inches='tight')
        plt.clf()

        # ----- Plot All Points' Precision Error Over Native Precision Error -----
        X = mean_subarrays(pd_analysis_prec.loc[
                (pd_analysis_prec['plugin'] == 'vanilla')
            ]['precision-error'].to_numpy())
        plt.plot([0, np.max(X)], [0, np.max(X)], '-', label="Native Precision Error", c='blue')
        i = 0
        for method in nn_names:
            Y = mean_subarrays(pd_analysis_prec.loc[
                (pd_analysis_prec['plugin'] == method)
            ]['precision-error'].to_numpy())
            plt.plot(X, Y, '+', label=method, c=colors[i], markersize=MARKERSIZE)
            i += 1
        plt.title("NN-assisted Precision Error vs Native Precision Error")
        plt.xlabel("Native Precision Error")
        plt.ylabel("NN-assisted Precision Error")
        plt.legend()
        #plt.xlim(0, 6)
        #plt.ylim(0, 6)
        plt.savefig(f'{figout_loc}VANIL_PRECISION_COMP.png', bbox_inches='tight')
        plt.clf()
        
        # Specify Resolution 192
        X = mean_subarrays(pd_analysis_prec.loc[
                (pd_analysis_prec['plugin'] == 'vanilla') &\
                (pd_analysis_prec['resolution'] == 192)
            ]['precision-error'].to_numpy())
        plt.plot([0, np.max(X)], [0, np.max(X)], '-', label="Native Precision Error", c='blue')
        i = 0
        for method in nn_names:
            Y = mean_subarrays(pd_analysis_prec.loc[
                (pd_analysis_prec['plugin'] == method) &\
                (pd_analysis_prec['resolution'] == 192)
            ]['precision-error'].to_numpy())
            plt.plot(X, Y, '+', label=method, c=colors[i], markersize=MARKERSIZE)
            i += 1
        plt.title("NN-assisted Precision Error vs Native Precision Error (192x192)")
        plt.xlabel("Native Precision Error")
        plt.ylabel("NN-assisted Precision Error")
        plt.legend()
        #plt.xlim(0, 6)
        #plt.ylim(0, 6)
        plt.savefig(f'{figout_loc}VANIL_PRECISION_192_COMP.png', bbox_inches='tight')
        plt.clf()
        
        # Specify Resolution 400
        X = mean_subarrays(pd_analysis_prec.loc[
                (pd_analysis_prec['plugin'] == 'vanilla') &\
                (pd_analysis_prec['resolution'] == 400)
            ]['precision-error'].to_numpy())
        plt.plot([0, np.max(X)], [0, np.max(X)], '-', label="Native Precision Error", c='blue')
        i = 0
        for method in nn_names:
            Y = mean_subarrays(pd_analysis_prec.loc[
                (pd_analysis_prec['plugin'] == method) &\
                (pd_analysis_prec['resolution'] == 400)
            ]['precision-error'].to_numpy())
            plt.plot(X, Y, '+', label=method, c=colors[i], markersize=MARKERSIZE)
            i += 1
        plt.title("NN-assisted Precision Error vs Native Precision Error (400x400)")
        plt.xlabel("Native Precision Error")
        plt.ylabel("NN-assisted Precision Error")
        plt.legend()
        #plt.xlim(0, 6)
        #plt.ylim(0, 6)
        plt.savefig(f'{figout_loc}VANIL_PRECISION_400_COMP.png', bbox_inches='tight')
        plt.clf()
        
        # ----- Plot All Points' Binned&Averaged Accuracy Error Over Native Binned&Averaged Accuracy Error -----
        from decimal import Decimal
        for resolution in (None, 192, 400):
            bins = {}
            #expand_bin_size_at = 12.0
            colors = COLORS
            if resolution is None:
                X = mean_subarrays(pd_analysis_acc.loc[
                    (pd_analysis_acc['plugin'] == 'vanilla')
                ]['accuracy-error'].to_numpy())
            else:
                X = mean_subarrays(pd_analysis_acc.loc[
                    (pd_analysis_acc['plugin'] == 'vanilla') &\
                    (pd_analysis_acc['resolution'] == resolution)
                ]['accuracy-error'].to_numpy())
            plt.plot([0, np.max(X)], [0, np.max(X)], '-', label="Native Accuracy Error", c='blue')
            i = 0
            for method in nn_names:
                bins[method] = []
                if resolution is None:
                    X = mean_subarrays(pd_analysis_acc.loc[
                        (pd_analysis_acc['plugin'] == 'vanilla')
                    ]['accuracy-error'].to_numpy())
                    Y = mean_subarrays(pd_analysis_acc.loc[
                        (pd_analysis_acc['plugin'] == method)
                    ]['accuracy-error'].to_numpy())
                else:
                    X = mean_subarrays(pd_analysis_acc.loc[
                        (pd_analysis_acc['plugin'] == 'vanilla') &\
                        (pd_analysis_acc['resolution'] == resolution)
                    ]['accuracy-error'].to_numpy())
                    Y = mean_subarrays(pd_analysis_acc.loc[
                        (pd_analysis_acc['plugin'] == method) &\
                        (pd_analysis_acc['resolution'] == resolution)
                    ]['accuracy-error'].to_numpy())
                vanillas = X

                percentile50 = np.nanpercentile(vanillas, 50)
                percentile90 = np.nanpercentile(vanillas, 90)
                percentile95 = np.nanpercentile(vanillas, 95)

                expand_bin_size_at = percentile90
                max_vanillas = round(np.nanmax(vanillas), 1)  # nearest tenth place
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

                for pt_idx in range(len(X)):
                    pt = X[pt_idx]
                    if pt < expand_bin_size_at + (bin_group_1_size / 2):
                        if (pt % bin_group_1_size) < (bin_group_1_size / 2):
                            modfix = float(Decimal(pt.item()) % bg1s_decimal)
                            idx = pt - modfix
                        else:
                            modfix = float(Decimal(pt.item()) % bg1s_decimal)
                            idx = pt + (bin_group_1_size - modfix)
                        idx = idx / bin_group_1_size
                        bins[method][int(round(idx))].append(Y[pt_idx])
                    else:
                        if (pt % bin_group_3_size) < (bin_group_3_size / 2):
                            modfix = float(Decimal(pt.item()) % bg3s_decimal)
                            idx = pt - modfix
                        else:
                            modfix = float(Decimal(pt.item()) % bg3s_decimal)
                            idx = pt + (bin_group_3_size - modfix)
                        #idx = round(idx)
                        idx = (expand_bin_size_at / bin_group_1_size) + ((idx - expand_bin_size_at) / bin_group_3_size)
                        if not np.isnan(idx):
                            bins[method][int(round(idx))].append(Y[pt_idx])
                X = []
                Y = []
                for idx in range(len(bins[method])):
                    if len(bins[method][idx]):
                        if idx <= (expand_bin_size_at / bin_group_1_size):
                            bin = bin_group_1_size * idx
                        else:
                            bin = expand_bin_size_at + bin_group_3_size * (idx - (expand_bin_size_at / bin_group_1_size))
                        #X.append(bin)
                        #Y.append(np.mean(bins[method][idx]))
                        for currpt in bins[method][idx]:
                            X.append(bin)
                            Y.append(currpt)
                #plt.plot(X, Y, '-o', markersize=5, label=method, c=colors[i])
                sns.lineplot(x=X, y=Y, markers=True, label=method, color=colors[i], marker='o')
                #plt.plot(X, Y, '-o', markersize=5, label=method, c=colors[i])
                i += 1
            
            plt.xlabel("Native Accuracy Error (degrees) (bin interval = {} -> {})".format(bin_group_1_size, bin_group_3_size))
            plt.ylabel("NN-assisted Accuracy Error (degrees)")
            #plt.xlim(0, 12)
            #plt.ylim(0, 12)
            plt.axvline(percentile50, linestyle=":", color="blue", label="50th percentile")
            plt.axvline(percentile90, linestyle=":", color="green", label="90th percentile")
            plt.axvline(percentile95, linestyle=":", color="red", label="95th percentile")
            plt.legend()
            #plt.xlim(0, percentile90)
            #plt.ylim(0, 22)
            if resolution is None:
                plt.title("NN-assisted Binned Mean Accuracy Error vs Native Accuracy Error")
                plt.savefig(f'{figout_loc}VANIL_ACC_BINNED_COMP.png', bbox_inches='tight')
                plt.xlim(0, percentile90)
                plt.ylim(0, 22)
                plt.savefig(f'{figout_loc}VANIL_ACC_BINNED_LIM_PERCENTILE90_COMP.png', bbox_inches='tight')
                plt.clf()
            else:
                plt.title("NN-assisted Binned Mean Accuracy Error vs Native Accuracy Error ({}x{})".format(resolution, resolution))
                plt.ylim(0, 70 if resolution == 192 else 100)
                plt.savefig(f'{figout_loc}VANIL_ACC_BINNED_{resolution}_COMP.png', bbox_inches='tight')
                plt.xlim(0, percentile90)
                plt.ylim(0, 22)
                plt.savefig(f'{figout_loc}VANIL_ACC_BINNED_LIM_PERCENTILE90_{resolution}_COMP.png', bbox_inches='tight')
                plt.clf()

        # ---------- Compare distribution of accuracy errors across subjects ----------
        print("-----All Data (Accuracy)-----")
        f, axs = plt.subplots(5, 2, sharex=True, sharey=True, figsize=(5,6))
        for subjnum in SUBJECTS:
            subjidx = subjnum - 1
            if subjidx > 2:
                subjidx -= 1
            GGG = flatten_np(pd_analysis_acc.loc[pd_analysis_acc['subject'] == subjnum]['accuracy-error'].to_numpy())
            print("(Subject {}) Mean: {}, std: {}".format(subjnum, np.nanmean(GGG), np.nanstd(GGG)))
            ax = axs[subjidx % 5][int(subjidx / 5)]
            ax.hist(GGG, 200)
            ax.set_title(f"Subject {subjnum}", fontsize='small')
            ax.set_xlim([0, 50])
            ax.axvline(x=DROPOUT, color='red', lw=0.5)
        f.suptitle("Accuracy Error Distribution Across Subjects")
        f.tight_layout()
        f.savefig(f'{figout_loc}HIST_ACC_ALL.png', dpi=600)
        plt.clf()
        plt.close()

        # ---------- SUMMARY DATA FOR RESOLUTIONS 192 AND 400 ----------
        from matplotlib.ticker import FuncFormatter
        plt.style.use('ggplot')
        
        def deg_suffixer(x, pos):
            return f'{int(x)}°' if int(x) == x else f'{x}°'
        deg_formatter = FuncFormatter(deg_suffixer)
        def percent_suffixer(x, pos):
            return f'{int(x)}%' if int(x) == x else f'{x}%'
        percent_formatter = FuncFormatter(percent_suffixer)

        fig192, axes192 = plt.subplots(1, 3, figsize=(16, 4))
        ax192_1, ax192_2, ax192_3 = axes192
        fig400, axes400 = plt.subplots(1, 3, figsize=(16, 4))
        ax400_1, ax400_2, ax400_3 = axes400
        for resolution in (192, 400):
            print("-----Resolution {} (Accuracy)-----".format(resolution))
            eccentricity_accs = {}
            acc_bins = {}
            
            for eccentricity in (0.0, 10.0, 15.0, 20.0):
                X = mean_subarrays(pd_analysis_acc.loc[
                    (pd_analysis_acc['plugin'] == 'vanilla') &\
                    (pd_analysis_acc['eccentricity'] == eccentricity) &\
                    (pd_analysis_acc['resolution'] == resolution)
                ]['accuracy-error'].to_numpy())
                if 'Native' not in eccentricity_accs:
                    eccentricity_accs['Native'] = X
                    acc_bins['Native'] = [eccentricity for i in range(len(X))]
                else:
                    eccentricity_accs['Native'] = np.concatenate((eccentricity_accs['Native'], X))
                    acc_bins['Native'] = np.concatenate((acc_bins['Native'], [eccentricity for i in range(len(X))]))
                for method in nn_names_ecc:
                    Y = mean_subarrays(pd_analysis_acc.loc[
                        (pd_analysis_acc['plugin'] == method) &\
                        (pd_analysis_acc['eccentricity'] == eccentricity) &\
                        (pd_analysis_acc['resolution'] == resolution)
                    ]['accuracy-error'].to_numpy())
                    if method not in eccentricity_accs:
                        eccentricity_accs[method] = Y
                        acc_bins[method] = [eccentricity for i in range(len(Y))]
                    else:
                        eccentricity_accs[method] = np.concatenate((eccentricity_accs[method], Y))
                        acc_bins[method] = np.concatenate((acc_bins[method], [eccentricity for i in range(len(Y))]))

            print("-----Resolution {} (Precision)-----".format(resolution))
            eccentricity_precs = {}
            prec_bins = {}
            for eccentricity in (0.0, 10.0, 15.0, 20.0):
                X = mean_subarrays(pd_analysis_prec.loc[
                        (pd_analysis_prec['plugin'] == 'vanilla') &\
                        (pd_analysis_prec['eccentricity'] == eccentricity) &\
                        (pd_analysis_prec['resolution'] == resolution)
                    ]['precision-error'].to_numpy())
                if 'Native' not in eccentricity_precs:
                    eccentricity_precs['Native'] = X
                    prec_bins['Native'] = [eccentricity for i in range(len(X))]
                else:
                    eccentricity_precs['Native'] = np.concatenate((eccentricity_precs['Native'], X))
                    prec_bins['Native'] = np.concatenate((prec_bins['Native'], [eccentricity for i in range(len(X))]))
                for method in nn_names_ecc:
                    Y = mean_subarrays(pd_analysis_prec.loc[
                        (pd_analysis_prec['plugin'] == method) &\
                        (pd_analysis_prec['eccentricity'] == eccentricity) &\
                        (pd_analysis_prec['resolution'] == resolution)
                    ]['precision-error'].to_numpy())
                    if method not in eccentricity_precs:
                        eccentricity_precs[method] = Y
                        prec_bins[method] = [eccentricity for i in range(len(Y))]
                    else:
                        eccentricity_precs[method] = np.concatenate((eccentricity_precs[method], Y))
                        prec_bins[method] = np.concatenate((prec_bins[method], [eccentricity for i in range(len(Y))]))

            FONT_SIZE = 15

            # ----- ROBUSTNESS -----

            i = 0
            if resolution == 192:
                ax = ax192_1
            else:
                ax = ax400_1
            for method in eccentricity_accs:
                newmeth = method
                if newmeth == 'Native':
                    newmeth = 'vanilla'
                
                X = []
                Y = []
                
                for eccentricity in (0.0, 10.0, 15.0, 20.0):
                    Y = np.concatenate((Y, grouped_dropouts[resolution, eccentricity, newmeth]))
                    X = np.concatenate((X, [eccentricity for _ in range(len(grouped_dropouts[resolution, eccentricity, newmeth]))])) 
                label = xlabel_dict[method]
                sns.lineplot(x=X, y=Y, ax=ax, markers=True, label=label, color=colors[i-1] if i > 0 else 'blue', marker='o')
                i += 1

            #ax.set_title("Dropout Rate", pad=20)
            ax.set_xlabel("Eccentricity")
            ax.set_ylabel("Dropout Rate")
            ax.set_xticks((0.0, 10.0, 15.0, 20.0))
            ax.yaxis.set_major_formatter(percent_formatter)
            ax.xaxis.set_major_formatter(deg_formatter)
            #plt.xlim(0, 12)
            #plt.ylim(0, 19 if resolution == 192 else 12.5)
            if figout_loc[-3:] == '2d/':
                if resolution == 192:
                    ax.set_ylim(0, 13)
                else:
                    ax.set_ylim(0, 25)
            else:
                if resolution == 192:
                    ax.set_ylim(0, 20)
                else:
                    ax.set_ylim(0, 30)
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                        ax.get_xticklabels() + ax.get_yticklabels() + ax.get_legend().get_texts()):
                item.set_fontsize(FONT_SIZE)
            ax.legend().remove()
            # ----- ACCURACY -----
            i = 0
            if resolution == 192:
                ax = ax192_2
            else:
                ax = ax400_2
            for method in eccentricity_accs:
                print("(accuracy) method:{}{}, dropout: {:.2f}%, mean (with 95% CIs): {}, std: {:.2f}".format(' '*(28-len(barlabel_dict[method])), barlabel_dict[method],
                    100*np.count_nonzero(np.isnan(eccentricity_accs[method]) | np.where(eccentricity_accs[method] >= DROPOUT, 1, 0)) / len(eccentricity_accs[method]),
                    mean_confidence_interval(eccentricity_accs[method]), np.nanstd(eccentricity_accs[method])))
                newmeth = method
                if newmeth == 'Native':
                    newmeth = 'vanilla'
                
                label = xlabel_dict[method]
                sns.lineplot(x=acc_bins[method], y=eccentricity_accs[method], ax=ax, markers=True, label=label, color=colors[i-1] if i > 0 else 'blue', marker='o')
                i += 1

            #ax.set_title("Accuracy Error", pad=20)
            ax.set_title(f'{resolution}x{resolution}px ({"Feature-Based" if figout_loc[-3:] == "2d/" else "3D Model-Based"})', pad=(22 if figout_loc[-3:] == "2d/"  else 0))
            ax.set_xlabel("Eccentricity")
            ax.set_ylabel("Accuracy Error")
            ax.set_xticks((0.0, 10.0, 15.0, 20.0))
            ax.yaxis.set_major_formatter(deg_formatter)
            ax.xaxis.set_major_formatter(deg_formatter)
            #plt.xlim(0, 12)
            #plt.ylim(0, 19 if resolution == 192 else 12.5)
            if figout_loc[-3:] == '2d/':
                if resolution == 192:
                    ax.set_ylim(0, 5)
                else:
                    ax.set_ylim(0, 5)
            else:
                if resolution == 192:
                    ax.set_ylim(0, 7)
                else:
                    ax.set_ylim(0, 7)
            ax.legend().remove()
            #ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05),
            #    fancybox=True, shadow=True, ncol=7)
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                        ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(FONT_SIZE)

            # ----- PRECISION -----
            i = 0
            if resolution == 192:
                ax = ax192_3
            else:
                ax = ax400_3
            for method in eccentricity_precs:
                print("(precision) method:{}{}, mean (with 95% CIs): {}, std: {:.2f}".format(' '*(28-len(barlabel_dict[method])), barlabel_dict[method],
                    mean_confidence_interval(eccentricity_precs[method]), np.nanstd(eccentricity_precs[method])))
                newmeth = method
                if newmeth == 'Native':
                    newmeth = 'vanilla'
                
                label = xlabel_dict[method]
                sns.lineplot(x=acc_bins[method], y=eccentricity_precs[method], ax=ax, markers=True, label=label, color=colors[i-1] if i > 0 else 'blue', marker='o')
                i += 1
            #ax.set_title("Precision Error", pad=20)
            ax.set_xlabel("Eccentricity")
            ax.set_ylabel("Precision Error")
            ax.set_xticks((0.0, 10.0, 15.0, 20.0))
            ax.yaxis.set_major_formatter(deg_formatter)
            ax.xaxis.set_major_formatter(deg_formatter)
            #plt.xlim(0, 12)
            #plt.ylim(0, 19 if resolution == 192 else 12.5)
            if figout_loc[-3:] == '2d/':
                if resolution == 192:
                    ax.set_ylim(0, 3)
                else:
                    ax.set_ylim(0, 7)
            else:
                if resolution == 192:
                    ax.set_ylim(0, 6)
                else:
                    ax.set_ylim(0, 6)
            ax.legend().remove()
            #ax2 = ax.twinx()
            #ax2.set_yticks([])
            #ax2.set_ylabel(f'{resolution}x{resolution}px\n({"Appearance-Based" if figout_loc[-3:] == "2d/" else "Model-Based"})', rotation=-90, labelpad=30)
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                        ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(FONT_SIZE)
            #for item in ([ax2.title, ax2.xaxis.label, ax2.yaxis.label] +
            #            ax2.get_xticklabels() + ax2.get_yticklabels()):
            #    item.set_fontsize(int(FONT_SIZE*1.5))

        handles, labels = ax192_1.get_legend_handles_labels()
        if figout_loc[-3:] == "2d/":
            fig192.legend(handles, labels, loc='upper center', fancybox=True, shadow=True, ncol=7, bbox_to_anchor=(0.5, 0.93), prop={'size': 12})
        fig192.tight_layout()
        #fig192.suptitle(f'{192}x{192}px\n({"Appearance-Based" if figout_loc[-3:] == "2d/" else "Model-Based"})', size=20, y=1.05)#), pad=20)
        fig192.savefig(f'{figout_loc}ecc_separated_192.png')#, bbox_inches='tight')
        fig192.clf()
        handles, labels = ax400_1.get_legend_handles_labels()
        if figout_loc[-3:] == "2d/":
            fig400.legend(handles, labels, loc='upper center', fancybox=True, shadow=True, ncol=7, bbox_to_anchor=(0.5, 0.93), prop={'size': 12})
        fig400.tight_layout()
        #fig400.suptitle(f'{400}x{400}px\n({"Appearance-Based" if figout_loc[-3:] == "2d/" else "Model-Based"})', size=20, y=1.05)#, pad=20)
        fig400.savefig(f'{figout_loc}ecc_separated_400.png')#, bbox_inches='tight')
        fig400.clf()
        plt.clf()
        #fig.legend
        exit()

        # ---------- Eccentricity-separated accuracy comparison ----------
        for resolution in (None, 192, 400):
            print("-----Resolution {} (Accuracy)-----".format(resolution))
            eccentricity_accs = {}
            acc_bins = {}
            for eccentricity in (0.0, 10.0, 15.0, 20.0):
                if resolution is None:
                    X = flatten_np(pd_analysis_acc.loc[
                        (pd_analysis_acc['plugin'] == 'vanilla') &\
                        (pd_analysis_acc['eccentricity'] == eccentricity)
                    ]['accuracy-error'].to_numpy())
                else:
                    X = flatten_np(pd_analysis_acc.loc[
                        (pd_analysis_acc['plugin'] == 'vanilla') &\
                        (pd_analysis_acc['eccentricity'] == eccentricity) &\
                        (pd_analysis_acc['resolution'] == resolution)
                    ]['accuracy-error'].to_numpy())
                if 'Native' not in eccentricity_accs:
                    eccentricity_accs['Native'] = X
                    acc_bins['Native'] = [eccentricity for i in range(len(X))]
                else:
                    eccentricity_accs['Native'] = np.concatenate((eccentricity_accs['Native'], X))
                    acc_bins['Native'] = np.concatenate((acc_bins['Native'], [eccentricity for i in range(len(X))]))
                for method in nn_names_ecc:
                    if resolution is None:
                        Y = flatten_np(pd_analysis_acc.loc[
                            (pd_analysis_acc['plugin'] == method) &\
                            (pd_analysis_acc['eccentricity'] == eccentricity)
                        ]['accuracy-error'].to_numpy())
                    else:
                        Y = flatten_np(pd_analysis_acc.loc[
                            (pd_analysis_acc['plugin'] == method) &\
                            (pd_analysis_acc['eccentricity'] == eccentricity) &\
                            (pd_analysis_acc['resolution'] == resolution)
                        ]['accuracy-error'].to_numpy())
                    if method not in eccentricity_accs:
                        eccentricity_accs[method] = Y
                        acc_bins[method] = [eccentricity for i in range(len(Y))]
                    else:
                        eccentricity_accs[method] = np.concatenate((eccentricity_accs[method], Y))
                        acc_bins[method] = np.concatenate((acc_bins[method], [eccentricity for i in range(len(Y))]))
            i = 0
            for method in eccentricity_accs:
                print("method:{}{}, dropout: {:.2f}%, mean (with 95% CIs): {}, std: {:.2f}".format(' '*(28-len(barlabel_dict[method])), barlabel_dict[method],
                    100*np.count_nonzero(np.isnan(eccentricity_accs[method]) | np.where(eccentricity_accs[method] >= DROPOUT, 1, 0)) / len(eccentricity_accs[method]),
                    mean_confidence_interval(eccentricity_accs[method]), np.nanstd(eccentricity_accs[method])))
                newmeth = method
                if newmeth == 'Native':
                    newmeth = 'vanilla'
                
                if resolution is None:
                    label = "({:.1f}%) ".format(np_dropouts_plugins[newmeth]) + xlabel_dict[method]
                else:
                    label = xlabel_dict[method]
                sns.lineplot(x=acc_bins[method], y=eccentricity_accs[method], markers=True, label=label, color=colors[i-1] if i > 0 else 'blue', marker='o')
                i += 1
            if resolution is None:
                plt.title("Accuracy Across Eccentricities (All Resolutions)")
                plt.xlabel("Eccentricity")
                plt.ylabel("Accuracy Error (degrees)")
                #plt.xlim(0, 12)
                if ELIMINATE_DROPOUTS:
                    plt.ylim(0, 19)
                else:
                    plt.ylim(0, 19)
                plt.legend()
                plt.savefig(f'{figout_loc}ecc_separated_acc_comp.png', bbox_inches='tight')
                plt.clf()
            else:
                plt.title("Accuracy Across Eccentricities ({}x{})".format(resolution, resolution))
                plt.xlabel("Eccentricity")
                plt.ylabel("Accuracy Error (degrees)")
                #plt.xlim(0, 12)
                #plt.ylim(0, 19 if resolution == 192 else 12.5)
                if ELIMINATE_DROPOUTS:
                    plt.ylim(0, 19)
                else:
                    plt.ylim(0, 19)
                plt.legend()
                plt.savefig(f'{figout_loc}ecc_separated_acc_{resolution}_comp.png', bbox_inches='tight')
                plt.clf()

        # ---------- Eccentricity-separated (and SUBJECT-separated) accuracy comparison ----------
        for resolution in (192, 400, None):
            print("-----Resolution {} (Accuracy BY SUBJECT)-----".format(resolution))
            f, axs = plt.subplots(5, 2, sharex=True)
            eccentricity_accs_subject = {n:{} for n in SUBJECTS}
            acc_bins_subject = {n:{} for n in SUBJECTS}
            for eccentricity in (0.0, 10.0, 15.0, 20.0):
                for subj in SUBJECTS:
                    if resolution is None:
                        X = flatten_np(pd_analysis_acc.loc[
                            (pd_analysis_acc['plugin'] == 'vanilla') &\
                            (pd_analysis_acc['eccentricity'] == eccentricity) &\
                            (pd_analysis_acc['subject'] == subj)
                        ]['accuracy-error'].to_numpy())
                    else:
                        X = flatten_np(pd_analysis_acc.loc[
                            (pd_analysis_acc['plugin'] == 'vanilla') &\
                            (pd_analysis_acc['eccentricity'] == eccentricity) &\
                            (pd_analysis_acc['resolution'] == resolution) &\
                            (pd_analysis_acc['subject'] == subj)
                        ]['accuracy-error'].to_numpy())
                    if 'Native' not in eccentricity_accs_subject[subj]:
                        eccentricity_accs_subject[subj]['Native'] = X
                        acc_bins_subject[subj]['Native'] = [eccentricity for i in range(len(X))]
                    else:
                        eccentricity_accs_subject[subj]['Native'] = np.concatenate((eccentricity_accs_subject[subj]['Native'], X))
                        acc_bins_subject[subj]['Native'] = np.concatenate((acc_bins_subject[subj]['Native'], [eccentricity for i in range(len(X))]))
                    for method in nn_names_ecc:
                        if resolution is None:
                            Y = flatten_np(pd_analysis_acc.loc[
                                (pd_analysis_acc['plugin'] == method) &\
                                (pd_analysis_acc['eccentricity'] == eccentricity) &\
                                (pd_analysis_acc['subject'] == subj)
                            ]['accuracy-error'].to_numpy())
                        else:
                            Y = flatten_np(pd_analysis_acc.loc[
                                (pd_analysis_acc['plugin'] == method) &\
                                (pd_analysis_acc['eccentricity'] == eccentricity) &\
                                (pd_analysis_acc['resolution'] == resolution) &\
                                (pd_analysis_acc['subject'] == subj)
                            ]['accuracy-error'].to_numpy())
                        if method not in eccentricity_accs_subject[subj]:
                            eccentricity_accs_subject[subj][method] = Y
                            acc_bins_subject[subj][method] = [eccentricity for i in range(len(Y))]
                        else:
                            eccentricity_accs_subject[subj][method] = np.concatenate((eccentricity_accs_subject[subj][method], Y))
                            acc_bins_subject[subj][method] = np.concatenate((acc_bins_subject[subj][method], [eccentricity for i in range(len(Y))]))
            for subj in SUBJECTS:
                print("SUBJECT {}".format(subj))
                subjidx = subj - 1
                if subjidx > 2:
                    subjidx -= 1
                i = 0
                for method in eccentricity_accs_subject[subj]:
                    if len(eccentricity_accs_subject[subj][method]) > 0:
                        curr_dropout = 100*np.count_nonzero(np.isnan(eccentricity_accs_subject[subj][method]) | np.where(eccentricity_accs_subject[subj][method] >= DROPOUT, 1, 0)) / len(eccentricity_accs_subject[subj][method])
                    else:
                        curr_dropout = 100
                    print("method:{}{}, dropout: {:.2f}%, mean (with 95% CIs): {}, std: {:.2f}".format(' '*(28-len(barlabel_dict[method])), barlabel_dict[method],
                        curr_dropout,
                        mean_confidence_interval(eccentricity_accs_subject[subj][method]), np.nanstd(eccentricity_accs_subject[subj][method])))
                    sns.lineplot(x=acc_bins_subject[subj][method], y=eccentricity_accs_subject[subj][method], markers=True, label=xlabel_dict[method], color=colors[i-1] if i > 0 else 'blue', marker='o', ax=axs[subjidx % 5][int(subjidx / 5)])
                    i += 1
            if resolution is None:
                f.suptitle("Accuracy Across Eccentricities (All Resolutions)")
                #f.xlabel("Eccentricity")
                #f.ylabel("Accuracy Error (degrees)")
                #plt.xlim(0, 12)
                #plt.ylim(0, 12)
                #axs[0][0].legend()
                for ii in range(5):
                    for jj in range(2):
                        if ii != 0 or jj != 0:
                            axs[ii][jj].set_title((ii + jj*5)+1 if (ii + jj*5)+1 < 4 else (ii + jj*5)+2, fontsize='small')
                            axs[ii][jj].get_legend().remove()
                f.savefig(f'{figout_loc}ecc_separated_subj_acc_comp.png', bbox_inches='tight')
                plt.clf()
            else:
                f.suptitle("Accuracy Across Eccentricities ({}x{})".format(resolution, resolution))
                #f.supxlabel("Eccentricity")
                #f.supylabel("Accuracy Error (degrees)")
                #plt.xlim(0, 12)
                #plt.ylim(0, 19 if resolution == 192 else 12.5)
                #axs[0][0].legend()
                for ii in range(5):
                    for jj in range(2):
                        if ii != 0 or jj != 0:
                            axs[ii][jj].set_title((ii + jj*5)+1 if (ii + jj*5)+1 < 4 else (ii + jj*5)+2, fontsize='small')
                            axs[ii][jj].get_legend().remove()
                f.savefig(f'{figout_loc}ecc_separated_subj_acc_{resolution}_comp.png', bbox_inches='tight')
                plt.clf()

        # ---------- Eccentricity-separated accuracy comparison (subtracted by native) ----------
        for resolution in (None, 192, 400):
            print("-----Resolution {} (Accuracy Averaged & Sub-native)-----".format(resolution))
            eccentricity_accs = {}
            acc_bins = {}
            for eccentricity in (0.0, 10.0, 15.0, 20.0):
                if resolution is None:
                    X = mean_subarrays(pd_analysis_acc.loc[
                            (pd_analysis_acc['plugin'] == 'vanilla') &\
                            (pd_analysis_acc['eccentricity'] == eccentricity)
                        ]['accuracy-error'].to_numpy())
                else:
                    X = mean_subarrays(pd_analysis_acc.loc[
                            (pd_analysis_acc['plugin'] == 'vanilla') &\
                            (pd_analysis_acc['eccentricity'] == eccentricity) &\
                            (pd_analysis_acc['resolution'] == resolution)
                        ]['accuracy-error'].to_numpy())
                if 'Native' not in eccentricity_accs:
                    eccentricity_accs['Native'] = np.array(X) - np.array(X)
                    acc_bins['Native'] = [eccentricity for i in range(len(X))]
                else:
                    eccentricity_accs['Native'] = np.concatenate((eccentricity_accs['Native'],
                        np.array(X) - np.array(X)))
                    acc_bins['Native'] = np.concatenate((acc_bins['Native'], [eccentricity for i in range(len(X))]))
                for method in nn_names_ecc:
                    if resolution is None:
                        Y = mean_subarrays(pd_analysis_acc.loc[
                            (pd_analysis_acc['plugin'] == method) &\
                            (pd_analysis_acc['eccentricity'] == eccentricity)
                        ]['accuracy-error'].to_numpy())
                    else:
                        Y = mean_subarrays(pd_analysis_acc.loc[
                            (pd_analysis_acc['plugin'] == method) &\
                            (pd_analysis_acc['eccentricity'] == eccentricity) &\
                            (pd_analysis_acc['resolution'] == resolution)
                        ]['accuracy-error'].to_numpy())
                    if method not in eccentricity_accs:
                        eccentricity_accs[method] = np.array(Y) - np.array(X)
                        acc_bins[method] = [eccentricity for i in range(len(Y))]
                    else:
                        eccentricity_accs[method] = np.concatenate(
                            (eccentricity_accs[method],
                            np.array(Y) - np.array(X))
                        )
                        acc_bins[method] = np.concatenate((acc_bins[method], [eccentricity for i in range(len(Y))]))
            i = 0
            for method in eccentricity_accs:
                sns.lineplot(x=acc_bins[method], y=eccentricity_accs[method], markers=True, label=xlabel_dict[method], color=colors[i-1] if i > 0 else 'blue', marker='o')
                i += 1
            if resolution is None:
                plt.title("Accuracy (Comp. to Native) Across Eccentricities")
                plt.xlabel("Eccentricity")
                plt.ylabel("Accuracy Error Relative to Native (degrees)")
                #plt.xlim(0, 12)
                #plt.ylim(0, 12)
                plt.legend()
                plt.savefig(f'{figout_loc}ecc_separated_acc_subnative_comp.png', bbox_inches='tight')
                plt.clf()
            else:
                plt.title("Accuracy (Comp. to Native) Across Eccentricities ({}x{})".format(resolution, resolution))
                plt.xlabel("Eccentricity")
                plt.ylabel("Accuracy Error Relative to Native (degrees)")
                #plt.xlim(0, 12)
                plt.ylim(-7.5 if resolution == 192 else -8, 14 if resolution == 192 else 2)
                plt.legend()
                plt.savefig(f'{figout_loc}ecc_separated_acc_subnative_{resolution}_comp.png', bbox_inches='tight')
                plt.clf()
        
        # ---------- Eccentricity-separated precision comparison ----------
        for resolution in (None, 192, 400):
            eccentricity_precs = {}
            prec_bins = {}
            for eccentricity in (0.0, 10.0, 15.0, 20.0):
                if resolution is None:
                    X = flatten_np(pd_analysis_prec.loc[
                            (pd_analysis_prec['plugin'] == 'vanilla') &\
                            (pd_analysis_prec['eccentricity'] == eccentricity)
                        ]['precision-error'].to_numpy())
                else:
                    X = flatten_np(pd_analysis_prec.loc[
                            (pd_analysis_prec['plugin'] == 'vanilla') &\
                            (pd_analysis_prec['eccentricity'] == eccentricity) &\
                            (pd_analysis_prec['resolution'] == resolution)
                        ]['precision-error'].to_numpy())
                if 'Native' not in eccentricity_precs:
                    eccentricity_precs['Native'] = X
                    prec_bins['Native'] = [eccentricity for i in range(len(X))]
                else:
                    eccentricity_precs['Native'] = np.concatenate((eccentricity_precs['Native'], X))
                    prec_bins['Native'] = np.concatenate((prec_bins['Native'], [eccentricity for i in range(len(X))]))
                for method in nn_names_ecc:
                    if resolution is None:
                        Y = flatten_np(pd_analysis_prec.loc[
                            (pd_analysis_prec['plugin'] == method) &\
                            (pd_analysis_prec['eccentricity'] == eccentricity)
                        ]['precision-error'].to_numpy())
                    else:
                        Y = flatten_np(pd_analysis_prec.loc[
                            (pd_analysis_prec['plugin'] == method) &\
                            (pd_analysis_prec['eccentricity'] == eccentricity) &\
                            (pd_analysis_prec['resolution'] == resolution)
                        ]['precision-error'].to_numpy())
                    if method not in eccentricity_precs:
                        eccentricity_precs[method] = Y
                        prec_bins[method] = [eccentricity for i in range(len(Y))]
                    else:
                        eccentricity_precs[method] = np.concatenate((eccentricity_precs[method], Y))
                        prec_bins[method] = np.concatenate((prec_bins[method], [eccentricity for i in range(len(Y))]))
            i = 0
            for method in eccentricity_precs:
                newmeth = method
                if newmeth == 'Native':
                    newmeth = 'vanilla'
                
                if resolution is None:
                    label = "({:.1f}%) ".format(np_dropouts_plugins[newmeth]) + xlabel_dict[method]
                else:
                    label = xlabel_dict[method]

                sns.lineplot(x=prec_bins[method], y=eccentricity_precs[method], markers=True, label=label, color=colors[i-1] if i > 0 else 'blue', marker='o')
                i += 1
            if resolution is None:
                plt.title("Precision Across Eccentricities (All Resolutions)")
                plt.xlabel("Eccentricity")
                plt.ylabel("Precision Error (degrees)")
                #plt.xlim(0, 12)
                #plt.ylim(0, 12)
                plt.legend()
                plt.savefig(f'{figout_loc}ecc_separated_prec_comp.png', bbox_inches='tight')
                plt.clf()
            else:
                plt.title("Precision Across Eccentricities ({}x{})".format(resolution, resolution))
                plt.xlabel("Eccentricity")
                plt.ylabel("Precision Error (degrees)")
                #plt.xlim(0, 12)
                plt.ylim(0, 6.5 if resolution == 192 else 8.5)
                plt.legend()
                plt.savefig(f'{figout_loc}ecc_separated_prec_{resolution}_comp.png', bbox_inches='tight')
                plt.clf()
        
        # ---------- Eccentricity-separated precision comparison (subtracted by native) ----------
        for resolution in (None, 192, 400):
            eccentricity_precs = {}
            prec_bins = []
            for eccentricity in (0.0, 10.0, 15.0, 20.0):
                if resolution is None:
                    X = mean_subarrays(pd_analysis_prec.loc[
                            (pd_analysis_prec['plugin'] == 'vanilla') &\
                            (pd_analysis_prec['eccentricity'] == eccentricity)
                        ]['precision-error'].to_numpy())
                else:
                    X = mean_subarrays(pd_analysis_prec.loc[
                            (pd_analysis_prec['plugin'] == 'vanilla') &\
                            (pd_analysis_prec['eccentricity'] == eccentricity) &\
                            (pd_analysis_prec['resolution'] == resolution)
                        ]['precision-error'].to_numpy())
                if 'Native' not in eccentricity_precs:
                    eccentricity_precs['Native'] = np.array(X) - np.array(X)
                    prec_bins = [eccentricity for i in range(len(X))]
                else:
                    eccentricity_precs['Native'] = np.concatenate((eccentricity_precs['Native'],
                        np.array(X) - np.array(X)))
                    prec_bins = np.concatenate((prec_bins, [eccentricity for i in range(len(X))]))
                for method in nn_names_ecc:
                    if resolution is None:
                        Y = mean_subarrays(pd_analysis_prec.loc[
                            (pd_analysis_prec['plugin'] == method) &\
                            (pd_analysis_prec['eccentricity'] == eccentricity)
                        ]['precision-error'].to_numpy())
                    else:
                        Y = mean_subarrays(pd_analysis_prec.loc[
                            (pd_analysis_prec['plugin'] == method) &\
                            (pd_analysis_prec['eccentricity'] == eccentricity) &\
                            (pd_analysis_prec['resolution'] == resolution)
                        ]['precision-error'].to_numpy())
                    if method not in eccentricity_precs:
                        eccentricity_precs[method] = np.array(Y) - np.array(X)
                    else:
                        eccentricity_precs[method] = np.concatenate((eccentricity_precs[method],
                            np.array(Y) - np.array(X)))
            i = 0
            for method in eccentricity_precs:
                sns.lineplot(x=prec_bins, y=eccentricity_precs[method], markers=True, label=xlabel_dict[method], color=colors[i-1] if i > 0 else 'blue', marker='o')
                i += 1
            if resolution is None:
                plt.title("Precision (Comp. to Native) Across Eccentricities")
                plt.xlabel("Eccentricity")
                plt.ylabel("Precision Error Relative to Native (degrees)")
                #plt.xlim(0, 12)
                #plt.ylim(0, 12)
                plt.legend()
                plt.savefig(f'{figout_loc}ecc_separated_prec_subnative_comp.png', bbox_inches='tight')
                plt.clf()
            else:
                plt.title("Precision (Comp. to Native) Across Eccentricities ({}x{})".format(resolution, resolution))
                plt.xlabel("Eccentricity")
                plt.ylabel("Precision Error Relative to Native (degrees)")
                #plt.xlim(0, 12)
                plt.ylim(-5 if resolution == 192 else -8, 4 if resolution == 192 else 1.5)
                plt.legend()
                plt.savefig(f'{figout_loc}ecc_separated_prec_subnative_{resolution}_comp.png', bbox_inches='tight')
                plt.clf()
        
        # ----- Plot All Points' Binned&Averaged Precision Error Over Native Binned&Averaged Precision Error -----
        from decimal import Decimal
        for resolution in (None, 192, 400):
            bins = {}
            #expand_bin_size_at = 2.0
            colors = COLORS
            if resolution is None:
                X = mean_subarrays(pd_analysis_prec.loc[
                    (pd_analysis_prec['plugin'] == 'vanilla')
                ]['precision-error'].to_numpy())
            else:
                X = mean_subarrays(pd_analysis_prec.loc[
                    (pd_analysis_prec['plugin'] == 'vanilla') &\
                    (pd_analysis_prec['resolution'] == resolution)
                ]['precision-error'].to_numpy())
            plt.plot([0, np.max(X)], [0, np.max(X)], '-', label="Native Precision Error", c='blue')
            i = 0
            for method in nn_names:
                bins[method] = []
                if resolution is None:
                    vanillas = mean_subarrays(pd_analysis_prec.loc[
                        (pd_analysis_prec['plugin'] == 'vanilla')
                    ]['precision-error'].to_numpy())
                else:
                    vanillas = mean_subarrays(pd_analysis_prec.loc[
                        (pd_analysis_prec['plugin'] == 'vanilla') &\
                        (pd_analysis_prec['resolution'] == resolution)
                    ]['precision-error'].to_numpy())
                percentile50 = np.nanpercentile(vanillas, 50)
                percentile90 = np.nanpercentile(vanillas, 90)
                percentile95 = np.nanpercentile(vanillas, 95)
                expand_bin_size_at = percentile90
                max_vanillas = round(np.nanmax(vanillas), 1)  # nearest tenth place
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
                if resolution is None:
                    Y = mean_subarrays(pd_analysis_prec.loc[
                        (pd_analysis_prec['plugin'] == method)
                    ]['precision-error'].to_numpy())
                else:
                    Y = mean_subarrays(pd_analysis_prec.loc[
                        (pd_analysis_prec['plugin'] == method) &\
                        (pd_analysis_prec['resolution'] == resolution)
                    ]['precision-error'].to_numpy())
                for pt_idx in range(len(vanillas)):
                    pt = vanillas[pt_idx]
                    if pt < expand_bin_size_at + (bin_group_1_size / 2):
                        if (pt % bin_group_1_size) < (bin_group_1_size / 2):
                            modfix = float(Decimal(pt.item()) % bg1s_decimal)
                            idx = pt - modfix
                        else:
                            modfix = float(Decimal(pt.item()) % bg1s_decimal)
                            idx = pt + (bin_group_1_size - modfix)
                        idx = idx / bin_group_1_size
                        bins[method][int(round(idx))].append(Y[pt_idx])
                    else:
                        modfix = float(Decimal(pt.item()) % bg3s_decimal)
                        if modfix < (bin_group_3_size / 2):
                            idx = pt - modfix
                        else:
                            idx = pt + (bin_group_3_size - modfix)
                        #idx = round(idx)
                        idx = (expand_bin_size_at / bin_group_1_size) + ((idx - expand_bin_size_at) / bin_group_3_size)
                        if not np.isnan(idx):
                            bins[method][int(round(idx))].append(Y[pt_idx])
                currX = []
                currY = []
                for idx in range(len(bins[method])):
                    if len(bins[method][idx]):
                        if idx <= (expand_bin_size_at / bin_group_1_size):
                            bin = bin_group_1_size * idx
                        else:
                            bin = expand_bin_size_at + bin_group_3_size * (idx - (expand_bin_size_at / bin_group_1_size))
                        #currX.append(bin)
                        #currY.append(np.mean(bins[method][idx]))
                        for currpt in bins[method][idx]:
                            currX.append(bin)
                            currY.append(currpt)
                #plt.plot(X, Y, '-o', markersize=5, label=method, c=colors[i])
                sns.lineplot(x=currX, y=currY, markers=True, label=method, color=colors[i], marker='o')
                i += 1
            
            if resolution is None:
                plt.title("NN-assisted Binned Mean Precision Error vs Native Precision Error")
                plt.xlabel("Native Precision Error (degrees) (bin interval = {} -> {})".format(bin_group_1_size, bin_group_3_size))
                plt.ylabel("NN-assisted Precision Error (degrees)")
                #plt.xlim(0, 12)
                #plt.ylim(0, 12)
                plt.axvline(percentile50, linestyle=":", color="blue", label="50th percentile")
                plt.axvline(percentile90, linestyle=":", color="green", label="90th percentile")
                plt.axvline(percentile95, linestyle=":", color="red", label="95th percentile")
                plt.legend()
                #plt.xlim(0, percentile90)
                #plt.ylim(0, 10)
                plt.savefig(f'{figout_loc}VANIL_PREC_BINNED_COMP.png', bbox_inches='tight')
                plt.xlim(0, percentile90)
                plt.ylim(0, 10)
                plt.savefig(f'{figout_loc}VANIL_PREC_BINNED_LIM_PERCENTILE90_COMP.png', bbox_inches='tight')
                plt.clf()
            else:
                plt.title("NN-assisted Binned Mean Precision Error vs Native Precision Error ({}x{})".format(resolution, resolution))
                plt.xlabel("Native Precision Error (degrees) (bin interval = {} -> {})".format(bin_group_1_size, bin_group_3_size))
                plt.ylabel("NN-assisted Precision Error (degrees)")
                #plt.xlim(0, 12)
                #plt.ylim(0, 12)
                plt.axvline(percentile50, linestyle=":", color="blue", label="50th percentile")
                plt.axvline(percentile90, linestyle=":", color="green", label="90th percentile")
                plt.axvline(percentile95, linestyle=":", color="red", label="95th percentile")
                plt.legend()
                #plt.xlim(0, percentile90)
                plt.ylim(0, 75 if resolution == 192 else 70)
                plt.savefig(f'{figout_loc}VANIL_PREC_BINNED_{resolution}_COMP.png', bbox_inches='tight')
                plt.xlim(0, percentile90)
                plt.ylim(0, 10)
                plt.savefig(f'{figout_loc}VANIL_PREC_BINNED_LIM_PERCENTILE90_{resolution}_COMP.png', bbox_inches='tight')
                plt.clf()

        # ----- Plot Generalized Accuracy Information For Native And Each NN -----
        colors = COLORS
        labels = np.concatenate((["Native"], [xlabel_dict[k] for k in nn_names]))
        # --------------------------------------------------------------------------------------
        
        def generate_summary_boxplot(X, Y, labels, barlabels, xlabels, ylabel, title, filename, ylimit, xlabel, Z, override_plotsize, group_size):
            from pylab import plot, show, savefig, xlim, figure, \
                        ylim, legend, boxplot, setp, axes
            from matplotlib.cbook import boxplot_stats
            color_list = np.concatenate((['blue'], COLORS))
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
            
            # boxplot pair
            bpX = X[~np.isnan(X)]
            bpY = Y[~np.isnan(Y)]
            bpZ = [z[~np.isnan(z)] for z in Z]
            bp = ax.boxplot([bpX, bpY, *bpZ], positions = [I for I in range(num_boxes)], widths = 0.6, showfliers=False)
            boxplots.append(bp)
            setBoxColors(bp, color_count=num_boxes)
            
            mass_medians[0].append(np.median(X[~np.isnan(X)]))
            mass_medians[1].append(np.median(Y[~np.isnan(Y)]))
            for zed in range(len(Z)):
                z = Z[zed]
                mass_medians[zed+2].append(np.median(z[~np.isnan(z)]))
            
            Xs[0].append(0)
            Xs[1].append(1)
            for zed in range(len(Z)):
                Xs[zed+2].append(zed+2)
            
            plots = []
            for i in range(len(mass_medians)):
                for j in range(len(Xs[0])):
                    h, = plt.plot([Xs[0][j], Xs[0][j]], [mass_medians[i][j], mass_medians[i][j]], '-', c=color_list[i])
           
                #h, = plt.plot(Xs[i], mass_medians[i], '-', c=color_list[i])
                plots.append(h)

            print(f"len Xs: {len(Xs)}, len X[0]: {len(Xs[0])}, j: {j}, len(mass_medians): {len(mass_medians)}")
            print(f"group_size: {group_size}")
            print(f"(len(mass_medians) - 1) - ((len(mass_medians) - 1) % group_size) + group_size - 1: {(len(mass_medians) - 1) - ((len(mass_medians) - 1) % group_size) + group_size - 1}")
            plt.plot([Xs[0][j], Xs[(len(mass_medians) - 1) - ((len(mass_medians) - 1) % group_size) + group_size - 1][j]], [mass_medians[0][j], mass_medians[0][j]], '--', c=color_list[0], linewidth=1)

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
            #ax.set_xticks(np.arange(len(labels))*(num_boxes+1) + int(np.floor((num_boxes+1)/2)))
            ax.set_xticklabels(xlabels)
            
            ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)
            #ax.tick_params(axis='x', which='major', labelsize=10)
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(9)
            #ax.xaxis.set_label_coords(-0.1, -0.1)
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
            legend(plots, xlabels)
            #if override_plotsize:
            #    savefig(filename, bbox_inches='tight')
            #else:
            #    savefig(filename)
            savefig(filename, bbox_inches='tight')
            plt.clf()
        # --------------------------------------------------------------------------------------
        
        X = flatten_np(pd_analysis_acc.loc[
            (pd_analysis_acc['plugin'] == 'vanilla')
        ]['accuracy-error'].to_numpy())
        Y = flatten_np(pd_analysis_acc.loc[
            (pd_analysis_acc['plugin'] == nn_names[0])
        ]['accuracy-error'].to_numpy())
        labels = np.concatenate((["Native"], [xlabel_dict[k] for k in nn_names]))
        barlabels = np.concatenate((["Native"], [barlabel_dict[k] for k in nn_names]))
        #xlabels = ['Native', 'EllSeg', 'ESFnet', 'RITnet Pupil']
        xlabels = np.concatenate((["Native"], [xlabel_dict[k] for k in nn_names]))
        ylabel = 'Accuracy Error (degrees)'
        title = 'Gaze Accuracy Errors Across Neural Networks'
        filename = f'{figout_loc}Generalized Accuracy TRUE.png'
        ylimit = None
        xlabel = 'Method'
        Z = [flatten_np(pd_analysis_acc.loc[
                (pd_analysis_acc['plugin'] == k)
            ]['accuracy-error'].to_numpy()) for k in nn_names[1:]]
        override_plotsize = False
        group_size = 2

        for resolution in (192, 400):
            fname = f'{figout_loc}/out_data_robustness_{resolution}.csv'
            with open(fname, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['Subject', 'Plugin', 'nanmean', 'nanmedian', 'nanstd'])
                grouped_dropouts_by_subject[192, 1, 'Detector2DESFnetPlugin']
                for subject in SUBJECTS:
                    Native = grouped_dropouts_by_subject[resolution, subject, 'vanilla']
                    EllSeg = grouped_dropouts_by_subject[resolution, subject, 'Detector2DRITnetEllsegV2AllvonePlugin']
                    ESFnet = grouped_dropouts_by_subject[resolution, subject, 'Detector2DESFnetPlugin']
                    ESFnetEmbeddedPupil = grouped_dropouts_by_subject[resolution, subject, 'Detector2DESFnetEmbeddedPlugin']
                    RITnetPupil = grouped_dropouts_by_subject[resolution, subject, 'Detector2DRITnetPupilPlugin']
                    EllSegEmbeddedIris = grouped_dropouts_by_subject[resolution, subject, 'Detector2DRITnetEllsegV2AllvoneEmbeddedIrisPlugin']
                    EllSegEmbeddedPupil = grouped_dropouts_by_subject[resolution, subject, 'Detector2DRITnetEllsegV2AllvoneEmbeddedPlugin']
                    writer.writerow([subject, 'Native',np.nanmean(Native),np.nanmedian(Native),np.nanstd(Native)])
                    writer.writerow([subject, 'EllSeg',np.nanmean(EllSeg),np.nanmedian(EllSeg),np.nanstd(EllSeg)])
                    writer.writerow([subject, 'EllSeg (Embedded Pupil)',np.nanmean(EllSegEmbeddedPupil),np.nanmedian(EllSegEmbeddedPupil),np.nanstd(EllSegEmbeddedPupil)])
                    writer.writerow([subject, 'EllSeg (Embedded Iris)',np.nanmean(EllSegEmbeddedIris),np.nanmedian(EllSegEmbeddedIris),np.nanstd(EllSegEmbeddedIris)])
                    writer.writerow([subject, 'ESFnet',np.nanmean(ESFnet),np.nanmedian(ESFnet),np.nanstd(ESFnet)])
                    writer.writerow([subject, 'ESFnet (Embedded Pupil)',np.nanmean(ESFnetEmbeddedPupil),np.nanmedian(ESFnetEmbeddedPupil),np.nanstd(ESFnetEmbeddedPupil)])
                    writer.writerow([subject, 'RITnet (Pupil)',np.nanmean(RITnetPupil),np.nanmedian(RITnetPupil),np.nanstd(RITnetPupil)])

        for resolution in (None, 192, 400):
            if resolution is None:
                fname = f'{figout_loc}/out_data.csv'
            else:
                fname = f'{figout_loc}/out_data_{resolution}.csv'
            with open(fname, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['Subject', 'Plugin', 'nanmean', 'nanmedian', 'nanstd'])
                for subject in SUBJECTS:
                    if resolution is None:
                        Native = mean_subarrays(pd_analysis_acc.loc[
                            (pd_analysis_acc['plugin'] == 'vanilla') &\
                            (pd_analysis_acc['subject'] == subject)
                        ]['accuracy-error'].to_numpy())
                        EllSeg = mean_subarrays(pd_analysis_acc.loc[
                            (pd_analysis_acc['plugin'] == 'Detector2DRITnetEllsegV2AllvonePlugin') &\
                            (pd_analysis_acc['subject'] == subject)
                        ]['accuracy-error'].to_numpy())
                        ESFnet = mean_subarrays(pd_analysis_acc.loc[
                            (pd_analysis_acc['plugin'] == 'Detector2DESFnetPlugin') &\
                            (pd_analysis_acc['subject'] == subject)
                        ]['accuracy-error'].to_numpy())
                        ESFnetEmbeddedPupil = mean_subarrays(pd_analysis_acc.loc[
                            (pd_analysis_acc['plugin'] == 'Detector2DESFnetEmbeddedPlugin') &\
                            (pd_analysis_acc['subject'] == subject)
                        ]['accuracy-error'].to_numpy())
                        RITnetPupil = mean_subarrays(pd_analysis_acc.loc[
                            (pd_analysis_acc['plugin'] == 'Detector2DRITnetPupilPlugin') &\
                            (pd_analysis_acc['subject'] == subject)
                        ]['accuracy-error'].to_numpy())
                        EllSegEmbeddedIris = mean_subarrays(pd_analysis_acc.loc[
                            (pd_analysis_acc['plugin'] == 'Detector2DRITnetEllsegV2AllvoneEmbeddedIrisPlugin') &\
                            (pd_analysis_acc['subject'] == subject)
                        ]['accuracy-error'].to_numpy())
                        EllSegEmbeddedPupil = mean_subarrays(pd_analysis_acc.loc[
                            (pd_analysis_acc['plugin'] == 'Detector2DRITnetEllsegV2AllvoneEmbeddedPlugin') &\
                            (pd_analysis_acc['subject'] == subject)
                        ]['accuracy-error'].to_numpy())
                    else:
                        Native = mean_subarrays(pd_analysis_acc.loc[
                            (pd_analysis_acc['plugin'] == 'vanilla') &\
                            (pd_analysis_acc['subject'] == subject) &\
                            (pd_analysis_acc['resolution'] == resolution)
                        ]['accuracy-error'].to_numpy())
                        EllSeg = mean_subarrays(pd_analysis_acc.loc[
                            (pd_analysis_acc['plugin'] == 'Detector2DRITnetEllsegV2AllvonePlugin') &\
                            (pd_analysis_acc['subject'] == subject) &\
                            (pd_analysis_acc['resolution'] == resolution)
                        ]['accuracy-error'].to_numpy())
                        ESFnet = mean_subarrays(pd_analysis_acc.loc[
                            (pd_analysis_acc['plugin'] == 'Detector2DESFnetPlugin') &\
                            (pd_analysis_acc['subject'] == subject) &\
                            (pd_analysis_acc['resolution'] == resolution)
                        ]['accuracy-error'].to_numpy())
                        ESFnetEmbeddedPupil = mean_subarrays(pd_analysis_acc.loc[
                            (pd_analysis_acc['plugin'] == 'Detector2DESFnetEmbeddedPlugin') &\
                            (pd_analysis_acc['subject'] == subject) &\
                            (pd_analysis_acc['resolution'] == resolution)
                        ]['accuracy-error'].to_numpy())
                        RITnetPupil = mean_subarrays(pd_analysis_acc.loc[
                            (pd_analysis_acc['plugin'] == 'Detector2DRITnetPupilPlugin') &\
                            (pd_analysis_acc['subject'] == subject) &\
                            (pd_analysis_acc['resolution'] == resolution)
                        ]['accuracy-error'].to_numpy())
                        EllSegEmbeddedIris = mean_subarrays(pd_analysis_acc.loc[
                            (pd_analysis_acc['plugin'] == 'Detector2DRITnetEllsegV2AllvoneEmbeddedIrisPlugin') &\
                            (pd_analysis_acc['subject'] == subject) &\
                            (pd_analysis_acc['resolution'] == resolution)
                        ]['accuracy-error'].to_numpy())
                        EllSegEmbeddedPupil = mean_subarrays(pd_analysis_acc.loc[
                            (pd_analysis_acc['plugin'] == 'Detector2DRITnetEllsegV2AllvoneEmbeddedPlugin') &\
                            (pd_analysis_acc['subject'] == subject) &\
                            (pd_analysis_acc['resolution'] == resolution)
                        ]['accuracy-error'].to_numpy())
                    writer.writerow([subject, 'Native',np.nanmean(Native),np.nanmedian(Native),np.nanstd(Native)])
                    writer.writerow([subject, 'EllSeg',np.nanmean(EllSeg),np.nanmedian(EllSeg),np.nanstd(EllSeg)])
                    writer.writerow([subject, 'EllSeg (Embedded Pupil)',np.nanmean(EllSegEmbeddedPupil),np.nanmedian(EllSegEmbeddedPupil),np.nanstd(EllSegEmbeddedPupil)])
                    writer.writerow([subject, 'EllSeg (Embedded Iris)',np.nanmean(EllSegEmbeddedIris),np.nanmedian(EllSegEmbeddedIris),np.nanstd(EllSegEmbeddedIris)])
                    writer.writerow([subject, 'ESFnet',np.nanmean(ESFnet),np.nanmedian(ESFnet),np.nanstd(ESFnet)])
                    writer.writerow([subject, 'ESFnet (Embedded Pupil)',np.nanmean(ESFnetEmbeddedPupil),np.nanmedian(ESFnetEmbeddedPupil),np.nanstd(ESFnetEmbeddedPupil)])
                    writer.writerow([subject, 'RITnet (Pupil)',np.nanmean(RITnetPupil),np.nanmedian(RITnetPupil),np.nanstd(RITnetPupil)])

        for resolution in (None, 192, 400):
            if resolution is None:
                fname = f'{figout_loc}/out_data_precision.csv'
            else:
                fname = f'{figout_loc}/out_data_precision_{resolution}.csv'
            with open(fname, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['Subject', 'Plugin', 'nanmean', 'nanmedian', 'nanstd'])
                for subject in SUBJECTS:
                    if resolution is None:
                        Native = mean_subarrays(pd_analysis_prec.loc[
                            (pd_analysis_prec['plugin'] == 'vanilla') &\
                            (pd_analysis_prec['subject'] == subject)
                        ]['precision-error'].to_numpy())
                        EllSeg = mean_subarrays(pd_analysis_prec.loc[
                            (pd_analysis_prec['plugin'] == 'Detector2DRITnetEllsegV2AllvonePlugin') &\
                            (pd_analysis_prec['subject'] == subject)
                        ]['precision-error'].to_numpy())
                        ESFnet = mean_subarrays(pd_analysis_prec.loc[
                            (pd_analysis_prec['plugin'] == 'Detector2DESFnetPlugin') &\
                            (pd_analysis_prec['subject'] == subject)
                        ]['precision-error'].to_numpy())
                        ESFnetEmbeddedPupil = mean_subarrays(pd_analysis_prec.loc[
                            (pd_analysis_prec['plugin'] == 'Detector2DESFnetEmbeddedPlugin') &\
                            (pd_analysis_prec['subject'] == subject)
                        ]['precision-error'].to_numpy())
                        RITnetPupil = mean_subarrays(pd_analysis_prec.loc[
                            (pd_analysis_prec['plugin'] == 'Detector2DRITnetPupilPlugin') &\
                            (pd_analysis_prec['subject'] == subject)
                        ]['precision-error'].to_numpy())
                        EllSegEmbeddedIris = mean_subarrays(pd_analysis_prec.loc[
                            (pd_analysis_prec['plugin'] == 'Detector2DRITnetEllsegV2AllvoneEmbeddedIrisPlugin') &\
                            (pd_analysis_prec['subject'] == subject)
                        ]['precision-error'].to_numpy())
                        EllSegEmbeddedPupil = mean_subarrays(pd_analysis_prec.loc[
                            (pd_analysis_prec['plugin'] == 'Detector2DRITnetEllsegV2AllvoneEmbeddedPlugin') &\
                            (pd_analysis_prec['subject'] == subject)
                        ]['precision-error'].to_numpy())
                    else:
                        Native = mean_subarrays(pd_analysis_prec.loc[
                            (pd_analysis_prec['plugin'] == 'vanilla') &\
                            (pd_analysis_prec['subject'] == subject) &\
                            (pd_analysis_prec['resolution'] == resolution)
                        ]['precision-error'].to_numpy())
                        EllSeg = mean_subarrays(pd_analysis_prec.loc[
                            (pd_analysis_prec['plugin'] == 'Detector2DRITnetEllsegV2AllvonePlugin') &\
                            (pd_analysis_prec['subject'] == subject) &\
                            (pd_analysis_prec['resolution'] == resolution)
                        ]['precision-error'].to_numpy())
                        ESFnet = mean_subarrays(pd_analysis_prec.loc[
                            (pd_analysis_prec['plugin'] == 'Detector2DESFnetPlugin') &\
                            (pd_analysis_prec['subject'] == subject) &\
                            (pd_analysis_prec['resolution'] == resolution)
                        ]['precision-error'].to_numpy())
                        ESFnetEmbeddedPupil = mean_subarrays(pd_analysis_prec.loc[
                            (pd_analysis_prec['plugin'] == 'Detector2DESFnetEmbeddedPlugin') &\
                            (pd_analysis_prec['subject'] == subject) &\
                            (pd_analysis_prec['resolution'] == resolution)
                        ]['precision-error'].to_numpy())
                        RITnetPupil = mean_subarrays(pd_analysis_prec.loc[
                            (pd_analysis_prec['plugin'] == 'Detector2DRITnetPupilPlugin') &\
                            (pd_analysis_prec['subject'] == subject) &\
                            (pd_analysis_prec['resolution'] == resolution)
                        ]['precision-error'].to_numpy())
                        EllSegEmbeddedIris = mean_subarrays(pd_analysis_prec.loc[
                            (pd_analysis_prec['plugin'] == 'Detector2DRITnetEllsegV2AllvoneEmbeddedIrisPlugin') &\
                            (pd_analysis_prec['subject'] == subject) &\
                            (pd_analysis_prec['resolution'] == resolution)
                        ]['precision-error'].to_numpy())
                        EllSegEmbeddedPupil = mean_subarrays(pd_analysis_prec.loc[
                            (pd_analysis_prec['plugin'] == 'Detector2DRITnetEllsegV2AllvoneEmbeddedPlugin') &\
                            (pd_analysis_prec['subject'] == subject) &\
                            (pd_analysis_prec['resolution'] == resolution)
                        ]['precision-error'].to_numpy())
                    writer.writerow(['SUBJECT {}'.format(subject)])
                    writer.writerow(['Native',np.nanmean(Native),np.nanmedian(Native),np.nanstd(Native)])
                    writer.writerow(['EllSeg',np.nanmean(EllSeg),np.nanmedian(EllSeg),np.nanstd(EllSeg)])
                    writer.writerow(['EllSeg (Embedded Pupil)',np.nanmean(EllSegEmbeddedPupil),np.nanmedian(EllSegEmbeddedPupil),np.nanstd(EllSegEmbeddedPupil)])
                    writer.writerow(['RITnet (Embedded Iris)',np.nanmean(EllSegEmbeddedIris),np.nanmedian(EllSegEmbeddedIris),np.nanstd(EllSegEmbeddedIris)])
                    writer.writerow(['ESFnet',np.nanmean(ESFnet),np.nanmedian(ESFnet),np.nanstd(ESFnet)])
                    writer.writerow(['ESFnet (Embedded Pupil)',np.nanmean(ESFnetEmbeddedPupil),np.nanmedian(ESFnetEmbeddedPupil),np.nanstd(ESFnetEmbeddedPupil)])
                    writer.writerow(['RITnet (Pupil)',np.nanmean(RITnetPupil),np.nanmedian(RITnetPupil),np.nanstd(RITnetPupil)])
        print("All done.")
        exit()
        

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
    subjects = SUBJECTS[1:]#[2, 3, 5, 6, 7, 8, 9, 10, 11]

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
    fig.savefig(f'{figout_loc}Velocity/out_{subID}_{label}.png')
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
    fig.write_html(f'{figout_loc}HTML/out_{subID}_{label}.html')

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
    fig.savefig(f'{figout_loc}Velocity/out_fil_{subID}_mean{round(np.mean(newthing), 2)}_{label}.png')

    plt.clf()
    plt.close(fig)

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
    fig.savefig(f'{figout_loc}Velocity/out_fil_{subID}_{label}_CalibOverlap.png')
    
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
    fig.savefig(f'{figout_loc}Velocity/out_fil_{subID}_{label}_AssessOverlap.png')
    plt.close(fig)


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
            
            setp(bp['caps'][i*2], color=color_list[i])
            setp(bp['caps'][i*2+1], color=color_list[i])
            
            setp(bp['whiskers'][i*2], color=color_list[i])
            setp(bp['whiskers'][i*2+1], color=color_list[i])
            try:
                setp(bp['fliers'][i], color='black')
            except:
                pass
            setp(bp['medians'][i], color=color_list[i])

    if override_plotsize:
        fig = figure(figsize=(20,15))
    else:
        fig = figure()
    ax = axes()
    
    num_boxes = 2 + len(Z)
    Xs = [[] for i in range(num_boxes)]
    boxplots = []
    mass_medians = [[] for i in range(num_boxes)]
    
    for i in range(0, len(X)):
        # boxplot pair
        bpX = X[i][~np.isnan(X[i])]
        bpY = Y[i][~np.isnan(Y[i])]
        bpZ = [z[i][~np.isnan(z[i])] for z in Z]
        bp = ax.boxplot([bpX, bpY, *bpZ], positions = [i*(num_boxes+1)+I for I in range(num_boxes)], widths = 0.6, showfliers=False)
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

    # set axes limits and labels
    if ylimit:
        ylim(0, ylimit)
    ax.set_xticks(np.arange(len(labels))*(num_boxes+1) + int(np.floor((num_boxes+1)/2)))
    ax.set_xticklabels(labels)
    
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    
    legend(plots, barlabels)
    if override_plotsize:
        savefig(filename, bbox_inches='tight')
    else:
        savefig(filename)

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
