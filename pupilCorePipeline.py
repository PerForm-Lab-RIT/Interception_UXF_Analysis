import os
import sys
import click
import csv
import importlib

import logging
import pickle

from dotenv import load_dotenv

import numpy as np

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
def main(allow_session_loading, skip_pupil_detection, vanilla_only, skip_vanilla, surpress_runtimewarnings, load_2d_pupils, min_calibration_confidence, show_filtered_out, display_world_video, core_shared_modules_loc, pipeline_loc, plugins_file):
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
    
    
if __name__ == "__main__":
    load_dotenv()
    main()
