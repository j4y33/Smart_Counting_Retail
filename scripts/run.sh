#!/bin/bash

run_path=$PYTHONPATH/run.py
traffic_script_path=$PYTHONPATH/scripts/tracks_to_traffic.py
copy_script_path=$PYTHONPATH/scripts/copy_results.py
tracks_path=$SMART_COUNTER_OUTPUTS/tracks.json
traffic_result_path=$SMART_COUNTER_OUTPUTS/traffic.csv
filters_path=$SMART_COUNTER_ASSETS/filter_params/params_$SMART_COUNTER_CAMERA_ID.json

dropbox_folder=/dropbox/stores/chausse-dantin-1/$SMART_COUNTER_OUTPUT_FOLDER/$SMART_COUNTER_CAMERA_ID
reports_folder=/storage/smart-counter/entrance/chausse-dantin-1/$SMART_COUNTER_OUTPUT_FOLDER/$SMART_COUNTER_CAMERA_ID

ROOT_PATH_FOR_DYNACONF=$SMART_COUNTER_PATH_FOR_DYNACONF python3 $run_path
python3 $traffic_script_path --tracks $tracks_path --mode $SMART_COUNTER_CAMERA_ID --output $traffic_result_path --params $filters_path
python3 $copy_script_path --traffic $traffic_result_path --processed_videos $SMART_COUNTER_OUTPUTS/processed_videos.txt --tracks $tracks_path --dropbox_folder $dropbox_folder --reports_folder $reports_folder