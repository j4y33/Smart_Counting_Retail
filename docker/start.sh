#!/bin/bash

dropbox_folder=/home/minisio1/Dropbox
workspace_folder=/home/minisio1/workspace
path_into_docker_script=/workspace/camai/smart-counter-solid/smart-counter-reid/scripts/run.sh

export PYTHONPATH=/workspace/camai/smart-counter-solid/smart-counter-reid
video_folder=/workspace/video-record/dvr/minio/minisio
export SMART_COUNTER_ASSETS=$PYTHONPATH/assets
output=$PYTHONPATH/res
export SMART_COUNTER_PATH_FOR_DYNACONF=$PYTHONPATH/configs/production
export SMART_COUNTER_VIDEO_SOURCE__CONFIG__TIME_LIMIT_HOURS=12
export SMART_COUNTER_LOGGING__LEVEL="INFO"

#export SMART_COUNTER_VIDEO_SOURCE__TYPE="FFmpegVideoSource"

cameras=(entrance cash-desk)
year=2021
month=6
day_start=1
day_end=3
start_time='09:00:00'
for ((day=$day_start; day<=$day_end; day++))
do
   for camera in "${cameras[@]}"
   do
      export SMART_COUNTER_CAMERA_ID=$camera
      output_folder=$(date -d $year-$month-$day +"%Y%m%d")
      export SMART_COUNTER_OUTPUT_FOLDER=$output_folder
      export SMART_COUNTER_OUTPUTS=$output/$output_folder/$camera
      #export SMART_COUNTER_VIDEO_SOURCE__CONFIG__SOURCE=/workspace/t/2020-12-12-15-00-03_2020-12-12-16-59-22_cash-desk.mp4
      export SMART_COUNTER_VIDEO_SOURCE__CONFIG__SOURCE=$video_folder/$camera
      start_datetime=$(date -d $year-$month-$day' '$start_time +"%F %T")
      export SMART_COUNTER_VIDEO_SOURCE__CONFIG__START_TIME=$start_datetime
      docker run --rm --env-file <(env | grep -e PYTHONPATH -e SMART_COUNTER) --memory="8g" --name camai_running  --runtime=nvidia -v /etc/timezone:/etc/timezone -v /media:/media -v $dropbox_folder:/dropbox -v /mnt:/storage -v $workspace_folder:/workspace camai_run_day_solid bash $path_into_docker_script
      rsync -avh -e "ssh -i $HOME/.ssh/camai_key" /mnt/smart-counter/entrance/ pi@10.147.17.161:/home/pi/camai/reporting/stores
   done
done

