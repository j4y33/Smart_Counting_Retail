
### Installation

1. Setup SSH keys to DVC host (no host defined, host needs to be created)
- Additional.[Setup DVC](https://github.com/camai-pro/org/blob/main/dvc.md#sftp)

1.1. Setup Docker
- skip this if you want to install app without docker
```
cd docker
docker build -t camai:u18-gst14-tf14-py3 .

cd ..
docker run --runtime=nvidia --name camai-dev \
    -e DISPLAY=$DISPLAY -it -v /tmp/.X11-unix/:/tmp/.X11-unix \
    -v ${pwd):/workspace/camai/smart-counter-reid camai:u18-gst14-tf14-py3

# copy ssh keys
docker cp ~/.ssh/id_rsa camai-dev:/root/.ssh/id_rsa
```

2. Setup APP
- locally or inside docker container
```

make install
make venv
source venv/bin/activate

make update
```


### Usage
```
export SMART_COUNTER_VIDEO_SOURCE__SOURCE="my_video.mp4"
ROOT_PATH_FOR_DYNACONF=$PWD/configs python run.py
```

### Notes
#### Configuration
- [dynaconf envvars](https://www.dynaconf.com/envvars/)
```
export SMART_COUNTER_VIDEO_SOURCE__SOURCE="my_video.mp4"
=
settings.video_source.source="my_video.mp4"

```

#### Enable Display
- if you use docker container, run next command from host
```
# to enable UI
xhost +local:

# enable display in APP
export SMART_COUNTER_UNITS__DATA_OVERLAY__DISPLAY__ENABLED=true
```