from smart_counter.sources import OpenCVVideoSource, FFmpegVideoSource
from smart_counter.models import PersonDetectron2, PersonEmbeddingTorchReid
from smart_counter.base import Model, Unit, Frame, VideoSource, read_source_info, SourceInfo
from smart_counter.units import PersonDetector, PersonEmbedding, Tracker, TracksFilePublisher
from smart_counter.structs import KeyPoint, BoundingBox, Detection, Track
import numpy as np
import os
import subprocess
import cv2
import mock
import pytest
import typing as typ
from fractions import Fraction
import datetime as dt
import json
import shutil
import pytz


@pytest.fixture(scope="session", autouse=True)
def init(request):
    """
    create video for tests before all tests and delete it after all test

    Parameters
    ----------
    request
        pytest var, used for possibility to add finalizer
    """
    gst_launch_str = 'gst-launch-1.0 videotestsrc num-buffers=100 ! x264enc tune=zerolatency pass=quant\
                        ! mp4mux ! filesink location=smoke_video.mp4 -e'
    stdout, stderr = subprocess.Popen(gst_launch_str.split(
        ' '), stdout=subprocess.PIPE, stderr=subprocess.STDOUT).communicate()
    request.addfinalizer(clear)


def clear():
    """
    delete test video
    """
    os.remove("smoke_video.mp4")


"""
first tulpe for test all frames getting with FFmpegVideoSource
second for test all frames getting with OpenCVVideoSource
third for test getting frames with FFmpegVideoSource with other fps
"""
test_sources_params = [
    (FFmpegVideoSource, 'smoke_video.mp4', {'fps': 30}),
    (OpenCVVideoSource, 'smoke_video.mp4', {'fps': 30}),
    (FFmpegVideoSource, 'smoke_video.mp4', {'fps': 1})
]


@pytest.mark.parametrize("source_class,video_file, config", test_sources_params)
def test_sources(source_class: typ.Type[VideoSource], video_file: str, config: dict):
    """
    tests for FFmpegVideoSource and OpenCVVideoSource

    Parameters
    ----------
    source_class
        class to test
    fps
        original video fps
    num_frames
        number of frames in original video
    video_file
        path to video file for tests
    config
        dict with config for source_class
    """
    real_duration_msec = 1000/config['fps']
    with source_class(video_file, config) as source:
        while True:
            frame = source.read()
            if frame is None:
                break
            assert frame.duration_msec == real_duration_msec, f"Expected duration_msec\
                 for frame {frame.duration_msec} != {real_duration_msec} actual duration_msec for frame"


test_model_params = [
    (PersonDetectron2, {'weights': 'assets/models/configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml',
                        'detector_threshold': 0.9},
     'assets/images/one_person.jpg')
]


@pytest.mark.parametrize("model_class,config,img", test_model_params)
def test_detectron_model(model_class: typ.Type[Model], config: dict, img: str):
    """
    test model predictions

    Parameters
    ----------
    model_class
        model_class
    config
        config dict for model
    img
        path to test img (not autocreated, need to contains person)
    """
    frame = cv2.imread(img)
    with model_class(config) as detector:
        res = detector.process_single(frame)
        assert len(res) > 0, f"Expected that model should find some detections on image"
        for el in res:
            assert type(el['box']) is BoundingBox, f"Expected type of box is BoundingBox,\
                but {type(el['box'])} is actual"
            assert type(el['keypoints']) is dict, f"Expected type of keypoints is dict,\
                but {type(el['keypoints'])} is actual"
            assert type(el['keypoints']['nose']) is KeyPoint, f"Expected type of keypoint is KeyPoint,\
                but {type(el['keypoints']['nose'])} is actual"
            assert el['score'] > config['detector_threshold'], f"Expected score \
                is bigger or equal with {config['detector_threshold']},\
                but actual is less {el['score']}"


test_extractor_params = [
    (PersonEmbeddingTorchReid, {'weights': 'assets/models/osnet_ain.pth'}, 'assets/images/one_person.jpg')
]


@pytest.mark.parametrize("model_class,config,img", test_extractor_params)
def test_extractor_model(model_class: typ.Type[Model], config: dict, img: str):
    """
    test model predictions

    Parameters
    ----------
    model_class
        model_class
    config
        config dict for model
    img
        path to test img
    """
    expected_shape = (512,)
    frame = cv2.imread(img)
    with model_class(config) as extractor:
        res = extractor.process_single(frame)
        assert res.shape == expected_shape, f"Result is expected to have shape {expected_shape},\
             but actual shape is {res.shape}"


def mocked_read(frame: np.ndarray) -> np.ndarray:
    """
    mocked method

    Returns
    -------
        100x100 zeros matrix
    """

    return Frame(frame=np.ones((100, 100, 3)),
                 offset=1,
                 counter=1,
                 stream_id=1,
                 width=100,
                 height=100,
                 bpp=None,
                 fps=1,
                 pix_fmt=None,
                 command=None,
                 start_time=dt.datetime(2000, 1, 1, 1, 1, 1),
                 source=SourceInfo(
        source='/my_video',                 # type: str
        width=100,                # type: int
        height=100,                # type: int
        fps=1
    ))


def mocked_process_single(image: np.ndarray) -> typ.List[typ.Any]:
    """
    mocked method

    Parameters
    ----------
    image
        original image

    Returns
    -------
    typ.List[]
        empty list
    """
    return []


test_detector_params = [
    (PersonDetector, PersonDetectron2,
     {'weights': 'assets/models/configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml', 'threshold': 0.9})
]


@mock.patch.object(Model, 'process_single', mocked_process_single)
@mock.patch.object(OpenCVVideoSource, 'read', mocked_read)
@mock.patch.object(OpenCVVideoSource, 'startup', lambda x: None)
@mock.patch.object(OpenCVVideoSource, 'shutdown', lambda x: None)
@pytest.mark.parametrize("detector_class,model_class,config", test_detector_params)
def test_detector(detector_class: typ.Type[Unit], model_class: typ.Type[Model], config: dict):
    """
    test detector unit
    mocked Model.process_single and VideoSource.read to test only detector_class

    Args:
        detector_class : detector class name
        model_class : model class name
        config : config dict for model
    """
    with model_class(config) as model:
        person_detector = detector_class(None, {'models': {'person_detector': model}})
        data = {}
        with OpenCVVideoSource('None') as source:
            while True:
                frame = source.read()
                person_detector.process(frame, data)
                break
        expected_key = 'persons'
        assert expected_key in data, f"Expected key '{expected_key}' in dict as result of processing"


test_extractor_params = [
    (PersonEmbedding, PersonEmbeddingTorchReid,
     {'weights': 'assets/models/osnet_ain.pth'})
]


@mock.patch.object(Model, 'process_single', mocked_process_single)
@mock.patch.object(OpenCVVideoSource, 'read', mocked_read)
@mock.patch.object(OpenCVVideoSource, 'startup', lambda x: None)
@mock.patch.object(OpenCVVideoSource, 'shutdown', lambda x: None)
@pytest.mark.parametrize("detector_class,model_class,config", test_extractor_params)
def test_extractor(detector_class: typ.Type[Unit], model_class: typ.Type[Model], config: dict):
    """
    test extractor unit
    mocked Model.process_single and VideoSource.read to test only detector_class

    Args:
        detector_class : detector class name
        model_class : model class name
        config : config dict for model
    """
    with model_class(config) as model:
        person_detector = detector_class(None, {'models': {'person_embedding': model}})
        data = {'persons': [Detection(
                box=BoundingBox(20, 20, 20, 20),
                score=float(0.96),
                keypoints={},
                side='front'
                )]}
        with OpenCVVideoSource('None') as source:
            while True:
                frame = source.read()
                person_detector.process(frame, data)
                break

        expected_key = 'persons_embeddings'
        assert expected_key in data, f"Expected key '{expected_key}' in dict as result of processing"
        expected_len = 1
        embeddings_actual_num = len(data[expected_key])
        assert embeddings_actual_num == expected_len, f"Excepted to extract {expected_len} embeddings,\
             but {embeddings_actual_num} embeddings were extracted"


test_tracker_params = [
    (Tracker, {'min_track_length': 2, 'max_embeddings_size': 1})
]


@mock.patch.object(OpenCVVideoSource, 'read', mocked_read)
@mock.patch.object(OpenCVVideoSource, 'startup', lambda x: None)
@mock.patch.object(OpenCVVideoSource, 'shutdown', lambda x: None)
@pytest.mark.parametrize("tracker_class,config", test_tracker_params)
def test_tracker(tracker_class: typ.Type[Unit], config: dict):
    """
    test Tracker unit
    mocked VideoSource.read to test only tracker_class

    Parameters
    ----------
    tracker_class
        tracker class to test
    config
        config dict
    """
    imitate_detection = Detection(
        box=BoundingBox(20, 20, 20, 20),
        score=float(0.96),
        keypoints={},
        side='front'
    )
    imitate_embedding = np.ones((512))
    with tracker_class(config, None) as tracker:
        with OpenCVVideoSource('None') as source:
            data = {'persons': [imitate_detection],
                    'persons_embeddings': {imitate_detection.det_id: imitate_embedding}}
            for i in range(config['min_track_length']-1):
                frame = source.read()
                tracker.process(frame, data)
                assert len(data['active_tracks']) == 0, f"Expected no tracks as {i} frames were processed\
                     and min_track_length is {config['min_track_length']},\
                          but it was {len(data['active_tracks'])} tracks"

            frame = source.read()
            tracker.process(frame, data)
            assert len(data['active_tracks']) == 1, f"Expected 1 track appeared,\
                 but actually it was {len(data['active_tracks'])} tracks"

            for track in data['active_tracks']:
                assert track.max_embeddings_size == config['max_embeddings_size'], f"Expected\
                     track.max_embeddings_size == {config['max_embeddings_size']} \
                         but actual track.max_embeddings_size is {track.max_embeddings_size}"
                assert len(track.embeddings) == config['max_embeddings_size'], f"Expected number of stored embeddigs\
                     {config['max_embeddings_size']}, but actual is {len(track.embeddings)}"
                assert len(track.track_objects) != 0, f"Expected number of track_objects in track !=0,\
                     but actual is {len(track.track_objects)}"
                for obj in track.track_objects:
                    assert type(obj.detection) == Detection, f"Expected track_object.detection to have Detection type,\
                         but actual is {type(obj.detection)}"

test_publisher_params = [
    (TracksFilePublisher, {'filename': 'res/00000001/tracks.json', 'res_folder': 'res', 'processed_day_datetime': '00000001'}, {'settings':{'camera_id':'entrance'}})
]


@mock.patch.object(OpenCVVideoSource, 'read', mocked_read)
@mock.patch.object(OpenCVVideoSource, 'startup', lambda x: None)
@mock.patch.object(OpenCVVideoSource, 'shutdown', lambda x: None)
@pytest.mark.parametrize("publisher_class,config,resources", test_publisher_params)
def test_publisher(publisher_class: typ.Type[Unit], config: dict, resources:dict):
    """
    test TracksFilePublisher unit
    mocked VideoSource.read to test only publisher_class

    Parameters
    ----------
    publisher_class
        unit class to test
    config
        config dict
    """
    imitate_detection = Detection(
        box=BoundingBox(20, 20, 20, 20),
        score=float(0.96),
        keypoints={},
        side='front'
    )
    tz = pytz.timezone('Europe/Paris')
    start_time = dt.datetime(1000, 1, 1, 1, 1, 1)
    imitate_track = Track(max_embeddings_size=1).add(
        detection=imitate_detection,
        offset=1,
        timestamp=start_time)

    imitate_track.add_embedding(np.ones((512)))
    data = {'completed_tracks': [imitate_track], 'active_tracks':[]}
    with publisher_class(config, resources) as publisher:
        with OpenCVVideoSource('None') as source:
            frame = source.read()
            publisher.process(frame, data)

            res_folder = config['res_folder'] + '/' + config['processed_day_datetime']
            file_name = res_folder + '/' + config['filename']
            with open(config['filename'], 'r') as f:
                expected_res = {"camera_id": resources['settings'].get('camera_id', ''),
                                "track_id": imitate_track.track_id,
                                "video": {
                    "fps": float(frame.fps),
                    "frame_width": frame.width,
                    "frame_heigth": frame.height},
                    "source": frame.source.source,
                    "cluster_id": imitate_track.track_id,
                    "start": start_time.astimezone().isoformat(timespec='microseconds'),
                    "end": start_time.astimezone().isoformat(timespec='microseconds'),
                    "trajectory": [{
                        "frame_timestamp": start_time.astimezone().isoformat(timespec='microseconds'),
                        "bounding_box": list(imitate_detection.box),
                        "keypoints": imitate_detection.keypoints,
                        "box_id": imitate_detection.det_id,
                        "confidence": imitate_detection.score,
                        "side": imitate_detection.side}]}
                res = json.loads(f.readline())
                assert res == expected_res, f"Expected json shoulf be equal to {expected_res}, but actual is {res}"
                shutil.rmtree(res_folder)
