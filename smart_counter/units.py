from loguru import logger
from pathlib import Path
import trafaret as t
import json
from enum import Enum
import datetime as dt
import numpy as np
import itertools
import typing as typ
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Point, Polygon
import cv2

from .error import ConfigError, ApplicationError
from .base import Unit, Frame, StreamInfo
from .displays import VideoDisplay
from .video_writers import FFmpegVideoWriter
from . import structs as st
from . import utils
from scipy.optimize import linear_sum_assignment


class PersonDetector(Unit):

    _config_schema = t.Dict({
        t.Key('min_box_shape', optional=True, default=(100, 100)): t.Tuple(t.Int(gte=1), t.Int(gte=1)),
        t.Key('iou_overlap_threshold', optional=True, default=0.2): t.Float,
        t.Key('zones_include', optional=True, default=[[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]):
        t.List(t.List(t.Tuple(t.Float(gte=0.0, lte=1.0), t.Float(gte=0.0, lte=1.0)))) >> (
            lambda x: list(map(lambda y: Polygon(y), x))),
    }, ignore_extra='*')

    def process(self, frame: Frame, data: dict) -> None:
        """
        Unit process method

        Args:
            frame : frame to process
            data : dict where results will be saved
        """
        persons = self.models['person_detector'].process_single(frame.frame)
        persons = self._exclude_overlaps(persons) if len(persons) > 1 else persons
        data['persons'] = [st.Detection(
            box=p['box'],                                                          # typ.Tuple[int, int, int, int]
            score=p['score'],                                                      # float
            keypoints=p['keypoints'],                                              # typ.Dict[str:KeyPoint]
            side=self.detect_side(p['keypoints'])                                  # str
        )for p in persons if self._validate_detection(p, frame)]

        logger.debug(f"frame #{frame.offset}: detected {len(data['persons'])} persons")

    def _validate_detection(self, detection: dict, frame: Frame) -> bool:
        """
        check if detection is valid

        Parameters
        ----------
        detection
            detection to check

        Returns
        -------
            True if valid otherwise False
        """
        center = Point((detection['box'].x + detection['box'].w/2)/frame.width,
                       (detection['box'].h + detection['box'].h/2)/frame.height)
        return (detection['box'].w > self.config['min_box_shape'][0]
                and detection['box'].h > self.config['min_box_shape'][1]
                and any(map(lambda x: x.contains(center), self.config['zones_include']))
                and len(detection['keypoints']) >= 11)

    def _exclude_overlaps(self, detections: typ.List[dict]) -> np.ndarray:
        """
        exclude detections with overlaps

        Args:
            detections : all detections

        Returns:
            detections without overlaps
        """
        # get boxes from detections
        detections_boxes = np.array([det['box'] for det in detections])
        # calculate iou matrix
        ious = utils.iou(detections_boxes, detections_boxes)
        # box iou with himself is 0
        np.fill_diagonal(ious, 0)
        # calculate sizes comparison mask
        size_matrix = np.array([[det1[2]*det1[3] < det2[2]*det2[3] for det1 in detections_boxes]
                                for det2 in detections_boxes])
        # calculate mask for detections
        ious = np.sum(ious*size_matrix, axis=1) < self.config['iou_overlap_threshold']
        # exclude detections
        return np.array(detections)[ious]

    @staticmethod
    def detect_side(keypoints: typ.Dict[str, st.KeyPoint]) -> str:
        """
        detect side by leypoints

        Parameters
        ----------
        keypoints
            dict with keypoints
        threshold
            threshold for keypoints score

        Returns
        -------
            side name
        """
        if all([kp in keypoints for kp in ['nose', 'left_eye', 'right_eye']]):
            return st.Side.FRONT
        elif all([kp in keypoints for kp in ['left_ear', 'right_ear', 'right_eye']]):
            return st.Side.BACK
        return st.Side.UNDEFINED


class PersonEmbedding(Unit):

    def process(self, frame: Frame, data: dict) -> None:
        """Unit process method executor."""
        embeddings = {}  # typ.Dict[str, np.ndarray]
        for p in data['persons']:
            crop_img = utils.crop_by_box(frame.frame, p.box)
            embeddings[p.det_id] = self.models['person_embedding'].process_single(crop_img)

        data['persons_embeddings'] = embeddings

        logger.debug(f"frame #{frame.offset}: created {len(data['persons_embeddings'])} embeddings")


class Tracker(Unit):

    _config_schema = t.Dict({
        t.Key('max_time_without_detections_ms', optional=True, default=32000): t.Int(gte=1),
        t.Key('min_track_length', optional=True, default=3): t.Int(gte=1),
        t.Key('max_track_length', optional=True, default=600): t.Int(gte=2),
        t.Key('sequential_track_id', optional=True, default=True): t.Bool,
        t.Key('cosine_distance_threshold', optional=True, default=0.5): t.Float,
        t.Key('max_embeddings_size', optional=True, default=1): t.Int(gte=1),
    }, ignore_extra='*')

    def on_startup(self) -> None:
        self.tracks = []  # type: typ.List[Track]

        if self.config['sequential_track_id']:
            st.Track.id_factory = st.sequential_id_factory('trk-', size=5)

        logger.debug(f"config | {self.__class__.__name__} | {self.config}")

    def process(self, frame: Frame, data: dict):
        """
        unit process method

        Parameters
        ----------
        frame
            frame to process
        data
            dict with all previous info
        """
        detections = data['persons']
        embeddings = data['persons_embeddings']
        if detections and self.tracks:
            match_results = self.match_boxes_by_embeddings(
                [track.embeddings for track in self.tracks],
                [[embeddings[det.det_id]] for det in detections],
                threshold=self.config['cosine_distance_threshold']
            )
        else:
            match_results = []

        # update tracks with matched detections
        for trk_ind, det_ind in match_results:
            self.tracks[trk_ind].add(
                detection=detections[det_ind],
                offset=frame.offset,
                timestamp=frame.timestamp)

        # create new tracks with unmatched detections
        unmatched_dets_ind = set(range(len(detections))) - set(det_ind for _, det_ind in match_results)
        for det_ind in unmatched_dets_ind:
            self.tracks.append(
                st.Track(max_embeddings_size=self.config['max_embeddings_size']).add(
                    detection=detections[det_ind],
                    offset=frame.offset,
                    timestamp=frame.timestamp))

        for track in self.tracks:
            det_id = track.last.detection.det_id
            if det_id in embeddings:
                track.add_embedding(embeddings[det_id])

        if self.tracks:
            durations_no_det = frame.timestamp - np.array([x.last.frame_timestamp for x in self.tracks])
            duration_sec = np.array([x.seconds for x in durations_no_det])
            durations = np.array([track.length for track in self.tracks])

            too_short = durations < self.config['min_track_length']
            too_long = durations > self.config['max_track_length']
            is_valid = ~too_short & ~too_long

            is_dead = duration_sec*1000 >= self.config['max_time_without_detections_ms']
            is_alive = ~is_dead

            data['active_tracks'] = list(itertools.compress(self.tracks,
                                                            is_alive & is_valid))

            data['completed_tracks'] = list(itertools.compress(self.tracks,
                                                               (is_valid & is_dead) | too_long))

            self.tracks = list(itertools.compress(self.tracks,
                                                  is_alive & (is_valid | too_short)))
        else:
            data['completed_tracks'] = []
            data['active_tracks'] = []

        logger.debug(f"frame #{frame.offset}: {len(data['active_tracks'])} active tracks")

        for trk in data['completed_tracks']:
            logger.debug(
                f"frame #{frame.offset}: track {trk.track_id} completed (with len {trk.length})")

    @staticmethod
    def match_boxes_by_embeddings(prev_embeddings: typ.List[typ.List[np.ndarray]],
                                  new_embeddings: typ.List[np.ndarray],
                                  threshold: float) -> typ.List[typ.Tuple[int, int]]:
        """
        matches embeddings by cosine distance

        Parameters
        ----------
        prev_embeddings
            previous embeddings
        new_embeddings
            new embeddings
        threshold
            threshold for cosine distance

        Returns
        -------
            list with matched pairs
        """
        # np.ndarray with shape (num_tracks, num_detections)
        distances = np.ones((len(prev_embeddings), len(new_embeddings)))
        det_embeddings = np.array(new_embeddings).squeeze(axis=1)
        for trk_id in range(len(prev_embeddings)):
            trk_embeddings = np.array(prev_embeddings[trk_id])
            distances[trk_id] = np.mean(utils.cosine_distance_vectorized(trk_embeddings, det_embeddings), axis=0)

        matched = []
        distances_clipped = np.clip(distances, 0.0, threshold)

        # lists with shape (min(num_tracks, num_detections))
        row_ind, col_ind = linear_sum_assignment(distances_clipped)

        return [(i, j) for i, j in zip(row_ind, col_ind) if distances[i, j] <= threshold]


class JsonEncoder(json.JSONEncoder):
    """Json Encoder for proper custom object logging"""

    def default(self, obj):
        if isinstance(obj, Enum):
            return str(obj.value)
        elif isinstance(obj, dt.datetime):
            # .astimezone(pytz.timezone("Europe/Kiev"))
            return obj.isoformat(timespec="microseconds")
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return json.JSONEncoder.default(self, obj)


class TracksFilePublisher(Unit):
    """Publish main info about completed tracks to the file"""

    _config_schema = t.Dict({
        t.Key('filename', default="tracks.json"): t.String(min_length=3),
    }, ignore_extra='*')

    def on_startup(self):
        self._filename = Path(self.config['filename']).absolute()
        self._filename.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._filename, mode='w')

        self._last_stream_info = None
        self._last_active_tracks = None

    def on_shutdown(self):
        if hasattr(self, '_file'):
            # Publish active tracks so not lost them on source end
            if hasattr(self, "_last_stream_info"):
                self._publish_tracks(self._last_stream_info, self._last_active_tracks)
                self._last_stream_info = None
            self._file.close()

    def process(self, frame: Frame, data: dict) -> None:
        """Publishes tracks info to file"""
        self._publish_tracks(frame, data['completed_tracks'])

        self._last_stream_info = frame
        self._last_active_tracks = data['active_tracks']

    def _publish_tracks(self, stream_info: StreamInfo, tracks: typ.List[st.Track]) -> None:
        if not stream_info:
            logger.warning("{self}: stream_info is None")
            return

        
        for track in tracks:
            track_info = {
                'camera_id': self._resources['settings'].get('camera_id', ''),
                'track_id': track.track_id,
                'video': {
                    'fps': float(stream_info.fps),
                    'frame_width': stream_info.width,
                    'frame_heigth': stream_info.height
                },
                'source': stream_info.source.source,
                'cluster_id': track.track_id,
                'start': track.first.frame_timestamp,
                'end': track.last.frame_timestamp,
                'trajectory': list(map(self._track_object_to_dict, track.track_objects))
            }
            logger.info(f"published track #{track.track_id} to {self._filename.name}")

            message = json.dumps(track_info, cls=JsonEncoder)
            self._file.write(message)
            self._file.write('\n')
        self._file.flush()

    @staticmethod
    def _track_object_to_dict(track_object: st.TrackObject) -> dict:
        """
        format TrackObject to dict for json dump

        Parameters
        ----------
        track_object
            object

        Returns
        -------
            dict
        """
        return {
            'frame_timestamp': track_object.frame_timestamp,
            'bounding_box': track_object.detection.box,
            'keypoints': track_object.detection.keypoints,
            'box_id': track_object.detection.det_id,
            'confidence': track_object.detection.score,
            'side': track_object.detection.side
        }


class TracksDataFilePublisher(Unit):
    """Publish events to the file"""

    _config_schema = t.Dict({
        t.Key('folder', default="tracks_data/"): t.String(min_length=3),
    }, ignore_extra='*')

    def on_startup(self):

        self._data_folder = Path(self.config['folder']).absolute()
        if self._data_folder.exists():
            raise ConfigError(
                'folder with data is already exists '
                'for {} ({})'.format(self, self._data_folder))
        self._data_folder.mkdir(parents=True, exist_ok=True)

    def process(self, frame: Frame, data: dict) -> None:
        """Publishes event to file"""
        for track in data['active_tracks']:
            if track.last.offset != frame.offset:
                continue
            with open(self._data_folder / (track.track_id+'.json'), 'a') as f:
                json_info = {
                    'box_id': track.last.detection.det_id,
                    'embedding': track.embeddings[-1].tolist(),
                    'img': ''
                }
                f.write(json.dumps(json_info)+'\n')


class OnlyCropsPublisher(Unit):
    """Publish events to the file"""

    _config_schema = t.Dict({
        t.Key('folder', default="tracks_crops/"): t.String(min_length=3),
        t.Key('tracks_num', default=200): t.Int(gte=1),
        t.Key('crop_limit', default=1000): t.Int(gte=1),
    }, ignore_extra='*')

    def on_startup(self):

        self._data_folder = Path(self.config['folder']).absolute()
        self._data_folder.mkdir(parents=True, exist_ok=True)

    def process(self, frame: Frame, data: dict) -> None:
        """Publishes event to file"""
        if len(list(self._data_folder.glob("*"))) > self.config['tracks_num'] or frame.counter > self.config['crop_limit']:
            raise ApplicationError(f'enough crops to save, {self.config["tracks_num"]}')

        for person in data['persons']:
            last_crop = utils.crop_by_box(frame.frame, person.box)
            cv2.imwrite(str(self._data_folder/(str(frame.counter)+'_'+person.det_id+'.jpg')),
                        cv2.cvtColor(last_crop, cv2.COLOR_RGB2BGR))
            with open(self._data_folder / 'embeddings.json', 'a') as f:
                json_info = {
                    'image_id': str(frame.counter)+'_'+person.det_id,
                    'embedding': data['persons_embeddings'][person.det_id].tolist()
                }
                f.write(json.dumps(json_info)+'\n')


class CropsPublisher(Unit):
    """Publish events to the file"""

    _config_schema = t.Dict({
        t.Key('folder', default="tracks_crops/"): t.String(min_length=3),
        t.Key('tracks_num', default=200): t.Int(gte=1),
    }, ignore_extra='*')

    def on_startup(self):

        self._data_folder = Path(self.config['folder']).absolute()
        self._data_folder.mkdir(parents=True, exist_ok=True)

    def process(self, frame: Frame, data: dict) -> None:
        """Publishes event to file"""
        if len(list(self._data_folder.glob("*"))) > self.config['tracks_num']:
            raise ApplicationError(f'enough crops to save, {self.config["tracks_num"]}')

        for track in data['active_tracks']:
            if track.last.offset != frame.offset:
                continue

            track_folder = self._data_folder / track.track_id
            track_folder.mkdir(parents=True, exist_ok=True)
            last_box = track.last.detection.box
            last_crop = utils.crop_by_box(frame.frame, last_box)
            image_id = track.last.detection.det_id
            cv2.imwrite(str(track_folder/(str(frame.counter)+'_'+image_id+'.jpg')),
                        cv2.cvtColor(last_crop, cv2.COLOR_RGB2BGR))
            with open(track_folder / 'embeddings.json', 'a') as f:
                json_info = {
                    'image_id': str(frame.counter)+'_'+image_id,
                    'embedding': track.embeddings[-1].tolist()
                }
                f.write(json.dumps(json_info)+'\n')


class DataOverlay(Unit):
    """Publish events to the file"""

    _FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
    _FONT_SIZE = 28
    _PERSON_BOX_LINE_SIZE = 4
    _PERSON_BOX_COLOR = (0, 200, 0, 255)
    _TRACK_TEXT_COLOR = 'white'
    _TRACK_TEXT_MARGIN = 5
    _TRACK_MAX_TIME_SINCE_LAST_DETECTION = dt.timedelta(seconds=0)  # seconds
    _MAX_TRAJECTORY_POINTS = 32

    _config_schema = t.Dict({
        t.Key('display', default={}): t.Dict({
            t.Key('enabled', default=False): t.Bool(),
        }, allow_extra='*'),
        t.Key('dvr', default={}): t.Dict({
            t.Key('enabled', default=False): t.Bool(),
        }, allow_extra='*')
    }, ignore_extra='*')

    def on_startup(self) -> None:
        self.font = ImageFont.truetype(self._FONT, size=self._FONT_SIZE)

        self._display = None
        if self.config['display']['enabled']:
            self._display = VideoDisplay(self.config['display'])
            self._display.startup()

        self._video_writer = None
        if self.config['dvr']['enabled']:
            cfg = FFmpegVideoWriter.from_stream(self.stream_info, self.config['dvr'])
            self._video_writer = FFmpegVideoWriter(filename=cfg['filename'],
                                                   config=cfg)
            self._video_writer.startup()

        self._colors = utils.ColorPicker()

    def on_shutdown(self) -> None:
        if hasattr(self._display, 'shutdown'):
            self._display.shutdown()

        if hasattr(self._video_writer, 'shutdown'):
            self._video_writer.shutdown()

    def process(self, frame: Frame, data: dict) -> None:
        if not self._display and not self._video_writer:
            return

        draw_frame = Image.fromarray(frame.frame.copy())
        draw = ImageDraw.Draw(draw_frame, 'RGBA')
        font = self.font
        frame_norm = np.array([*frame.size, *frame.size])

        # draw tracks
        for trk in data['active_tracks']:

            if (frame.timestamp - trk.last.frame_timestamp) > self._TRACK_MAX_TIME_SINCE_LAST_DETECTION:
                continue

            det = trk.last.detection

            # get unique track color
            # convert rgb to rgba
            box_color = tuple(self._colors.get(trk.track_id) + [255])

            x, y, w, h = det.box

            # draw box
            draw.rectangle([x, y, x + w, y + h],
                           outline=box_color,
                           width=self._PERSON_BOX_LINE_SIZE)

            # draw text
            lines = [str(trk.track_id), str(det.side.value), f"{det.score:.2f}"]
            ty = y
            for line in lines:
                _, th = self.font.getsize(line)
                draw.text((x + self._TRACK_TEXT_MARGIN, ty - th - self._TRACK_TEXT_MARGIN),
                          font=self.font,
                          fill=self._TRACK_TEXT_COLOR,
                          text=line)
                ty += (th + self._TRACK_TEXT_MARGIN)

            # draw tail
            n = trk.length - 1
            n_from = 0 if n <= self._MAX_TRAJECTORY_POINTS else n - self._MAX_TRAJECTORY_POINTS

            target_trajectory = list(reversed([to.detection.top
                                               for to in trk.track_objects[n_from:n]]))

            self._draw_trajectory_tail(draw=draw, points=target_trajectory,
                                       max_size=self._MAX_TRAJECTORY_POINTS,
                                       color=box_color)

        if self._display:
            self._display.display(draw_frame.convert('RGB'))

        if self._video_writer:
            self._video_writer.write(draw_frame.convert('RGB'))

    @staticmethod
    def _draw_trajectory_tail(*, draw: ImageDraw,  points: typ.List[st.Point],
                              max_size: int = 128, color=(0, 0, 255, 255),
                              min_line_width: int = 4,
                              max_line_width: int = 8):
        """Draws track's tail"""
        if len(points) <= 0:
            return

        # manually selected parameter
        thickness_scaler = 3 ** 3 / max_size if max_size > 0 else 1
        width_step = (max_line_width - min_line_width) / max_size

        for i in range(1, len(points)):
            pt_from = points[i - 1]
            pt_to = points[i]

            step = 1 if (max_size - i) == 0 else abs(max_size - i)
            thickness = max(min_line_width, int(max_line_width - i * width_step))
            draw.line([(pt_from.x, pt_from.y), (pt_to.x, pt_to.y)], fill=color, width=thickness)


# ***************** debug_units.py *************************
"""
Using PersonsFilePublisher publish detections per frame (using powerful PC)
Using PersonsFileReader re-create detections per frame without any models (using less performance PC)
"""


class PersonsFileReader(Unit):

    _config_schema = t.Dict({
        t.Key('filename', default="persons.json"): t.String(min_length=3),
    }, ignore_extra='*')

    def on_startup(self):
        self._filename = Path(self.config['filename']).absolute()
        self._file = open(self._filename, mode='r')
        self._eof = False

        self._last_frame_info = None

    def on_shutdown(self):
        if hasattr(self, '_file'):
            self._file.close()

    def process(self, frame: Frame, data: dict) -> None:
        data['persons'] = []  # typ.List[st.Detection]
        data['persons_embeddings'] = {}  # typ.Dict[str, np.ndarray]

        if self._eof:
            return

        if not self._last_frame_info:
            # read until we get anything with close frame offset
            while not self._last_frame_info or self._last_frame_info['frame_offset'] < frame.offset:
                line = self._file.readline()
                if len(line) <= 0:
                    self._eof = True
                    return

                self._last_frame_info = json.loads(line)

        if frame.offset == self._last_frame_info['frame_offset']:
            # scale box to current frame size
            scale_x = frame.width / self._last_frame_info['video']['frame_width']
            scale_y = frame.height / self._last_frame_info['video']['frame_height']

            box_scalier = np.array([scale_x, scale_y, scale_x, scale_y], dtype=np.float32)
            for p in self._last_frame_info['persons']:
                box_wrt_frame = list(map(int, p['box'] * box_scalier))

                det = st.Detection(
                    box=st.BoundingBox(*box_wrt_frame),
                    score=float(p['score']),
                    keypoints={k: st.KeyPoint(*pt) for k, pt in p['keypoints'].items()},
                    side=st.Side(p['side'])
                )
                data['persons'].append(det)
                data['persons_embeddings'][det.det_id] = p['embedding']

            self._last_frame_info = None


class PersonsFilePublisher(Unit):
    """Publishes st.Detection and Embeddings for each frame"""

    _config_schema = t.Dict({
        t.Key('filename', default="persons.json"): t.String(min_length=3),
    }, ignore_extra='*')

    def on_startup(self):
        self._filename = Path(self.config['filename']).absolute()
        self._filename.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._filename, mode='w')

    def on_shutdown(self):
        if hasattr(self, '_file'):
            self._file.close()

    def process(self, frame: Frame, data: dict) -> None:
        """Publishes tracks info to file"""

        persons = []
        for det in data['persons']:
            persons.append({
                **det.as_dict(),
                "embedding": data['persons_embeddings'][det.det_id]
            })

        if len(persons) > 0:
            message = {
                'video': {
                    'fps': float(frame.fps),
                    'frame_width': frame.width,
                    'frame_height': frame.height
                },
                'source': frame.source.source,
                'frame_timestamp': frame.timestamp,
                'frame_offset': frame.offset,
                'persons': persons
            }
            self._file.write(json.dumps(message, cls=JsonEncoder))
            self._file.write('\n')
            self._file.flush()


class ProcessedVideosFileWriter(Unit):
    _config_schema = t.Dict({
        t.Key('filename', default="processed_videos.txt"): t.String(min_length=3),
    }, ignore_extra='*')

    def on_startup(self):
        self._filename = Path(self.config['filename']).absolute()
        self._filename.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._filename, mode='w')
        self._curr_source = None

    def on_shutdown(self):
        if hasattr(self, '_file'):
            self._file.close()
        self._curr_source = None

    def process(self, frame: Frame, data: dict) -> None:
        """Write processed videos to file"""
        if not self._curr_source or self._curr_source != frame.source.source:
            self._curr_source = frame.source.source
            self._file.write(self._curr_source.split('/')[-1]+'\n')
            self._file.flush()
