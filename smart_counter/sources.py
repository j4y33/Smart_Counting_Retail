import struct
import cv2
import math
from loguru import logger
import trafaret as t
from fractions import Fraction
import datetime as dt
import typing as typ
import queue
from functools import partial
import threading
import ffmpeg
import shlex
import subprocess
import numpy as np
import re
from pathlib import Path, PosixPath
import pytz


from . import base
from .error import ConfigError, VideoSourceError


class KerberosVideoSource(base.VideoSource):

    _config_schema = t.Dict({
        t.Key('type'): t.String(min_length=3),
        t.Key('start_time', optional=True): t.DateTime(),
        t.Key('files_limit', optional=True): t.Int(gte=1),
        t.Key('fps', optional=True): t.Or(t.Int(gt=0), t.Regexp(r'\d+\/\d+')) >> Fraction,
        t.Key('width', optional=True): t.Int(gte=16),
        t.Key('height', optional=True): t.Int(gte=16),
        t.Key('pix_fmt', default='RGB'): t.Enum(
            'RGB', 'RGBA', 'BGR', 'BGRA', 'GRAY'),
        t.Key('frames_offset', optional=True): t.Int(gt=0),
        t.Key('frames_limit', optional=True): t.Int(gt=0),
        t.Key('time_limit_hours', optional=True): t.Int(gte=1),
        t.Key('cmd_dir', optional=True): t.String,
        t.Key('use_gpu', default=False): t.Bool,
        t.Key('crop', optional=True): t.String(min_length=1),
        t.Key('queue_size', default=20): t.Int(gt=0, lt=1000),
        t.Key('tz', default='Europe/Paris'): t.String(min_length=1),
    }, ignore_extra='*')

    @staticmethod
    def extract_datetime_from_filename(filename, tz):
        return dt.datetime.fromtimestamp(int(filename.split('_')[0]), tz=tz)

    @staticmethod
    def _get_videos(all_videos: typ.List[Path],
                    start_time: dt.datetime,
                    end_time: dt.datetime,
                    files_limit: int,
                    tz) -> typ.List[PosixPath]:
        """
        get videos suited by time

        Parameters
        ----------
        all_videos
            sorted list of all videos
        start_time
        end_time
        files_limit
            maximum possible number of files to process or None if there is no limit

        Returns
        -------
            pathes of videos to process
        """
        videos = [video for video in all_videos if end_time >=
                  KerberosVideoSource.extract_datetime_from_filename(video.name, tz) >= start_time]
        return videos[:files_limit] if files_limit else videos

    def __init__(self, source: str, config: typ.Optional[dict] = None):
        super().__init__(source, config)
        source_path = Path(source).absolute()
        if not source_path.is_dir():
            raise VideoSourceError(f'source {source_path} must be a folder, that exists')

        all_videos = sorted([f for f in source_path.glob('*.mp4')])
        if len(all_videos) == 0:
            raise VideoSourceError(f'source {source_path} must contain at least 1 video')

        self._tz = pytz.timezone(self.config['tz'])
        self._start_time = self.config.get(
            'start_time', self.extract_datetime_from_filename(all_videos[0].name, self._tz))
        self._start_time = self._tz.localize(self._start_time)
        if 'time_limit_hours' in self.config:
            self._end_datetime = self._start_time+dt.timedelta(hours=self.config['time_limit_hours'])
        else:
            self._end_datetime = self.extract_datetime_from_filename(all_videos[-1].name, self._tz)

        self._videos = self._get_videos(all_videos,
                                        self._start_time,
                                        self._end_datetime,
                                        self.config.get('files_limit', None),
                                        self._tz)
        if len(self._videos) == 0:
            raise VideoSourceError(f'at least 1 video should be processed, not 0')

        source_cls_by_name = {u.__name__: u for u in base.VideoSource.sub_classes()}
        self._source_cls = source_cls_by_name.get(self.config.get('type', ''), '')
        if self._source_cls == '':
            raise VideoSourceError(f'source class {self.config.get("type", "")} wasn\'t found in sublasses')
        self._capture = None
        self._counter = 0
        self._index = 0

    def _startup_capture(self):
        curr_video = self._videos[self._index]
        curr_video_config = self.config.copy()
        curr_video_config['start_time'] = self.extract_datetime_from_filename(curr_video.name, self._tz)
        self._capture = self._source_cls(str(curr_video), curr_video_config)
        self._index += 1
        self._capture.startup()
        self._stream_info = self._capture.stream_info

    def _shutdown_capture(self):
        if self._capture is not None:
            self._capture.shutdown()
            self._capture = None

    def startup(self):
        self._startup_capture()

    def shutdown(self):
        self._shutdown_capture()
        self._index = 0
        self._videos = []

    def read(self) -> base.Frame:
        """Reads and returns frame"""
        frame = self._capture.read()
        if frame is None and self._index < len(self._videos):
            self._shutdown_capture()
            self._startup_capture()
            frame = self._capture.read()
            if frame is None:
                raise VideoSourceError(f'error for next video from list {self._capture}')

        self._counter += 1
        source_info = self._stream_info.as_dict(recursive=False)
        return base.Frame(
            frame=frame.frame,
            counter=self._counter,
            offset=frame.counter,
            **source_info
        ) if frame else None


class OpenCVVideoSource(base.VideoSource):
    _config_schema = t.Dict({
        t.Key('fps', optional=True): t.Or(t.Int(gt=0), t.Regexp(r'\d+\/\d+')) >> Fraction,
        t.Key('width', optional=True): t.Int(gte=16),
        t.Key('height', optional=True): t.Int(gte=16),
        t.Key('start_time', default=dt.datetime.now()): t.DateTime(),
    }, ignore_extra='*')

    def __init__(self, source: str, config: typ.Optional[dict] = None):
        super().__init__(source, config)

        self._capture = None  # type: cv2.VideoCapture
        self._offeset = 0

    def read(self) -> base.Frame:
        """Reads and returns frame"""

        if self._capture is None:
            raise VideoSourceError(f'{self} not started!')

        is_good, frame = self._capture.read()
        if not is_good:
            return None

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        self._offset += 1

        return base.Frame(frame=frame,
                          offset=self._offset,
                          counter=self._offset,
                          **self._stream_info.as_dict(recursive=False))

    def _source_info(self, cap: cv2.VideoCapture) -> base.SourceInfo:
        """Get original source video information"""
        try:
            return base.SourceInfo(
                source=self.source,
                width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                fps=Fraction(cap.get(cv2.CAP_PROP_FPS))
            )

        except struct.error as err:
            raise VideoSourceError(
                'Error get video information from "{}": {}'.format(self.source, err))

    def _stream_builder(self, cap: cv2.VideoCapture) -> base.StreamInfo:
        source_info = self._source_info(cap)
        return base.StreamInfo(
            stream_id=self.stream_id,
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=Fraction(cap.get(cv2.CAP_PROP_FPS)),
            start_time=self.config['start_time'],
            source=source_info,
            bpp=24,
            pix_fmt=base.PixFmt.BGR,
            command=''
        )

    def startup(self):
        """Initialization"""
        if self._capture is not None:
            raise VideoSourceError(
                'Can\'t start new <{self}> capture while previous not closed!')

        logger.info(f'Starting {self} ({self.source})')

        capture = cv2.VideoCapture(self.source)
        if not capture or not capture.isOpened():
            raise ConfigError(
                '{self} can\'t open source: {self.source}')

        stream_info = self._stream_builder(capture)

        self._capture = capture
        self._stream_info = stream_info
        self._offset = 0

    def shutdown(self):
        """Shutdown and cleanup"""
        if self._capture is not None:
            self._capture.release()
            self._capture = None


class FFmpegVideoSource(base.VideoSource):
    """Video streaming on FFmpeg lib.
    Options:
        - fps - updated stream FPS (e.g fps=24 or fps='3000/101', can't be more than source stream FPS)
        - width - frame resize width
        - height - frame resize height
        - pix_fmt - frame pixels format (e.g. RGB or RGBA)
        - frames_offset - number of frames to skip from the beginning (for video file only)
        - frames_limit - number of frames to process
        - time_offset_msec - time in msec to skip from the beginning (for video file only)
        - time_limit_msec - time in msec to process
        - start_time - timestamp when video started or recorded (default is stream start)
        - cmd_dir - path to dir with with ffmpeg and ffprobe binaries to use
        - use_gpu - use GPU video decoding acceleration if available
        - crop - FFmpeg frame crop string (http://ffmpeg.org/ffmpeg-filters.html#crop)
        - queue_size - frames buffer size (for stream source LeakyQueue will be used)
    Usage example:
        >>> with FFmpegVideoSource('video.mp4') as source:
        ...     while True:
        ...         frame = source.read()
        ...         if frame is None:
        ...             break
        ...         print(frame.frame[100,100])
        >>>
    """

    _pix_fmt_ffmpeg = {
        base.PixFmt.RGB: 'rgb24',
        base.PixFmt.RGBA: 'rgba',
        base.PixFmt.BGR: 'bgr24',
        base.PixFmt.BGRA: 'bgra',
        base.PixFmt.GRAY: 'gray',
    }

    _config_schema = t.Dict({
        t.Key('fps', optional=True): t.Or(t.Int(gt=0), t.Regexp(r'\d+\/\d+')) >> Fraction,
        t.Key('width', optional=True): t.Int(gte=16),
        t.Key('height', optional=True): t.Int(gte=16),
        t.Key('pix_fmt', default='RGB'): t.Enum(
            'RGB', 'RGBA', 'BGR', 'BGRA', 'GRAY') >> base.PixFmt,
        t.Key('frames_offset', optional=True): t.Int(gt=0),
        t.Key('frames_limit', optional=True): t.Int(gt=0),
        t.Key('time_offset_msec', optional=True): t.Int(gt=0),
        t.Key('time_limit_msec', optional=True): t.Int(gt=0),
        t.Key('start_time', default=dt.datetime.now()): t.DateTime(),
        t.Key('cmd_dir', optional=True): t.String,
        t.Key('use_gpu', default=False): t.Bool,
        t.Key('crop', optional=True): t.String(min_length=1),
        t.Key('queue_size', default=20): t.Int(gt=0, lt=1000),
    }, ignore_extra='*')

    def __init__(self, source: str, config: typ.Optional[dict] = None):
        super().__init__(source, config)
        ffmpeg_cmd = 'ffmpeg'
        ffprobe_cmd = 'ffprobe'

        if 'cmd_dir' in self.config:
            cmd_dir = Path(self.config['cmd_dir'])
            if not cmd_dir.exists():
                raise ConfigError(
                    'Wrong option "cmd_dir": directory "{}" not exists!'.format(cmd_dir))

            if not Path(self.config['cmd_dir']).is_dir():
                raise ConfigError(
                    'Wrong option "cmd_dir": path "{}" should be dir!'.format(cmd_dir))

            ffmpeg_cmd = str(cmd_dir / ffmpeg_cmd)
            ffprobe_cmd = str(cmd_dir / ffprobe_cmd)

        if 'frames_offset' in self.config and 'time_offset_msec' in self.config:
            raise ConfigError(
                'Prohibited to use "frames_offset" and "time_offset_msec" at the same time')

        if 'frames_limit' in self.config and 'time_limit_msec' in self.config:
            raise ConfigError(
                'Prohibited to use "frames_limit" and "time_limit_msec" at the same time')

        self._ffmpeg_cmd = ffmpeg_cmd
        self._ffprobe_cmd = ffprobe_cmd

        self._counter = 0
        self._frames_offset_from_start = 0

        self._process = None         # type: subprocess.Popen

        queue_cls = queue.Queue if self.source_type == base.SourceType.FILE \
            else partial(base.LeakyQueue, on_drop=self._on_drop)

        self._queue = queue_cls(self.config['queue_size'])
        self._errors_queue = queue.Queue(self.config['queue_size'])
        self._end_event = threading.Event()
        self._reading_thread = threading.Thread(target=self._frames_reading_loop)

    def _on_drop(self, queue: base.LeakyQueue, frame: base.Frame) -> None:
        logger.warning(
            f'Frame #{frame.offset} for {self} is dropped (totally dropped {queue.dropped} frames)')

    def _search_decoder(self, codec: str) -> typ.Optional[str]:
        """Search decoder for video codec.
        FFMPEG can support accelerated decoder.
        NVDEC decoders presented in FFMPEG with suffix "_cuvid"
        See: https://trac.ffmpeg.org/wiki/HWAccelIntro
        Steps:
            - Extract additional codec decoders (e.g. `h264_cuvid` decoder on GPU for `h264` codec).
            - Search for extra decoder (if no extra decoder than use default)
        """
        codecs_out = subprocess.run([self._ffmpeg_cmd, '-hide_banner', '-codecs'], stdout=subprocess.PIPE)
        extra_decoders = {
            fcodec.decode(): set(decoders.decode().split(' '))
            for fcodec, decoders in
            re.findall(br'[DEVASIL.]{6} ([\w\d_]+) .* \(decoders: ([\w\d\s_]+) \)', codecs_out.stdout)
        }
        decoders = extra_decoders.get(codec)

        if decoders is not None and '{}_cuvid'.format(codec) in decoders:
            return '{}_cuvid'.format(codec)

        logger.warning(f'FFMPEG does not support GPU decoder for "{codec}" codec')
        return None

    def _stream_builder(self) -> typ.Tuple[typ.List[str], base.StreamInfo]:
        """Build FFMPEG command line for video streaming
        :return (shell command args, StreamInfo)
        """
        source_info = base.read_source_info(self.source, ffprobe_cmd=self._ffprobe_cmd)

        width = source_info.width
        height = source_info.height
        fps = source_info.fps
        pix_fmt = self.config['pix_fmt']

        input_params = {}
        output_params = {}
        filters = []

        if 'fps' in self.config:
            fps = self.config['fps']
            if fps > source_info.fps:
                raise VideoSourceError(
                    'Video source FPS={} can\'t be more '
                    'than original source FPS={}'.format(fps, source_info.fps))

            filters.append('fps=fps={}'.format(fps))

        if 'width' in self.config and 'height' in self.config:
            width, height = self.config['width'], self.config['height']
            filters.append('scale={}:{}'.format(width, height))
            output_params['sws_flags'] = 'neighbor'  # nearest neighbor rescaling algorithm

        if source_info.pix_fmt != self._pix_fmt_ffmpeg[pix_fmt]:
            output_params['pix_fmt'] = self._pix_fmt_ffmpeg[pix_fmt]

        if self.source_type == base.SourceType.FILE:
            time_offset_sec = 0

            if 'frames_offset' in self.config:
                time_offset_sec = float(self.config['frames_offset'] / source_info.fps)

            elif 'time_offset_msec' in self.config:
                time_offset_sec = self.config['time_offset_msec'] / 1000.0

            if (time_offset_sec * 1000) >= 1.0:
                input_params['ss'] = '{:0.3f}'.format(time_offset_sec)
                # frames offset with new FPS
                self._frames_offset_from_start = math.ceil(fps * time_offset_sec)

        # TODO: check and change with custom counter
        if 'frames_limit' in self.config:
            output_params['frames:v'] = self.config['frames_limit']

        elif 'time_limit_msec' in self.config:
            output_params['to'] = '{:0.3f}'.format((self.config["time_limit_msec"] / 1000.0))

        if self.config['use_gpu']:
            decoder = self._search_decoder(source_info.codec)
            if decoder:
                input_params['codec:v'] = decoder

        if 'crop' in self.config:
            # FIXME: width, height changed after crop (check together with scale)
            filters.append('crop={}'.format(self.config['crop']))

        if filters:
            output_params['filter:v'] = ', '.join(filters)

        cmd_args = (
            ffmpeg
            .input(self.source, **input_params)
            .output('pipe:', format='rawvideo', **output_params)
            .global_args('-hide_banner')
            .global_args('-loglevel', 'error')
            .global_args('-y')
            .compile(cmd=self._ffmpeg_cmd)
        )

        if pix_fmt.bpp() % 8 != 0:
            raise VideoSourceError(
                'FFmpeg VideoSource can\'t use pixel format {pix_fmt} with '
                'not integer number of bytes'.format(pix_fmt=pix_fmt))

        return cmd_args, base.StreamInfo(
            stream_id=self.stream_id,
            width=width,
            height=height,
            bpp=pix_fmt.bpp(),
            fps=fps,
            pix_fmt=pix_fmt,
            command=' '.join(map(shlex.quote, cmd_args)),
            start_time=self.config['start_time'],
            source=source_info
        )

    def _read_frame_from_process(self) -> typ.Optional[base.Frame]:
        """Get next frame from FFmpeg subprocess stdout"""
        width = self._stream_info.width
        height = self._stream_info.height
        bytes_per_pixel = self._stream_info.bpp // 8
        frame_size = width * height * bytes_per_pixel

        frame_bytes = self._process.stdout.read(frame_size)
        if len(frame_bytes) == 0:
            return None

        if len(frame_bytes) != frame_size:
            raise VideoSourceError(
                'Frame truncated: got {} bytes instead of {}'.format(
                    len(frame_bytes), frame_size))

        frame = np.frombuffer(frame_bytes, np.uint8).reshape([height, width, bytes_per_pixel])

        self._counter += 1

        return base.Frame(
            frame=frame,
            counter=self._counter,
            offset=self._frames_offset_from_start + self._counter,
            **self._stream_info.as_dict(recursive=False)
        )

    def _frames_reading_loop(self) -> None:
        """Read frames into separate thread and push into queue"""
        while not self._end_event.is_set():
            try:
                if self._process is None:
                    raise VideoSourceError('{} not started!'.format(self))

                frame = self._read_frame_from_process()
                self._queue.put(frame)

            except subprocess.SubprocessError as err:
                logger.warning(f'{self} got read frame error: {err}. Skipping...')

            except VideoSourceError as err:
                logger.error(f'{self} got read frame error: {err}. Terminating...')
                self._end_event.set()

                try:
                    self._errors_queue.put_nowait(err)
                except queue.Full:
                    logger.warning(f'Too many {self} errors. Skipping error: {err}')
            else:
                if frame is None:
                    self._end_event.set()

    def startup(self) -> None:
        """Compile command and run in subprocess"""
        if self._process is not None:
            raise VideoSourceError(
                'Can\'t start new <{}> subprocess '
                'while previous not closed!'.format(self))

        if self.source_type == base.SourceType.OTHER:
            raise ConfigError(
                'Unknown or missing source: {}'.format(self.source))

        self._end_event.clear()
        self._counter = 0

        cmd_args, self._stream_info = self._stream_builder()

        logger.info(f'Running: {self._stream_info.command}')
        process = subprocess.Popen(cmd_args, stdout=subprocess.PIPE, universal_newlines=False)

        self._process = process
        self._reading_thread.start()

    @ staticmethod
    def _clean_queue(q: queue.Queue):
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break

    def shutdown(self) -> None:
        """Destroy subprocess"""
        self._end_event.set()

        self._clean_queue(self._queue)
        self._clean_queue(self._errors_queue)

        if self._reading_thread.is_alive():
            self._reading_thread.join(timeout=1)

        if self._process is not None:
            logger.info(f'Stopping {self} subprocess')
            self._process.stdout.close()
            self._process.kill()
            logger.info(f'Waiting for {self} process to destroy')
            try:
                self._process.wait(timeout=4)

            except subprocess.TimeoutExpired:
                logger.error(f'Error correctly stop {self} subprocess. Terminating...')
                self._process.terminate()
            finally:
                self._process = None

                # Workaround to restore TTY echo settings
                subprocess.run(['stty', 'echo'])

        logger.info(f'{self} successfully destroyed')

    def read(self) -> typ.Optional[base.Frame]:
        """Get next frame from source (return None if stream is closed)"""
        if not self._end_event.is_set() and self._process is None:
            raise VideoSourceError('{} not started!'.format(self))

        try:
            raise self._errors_queue.get_nowait()
        except queue.Empty:
            pass

        if self._end_event.is_set() and self._queue.empty():
            return None

        while True:
            try:
                return self._queue.get(timeout=0.5)
            except queue.Empty:
                if self._end_event.is_set():
                    return None
