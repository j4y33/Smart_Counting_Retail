import abc
import shlex
import queue
import logging
import threading
import subprocess
import typing as typ
from pathlib import Path
from fractions import Fraction
from copy import deepcopy

import ffmpeg
import PIL.Image
import numpy as np
import trafaret as t

from . import utils
from .base import StreamInfo
from .error import ConfigError, VideoWriterError


class BaseVideoWriter(abc.ABC):
    """Base VideoWriter class

    :param filename: path to the video file to write
    :param config: additional video source processing options
    :param source: source instance (we can get config from video source)
    """

    _config_schema = t.Dict(allow_extra='*')    # should be redefined in the inherited class

    def __init__(self, filename: str, config: typ.Optional[dict] = None):
        try:
            config = self._config_schema.check(config or {})
        except (t.DataError, ValueError) as err:
            raise ConfigError(
                'Wrong {} configuration: {}'.format(self, err))

        self._filename = filename
        self._config = config

        self._log = logging.getLogger('sg.vw.{}'.format(self.__class__.__name__))

    @property
    def filename(self) -> str:
        return self._filename

    @property
    def config(self) -> dict:
        return self._config

    @property
    def log(self) -> logging.Logger:
        return self._log

    def __str__(self) -> str:
        return '{} [{}]'.format(self.__class__.__name__, self.filename)

    def __repr__(self) -> str:
        return '<{}>'.format(self)

    def __enter__(self):
        self.startup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    def startup(self) -> None:
        """Video writer initializer"""

    def shutdown(self) -> None:
        """Video writer destroyer"""

    @abc.abstractmethod
    def write(self, draw_frame: np.ndarray) -> None:
        """Write next frame to the video file"""

    @staticmethod
    def from_stream(stream_info: StreamInfo,
                    config: typ.Dict[str, typ.Any] = None) -> typ.Dict[str, typ.Any]:
        base_config = deepcopy(config) or {}

        for field in ['fps', 'width', 'height']:
            if field not in base_config:
                base_config[field] = getattr(stream_info, field)

        return base_config


_ffmpeg_presets = [
    'ultrafast',
    'superfast',
    'veryfast',
    'faster',
    'fast',
    'medium',
    'slow',
    'slower',
    'veryslow',
]


class FFmpegVideoWriter(BaseVideoWriter):
    """Video writer based on FFmpeg lib.

    Options:
        - fps - video file FPS (e.g fps=24 or fps='3000/101', default: get from source)
        - width - video width size (default: get from source)
        - height - video height size (default: get from source))
        - crf - constant rate factor (ffmpeg quality parameter*)
        - preset - encoding speed to compression ratio (ffmpeg quality parameter*)
        - codec - encoding codec ("ffmpeg -codecs", https://trac.ffmpeg.org/wiki/HWAccelIntro)
        - cmd_dir - path to dir with with ffmpeg and ffprobe binaries to use (optional)

    *FFmpeg quality configuration: https://trac.ffmpeg.org/wiki/Encode/H.264
    """
    _config_schema = t.Dict({
        t.Key('fps', optional=True): t.Or(t.Int(gt=0), t.Regexp(r'\d+\/\d+')) >> Fraction,
        t.Key('width', optional=True): t.Int(gte=16),
        t.Key('height', optional=True): t.Int(gte=16),
        t.Key('crf', optional=True): t.Int(gte=0, lte=63),
        t.Key('preset', optional=True): t.Enum(*_ffmpeg_presets),
        t.Key('codec', optional=True): t.String(min_length=2, max_length=10),
        t.Key('resize', default=False): t.Bool,
        t.Key('cmd_dir', optional=True): t.String,
    }, ignore_extra='*')

    def __init__(self, filename: str, config: typ.Optional[dict] = None):
        super().__init__(filename, config)

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

        self._ffmpeg_cmd = ffmpeg_cmd
        self._ffprobe_cmd = ffprobe_cmd

        self._process = None         # type: typ.Optional[subprocess.Popen]

        self._queue = queue.Queue(maxsize=2)
        self._end_event = threading.Event()
        self._writing_thread = threading.Thread(target=self._frames_writing_loop)

    def _stream_builder(self) -> subprocess.Popen:
        """Build FFmpeg subprocess for video writing"""

        _missing_err = 'Can\'t identify {} "{{}}" value. ' \
                       'Should be either "config" or "source" parameters defined'

        # Fill in missing config fields from source stream info
        for field in ['fps', 'width', 'height']:
            if field not in self.config:
                raise ConfigError(_missing_err.format(field))

        input_params = dict(
            pix_fmt='rgb24',
            s='{}x{}'.format(self.config['width'], self.config['height']),
            framerate=self.config['fps'])

        output_params = dict(
            pix_fmt='yuv420p')

        if 'crf' in self.config:
            output_params['crf'] = self.config['crf']

        if 'preset' in self.config:
            output_params['preset'] = self.config['preset']

        if 'codec' in self.config:
            output_params['codec:v'] = self.config['codec']

        return (
            ffmpeg
            .input('pipe:', format='rawvideo', **input_params)
            .output(self.filename, **output_params)
            .overwrite_output()
            .global_args('-hide_banner')
            .global_args('-loglevel', 'error')
            .global_args('-y')
            .run_async(pipe_stdin=True)
        )

    def startup(self) -> None:
        """Prepare ffmpeg pipe input subprocess and start writing thread"""
        if self._end_event.is_set() and self._process is not None:
            raise VideoWriterError(
                'Can\'t start new <{}> subprocess '
                'while previous not closed!'.format(self))

        self._end_event.clear()

        self._process = self._stream_builder()
        self.log.info('Running: %s', ' '.join(map(shlex.quote, self._process.args)))
        self._writing_thread.start()

    @staticmethod
    def _clean_queue(q: queue.Queue):
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break

    def shutdown(self) -> None:
        """Destroy video writing subprocess and thread"""
        self._end_event.set()

        self._clean_queue(self._queue)

        if self._writing_thread.is_alive():
            self._writing_thread.join(timeout=1)

        if self._process is not None:
            self.log.info('Stopping %s subprocess', self)
            try:
                self._process.stdin.close()
                self._process.wait(timeout=4)
            except subprocess.TimeoutExpired:
                pass

            self.log.info('Waiting for %s process to destroy', self)
            try:
                if not self._process.stdin.closed:
                    self._process.kill()
                    self._process.wait(timeout=4)
            except subprocess.TimeoutExpired:
                self.log.error('Error correctly stop %s subprocess. Terminating...', self)
                self._process.terminate()
            finally:
                self._process = None

        self.log.info('%s successfully destroyed', self)

    def _frames_writing_loop(self) -> None:
        """Get frames from queue and write into video file"""
        while not self._end_event.is_set():
            try:
                draw_frame = self._queue.get(timeout=1)
            except queue.Empty:
                continue

            try:
                self._process.stdin.write(draw_frame.tobytes())
            except subprocess.SubprocessError as err:
                self.log.warning('%s got write frame error: %s. Stop writing...', self, err)
                self._end_event.set()

    def write(self, draw_frame: typ.Union[np.ndarray, PIL.Image.Image]) -> None:
        """Write next frame to the video file"""
        if not self._end_event.is_set() and self._process is None:
            raise VideoWriterError('{} not started!'.format(self))

        if self._end_event.is_set():
            return

        draw_frame = np.array(draw_frame).astype(np.uint8)

        if len(draw_frame.shape) != 3:
            raise VideoWriterError('{} got wrong frame shape {}'.format(self, draw_frame.shape))

        height, width, bpp = draw_frame.shape
        if bpp != 3:
            raise VideoWriterError('{} supports only RGB pixel format'.format(self))

        if (width, height) != (self.config['width'], self.config['height']):
            if self.config['resize']:
                draw_frame = utils.resize(
                    draw_frame, (self.config['width'], self.config['height']))
            else:
                raise VideoWriterError('{} got wrong frame size {}x{} (expected {}x{})'.format(
                    self, width, height, self.config['width'], self.config['height']))

        self._queue.put(draw_frame)