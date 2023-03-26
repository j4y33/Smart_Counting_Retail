import abc
import random
import attr
import ffmpeg
import numpy as np
import trafaret as t
import typing as typ
import datetime as dt
from pathlib import Path
from fractions import Fraction
from loguru import logger
from enum import Enum
from queue import Queue
from .error import (
    ConfigError, SpyglassError, UnitError, VideoSourceError)


class _EnumWithNames(Enum):
    def __repr__(self):
        return str(self)

    @classmethod
    def names(cls) -> typ.List[str]:
        return list(cls.__members__.keys())


class SourceType(_EnumWithNames):
    FILE = 'FILE'
    FILES = 'FILES'
    STREAM = 'STREAM'
    CAMERA = 'CAMERA'
    OTHER = "OTHER"

    @classmethod
    def resolve(cls, source: str) -> 'SourceType':
        """Define source type"""
        if isinstance(source, list):
            return cls.FILES
        if source.startswith('/dev/video') or source.startswith('/device'):
            return cls.CAMERA
        elif any(source.startswith('{}://'.format(schema)) for schema in ['http', 'https', 'rtsp']):
            return cls.STREAM
        elif Path(source).exists():
            return cls.FILE
        return cls.OTHER

    @property
    def is_live(self) -> bool:
        """Checks if source_type is live"""
        return self in {self.STREAM, self.CAMERA}


class PixFmt(_EnumWithNames):
    RGB = 'RGB'
    RGBA = 'RGBA'
    BGR = 'BGR'
    BGRA = 'BGRA'
    GRAY = 'GRAY'
    GRAY16 = 'GRAY16'

    def bpp(self):
        """Bits per pixel"""
        return {
            PixFmt.RGB: 24,
            PixFmt.RGBA: 32,
            PixFmt.BGR: 24,
            PixFmt.BGRA: 32,
            PixFmt.GRAY: 8,
            PixFmt.GRAY16: 16,
        }[self]


class _ABCMeta(abc.ABCMeta):
    """Unit abstract Meta class."""

    _sub_classes = list()

    def __new__(mcs, name: str, bases: tuple, namespace: dict):
        current_cls = super().__new__(mcs, name, bases, namespace)
        if bases:
            mcs._sub_classes.append(current_cls)
        return current_cls

    @classmethod
    def sub_classes(cls) -> typ.List[typ.Type['Model']]:
        """Return Unit interface implementation classes"""
        return cls._sub_classes.copy()


class Model(metaclass=_ABCMeta):
    """ Interface for Units, that could be injected in pipeline """

    _config_schema = t.Dict(allow_extra='*')    # should be redefined in the inherited class

    def __init__(self, config: dict):
        try:
            config = self._config_schema.check(config or {})
        except (t.DataError, ValueError) as err:
            raise ConfigError(
                'Wrong VideoSource configuration '
                'for {} ({}): {}'.format(self, config, err))

        self._config = config

    def __str__(self) -> str:
        """Representation string."""
        return self.__class__.__name__

    @property
    def config(self) -> dict:
        """Unit config."""
        return self._config

    @abc.abstractmethod
    def process_single(self, image: np.ndarray, **extra):
        """Model method executor."""

    def on_startup(self) -> None:
        """Additional operations to do after model startup"""

    def on_shutdown(self):
        """Additional operations to do before model shutdown"""

    def startup(self):
        """Unit driver initialization"""
        logger.info(f'Starting {self} model')

        try:
            self.on_startup()
        except SpyglassError as err:
            raise err
        except Exception as err:
            err_msg = f"{self} model startup error: {err}"
            logger.error(err_msg)
            raise UnitError(err_msg)

        logger.info(f"{self} started")

    def shutdown(self):
        """Model shutdown and cleanup"""
        logger.info(f'Shutdown {self} unit')
        try:
            self.on_shutdown()
        except SpyglassError as err:
            logger.error(f'{self} model shutdown error:{err}. Skipping...')
        except Exception as err:
            logger.error(f'{self} model shutdown error:{err}. Skipping...')

    def __enter__(self):
        self.startup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


Rect = typ.NamedTuple(
    'Rect', [
        ('width', int),
        ('height', int)])


@attr.s(slots=True, frozen=True)
class SourceInfo:
    """Origin Video Source raw information.
    Documentation:
        - codec: https://en.wikipedia.org/wiki/Comparison_of_video_codecs
        - container: https://en.wikipedia.org/wiki/Comparison_of_video_container_formats
    :param source: video source path/url
    :param codec: name of the codec to decode video stream (e.g. h264)
    :param container: name of the video container (e.g. mp4)
    :param width: origin video width
    :param height: origin video height
    :param pix_fmt: origin video pixel format (e.g. yuv420p)
    :param fps: origin stream FPS (e.g. Fraction(24) or Fraction('30000/1001'))
    """
    source = attr.ib()                 # type: str
    width = attr.ib()                  # type: int
    height = attr.ib()                 # type: int
    fps = attr.ib()                    # type: Fraction
    pix_fmt = attr.ib(default='')      # type: str
    codec = attr.ib(default='')        # type: str
    container = attr.ib(default='')    # type: str

    @property
    def size(self) -> Rect:
        return Rect(self.width, self.height)

    def as_dict(self, recursive: bool = True) -> dict:
        return attr.asdict(self, recurse=recursive)


@attr.s(slots=True, frozen=True)
class StreamInfo:
    """Real stream info.
    StreamInfo is SourceInfo after all manipulations (described in VideoSource config)
    It represents real parameters that is used to produce frames.
    Used as common values for all Frames.
    E.g.:
        - OriginInfo.fps = 24
        - Video Source option {fps: 1}
        - StreamInfo.fps = 1
    :param stream_id: unique stream identifier
    :param width: frame width
    :param height: frame heights
    :param bpp: bits per pixel (e.g. RGB -> 24, RGBA -> 32)
    :param fps: number of frames/sec produced by Video Source (e.g. Fraction(24) or Fraction('30000/1001'))
    :param pix_fmt: format of the pixel (RGB, RGBA)
    :param command: escaped stream run command (in case when we use shell cmd to run stream)
    :param start_time: timestamp when video started or recorded
    :param source: origin video source information
    :param size: (width, height) named tuple
    """
    stream_id = attr.ib()     # type: str
    width = attr.ib()         # type: int
    height = attr.ib()        # type: int
    bpp = attr.ib()           # type: int
    fps = attr.ib()           # type: Fraction
    pix_fmt = attr.ib()       # type: PixFmt
    command = attr.ib()       # type: str
    start_time = attr.ib()    # type: dt.datetime
    source = attr.ib()        # type: SourceInfo

    @property
    def size(self) -> Rect:
        return Rect(self.width, self.height)

    def as_dict(self, recursive: bool = True) -> dict:
        return attr.asdict(self, recurse=recursive)


@attr.s(slots=True, frozen=True)
class Frame(StreamInfo):
    """Video frame container.
    :param frame: numpy frame data with shape[width, height, pixels]
    :param counter: real number of frames extracted from video source
    :param offset: number of frames from the beginning of file or stream start (config[frames_offset] + counter)
    :param meta: additional meta information about frame
    :param duration_msec: how long one frame will be displayed in the video (1000 / fps)
    :param elapsed_msec: time elapsed from the beginning (duration_msec * offset)
    :param timestamp: stream or video file timestamp (strat time + elapsed_msec)
    """
    frame = attr.ib()      # type: np.ndarray
    counter = attr.ib()    # type: int
    offset = attr.ib()     # type: int

    meta = attr.ib(default=dict())  # type: typ.Dict[str, typ.Any]

    @property
    def duration_msec(self) -> float:
        """Frame duration in milliseconds"""
        return float(1000 / self.fps)

    @property
    def elapsed_msec(self) -> float:
        """Number of milliseconds elapsed from the beginning"""
        return self.duration_msec * self.offset

    @property
    def timestamp(self) -> dt.datetime:
        """Frame timestamp from the beginning"""
        return self.start_time + dt.timedelta(milliseconds=self.elapsed_msec)


class LeakyQueue(Queue):
    """Queue that contains only the last actual items and drops the oldest one."""

    def __init__(self, maxsize: int = 100, on_drop: typ.Optional[typ.Callable[['LeakyQueue', Frame], None]] = None):
        super().__init__(maxsize=maxsize)
        self._dropped = 0
        self._on_drop = on_drop or (lambda queue, item: None)

    def put(self, item, block=True, timeout=None):
        if self.full():
            dropped_item = self.get_nowait()
            self._dropped += 1
            self._on_drop(self, dropped_item)
        super().put(item, block=block, timeout=timeout)

    @property
    def dropped(self):
        return self._dropped


class VideoSource(metaclass=_ABCMeta):
    """Interface for VideoSource, that reads frames for pipeline"""

    _config_schema = t.Dict(allow_extra='*')    # should be redefined in the inherited class

    def __init__(self, source: str, config: typ.Optional[dict] = None):
        self._stream_id = '{:04X}'.format(random.randrange(2**16))

        try:
            config = self._config_schema.check(config or {})
        except (t.DataError, ValueError) as err:
            raise ConfigError(
                'Wrong VideoSource configuration '
                'for {} ({}): {}'.format(self, source, err))

        self._source = source
        self._source_type = SourceType.resolve(source)
        self._config = config
        self._stream_info = None

    @property
    def source(self) -> str:
        return self._source

    @property
    def source_type(self) -> SourceType:
        return self._source_type

    @property
    def config(self) -> dict:
        return self._config

    @property
    def stream_id(self) -> str:
        return self._stream_id

    @property
    def stream_info(self) -> typ.Optional[StreamInfo]:
        return self._stream_info

    def __str__(self) -> str:
        return '{} [{}]'.format(self.__class__.__name__, self._stream_id)

    @abc.abstractmethod
    def read(self) -> typ.Optional[Frame]:
        """Reads and returns frame"""

    def startup(self):
        """Initialization"""

    def shutdown(self):
        """Shutdown and cleanup"""

    def __enter__(self):
        self.startup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    def __iter__(self):
        """Iterate over frames while not end"""
        while True:
            frame = self.read()
            if frame is None:
                return
            yield frame


def read_source_info(source: str, ffprobe_cmd='ffprobe') -> SourceInfo:
    """Get original source video information"""
    try:
        probe = ffmpeg.probe(source, cmd=ffprobe_cmd)

        stream_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        return SourceInfo(
            source=source,
            codec=stream_info['codec_name'],
            container=probe['format']['format_name'],
            width=stream_info['width'],
            height=stream_info['height'],
            pix_fmt=stream_info['pix_fmt'],
            fps=Fraction(stream_info['avg_frame_rate'])
        )

    except (ffmpeg.Error, KeyError, ValueError, StopIteration) as err:
        raise VideoSourceError(
            'Error get video information from "{}": {}'.format(source, err))


class Unit(metaclass=_ABCMeta):

    """ Interface for Units, that could be injected in pipeline """

    _config_schema = t.Dict(allow_extra='*')    # should be redefined in the inherited class

    def __init__(self, config: dict, resources: typ.Dict[str, typ.Any]):
        # TODO: types checks
        try:
            config = self._config_schema.check(config or {})
        except (t.DataError, ValueError) as err:
            raise ConfigError(
                'Wrong VideoSource configuration '
                'for {} ({}): {}'.format(self, config, err))

        self._config = config
        self._resources = resources

    @property
    def config(self) -> dict:
        """Unit config."""
        return self._config

    @property
    def models(self) -> typ.Dict[str, Model]:
        """AIModels mapping Dict(model_id: Model instance)."""
        return self._resources['models']

    @property
    def stream_info(self) -> StreamInfo:
        return self._resources['stream_info']

    @abc.abstractmethod
    def process(self, frame: Frame, data: dict):
        """Unit process method executor."""

    def on_startup(self) -> None:
        """Additional operations to do after unit startup"""

    def on_shutdown(self):
        """Additional operations to do before unit shutdown"""

    def startup(self):
        """Unit driver initialization"""
        logger.info(f'Starting {self} unit')

        try:
            self.on_startup()
        except SpyglassError as err:
            raise err
        except Exception as err:
            err_msg = f"{self} unit startup error: {err}"
            logger.error(err_msg)
            raise UnitError(err_msg)

        logger.info(f"{self} started")

    def __str__(self) -> str:
        """Representation string."""
        return self.__class__.__name__

    def shutdown(self):
        """Unit driver shutdown and cleanup"""
        logger.info(f'Shutdown {self} unit')
        try:
            self.on_shutdown()
        except SpyglassError as err:
            logger.error(f'{self} unit shutdown error:{err}. Skipping...')
        except Exception as err:
            logger.error(f'{self} unit shutdown error:{err}. Skipping...')

    def __enter__(self):
        self.startup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
