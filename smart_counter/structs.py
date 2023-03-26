import typing as typ
from random import randint
import datetime as dt
from uuid import uuid4
from enum import Enum

import attr
import numpy as np

from .base import _EnumWithNames


def uuid_id_factory() -> typ.Callable:
    """Unique track id"""
    def id_gen() -> str:
        return uuid4().hex
    return id_gen


def format_time(datetime: dt.datetime, *, timezone: dt.timezone = None) -> str:
    dt_ = datetime.replace(tzinfo=timezone) if timezone else datetime.astimezone()
    return dt.datetime.isoformat(dt_, timespec="microseconds")


def random_id_factory(prefix='', size: int = 8) -> typ.Callable:
    def id_gen() -> str:
        template = '{}{{:0{}X}}'.format(prefix, size)
        return template.format(randint(0, 2 ** (size * 4)))
    return id_gen


def sequential_id_factory(prefix='', size: int = 8) -> typ.Callable:
    counter = 0

    def id_gen() -> str:
        nonlocal counter
        counter += 1
        template = '{}{{:0{}d}}'.format(prefix, size)
        return template.format(counter)

    return id_gen


Point = typ.NamedTuple(
    'Point', [
        ('x', int),
        ('y', int)])


KeyPoint = typ.NamedTuple(
    'KeyPoint', [
        ('x', int),
        ('y', int),
        ('score', float)])


BoundingBox = typ.NamedTuple(
    'BoundingBox', [
        ('x', int),
        ('y', int),
        ('w', int),
        ('h', int)])


class Side(_EnumWithNames):
    FRONT = 'front'
    BACK = 'back'
    UNDEFINED = 'side'


@attr.s(slots=True, frozen=True)
class Detection:

    id_factory = uuid_id_factory()                      # type: typ.ClassVar[typ.Callable]

    det_id = attr.ib(init=False)                        # type: str
    keypoints = attr.ib()                               # type: Dict[str, Keypoint]
    box = attr.ib()                                     # type: BoundingBox
    score = attr.ib()                                   # type: float
    side = attr.ib()                                    # type: Side

    @det_id.default
    def det_id_factory(self) -> str:
        return self.__class__.id_factory()

    @property
    def top(self) -> Point:
        """Returns top-center point of object"""
        return Point(*[self.box.x + self.box.w // 2, self.box.y])

    def as_dict(self, recursive: bool = True) -> dict:
        return attr.asdict(self, recurse=recursive)


TrackObject = typ.NamedTuple(
    'TrackObject', [
        ('detection', Detection),
        ('offset', int),
        ('frame_timestamp', dt.datetime)])


@attr.s(slots=True, frozen=False)
class Track:

    id_factory = uuid_id_factory()                      # type: typ.ClassVar[typ.Callable]

    track_id = attr.ib(init=False)                      # type: str
    track_objects = attr.ib(init=False, factory=list)   # type: typ.List[TrackObject]

    max_embeddings_size = attr.ib()                     # type: int
    _embeddings = attr.ib(init=False, factory=list)     # type: typ.List[np.ndarray]

    @track_id.default
    def track_id_factory(self) -> str:
        return self.__class__.id_factory()

    @property
    def embeddings(self) -> typ.List[np.ndarray]:
        return self._embeddings

    @property
    def first(self) -> typ.Optional[TrackObject]:
        return self.track_objects[0] if self.track_objects else None

    @property
    def last(self) -> typ.Optional[TrackObject]:
        """Last registered TrackObject if any else None"""
        return self.track_objects[-1] if self.track_objects else None

    @property
    def length(self) -> int:
        """Number of objects in the track"""
        return len(self.track_objects)

    # FIXME: probably track line shoudn't be here
    def track_line(self) -> typ.List[Point]:
        """Track points"""
        return [obj.detection.top for obj in self.track_objects]

    def add_embedding(self, embedding: np.ndarray) -> 'Track':
        """
        add new embedding to the embedding list
        """
        self._embeddings.append(embedding)
        if len(self._embeddings) > self.max_embeddings_size:
            self._embeddings = self._embeddings[1:]
        return self

    def add(self, detection: Detection,
            offset: int,
            timestamp: dt.datetime) -> 'Track':
        """Add track object to the track.
        :param detection: detected object assigned to the track
        :param offset: frame offset
        :param timestamp: frame timestamp
        """
        self.track_objects.append(TrackObject(detection, offset, timestamp))
        return self
