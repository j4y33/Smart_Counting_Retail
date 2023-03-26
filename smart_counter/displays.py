import queue
import logging
import threading
import typing as typ

from loguru import logger

import PIL.Image
import numpy as np
import trafaret as t

from . import utils
from .error import ConfigError, VideoDisplayError

try:
    import cv2
except ImportError:
    cv2 = None


class VideoDisplay:
    """Video display based on OpenCV lib.

    Options:
        - title - window title (default: empty)
        - width - display width size (default: get from frame)
        - height - display height size (default: get from frame))
    """
    _config_schema = t.Dict({
        t.Key('title', default=''): t.String(allow_blank=True),
        t.Key('width', optional=True): t.Int(gte=16),
        t.Key('height', optional=True): t.Int(gte=16),
    }, ignore_extra='*')

    def __init__(self, config: typ.Optional[dict] = None):
        try:
            config = self._config_schema.check(config or {})
        except (t.DataError, ValueError) as err:
            raise ConfigError(
                'Wrong {} configuration: {}'.format(self, err))

        self._config = config

        self._queue = queue.Queue(maxsize=2)
        self._end_event = threading.Event()
        self._display_thread = threading.Thread(target=self._frames_display_loop)

    @property
    def config(self) -> dict:
        return self._config

    def __str__(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return '<{}>'.format(self)

    def __enter__(self):
        self.startup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    def startup(self) -> None:
        """Start video display thread"""
        logger.info('Starting {self}')
        self._end_event.clear()
        self._display_thread.start()

    @staticmethod
    def _clean_queue(q: queue.Queue):
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break

    def shutdown(self) -> None:
        """Destroy video display thread and close window"""
        self._end_event.set()
        self._clean_queue(self._queue)

        if self._display_thread.is_alive():
            self._display_thread.join(timeout=1)

        if cv2:
            cv2.destroyAllWindows()

        logger.info(f'{self} successfully destroyed')

    def _frames_display_loop(self) -> None:
        """Display frames from queue"""
        while not self._end_event.is_set():
            try:
                draw_frame = self._queue.get(timeout=1)
            except queue.Empty:
                continue

            if cv2:
                cv2.imshow(self.config['title'], cv2.cvtColor(draw_frame, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

    def display(self, draw_frame: typ.Union[np.ndarray, PIL.Image.Image]) -> None:
        """Display next video frame"""
        if cv2 is None or self._end_event.is_set():
            return

        draw_frame = np.array(draw_frame).astype(np.uint8)

        if len(draw_frame.shape) != 3:
            raise VideoDisplayError('{} got wrong frame shape {}'.format(self, draw_frame.shape))

        height, width, bpp = draw_frame.shape
        if bpp != 3:
            raise VideoDisplayError('{} supports only RGB pixel format'.format(self))

        display_size = (self.config.get('width'), self.config.get('height'))
        if all(display_size) and (width, height) != display_size:
            draw_frame = utils.resize(draw_frame, display_size)

        self._queue.put(draw_frame)
