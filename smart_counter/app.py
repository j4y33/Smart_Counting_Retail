import signal
import typing as typ
import threading

from .config import settings
from .base import Model, Unit, VideoSource
from .logging import logger
from .utils import (
    camel_to_snake_upper,
    dict_to_lower_keys
)

from .error import (
    SpyglassError,
    ConfigError,
    ServiceExit,
    VideoSourceError
)


_Stream = typ.NamedTuple(
    'Stream', [
        ('source', VideoSource),
        ('pipeline', typ.List[Unit])])


class Application:

    def __init__(self):

        # FIXME dicts are the same and represents all sub classes for _ABCMeta base class
        self._models_cls_by_name = {m.__name__: m for m in Model.sub_classes()}
        self._unit_cls_by_name = {u.__name__: u for u in Unit.sub_classes()}
        self._source_cls_by_name = {u.__name__: u for u in VideoSource.sub_classes()}

        self._stream = None
        self._resources = {}

        self._end_event = threading.Event()
        self._shutdown_reason = 'All processes completed'

    def _startup(self):
        self._resources['settings'] = {
            "camera_id": settings.camera_id
        }
        video_source_cls = self._source_cls_by_name[settings.video_source.type]
        video_source_cfg = dict_to_lower_keys(settings.video_source.config.to_dict())
        video_source = video_source_cls(settings.video_source.config.source, video_source_cfg)

        models = {}
        if hasattr(settings, 'models'):
            for model_type, model_config in settings.models.items():
                cfg = dict_to_lower_keys(model_config)
                model_cls_name = cfg.pop('type')
                models[model_type.lower()] = self._models_cls_by_name[model_cls_name](cfg)

        self._resources['models'] = models

        units = []
        for unit_name in settings.pipeline:
            unit_name_snake_upper = camel_to_snake_upper(unit_name)
            unit_config = settings.units.get(unit_name_snake_upper)

            unit_cls, cfg = self._unit_cls_by_name[unit_name], {}
            if unit_config:
                cfg = dict_to_lower_keys(unit_config)
                unit_cls_name = cfg.get("type", unit_name)
                unit_cls = self._unit_cls_by_name[unit_cls_name]

            units.append(unit_cls(config=cfg, resources=self._resources))

        video_source.startup()
        self._resources['stream_info'] = video_source._stream_info

        items = list(models.values()) + units
        for item in items:
            item.startup()

        self._stream = _Stream(source=video_source, pipeline=units)

    def __str__(self) -> str:
        return self.__class__.__name__

    def _shutdown(self):
        self._stream.source.shutdown()

        for unit in self._stream.pipeline:
            unit.shutdown()

        for model in self._resources['models'].values():
            model.shutdown()

        logger.info(f"{self} shutdown done")

    def _signal_handler(self, signum, frame):
        raise ServiceExit(f'Shutdown signal: {signum}')

    def _terminating(self) -> bool:
        """Check is app terminating"""
        return self._end_event.is_set()

    def terminate(self, reason: str) -> None:
        """Set end event and reason to terminate threads"""
        logger.info('Got terminate request: {reason}')
        if not self._terminating():
            self._shutdown_reason = reason
            self._end_event.set()

    def _run_pipeline(self) -> None:
        logger.info('Stream loop started')

        error_counter = 0
        last_error = Exception('Unknown error')
        max_attempts = settings.app_config.attempts_before_shutdown

        counter = 0
        while not self._terminating():

            if error_counter >= max_attempts:
                err_msg = f'Stream got too many errors: {last_error}'
                logger.error(err_msg)
                self.terminate(err_msg)
                break

            try:
                frame = self._stream.source.read()  # Frame
                if frame is None:
                    break

                data = dict()
                for unit in self._stream.pipeline:
                    if self._terminating():
                        break
                    unit.process(frame, data)

                error_counter = 0

            except VideoSourceError as err:
                err_msg = 'Stream got video source error: {}. ' \
                          'Terminating...'.format(err)
                logger.error(err_msg)
                self.terminate(err_msg)
                break

            except SpyglassError as err:
                logger.exception(f'Stream error: {err}')
                if "Shutdown signal" in str(err):
                    raise ServiceExit(str(err))
                error_counter += 1
                last_error = err

        logger.info('Stream loop complete')

    def run(self) -> None:
        """Run application"""

        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        try:
            logger.info('Starting application')
            self._startup()

        except ServiceExit as err:
            logger.info('Applcation got terminate request during startup')
            self.terminate(str(err))
            return

        except ConfigError as err:
            logger.error(f'Configuration error: {err}')
            return

        except SpyglassError as err:
            logger.error(
                f'Got {err.__class__.__name__} error during application startup: {err}')
            return

        try:

            self._run_pipeline()

        except ServiceExit as err:
            self.terminate(str(err))

        except Exception as err:
            logger.exception(f'Unexpected main process error: {err}')
            self.terminate(f'Unexpected error: {err}')

        finally:
            logger.info(
                f'Shutdown Spyglass application (reason: {self._shutdown_reason})')

            self._shutdown()

        logger.info('Done')
