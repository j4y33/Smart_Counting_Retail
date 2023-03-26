
class SpyglassError(Exception):
    """Base framework error."""


class ConfigError(SpyglassError):
    """Configuration error"""


class ApplicationError(SpyglassError):
    """Application load error"""


class ServiceExit(SpyglassError):
    """Signal to stop the application"""


class UnitError(SpyglassError):
    """Unit error"""


class VideoSourceError(SpyglassError):
    """Video source error"""


class VideoWriterError(SpyglassError):
    """Video writer error"""


class VideoDisplayError(SpyglassError):
    """Video display error"""


class AIModelError(SpyglassError):
    """Common error for any AIModel error"""


class WeightsLoaderError(AIModelError):
    """Load wights error"""
