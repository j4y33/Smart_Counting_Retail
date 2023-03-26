import sys
from loguru import logger
from .config import settings

logger.remove()  # remove default logger

logger.add(sys.stderr,
           format=settings.logging.format,
           level=settings.logging.level)
