import logging
import sys

# Create logger with proper handler
logger = logging.getLogger("BanglaSpeech2Text")
logger.setLevel(logging.INFO)

# Check if handlers already exist to avoid duplicates
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

from banglaspeech2text.speech2text import Speech2Text

__all__ = ["Speech2Text"]
