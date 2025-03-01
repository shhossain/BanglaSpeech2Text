import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BanglaSpeech2Text")


from banglaspeech2text.speech2text import Speech2Text


__all__ = ["Speech2Text"]
