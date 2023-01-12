import logging
from banglaspeech2text.utils.download_path import get_app_path

app_name = "BanglaSpeech2Text"

# create a logger
logger = logging.getLogger(__name__)
# logger.setLevel(logging.CRITICAL)

# create a file handler
_handler = logging.FileHandler('{}.log'.format(app_name))
_handler.setLevel(logging.DEBUG)

# create a logging format
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
_handler.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(_handler)

# app_path = get_app_path(app_name)

models_download_repo = "https://media.githubusercontent.com/media/shhossain/whisper_bangla_models"


__all__ = [
    'get_app_path', 'logger', 'models_download_repo', 'app_name'
]
