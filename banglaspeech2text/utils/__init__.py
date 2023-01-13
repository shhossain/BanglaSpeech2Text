import logging
from banglaspeech2text.utils.download_path import get_app_path

app_name = "BanglaSpeech2Text"

FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(format=FORMAT)

# create a logger
logger = logging.getLogger(app_name)
# logger.setLevel(logging.CRITICAL)


# app_path = get_app_path(app_name)

models_download_repo = "https://raw.githubusercontent.com/shhossain/whisper_bangla_models/"


__all__ = [
    'get_app_path', 'logger', 'models_download_repo', 'app_name'
]
