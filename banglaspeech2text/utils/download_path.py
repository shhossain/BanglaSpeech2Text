# Determine Download Path
import os
import sys


def get_user_path():
    """Get user home path"""
    if "BanglaSpeech2Text" in os.environ:
        if os.environ['BanglaSpeech2Text']:
            return os.environ['BanglaSpeech2Text']

    if sys.platform == 'win32':
        return os.path.expanduser('~')
    else:
        return os.path.expanduser('~/Downloads')


def get_app_path(app_name: str) -> str:
    path = os.path.join(get_user_path(), app_name)
    if not os.path.exists(path):
        os.mkdir(path)
    return path
