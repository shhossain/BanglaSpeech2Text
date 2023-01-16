# Determine Download Path
import os
app_name = "BanglaSpeech2Text"

def get_user_path():
    """Get user home path"""
    if app_name in os.environ:
        # print('app_name in os.environ')
        if not os.path.exists(os.environ[app_name]):
            os.makedirs(os.environ[app_name])
        return os.environ[app_name]

    return os.path.expanduser("~")


def get_app_path(app_name: str) -> str:
    path = os.path.join(get_user_path(), app_name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path
