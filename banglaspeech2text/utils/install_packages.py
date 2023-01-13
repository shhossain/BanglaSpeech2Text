from banglaspeech2text.utils import get_app_path, logger, app_name
import subprocess
import os
from pySmartDL import SmartDL
import subprocess
import os
import ctypes
import platform
from elevate import elevate


def is_admin():
    if platform.system() == 'Windows':
        try:
            return ctypes.windll.shell32.IsUserAnAdmin()
        except:
            return False
    else:
        return os.getuid() == 0  # type: ignore


def admin_required(func):
    def wrapper(*args, **kwargs):
        if not is_admin():
            elevate()
        return func(*args, **kwargs)
    return wrapper


win_url = r"https://github.com/git-for-windows/git/releases/download/v2.39.0.windows.2/Git-2.39.0.2-64-bit.exe"


def install_git_windows():
    if not os.path.exists("C:\\Program Files\\Git"):
        logger.info("Installing Git for Windows")
        path = os.path.join(get_app_path(app_name), "Git.exe")
        if not os.path.exists(path):
            obj = SmartDL(win_url, path)
            obj.start()
        subprocess.run([path])
        logger.info("Git for Windows installed successfully")
    else:
        logger.info("Git for Windows already installed")


@admin_required
def install_git_linux():
    logger.info("Installing Git for Linux")
    subprocess.run(["sudo", "apt", "install", "git", "-y"])
    subprocess.run(["sudo", "apt", "install", "git-lfs", "-y"])
    logger.info("Git for Linux installed successfully")


__all__ = ["install_git_windows", "install_git_linux"]
