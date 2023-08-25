import subprocess
import elevate
import platform
from tqdm.auto import tqdm
import requests
import zipfile
import os
import ctypes
import shutil


def get_cache_dir() -> str:
    return os.path.join(os.path.expanduser("~"), ".banglaspeech2text")


def is_root() -> bool:
    if platform.system() == "Windows":
        return ctypes.windll.shell32.IsUserAnAdmin() != 0
    else:
        return os.geteuid() == 0  # type: ignore


def ffmpeg_installed() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False


def download_ffmpeg() -> None:
    if platform.system() == "Windows":
        url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
        cache_dir = get_cache_dir()
        path = os.path.join(cache_dir, "ffmpeg-release-essentials.zip")
        ffmpeg_path = os.path.join(cache_dir, "ffmpeg-release-essentials")

        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024 * 1024  # 1 MB
        progress_bar = tqdm(
            total=total_size_in_bytes,
            unit="iB",
            unit_scale=True,
            desc="Downloading ffmpeg",
        )
        with open(path, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()

        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(ffmpeg_path)
        os.remove(path)

        # recursively search for ffmpeg.exe
        ffmpeg_exe_path = None
        for root, dirs, files in os.walk(ffmpeg_path):
            if "ffmpeg.exe" in files:
                ffmpeg_exe_path = os.path.join(root, "ffmpeg.exe")
                break

        if ffmpeg_exe_path is None:
            raise RuntimeError(
                "Error while installing ffmpeg: ffmpeg.exe not found. Please install ffmpeg manually. See https://ffmpeg.org/download.html for more info."
            )

        python_path = None
        possible_paths = ["python", "python3", "py"]
        for path in possible_paths:
            if shutil.which(path):
                python_path = path
                break

        if python_path is None:
            raise RuntimeError(
                "Error while installing ffmpeg: Python path not found. Please install ffmpeg manually. See https://ffmpeg.org/download.html for more info."
            )

        scripts_path = os.path.join(python_path, "scripts")

        # copy ffmpeg.exe to scripts folder
        shutil.copy(ffmpeg_exe_path, scripts_path)

    else:
        # check for active package managers
        package_managers = ["apt-get", "brew", "dnf", "yum", "zypper", "pacman"]
        package_manager = None
        for pm in package_managers:
            if shutil.which(pm):
                package_manager = pm
                break

        if package_manager is None:
            raise RuntimeError(
                "No package manager found. Please install ffmpeg manually. See https://ffmpeg.org/download.html for more info."
            )

        # install ffmpeg
        if not is_root():
            elevate.elevate(graphical=False)
        cmd = [package_manager, "install", "-y", "ffmpeg"]

        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL)
        except Exception as e:
            raise RuntimeError(
                f"Error while installing ffmpeg: {e}\nPlease install ffmpeg manually. See https://ffmpeg.org/download.html for more info."
            )

        # update current session
        possible_paths = [
            "/usr/bin/ffmpeg",
            "/usr/local/bin/ffmpeg",
            "/snap/bin/ffmpeg",
        ]

        for path in possible_paths:
            if os.path.exists(path):
                os.environ["PATH"] += f":{path}"
                break


if not ffmpeg_installed():
    download_ffmpeg()


from banglaspeech2text.utils.models import all_models, nice_model_list, get_model
from banglaspeech2text.utils.helpers import (
    safe_name,
    get_wer_value,
    convert_file_size,
    audiosegment_to_librosawav,
    seg_to_bytes,
    split_audio,
)


__all__ = [
    "all_models",
    "nice_model_list",
    "get_model",
    "safe_name",
    "get_wer_value",
    "convert_file_size",
    "audiosegment_to_librosawav",
    "seg_to_bytes",
    "split_audio",
]
