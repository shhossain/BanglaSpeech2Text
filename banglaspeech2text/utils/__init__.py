import subprocess
import elevate
import platform
from tqdm.auto import tqdm
import requests
import zipfile
import os
import ctypes
import shutil
import warnings
import sys
from pathlib import Path



pcache_dir = Path.home() / ".banglaspeech2text" 
cache_dir: str = os.getenv("BANGLASPEECH2TEXT_CACHE_DIR", str(pcache_dir))
os.makedirs(cache_dir, exist_ok=True)



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


def check_ffmpeg_health(exe_path) -> bool:
    # check if exe is corupts or not
    try:
        subprocess.run([exe_path, "-version"], stdout=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False


def download_ffmpeg() -> None:
    pbar = tqdm(total=2, desc="Installing ffmpeg")
    if platform.system() == "Windows":
        url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
        path = os.path.join(cache_dir, "ffmpeg-release-essentials.zip")
        ffmpeg_path = os.path.join(cache_dir, "ffmpeg-release-essentials")
        ffmpeg_exe_path = None

        for _ in range(3):
            if not os.path.exists(ffmpeg_path):
                if not os.path.exists(path):
                    headers = {
                        "Accept": "*/*",
                        "Connection": "keep-alive",
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36 OPR/52.0.2871.40",
                    }
                    response = requests.get(url, headers=headers, stream=True)
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

                # check if zip file is corrupted
                try:
                    with zipfile.ZipFile(path, "r") as zip_ref:
                        zip_ref.extractall(ffmpeg_path)
                except zipfile.BadZipFile:
                    os.remove(path)
                    continue

                os.remove(path)

                # recursively search for ffmpeg.exe
                for root, dirs, files in os.walk(ffmpeg_path):
                    if "ffmpeg.exe" in files:
                        ffmpeg_exe_path = os.path.join(root, "ffmpeg.exe")
                        break

            if ffmpeg_exe_path is None:
                shutil.rmtree(ffmpeg_path)
            else:
                if check_ffmpeg_health(ffmpeg_exe_path):
                    break
                else:
                    shutil.rmtree(ffmpeg_path)
                    ffmpeg_exe_path = None

        pbar.update(1)

        if ffmpeg_exe_path is None:
            raise FileNotFoundError(
                "Error while installing ffmpeg: ffmpeg.exe not found. Please install ffmpeg manually. See https://ffmpeg.org/download.html for more info."
            )

        python_path = sys.exec_prefix

        scripts_path = os.path.join(python_path, "scripts")

        # copy ffmpeg.exe to scripts folder
        shutil.copy(ffmpeg_exe_path, scripts_path)
        pbar.update(1)

    else:
        # check for active package managers
        package_managers = ["apt-get", "brew", "apt", "dnf", "yum", "pacman"]
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
        if not is_root() and package_manager != "brew":
            elevate.elevate(graphical=False)

        pbar.update(1)
        cmd = [package_manager, "install", "ffmpeg", "-y"]
        if package_manager == "brew":
            cmd.pop()

        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL)
        except Exception as e:
            raise RuntimeError(
                f"Error while installing ffmpeg: {e}\nPlease install ffmpeg manually. See https://ffmpeg.org/download.html for more info."
            )

        pbar.update(1)

    pbar.close()


if not ffmpeg_installed():
    warnings.warn("ffmpeg not found. Trying to install it automatically.")
    download_ffmpeg()


from banglaspeech2text.utils.models import all_models, nice_model_list, get_model
from banglaspeech2text.utils.helpers import (
    safe_name,
    get_wer_value,
    convert_file_size,
    audiosegment_to_librosawav,
    seg_to_bytes,
    split_audio,
    get_generation_model,
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
    "get_generation_model",
]
