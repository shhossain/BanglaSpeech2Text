import subprocess
import elevate
import platform
from tqdm.auto import tqdm
import requests
import zipfile
import os


def ffmpeg_installed() -> bool:
    check_path = os.path.join(get_cache_dir(), "ffmpeg_installed")
    if os.path.exists(check_path):
        return True

    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL)
        with open(check_path, "w") as f:
            f.write("1")
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
        progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
        with open(path, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()

        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(ffmpeg_path)
        os.remove(path)

        # recursively search for ffmpeg.exe
        bin_path = None
        for root, dirs, files in os.walk(ffmpeg_path):
            if "ffmpeg.exe" in files:
                bin_path = root
                break

        if bin_path is None:
            raise RuntimeError(
                "Error while installing ffmpeg: ffmpeg.exe not found. Please install ffmpeg manually. See https://ffmpeg.org/download.html for more info."
            )

        elevate.elevate(show_console=False)
        cmd = [
            "setx",
            "PATH",
            f"%PATH%;{bin_path}",
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL)
        except Exception as e:
            raise RuntimeError(
                f"Error while installing ffmpeg: {e}\nPlease install ffmpeg manually. See https://ffmpeg.org/download.html for more info."
            )

        # update current session
        os.environ["PATH"] += f";{bin_path}"

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
        elevate.elevate(graphical=False)
        cmd = [package_manager, "install", "-y", "ffmpeg"]

        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL)
        except Exception as e:
            raise RuntimeError(
                f"Error while installing ffmpeg: {e}\nPlease install ffmpeg manually. See https://ffmpeg.org/download.html for more info."
            )

        # update current by sourcing .bashrc
        cmd = ["source", "~/.bashrc"]
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL)
        except Exception as e:
            raise RuntimeError(f"Restart your terminal to use this package.")


if ffmpeg_installed():
    download_ffmpeg()


import re
import shutil
from typing import Union
import warnings
import numpy as np
from pydub import AudioSegment
from pydub.silence import split_on_silence
import io


def safe_name(name, author) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-\.]", "", f"{name}-/{author}")


def get_cache_dir() -> str:
    return os.path.join(os.path.expanduser("~"), ".banglaspeech2text")


def get_wer_value(text, max_wer=1000) -> float:
    pattern = r"(?:wer)[:\s]+(\d+(?:\.\d+)?)"
    wer_values = re.findall(pattern, text, re.IGNORECASE)
    if wer_values:
        return float(wer_values[0])
    else:
        return max_wer


def convert_file_size(size_bytes, decimal_places=1) -> str:
    units = ["bytes", "KB", "MB", "GB", "TB"]
    size = size_bytes
    unit_index = 0

    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1

    return f"~{round(size, decimal_places)} {units[unit_index]}"


def audiosegment_to_librosawav(audiosegment):
    channel_sounds = audiosegment.split_to_mono()
    samples = [s.get_array_of_samples() for s in channel_sounds]

    fp_arr = np.array(samples).T.astype(np.float32)
    fp_arr /= np.iinfo(samples[0].typecode).max
    fp_arr = fp_arr.reshape(-1)
    return fp_arr


def seg_to_bytes(seg: AudioSegment) -> bytes:
    wav_data = io.BytesIO()
    seg = seg.set_channels(1).set_frame_rate(16000)
    seg.export(wav_data, format="wav")
    return wav_data.getvalue()


def split_audio(
    data: Union[bytes, np.ndarray, AudioSegment, str],
    min_silence_length: float = 1000,
    silence_threshold: float = 16,
    padding: int = 300,
) -> list[AudioSegment]:
    if isinstance(data, np.ndarray):
        warnings.warn(
            "Using numpy array for longer audio is not recommended. This may cause error. Use str, bytes, AudioData, AudioSegment or BytesIO instead."
        )
        segment = AudioSegment(
            data.tobytes(),
            frame_rate=16000,
            sample_width=data.dtype.itemsize,
            channels=1,
        )
    elif isinstance(data, bytes):
        segment = AudioSegment.from_file(io.BytesIO(data))
    else:
        segment = data

    segments: list[AudioSegment] = split_on_silence(
        segment,
        min_silence_len=min_silence_length,  # type: ignore
        silence_thresh=segment.dBFS - abs(silence_threshold),  # type: ignore
        keep_silence=padding,
    )

    return segments


__all__ = [
    "safe_name",
    "get_cache_dir",
    "get_wer_value",
    "convert_file_size",
    "audiosegment_to_librosawav",
    "seg_to_bytes",
    "split_audio",
]
