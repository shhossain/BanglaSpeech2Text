import json
import os
import re
from typing import Optional, Union
import tempfile

APP_NAME = "banglaspeech2text"


def get_app_dir() -> str:
    path = os.path.join(os.path.expanduser("~"), f".{APP_NAME}")
    os.makedirs(path, exist_ok=True)
    return path


def get_app_temp_dir() -> str:
    path = os.path.join(tempfile.gettempdir(), APP_NAME)
    os.makedirs(path, exist_ok=True)
    return path


def safe_name(name, author) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-\.]", "", f"{name}-/{author}")


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


def safe_json(
    file_path: str, read: bool = True, data: Optional[dict] = None
) -> Union[dict, None, bool]:
    if read:
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    if not data:
                        return None
                    return data
            except Exception as e:
                return None
        else:
            return None
    else:
        try:
            with open(file_path, "w") as f:
                json.dump(data, f)
                return True
        except Exception as e:
            return False
