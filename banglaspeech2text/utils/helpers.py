import re
import os

def safe_name(name,author) -> str:
    return re.sub(r'[^a-zA-Z0-9_\-\.]', '', f"{name}-/{author}")


def get_cache_dir() -> str:
    return os.path.join(os.path.expanduser("~"), ".banglaspeech2text")

def get_wer_value(text, max_wer=1000) -> float:
    pattern = r'(?:wer)[:\s]+(\d+(?:\.\d+)?)'
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


__all__ = ["safe_name", "get_cache_dir", "get_wer_value", "convert_file_size"]