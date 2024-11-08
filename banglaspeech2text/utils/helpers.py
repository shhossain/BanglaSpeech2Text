import re
from typing import Union
import warnings
import numpy as np
from pydub import AudioSegment
from pydub.silence import split_on_silence
import io


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
    byt = wav_data.getvalue()
    wav_data.close()
    return byt


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



def get_generation_model(model_name: str) -> str:
    mapping = {
        "tiny": "openai/whisper-tiny",
        "base": "openai/whisper-base",
        "small": "openai/whisper-small",
        "medium": "openai/whisper-medium",
        "large-v3-turbo": "openai/whisper-large-v3-turbo",
        "large-v3": "openai/whisper-large-v3",
        "large-v2": "openai/whisper-large-v2",
        "large": "openai/whisper-large",
    }

    for key in mapping:
        if key in model_name:
            return mapping[key]
    
    return model_name