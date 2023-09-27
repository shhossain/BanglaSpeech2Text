from banglaspeech2text.utils import (
    all_models,
    nice_model_list,
    get_model,
    get_wer_value,
    convert_file_size,
    seg_to_bytes,
    split_audio,
)
import os
from pathlib import Path

os.environ["HF_HOME"] = os.getenv(
    "BANGLASPEECH2TEXT_CACHE_DIR", str(Path.home() / ".banglaspeech2text")
)
CACHE_DIR = Path(os.getenv("HF_HOME", str(Path.home() / ".banglaspeech2text")))
CACHE_DIR.mkdir(exist_ok=True, parents=True)


from pprint import pformat
from typing import Optional, Union
import io
import warnings
import numpy as np

import requests

from speech_recognition import AudioData
import transformers
import re
import yaml
import json
from pydub import AudioSegment
from io import BytesIO


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


class Model:
    def __init__(self, name: str, **kw):
        self.kw = kw
        self.raw_name = name
        local = False
        if "/" in name:
            if not os.path.exists(name):
                self.name = name
                self.author = name.split("/")[0]
            else:
                local = True
        elif "\\" in name:
            if os.path.exists(name):
                local = True
        else:
            bst = get_model(name)
            self.name = bst["name"]
            self.author = bst["author"]
            last_part = bst["url"].split("/")[-2:]
            self.raw_name = "/".join(last_part)

        if local:
            self.name = name
            self.author = "local"
            local = True

        self.cache_path = CACHE_DIR
        self.save_name = f"models--{self.raw_name.replace('/', '--')}"

        if kw.get("load_pipeline", "True") == "True":
            self.pipeline = transformers.pipeline(
                task="automatic-speech-recognition", model=self.raw_name, **kw
            )

        model_dir = self.cache_path / "hub" / self.save_name
        self.model_path = model_dir
        snapshots = model_dir / "snapshots"
        if snapshots.exists():
            folders = snapshots.glob("*")
            if folders:
                latest_folder = sorted(folders, key=os.path.getmtime)[-1]
                self.model_path = latest_folder

        self.__MAX_WER_SCORE = 1000
        self._type: str = ""
        self._license: str = ""
        self._description: str = ""
        self._url: str = ""
        self._wer: float = self.__MAX_WER_SCORE
        self._size: str = ""
        self._lang: str = ""

        self.load_details()

    def load_details(self, force_reload=False) -> None:
        details_path = self.model_path / "details.json"
        data: dict = safe_json(str(details_path), data=None)  # type: ignore
        if not data or force_reload:
            data = {}

            mdl = get_model(self.raw_name, raise_error=False)
            if mdl is not None:
                data["type"] = mdl["type"]
                data["license"] = mdl["license"]
                data["description"] = mdl["description"]
                data["url"] = mdl["url"]
                data["wer"] = mdl["wer"]
                data["size"] = mdl["size"]
                data["lang"] = mdl["lang"]
            else:
                if "base" in self.name:
                    data["type"] = "base"
                elif "large" in self.name:
                    data["type"] = "large"
                elif "tiny" in self.name:
                    data["type"] = "tiny"
                elif "small" in self.name:
                    data["type"] = "small"
                elif "medium" in self.name:
                    data["type"] = "medium"
                else:
                    data["type"] = "unknown"

                data["url"] = f"https://huggingface.co/{self.raw_name}"
                files = self.model_path.glob("*")
                model_file = [f for f in files if "_model" in f.name]
                if model_file:
                    model_file = model_file[0]
                    data["size"] = convert_file_size(model_file.stat().st_size)
                else:
                    data["size"] = convert_file_size(0)

                try:
                    url = f"{data['url']}/raw/main/README.md"
                    res = requests.get(url)
                    if res.status_code == 200:
                        text = res.text
                        pattern = r"---\n(.*?)\n---"
                        mtc = re.search(pattern, text, re.DOTALL)
                        data["description"] = text
                        if mtc:
                            ydata = mtc.group(1)
                            pardata = yaml.safe_load(ydata)
                            data["license"] = pardata.get("license", "unknown")
                            data["lang"] = pardata.get("language", "bn")
                        else:
                            raise Exception("No match found")

                        data["wer"] = get_wer_value(text, max_wer=self.__MAX_WER_SCORE)

                except Exception as e:
                    data["license"] = "unknown"
                    data["lang"] = "bn"
                    data["wer"] = self.__MAX_WER_SCORE
                    data["description"] = "No description found"

            safe_json(str(details_path), read=False, data=data)

        self._type = data["type"]
        self._license = data["license"]
        self._description = data["description"]
        self._url = data["url"]
        self._wer = data["wer"]
        self._size = data["size"]
        self._lang = data["lang"]
    

    def __repr__(self):
        return f"Model(name={self.name}, type={self._type})"

    def __str__(self):
        txt = f"Model: {self.name}\n"
        txt += f"Type: {self._type}\n"
        txt += f"Author: {self.author}\n"
        txt += f"License: {self._license}\n"
        txt += f"Size: {self._size}\n"
        txt += f"WER: {self._wer}\n"
        txt += f"URL: {self._url}\n"
        return txt

    # Removed methods
    def recognize(self, audio) -> None:
        raise NotImplementedError(
            """This method is removed. Use Speech2Text class instead.\n\nExamples:
            >>> from banglaspeech2text import Speech2Text
            >>> stt = Speech2Text()
            >>> stt.recognize("test.wav")
            
            >>> stt = Speech2Text("tiny")
            >>> stt.recognize("test.wav") """
        )

    def __call__(self, audio) -> None:
        self.recognize(audio)

    def transcribe(self, audio) -> None:
        self.recognize(audio)


class Models:
    def __init__(self) -> None:
        self.models = all_models

    def __str__(self) -> str:
        return f"{nice_model_list()}\n\nFor more models, visit https://huggingface.co/models?pipeline_tag=automatic-speech-recognition&language=bn&sort=likes"

    def __repr__(self) -> str:
        return pformat(all_models)

    def __getitem__(self, key: int) -> str:
        # 0 for tiny, 1 for small, 2 for medium, 3 for base, 4 for large
        models = ["tiny", "small", "medium", "base", "large"]
        if key < 0 or key > 4:
            raise IndexError("Index out of range. Index must be between 0 and 4")
        return models[key]


class Speech2Text:
    def __init__(
        self,
        model: str = "base",
        cache_path: Optional[str] = None,
        use_gpu: bool = True,
        **kw,
    ):  # type: ignore
        """
        Speech to text model
        Args:
            model (str, optional): Model name. Defaults to "base".
            use_gpu (bool, optional): Use GPU or not. Defaults to auto detect.
        **kw:
            Keyword arguments are passed to the transformers pipeline
        Examples:
            >>> from bangla_stt import Speech2Text
            >>> stt = Speech2Text()
            >>> stt.recognize("test.wav")
            >>>
            >>> stt = Speech2Text("tiny")
            >>> stt.recognize("test.wav")
        """

        # if kw.get("device", None) is None and kw.get("device_map", None) is None:
        #     if use_gpu:
        #         kw["device"] = "cuda:0"
        #     else:
        #         kw["device"] = "cpu"

        if cache_path is not None:
            warnings.warn(
                'cache_path is removed. Use os.environ["BANGLASPEECH2TEXT_CACHE_DIR"] to set cache path'
            )

        if not use_gpu and "device" not in kw and "device_map" not in kw:
            kw["device"] = "cpu"

        self.kw = kw
        self.model = Model(model, **kw)
        self.use_gpu = use_gpu

    @property
    def cache_path(self) -> str:
        return str(self.model.cache_path)
    
    @property
    def model_path(self) -> str:
        return str(self.model.model_path)

    @property
    def model_name(self) -> str:
        return self.model.name

    @property
    def model_author(self) -> str:
        return self.model.author

    @property
    def model_type(self) -> str:
        return self.model._type

    @property
    def model_license(self) -> str:
        return self.model._license

    @property
    def model_description(self) -> str:
        return self.model._description

    @property
    def model_url(self) -> str:
        return self.model._url

    @property
    def model_wer(self) -> float:
        return self.model._wer

    @property
    def model_size(self) -> str:
        return self.model._size

    @property
    def model_lang(self) -> str:
        return self.model._lang

    @property
    def model_details(self) -> str:
        return self.model.__str__()

    @property
    def pipeline(self) -> transformers.Pipeline:
        return self.model.pipeline

    def reload_model_details(self, force_reload=True) -> None:
        """
        Reload model details from huggingface.co
        Args:
            force_reload: If True, ignore cache and reload the details
        """
        self.model.load_details(force_reload=force_reload)

    def transcribe(self, audio_path: str, *args, **kw):
        """
        Transcribe an audio file to text
        Args:
            audio_path (str): Path to the audio file

        Check recognize method for more arguments
        Returns:
            str: Transcribed text
        """
        warnings.warn("transcribe is deprecated. Use recognize instead")
        return self.recognize(audio_path, *args, **kw)

    def _preprocess(
        self,
        audio: Union[bytes, np.ndarray, str, AudioData, AudioSegment, BytesIO],
        split,
        convert_func=None,
    ) -> Union[bytes, np.ndarray, AudioSegment, str]:
        data: Union[np.ndarray, bytes, AudioSegment, str] = None  # type: ignore

        if convert_func is not None:
            audio = convert_func(audio)

        if isinstance(audio, AudioData):
            wav_data = audio.get_wav_data(convert_rate=16000)
            f = io.BytesIO(wav_data)
            data = f.getvalue()
            f.close()

        elif isinstance(audio, str):
            if not split:
                data = audio
            else:
                data = AudioSegment.from_file(audio)

        elif isinstance(audio, np.ndarray):
            data = audio

        elif isinstance(audio, AudioSegment):
            data = audio
            if not split:
                data = seg_to_bytes(data)

        elif isinstance(audio, BytesIO):
            data = audio.getvalue()

        elif isinstance(audio, bytes):
            data = audio

        else:
            try:
                path = str(audio)
                if os.path.exists(path):
                    if not split:
                        data = path
                    else:
                        data = AudioSegment.from_file(path)
            except Exception as e:
                pass

            raise TypeError(
                "Invalid audio type. Must be one of str, bytes, np.ndarray, AudioData, AudioSegment, BytesIO, Path like object or provide a convert_func"
            )

        return data

    def recognize(
        self,
        audio: Union[bytes, np.ndarray, str, AudioData, AudioSegment, BytesIO],
        split: bool = False,
        min_silence_length: float = 1000,
        silence_threshold: float = 16,
        padding: int = 300,
        convert_func=None,
        *args,
        **kw,
    ) -> Union[str, list[str]]:
        """
        Recognize an audio to text.

        Args:
            audio (str, bytes, np.ndarray, AudioData, AudioSegment, BytesIO):
                str: Path to the audio file
                bytes or BytesIO: Audio data in bytes
                np.ndarray: Audio data in numpy array
                AudioData: AudioData object from SpeechRecognition library
                AudioSegment: AudioSegment object from pydub library

            split (bool, optional): Split audio into chunks. Defaults to False.
                min_silence_length (float, optional): Minimum silence length in ms. Defaults to 1000
                silence_threshold (float, optional): Average db of audio minus this value is considered as silence. Defaults to 16
                padding (int, optional): Pad beginning and end of splited audio by this ms. Defaults to 300

            convert_func (function, optional): Function to convert audio to supported types. Defaults to None.

            Extra arguments are passed to the transformers pipeline
        Returns:
            str: Transcribed text
            list[str] if split is True
        """

        data = self._preprocess(audio, split, convert_func)
        if split:
            segments = split_audio(data, min_silence_length, silence_threshold, padding)
            segments = [seg_to_bytes(seg) for seg in segments]
            results = self.pipeline(segments, *args, **kw)  # type: ignore
            results: list[str] = [result["text"] for result in results]  # type: ignore

            return results

        return self.pipeline(data, *args, **kw)["text"]  # type: ignore

    def generate(
        self,
        audio: Union[bytes, np.ndarray, str, AudioData, AudioSegment, BytesIO],
        min_silence_length: float = 1000,
        silence_threshold: float = 16,
        padding: int = 300,
        convert_func=None,
        *args,
        **kw,
    ):
        """
        Generate text from an audio.

        Args:
            audio (str, bytes, np.ndarray, AudioData, AudioSegment, BytesIO):
                str: Path to the audio file
                bytes or BytesIO: Audio data in bytes
                np.ndarray: Audio data in numpy array
                AudioData: AudioData object from SpeechRecognition library
                AudioSegment: AudioSegment object from pydub library

            split (bool, optional): Split audio into chunks. Defaults to True.
                min_silence_length (float, optional): Minimum silence length in ms. Defaults to 1000
                silence_threshold (float, optional): Average db of audio minus this value is considered as silence. Defaults to 16
                padding (int, optional): Pad beginning and end of splited audio by this ms. Defaults to 300

            convert_func (function, optional): Function to convert audio to supported types. Defaults to None.

            Extra arguments are passed to the transformers pipeline

        Examples:
            >>> for text in stt.generate_text("audio.wav"):
            >>>     print(text)
        """

        data = self._preprocess(audio, True, convert_func)
        segments = split_audio(data, min_silence_length, silence_threshold, padding)

        for seg in segments:
            yield self.pipeline(seg, *args, **kw)["text"]  # type: ignore

    def generate_text(self, *args, **kw):
        warnings.warn("generate_text is deprecated. Use generate instead")
        return self.generate(*args, **kw)

    def __call__(
        self,
        audio: Union[bytes, np.ndarray, str, AudioData, AudioSegment, BytesIO],
        *args,
        **kw,
    ) -> Union[str, list[str]]:
        """
        Recognize an audio to text.
        Args:
            audio (str, bytes, np.ndarray, AudioData, AudioSegment, BytesIO):
                str: Path to the audio file
                bytes or BytesIO: Audio data in bytes
                np.ndarray: Audio data in numpy array
                AudioData: AudioData object from SpeechRecognition library
                AudioSegment: AudioSegment object from pydub library

            split (bool, optional): Split audio into chunks. Defaults to False.
                min_silence_length (float, optional): Minimum silence length in ms. Defaults to 500
                silence_threshold (float, optional): Silence threshold in dBFS. Defaults to -16
                padding (int, optional): Pad beginning and end of splited audio by this ms. Defaults to 300
                text_divider (str, optional): Divide output text by this string. Defaults to newline

            convert_func (function, optional): Function to convert audio to supported types. Defaults to None.

            Extra arguments are passed to the transformers pipeline
        Returns:
            str: Transcribed text
        """
        return self.recognize(audio, *args, **kw)

    def __repr__(self) -> str:
        return f"Speech2Text(model={self.model_name}, use_gpu={self.use_gpu})"

    def __str__(self) -> str:
        return self.model_details

    @staticmethod
    def list_models() -> Models:
        """
        List all available models
        Returns:
            List of models
        """
        return Models()


__all__ = ["Speech2Text", "Models"]

if __name__ == "__main__":
    print(Models())
