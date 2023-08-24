from banglaspeech2text.utils import (
    all_models,
    nice_model_list,
    safe_name,
    get_cache_dir,
    get_model,
    get_wer_value,
    convert_file_size,
    seg_to_bytes,
    split_audio,
)


from pprint import pformat
from typing import Optional, Union
import io
import warnings
import numpy as np

import requests
import os
from speech_recognition import AudioData
import transformers
import re
import yaml
import json
from pydub import AudioSegment
from io import BytesIO


class Model:
    def __init__(self, name: str, cache_path: Optional[str] = None, **kw):
        self.kw = kw
        self.raw_name = name
        local = False
        if "/" in name:
            if not os.path.exists(name):
                self.name = name
                self.author = name.split("/")[0]
                self.save_name = safe_name(name.split("/")[1], self.author)
            else:
                self.name = name
                self.author = "local"
                self.save_name = name
                local = True
        else:
            bst = get_model(name)
            self.name = bst["name"]
            self.author = bst["author"]
            self.save_name = safe_name(self.name, self.author)
            last_part = bst["url"].split("/")[-2:]
            self.raw_name = "/".join(last_part)

        # fix save path
        self.cache_dir = get_cache_dir()
        cache_dir_models = os.path.join(self.cache_dir, "models")
        if not os.path.exists(cache_dir_models):
            os.makedirs(cache_dir_models)

        if not local:
            if cache_path is None or not cache_path:
                cache_path = os.path.join(cache_dir_models, self.save_name)
            else:
                cache_path = os.path.join(str(cache_path), "models", self.save_name)
        else:
            cache_path = name

        # check if model is downloaded
        self.cache_path = cache_path

        if kw.get("load_pipeline", True):
            if not os.path.exists(cache_path):
                self.pipeline = self._get_pipeline(cache_path)
            else:
                self.pipeline = transformers.pipeline(
                    task="automatic-speech-recognition", model=cache_path, **kw
                )

        self.__MAX_WER_SCORE = 1000

        self._type: str = ""
        self._license: str = ""
        self._description: str = ""
        self._url: str = ""
        self._wer: float = self.__MAX_WER_SCORE
        self._size: str = ""
        self._lang: str = ""

        self.load_details()

    def _get_pipeline(self, cache_path: str) -> transformers.Pipeline:
        pipeline = transformers.pipeline
        pipe = pipeline(
            task="automatic-speech-recognition", model=self.raw_name, **self.kw
        )
        pipe.save_pretrained(cache_path)
        return pipe

    def load_details(self, force_reload=False) -> None:
        details_path = os.path.join(self.cache_path, "details.json")
        if not os.path.exists(details_path) or force_reload:
            mdl = get_model(self.raw_name, raise_error=False)
            if mdl is not None:
                self._type = mdl["type"]
                self._license = mdl["license"]
                self._description = mdl["description"]
                self._url = mdl["url"]
                self._wer = mdl["wer"]
                self._size = mdl["size"]
                self.save_details(details_path)
            else:
                if "base" in self.name:
                    self._type = "base"
                elif "large" in self.name:
                    self._type = "large"
                elif "tiny" in self.name:
                    self._type = "tiny"
                elif "small" in self.name:
                    self._type = "small"
                elif "medium" in self.name:
                    self._type = "medium"
                else:
                    self._type = "unknown"

                self._url = f"https://huggingface.co/{self.raw_name}"

                # check cache path files and get the file with "model" in it and get the size. 3.06 GB = ~3.1 GB, 346 MB = ~350 MB
                files = os.listdir(self.cache_path)
                for file in files:
                    if "_model" in file:
                        file_path = os.path.join(self.cache_path, file)
                        size = os.path.getsize(file_path)
                        self._size = convert_file_size(size)
                        break

                try:
                    url = f"{self._url}/raw/main/README.md"
                    res = requests.get(url)
                    if res.status_code == 200:
                        text = res.text
                        self._description = text

                        pattern = r"---\n(.*?)\n---"
                        data = re.search(pattern, text, re.DOTALL)

                        if data:
                            data = data.group(1)
                            pardata = yaml.safe_load(data)
                            self._license = pardata.get("license", "unknown")
                            self._lang = pardata.get("language", None)
                            self._wer = get_wer_value(
                                text, max_wer=self.__MAX_WER_SCORE
                            )

                except Exception as e:
                    pass

                self.save_details(details_path)

        else:
            with open(details_path, "r") as f:
                details = json.load(f)
            self._type = details["type"]
            self._license = details["license"]
            self._description = details["description"]
            self._url = details["url"]
            self._wer = details["wer"]
            self._size = details["size"]
            self._lang = details["lang"]

    def save_details(self, details_path: str) -> None:
        details = {
            "type": self._type,
            "license": self._license,
            "description": self._description,
            "url": self._url,
            "wer": self._wer,
            "size": self._size,
            "lang": self._lang,
        }
        with open(details_path, "w") as f:
            json.dump(details, f)

    @property
    def type(self) -> str:
        return self._type

    @property
    def license(self) -> str:
        return self._license

    @property
    def description(self) -> str:
        return self._description

    @property
    def url(self) -> str:
        return self._url

    @property
    def wer(self) -> float:
        return self._wer

    @property
    def size(self) -> str:
        return self._size

    @property
    def lang(self) -> str:
        return self._lang

    def __repr__(self):
        return f"Model(name={self.name}, type={self.type})"

    def __str__(self):
        txt = f"Model: {self.name}\n"
        txt += f"Type: {self.type}\n"
        txt += f"Author: {self.author}\n"
        txt += f"License: {self.license}\n"
        txt += f"Size: {self.size}\n"
        txt += f"WER: {self.wer}\n"
        txt += f"URL: {self.url}\n"
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
        use_gpu: bool = False,
        **kw,
    ):  # type: ignore
        """
        Speech to text model
        Args:
            model (str, optional): Model name. Defaults to "base".
            cache_path (str, optional): Cache path to store the model. Defaults to None.
            use_gpu (bool, optional): Use GPU or not. Defaults to False.
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

        if kw.get("device", None) is None and kw.get("device_map", None) is None:
            if use_gpu:
                kw["device"] = "cuda:0"
            else:
                kw["device"] = "cpu"

        self.kw = kw
        self.model = Model(model, cache_path=cache_path, **kw)
        self.use_gpu = use_gpu

    @property
    def cache_path(self) -> str:
        return self.model.cache_path

    @property
    def model_name(self) -> str:
        return self.model.name

    @property
    def model_author(self) -> str:
        return self.model.author

    @property
    def model_type(self) -> str:
        return self.model.type

    @property
    def model_license(self) -> str:
        return self.model.license

    @property
    def model_description(self) -> str:
        return self.model.description

    @property
    def model_url(self) -> str:
        return self.model.url

    @property
    def model_wer(self) -> float:
        return self.model.wer

    @property
    def model_size(self) -> str:
        return self.model.size

    @property
    def model_lang(self) -> str:
        return self.model.lang

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

    def _pipeline_recognize(self, audio, *args, **kw) -> str:
        return self.pipeline(audio, *args, **kw)["text"]  # type: ignore

    def transcribe(self, audio_path: str, *args, **kw) -> str:
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

        else:
            raise TypeError(
                "Invalid audio type. Must be one of str, bytes, np.ndarray, AudioData, AudioSegment, BytesIO or provide a convert_func"
            )

        return data

    def recognize(
        self,
        audio: Union[bytes, np.ndarray, str, AudioData, AudioSegment, BytesIO],
        split: bool = False,
        min_silence_length: float = 1000,
        silence_threshold: float = 16,
        padding: int = 300,
        text_divider: str = "\n",
        convert_func=None,
        *args,
        **kw,
    ) -> str:
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
                text_divider (str, optional): Divide output text by this string. Defaults to newline

            convert_func (function, optional): Function to convert audio to supported types. Defaults to None.

            Extra arguments are passed to the transformers pipeline
        Returns:
            str: Transcribed text
        """

        data = self._preprocess(audio, split, convert_func)
        if split:
            segments = split_audio(data, min_silence_length, silence_threshold, padding)
            text = ""
            for seg in segments:
                text += (
                    self._pipeline_recognize(seg_to_bytes(seg), *args, **kw)
                    + text_divider
                )
            return text

        return self._pipeline_recognize(data, *args, **kw)

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
                text_divider (str, optional): Divide output text by this string. Defaults to newline

            convert_func (function, optional): Function to convert audio to supported types. Defaults to None.

            Extra arguments are passed to the transformers pipeline

        Examples:
            >>> for text in stt.generate_text("audio.wav"):
            >>>     print(text)
        """

        data = self._preprocess(audio, True, convert_func)
        segments = split_audio(data, min_silence_length, silence_threshold, padding)

        for seg in segments:
            yield self._pipeline_recognize(seg_to_bytes(seg), *args, **kw)

    def generate_text(self, *args, **kw):
        warnings.warn("generate_text is deprecated. Use generate instead")
        return self.generate(*args, **kw)

    def __call__(
        self,
        audio: Union[bytes, np.ndarray, str, AudioData, AudioSegment, BytesIO],
        *args,
        **kw,
    ) -> str:
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
        return f"""Speech2Text(model={self.model_name}, use_gpu={self.use_gpu})
        {self.model_details}"""

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
