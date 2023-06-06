from pprint import pformat
from typing import  Union
import io
import librosa
import numpy as np
from banglaspeech2text.utils import (
    all_models,
    nice_model_list,
    safe_name,
    get_cache_dir,
    get_model,
    get_wer_value,
    convert_file_size,
)
import requests
import os
from speech_recognition import AudioData
import transformers
import re
import yaml
import json



class Model:
    def __init__(self, name: str, cache_path: str = None, **kw):  # type: ignore
        self.kw = kw
        self.raw_name = name
        if "/" in name:
            self.name = name
            self.author = name.split("/")[0]
            self.save_name = safe_name(name.split("/")[1], self.author)
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

        if cache_path is None:
            cache_path = os.path.join(cache_dir_models, self.save_name)
        else:
            cache_path = os.path.join(str(cache_path), "models", self.save_name)

        # check if model is downloaded
        self.cache_path = cache_path
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
                            self._wer = get_wer_value(text, max_wer=self.__MAX_WER_SCORE)

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
        return f"ModelDict(name={self.name}, type={self.type})"

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
        raise NotImplementedError("""This method is removed. Use Speech2Text class instead.\n\nExamples:
            >>> from bangla_stt import Speech2Text
            >>> stt = Speech2Text()
            >>> stt.recognize("test.wav")
            
            >>> stt = Speech2Text("tiny")
            >>> stt.recognize("test.wav") """)
    
    def __call__(self, audio) -> None:
        self.recognize(audio)
    
    def transcribe(self, audio) -> None:
        self.recognize(audio)
    


class Models:
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
        self, model: str = "base", cache_path: str = None, use_gpu: bool = False, **kw): # type: ignore
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
                kw["device"] = "cuda"
            else:
                kw["device"] = "cpu"

        self.kw = kw
        self.model = Model(model, cache_path=cache_path, **kw)
        self.use_gpu = use_gpu
    
    @property
    def model_name(self) -> str:
        return self.model.name
    
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
    
    def reload_model_details(self, force_reload=False) -> None:
        """
        Reload model details from huggingface.co
        Args:
            force_reload: If True, reload details from huggingface.co
        """
        self.model.load_details(force_reload=force_reload)
    
    

    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe an audio file to text
        Args:
            audio_path (str): Path to the audio file
        Returns:
            str: Transcribed text
        """
        return self.pipeline(audio_path)["text"]  # type: ignore

    def recognize(self, audio: Union[bytes, np.ndarray, str, AudioData]) -> str:
        """
        Recognize an audio to text
        Args:
            audio (str, bytes, np.ndarray, AudioData): Audio to recognize
        Returns:
            str: Transcribed text
        """
        data: np.ndarray = np.array([])
        if isinstance(audio, AudioData):
            wav_data = audio.get_wav_data(convert_rate=16000)
            f = io.BytesIO(wav_data)
            data, _ = librosa.load(f, sr=16000)
        elif isinstance(audio, str):
            data, _ = librosa.load(audio, sr=16000)
        elif isinstance(audio, bytes):
            f = io.BytesIO(audio)
            data, _ = librosa.load(f, sr=16000)
        elif isinstance(audio, np.ndarray):
            data = audio
        else:
            raise TypeError("Invalid audio type. Must be one of str, bytes, np.ndarray, AudioData")

        return self.pipeline(data)["text"]  # type: ignore
    

    def __call__(self, audio: Union[bytes, np.ndarray, str, AudioData]) -> str:
        """
        Recognize an audio to text
        Args:
            audio (str, bytes, np.ndarray, AudioData): Audio to recognize
        Returns:
            str: Transcribed text
        """
        return self.recognize(audio)
    
    def __repr__(self) -> str:
       return f"Speech2Text(model={self.model_name}, use_gpu={self.use_gpu})"
   
    def __str__(self) -> str:
        return f"""Speech2Text(model={self.model_name}, use_gpu={self.use_gpu})
        {self.model_details}"""

    @staticmethod
    def list_models():
        """
        List all available models
        Returns:
            List of models
        """
        return Models()


__all__ = ["Speech2Text", "Models"]

if __name__ == "__main__":
    print(Models())
