from typing import Optional, Union
from banglaspeech2text.utils import app_name, logger, get_app_path
from banglaspeech2text.utils.download_models import ModelType, get_model, available_models, ModelDict
import os
import torch
from speech_recognition import AudioData
from uuid import uuid4
from threading import Thread


class Model:
    def __init__(self, model: Union[str, ModelType, ModelDict] = ModelType.base, download_path=None, device: Optional[Union[int, str, "torch.device"]] = None, force=False, verbose=False, **kwargs):
        """
        Args:
            model_name_or_type (str or ModelType): Model name or type 
            download_path (str): Path to download model
            device (str): Device to use for inference (cpu, cuda, cuda:0, cuda:1, etc.)
            force (bool): Force download model
            verbose (bool): Verbose mode

        **kwargs are passed to transformers.pipeline
        See more at https://huggingface.co/transformers/main_classes/pipelines.html#transformers.pipeline
        """

        if verbose:
            logger.setLevel("INFO")
        else:
            logger.setLevel("ERROR")

        if download_path is not None:
            if not os.path.exists(download_path):
                raise ValueError(f"{download_path} does not exist")
            os.environ[app_name] = download_path

        self.model: ModelDict = None  # type: ignore
        if isinstance(model, ModelDict):
            self.model = model
        else:
            self.model = get_model(model, force=force)

        self.device = device
        self.kwargs = kwargs

        self.task = "automatic-speech-recognition"
        self.pipe = None

    def load(self):
        logger.info("Loading model")
        from transformers import pipeline

        if not self.model.is_downloaded():
            self.model.download()

        self.pipe = pipeline(self.task, model=self.model.path, device=self.device, **self.kwargs)
 # type: ignore
    @property
    def available_models(self):
        return available_models()

    def __get_wav_from_audiodata(self, data: AudioData):
        temp_audio_file = f"{uuid4()}.wav"
        path = os.path.join(get_app_path(app_name), temp_audio_file)

        with open(path, "wb") as f:
            f.write(data.get_wav_data())

        return path

    def transcribe(self, audio_file) -> dict:
        data: dict = self.pipe(audio_file)  # type: ignore
        Thread(target=os.remove, args=(audio_file,)).start()
        return data

    def recognize(self, audio) -> dict:
        if isinstance(audio, AudioData):
            audio = self.__get_wav_from_audiodata(audio)
        return self.transcribe(audio)  # type: ignore

    def __call__(self, audio) -> dict:
        return self.recognize(audio)

    def __repr__(self):
        return f"Model(name={self.model.name}, type={self.model.type})"

    def __str__(self):
        return self.__repr__()


__all__ = [
    "Model",
    "available_models",
    "ModelType",
]

if __name__ == "__main__":
    model = Model()
    model.load()
    print(model.available_models)
    print(model)

    # audio_file = "data/audios/0.wav"
