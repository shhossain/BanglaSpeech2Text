from io import BytesIO
from pathlib import Path
import random
from typing import Any, BinaryIO, Iterable, Literal, Optional, Union, overload
import logging
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment
from numpy import ndarray
from banglaspeech2text.utils.converter import get_ct2_model_path
from banglaspeech2text.utils.helpers import get_app_temp_dir
from banglaspeech2text.utils.models import BanglaASRModels, ModelMetadata
import torch

# Get a child logger that inherits from the main logger
logger = logging.getLogger("BanglaSpeech2Text.speech2text")


class Speech2Text(WhisperModel):
    def __init__(
        self,
        model_size_or_path: str = "large",
        device="auto",
        device_index=0,
        compute_type="default",
        cpu_threads=0,
        num_workers=1,
        skip_conversion=False,
        ct_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        self.model_metadata = ModelMetadata(model_size_or_path)
        logger.info(f"Initializing Speech2Text with model: {model_size_or_path}")

        if compute_type == "default":
            compute_type = "float16" if torch.cuda.is_available() else "int8"
            logger.info(f"Using compute type: {compute_type}")

        self.model_path = model_size_or_path
        if not skip_conversion:
            self.model_path = get_ct2_model_path(
                self.model_metadata.raw_name,
                self.model_metadata.cache_path,
                compute_type,
            )

        super().__init__(
            self.model_path,
            device,
            device_index,
            compute_type,
            cpu_threads,
            num_workers,
            **kwargs,
            **(ct_kwargs or {}),
        )

    @overload
    def recognize(
        self,
        audio: Any,
        return_segments: Literal[False],
        **kw,
    ) -> str: ...

    @overload
    def recognize(
        self,
        audio: Any,
        return_segments: Literal[True],
        **kw,
    ) -> Iterable[Segment]: ...

    @overload
    def recognize(
        self,
        audio: Any,
        **kw,
    ) -> str: ...

    def recognize(
        self,
        audio: Any,
        return_segments: bool = False,
        **kw,
    ) -> Union[Iterable[Segment], str]:

        if "language" not in kw:
            kw["language"] = "bn"

        audio = self._preprocess(audio)
        segments, _ = self.transcribe(
            audio, append_punctuations="\"'.。,，!！?？:：”)]}、।", **kw
        )

        if return_segments:
            return segments
        else:
            return "".join([segment.text for segment in segments])

    def _preprocess(self, audio: Any) -> Union[str, BinaryIO, ndarray]:
        class_name = audio.__class__.__name__
        temp_file = Path(get_app_temp_dir()) / f"{random.randint(0, 100000)}.wav"
        if class_name == "AudioData":  # from speech_recognition
            with temp_file.open("wb") as f:
                f.write(audio.get_wav_data())
        elif class_name == "AudioSegment":  # from pydub
            audio = audio.set_frame_rate(16000)
            audio.export(temp_file, format="wav")
        elif isinstance(audio, bytes):
            with temp_file.open("wb") as f:
                f.write(audio)
        elif isinstance(audio, BytesIO):
            with temp_file.open("wb") as f:
                f.write(audio.read())

        elif isinstance(audio, Path):
            temp_file = audio

        if temp_file.exists():
            audio = str(temp_file)

        if not (
            isinstance(audio, str)
            or isinstance(audio, ndarray)
            or isinstance(audio, ndarray)
        ):
            raise ValueError("Invalid audio input")

        return audio

    def __call__(
        self, audio: Any, return_segments: bool = False, **kw
    ) -> Union[Iterable[Segment], str]:
        """
        Shorthand for `recognize` method.
        """
        return self.recognize(audio, return_segments, **kw)

    def __repr__(self):
        return f"Speech2Text(model_path={self.model_path})"

    def __str__(self):
        return f"Speech2Text(model_path={self.model_path})"

    @staticmethod
    def list_models():
        return BanglaASRModels()
