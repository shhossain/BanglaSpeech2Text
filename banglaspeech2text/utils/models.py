import json
import os
from pathlib import Path
from pprint import pformat
import re
import requests
import yaml
from banglaspeech2text.utils.helpers import convert_file_size, get_wer_value, safe_json
import threading
from dataclasses import dataclass
import logging

# Get a child logger that inherits from the main logger
logger = logging.getLogger("BanglaSpeech2Text.models")

current_dir = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(current_dir, "listed_models.json"), "r") as f:
    all_models = json.load(f)


@dataclass
class ModelMetadata:
    """Metadata about a model."""

    raw_name: str

    def __post_init__(self):
        self.cache_path = Path(os.path.expanduser("~/.cache/banglaspeech2text"))
        logger.debug(f"Model cache path: {self.cache_path}")

    def __init__(self, name: str, **kw):
        self.kw = kw
        self.raw_name = name
        local = False

        if os.path.exists(name):
            local = True
        elif "/" in name:
            self.name = name
            self.author = name.split("/")[0]
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

        self.cache_path = Path(
            os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        )
        self.save_name = f"models--{self.raw_name.replace('/', '--')}"

        self.__MAX_WER_SCORE = 1000
        self.type: str = ""
        self.license: str = ""
        self.description: str = ""
        self.url: str = ""
        self.wer: float = self.__MAX_WER_SCORE
        self.size: str = ""
        self.lang: str = ""

        model_dir = self.cache_path / "hub" / self.save_name
        self.model_path = model_dir
        snapshots = model_dir / "snapshots"
        if snapshots.exists():
            folders = snapshots.glob("*")
            if folders:
                latest_folder = sorted(folders, key=os.path.getmtime)[-1]
                self.model_path = latest_folder

        threading.Thread(target=self.load_details).start()

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

        self.type = data["type"]
        self.license = data["license"]
        self.description = data["description"]
        self.url = data["url"]
        self.wer = data["wer"]
        self.size = data["size"]
        self.lang = data["lang"]

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


class BanglaASRModels:
    """Available Bangla ASR models."""

    def __init__(self):
        self.models = all_models

    def __call__(self):
        return self.models

    def __str__(self) -> str:
        return f"{nice_model_list()}\n\nFor more models, visit https://huggingface.co/models?pipeline_tag=automatic-speech-recognition&language=bn&sort=likes"

    def __repr__(self) -> str:
        return pformat(all_models)

    def __getitem__(self, key: int) -> str:
        # 0 for tiny, 1 for small, 2 for medium, 3 for base, 4 for large
        models = ["tiny", "small", "medium", "base", "large"]
        if key < 0 or key > 4:
            raise IndexError("Index out of range. Index must be between 0 and 4")
        return get_best_model(models[key])


# nice list of models
def nice_model_list() -> str:
    txt = "Available models:\n"
    for model_type in all_models:
        txt += f"\t{model_type}:\n"
        for model in all_models[model_type]:
            txt += f"\t\t{model['name']}\t{model['wer']} WER\t{model['size']}\tby {model['author']} ({model['license']})\n"
        txt += "\n"
    return txt


def get_best_model(type: str = "base"):
    if type not in all_models:
        raise ValueError(
            f"Model type {type} not found. Please choose from {list(all_models.keys())}"
        )
    return sorted(all_models[type], key=lambda x: x["wer"])[0]


def get_model(name: str, raise_error: bool = True) -> dict:
    for model_type in all_models:
        for model in all_models[model_type]:
            if model["name"] == name:
                return model

    # check if it's type return the best model of that type lower WER
    if name in all_models:
        return get_best_model(name)

    if raise_error:
        raise ValueError(
            f"Model {name} not found. Please choose from:\n{nice_model_list()}"
        )

    return None  # type: ignore
