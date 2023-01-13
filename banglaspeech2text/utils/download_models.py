from typing import List, Optional, Union, overload
from pySmartDL import SmartDL
from banglaspeech2text.utils import get_app_path, logger, models_download_repo, app_name
from enum import Enum
import os
# import zipfile
import requests
from git.repo import Repo
from git.exc import InvalidGitRepositoryError
from pprint import pprint
import shutil

from git import RemoteProgress
from tqdm import tqdm


class CloneProgress(RemoteProgress):
    def __init__(self):
        super().__init__()
        self.pbar = tqdm()

    def update(self, op_code, cur_count, max_count=None, message=''):
        self.pbar.total = max_count
        self.pbar.n = cur_count
        self.pbar.refresh()


class ModelType(Enum):
    tiny = "tiny"
    base = "base"
    small = "small"
    medium = "medium"
    large = "large"

# {
#   "url": "https://github.com/shhossain/whisper_base_bn_sifat",
#   "name": "whisper-base-bn-sifat",
#   "type": "base",
#   "host": "github.com",
#   "cloneable": true,
#   "private": false,
#   "license": "apache-2.0",
#   "author": "Sifat"
# }


class ModelDict:
    def __init__(self, model_dict: Optional[dict] = None) -> None:
        model_dict = model_dict or {}

        self._name = model_dict.get("name",  None)
        self._type = model_dict.get("type",  None)
        self._wer = model_dict.get("wer",  None)
        self._size = model_dict.get("size",  None)
        self._config_url = model_dict.get("config_url",  None)
        self._path = model_dict.get("path",  None)

        self._downloaded = False if self._path is None else True
        self._config = None
        self.repo = None

    def __load_config(self):
        if self._config is None:
            # print(self._config_url,'config_url')
            res = requests.get(self._config_url)  # type: ignore
            self._config = res.json()
        return self._config

    def get_config(self, key):
        return self.__load_config().get(key, None)

    def clone(self, path: str):
        # check if path exists

        url = self.get_config("url")
        if url is None:
            raise ValueError("url is None")

        # check if repo exists
        do_clone = False
        if os.path.exists(path):
            try:
                self.repo = Repo(path)
            except InvalidGitRepositoryError:
                logger.error(f"Invalid git repository at {path}")
                logger.info(f"Removing {path} and cloning again")
                shutil.rmtree(path)
                do_clone = True
        else:
            logger.info(f"Cloning {url} to {path}")
            do_clone = True
        if do_clone:
            self.repo = Repo.clone_from(
                url, path, progress=CloneProgress())  # type: ignore

        self._path = path

    def show_config(self):
        json = self.__load_config()
        pprint(json)
        return json

    def is_downloaded(self):
        return self._downloaded

    def download(self):
        model = get_model(self.name)
        self._path = model.path
        self._downloaded = True

    def __return_val(self, val, name) -> str:
        if val is None:
            raise ValueError(f"{name} is None")
        return val

    @property
    def name(self): return self.__return_val(self._name, "name")
    @property
    def type(self): return self.__return_val(self._type, "type")
    @property
    def wer(self): return self.__return_val(self._wer, "wer")
    @property
    def size(self): return self.__return_val(self._size, "size")
    @property
    def url(self): return self.__return_val(self._config_url, "config_url")
    @property
    def path(self): return self.__return_val(self._path, "path")

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        if key == "name":
            self._name = value
        elif key == "type":
            self._type = value
        elif key == "wer":
            self._wer = value
        elif key == "size":
            self._size = value
        elif key == "url":
            self._config_url = value
        elif key == "path":
            self._path = value

    def __repr__(self):
        return f"Model(name={self.name}, type={self.type}, wer={self.wer}, size={self.size})"

    def __str__(self):
        return f"{self.name}({self.type})"
    


def get_models(url="https://raw.githubusercontent.com/shhossain/whisper_bangla_models/main/all_models.csv", force=False):
    all_models_path = os.path.join(get_app_path(app_name), "all_models.csv")
    if not os.path.exists(all_models_path) or force:
        logger.info("Downloading all_models.csv")
        obj = SmartDL(url, all_models_path)
        obj.start()

    with open(all_models_path, "r") as f:
        models = f.readlines()[1:]

    # name, type, wer, size
    models = [model.strip().split(",") for model in models]

    config_json = "{}/main/{}/config.json"

    # check if url is valid
    r = requests.head(url)
    if r.status_code != 200:
        raise Exception("models_download_repo is not valid")

    models = [
        {
            "name": model[0],
            "type": model[1],
            "wer": model[2],
            "size": model[3],
            "config_url": config_json.format(models_download_repo, model[0])
        } for model in models
    ]

    models = [ModelDict(model) for model in models]
    keep_models = []
    for model in models:
        try:
            model.get_config("url")
            keep_models.append(model)
        except Exception as e:
            logger.error(f"Error loading config for {model.name}: {e}")

    
    return keep_models


class AvailableModels:
    def __init__(self, models):
        self.models = models

    def __str__(self) -> str:
        longest_name = max([len(model['name']) for model in self.models])
        longest_type = max([len(model['type']) for model in self.models])
        longest_wer = max([len(model['wer']) for model in self.models])+1
        longest_size = max([len(model['size']) for model in self.models])

        text = "Available Models:\n"
        text += "Name".ljust(longest_name) + " | "
        text += "Type".ljust(longest_type) + " | "
        text += "WER".ljust(longest_wer) + " | "

        text += "Size(MB)".ljust(longest_size) + "\n"
        text += "-" * (longest_name + longest_type +
                       longest_wer + longest_size + 15) + "\n"

        for model in self.models:
            text += model['name'].ljust(longest_name) + " | "
            text += model['type'].ljust(longest_type) + " | "
            text += model['wer'].ljust(longest_wer) + " | "
            text += model['size'].ljust(longest_size) + "\n"

        return text

    def __repr__(self) -> str:
        return self.__str__()

    def __iter__(self):
        return iter(self.models)

    @overload
    def __getitem__(self, index: int) -> ModelDict:
        ...

    @overload
    def __getitem__(self, index: str) -> ModelDict:
        ...

    @overload
    def __getitem__(self, index: ModelType) -> List[ModelDict]:
        ...

    # type:ignore
    def __getitem__(self, index: Union[int, str, ModelType]) -> Union[ModelDict, List[ModelDict]]:
        if isinstance(index, int):
            return self.models[index]
        elif isinstance(index, str):
            index = index.lower()
            if index in ['tiny', 'base', 'small', 'medium', 'large']:
                return [model for model in self.models if model['type'] == index]
            found = False
            for model in self.models:
                if model['name'] == index:
                    found = True
                    return model

            if not found:
                raise ValueError("Model not found")

        elif isinstance(index, ModelType):
            return [model for model in self.models if model['type'] == index.value]
        else:
            raise TypeError("Index must be int or str")

    def __contains__(self, item: str):
        for model in self.models:
            if model['name'] == item:
                return True
        return False


def available_models(force=False) -> AvailableModels:
    """Returns a list of available models
        Args:
            force (bool, optional): Force download. Defaults to False.
    """
    
    models = get_models(force=force)
    return AvailableModels(models)


# def download(url, path):
#     obj = SmartDL(url, path)
#     obj.start(False)
#     return obj


# def download_model(model: ModelDict, force=False):
#     model_path = os.path.join(get_app_path(app_name), model['name'])
#     zip_path = os.path.join(model_path, "model.zip")
#     zip_path_exists = os.path.exists(zip_path)

#     if not os.path.exists(model_path) or force or not zip_path_exists:
#         if not os.path.exists(model_path):
#             os.mkdir(model_path)
#         logger.info("Downloading {}".format(model["name"]), model_path)
#         # obj = SmartDL(model["url"], model_path)
#         # obj.start(True)

#         obj = download(model["url"], zip_path)

#         try:
#             while not obj.isFinished():
#                 pass
#         except KeyboardInterrupt:
#             obj.stop()
#             raise KeyboardInterrupt

#     else:
#         logger.info("Getting model from {}".format(model_path))

#     return model_path


# def extract_model(model_path):
#     if os.path.exists(os.path.join(model_path, 'extracted.txt')):
#         return model_path
#     else:
#         logger.info("Extracting model from {}".format(model_path))
#         zip_ref = zipfile.ZipFile(os.path.join(model_path, "model.zip"), 'r')
#         zip_ref.extractall(model_path)
#         zip_ref.close()
#         with open(os.path.join(model_path, 'extracted.txt'), 'w') as f:
#             f.write("Extracted")
#         return model_path


# def get_model(model_name_or_type: Union[str, ModelType], force=False) -> ModelDict:
#     if isinstance(model_name_or_type, ModelType):
#         model_name_or_type = model_name_or_type.value

#     models = get_models()
#     mnt = model_name_or_type.lower()  # type:ignore

#     # if it is a type get the best wer score model (means lowest wer)
#     # otherwise get the model with the exact name

#     if mnt in [model_type.value for model_type in ModelType]:
#         # check if the model type is available
#         model = [model for model in models if model["type"] == mnt]
#         if model and not force:
#             if os.path.exists(os.path.join(get_app_path(app_name), model[0]["name"])):
#                 modeL = [model[0]]
#         model = sorted(model, key=lambda x: float(x["wer"]))  # type:ignore
#     else:
#         model = [model for model in models if model["name"] == mnt]

#     if not model:
#         logger.error("Model {} not found".format(model_name_or_type))
#         print(available_models())
#         raise ValueError("Model {} not found".format(model_name_or_type))

#     model = model[0]
#     # print(model)
#     model_path = download_model(model, force=force)
#     model_path = extract_model(model_path)

#     model["path"] = model_path
#     return model


def get_model(model_name_or_type: Union[str, ModelType], force=False) -> ModelDict:
    if isinstance(model_name_or_type, ModelType):
        model_name_or_type = model_name_or_type.value

    models = get_models()
    mnt = model_name_or_type.lower()  # type:ignore

    # if it is a type get the best wer score model (means lowest wer)
    # otherwise get the model with the exact name

    if mnt in [model_type.value for model_type in ModelType]:
        # check if the model type is available
        model = [model for model in models if model["type"] == mnt]
        if model and not force:
            if os.path.exists(os.path.join(get_app_path(app_name), model[0]["name"])):
                model = [model[0]]
        model = sorted(model, key=lambda x: float(x["wer"]))  # type:ignore
    else:
        model = [model for model in models if model["name"] == mnt]

    if not model:
        logger.error("Model {} not found".format(model_name_or_type))
        print(available_models())
        raise ValueError("Model {} not found".format(model_name_or_type))

    model = model[0]
    path = os.path.join(get_app_path(app_name), model["name"])
    model.clone(path)
    logger.info("Getting model from {}".format(path))
    return model
