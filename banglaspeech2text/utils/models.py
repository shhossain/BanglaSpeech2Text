import json
import os

current_dir = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(current_dir, "listed_models.json"), "r") as f:
    all_models = json.load(f)


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


__all__ = ["nice_model_list", "get_best_model", "get_model"]