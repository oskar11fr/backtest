import yaml
import pandas as pd

from typing import Dict, Tuple, List

import shelve

def save_obj(obj, obj_name: str) -> None:
    DATA_FOLDER_PATH: str = get_configs()["PATH"]["DATA_FOLDER_PATH"]
    with shelve.open(DATA_FOLDER_PATH + "/strategy") as db:
        db[obj_name] = obj
    return

def load_obj(obj_name: str) -> pd.DataFrame:
    DATA_FOLDER_PATH: str = get_configs()["PATH"]["DATA_FOLDER_PATH"]
    with shelve.open(DATA_FOLDER_PATH + "/strategy") as db:
        if obj_name in db.keys():
            portfolio_df = db[obj_name]
            return portfolio_df
        else:
            return None

def get_configs() -> Dict[str, str]:
    """
    Loads configuration settings from a YAML file.

    Returns
    -------
    Dict[str, str]
        A dictionary containing configuration settings.

    Raises
    ------
    FileNotFoundError
        If the configuration file is not found.
    yaml.YAMLError
        If the YAML file contains errors or is improperly formatted.
    """
    try:
        with open('./backtester/engine/configs.yml', 'r') as file:
            confs = yaml.safe_load(file)
            return confs
    except FileNotFoundError as e:
        raise FileNotFoundError("The configuration file './backtester/engine/configs.yml' was not found.") from e
    except yaml.YAMLError as e:
        raise ValueError("Error parsing the YAML configuration file.") from e
    