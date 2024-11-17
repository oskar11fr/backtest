import yaml
import pandas as pd

from backtester import BacktestEngine
from typing import Dict, Tuple, List


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


def bundle_strategies(
        strats: Dict[str, BacktestEngine]
    ) -> Tuple[List[str], Dict[str, pd.Series]]:
    """
    Bundles strategies into a dictionary of capital returns and extracts strategy names.

    Parameters
    ----------
    strats : Dict[str, BacktestEngine]
        A dictionary where the key is the strategy name, and the value is a `BacktestEngine` object.

    Returns
    -------
    Tuple[List[str], Dict[str, pd.Series]]
        A tuple containing:
        - A list of strategy names.
        - A dictionary where keys are strategy names and values are normalized capital returns.

    Raises
    ------
    KeyError
        If the portfolio DataFrame does not contain the "capital" column.
    """
    try:
        capital_rets = {
            name: strat.portfolio_df["capital"].rename("close") / 1000 
            for name, strat in strats.items()
        }
        names = list(capital_rets.keys())
        return names, capital_rets
    except KeyError as e:
        raise KeyError("One or more strategies are missing the 'capital' column in their portfolio DataFrame.") from e
