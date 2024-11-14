import yaml
from backtester import BacktestEngine


def get_configs() -> dict[str,str]:
    with open('./backtester/engine/configs.yml', 'r') as file:
        confs = yaml.safe_load(file)
        return confs
    

def bundle_strategies(strats: dict[str, BacktestEngine]):
    capital_rets = {
        name: strat.portfolio_df["capital"].rename("close") / 100 for name, strat in strats.items()
    }
    names = list(capital_rets.keys())
    return names, capital_rets