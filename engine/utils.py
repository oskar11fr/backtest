import yaml


def get_configs() -> dict[str,str]:
    with open('engine/configs.yml', 'r') as file:
        confs = yaml.safe_load(file)
        return confs