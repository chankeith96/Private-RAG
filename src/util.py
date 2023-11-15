import yaml


def get_config():
    # Load configuration file
    with open("../conf/config.yaml", "r") as file:
        config = yaml.safe_load(file)
    return config
