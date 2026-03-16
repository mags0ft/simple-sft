"""
Loads in the YAML configuration file.
"""

from functools import cache

import yaml
from os import getenv

from dotenv import load_dotenv

from constants import YAML_CONFIG_ENV

load_dotenv()


@cache
def load_config() -> dict:
    """
    Loads the configuration from the YAML file.
    """

    with open(getenv(YAML_CONFIG_ENV), "r") as f:
        return yaml.load(f, yaml.FullLoader)


config: dict = load_config()
