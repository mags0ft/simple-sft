"""
Loads in the YAML configuration file.
"""

from functools import cache

import yaml
from os import getenv

from dotenv import load_dotenv

from constants import YAML_CONFIG_ENV
from logging_manager import logger

load_dotenv()


@cache
def load_config() -> dict:
    """
    Loads the configuration from the YAML file.
    """
    path = getenv(YAML_CONFIG_ENV, "./config.yml")
    logger.debug("Loading config from %s", path)

    with open(path, "r") as f:
        cfg = yaml.load(f, yaml.FullLoader)

    logger.info(
        "Loaded configuration (%d top-level keys)",
        len(cfg) if isinstance(cfg, dict) else 0,
    )

    return cfg


config: dict = load_config()
