"""
Handles logging functionality.
"""

import logging
from os import getenv

from dotenv import load_dotenv


load_dotenv()


__version__ = "0.1.0"


def create_logger() -> logging.Logger:
    """
    Factory for a logger that is used across the project.
    """

    logger = logging.getLogger("simple-sft")
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch2 = logging.FileHandler(getenv("SIMPLESFT_LOG_FILE", "simple-sft.log"))
    ch.setLevel(logging.DEBUG)
    ch2.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # add formatter to ch
    ch.setFormatter(formatter)
    ch2.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)
    logger.addHandler(ch2)

    return logger


logger = create_logger()

logger.info(f"simple-sft {__version__}")
