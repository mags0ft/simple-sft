"""
Handles logging functionality.
"""

import logging


def create_logger() -> logging.Logger:
    """
    Factory for a logger that is used across the project.
    """

    logger = logging.getLogger("simple-sft")
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)

    return logger


logger = create_logger()
