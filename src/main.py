"""
The main file to control the program execution flow.
"""

from config_reader import config
from scheduler import main_flow
from logging_manager import logger


def main() -> None:
    """
    Controls the main execution flow.
    """

    logger.debug("Starting main flow")
    main_flow()
    logger.debug("Main flow finished")


if __name__ == "__main__":
    main()
