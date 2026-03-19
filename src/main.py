"""
The main file to control the program execution flow.
"""

from config_reader import config


def main() -> None:
    """
    Controls the main execution flow.
    """

    print(config)


if __name__ == "__main__":
    main()
