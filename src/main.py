"""
The main file to control the program execution flow.
"""

from config_reader import load_config


def main() -> None:
    """
    Controls the main execution flow.
    """

    config = load_config()

    print(config)


if __name__ == "__main__":
    main()
