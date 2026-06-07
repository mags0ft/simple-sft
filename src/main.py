"""
The main file to control the program execution flow.
"""

from argparse import ArgumentParser

from scheduler import main_flow
from logging_manager import logger


def main() -> None:
    """
    Controls the main execution flow.
    """

    parser = ArgumentParser(
        description="simple-sft: Generate high-quality synthetic datasets for fine-tuning."
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="The name of the run, unique identifier.",
    )
    parser.add_argument(
        "--new-run",
        action="store_true",
        default=False,
        help="Flag to indicate a new run.",
    )
    parser.add_argument(
        "--resume-run",
        action="store_true",
        default=False,
        help="Flag to indicate resuming an existing run.",
    )
    parser.add_argument(
        "--generate-templates-only",
        action="store_true",
        default=False,
        help="Flag to indicate generating templates only. Will not actually generate the dataset, just the templates.",
    )

    args = parser.parse_args()

    if args.new_run and args.resume_run:
        logger.error("Cannot specify both --new-run and --resume-run.")
        return

    if not args.new_run and not args.resume_run and not args.generate_templates_only:
        logger.error("Did you mean to start a new run? Use --new-run.\nNothing to do. Exiting.")
        return

    if args.resume_run and not args.run_name:
        logger.error("Must specify --run-name when using --resume-run.")
        return

    main_flow(
        run_name=args.run_name,
        new_run=args.new_run,
        resume_run=args.resume_run,
        generate_templates_only=args.generate_templates_only,
    )

    logger.info("Finished.")


if __name__ == "__main__":
    main()
