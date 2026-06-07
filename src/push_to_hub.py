from argparse import ArgumentParser

from datasets import load_dataset


def push_to_hub(
    dataset_name: str,
    dataset_path: str,
    private: bool = False,
    token: str | None = None,
) -> None:
    """
    Pushes the generated conversations to the Hugging Face Hub as a dataset.
    """

    dataset = load_dataset("json", data_files=dataset_path, split="train")

    dataset.push_to_hub(
        dataset_name,
        private=private,
        token=token,
    )


def main() -> None:
    parser = ArgumentParser(
        description="Push generated conversations to Hugging Face Hub"
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name of the dataset on Hugging Face Hub",
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the generated dataset file (JSONL format)",
    )
    parser.add_argument(
        "--private", action="store_true", help="Whether to make the dataset private"
    )

    args = parser.parse_args()

    push_to_hub(
        dataset_name=args.name,
        dataset_path=args.path,
        private=args.private,
    )


if __name__ == "__main__":
    main()
