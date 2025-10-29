import sys

from src import predict as predict_module
from src import train as train_module


def main() -> None:
    if len(sys.argv) <= 1:
        train_module.main([])
        return

    command, *rest = sys.argv[1:]

    if command == "train":
        train_module.main(rest)
    elif command == "predict":
        predict_module.main(rest)
    else:
        raise SystemExit(f"Unsupported command '{command}'. Use 'train' or 'predict'.")


if __name__ == "__main__":
    main()

