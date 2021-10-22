from genericpath import isfile
import pathlib


def find_checkpoint_file(checkpoint_path):
    checkpoint_path = pathlib.Path(checkpoint_path)

    if checkpoint_path.is_file():
        return checkpoint_path
    else:
        checkpoint_files = sorted(list(checkpoint_path.rglob("*.ckpt")))
        return checkpoint_files[-1]
