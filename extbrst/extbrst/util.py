from pathlib import Path


def get_nth_ancestor(relpath: str, n: int) -> Path:
    return list(Path(relpath).absolute().parents)[n]
