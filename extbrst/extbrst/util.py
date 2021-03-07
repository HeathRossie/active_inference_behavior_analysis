from pathlib import Path
from typing import Any

import numpy as np
from nptyping import NDArray


def get_nth_ancestor(relpath: str, n: int) -> Path:
    return list(Path(relpath).absolute().parents)[n]


def randomize(v: NDArray[1, Any]) -> NDArray[1, Any]:
    return np.random.choice(v, size=len(v), replace=False)
