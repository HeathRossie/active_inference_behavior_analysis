from itertools import product
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import yaml
from nptyping import NDArray

from aibes.model import Agent


def get_nth_ancestor(relpath: str, n: int) -> Path:
    return list(Path(relpath).absolute().parents)[n]


def randomize(v: NDArray[1, Any]) -> NDArray[1, Any]:
    return np.random.choice(v, size=len(v), replace=False)


def colnames(agent: Agent, condition: str) -> List[str]:
    k = agent.k
    colnames = [condition, "reward", "action"]
    colnames += [f"pragmatic_{i}" for i in range(k)]
    colnames += [f"epistemic_{i}" for i in range(k)]
    colnames += [f"response_probability_{i}" for i in range(k)]
    return colnames


def load_yaml(path: str) -> dict:
    f = open(path, "r")
    d = yaml.safe_load(f)
    return d


def all_parameters_combination(yaml: dict) -> List[Tuple]:
    return list(product(*yaml.values()))
