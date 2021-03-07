from abc import ABCMeta, abstractmethod, abstractproperty
from typing import Any, Iterable, List, Sequence, Union

import numpy as np
import scipy.stats as st
from nptyping import NDArray

from extbrst.util import randomize


def _exp_rng(mean: float, n: int, _min: float) -> NDArray[1, float]:
    return st.expon.ppf(np.linspace(0.01, 0.99, n), scale=mean, loc=_min)


def _geom_rng(mean: float, n: int, _min: float) -> NDArray[1, float]:
    return st.geom.ppf(np.linspace(0.01, 0.99, n),
                       p=1 / (mean - _min),
                       loc=_min)


class Schedule(metaclass=ABCMeta):
    @abstractmethod
    def step(self, count: Union[Any, Iterable[Any]],
             action: Union[int, Iterable[int]]) -> Union[int, Iterable[int]]:
        pass

    @abstractmethod
    def config(self, val: float, n: int, _min: float):
        pass

    @abstractmethod
    def finished(self) -> bool:
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractproperty
    def val(self) -> Any:
        pass

    @abstractproperty
    def n(self) -> Any:
        pass

    @abstractproperty
    def vals(self) -> Any:
        pass


class VariableInterval(Schedule):
    def __init__(self, val: float, n: int, _min: float):
        self.__val = val
        self.__min = _min
        self.__n = n
        self.__count = 0
        self.__intervals = randomize(_exp_rng(val, n, _min))
        self.__interval = self.__intervals[self.__count]

    def config(self, val: float, n: int, _min: float):
        self.__val = val
        self.__min = _min
        self.__n = n
        self.__count = 0
        self.__intervals = randomize(_exp_rng(val, n, _min))
        self.__interval = self.__intervals[self.__count]

    def step(self, count: float, action: int) -> int:
        self.__interval -= count
        if self.__interval <= 0. and action == 1:
            self.__count += 1
            if not self.finished():
                self.__interval = self.__intervals[self.__count]
            return 1
        return 0

    def finished(self) -> bool:
        return self.__count >= self.__n

    def reset(self):
        self.__count = 0
        self.__intervals = randomize(self.__intervals)
        self.__interval = self.__intervals[self.__count]

    @property
    def val(self) -> float:
        return self.__val

    @property
    def n(self) -> int:
        return self.__n

    @property
    def vals(self) -> NDArray[1, float]:
        return self.__intervals


class VariableRatio(Schedule):
    def __init__(self, val: float, n: int, _min: float):
        self.__val = val
        self.__min = _min
        self.__n = n
        self.__count = 0
        self.__responses = randomize(_geom_rng(val, n, _min))
        self.__response = self.__responses[self.__count]

    def config(self, val: float, n: int, _min: float):
        self.__val = val
        self.__min = _min
        self.__n = n
        self.__count = 0
        self.__responses = randomize(_geom_rng(val, n, _min))
        self.__response = self.__responses[self.__count]

    def step(self, count: int, action: int) -> int:
        self.__response -= count
        if self.__response <= 0 and action == 1:
            self.__count += 1
            if not self.finished():
                self.__response = self.__responses[self.__count]
            return 1
        return 0

    def finished(self) -> bool:
        return self.__count >= self.__n

    def reset(self):
        self.__count = 0
        self.__responses = randomize(self.__responses)
        self.__response = self.__responses[self.__count]

    @property
    def val(self) -> float:
        return self.__val

    @property
    def n(self) -> int:
        return self.__n

    @property
    def vals(self) -> NDArray[1, float]:
        return self.__responses


class Extinction(Schedule):
    def __init__(self, val: float):
        self.__val = val
        self.__count = 0

    def config(self, val: float, n: int = 0, _min: float = 0):
        _, _ = n, _min
        self.__val = val

    def step(self, count: float, action: int) -> int:
        _ = action
        self.__count += count
        return 0

    def finished(self) -> bool:
        return self.__count >= self.__val

    def reset(self):
        self.__count = 0

    @property
    def val(self) -> float:
        return self.__val

    @property
    def n(self) -> None:
        pass

    @property
    def vals(self) -> None:
        pass


class ConcurrentSchedule(Schedule):
    def __init__(self, schedules: List[Schedule]):
        self.__schedules = schedules

    def config(self, val: List[Any], n: List[int], _min: List[float]):
        for i in range(len(self.__schedules)):
            self.__schedules[i].config(val[i], n[i], _min[i])

    def step(self, count: Sequence, action: Sequence) -> NDArray[1, int]:
        rewards: NDArray[1, int] = np.zeros(len(self.__schedules))
        for i in range(len(self.__schedules)):
            rew = self.__schedules[i].step(count[i], action[i])
            rewards[i] = rew
        return rewards

    def finished(self) -> bool:
        return sum(s.finished() for s in self.__schedules) > 0

    def reset(self):
        for i in range(len(self.__schedules)):
            self.__schedules[i].reset()

    @property
    def val(self) -> List[Any]:
        return [s.val for s in self.__schedules]

    @property
    def n(self) -> List[int]:
        return [s.n for s in self.__schedules]

    @property
    def vals(self) -> List[Any]:
        return [s.vals for s in self.__schedules]
