from abc import ABCMeta, abstractmethod
from typing import Any

import numpy as np
import scipy.stats as st

from extbrst.util import randomize


class Schedule(metaclass=ABCMeta):
    @abstractmethod
    def step(self, count: Any, action: int) -> Any:
        pass

    @abstractmethod
    def config(self, val: float, n: int, _min: float = 0.) -> int:
        pass

    @abstractmethod
    def finished(self) -> bool:
        pass


class VariableInterval(Schedule):
    def __init__(self):
        self.__mean_intervals = None
        self.__count = 0

    def config(self, val: float, n: int, _min: float):
        self.__val = val
        self.__min = _min
        self.__n = n
        intervals = st.expon.ppf(np.linspace(0.01, 0.99, n),
                                 scale=val,
                                 loc=_min)
        self.__intervals = randomize(intervals)
        self.__interval = self.__intervals[self.__count]

    def step(self, count: float, action: int) -> int:
        self.__interval -= count
        if self.__interval <= 0. and action == 1:
            self.__count += 1
            if not self.finished():
                self.__interval = self.__intervals[self.__count]
                print(self.__interval)
            return 1
        return 0

    def finished(self) -> bool:
        return self.__count >= self.__n


class FixedRatio(Schedule):
    def __init__(self):
        self.__required_responses = None
        self.__count = 0

    def config(self, val: float, n: int, _min: int):
        self.__val = val
        self.__min = _min
        self.__n = n
        self.__required_responses = np.array([val for _ in range(n)])
        self.__required_response = self.__required_responses[self.__count]

    def step(self, count: int, action: int) -> int:
        self.__required_response -= count
        if action == 1:
            self.__count += 1
            if not self.finished:
                self.__required_response = self.__required_responses[
                    self.__count]
            return 1
        return 0

    def finished(self) -> bool:
        return self.__count >= self.__n


class VariableRatio(Schedule):
    def __init__(self):
        self.__required_responses = None
        self.__count = 0

    def config(self, val: float, n: int, _min: int):
        self.__val = val
        self.__min = _min
        self.__n = n
        required_responses = st.geom.ppf(np.linspace(0.01, 0.99, n),
                                         p=1 / (val - _min),
                                         loc=_min)
        self.__required_responses = randomize(required_responses)
        self.__required_response = self.__required_responses[self.__count]

    def step(self, count: int, action: int) -> int:
        self.__required_response -= count
        if action == 1 and self.__required_response <= 0:
            self.__count += 1
            if not self.finished():
                self.__required_response = self.__required_responses[
                    self.__count]
            return 1
        return 0

    def finished(self) -> bool:
        return self.__count >= self.__n


class Extinction(Schedule):
    def __init__(self):
        self.__count = 0

    def config(self, val: float, n: int, _min: int):
        self.__n = n
        pass

    def step(self, count: int, action: int) -> int:
        self.__count += 1
        return 0

    def finished(self) -> bool:
        return self.__count >= self.__n
