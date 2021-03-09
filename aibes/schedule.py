from abc import ABCMeta, abstractmethod, abstractproperty
from typing import Any, Iterable, List, Sequence, Union

import numpy as np
import scipy.stats as st
from nptyping import NDArray

from aibes.types import (Action, Duration, Interval, NumberOfReward,
                         RequiredResponse, Reward)
from aibes.util import randomize


def _exp_rng(mean: Interval, n: NumberOfReward,
             _min: Interval) -> NDArray[1, Interval]:
    return st.expon.ppf(np.linspace(0.01, 0.99, n), scale=mean, loc=_min)


def _geom_rng(mean: RequiredResponse, n: NumberOfReward,
              _min: RequiredResponse) -> NDArray[1, RequiredResponse]:
    return st.geom.ppf(np.linspace(0.01, 0.99, n),
                       p=1 / (mean - _min),
                       loc=_min)


class Schedule(metaclass=ABCMeta):
    @abstractmethod
    def step(self, count: Union[Any, Iterable[Any]],
             action: Union[int, Iterable[int]]) -> Union[int, Iterable[int]]:
        pass

    @abstractmethod
    def config(self, val: float, n: NumberOfReward, _min: float):
        pass

    @abstractmethod
    def finished(self) -> bool:
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def forever(self):
        self.__repeat = True

    @abstractmethod
    def once(self):
        self.__repeat = False

    @abstractproperty
    def val(self) -> Any:
        pass

    @abstractproperty
    def n(self) -> Any:
        pass

    @abstractproperty
    def vals(self) -> Any:
        pass

    @abstractproperty
    def repeat(self) -> bool:
        return self.__repeat


class VariableInterval(Schedule):
    def __init__(self, val: Interval, n: NumberOfReward, _min: Interval):
        self.__val = val
        self.__min = _min
        self.__n = n
        self.__count = 0
        self.__intervals = randomize(_exp_rng(val, n, _min))
        self.__interval = self.__intervals[self.__count]
        self.__repeat = False

    def config(self, val: Interval, n: NumberOfReward, _min: float):
        self.__val = val
        self.__min = _min
        self.__n = n
        self.__count = 0
        self.__intervals = randomize(_exp_rng(val, n, _min))
        self.__interval = self.__intervals[self.__count]

    def step(self, count: Interval, action: Action) -> int:
        self.__interval -= count
        if self.__interval <= 0. and action == 1:
            self.__count += 1
            if not self.finished():
                self.__interval = self.__intervals[self.__count]
            elif self.__repeat:
                self.reset()
            return 1
        return 0

    def finished(self) -> bool:
        return self.__count >= self.__n

    def reset(self):
        self.__count = 0
        self.__intervals = randomize(self.__intervals)
        self.__interval = self.__intervals[self.__count]

    def forever(self):
        self.__repeat = True

    def once(self):
        self.__repeat = False

    @property
    def val(self) -> Interval:
        return self.__val

    @property
    def n(self) -> NumberOfReward:
        return self.__n

    @property
    def vals(self) -> NDArray[1, Interval]:
        return self.__intervals

    @property
    def repeat(self) -> bool:
        return self.__repeat


class VariableRatio(Schedule):
    def __init__(self, val: RequiredResponse, n: NumberOfReward,
                 _min: RequiredResponse):
        self.__val = val
        self.__min = _min
        self.__n = n
        self.__count = 0
        self.__responses = randomize(_geom_rng(val, n, _min))
        self.__response = self.__responses[self.__count]
        self.__repeat = False

    def config(self, val: RequiredResponse, n: NumberOfReward, _min: float):
        self.__val = val
        self.__min = _min
        self.__n = n
        self.__count = 0
        self.__responses = randomize(_geom_rng(val, n, _min))
        self.__response = self.__responses[self.__count]

    def step(self, count: Action, action: Action) -> int:
        self.__response -= count
        if self.__response <= 0 and action == 1:
            self.__count += 1
            if not self.finished():
                self.__response = self.__responses[self.__count]
            elif self.__repeat:
                self.reset()
            return 1
        return 0

    def finished(self) -> bool:
        return self.__count >= self.__n

    def reset(self):
        self.__count = 0
        self.__responses = randomize(self.__responses)
        self.__response = self.__responses[self.__count]

    def forever(self):
        self.__repeat = True

    def once(self):
        self.__repeat = False

    @property
    def val(self) -> RequiredResponse:
        return self.__val

    @property
    def n(self) -> NumberOfReward:
        return self.__n

    @property
    def vals(self) -> NDArray[1, RequiredResponse]:
        return self.__responses

    @property
    def repeat(self) -> bool:
        return self.__repeat


class Extinction(Schedule):
    def __init__(self, val: float):
        self.__val = val
        self.__count = 0
        self.__repeat = False

    def config(self, val: Duration, n: NumberOfReward = 0, _min: Duration = 0):
        _, _ = n, _min
        self.__val = val

    def step(self, count: Interval, action: Action) -> int:
        _ = action
        self.__count += count
        return 0

    def finished(self) -> bool:
        if self.__repeat:
            return False
        return self.__count >= self.__val

    def reset(self):
        self.__count = 0

    def forever(self):
        self.__repeat = True

    def once(self):
        self.__repeat = False

    @property
    def val(self) -> Duration:
        return self.__val

    @property
    def n(self) -> None:
        pass

    @property
    def vals(self) -> None:
        pass

    @property
    def repeat(self) -> bool:
        return self.__repeat


class ConcurrentSchedule(Schedule):
    def __init__(self, schedules: List[Schedule]):
        self.__schedules = schedules
        self.__repeat = False

    def config(self, val: List[Any], n: List[int], _min: List[float]):
        for i in range(len(self.__schedules)):
            self.__schedules[i].config(val[i], n[i], _min[i])

    def step(self, count: Sequence, action: Sequence) -> NDArray[1, int]:
        rewards: NDArray[1, Reward] = np.zeros(len(self.__schedules))
        for i in range(len(self.__schedules)):
            rew = self.__schedules[i].step(count[i], action[i])
            rewards[i] = rew
        return rewards

    def finished(self) -> bool:
        return sum(s.finished() for s in self.__schedules) > 0

    def reset(self):
        for i in range(len(self.__schedules)):
            self.__schedules[i].reset()

    def forever(self):
        self.__repeat = True
        for i in range(len(self.__schedules)):
            self.__schedules[i].forever()

    def once(self):
        self.__repeat = False
        for i in range(len(self.__schedules)):
            self.__schedules[i].once()

    @property
    def val(self) -> List[Any]:
        return [s.val for s in self.__schedules]

    @property
    def n(self) -> List[int]:
        return [s.n for s in self.__schedules]

    @property
    def vals(self) -> List[Any]:
        return [s.vals for s in self.__schedules]

    @property
    def repeat(self) -> bool:
        return self.__repeat
