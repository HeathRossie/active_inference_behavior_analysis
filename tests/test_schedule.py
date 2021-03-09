from typing import List, Tuple

import aibes.schedule as s
import numpy as np
import pytest


@pytest.mark.parametrize("ratio, n", [(5., 100), (10., 100), (100., 100),
                                      (5., 1000), (5., 10000)])
def test_vr_initialize(ratio: float, n: int):
    _min = 0
    vr = s.VariableRatio(ratio, n, _min)
    unexpected = s._geom_rng(ratio, n, _min)
    assert not np.array_equal(vr.vals, unexpected)
    vr.vals.sort()
    assert np.array_equal(vr.vals, unexpected)


@pytest.mark.parametrize("interval, n", [(5., 100), (10., 100), (100., 100),
                                         (5., 1000), (5., 10000)])
def test_vi_initialize(interval: float, n: int):
    _min = 0
    vi = s.VariableInterval(interval, n, _min)
    unexpected = s._exp_rng(interval, n, _min)
    assert not np.array_equal(vi.vals, unexpected)
    vi.vals.sort()
    assert np.array_equal(vi.vals, unexpected)


@pytest.mark.parametrize("interval, timestep", [
    (1., .01),
    (1., .1),
    (1., .5),
    (1., 1.),
    (10., .01),
    (10., .1),
    (10., .5),
    (10., 1.),
    (100., .01),
    (100., .1),
    (100., .5),
    (100., 1.),
])
def test_vi_step(interval, timestep: float):
    vi = s.VariableInterval(interval, 10, 1.)
    intervals = vi.vals
    expected_rewards: List[int] = []
    observed_rewards: List[int] = []
    for interval in intervals:
        while True:
            action = int(np.random.uniform() < 0.9)
            interval -= timestep
            observed = vi.step(timestep, action)
            observed_rewards.append(observed)
            if action and interval <= 0.:
                expected_rewards.append(1)
                break
            else:
                expected_rewards.append(0)
    assert expected_rewards == observed_rewards
    assert vi.finished()
    assert vi.step(timestep, 1) == 0


@pytest.mark.parametrize("req", [1., 10., 100.])
def test_vr_step(req):
    vr = s.VariableRatio(req, 10, 0.)
    required_responses = vr.vals
    expected_rewards: List[int] = []
    observed_rewards: List[int] = []
    for req in required_responses:
        while True:
            action = int(np.random.uniform() < 0.9)
            req -= action
            observed = vr.step(action, action)
            observed_rewards.append(observed)
            if action and req <= 0.:
                expected_rewards.append(1)
                break
            else:
                expected_rewards.append(0)
    assert expected_rewards == observed_rewards
    assert vr.finished()
    assert vr.step(1, 1) == 0


@pytest.mark.parametrize("interval, timestep, num_reward", [
    (10., .1, 20),
    (10., .1, 30),
    (10., .1, 21),
    (10., .1, 29),
])
def test_vi_forever(interval, timestep: float, num_reward: int):
    vi = s.VariableInterval(interval, 10, 1.)
    vi.forever()
    expected_rewards: List[int] = []
    observed_rewards: List[int] = []
    cumulative_rewards = 0
    while cumulative_rewards < num_reward:
        intervals = vi.vals
        for interval in intervals:
            while True:
                action = int(np.random.uniform() < 0.9)
                interval -= timestep
                observed = vi.step(timestep, action)
                observed_rewards.append(observed)
                if action and interval <= 0.:
                    expected_rewards.append(1)
                    cumulative_rewards += 1
                    break
                else:
                    expected_rewards.append(0)
            if cumulative_rewards >= num_reward:
                break
    assert expected_rewards == observed_rewards
    assert cumulative_rewards == num_reward


@pytest.mark.parametrize("req, num_reward", [(10., 20), (10., 30), (10., 21),
                                             (10., 29)])
def test_vr_forever(req, num_reward):
    vr = s.VariableRatio(req, 10, 0.)
    vr.forever()
    expected_rewards: List[int] = []
    observed_rewards: List[int] = []
    cumulative_rewards = 0
    while cumulative_rewards < num_reward:
        required_responses = vr.vals
        for req in required_responses:
            while True:
                action = int(np.random.uniform() < 0.9)
                req -= action
                observed = vr.step(action, action)
                observed_rewards.append(observed)
                if action and req <= 0.:
                    expected_rewards.append(1)
                    cumulative_rewards += 1
                    break
                else:
                    expected_rewards.append(0)
            if cumulative_rewards >= num_reward:
                break
    assert expected_rewards == observed_rewards
    assert cumulative_rewards == num_reward


@pytest.mark.parametrize("n", [1, 10, 100])
def test_conc_step(n):
    vi = s.VariableInterval(10., n, 1.)
    vr = s.VariableRatio(10, n, 0.)
    conc = s.ConcurrentSchedule([vi, vr])

    cumulative_rewards_vi = 0
    cumulative_rewards_vr = 0

    intervals = vi.vals
    required_responses = vr.vals
    interval = intervals[cumulative_rewards_vi]
    req = required_responses[cumulative_rewards_vr]

    expected_rewards: List[Tuple[int, int]] = []
    observed_rewards: List[Tuple[int, int]] = []

    while not conc.finished():
        while True:
            if np.random.uniform() <= 0.5:
                actions = np.array([1, 0])
            else:
                actions = np.array([0, 1])

            counts = actions.copy()
            counts[0] = 1.

            interval -= 1.
            req -= actions[1]
            obs = conc.step(counts.tolist(), actions.tolist())

            if actions[0] == 1 and interval <= 0.:
                exp0 = 1
                cumulative_rewards_vi += 1
                if not cumulative_rewards_vi == n:
                    interval = intervals[cumulative_rewards_vi]
            else:
                exp0 = 0

            if actions[1] == 1 and req <= 0.:
                exp1 = 1
                cumulative_rewards_vr += 1
                if not n == cumulative_rewards_vr:
                    req = required_responses[cumulative_rewards_vr]
            else:
                exp1 = 0

            expected_rewards.append((exp0, exp1))
            observed_rewards.append((obs[0], obs[1]))
            if exp0 == 1 or exp1 == 1:
                break

    assert expected_rewards == observed_rewards
