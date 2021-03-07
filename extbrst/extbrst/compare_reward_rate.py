from typing import Any, Iterable, List, Tuple

import numpy as np
from nptyping import NDArray

from extbrst.model import (Action, Agent, GAIAgent, Prediction, Probability,
                           Reward)
from extbrst.schedule import (Extinction, Schedule, VariableInterval,
                              VariableRatio)

NumberOfTrial = int
Result = Tuple[Action, Reward, Prediction, Probability]
OutputData = Tuple[Probability, Action, Reward, Prediction, Probability]


class ConcurrentSchedule(object):
    def __init__(self, schedules: List[Schedule]):
        self.__schedules = schedules

    def step(self, counts: Iterable[Any],
             actions: Iterable[Action]) -> NDArray[1, Reward]:
        rewards: List[int] = []
        for s, c, a in zip(self.__schedules, counts, actions):
            rewards.append(s.step(c, a))
        return np.array(rewards)

    def finished(self) -> bool:
        return self.__schedules[0].finished()


def trial_process(agent: Agent, schedule: ConcurrentSchedule,
                  timestep: float) -> Result:
    preds = agent.predict()
    probs = agent.calculate_response_probs(-preds)
    actions = agent.choose_action(probs)
    counts = actions.copy().tolist()
    counts[1] = timestep
    rewards = schedule.step(counts, actions)
    agent.update(rewards, actions.astype(np.uint8))
    ret = actions[0], rewards[0], preds[0], probs[0]
    return ret


def run(agent: Agent, schedule: ConcurrentSchedule,
        timestep: float) -> List[Result]:
    results: List[Result] = []
    while not conc.finished():
        results.append(trial_process(agent, conc, timestep))
    return results


if __name__ == '__main__':
    from pandas import DataFrame

    from extbrst.model import GAIAgent
    from extbrst.util import get_nth_ancestor

    requirements: List[Probability] = [1.]
    extinction: Probability = 0.
    baseline_lenght: NumberOfTrial = 200
    extinction_lenght: NumberOfTrial = 1000

    results: List[List[OutputData]] = []
    # run simulations for each baseline reward probability
    for rq in requirements:
        agent = GAIAgent(0.5, 2)
        vr = VariableRatio()
        vr.config(rq, baseline_lenght, 0)
        vi = VariableInterval()
        vi.config(30., 1000, 1.)
        conc = ConcurrentSchedule([vr, vi])
        _baseline_result = run(agent, conc, 1.)
        baseline_result: List[OutputData] = \
            list(map(lambda br: (rq, ) + br, _baseline_result))
        ext = Extinction()
        ext.config(1., extinction_lenght, 0)
        vi.config(30., 1000, 1.)
        conc = ConcurrentSchedule([ext, vi])
        _ext_result = run(agent, conc, 1.)
        ext_result: List[OutputData] = \
            list(map(lambda er: (extinction, ) + er, _ext_result))
        results.append(baseline_result + ext_result)

    data_dir = get_nth_ancestor(__file__, 1).joinpath("data")
    if not data_dir.exists():
        data_dir.mkdir()
    filename = data_dir.joinpath("compare_reward_rate.csv")

    # `sum` can flatten list of list (f: List[List[T]] => List[T])
    merged_result = sum(results, [])
    df = DataFrame(
        merged_result,
        columns=["reward_prob", "action", "reward", "G", "action_prob"])
    df.to_csv(filename, index=False)
