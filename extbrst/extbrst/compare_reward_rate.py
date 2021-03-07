from typing import List, Tuple

import numpy as np

from extbrst.model import Action, Agent, Prediction, Probability, Reward
from extbrst.schedule import (ConcurrentSchedule, Extinction, VariableInterval,
                              VariableRatio)

NumberOfTrial = int
Result = Tuple[Action, Reward, Prediction, Probability]
OutputData = Tuple[Probability, Action, Reward, Prediction, Probability]


def run_baseline(agent: Agent, schedule: ConcurrentSchedule, timestep: float):
    def trial_process(agent: Agent, schedule: ConcurrentSchedule,
                      timestep: float) -> Result:
        preds = agent.predict()
        probs = agent.calculate_response_probs(-preds)
        actions = agent.choose_action(probs)
        counts = actions.copy().tolist()
        counts[1:] = [timestep for _ in range(len(counts) - 1)]
        rewards = schedule.step(counts, actions.tolist())
        agent.update(rewards, actions.astype(np.uint8))
        ret = actions[0], rewards[0], preds[0], probs[0]
        return ret

    results: List[Result] = []
    while not schedule.finished():
        results.append(trial_process(agent, schedule, timestep))
    return results


def run_extinction(agent: Agent, schedule: ConcurrentSchedule,
                   timestep: float):
    counts = [timestep for _ in range(len(schedule.val))]

    def trial_process(agent: Agent, schedule: ConcurrentSchedule,
                      timestep: float) -> Result:
        preds = agent.predict()
        probs = agent.calculate_response_probs(-preds)
        actions = agent.choose_action(probs)
        rewards = schedule.step(counts, actions.tolist())
        agent.update(rewards, actions.astype(np.uint8))
        ret = actions[0], rewards[0], preds[0], probs[0]
        return ret

    results: List[Result] = []
    while not schedule.finished():
        results.append(trial_process(agent, schedule, timestep))
    return results


if __name__ == '__main__':
    from pandas import DataFrame

    from extbrst.model import GAIAgent
    from extbrst.util import get_nth_ancestor

    requirements: List[Probability] = [1 + 1e-8]
    extinction: Probability = 0.
    baseline_lenght: NumberOfTrial = 500
    extinction_lenght: NumberOfTrial = 3600
    num_alt = 10

    results: List[List[OutputData]] = []
    # run simulations for each VR schedule
    for rq in requirements:
        agent = GAIAgent(1., 1 + num_alt)  # operant + alternative behaviors
        baseline_schedule = VariableRatio(rq, baseline_lenght, 0)
        alternative_schedules = [
            VariableInterval(120., 1000, 1.) for _ in range(num_alt)
        ]
        [s.forever() for s in alternative_schedules]

        baseline_schedules = ConcurrentSchedule([baseline_schedule] +
                                                alternative_schedules)
        _baseline_result = run_baseline(agent, baseline_schedules, 1.)

        ext = Extinction(extinction_lenght)
        [alt.reset for alt in alternative_schedules]
        extinction_schedules = ConcurrentSchedule([ext] +
                                                  alternative_schedules)
        _ext_result = run_extinction(agent, extinction_schedules, 1.)
        baseline_result: List[OutputData] = \
            list(map(lambda br: (1 / rq, ) + br, _baseline_result))
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
