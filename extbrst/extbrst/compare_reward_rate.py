from typing import List, Tuple

import numpy as np

from extbrst.model import Action, Agent, Prediction, Probability, Reward

NumberOfTrial = int
Result = Tuple[Action, Reward, Prediction, Probability]
OutputData = Tuple[Probability, Action, Reward, Prediction, Probability]


def reward_function(prob: Probability) -> Reward:
    return int(np.random.uniform() < prob)


def trial_process(agent: Agent, reward_prob: Probability) -> Result:
    pred = agent.predict()
    prob = agent.calculate_response_prob(-pred)
    action = agent.emit_action(prob)
    reward = reward_function(reward_prob)
    agent.update(reward, action)
    return action, reward, pred, prob


def run(agent: Agent, trial: int, reward_prob: Probability) -> List[Result]:
    return list(map(lambda _: trial_process(agent, reward_prob), range(trial)))


if __name__ == '__main__':
    from pandas import DataFrame

    from extbrst.model import GAIAgent
    from extbrst.util import get_nth_ancestor

    baseline_reward_probs: List[Probability] = [
        1., 0.75, 0.5, 0.25, 0.1, 0.05, 0.01
    ]
    extinction: Probability = 0.
    baseline_lenght: NumberOfTrial = 200
    extinction_lenght: NumberOfTrial = 200

    results: List[List[OutputData]] = []
    # run simulations for each baseline reward probability
    for blrp in baseline_reward_probs:
        agent = GAIAgent(1., bias=5.)
        _baseline_result = run(agent, trial=baseline_lenght, reward_prob=blrp)
        _ext_result = run(agent,
                          trial=extinction_lenght,
                          reward_prob=extinction)
        baseline_result: List[OutputData] = \
            list(map(lambda br: (blrp, ) + br, _baseline_result))
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
