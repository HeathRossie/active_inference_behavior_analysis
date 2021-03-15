from typing import List, Tuple

import numpy as np
import pandas as pd
from aibes.model import Action, Agent, Prediction, Probability, Reward
from aibes.schedule import (ConcurrentSchedule, Extinction, VariableInterval,
                            VariableRatio)
from aibes.types import (Duration, Interval, NumberOfOptions, NumberOfReward,
                         RequiredResponse)
from aibes.util import all_parameters_combination, colnames, load_yaml
from nptyping import NDArray

Result = Tuple[Action, Reward, NDArray[1, Prediction], NDArray[1, Prediction],
               NDArray[1, Probability]]
OutputData = Tuple[Probability, Action, Reward, Prediction, Probability]


def run_baseline(agent: Agent, schedule: ConcurrentSchedule,
                 timestep: Interval):
    def trial_process(agent: Agent, schedule: ConcurrentSchedule,
                      timestep: float) -> Result:
        pragmatic, epistemic = agent.predict()
        preds = pragmatic + epistemic
        probs = agent.calculate_response_probs(-preds)
        actions = agent.choose_action(probs)
        counts = actions.copy().tolist()
        counts[1:] = [timestep for _ in range(len(counts) - 1)]
        rewards = schedule.step(counts, actions.tolist())
        agent.update(rewards, actions.astype(np.uint8))
        chosen_action: Action = np.argmax(actions)
        reward: Reward = np.max(rewards)
        return chosen_action, reward, pragmatic, epistemic, probs

    ret_actions, ret_rewards = np.zeros(1), np.zeros(1)
    ret_pragmatics, ret_epistemics = np.zeros(agent.k), np.zeros(agent.k)
    ret_probs = np.zeros(agent.k)
    while not schedule.finished():
        action, reward, pragmatics, epistemics, probs = \
            trial_process(agent, schedule, timestep)
        ret_actions = np.vstack((ret_actions, action))
        ret_rewards = np.vstack((ret_rewards, reward))
        ret_pragmatics = np.vstack((ret_pragmatics, pragmatics))
        ret_epistemics = np.vstack((ret_epistemics, epistemics))
        ret_probs = np.vstack((ret_probs, probs))
    return np.hstack(
        (ret_actions, ret_rewards, ret_pragmatics, ret_epistemics, ret_probs))


def run_extinction(agent: Agent, schedule: ConcurrentSchedule,
                   timestep: Interval):
    counts = [timestep for _ in range(len(schedule.val))]

    def trial_process(agent: Agent, schedule: ConcurrentSchedule,
                      timestep: Interval) -> Result:
        pragmatic, epistemic = agent.predict()
        preds = pragmatic + epistemic
        probs = agent.calculate_response_probs(-preds)
        actions = agent.choose_action(probs)
        rewards = schedule.step(counts, actions.tolist())
        agent.update(rewards, actions.astype(np.uint8))
        chosen_action: Action = np.argmax(actions)
        reward: Reward = np.max(rewards)
        return chosen_action, reward, pragmatic, epistemic, probs

    ret_actions, ret_rewards = np.zeros(1), np.zeros(1)
    ret_pragmatics, ret_epistemics = np.zeros(agent.k), np.zeros(agent.k)
    ret_probs = np.zeros(agent.k)
    while not schedule.finished():
        action, reward, pragmatics, epistemics, probs = \
            trial_process(agent, schedule, timestep)
        ret_actions = np.vstack((ret_actions, action))
        ret_rewards = np.vstack((ret_rewards, reward))
        ret_pragmatics = np.vstack((ret_pragmatics, pragmatics))
        ret_epistemics = np.vstack((ret_epistemics, epistemics))
        ret_probs = np.vstack((ret_probs, probs))
    return np.hstack(
        (ret_actions, ret_rewards, ret_pragmatics, ret_epistemics, ret_probs))


if __name__ == '__main__':
    import argparse as ap

    from aibes.model import (GAIDynamicAgent, GAIStaticAgent, SAIDynamicAgent,
                             SAIStaticAgent)
    from aibes.util import get_nth_ancestor
    from pandas import DataFrame

    clap = ap.ArgumentParser(description="about this simulation")
    clap.add_argument("--agent-type", "-A", type=str, default="GAIStaticAgent")
    clap.add_argument("--requirement", "-r", type=float, default=1.)
    clap.add_argument("--num-reward", "-R", type=int, default=300)
    clap.add_argument("--num-alternatives", "-a", type=int, default=1)
    clap.add_argument("--mean-interval", "-i", type=float, default=120.)
    clap.add_argument("--extinction-duration", "-e", type=float, default=1200.)
    clap.add_argument("--timestep", "-t", type=float, default=1.)
    clap.add_argument("--learning-rate-alpha", type=float, default=.1)
    clap.add_argument("--learning-rate-beta", type=float, default=.1)
    clap.add_argument("--lamb", "-L", type=float, default=1.)
    clap.add_argument("--yaml", "-y", type=str)
    args = clap.parse_args()

    if args.yaml is None:
        req: List[RequiredResponse] = [args.requirement]
        nr: List[NumberOfReward] = [args.num_reward]
        na: List[NumberOfOptions] = [args.num_alternatives]
        mi: List[Interval] = [args.mean_interval]
        ed: List[Duration] = [args.extinction_duration]
        ts: List[Interval] = [args.timestep]
        lra: List[float] = [args.learning_rate_alpha]
        lrb: List[float] = [args.learning_rate_beta]
        l: List[float] = [args.lamb]
        at: List[str] = [args.agent_type]
        params = list(zip(at, req, nr, na, mi, ed, ts, lra, lrb, l))
    else:
        config = load_yaml(args.yaml)
        params = all_parameters_combination(config)

    for agent_type, requirement, num_rewards, num_alts, mean_interval, \
            extinction_duration, timestep, lr_alpha, lr_beta, lamb in params:
        if agent_type == "GAIStaticAgent":
            AgentClass = GAIStaticAgent
        elif agent_type == "GAIDynamicAgent":
            AgentClass = GAIDynamicAgent
        elif agent_type == "SAIStaticAgent":
            AgentClass = SAIStaticAgent
        elif agent_type == "SAIDynamicAgent":
            AgentClass = SAIDynamicAgent
        else:
            print(
                f"""{agent_type} does not exists.\nAvailbele arguments are `GAIDynamicAgent`, `GAIDynamicAgent`, `SAIStaticAgent`, and `SAIDynamicAgent`)"""
            )
            exit()

        agent = AgentClass(lamb, 1 + num_alts, lr_alpha, lr_beta)

        target_schedule = VariableRatio(requirement + 1e-8, num_rewards, 0)
        alternative_schedules = [
            VariableInterval(mean_interval, 1000, timestep)
            for _ in range(num_alts)
        ]
        [s.forever() for s in alternative_schedules]

        baseline_schedules = ConcurrentSchedule([target_schedule] +
                                                alternative_schedules)
        baseline_result = run_baseline(agent, baseline_schedules, 1.)
        nrow, _ = baseline_result.shape
        reward_probs = np.full((nrow, 1), 1 / requirement)

        target_schedules = Extinction(extinction_duration)
        [alt.reset for alt in alternative_schedules]
        extinction_schedules = ConcurrentSchedule([target_schedules] +
                                                  alternative_schedules)
        ext_result = run_extinction(agent, extinction_schedules, 1.)
        nrow, _ = ext_result.shape
        reward_probs = np.vstack((reward_probs, np.full((nrow, 1), 0.)))

        data_dir = get_nth_ancestor(__file__, 0).joinpath("data")
        if not data_dir.exists():
            data_dir.mkdir()
        filename = data_dir.joinpath(
            f"{agent_type}_lra-{lr_alpha}_lrb-{lr_beta}_lambda-{lamb}_VR-{requirement}_nrewards-{num_rewards}_nalts-{num_alts}_VI-{mean_interval}.csv"
        )

        merged_result = np.vstack((baseline_result, ext_result))
        output_data = pd.DataFrame(np.hstack((reward_probs, merged_result)),
                                   columns=colnames(agent,
                                                    "reward_probability"))
        output_data.to_csv(filename, index=False)
