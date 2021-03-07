from abc import ABC, ABCMeta

import numpy as np
from nptyping import NDArray
from scipy.special import betaln, digamma

Prediction = float
Probability = float
Action = int
Reward = int
NumberOfOptions = int


class Agent(metaclass=ABCMeta):
    # TODO: parameterize the learning rate
    def update(self, reward: NDArray[1, Reward], action: NDArray[1, Action]):
        pass

    def predict(self) -> NDArray[1, Prediction]:
        pass

    def calculate_response_probs(
            self, pred: NDArray[1, Prediction]) -> NDArray[1, Probability]:
        pass

    def choose_action(self, prob: NDArray[1,
                                          Probability]) -> NDArray[1, Action]:
        return 0


class GAIAgent(Agent):
    def __init__(self, lamb: float, k: NumberOfOptions):
        self.__alpha = np.exp(2 * lamb)
        self.__alpha_t = np.ones(k)
        self.__beta_t = np.ones(k)
        self.__k = k

    def update(self, reward: NDArray[1, Reward], action: NDArray[1, Action]):
        self.__alpha_t += reward * action * 0.1
        self.__beta_t += (1 - reward) * action * 0.1

    def predict(self) -> NDArray[1, Prediction]:
        nu_t = self.__alpha_t + self.__beta_t
        mu_t = self.__alpha_t / nu_t
        kl_div_a = -betaln(self.__alpha_t, self.__beta_t) \
            + (self.__alpha_t - self.__alpha) * digamma(self.__alpha_t) \
            + (self.__beta_t - 1) * digamma(self.__beta_t) \
            + (self.__alpha + 1 - nu_t) * digamma(nu_t)
        h_a = - mu_t * digamma(self.__alpha_t + 1) \
            - (1 - mu_t) * digamma(self.__beta_t + 1) \
            + digamma(nu_t + 1)
        return kl_div_a + h_a

    def calculate_response_probs(
            self, preds: NDArray[1, Prediction]) -> NDArray[1, Probability]:
        pmax = np.max(preds)
        pexp = np.exp(preds - pmax)
        return pexp / np.sum(pexp)

    def choose_action(self, probs: NDArray[1,
                                           Probability]) -> NDArray[1, Action]:
        act = np.random.choice(self.__k, p=probs)
        return np.identity(self.__k)[act]


class SAIAgent(Agent):
    def __init__(self, lamb: float, k: NumberOfOptions):
        self.__lambda = lamb
        self.__alpha_t = np.ones(k)
        self.__beta_t = np.ones(k)
        self.__k = k

    def update(self, reward: Reward, action: Action):
        self.__alpha_t += reward * action * 0.1
        self.__beta_t[action] = (1 - reward) * action * 0.1

    def predict(self) -> NDArray[1, Prediction]:
        nu_t = self.__alpha_t + self.__beta_t
        mu_t = self.__alpha_t / nu_t

        kl_div_a = - self.__lambda * (2 * mu_t - 1) \
            + mu_t * np.log(mu_t) \
            + (1 - mu_t) * np.log(1 - mu_t)
        h_a = - mu_t * digamma(self.__alpha_t + 1) \
            - (1 - mu_t) * digamma(self.__beta_t + 1) \
            + digamma(nu_t + 1)
        return kl_div_a + h_a

    def calculate_response_probs(
            self, preds: NDArray[1, Prediction]) -> NDArray[1, Probability]:
        pmax = np.max(preds)
        pexp = np.exp(preds - pmax)
        return pexp / np.sum(pexp)

    def choose_action(self, probs: NDArray[1,
                                           Probability]) -> NDArray[1, Action]:
        act = np.random.choice(self.__k, p=probs)
        return np.identity(self.__k)[act]
