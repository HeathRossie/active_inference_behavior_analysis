from abc import ABCMeta, abstractmethod, abstractproperty
from typing import Tuple

import numpy as np
from nptyping import NDArray
from scipy.special import betaln, digamma

from aibes.types import (Action, NumberOfOptions, Prediction, Probability,
                         Reward)


class Agent(metaclass=ABCMeta):
    @abstractmethod
    def update(self, reward: NDArray[1, Reward], action: NDArray[1, Action]):
        pass

    @abstractmethod
    def predict(self) -> Tuple[NDArray[1, Prediction], NDArray[1, Prediction]]:
        pass

    @abstractmethod
    def calculate_response_probs(
            self, pred: NDArray[1, Prediction]) -> NDArray[1, Probability]:
        pass

    @abstractmethod
    def choose_action(self, prob: NDArray[1,
                                          Probability]) -> NDArray[1, Action]:
        pass

    @abstractproperty
    def alpha_t(self) -> NDArray[1, float]:
        pass

    @abstractproperty
    def beta_t(self) -> NDArray[1, float]:
        pass

    @abstractproperty
    def k(self) -> int:
        pass


class GAIStaticAgent(Agent):
    """
    An implementation of the model proposed by Markovic, et al. (2021).
    Original article and author's implementations are available on https://arxiv.org/pdf/2101.08699.pdf and https://github.com/dimarkov/aibandits respectively.
    `GAIAgent` is who tried to minimize expected free energy directly.
    """
    def __init__(self, lamb: float, k: NumberOfOptions, lr: float):
        self.__alpha = np.exp(2 * lamb)
        self.__alpha_t: NDArray[1, float] = np.ones(k)
        self.__beta_t: NDArray[1, float] = np.ones(k)
        self.__lr = lr
        self.__k = k

    def update(self, reward: NDArray[1, Reward], action: NDArray[1, Action]):
        self.__alpha_t += reward * action * self.__lr
        self.__beta_t += (1 - reward) * action * self.__lr

    def predict(self) -> Tuple[NDArray[1, Prediction], NDArray[1, float]]:
        nu_t = self.__alpha_t + self.__beta_t
        mu_t = self.__alpha_t / nu_t
        kl_div_a = -betaln(self.__alpha_t, self.__beta_t) \
            + (self.__alpha_t - self.__alpha) * digamma(self.__alpha_t) \
            + (self.__beta_t - 1) * digamma(self.__beta_t) \
            + (self.__alpha + 1 - nu_t) * digamma(nu_t)
        h_a = - mu_t * digamma(self.__alpha_t + 1) \
            - (1 - mu_t) * digamma(self.__beta_t + 1) \
            + digamma(nu_t + 1)
        epistemic = \
            mu_t * (-np.log(mu_t) + digamma(self.__alpha_t + 1) - digamma(nu_t + 1)) \
            + (1 - mu_t) * (-np.log(1 - mu_t) + digamma(self.__beta_t + 1) - digamma(nu_t + 1))
        pragmatic = kl_div_a + h_a - epistemic
        return pragmatic, epistemic

    def calculate_response_probs(
            self, preds: NDArray[1, Prediction]) -> NDArray[1, Probability]:
        pmax = np.max(preds)
        pexp = np.exp(preds - pmax)
        return pexp / np.sum(pexp)

    def choose_action(self, probs: NDArray[1,
                                           Probability]) -> NDArray[1, Action]:
        act = np.random.choice(self.__k, p=probs)
        return np.identity(self.__k)[act]

    @property
    def alpha_t(self) -> NDArray[1, float]:
        return self.__alpha_t

    @property
    def beta_t(self) -> NDArray[1, float]:
        return self.__beta_t

    @property
    def k(self) -> int:
        return self.__k


class SAIStaticAgent(Agent):
    """
    An implementation of the model proposed by Markovic, et al. (2021).
    Original article and author's implementations are available on https://arxiv.org/pdf/2101.08699.pdf and https://github.com/dimarkov/aibandits respectively.
    `SAIAgent` is who tried to minimize expected surprisal instead of expected free energy.
    """
    def __init__(self, lamb: float, k: NumberOfOptions, lr: float):
        self.__lambda = lamb
        self.__alpha_t: NDArray[1, float] = np.ones(k)
        self.__beta_t: NDArray[1, float] = np.ones(k)
        self.__lr = lr
        self.__k = k

    def update(self, reward: Reward, action: Action):
        self.__alpha_t += reward * action * self.__lr
        self.__beta_t += (1 - reward) * action * self.__lr

    def predict(self) -> Tuple[NDArray[1, Prediction], NDArray[1, Prediction]]:
        nu_t = self.__alpha_t + self.__beta_t
        mu_t = self.__alpha_t / nu_t

        kl_div_a = - self.__lambda * (2 * mu_t - 1) \
            + mu_t * np.log(mu_t) \
            + (1 - mu_t) * np.log(1 - mu_t)
        h_a = - mu_t * digamma(self.__alpha_t + 1) \
            - (1 - mu_t) * digamma(self.__beta_t + 1) \
            + digamma(nu_t + 1)
        epistemic = \
            mu_t * (-np.log(mu_t) + digamma(self.__alpha_t + 1) - digamma(nu_t + 1)) \
            + (1 - mu_t) * (-np.log(1 - mu_t) + digamma(self.__beta_t + 1) - digamma(nu_t + 1))
        pragmatic = kl_div_a + h_a - epistemic
        return pragmatic, epistemic

    def calculate_response_probs(
            self, preds: NDArray[1, Prediction]) -> NDArray[1, Probability]:
        pmax = np.max(preds)
        pexp = np.exp(preds - pmax)
        return pexp / np.sum(pexp)

    def choose_action(self, probs: NDArray[1,
                                           Probability]) -> NDArray[1, Action]:
        act = np.random.choice(self.__k, p=probs)
        return np.identity(self.__k)[act]

    @property
    def alpha_t(self) -> NDArray[1, float]:
        return self.__alpha_t

    @property
    def beta_t(self) -> NDArray[1, float]:
        return self.__beta_t

    @property
    def k(self) -> int:
        return self.__k


class GAIDynamicAgent(Agent):
    def __init__(self, lamb: float, k: NumberOfOptions, lr: float):
        self.__alpha = np.exp(2 * lamb)
        self.__alpha_t: NDArray[1, float] = np.ones(k)
        self.__alpha_zero = self.__alpha_t.copy()
        self.__beta_t: NDArray[1, float] = np.ones(k)
        self.__beta_zero = self.__beta_t.copy()
        self.__a = np.full(k, 0.5)
        self.__b = np.full(k, 20.)
        self.__omega = np.zeros(k)
        self.__lr = lr
        self.__k = k

    def update(self, reward: NDArray[1, Reward], action: NDArray[1, Action]):
        m = self.__a / (self.__a + self.__b)
        mu = self.__alpha_t / (self.__alpha_t + self.__beta_t)
        eta = .5 * self.__omega / (.5 * self.__omega +
                                   (mu * reward + (1 - mu) *
                                    (1 - reward)) * (1 - self.__omega))
        self.__omega = m * (1 - eta)
        alpha_t_new = (1 - eta) * self.__alpha_t \
            + eta * self.__alpha_zero \
            + reward
        self.__alpha_t += (alpha_t_new - self.__alpha_t) * action * self.__lr
        beta_t_new = (1 - eta) * self.__beta_t \
            + eta * self.__beta_zero \
            + (1 - reward)
        self.__beta_t += (beta_t_new - self.__beta_t) * action * self.__lr
        self.__a += self.__omega
        self.__b += 1 - eta - self.__omega

    def predict(self) -> Tuple[NDArray[1, Prediction], NDArray[1, Prediction]]:
        nu_t = self.__alpha_t + self.__beta_t
        mu_t = self.__alpha_t / nu_t
        kl_div_a = -betaln(self.__alpha_t, self.__beta_t) \
            + (self.__alpha_t - self.__alpha) * digamma(self.__alpha_t) \
            + (self.__beta_t - 1) * digamma(self.__beta_t) \
            + (self.__alpha + 1 - nu_t) * digamma(nu_t)
        h_a = - mu_t * digamma(self.__alpha_t + 1) \
            - (1 - mu_t) * digamma(self.__beta_t + 1) \
            + digamma(nu_t + 1)
        epistemic = \
            mu_t * (-np.log(mu_t) + digamma(self.__alpha_t + 1) - digamma(nu_t + 1)) \
            + (1 - mu_t) * (-np.log(1 - mu_t) + digamma(self.__beta_t + 1) - digamma(nu_t + 1))
        pragmatic = kl_div_a + h_a - epistemic
        return pragmatic, epistemic

    def calculate_response_probs(
            self, preds: NDArray[1, Prediction]) -> NDArray[1, Probability]:
        pmax = np.max(preds)
        pexp = np.exp(preds - pmax)
        return pexp / np.sum(pexp)

    def choose_action(self, probs: NDArray[1,
                                           Probability]) -> NDArray[1, Action]:
        act = np.random.choice(self.__k, p=probs)
        return np.identity(self.__k)[act]

    @property
    def alpha_t(self) -> NDArray[1, float]:
        return self.__alpha_t

    @property
    def beta_t(self) -> NDArray[1, float]:
        return self.__beta_t

    @property
    def k(self) -> int:
        return self.__k


class SAIDynamicAgent(Agent):
    """
    An implementation of the model proposed by Markovic, et al. (2021).
    Original article and author's implementations are available on https://arxiv.org/pdf/2101.08699.pdf and https://github.com/dimarkov/aibandits respectively.
    `SAIAgent` is who tried to minimize expected surprisal instead of expected free energy.
    """
    def __init__(self, lamb: float, k: NumberOfOptions, lr: float):
        self.__lambda = lamb
        self.__alpha_t: NDArray[1, float] = np.ones(k)
        self.__alpha_zero = self.__alpha_t.copy()
        self.__beta_t: NDArray[1, float] = np.ones(k)
        self.__beta_zero = self.__beta_t.copy()
        self.__a = np.full(k, 0.5)
        self.__b = np.full(k, 20.)
        self.__omega = np.zeros(k)
        self.__lr = lr
        self.__k = k

    def update(self, reward: NDArray[1, Reward], action: NDArray[1, Action]):
        m = self.__a / (self.__a + self.__b)
        mu = self.__alpha_t / (self.__alpha_t + self.__beta_t)
        eta = .5 * self.__omega / (.5 * self.__omega +
                                   (mu * reward + (1 - mu) *
                                    (1 - reward)) * (1 - self.__omega))
        self.__omega = m * (1 - eta)
        alpha_t_new = (1 - eta) * self.__alpha_t \
            + eta * self.__alpha_zero \
            + reward
        self.__alpha_t += (alpha_t_new - self.__alpha_t) * action * self.__lr
        beta_t_new = (1 - eta) * self.__beta_t \
            + eta * self.__beta_zero \
            + (1 - reward)
        self.__beta_t += (beta_t_new - self.__beta_t) * action * self.__lr
        self.__a += self.__omega
        self.__b += 1 - eta - self.__omega

    def predict(self) -> Tuple[NDArray[1, Prediction], NDArray[1, Prediction]]:
        nu_t = self.__alpha_t + self.__beta_t
        mu_t = self.__alpha_t / nu_t

        kl_div_a = - self.__lambda * (2 * mu_t - 1) \
            + mu_t * np.log(mu_t) \
            + (1 - mu_t) * np.log(1 - mu_t)
        h_a = - mu_t * digamma(self.__alpha_t + 1) \
            - (1 - mu_t) * digamma(self.__beta_t + 1) \
            + digamma(nu_t + 1)
        epistemic = \
            mu_t * (-np.log(mu_t) + digamma(self.__alpha_t + 1) - digamma(nu_t + 1)) \
            + (1 - mu_t) * (-np.log(1 - mu_t) + digamma(self.__beta_t + 1) - digamma(nu_t + 1))
        pragmatic = kl_div_a + h_a - epistemic
        return pragmatic, epistemic

    def calculate_response_probs(
            self, preds: NDArray[1, Prediction]) -> NDArray[1, Probability]:
        pmax = np.max(preds)
        pexp = np.exp(preds - pmax)
        return pexp / np.sum(pexp)

    def choose_action(self, probs: NDArray[1,
                                           Probability]) -> NDArray[1, Action]:
        act = np.random.choice(self.__k, p=probs)
        return np.identity(self.__k)[act]

    @property
    def alpha_t(self) -> NDArray[1, float]:
        return self.__alpha_t

    @property
    def beta_t(self) -> NDArray[1, float]:
        return self.__beta_t

    @property
    def k(self) -> int:
        return self.__k
