from abc import ABCMeta, abstractmethod, abstractproperty
from typing import Any, Dict, List, Tuple

import numpy as np
from nptyping import NDArray
from scipy.special import betaln, digamma

from aibes.types import (Action, NumberOfOptions, Parameters, Prediction,
                         Probability, Reward)


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

    @abstractmethod
    def get_params(self, names: List[str]) -> Tuple[Any, ...]:
        pass

    @abstractmethod
    def set_params(self, names_and_vals: List[Tuple[str, Any]]):
        pass

    @abstractproperty
    def params(self) -> Parameters:
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
    def __init__(self,
                 lamb: float,
                 k: NumberOfOptions,
                 lr_alpha: float,
                 lr_beta: float,
                 bias: float = 6.):
        self.__alpha = np.exp(2 * lamb)
        self.__params: Parameters = {
            "alpha": np.exp(2 * lamb),
            "alpha_t": np.ones(k),
            "beta_t": np.ones(k)
        }
        self.__lr_alpha = lr_alpha
        self.__lr_beta = lr_beta
        self.__bias = bias
        self.__k = k

    def update(self, reward: NDArray[1, Reward], action: NDArray[1, Action]):
        alpha_t, beta_t = self.get_params(["alpha_t", "beta_t"])
        alpha_t += reward * action * self.__lr_alpha
        beta_t += (1 - reward) * action * self.__lr_beta
        self.set_params([("alpha_t", alpha_t), ("beta_t", beta_t)])

    def predict(self) -> Tuple[NDArray[1, Prediction], NDArray[1, float]]:
        alpha, alpha_t, beta_t = self.get_params(
            ["alpha", "alpha_t", "beta_t"])
        nu_t = alpha_t + beta_t
        mu_t = alpha_t / nu_t
        kl_div_a = -betaln(alpha_t, beta_t) \
            + (alpha_t - alpha) * digamma(alpha_t) \
            + (beta_t - 1) * digamma(beta_t) \
            + (alpha + 1 - nu_t) * digamma(nu_t)
        h_a = - mu_t * digamma(alpha_t + 1) \
            - (1 - mu_t) * digamma(beta_t + 1) \
            + digamma(nu_t + 1)
        epistemic = \
            mu_t * (-np.log(mu_t) + digamma(alpha_t + 1) - digamma(nu_t + 1)) \
            + (1 - mu_t) * (-np.log(1 - mu_t) + digamma(beta_t + 1) - digamma(nu_t + 1))
        pragmatic = kl_div_a + h_a - epistemic
        return pragmatic, epistemic

    def calculate_response_probs(
            self, preds: NDArray[1, Prediction]) -> NDArray[1, Probability]:
        if self.__k == 1:
            return 1 / (1 + np.exp(-(preds + self.__bias)))
        pmax = np.max(preds)
        pexp = np.exp(preds - pmax)
        return pexp / np.sum(pexp)

    def choose_action(self, probs: NDArray[1,
                                           Probability]) -> NDArray[1, Action]:
        if self.__k == 1:
            return (np.random.uniform() <= probs).astype(np.uint8)
        act = np.random.choice(self.__k, p=probs)
        return np.identity(self.__k)[act]

    def get_params(self, names: List[str]) -> Tuple[Any, ...]:
        return tuple([self.__params[p] for p in names])

    def set_params(self, names_and_vals: List[Tuple[str, Any]]):
        for n, v in names_and_vals:
            self.__params[n] = v

    @property
    def params(self) -> Parameters:
        return self.__params

    @property
    def k(self) -> int:
        return self.__k


class SAIStaticAgent(Agent):
    """
    An implementation of the model proposed by Markovic, et al. (2021).
    Original article and author's implementations are available on https://arxiv.org/pdf/2101.08699.pdf and https://github.com/dimarkov/aibandits respectively.
    `SAIAgent` is who tried to minimize expected surprisal instead of expected free energy.
    """
    def __init__(self,
                 lamb: float,
                 k: NumberOfOptions,
                 lr_alpha: float,
                 lr_beta: float,
                 bias: float = 0.):
        self.__lambda = lamb
        self.__params: Parameters = {
            "lambda": lamb,
            "alpha_t": np.ones(k),
            "beta_t": np.ones(k)
        }
        self.__lr_alpha = lr_alpha
        self.__lr_beta = lr_beta
        self.__bias = bias
        self.__k = k

    def update(self, reward: Reward, action: Action):
        alpha_t, beta_t = self.get_params(["alpha_t", "beta_t"])
        alpha_t += reward * action * self.__lr_alpha
        beta_t += (1 - reward) * action * self.__lr_beta
        self.set_params([("alpha_t", alpha_t), ("beta_t", beta_t)])

    def predict(self) -> Tuple[NDArray[1, Prediction], NDArray[1, Prediction]]:
        lamb, alpha_t, beta_t = self.get_params(
            ["lambda", "alpha_t", "beta_t"])
        nu_t = alpha_t + beta_t
        mu_t = alpha_t / nu_t

        kl_div_a = - lamb * (2 * mu_t - 1) \
            + mu_t * np.log(mu_t) \
            + (1 - mu_t) * np.log(1 - mu_t)
        h_a = - mu_t * digamma(alpha_t + 1) \
            - (1 - mu_t) * digamma(beta_t + 1) \
            + digamma(nu_t + 1)
        epistemic = \
            mu_t * (-np.log(mu_t) + digamma(alpha_t + 1) - digamma(nu_t + 1)) \
            + (1 - mu_t) * (-np.log(1 - mu_t) + digamma(beta_t + 1) - digamma(nu_t + 1))
        pragmatic = kl_div_a + h_a - epistemic
        return pragmatic, epistemic

    def calculate_response_probs(
            self, preds: NDArray[1, Prediction]) -> NDArray[1, Probability]:
        if self.__k == 1:
            return 1 / (1 + np.exp(-(preds + self.__bias)))
        pmax = np.max(preds)
        pexp = np.exp(preds - pmax)
        return pexp / np.sum(pexp)

    def choose_action(self, probs: NDArray[1,
                                           Probability]) -> NDArray[1, Action]:
        if self.__k:
            return (np.random.uniform() <= probs).astype(np.uint8)
        act = np.random.choice(self.__k, p=probs)
        return np.identity(self.__k)[act]

    def get_params(self, names: List[str]) -> Tuple[Any, ...]:
        return tuple([self.__params[p] for p in names])

    def set_params(self, names_and_vals: List[Tuple[str, Any]]):
        for n, v in names_and_vals:
            self.__params[n] = v

    @property
    def params(self) -> Parameters:
        return self.__params

    @property
    def k(self) -> int:
        return self.__k


class GAIDynamicAgent(Agent):
    def __init__(self,
                 lamb: float,
                 k: NumberOfOptions,
                 lr_alpha: float,
                 lr_beta: float,
                 bias: float = 4.):
        self.__params: Parameters = {
            "alpha": np.exp(2 * lamb),
            "alpha_0": np.ones(k),
            "alpha_t": np.ones(k),
            "beta_0": np.ones(k),
            "beta_t": np.ones(k),
            "a": np.full(k, 0.5),
            "b": np.full(k, 20.),
            "omega": np.zeros(k)
        }
        self.__lr_alpha = lr_alpha
        self.__lr_beta = lr_beta
        self.__bias = bias
        self.__k = k

    def update(self, reward: NDArray[1, Reward], action: NDArray[1, Action]):
        a, b, alpha_t, alpha_0, beta_t, beta_0, omega = self.get_params(
            ["a", "b", "alpha_t", "alpha_0", "beta_t", "beta_0", "omega"])

        m = a / (a + b)
        mu = alpha_t / (alpha_t + beta_t)
        eta = .5 * omega / (.5 * omega + (mu * reward + (1 - mu) *
                                          (1 - reward)) * (1 - omega))
        omega = m * (1 - eta)
        alpha_t_new = (1 - eta) * alpha_t \
            + eta * alpha_0 \
            + reward
        alpha_t += (alpha_t_new - alpha_t) * action * self.__lr_alpha
        beta_t_new = (1 - eta) * beta_t \
            + eta * beta_0 \
            + (1 - reward)
        beta_t += (beta_t_new - beta_t) * action * self.__lr_beta
        a += omega
        b += 1 - eta - omega
        self.set_params([("alpha_t", alpha_t), ("beta_t", beta_t),
                         ("omega", omega), ("a", a), ("b", b)])

    def predict(self) -> Tuple[NDArray[1, Prediction], NDArray[1, Prediction]]:
        alpha, alpha_t, beta_t = self.get_params(
            ["alpha", "alpha_t", "beta_t"])
        nu_t = alpha_t + beta_t
        mu_t = alpha_t / nu_t
        kl_div_a = -betaln(alpha_t, beta_t) \
            + (alpha_t - alpha) * digamma(alpha_t) \
            + (beta_t - 1) * digamma(beta_t) \
            + (alpha + 1 - nu_t) * digamma(nu_t)
        h_a = - mu_t * digamma(alpha_t + 1) \
            - (1 - mu_t) * digamma(beta_t + 1) \
            + digamma(nu_t + 1)
        epistemic = \
            mu_t * (-np.log(mu_t) + digamma(alpha_t + 1) - digamma(nu_t + 1)) \
            + (1 - mu_t) * (-np.log(1 - mu_t) + digamma(beta_t + 1) - digamma(nu_t + 1))
        pragmatic = kl_div_a + h_a - epistemic
        return pragmatic, epistemic

    def calculate_response_probs(
            self, preds: NDArray[1, Prediction]) -> NDArray[1, Probability]:
        if self.__k == 1:
            return 1 / (1 + np.exp(-(preds + self.__bias)))
        pmax = np.max(preds)
        pexp = np.exp(preds - pmax)
        return pexp / np.sum(pexp)

    def choose_action(self, probs: NDArray[1,
                                           Probability]) -> NDArray[1, Action]:
        if self.__k == 1:
            return (np.random.uniform() <= probs).astype(np.uint8)
        act = np.random.choice(self.__k, p=probs)
        return np.identity(self.__k)[act]

    def get_params(self, names: List[str]) -> Tuple[Any, ...]:
        return tuple([self.__params[p] for p in names])

    def set_params(self, names_and_vals: List[Tuple[str, Any]]):
        for n, v in names_and_vals:
            self.__params[n] = v

    @property
    def params(self) -> Parameters:
        return self.__params

    @property
    def k(self) -> int:
        return self.__k


class SAIDynamicAgent(Agent):
    """
    An implementation of the model proposed by Markovic, et al. (2021).
    Original article and author's implementations are available on https://arxiv.org/pdf/2101.08699.pdf and https://github.com/dimarkov/aibandits respectively.
    `SAIAgent` is who tried to minimize expected surprisal instead of expected free energy.
    """
    def __init__(self,
                 lamb: float,
                 k: NumberOfOptions,
                 lr_alpha: float,
                 lr_beta: float,
                 bias: float = 0.):
        self.__params: Parameters = {
            "lambda": lamb,
            "alpha_0": np.ones(k),
            "alpha_t": np.ones(k),
            "beta_0": np.ones(k),
            "beta_t": np.ones(k),
            "a": np.full(k, 0.5),
            "b": np.full(k, 20.),
            "omega": np.zeros(k)
        }
        self.__lr_alpha = lr_alpha
        self.__lr_beta = lr_beta
        self.__bias = bias
        self.__k = k

    def update(self, reward: NDArray[1, Reward], action: NDArray[1, Action]):
        a, b, alpha_t, alpha_0, beta_t, beta_0, omega = self.get_params(
            ["a", "b", "alpha_t", "alpha_0", "beta_t", "beta_0", "omega"])
        m = a / (a + b)
        mu = alpha_t / (alpha_t + beta_t)
        eta = .5 * omega / (.5 * omega + (mu * reward + (1 - mu) *
                                          (1 - reward)) * (1 - omega))
        omega = m * (1 - eta)
        alpha_t_new = (1 - eta) * alpha_t \
            + eta * alpha_0 \
            + reward
        alpha_t += (alpha_t_new - alpha_t) * action * self.__lr_alpha
        beta_t_new = (1 - eta) * beta_t \
            + eta * beta_0 \
            + (1 - reward)
        beta_t += (beta_t_new - beta_t) * action * self.__lr_beta
        a += omega
        b += 1 - eta - omega
        self.set_params([("alpha_t", alpha_t), ("beta_t", beta_t),
                         ("omega", omega), ("a", a), ("b", b)])

    def predict(self) -> Tuple[NDArray[1, Prediction], NDArray[1, Prediction]]:
        lamb, alpha_t, beta_t = self.get_params(
            ["lambda", "alpha_t", "beta_t"])
        nu_t = alpha_t + beta_t
        mu_t = alpha_t / nu_t

        kl_div_a = - lamb * (2 * mu_t - 1) \
            + mu_t * np.log(mu_t) \
            + (1 - mu_t) * np.log(1 - mu_t)
        h_a = - mu_t * digamma(alpha_t + 1) \
            - (1 - mu_t) * digamma(beta_t + 1) \
            + digamma(nu_t + 1)
        epistemic = \
            mu_t * (-np.log(mu_t) + digamma(alpha_t + 1) - digamma(nu_t + 1)) \
            + (1 - mu_t) * (-np.log(1 - mu_t) + digamma(beta_t + 1) - digamma(nu_t + 1))
        pragmatic = kl_div_a + h_a - epistemic
        return pragmatic, epistemic

    def calculate_response_probs(
            self, preds: NDArray[1, Prediction]) -> NDArray[1, Probability]:
        if self.__k == 1:
            return 1 / (1 + np.exp(-(preds + self.__bias)))
        pmax = np.max(preds)
        pexp = np.exp(preds - pmax)
        return pexp / np.sum(pexp)

    def choose_action(self, probs: NDArray[1,
                                           Probability]) -> NDArray[1, Action]:
        if self.__k == 1:
            return (np.random.uniform() <= probs).astype(np.uint8)
        act = np.random.choice(self.__k, p=probs)
        return np.identity(self.__k)[act]

    def get_params(self, names: List[str]) -> Tuple[Any, ...]:
        return tuple([self.__params[p] for p in names])

    def set_params(self, names_and_vals: List[Tuple[str, Any]]):
        for n, v in names_and_vals:
            self.__params[n] = v

    @property
    def params(self) -> Parameters:
        return self.__params

    @property
    def k(self) -> int:
        return self.__k
