import aibes.model as m
import numpy as np
import pytest

AGENT_CLASSES = [
    m.GAIStaticAgent, m.SAIStaticAgent, m.GAIDynamicAgent, m.SAIDynamicAgent
]
GAIAGENT_CLASSES = [m.GAIStaticAgent, m.GAIDynamicAgent]
SAIAGENT_CLASSES = [m.SAIStaticAgent, m.SAIDynamicAgent]


@pytest.mark.parametrize("k, expected", [(1, np.ones(1)), (2, np.ones(2)),
                                         (3, np.ones(3)), (100, np.ones(100))])
def test_model_initialize(k, expected):
    for ac in AGENT_CLASSES:
        agent = ac(1., k, 0.1)
        agent = m.GAIStaticAgent(1., k, 0.1)
        alpha_t, beta_t = agent.get_params(["alpha_t", "beta_t"])
        assert sum(alpha_t == expected)
        assert sum(beta_t == expected)


@pytest.mark.parametrize(
    "rewards, actions, lr, expected_alpha, expected_beta", [
        (np.zeros(2), np.zeros(2), 0.1, np.ones(2), np.ones(2)),
        (np.zeros(2), np.ones(2), 0.1, np.ones(2), np.full(2, 1.1)),
        (np.ones(2), np.ones(2), 0.1, np.full(2, 1.1), np.ones(2)),
        (np.ones(2), np.zeros(2), 0.1, np.ones(2), np.ones(2)),
        (np.ones(2), np.ones(2), 0.2, np.full(2, 1.2), np.ones(2)),
        (np.array([1, 0]), np.ones(2), 0.1, np.array([1.1, 1
                                                      ]), np.array([1., 1.1])),
        (np.array([1, 0]), np.ones(2), 0.2, np.array([1.2, 1
                                                      ]), np.array([1., 1.2])),
        (np.array([0, 1]), np.ones(2), 0.1, np.array([1., 1.1
                                                      ]), np.array([1.1, 1.])),
        (np.array([0, 1]), np.ones(2), 0.2, np.array([1., 1.2
                                                      ]), np.array([1.2, 1.])),
    ])
def test_update(rewards, actions, lr, expected_alpha, expected_beta):
    k = 2
    for ac in AGENT_CLASSES:
        agent: m.Agent = ac(1., k, lr)
        agent.update(rewards, actions)
        alpha_t, beta_t = agent.get_params(["alpha_t", "beta_t"])
        assert sum(alpha_t == expected_alpha) == k
        assert sum(beta_t == expected_beta) == k


def G(alpha_t, beta_t, alpha):
    """
    Original implemation by Markovic (https://github.com/dimarkov/aibandits/blob/master/choice_algos.py)
    """
    from scipy.special import betaln, digamma
    nu_t = alpha_t + beta_t
    mu_t = alpha_t / nu_t

    KL_a = - betaln(alpha_t, beta_t) + (alpha_t - alpha) * digamma(alpha_t)\
             + (beta_t - 1) * digamma(beta_t) + (alpha + 1 - nu_t) * digamma(nu_t)
    H_a = -mu_t * digamma(alpha_t + 1) - (
        1 - mu_t) * digamma(beta_t + 1) + digamma(nu_t + 1)

    return KL_a + H_a


@pytest.mark.parametrize("alpha_t, beta_t, expected", [
    (np.array([1, 1]), np.array(
        [1, 1]), G(np.array([1, 1]), np.array([1, 1]), np.exp(2))),
    (np.array([2, 1]), np.array(
        [1, 1]), G(np.array([2, 1]), np.array([1, 1]), np.exp(2))),
    (np.array([1, 2]), np.array(
        [1, 1]), G(np.array([1, 2]), np.array([1, 1]), np.exp(2))),
    (np.array([1, 1]), np.array(
        [2, 1]), G(np.array([1, 1]), np.array([2, 1]), np.exp(2))),
    (np.array([1, 1]), np.array(
        [1, 2]), G(np.array([1, 1]), np.array([1, 2]), np.exp(2))),
    (np.array([2, 2]), np.array(
        [2, 2]), G(np.array([2, 2]), np.array([2, 2]), np.exp(2))),
    (np.array([100, 100]), np.array(
        [100, 100]), G(np.array([100, 100]), np.array([100, 100]), np.exp(2))),
])
def test_predict_gaia(alpha_t, beta_t, expected):
    k = 2
    lr = 0.1
    for ac in GAIAGENT_CLASSES:
        agent: m.Agent = ac(1., k, lr)
        agent.set_params([("alpha_t", alpha_t), ("beta_t", beta_t)])
        pragmatic, epistemic = agent.predict()
        preds = pragmatic + epistemic
        print(preds, expected)
        assert sum(preds == expected) == 2


def S(alpha_t, beta_t, lam):
    """
    Original implemation by Markovic (https://github.com/dimarkov/aibandits/blob/master/choice_algos.py)
    """
    from scipy.special import digamma
    nu_t = alpha_t + beta_t
    mu_t = alpha_t / nu_t

    KL_a = -lam * (2 * mu_t -
                   1) + mu_t * np.log(mu_t) + (1 - mu_t) * np.log(1 - mu_t)
    H_a = -mu_t * digamma(alpha_t + 1) - (
        1 - mu_t) * digamma(beta_t + 1) + digamma(nu_t + 1)

    return KL_a + H_a


@pytest.mark.parametrize("alpha_t, beta_t, expected", [
    (np.array([1, 1]), np.array(
        [1, 1]), S(np.array([1, 1]), np.array([1, 1]), 1.)),
    (np.array([2, 1]), np.array(
        [1, 1]), S(np.array([2, 1]), np.array([1, 1]), 1.)),
    (np.array([1, 2]), np.array(
        [1, 1]), S(np.array([1, 2]), np.array([1, 1]), 1.)),
    (np.array([1, 1]), np.array(
        [2, 1]), S(np.array([1, 1]), np.array([2, 1]), 1.)),
    (np.array([1, 1]), np.array(
        [1, 2]), S(np.array([1, 1]), np.array([1, 2]), 1.)),
    (np.array([2, 2]), np.array(
        [2, 2]), S(np.array([2, 2]), np.array([2, 2]), 1.)),
    (np.array([100, 100]), np.array(
        [100, 100]), S(np.array([100, 100]), np.array([100, 100]), 1.)),
])
def test_predict_saia(alpha_t, beta_t, expected):
    k = 2
    lr = 0.1
    for ac in SAIAGENT_CLASSES:
        agent: m.Agent = ac(1., k, lr)
        agent.set_params([("alpha_t", alpha_t), ("beta_t", beta_t)])
        pragmatic, epistemic = agent.predict()
        preds = pragmatic + epistemic
        assert sum(preds == expected) == 2
