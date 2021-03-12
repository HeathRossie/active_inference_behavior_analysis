library(tidyverse)

# functions used in all simulations
betaln <- function(a, b) {
  log(beta(a, b))
}

calc_expected_free_energy <- function(alpha, alpha_t, beta_t) {
  nu_t <- alpha_t + beta_t
  mu_t <- alpha_t / nu_t
  kl_div_a <- -betaln(alpha_t, beta_t) +
    (alpha_t - alpha) * digamma(alpha_t) +
    (beta_t - 1) * digamma(beta_t) +
    (alpha + 1 - nu_t) * digamma(nu_t)

  h_a <- - mu_t * digamma(alpha_t + 1) -
    (1 - mu_t) * digamma(beta_t + 1) +
    digamma(nu_t + 1)

  kl_div_a + h_a
}

# approximate g analyticaly under given parameter in the original model

# expected value of binominal distribution
expected_reward <- function(p, n) {
  p * n
}

expected_extinction <- function(p, n) {
  (1 - p) * n
}

# variation of a between n and n + 1
# p(n + 1) - pn
# = pn + p - pn
# = p
delta_alpha <- function(p) {
  p
}

# variation of b between n and n + 1
# q(n + 1) - qn (where q = 1 - p)
# = qn + q - qn
# = q = (1 - p)
delta_beta <- function(p) {
  1 - p
}

approx_expected_free_energy <- function(p1, n1, p2, n2, lambda) {
  alpha <- exp(2 * lambda)
  alpha_t <- 1.
  beta_t <- 1.

  component_1 <- seq_len(n1) %>%
    lapply(., function(i) {
      alpha_t <<- alpha_t + delta_alpha(p1)
      beta_t <<- beta_t + delta_beta(p1)
      g <- calc_expected_free_energy(alpha, alpha_t, beta_t)
      data.frame(p = p1, t = i, g = g)
  }) %>%
    do.call(rbind, .)

  component_2 <- seq_len(n2) %>%
    lapply(., function(i) {
      alpha_t <<- alpha_t + delta_alpha(p2)
      beta_t <<- beta_t + delta_beta(p2)
      g <- calc_expected_free_energy(alpha, alpha_t, beta_t)
      data.frame(p = p2, t = i + n1, g = g)
  }) %>%
    do.call(rbind, .)

  rbind(component_1, component_2)
}


# fixed across all conditions
n1 <- 100
n2 <- 400
lambda <- 0

# approximate g under extinction
baseline_probs <- seq(1., 0.0, by = -0.1)

ext_approx <- baseline_probs %>%
  lapply(., function(p1) {
    p2 <- 0.
    name <- paste0(p1, "=>", p2)

    res <- approx_expected_free_energy(p1, n1, p2, n2, lambda)
    res$cond <- rep(name, nrow(res))
    res
}) %>%
  do.call(rbind, .)

ext_approx <- transform(ext_approx,
                        cond = factor(cond, levels = unique(ext_approx$cond)))

ggplot(data = ext_approx) +
  geom_line(aes(x = t, y = g, color = cond))

ggplot(data = ext_approx) +
  geom_line(aes(x = t, y = g)) +
  facet_wrap(~cond)


# approximate g under acquisition
acquisition_prob <- seq(0., 1., by = 0.1)

acq_approx <- acquisition_prob %>%
  lapply(., function(p1) {
    p2 <- 1.
    name <- paste0(p1, "=>", p2)

    res <- approx_expected_free_energy(p1, n1, p2, n2, lambda)
    res$cond <- rep(name, nrow(res))
    res
}) %>%
  do.call(rbind, .)

acq_approx <- transform(acq_approx,
                        cond = factor(cond,
                                      levels = unique(acq_approx$cond)))

ggplot(data = acq_approx) +
  geom_line(aes(x = t, y = g, color = cond))

ggplot(data = acq_approx) +
  geom_line(aes(x = t, y = g)) +
  facet_wrap(~cond)

# find boundary condition where epistemic value decay monotnically
first_order_differential <- function(onset, offset, g) {
  dg <- diff(g)
  ret_dg <- dg[onset:offset]
  t <- seq_len(offset - onset + 1) + onset
  data.frame(t = t, dg = ret_dg)
}

onset <- n1 - 10
offset <- n1 + 10

dg_ext_approx <- ext_approx %>%
  split(., .$cond) %>%
  lapply(., function(d) {
    dg <- first_order_differential(onset, offset, d$g)
    dg$cond <- rep(unique(d$cond), nrow(dg))
    dg
}) %>%
  do.call(rbind, .)

dg_acq_approx <- acq_approx %>%
  split(., .$cond) %>%
  lapply(., function(d) {
    dg <- first_order_differential(onset, offset, d$g)
    dg$cond <- rep(unique(d$cond), nrow(dg))
    dg
}) %>%
  do.call(rbind, .)

# found boundary conditions
# 1. - 0.8: burst
# 0.8 - 0.6: resist
# 0.6 - 0.5: constant
# 0.5 - 0.1: promote
# 0.1 - 0.: constant
ggplot(data = dg_ext_approx) +
  geom_line(aes(x = t, y = dg)) +
  geom_hline(yintercept = 0., color = "red") +
  facet_wrap(~cond, scales = "free")

ggplot(data = dg_acq_approx) +
  geom_line(aes(x = t, y = dg)) +
  geom_hline(yintercept = 0., color = "red") +
  facet_wrap(~cond, scales = "free")

# NOTE: the boundary condition depends on `lambda`
# so the boundary condition found here is not universal

# extended model prometing to burst which has asymmetric leraning rate

# expected value of binominal distribution with learning rate
expected_reward_ex <- function(p, n, lr) {
  lr * p * n
}

expected_extinction_ex <- function(p, n, lr) {
  lr * (1 - p) * n
}

# variation of a between n and n + 1 with learning rate
# lr*p(n + 1) - lr*pn
# = lr*pn + lr*p - lr*pn
# = lr*p
delta_alpha_ex <- function(p, lr) {
  lr * p
}

# variation of b between n and n + 1 with learning rate
# lr*q(n + 1) - lr*qn (where q = 1 - p)
# = lr*qn + lr*q - lr*qn
# = lr*q = lr*(1 - p)
delta_beta_ex <- function(p, lr) {
  lr * (1 - p)
}


approx_expected_free_energy_ex <- function(p1, n1, lra, p2, n2, lrb, lambda) {
  alpha <- exp(2 * lambda)
  alpha_t <- 1.
  beta_t <- 1.

  component_1 <- seq_len(n1) %>%
    lapply(., function(i) {
      alpha_t <<- alpha_t + delta_alpha_ex(p1, lra)
      beta_t <<- beta_t + delta_beta_ex(p1, lrb)
      g <- calc_expected_free_energy(alpha, alpha_t, beta_t)
      data.frame(p = p1, t = i, g = g)
  }) %>%
    do.call(rbind, .)

  component_2 <- seq_len(n2) %>%
    lapply(., function(i) {
      alpha_t <<- alpha_t + delta_alpha_ex(p2, lra)
      beta_t <<- beta_t + delta_beta_ex(p2, lrb)
      g <- calc_expected_free_energy(alpha, alpha_t, beta_t)
      data.frame(p = p2, t = i + n1, g = g)
  }) %>%
    do.call(rbind, .)

  rbind(component_1, component_2)
}

# fixed across all conditions
# NOTE: extinction bursts are more likely to occur when `lra` > `lrb`
# because predicted reward probability greater than 0.5
lra <- 0.1
lrb <- 0.01

# approximate g under extinction with extended model
ext_approx_ex <- baseline_probs %>%
  lapply(., function(p1) {
    p2 <- 0.
    name <- paste0(p1, "=>", p2)

    res <- approx_expected_free_energy_ex(p1, n1, lra, p2, n2, lrb, lambda)
    res$cond <- rep(name, nrow(res))
    res
}) %>%
  do.call(rbind, .)

ext_approx_ex <- transform(ext_approx_ex,
                        cond = factor(cond, levels = unique(ext_approx_ex$cond)))

ggplot(data = ext_approx_ex) +
  geom_line(aes(x = t, y = g, color = cond))

ggplot(data = ext_approx_ex) +
  geom_line(aes(x = t, y = g)) +
  facet_wrap(~cond)

# find boundary condition where epistemic value decay monotnically
dg_ext_approx_ex <- ext_approx_ex %>%
  split(., .$cond) %>%
  lapply(., function(d) {
    dg <- first_order_differential(onset, offset, d$g)
    dg$cond <- rep(unique(d$cond), nrow(dg))
    dg
}) %>%
  do.call(rbind, .)

# found boundary conditions
# depending on `lra` and `lrb`
# 1. - 0.3: burst
# 0.3 - 0.1: promote
# 0.1 - 0.: constant
ggplot(data = dg_ext_approx_ex) +
  geom_line(aes(x = t, y = dg)) +
  geom_hline(yintercept = 0., color = "red") +
  facet_wrap(~cond, scales = "free")

# NOTE: the boundary condition depends on `lambda`, `lra`, and `lrb`
# so the boundary condition found here is not universal
