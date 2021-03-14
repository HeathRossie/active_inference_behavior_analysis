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

# find boundary conditions under given parameters in the original model

expected_alpha <- function(p, n) {
  p * n
}

expected_beta <- function(p, n) {
  (1 - p) * n
}

burst_intensity <- function(lambda, p, n) {
  alpha_t <- 1.
  beta_t <- 1.
  alpha <- exp(2 * lambda)
  alpha_t <- alpha_t + expected_alpha(p, n)
  beta_t <- beta_t + expected_beta(p, n)
  g0 <- calc_expected_free_energy(alpha, alpha_t, beta_t)
  alpha_t <- alpha_t + expected_alpha(0., 1)
  beta_t <- beta_t + expected_beta(0., 1)
  g1 <- calc_expected_free_energy(alpha, alpha_t, beta_t)
  - (g1 - g0)
}

detect_burst_prob_bound <- function(params, intensity) {
  d <- data.frame(prob = params, intensity = intensity)
  gz <- d %>% filter(intensity > 0)
  gz %>% filter(intensity == min(gz$intensity))
}

find_boundary_conditions <- function(lambda, probs, n) {
  prob_bounds <- lambda %>%
    lapply(., function(lambda) {
      intensity <- burst_intensity(lambda, probs, n)
      bound <- detect_burst_prob_bound(probs, intensity)
      bound
  }) %>%
    do.call(rbind, .)
  prob_bounds$lambda <- lambda
  prob_bounds
}

# parameter space to explore
probs <- seq(0.001, 1., by = 0.001)
lambdas <- seq(0, 1.99, by = 0.025)
n <- seq(100, 500, by = 100)

boundary_prob_per_lambda <- n %>%
  lapply(., function(n) {
    boundary_condtions <- find_boundary_conditions(lambdas, probs, n)
    boundary_condtions$n <- rep(n, nrow(boundary_condtions))
    boundary_condtions
  }) %>%
  do.call(rbind, .)

ggplot(data = boundary_prob_per_lambda) +
  geom_line(aes(x = lambda, y = prob, color = as.factor(n)))

# find boundary conditions of original model under given parameters
# in the extended model which has asymmetric leraning rate

expected_alpha_ex <- function(p, n, lr) {
  lr * p * n
}

expected_beta_ex <- function(p, n, lr) {
  lr * (1 - p) * n
}

# calculate burst intensity with extended model
burst_intensity_ex <- function(lambda, p, n, lra, lrb) {
  alpha_t <- 1.
  beta_t <- 1.
  alpha <- exp(2 * lambda)
  alpha_t <- alpha_t + expected_alpha_ex(p, n, lra)
  beta_t <- beta_t + expected_beta_ex(p, n, lrb)
  g0 <- calc_expected_free_energy(alpha, alpha_t, beta_t)
  alpha_t <- alpha_t + expected_alpha_ex(0., 1, lra)
  beta_t <- beta_t + expected_beta_ex(0., 1, lrb)
  g1 <- calc_expected_free_energy(alpha, alpha_t, beta_t)
  - (g1 - g0)
}

boundary_condtions_ex <- lambdas %>%
  lapply(., function(lambda) {
    intensity <- burst_intensity_ex(lambda, probs, n, lra, lrb)
    bound <- detect_burst_prob_bound(probs, intensity)
}) %>%
  do.call(rbind, .)

find_boundary_conditions_ex <- function(lambda, probs, n, lra, lrb) {
  prob_bounds <- lambda %>%
    lapply(., function(lambda) {
    intensity <- burst_intensity_ex(lambda, probs, n, lra, lrb)
    bound <- detect_burst_prob_bound(probs, intensity)
    bound
  }) %>%
    do.call(rbind, .)

  prob_bounds$lambda <- lambda[seq_len(nrow(prob_bounds))]
  prob_bounds
}

# parameter space to explore
lra <- seq(0.2, 1., by = 0.2)
lrb <- seq(0.2, 0.01, by = 0.02)
params <- expand.grid(lra, lrb, n)
colnames(params) <- c("lra", "lrb", "n")
params$idx <- seq_len(nrow(params))

boundary_prob_per_lambda_ex <- params %>%
  split(., .$idx) %>%
  lapply(., function(p) {
    n <- p$n
    lra <- p$lra
    lrb <- p$lrb
    boundary_condtions <- find_boundary_conditions_ex(lambdas, probs, n, lra, lrb)
    boundary_condtions$n <- rep(n, nrow(boundary_condtions))
    boundary_condtions$lra <- rep(lra, nrow(boundary_condtions))
    boundary_condtions$lrb <- rep(lrb, nrow(boundary_condtions))
    boundary_condtions
  }) %>%
  do.call(rbind, .)

ggplot(data = boundary_prob_per_lambda_ex) +
  geom_line(aes(x = lambda, y = prob, color = as.factor(n))) +
  facet_grid(~lra~lrb)
