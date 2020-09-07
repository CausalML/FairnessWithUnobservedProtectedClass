library(tidyverse)
source("HMDA_helper.R")

#################
# geolocation as proxy
#################
proxy_county = read_csv("small_proxy_county.csv")
# estimate the total race fraction P(A = alpha) for alpha = White, Black, API
raw_race_prob = table(proxy_county$race)/nrow(proxy_county)
# estimate P(\hat Y = 1 | Z) for Z = geolocation and merge it with the whole data
new_yhat1_est = proxy_county %>%
  group_by(county_id) %>%
  summarise(n = n(), nyhat1 = sum(outcome == 1)) %>%
  mutate(
    yhat1b = nyhat1/n
  ) %>%
  select(county_id, yhat1b)
proxy_county = proxy_county %>%
  left_join(new_yhat1_est, by = c("county_id"))
# initialize a dataframe for the results
bound_df = data.frame(PI_lower = rep(0, 3), PI_upper = rep(0, 3),
                      CI_lower = rep(0, 3), CI_upper = rep(0, 3),
                      truth = rep(0, 3),
                      race = (c("White-vs-Rest", "Black-vs-Rest", "API-vs-Rest")))

# white versus rest
mu_L_white = compute_mu_L(proxy_county$White, proxy_county$yhat1, proxy_county$outcome,  raw_race_prob["White"])
    # \hat mu(white, w^L)
mu_U_other = compute_mu_U(1 - proxy_county$White, proxy_county$yhat1, proxy_county$outcome,  1 - raw_race_prob["White"])
    # \hat mu(not white, w^U)
mu_U_white = compute_mu_U(proxy_county$White, proxy_county$yhat1, proxy_county$outcome,  raw_race_prob["White"])
    # \hat mu(white, w^U)
mu_L_other = compute_mu_L(1 - proxy_county$White, proxy_county$yhat1, proxy_county$outcome,  1 - raw_race_prob["White"])
    # \hat mu(not white, w^L)
var_L = compute_var_L(raw_race_prob["White"], 1 - raw_race_prob["White"],
                      proxy_county$yhat1, proxy_county$White,  1 - proxy_county$White,
                      proxy_county$outcome, mu_L_white - mu_U_other)
    # \hat V_L for \hat mu(white, w^L) - \hat mu(not white, w^U)
var_U = compute_var_U(raw_race_prob["White"], 1 - raw_race_prob["White"],
                      proxy_county$yhat1, proxy_county$White,  1 - proxy_county$White,
                      proxy_county$outcome, mu_U_white - mu_L_other)
    # \hat V_U for \hat mu(white, w^U) - \hat mu(not white, w^L)
bound_df$PI_lower[bound_df$race == "White-vs-Rest"] = mu_L_white - mu_U_other
bound_df$PI_upper[bound_df$race == "White-vs-Rest"] = mu_U_white - mu_L_other
bound_df$CI_lower[bound_df$race == "White-vs-Rest"] =
  mu_L_white - mu_U_other - qnorm(1 - 0.025) * sqrt(var_L)/sqrt(nrow(proxy_county))
bound_df$CI_upper[bound_df$race == "White-vs-Rest"] =
  mu_U_white - mu_L_other + qnorm(1 - 0.025) * sqrt(var_U)/sqrt(nrow(proxy_county))
# Black versus rest
mu_L_Black = compute_mu_L(proxy_county$Black, proxy_county$yhat1, proxy_county$outcome,  raw_race_prob["Black"])
mu_U_other = compute_mu_U(1 - proxy_county$Black, proxy_county$yhat1, proxy_county$outcome,  1 - raw_race_prob["Black"])
mu_U_Black = compute_mu_U(proxy_county$Black, proxy_county$yhat1, proxy_county$outcome,  raw_race_prob["Black"])
mu_L_other = compute_mu_L(1 - proxy_county$Black, proxy_county$yhat1, proxy_county$outcome,  1 - raw_race_prob["Black"])
var_L = compute_var_L(raw_race_prob["Black"], 1 - raw_race_prob["Black"],
                      proxy_county$yhat1, proxy_county$Black,  1 - proxy_county$Black,
                      proxy_county$outcome, mu_L_Black - mu_U_other)
var_U = compute_var_U(raw_race_prob["Black"], 1 - raw_race_prob["Black"],
                      proxy_county$yhat1, proxy_county$Black,  1 - proxy_county$Black,
                      proxy_county$outcome, mu_U_Black - mu_L_other)
bound_df$PI_lower[bound_df$race == "Black-vs-Rest"] = mu_L_Black - mu_U_other
bound_df$PI_upper[bound_df$race == "Black-vs-Rest"] = mu_U_Black - mu_L_other
bound_df$CI_lower[bound_df$race == "Black-vs-Rest"] =
  mu_L_Black - mu_U_other - qnorm(1 - 0.025) * sqrt(var_L)/sqrt(nrow(proxy_county))
bound_df$CI_upper[bound_df$race == "Black-vs-Rest"] =
  mu_U_Black - mu_L_other + qnorm(1 - 0.025) * sqrt(var_U)/sqrt(nrow(proxy_county))
# API versus the rest
mu_L_API = compute_mu_L(proxy_county$API, proxy_county$yhat1, proxy_county$outcome,  raw_race_prob["API"])
mu_U_other = compute_mu_U(1 - proxy_county$API, proxy_county$yhat1, proxy_county$outcome,  1 - raw_race_prob["API"])
mu_U_API = compute_mu_U(proxy_county$API, proxy_county$yhat1, proxy_county$outcome,  raw_race_prob["API"])
mu_L_other = compute_mu_L(1 - proxy_county$API, proxy_county$yhat1, proxy_county$outcome,  1 - raw_race_prob["API"])
var_L = compute_var_L(raw_race_prob["API"], 1 - raw_race_prob["API"],
                      proxy_county$yhat1, proxy_county$API,  1 - proxy_county$API,
                      proxy_county$outcome, mu_L_API - mu_U_other)
var_U = compute_var_U(raw_race_prob["API"], 1 - raw_race_prob["API"],
                      proxy_county$yhat1, proxy_county$API,  1 - proxy_county$API,
                      proxy_county$outcome, mu_U_API - mu_L_other)
bound_df$PI_lower[bound_df$race == "API-vs-Rest"] = mu_L_API - mu_U_other
bound_df$PI_upper[bound_df$race == "API-vs-Rest"] = mu_U_API - mu_L_other
bound_df$CI_lower[bound_df$race == "API-vs-Rest"] =
  mu_L_API - mu_U_other - qnorm(1 - 0.025) * sqrt(var_L)/sqrt(nrow(proxy_county))
bound_df$CI_upper[bound_df$race == "API-vs-Rest"] =
  mu_U_API - mu_L_other + qnorm(1 - 0.025) * sqrt(var_U)/sqrt(nrow(proxy_county))
# disparity using true race
bound_df[bound_df$race == "White-vs-Rest", "truth"] =
  compute_true_dd(proxy_county, "White")
bound_df[bound_df$race == "Black-vs-Rest", "truth"] =
  compute_true_dd(proxy_county, "Black")
bound_df[bound_df$race == "API-vs-Rest", "truth"] =
  compute_true_dd(proxy_county, "API")
write_csv(bound_df, "CI_geolocation.csv")


####################
#  Income as proxy
####################
proxy_income = read_csv("small_proxy_income.csv")
raw_race_prob = table(proxy_income$race)/nrow(proxy_income)
bound_df = data.frame(PI_lower = rep(0, 3), PI_upper = rep(0, 3),
                      CI_lower = rep(0, 3), CI_upper = rep(0, 3),
                      truth = rep(0, 3),
                      race = (c("White-vs-Rest", "Black-vs-Rest", "API-vs-Rest")))
# white vs other
mu_L_white = compute_mu_L(proxy_income$White, proxy_income$yhat1, proxy_income$outcome,  raw_race_prob["White"])
mu_U_other = compute_mu_U(1 - proxy_income$White, proxy_income$yhat1, proxy_income$outcome,  1 - raw_race_prob["White"])
mu_U_white = compute_mu_U(proxy_income$White, proxy_income$yhat1, proxy_income$outcome,  raw_race_prob["White"])
mu_L_other = compute_mu_L(1 - proxy_income$White, proxy_income$yhat1, proxy_income$outcome,  1 - raw_race_prob["White"])
var_L = compute_var_L(raw_race_prob["White"], 1 - raw_race_prob["White"],
                      proxy_income$yhat1, proxy_income$White,  1 - proxy_income$White,
                      proxy_income$outcome, mu_L_white - mu_U_other)
var_U = compute_var_U(raw_race_prob["White"], 1 - raw_race_prob["White"],
                      proxy_income$yhat1, proxy_income$White,  1 - proxy_income$White,
                      proxy_income$outcome, mu_U_white - mu_L_other)
bound_df$PI_lower[bound_df$race == "White-vs-Rest"] = mu_L_white - mu_U_other
bound_df$PI_upper[bound_df$race == "White-vs-Rest"] = mu_U_white - mu_L_other
bound_df$CI_lower[bound_df$race == "White-vs-Rest"] =
  mu_L_white - mu_U_other - qnorm(1 - 0.025) * sqrt(var_L)/sqrt(nrow(proxy_income))
bound_df$CI_upper[bound_df$race == "White-vs-Rest"] =
  mu_U_white - mu_L_other + qnorm(1 - 0.025) * sqrt(var_U)/sqrt(nrow(proxy_income))
# Black vs rest
mu_L_Black = compute_mu_L(proxy_income$Black, proxy_income$yhat1, proxy_income$outcome,  raw_race_prob["Black"])
mu_U_other = compute_mu_U(1 - proxy_income$Black, proxy_income$yhat1, proxy_income$outcome,  1 - raw_race_prob["Black"])
mu_U_Black = compute_mu_U(proxy_income$Black, proxy_income$yhat1, proxy_income$outcome,  raw_race_prob["Black"])
mu_L_other = compute_mu_L(1 - proxy_income$Black, proxy_income$yhat1, proxy_income$outcome,  1 - raw_race_prob["Black"])
var_L = compute_var_L(raw_race_prob["Black"], 1 - raw_race_prob["Black"],
                      proxy_income$yhat1, proxy_income$Black,  1 - proxy_income$Black,
                      proxy_income$outcome, mu_L_Black - mu_U_other)
var_U = compute_var_U(raw_race_prob["Black"], 1 - raw_race_prob["Black"],
                      proxy_income$yhat1, proxy_income$Black,  1 - proxy_income$Black,
                      proxy_income$outcome, mu_U_Black - mu_L_other)
bound_df$PI_lower[bound_df$race == "Black-vs-Rest"] = mu_L_Black - mu_U_other
bound_df$PI_upper[bound_df$race == "Black-vs-Rest"] = mu_U_Black - mu_L_other
bound_df$CI_lower[bound_df$race == "Black-vs-Rest"] =
  mu_L_Black - mu_U_other - qnorm(1 - 0.025) * sqrt(var_L)/sqrt(nrow(proxy_income))
bound_df$CI_upper[bound_df$race == "Black-vs-Rest"] =
  mu_U_Black - mu_L_other + qnorm(1 - 0.025) * sqrt(var_U)/sqrt(nrow(proxy_income))
# API versus the rest
mu_L_API = compute_mu_L(proxy_income$API, proxy_income$yhat1, proxy_income$outcome,  raw_race_prob["API"])
mu_U_other = compute_mu_U(1 - proxy_income$API, proxy_income$yhat1, proxy_income$outcome,  1 - raw_race_prob["API"])
mu_U_API = compute_mu_U(proxy_income$API, proxy_income$yhat1, proxy_income$outcome,  raw_race_prob["API"])
mu_L_other = compute_mu_L(1 - proxy_income$API, proxy_income$yhat1, proxy_income$outcome,  1 - raw_race_prob["API"])
var_L = compute_var_L(raw_race_prob["API"], 1 - raw_race_prob["API"],
                      proxy_income$yhat1, proxy_income$API,  1 - proxy_income$API,
                      proxy_income$outcome, mu_L_API - mu_U_other)
var_U = compute_var_U(raw_race_prob["API"], 1 - raw_race_prob["API"],
                      proxy_income$yhat1, proxy_income$API,  1 - proxy_income$API,
                      proxy_income$outcome, mu_U_API - mu_L_other)
bound_df$PI_lower[bound_df$race == "API-vs-Rest"] = mu_L_API - mu_U_other
bound_df$PI_upper[bound_df$race == "API-vs-Rest"] = mu_U_API - mu_L_other
bound_df$CI_lower[bound_df$race == "API-vs-Rest"] =
  mu_L_API - mu_U_other - qnorm(1 - 0.025) * sqrt(var_L)/sqrt(nrow(proxy_income))
bound_df$CI_upper[bound_df$race == "API-vs-Rest"] =
  mu_U_API - mu_L_other + qnorm(1 - 0.025) * sqrt(var_U)/sqrt(nrow(proxy_income))

bound_df[bound_df$race == "White-vs-Rest", "truth"] =
  compute_true_dd(proxy_income, "White")
bound_df[bound_df$race == "Black-vs-Rest", "truth"] =
  compute_true_dd(proxy_income, "Black")
bound_df[bound_df$race == "API-vs-Rest", "truth"] =
  compute_true_dd(proxy_income, "API")
write_csv(bound_df, "CI_income.csv")

#####################
#  Both Income and Geolocation as proxy:
#####################
proxy_county_income = read_csv("small_proxy_county_income.csv")
raw_race_prob = table(proxy_county_income$race)/nrow(proxy_county_income)
bound_df = data.frame(PI_lower = rep(0, 3), PI_upper = rep(0, 3),
                      CI_lower = rep(0, 3), CI_upper = rep(0, 3),
                      truth = rep(0, 3),
                      race = (c("White-vs-Rest", "Black-vs-Rest", "API-vs-Rest")))
# white vs other
mu_L_white = compute_mu_L(proxy_county_income$White, proxy_county_income$yhat1, proxy_county_income$outcome,  raw_race_prob["White"])
mu_U_other = compute_mu_U(1 - proxy_county_income$White, proxy_county_income$yhat1, proxy_county_income$outcome,  1 - raw_race_prob["White"])
mu_U_white = compute_mu_U(proxy_county_income$White, proxy_county_income$yhat1, proxy_county_income$outcome,  raw_race_prob["White"])
mu_L_other = compute_mu_L(1 - proxy_county_income$White, proxy_county_income$yhat1, proxy_county_income$outcome,  1 - raw_race_prob["White"])
var_L = compute_var_L(raw_race_prob["White"], 1 - raw_race_prob["White"],
                      proxy_county_income$yhat1, proxy_county_income$White,  1 - proxy_county_income$White,
                      proxy_county_income$outcome, mu_L_white - mu_U_other)
var_U = compute_var_U(raw_race_prob["White"], 1 - raw_race_prob["White"],
                      proxy_county_income$yhat1, proxy_county_income$White,  1 - proxy_county_income$White,
                      proxy_county_income$outcome, mu_U_white - mu_L_other)
bound_df$PI_lower[bound_df$race == "White-vs-Rest"] = mu_L_white - mu_U_other
bound_df$PI_upper[bound_df$race == "White-vs-Rest"] = mu_U_white - mu_L_other
bound_df$CI_lower[bound_df$race == "White-vs-Rest"] =
  mu_L_white - mu_U_other - qnorm(1 - 0.025) * sqrt(var_L)/sqrt(nrow(proxy_county_income))
bound_df$CI_upper[bound_df$race == "White-vs-Rest"] =
  mu_U_white - mu_L_other + qnorm(1 - 0.025) * sqrt(var_U)/sqrt(nrow(proxy_county_income))
# Black vs rest
mu_L_Black = compute_mu_L(proxy_county_income$Black, proxy_county_income$yhat1, proxy_county_income$outcome,  raw_race_prob["Black"])
mu_U_other = compute_mu_U(1 - proxy_county_income$Black, proxy_county_income$yhat1, proxy_county_income$outcome,  1 - raw_race_prob["Black"])
mu_U_Black = compute_mu_U(proxy_county_income$Black, proxy_county_income$yhat1, proxy_county_income$outcome,  raw_race_prob["Black"])
mu_L_other = compute_mu_L(1 - proxy_county_income$Black, proxy_county_income$yhat1, proxy_county_income$outcome,  1 - raw_race_prob["Black"])
var_L = compute_var_L(raw_race_prob["Black"], 1 - raw_race_prob["Black"],
                      proxy_county_income$yhat1, proxy_county_income$Black,  1 - proxy_county_income$Black,
                      proxy_county_income$outcome, mu_L_Black - mu_U_other)
var_U = compute_var_U(raw_race_prob["Black"], 1 - raw_race_prob["Black"],
                      proxy_county_income$yhat1, proxy_county_income$Black,  1 - proxy_county_income$Black,
                      proxy_county_income$outcome, mu_U_Black - mu_L_other)
bound_df$PI_lower[bound_df$race == "Black-vs-Rest"] = mu_L_Black - mu_U_other
bound_df$PI_upper[bound_df$race == "Black-vs-Rest"] = mu_U_Black - mu_L_other
bound_df$CI_lower[bound_df$race == "Black-vs-Rest"] =
  mu_L_Black - mu_U_other - qnorm(1 - 0.025) * sqrt(var_L)/sqrt(nrow(proxy_county_income))
bound_df$CI_upper[bound_df$race == "Black-vs-Rest"] =
  mu_U_Black - mu_L_other + qnorm(1 - 0.025) * sqrt(var_U)/sqrt(nrow(proxy_county_income))
# API versus the rest
mu_L_API = compute_mu_L(proxy_county_income$API, proxy_county_income$yhat1, proxy_county_income$outcome,  raw_race_prob["API"])
mu_U_other = compute_mu_U(1 - proxy_county_income$API, proxy_county_income$yhat1, proxy_county_income$outcome,  1 - raw_race_prob["API"])
mu_U_API = compute_mu_U(proxy_county_income$API, proxy_county_income$yhat1, proxy_county_income$outcome,  raw_race_prob["API"])
mu_L_other = compute_mu_L(1 - proxy_county_income$API, proxy_county_income$yhat1, proxy_county_income$outcome,  1 - raw_race_prob["API"])
var_L = compute_var_L(raw_race_prob["API"], 1 - raw_race_prob["API"],
                      proxy_county_income$yhat1, proxy_county_income$API,  1 - proxy_county_income$API,
                      proxy_county_income$outcome, mu_L_API - mu_U_other)
var_U = compute_var_U(raw_race_prob["API"], 1 - raw_race_prob["API"],
                      proxy_county_income$yhat1, proxy_county_income$API,  1 - proxy_county_income$API,
                      proxy_county_income$outcome, mu_U_API - mu_L_other)
bound_df$PI_lower[bound_df$race == "API-vs-Rest"] = mu_L_API - mu_U_other
bound_df$PI_upper[bound_df$race == "API-vs-Rest"] = mu_U_API - mu_L_other
bound_df$CI_lower[bound_df$race == "API-vs-Rest"] =
  mu_L_API - mu_U_other - qnorm(1 - 0.025) * sqrt(var_L)/sqrt(nrow(proxy_county_income))
bound_df$CI_upper[bound_df$race == "API-vs-Rest"] =
  mu_U_API - mu_L_other + qnorm(1 - 0.025) * sqrt(var_U)/sqrt(nrow(proxy_county_income))
bound_df[bound_df$race == "White-vs-Rest", "truth"] =
  compute_true_dd(proxy_county_income, "White")
bound_df[bound_df$race == "Black-vs-Rest", "truth"] =
  compute_true_dd(proxy_county_income, "Black")
bound_df[bound_df$race == "API-vs-Rest", "truth"] =
  compute_true_dd(proxy_county_income, "API")
write_csv(bound_df, "CI_income_geolocation.csv")
