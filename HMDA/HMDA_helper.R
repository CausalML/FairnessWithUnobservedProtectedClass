####################
#  Cleaning data
####################
get_obs_state_missing <- function(data){
  is.na(data$state_code)
}

get_obs_county_missing <- function(data){
  is.na(data$county_name)
}
remove_obs <- function(data, fun_obs){
  obs = fun_obs(data)
  filter(data, !obs)
}
make_race <- function(data){
  data$race = ""

  data$race[(data$applicant_ethnicity_name == "Not Hispanic or Latino") &
              ((data$applicant_race_name_1 == "Asian") | (data$applicant_race_name_1 == "Native Hawaiian or Other Pacific Islander")) &
              (is.na(data$applicant_race_name_2))] = "API"

  data$race[(data$applicant_ethnicity_name == "Not Hispanic or Latino") &
              (data$applicant_race_name_1 == "Black or African American") &
              (is.na(data$applicant_race_name_2))] = "Black"

  data$race[(data$applicant_ethnicity_name == "Not Hispanic or Latino") &
              (data$applicant_race_name_1 == "White") &
              (is.na(data$applicant_race_name_2))] = "White"

  select(data, -one_of(c("applicant_ethnicity_name", "applicant_race_name_1", "applicant_race_name_2"
                         )))
}
get_obs_race_empty <- function(data){
  data$race == ""
}


##############
# Figure 3
##############
extract_race_prob <- function(data){
  data %>% select(White, Black, API) %>%
    gather(key = "race", value = "prob")
}
compute_entropy_race <- function(data){
  race_temp = data[, c("White", "Black", "API")]

  temp_white = xlogx(race_temp$White)
  temp_API = xlogx(race_temp$API)
  temp_Black = xlogx(race_temp$Black)
  -mean(temp_white + temp_API + temp_Black, na.rm =T)
}
compute_entropy_binary_outcome <- function(data){
  outcome_temp = data.frame(yhat1 = data$yhat1, yhat0 = 1 - data$yhat1)
  -mean(xlogx(outcome_temp$yhat1) + xlogx(outcome_temp$yhat0))
}
compute_entropy_four_outcome <- function(outcome){
  -mean(xlogx(outcome$py1yhat1) + xlogx(outcome$py0yhat1) + xlogx(outcome$py1yhat0) + xlogx(outcome$py0yhat0))
}
xlogx <- function(x){
  ifelse(x == 0, 0, x*log(x))
}

#################
# Confidence interval for HMDA
#################
compute_true_dd <- function(dataset, race){
  # compute pairwise disparity w.r.t true racce
  colMeans(dataset[dataset$race == race, "outcome"]) - colMeans(dataset[dataset$race != race, "outcome"])
}
compute_mu_L <- function(race_prob, outcome_prob, outcome, raw_prob_race){
  # compute \hat mu(alpha, w^L) with P(A = alpha|Z) given by race_prob
  1/raw_prob_race * mean(ifelse(outcome_prob + race_prob - 1 >= 0, 1, 0) * (outcome + race_prob - 1))
}
compute_mu_U <- function(race_prob, outcome_prob, outcome, raw_prob_race){
  # compute \hat mu(alpha, w^U) with P(A = alpha|Z) given by race_prob
  1/raw_prob_race * mean(ifelse(outcome_prob - race_prob <= 0, 1, 0) * (outcome -race_prob) + race_prob)
}
compute_var_L <- function(pa, pb, eta_1, eta_a, eta_b, outcome, lb){
  # compute \hat V_L for \hat mu(a, w^L) - \hat mu(b, w^U)
  ind1 = ifelse(eta_1 + eta_a - 1 >= 0, 1, 0)
  ind2 = ifelse(eta_1 - eta_b <= 0, 1, 0)
  mean((ind1 * (outcome + eta_a - 1)/pa - (ind2 * (outcome - eta_b) + eta_b)/pb - lb)^2)
}
compute_var_U <- function(pa, pb, eta_1, eta_a, eta_b, outcome, ub){
  # compute \hat V_U for \hat mu(a, w^U) - \hat mu(b, w^L)
  ind1 = ifelse(eta_1 - eta_a <= 0, 1, 0)
  ind2 = ifelse(eta_1 + eta_b - 1 >= 0, 1, 0)
  mean(((ind1 * (outcome - eta_a) + eta_a)/pa - ind2 * (outcome + eta_b - 1)/pb - ub)^2)
}
