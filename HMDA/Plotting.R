library(lemon)
library(tidyverse)
source("HMDA_helper.R")

##################
# Figure 3
##################
proxy_county = read_csv("small_proxy_county.csv")
race_county = extract_race_prob(proxy_county)
outcome_county = proxy_county$yhat1

proxy_income = read_csv("small_proxy_income.csv")
race_income = extract_race_prob(proxy_income)
outcome_income = proxy_income$yhat1

proxy_both = read_csv("small_proxy_county_income.csv")
colnames(proxy_both)[1] = "yhat1"
race_both = extract_race_prob(proxy_both)
outcome_both = proxy_both$yhat1

outcome_prob = data.frame(county = outcome_county, income = outcome_income,
                          both = outcome_both)
breaks = seq(from = -0.1, to = 1.1, length = 30)

cat("race prob. entropy with only income", compute_entropy_race(proxy_income), "\n")
cat("race prob. entropy with only geolocation", compute_entropy_race(proxy_county), "\n")
cat("race prob. entropy with both geolocation and income", compute_entropy_race(proxy_both), "\n")
cat("outcome prob. entropy with only income",compute_entropy_binary_outcome(proxy_income), "\n")
cat("outcome prob. entropy with only geolocation",compute_entropy_binary_outcome(proxy_county), "\n")
cat("outcome prob. entropy with both geolocation and income", compute_entropy_binary_outcome(proxy_both), "\n")


plt_outcome_county = outcome_prob %>%
  ggplot(aes(x = county)) + geom_histogram(position = 'identity', color = 'black', fill = 'white', breaks = breaks) +
  ggtitle("Only Geolocation") +
  xlab("Outcome Probabilities (Entropy 0.493)") + ylab("") + xlim(c(-0.1, 1.1)) + ylim(c(0, 9000)) +
  theme(plot.title = element_text(hjust = 0.5))
plt_outcome_income = outcome_prob %>%
  ggplot(aes(x = income)) + geom_histogram(position = 'identity', color = 'black', fill = 'white', breaks = breaks) +
  ggtitle("Only Income") +
  xlab("Outcome Probabilities (Entropy 0.496)") + ylab("") +
  xlim(c(-0.1, 1.1)) + ylim(c(0, 9000)) +
  theme(plot.title = element_text(hjust = 0.5))
plt_outcome_both = outcome_prob %>%
  ggplot(aes(x = both)) + geom_histogram(position = 'identity', color = 'black', fill = 'white', breaks = breaks) +
  ggtitle("Both Income and Geolocation") +
  xlab("Outcome Probabilities (Entropy 0.488)") + ylab("") + xlim(c(-0.1, 1.1)) + ylim(c(0, 9000)) +
  theme(plot.title = element_text(hjust = 0.5))

plt_race_county = race_county %>% ggplot(aes(x = prob, color = race, fill = race)) +
  geom_histogram(position = 'dodge', alpha = 0, breaks = breaks) +
  ggtitle("Only Geolocation") +
  xlab("Race Probabilities (Entropy 0.374)") + ylab("") +  xlim(c(-0.1, 1.1)) + ylim(c(0, 15000)) +
  theme(plot.title = element_text(hjust = 0.5))
plt_race_income = race_income %>% ggplot(aes(x = prob, color = race, fill = race)) +
  geom_histogram(position = 'dodge', alpha = 0, breaks = breaks) +
  ggtitle("Only Income") +
  xlab("Race Probabilities (Entropy 0.464)") + ylab("") +  xlim(c(-0.1, 1.1)) + ylim(c(0, 15000)) +
  theme(plot.title = element_text(hjust = 0.5))
plt_race_both = race_both %>% ggplot(aes(x = prob, color = race, fill = race)) +
  geom_histogram(position = 'dodge', alpha = 0, breaks = breaks) +
  ggtitle("Both Income and Geolocation") +
  xlab("Race Probabilities (Entropy 0.366)") + ylab("") + xlim(c(-0.1, 1.1)) + ylim(c(0, 15000)) +
  theme(plot.title = element_text(hjust = 0.5))

grid_arrange_shared_legend(plt_race_income, plt_race_county,plt_race_both,
                           ncol = 3, nrow = 1)
gridExtra::grid.arrange(plt_outcome_income, plt_outcome_county, plt_outcome_both, nrow = 1)


###################
# Figure 4
###################
bound_df_geolocation = read_csv("CI_geolocation.csv")
bound_df_income = read_csv("CI_income.csv")
bound_df_both = read_csv("CI_income_geolocation.csv")

plot_county = ggplot(bound_df_geolocation) +
  geom_errorbar(aes(x = race, ymin = PI_lower, ymax = PI_upper), width = 0.4) +
  geom_errorbar(aes(x = race, ymin = CI_lower, ymax = PI_lower), linetype = "dashed",  width = 0.4) +
  geom_errorbar(aes(x = race, ymin = PI_upper, ymax = CI_upper), linetype = "dashed", width = 0.4) +  ggtitle("Only Geolocation") +
  xlab("Race") + ylab("") +
  theme(plot.title = element_text(hjust = 0.5)) +
  ylim(-1, 1) +
  geom_point(aes(x = race, y = truth), colour = "red", shape = "*", size =  9)+
  geom_hline(yintercept = 0)

plot_income = ggplot(bound_df_income) +
  geom_errorbar(aes(x = race, ymin = PI_lower, ymax = PI_upper), width = 0.4) +
  geom_errorbar(aes(x = race, ymin = CI_lower, ymax = PI_lower), linetype = "dashed",  width = 0.4) +
  geom_errorbar(aes(x = race, ymin = PI_upper, ymax = CI_upper), linetype = "dashed", width = 0.4) +  ggtitle("Only Income") +
  xlab("Race") + ylab("Demographic Disparity") +
  theme(plot.title = element_text(hjust = 0.5)) +
  ylim(-1, 1) +
  geom_point(aes(x = race, y = truth), colour = "red", shape = "*", size =  9) +
  geom_hline(yintercept = 0)

plot_county_income = ggplot(bound_df_both) +
  geom_errorbar(aes(x = race, ymin = PI_lower, ymax = PI_upper), width = 0.4) +
  geom_errorbar(aes(x = race, ymin = CI_lower, ymax = PI_lower), linetype = "dashed",  width = 0.4) +
  geom_errorbar(aes(x = race, ymin = PI_upper, ymax = CI_upper), linetype = "dashed", width = 0.4) +
  ggtitle("Both Income and Geolocation") +
  xlab("Race") + ylab("") +
  theme(plot.title = element_text(hjust = 0.5)) +
  ylim(-1, 1) +
  geom_point(aes(x = race, y = truth), colour = "red", shape = "*", size =  9) +
  geom_hline(yintercept = 0)

gridExtra::grid.arrange(plot_income, plot_county, plot_county_income, nrow = 1)

