################
# Figure 6
################
# extracting proxy probabilities
proxy_genetic = read_csv("genetic_as_proxy.csv")
race_genetic = proxy_genetic[, c("white_prob", "black_prob", "asian_prob")]
colnames(race_genetic) = c("White", "Black", "Asian")
outcome_genetic = proxy_genetic[, c("py1yhat1", "py0yhat1", "py1yhat0", "py0yhat0")]
colnames(proxy_genetic)[4:6] = c("White", "Black", "API")

proxy_medicine = read_csv("medicine_as_proxy.csv")
race_medicine = proxy_medicine[, c("white_prob", "black_prob", "asian_prob")]
colnames(race_medicine) = c("White", "Black", "Asian")
outcome_medicine = proxy_medicine[, c("py1yhat1", "py0yhat1", "py1yhat0", "py0yhat0")]
colnames(proxy_medicine)[4:6] = c("White", "Black", "API")


proxy_medicine_genetic = read_csv("medicine_genetic_as_proxy.csv")
race_medicine_genetic = proxy_medicine_genetic[, c("white_prob", "black_prob", "asian_prob")]
colnames(race_medicine_genetic) = c("White", "Black", "Asian")
outcome_medicine_genetic = proxy_medicine_genetic[, c("py1yhat1", "py0yhat1", "py1yhat0", "py0yhat0")]
colnames(proxy_medicine_genetic)[4:6] = c("White", "Black", "API")
# pool conditional outcome probs py1yhat1, py0yhat1, py1yhat0, py0yhat0 together for each type of proxy
outcome_medicine_compile = unlist((outcome_medicine))
outcome_genetic_compile = unlist((outcome_genetic))
outcome_medicine_genetic_compile = unlist(outcome_medicine_genetic)

outcome = data.frame(medicine = outcome_medicine_compile, genetic = outcome_genetic_compile,
                     both = outcome_medicine_genetic_compile)
rownames(outcome) = NULL
# compute entropy
compute_entropy_four_outcome(outcome_medicine_genetic)
compute_entropy_four_outcome(outcome_genetic)
compute_entropy_four_outcome(outcome_medicine)
compute_entropy_race(proxy_medicine_genetic)
compute_entropy_race(proxy_medicine)
compute_entropy_race(proxy_genetic)
# plotting
plt_genetic = race_genetic %>%
  gather(key = "race", value = "prob") %>%
  ggplot(aes(x = prob, color = race, fill = race)) +
  geom_histogram(position = 'dodge', alpha = 0) +
  ggtitle("Only Genetic") +
  xlab("Race Probabilities (Entropy 0.142)") + ylab("") +
  ylim(c(0, 5000)) +
  theme(plot.title = element_text(hjust = 0.5))
plt_medicine = race_medicine %>%
  gather(key = "race", value = "prob") %>%
  ggplot(aes(x = prob, color = race, fill = race)) +
  geom_histogram(position = 'dodge', alpha = 0) +
  ggtitle("Only Medicine") +
  xlab("Race Probabilities (Entropy 0.379)") + ylab("") +
  ylim(c(0, 5000)) +
  theme(plot.title = element_text(hjust = 0.5))
plt_medicine_genetic = race_medicine_genetic %>%
  gather(key = "race", value = "prob") %>%
  ggplot(aes(x = prob, color = race, fill = race)) +
  geom_histogram(position = 'dodge', alpha = 0) +
  ggtitle("Both Genetic and Medicine") +
  xlab("Race Probabilities (Entropy 0.042)") + ylab("") +
  ylim(c(0, 5000)) +
  theme(plot.title = element_text(hjust = 0.5))
grid_arrange_shared_legend(plt_medicine, plt_genetic,plt_medicine_genetic,
                           ncol = 3, nrow = 1)
plt_outcome_medicine = outcome %>% ggplot(aes(x = medicine)) +
  geom_histogram(position = 'identity', color = 'black', fill = 'white') +
  ggtitle("Only Medicine") +
  xlab("Outcome Probabilities (Entropy 0.903)") + ylab("") + ylim(c(0, 10500)) +
  theme(plot.title = element_text(hjust = 0.5))
plt_outcome_genetic = outcome %>% ggplot(aes(x = genetic)) +
  geom_histogram(position = 'identity', color = 'black', fill = 'white') +
  ggtitle("Only Genetic") +
  xlab("Outcome Probabilities (Entropy 0.647)") + ylab("") +  ylim(c(0, 10500)) +
  theme(plot.title = element_text(hjust = 0.5))
plt_outcome_genetic_medicine = outcome %>% ggplot(aes(x = both)) +
  geom_histogram(position = 'identity', color = 'black', fill = 'white') +
  ggtitle("Both Genetic and Medicine") +
  xlab("Outcome Probabilities (Entropy 0.372)") + ylab("") + ylim(c(0, 10500)) +
  theme(plot.title = element_text(hjust = 0.5))
gridExtra::grid.arrange(plt_outcome_medicine, plt_outcome_genetic, plt_outcome_genetic_medicine, nrow = 1)


######################
#  Figure 7
######################
bound_df_medicine = read_csv("CI_medicine.csv")
bound_df_genetic = read_csv("CI_genetic.csv")
bound_df_medicine_genetic = read_csv("CI_medicine_genetic.csv")
plot_medicine = ggplot(bound_df_medicine) +
  geom_errorbar(aes(x = race, ymin = PI_lower, ymax = PI_upper), width = 0.4) +
  geom_errorbar(aes(x = race, ymin = CI_lower, ymax = PI_lower), linetype = "dashed",  width = 0.4) +
  geom_errorbar(aes(x = race, ymin = PI_upper, ymax = CI_upper), linetype = "dashed", width = 0.4) +
  ggtitle("Medicine Only") +
  xlab("Race") + ylab("TPRD") +
  theme(plot.title = element_text(hjust = 0.5)) +
  geom_hline(yintercept = 0) +
  ylim(-1, 1) +
  geom_point(aes(x = race, y = truth), colour = "red", shape = "*", size =  9)
plot_genetic = ggplot(bound_df_genetic) +
  geom_errorbar(aes(x = race, ymin = PI_lower, ymax = PI_upper), width = 0.4) +
  geom_errorbar(aes(x = race, ymin = CI_lower, ymax = PI_lower), linetype = "dashed",  width = 0.4) +
  geom_errorbar(aes(x = race, ymin = PI_upper, ymax = CI_upper), linetype = "dashed", width = 0.4) +
  ggtitle("Genetic Only") +
  xlab("Race") + ylab("") +
  theme(plot.title = element_text(hjust = 0.5)) +
  geom_hline(yintercept = 0) +
  ylim(-1, 1) +
  geom_point(aes(x = race, y = truth), colour = "red", shape = "*", size =  9)
plot_medicine_genetic = ggplot(bound_df_medicine_genetic) +
  geom_errorbar(aes(x = race, ymin = PI_lower, ymax = PI_upper), width = 0.4) +
  geom_errorbar(aes(x = race, ymin = CI_lower, ymax = PI_lower), linetype = "dashed",  width = 0.4) +
  geom_errorbar(aes(x = race, ymin = PI_upper, ymax = CI_upper), linetype = "dashed", width = 0.4) +
  ggtitle("Both Genetic and Medicine") +
  xlab("Race") + ylab("") +
  theme(plot.title = element_text(hjust = 0.5)) +
  geom_hline(yintercept = 0) +
  ylim(-1, 1) +
  geom_point(aes(x = race, y = truth), colour = "red", shape = "*", size =  9)
gridExtra::grid.arrange(plot_medicine, plot_genetic, plot_medicine_genetic, nrow = 1)






