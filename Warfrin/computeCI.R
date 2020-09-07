 library(tidyverse)

source("warfarin_helper.R")

#########################
# Only Medicine as proxy
#########################
proxy_medicine = read_csv("medicine_as_proxy.csv")
n = nrow(proxy_medicine)
r = 1/2 # sample size ratio of each dataset over the total sample size
primary_index = sample(1:n, n/2, replace = FALSE)
  # the first n/2 obs correspond to D_{pri} in paper
  # the last n/2 obs correspond to D_{aux} in paper
bound_df_medicine = data.frame(PI_lower = rep(0, 3), PI_upper = rep(0, 3),
                               CI_lower = rep(0, 3), CI_upper = rep(0, 3),
                               truth = rep(0, 3),
                               race = (c("White-vs-Rest", "Black-vs-Rest", "Asian-vs-Rest")))
proxy_medicine = read_csv("medicine_as_proxy.csv")
proxy_medicine = proxy_medicine %>%
  mutate(
    White_prob = white_prob,
    Black_prob = black_prob,
    Asian_prob = asian_prob
  )
###  White vs rest
# estimate the quantities for alpha = white in equation (35)
Wbar_L_11_White = compute_Wbar_L_hyy(primary_index,
                                     proxy_medicine[, "White_prob"], proxy_medicine[, "py1yhat1"],
                                     ifelse(proxy_medicine$race == "White.", 1, 0),
                                     ifelse((proxy_medicine$Yhat == 1) & (proxy_medicine$Y == 1), 1, 0))
Wbar_L_01_White = compute_Wbar_L_hyy(primary_index,
                                     proxy_medicine[, "White_prob"], proxy_medicine[, "py1yhat0"],
                                     ifelse(proxy_medicine$race == "White.", 1, 0),
                                     ifelse((proxy_medicine$Yhat == 0) & (proxy_medicine$Y == 1), 1, 0))
Wbar_U_11_White = compute_Wbar_U_hyy(primary_index,
                                     proxy_medicine[, "White_prob"], proxy_medicine[, "py1yhat1"],
                                     ifelse(proxy_medicine$race == "White.", 1, 0),
                                     ifelse((proxy_medicine$Yhat == 1) & (proxy_medicine$Y == 1), 1, 0))
Wbar_U_01_White = compute_Wbar_U_hyy(primary_index,
                                     proxy_medicine[, "White_prob"], proxy_medicine[, "py1yhat0"],
                                     ifelse(proxy_medicine$race == "White.", 1, 0),
                                     ifelse((proxy_medicine$Yhat == 0) & (proxy_medicine$Y == 1), 1, 0))
# estimate the quantity \hat mu'_{\hat y, y}(alpha; \tilde w^L, \tilde w^U) on page 39 for alpha = white in
mu_White_WL_WU_11 = (truncate(Wbar_L_11_White)/(truncate(Wbar_L_11_White) + truncate(Wbar_U_01_White)))
mu_White_WU_WL_11 = (truncate(Wbar_U_11_White)/(truncate(Wbar_U_11_White) + truncate(Wbar_L_01_White)))
mu_White_WL_WU_01 = truncate(Wbar_L_01_White)/(truncate(Wbar_L_01_White) + truncate(Wbar_U_11_White))
mu_White_WU_WL_01 = truncate(Wbar_U_01_White)/(truncate(Wbar_U_01_White) + truncate(Wbar_L_11_White))
# do the same for alpha = non_white
Wbar_L_11_Other = compute_Wbar_L_hyy(primary_index,
                                     1 - proxy_medicine[, "White_prob"], proxy_medicine[, "py1yhat1"],
                                     1 - ifelse(proxy_medicine$race == "White.", 1, 0),
                                     ifelse((proxy_medicine$Yhat == 1) & (proxy_medicine$Y == 1), 1, 0))
Wbar_L_01_Other = compute_Wbar_L_hyy(primary_index,
                                     1 - proxy_medicine[, "White_prob"], proxy_medicine[, "py1yhat0"],
                                     1- ifelse(proxy_medicine$race == "White.", 1, 0),
                                     ifelse((proxy_medicine$Yhat == 0) & (proxy_medicine$Y == 1), 1, 0))
Wbar_U_11_Other = compute_Wbar_U_hyy(primary_index,
                                     1 - proxy_medicine[, "White_prob"], proxy_medicine[, "py1yhat1"],
                                     1 - ifelse(proxy_medicine$race == "White.", 1, 0),
                                     ifelse((proxy_medicine$Yhat == 1) & (proxy_medicine$Y == 1), 1, 0))
Wbar_U_01_Other = compute_Wbar_U_hyy(primary_index,
                                     1 - proxy_medicine[, "White_prob"], proxy_medicine[, "py1yhat0"],
                                     1 - ifelse(proxy_medicine$race == "White.", 1, 0),
                                     ifelse((proxy_medicine$Yhat == 0) & (proxy_medicine$Y == 1), 1, 0))
mu_Other_WL_WU_11 = (truncate(Wbar_L_11_Other)/(truncate(Wbar_L_11_Other) + truncate(Wbar_U_01_Other)))
mu_Other_WU_WL_11 = (truncate(Wbar_U_11_Other)/(truncate(Wbar_U_11_Other) + truncate(Wbar_L_01_Other)))
mu_Other_WL_WU_01 = truncate(Wbar_L_01_Other)/(truncate(Wbar_L_01_Other) + truncate(Wbar_U_11_Other))
mu_Other_WU_WL_01 = truncate(Wbar_U_01_Other)/(truncate(Wbar_U_01_Other) + truncate(Wbar_L_11_Other))
# compute the estimated partial identification bounds
bound_df_medicine$PI_lower[bound_df_medicine$race == "White-vs-Rest"] = mu_White_WL_WU_11 - mu_Other_WU_WL_11
bound_df_medicine$PI_upper[bound_df_medicine$race == "White-vs-Rest"] = mu_White_WU_WL_11 - mu_Other_WL_WU_11
# compute the estimated variance \hat V_L of the upper bound estimator, namely equation (39)
consta_UL = mu_White_WU_WL_01/(Wbar_L_11_White + Wbar_U_01_White)
consta_LU = mu_White_WL_WU_11/(Wbar_L_11_White + Wbar_U_01_White)
constb_UL = mu_Other_WU_WL_11/(Wbar_U_11_Other + Wbar_L_01_Other)
constb_LU = mu_Other_WL_WU_01/(Wbar_U_11_Other + Wbar_L_01_Other)
VL = compute_VL(r, primary_index,
                consta_UL, consta_LU, constb_UL, constb_LU,
                proxy_medicine[, "White_prob"], 1 - proxy_medicine[, "White_prob"],
                ifelse(proxy_medicine$race == "White.", 1, 0),
                1 - ifelse(proxy_medicine$race == "White.", 1, 0),
                proxy_medicine[, "py1yhat1"], proxy_medicine[, "py1yhat0"],
                ifelse((proxy_medicine$Yhat == 1) & (proxy_medicine$Y == 1), 1, 0),
                ifelse((proxy_medicine$Yhat == 0) & (proxy_medicine$Y == 1), 1, 0))
# estimate \hat V_U
consta_UL = mu_White_WU_WL_11/(Wbar_U_11_White + Wbar_L_01_White)
consta_LU = mu_White_WL_WU_01/(Wbar_U_11_White + Wbar_L_01_White)
constb_UL = mu_Other_WU_WL_01/(Wbar_L_11_Other + Wbar_U_01_Other)
constb_LU = mu_Other_WL_WU_11/(Wbar_L_11_Other + Wbar_U_01_Other)
VU = compute_VU(r, primary_index,
                consta_UL, consta_LU, constb_UL, constb_LU,
                proxy_medicine[, "White_prob"], 1 - proxy_medicine[, "White_prob"],
                ifelse(proxy_medicine$race == "White.", 1, 0),
                1 - ifelse(proxy_medicine$race == "White.", 1, 0),
                proxy_medicine[, "py1yhat1"], proxy_medicine[, "py1yhat0"],
                ifelse((proxy_medicine$Yhat == 1) & (proxy_medicine$Y == 1), 1, 0),
                ifelse((proxy_medicine$Yhat == 0) & (proxy_medicine$Y == 1), 1, 0))
bound_df_medicine$CI_lower[bound_df_medicine$race == "White-vs-Rest"] =
  mu_White_WL_WU_11 - mu_Other_WU_WL_11 - qnorm(1 - 0.025) * sqrt(VL)/sqrt(nrow(proxy_medicine)/2)
bound_df_medicine$CI_upper[bound_df_medicine$race == "White-vs-Rest"] =
  mu_White_WU_WL_11 - mu_Other_WL_WU_11 + qnorm(1 - 0.025) * sqrt(VU)/sqrt(nrow(proxy_medicine)/2)
### Black vs rest
Wbar_L_11_Black = compute_Wbar_L_hyy(primary_index,
                                     proxy_medicine[, "Black_prob"], proxy_medicine[, "py1yhat1"],
                                     ifelse(proxy_medicine$race == "Black.", 1, 0),
                                     ifelse((proxy_medicine$Yhat == 1) & (proxy_medicine$Y == 1), 1, 0))
Wbar_L_01_Black = compute_Wbar_L_hyy(primary_index,
                                     proxy_medicine[, "Black_prob"], proxy_medicine[, "py1yhat0"],
                                     ifelse(proxy_medicine$race == "Black.", 1, 0),
                                     ifelse((proxy_medicine$Yhat == 0) & (proxy_medicine$Y == 1), 1, 0))
Wbar_U_11_Black = compute_Wbar_U_hyy(primary_index,
                                     proxy_medicine[, "Black_prob"], proxy_medicine[, "py1yhat1"],
                                     ifelse(proxy_medicine$race == "Black.", 1, 0),
                                     ifelse((proxy_medicine$Yhat == 1) & (proxy_medicine$Y == 1), 1, 0))
Wbar_U_01_Black = compute_Wbar_U_hyy(primary_index,
                                     proxy_medicine[, "Black_prob"], proxy_medicine[, "py1yhat0"],
                                     ifelse(proxy_medicine$race == "Black.", 1, 0),
                                     ifelse((proxy_medicine$Yhat == 0) & (proxy_medicine$Y == 1), 1, 0))
mu_Black_WL_WU_11 = (truncate(Wbar_L_11_Black)/(truncate(Wbar_L_11_Black) + truncate(Wbar_U_01_Black)))
mu_Black_WU_WL_11 = (truncate(Wbar_U_11_Black)/(truncate(Wbar_U_11_Black) + truncate(Wbar_L_01_Black)))
mu_Black_WL_WU_01 = truncate(Wbar_L_01_Black)/(truncate(Wbar_L_01_Black) + truncate(Wbar_U_11_Black))
mu_Black_WU_WL_01 = truncate(Wbar_U_01_Black)/(truncate(Wbar_U_01_Black) + truncate(Wbar_L_11_Black))

Wbar_L_11_Other = compute_Wbar_L_hyy(primary_index,
                                     1 - proxy_medicine[, "Black_prob"], proxy_medicine[, "py1yhat1"],
                                     1 - ifelse(proxy_medicine$race == "Black.", 1, 0),
                                     ifelse((proxy_medicine$Yhat == 1) & (proxy_medicine$Y == 1), 1, 0))
Wbar_L_01_Other = compute_Wbar_L_hyy(primary_index,
                                     1 - proxy_medicine[, "Black_prob"], proxy_medicine[, "py1yhat0"],
                                     1- ifelse(proxy_medicine$race == "Black.", 1, 0),
                                     ifelse((proxy_medicine$Yhat == 0) & (proxy_medicine$Y == 1), 1, 0))
Wbar_U_11_Other = compute_Wbar_U_hyy(primary_index,
                                     1 - proxy_medicine[, "Black_prob"], proxy_medicine[, "py1yhat1"],
                                     1 - ifelse(proxy_medicine$race == "Black.", 1, 0),
                                     ifelse((proxy_medicine$Yhat == 1) & (proxy_medicine$Y == 1), 1, 0))
Wbar_U_01_Other = compute_Wbar_U_hyy(primary_index,
                                     1 - proxy_medicine[, "Black_prob"], proxy_medicine[, "py1yhat0"],
                                     1 - ifelse(proxy_medicine$race == "Black.", 1, 0),
                                     ifelse((proxy_medicine$Yhat == 0) & (proxy_medicine$Y == 1), 1, 0))
mu_Other_WL_WU_11 = (truncate(Wbar_L_11_Other)/(truncate(Wbar_L_11_Other) + truncate(Wbar_U_01_Other)))
mu_Other_WU_WL_11 = (truncate(Wbar_U_11_Other)/(truncate(Wbar_U_11_Other) + truncate(Wbar_L_01_Other)))
mu_Other_WL_WU_01 = truncate(Wbar_L_01_Other)/(truncate(Wbar_L_01_Other) + truncate(Wbar_U_11_Other))
mu_Other_WU_WL_01 = truncate(Wbar_U_01_Other)/(truncate(Wbar_U_01_Other) + truncate(Wbar_L_11_Other))

bound_df_medicine$PI_lower[bound_df_medicine$race == "Black-vs-Rest"] = mu_Black_WL_WU_11 - mu_Other_WU_WL_11
bound_df_medicine$PI_upper[bound_df_medicine$race == "Black-vs-Rest"] = mu_Black_WU_WL_11 - mu_Other_WL_WU_11

consta_UL = mu_Black_WU_WL_01/(Wbar_L_11_Black + Wbar_U_01_Black)
consta_LU = mu_Black_WL_WU_11/(Wbar_L_11_Black + Wbar_U_01_Black)
constb_UL = mu_Other_WU_WL_11/(Wbar_U_11_Other + Wbar_L_01_Other)
constb_LU = mu_Other_WL_WU_01/(Wbar_U_11_Other + Wbar_L_01_Other)
VL = compute_VL(r, primary_index,
                consta_UL, consta_LU, constb_UL, constb_LU,
                proxy_medicine[, "Black_prob"], 1 - proxy_medicine[, "Black_prob"],
                ifelse(proxy_medicine$race == "Black.", 1, 0),
                1 - ifelse(proxy_medicine$race == "Black.", 1, 0),
                proxy_medicine[, "py1yhat1"], proxy_medicine[, "py1yhat0"],
                ifelse((proxy_medicine$Yhat == 1) & (proxy_medicine$Y == 1), 1, 0),
                ifelse((proxy_medicine$Yhat == 0) & (proxy_medicine$Y == 1), 1, 0))
consta_UL = mu_Black_WU_WL_11/(Wbar_U_11_Black + Wbar_L_01_Black)
consta_LU = mu_Black_WL_WU_01/(Wbar_U_11_Black + Wbar_L_01_Black)
constb_UL = mu_Other_WU_WL_01/(Wbar_L_11_Other + Wbar_U_01_Other)
constb_LU = mu_Other_WL_WU_11/(Wbar_L_11_Other + Wbar_U_01_Other)
VU = compute_VU(r, primary_index,
                consta_UL, consta_LU, constb_UL, constb_LU,
                proxy_medicine[, "Black_prob"], 1 - proxy_medicine[, "Black_prob"],
                ifelse(proxy_medicine$race == "Black.", 1, 0),
                1 - ifelse(proxy_medicine$race == "Black.", 1, 0),
                proxy_medicine[, "py1yhat1"], proxy_medicine[, "py1yhat0"],
                ifelse((proxy_medicine$Yhat == 1) & (proxy_medicine$Y == 1), 1, 0),
                ifelse((proxy_medicine$Yhat == 0) & (proxy_medicine$Y == 1), 1, 0))
bound_df_medicine$CI_lower[bound_df_medicine$race == "Black-vs-Rest"] =
  mu_Black_WL_WU_11 - mu_Other_WU_WL_11 - qnorm(1 - 0.025) * sqrt(VL)/sqrt(nrow(proxy_medicine)/2)
bound_df_medicine$CI_upper[bound_df_medicine$race == "Black-vs-Rest"] =
  mu_Black_WU_WL_11 - mu_Other_WL_WU_11 + qnorm(1 - 0.025) * sqrt(VU)/sqrt(nrow(proxy_medicine)/2)
### Asian vs rest
Wbar_L_11_Asian = compute_Wbar_L_hyy(primary_index,
                                     proxy_medicine[, "Asian_prob"], proxy_medicine[, "py1yhat1"],
                                     ifelse(proxy_medicine$race == "Asian.", 1, 0),
                                     ifelse((proxy_medicine$Yhat == 1) & (proxy_medicine$Y == 1), 1, 0))
Wbar_L_01_Asian = compute_Wbar_L_hyy(primary_index,
                                     proxy_medicine[, "Asian_prob"], proxy_medicine[, "py1yhat0"],
                                     ifelse(proxy_medicine$race == "Asian.", 1, 0),
                                     ifelse((proxy_medicine$Yhat == 0) & (proxy_medicine$Y == 1), 1, 0))
Wbar_U_11_Asian = compute_Wbar_U_hyy(primary_index,
                                     proxy_medicine[, "Asian_prob"], proxy_medicine[, "py1yhat1"],
                                     ifelse(proxy_medicine$race == "Asian.", 1, 0),
                                     ifelse((proxy_medicine$Yhat == 1) & (proxy_medicine$Y == 1), 1, 0))
Wbar_U_01_Asian = compute_Wbar_U_hyy(primary_index,
                                     proxy_medicine[, "Asian_prob"], proxy_medicine[, "py1yhat0"],
                                     ifelse(proxy_medicine$race == "Asian.", 1, 0),
                                     ifelse((proxy_medicine$Yhat == 0) & (proxy_medicine$Y == 1), 1, 0))
mu_Asian_WL_WU_11 = (truncate(Wbar_L_11_Asian)/(truncate(Wbar_L_11_Asian) + truncate(Wbar_U_01_Asian)))
mu_Asian_WU_WL_11 = (truncate(Wbar_U_11_Asian)/(truncate(Wbar_U_11_Asian) + truncate(Wbar_L_01_Asian)))
mu_Asian_WL_WU_01 = truncate(Wbar_L_01_Asian)/(truncate(Wbar_L_01_Asian) + truncate(Wbar_U_11_Asian))
mu_Asian_WU_WL_01 = truncate(Wbar_U_01_Asian)/(truncate(Wbar_U_01_Asian) + truncate(Wbar_L_11_Asian))

Wbar_L_11_Other = compute_Wbar_L_hyy(primary_index,
                                     1 - proxy_medicine[, "Asian_prob"], proxy_medicine[, "py1yhat1"],
                                     1 - ifelse(proxy_medicine$race == "Asian.", 1, 0),
                                     ifelse((proxy_medicine$Yhat == 1) & (proxy_medicine$Y == 1), 1, 0))
Wbar_L_01_Other = compute_Wbar_L_hyy(primary_index,
                                     1 - proxy_medicine[, "Asian_prob"], proxy_medicine[, "py1yhat0"],
                                     1- ifelse(proxy_medicine$race == "Asian.", 1, 0),
                                     ifelse((proxy_medicine$Yhat == 0) & (proxy_medicine$Y == 1), 1, 0))
Wbar_U_11_Other = compute_Wbar_U_hyy(primary_index,
                                     1 - proxy_medicine[, "Asian_prob"], proxy_medicine[, "py1yhat1"],
                                     1 - ifelse(proxy_medicine$race == "Asian.", 1, 0),
                                     ifelse((proxy_medicine$Yhat == 1) & (proxy_medicine$Y == 1), 1, 0))
Wbar_U_01_Other = compute_Wbar_U_hyy(primary_index,
                                     1 - proxy_medicine[, "Asian_prob"], proxy_medicine[, "py1yhat0"],
                                     1 - ifelse(proxy_medicine$race == "Asian.", 1, 0),
                                     ifelse((proxy_medicine$Yhat == 0) & (proxy_medicine$Y == 1), 1, 0))
mu_Other_WL_WU_11 = (truncate(Wbar_L_11_Other)/(truncate(Wbar_L_11_Other) + truncate(Wbar_U_01_Other)))
mu_Other_WU_WL_11 = (truncate(Wbar_U_11_Other)/(truncate(Wbar_U_11_Other) + truncate(Wbar_L_01_Other)))
mu_Other_WL_WU_01 = truncate(Wbar_L_01_Other)/(truncate(Wbar_L_01_Other) + truncate(Wbar_U_11_Other))
mu_Other_WU_WL_01 = truncate(Wbar_U_01_Other)/(truncate(Wbar_U_01_Other) + truncate(Wbar_L_11_Other))

bound_df_medicine$PI_lower[bound_df_medicine$race == "Asian-vs-Rest"] = mu_Asian_WL_WU_11 - mu_Other_WU_WL_11
bound_df_medicine$PI_upper[bound_df_medicine$race == "Asian-vs-Rest"] = mu_Asian_WU_WL_11 - mu_Other_WL_WU_11
consta_UL = mu_Asian_WU_WL_01/(Wbar_L_11_Asian + Wbar_U_01_Asian)
consta_LU = mu_Asian_WL_WU_11/(Wbar_L_11_Asian + Wbar_U_01_Asian)
constb_UL = mu_Other_WU_WL_11/(Wbar_U_11_Other + Wbar_L_01_Other)
constb_LU = mu_Other_WL_WU_01/(Wbar_U_11_Other + Wbar_L_01_Other)
VL = compute_VL(r, primary_index,
                consta_UL, consta_LU, constb_UL, constb_LU,
                proxy_medicine[, "Asian_prob"], 1 - proxy_medicine[, "Asian_prob"],
                ifelse(proxy_medicine$race == "Asian.", 1, 0),
                1 - ifelse(proxy_medicine$race == "Asian.", 1, 0),
                proxy_medicine[, "py1yhat1"], proxy_medicine[, "py1yhat0"],
                ifelse((proxy_medicine$Yhat == 1) & (proxy_medicine$Y == 1), 1, 0),
                ifelse((proxy_medicine$Yhat == 0) & (proxy_medicine$Y == 1), 1, 0))
consta_UL = mu_Asian_WU_WL_11/(Wbar_U_11_Asian + Wbar_L_01_Asian)
consta_LU = mu_Asian_WL_WU_01/(Wbar_U_11_Asian + Wbar_L_01_Asian)
constb_UL = mu_Other_WU_WL_01/(Wbar_L_11_Other + Wbar_U_01_Other)
constb_LU = mu_Other_WL_WU_11/(Wbar_L_11_Other + Wbar_U_01_Other)
VU = compute_VU(r, primary_index,
                consta_UL, consta_LU, constb_UL, constb_LU,
                proxy_medicine[, "Asian_prob"], 1 - proxy_medicine[, "Asian_prob"],
                ifelse(proxy_medicine$race == "Asian.", 1, 0),
                1 - ifelse(proxy_medicine$race == "Asian.", 1, 0),
                proxy_medicine[, "py1yhat1"], proxy_medicine[, "py1yhat0"],
                ifelse((proxy_medicine$Yhat == 1) & (proxy_medicine$Y == 1), 1, 0),
                ifelse((proxy_medicine$Yhat == 0) & (proxy_medicine$Y == 1), 1, 0))
bound_df_medicine$CI_lower[bound_df_medicine$race == "Asian-vs-Rest"] =
  mu_Asian_WL_WU_11 - mu_Other_WU_WL_11 - qnorm(1 - 0.025) * sqrt(VL)/sqrt(nrow(proxy_medicine)/2)
bound_df_medicine$CI_upper[bound_df_medicine$race == "Asian-vs-Rest"] =
  mu_Asian_WU_WL_11 - mu_Other_WL_WU_11 + qnorm(1 - 0.025) * sqrt(VU)/sqrt(nrow(proxy_medicine)/2)

#########################
# Only Geonetic
#########################
bound_df_genetic = data.frame(PI_lower = rep(0, 3), PI_upper = rep(0, 3),
                              CI_lower = rep(0, 3), CI_upper = rep(0, 3),
                              truth = rep(0, 3),
                              race = (c("White-vs-Rest", "Black-vs-Rest", "Asian-vs-Rest")))

proxy_genetic = read_csv("genetic_as_proxy.csv")
proxy_genetic = proxy_genetic %>%
  mutate(
    White_prob = white_prob,
    Black_prob = black_prob,
    Asian_prob = asian_prob
  )

### White vs rest
Wbar_L_11_White = compute_Wbar_L_hyy(primary_index,
                                     proxy_genetic[, "White_prob"], proxy_genetic[, "py1yhat1"],
                                     ifelse(proxy_genetic$race == "White.", 1, 0),
                                     ifelse((proxy_genetic$Yhat == 1) & (proxy_genetic$Y == 1), 1, 0))
Wbar_L_01_White = compute_Wbar_L_hyy(primary_index,
                                     proxy_genetic[, "White_prob"], proxy_genetic[, "py1yhat0"],
                                     ifelse(proxy_genetic$race == "White.", 1, 0),
                                     ifelse((proxy_genetic$Yhat == 0) & (proxy_genetic$Y == 1), 1, 0))
Wbar_U_11_White = compute_Wbar_U_hyy(primary_index,
                                     proxy_genetic[, "White_prob"], proxy_genetic[, "py1yhat1"],
                                     ifelse(proxy_genetic$race == "White.", 1, 0),
                                     ifelse((proxy_genetic$Yhat == 1) & (proxy_genetic$Y == 1), 1, 0))
Wbar_U_01_White = compute_Wbar_U_hyy(primary_index,
                                     proxy_genetic[, "White_prob"], proxy_genetic[, "py1yhat0"],
                                     ifelse(proxy_genetic$race == "White.", 1, 0),
                                     ifelse((proxy_genetic$Yhat == 0) & (proxy_genetic$Y == 1), 1, 0))
mu_White_WL_WU_11 = (truncate(Wbar_L_11_White)/(truncate(Wbar_L_11_White) + truncate(Wbar_U_01_White)))
mu_White_WU_WL_11 = (truncate(Wbar_U_11_White)/(truncate(Wbar_U_11_White) + truncate(Wbar_L_01_White)))
mu_White_WL_WU_01 = truncate(Wbar_L_01_White)/(truncate(Wbar_L_01_White) + truncate(Wbar_U_11_White))
mu_White_WU_WL_01 = truncate(Wbar_U_01_White)/(truncate(Wbar_U_01_White) + truncate(Wbar_L_11_White))

Wbar_L_11_Other = compute_Wbar_L_hyy(primary_index,
                                     1 - proxy_genetic[, "White_prob"], proxy_genetic[, "py1yhat1"],
                                     1 - ifelse(proxy_genetic$race == "White.", 1, 0),
                                     ifelse((proxy_genetic$Yhat == 1) & (proxy_genetic$Y == 1), 1, 0))
Wbar_L_01_Other = compute_Wbar_L_hyy(primary_index,
                                     1 - proxy_genetic[, "White_prob"], proxy_genetic[, "py1yhat0"],
                                     1- ifelse(proxy_genetic$race == "White.", 1, 0),
                                     ifelse((proxy_genetic$Yhat == 0) & (proxy_genetic$Y == 1), 1, 0))
Wbar_U_11_Other = compute_Wbar_U_hyy(primary_index,
                                     1 - proxy_genetic[, "White_prob"], proxy_genetic[, "py1yhat1"],
                                     1 - ifelse(proxy_genetic$race == "White.", 1, 0),
                                     ifelse((proxy_genetic$Yhat == 1) & (proxy_genetic$Y == 1), 1, 0))
Wbar_U_01_Other = compute_Wbar_U_hyy(primary_index,
                                     1 - proxy_genetic[, "White_prob"], proxy_genetic[, "py1yhat0"],
                                     1 - ifelse(proxy_genetic$race == "White.", 1, 0),
                                     ifelse((proxy_genetic$Yhat == 0) & (proxy_genetic$Y == 1), 1, 0))
mu_Other_WL_WU_11 = (truncate(Wbar_L_11_Other)/(truncate(Wbar_L_11_Other) + truncate(Wbar_U_01_Other)))
mu_Other_WU_WL_11 = (truncate(Wbar_U_11_Other)/(truncate(Wbar_U_11_Other) + truncate(Wbar_L_01_Other)))
mu_Other_WL_WU_01 = truncate(Wbar_L_01_Other)/(truncate(Wbar_L_01_Other) + truncate(Wbar_U_11_Other))
mu_Other_WU_WL_01 = truncate(Wbar_U_01_Other)/(truncate(Wbar_U_01_Other) + truncate(Wbar_L_11_Other))

bound_df_genetic$PI_lower[bound_df_genetic$race == "White-vs-Rest"] = mu_White_WL_WU_11 - mu_Other_WU_WL_11
bound_df_genetic$PI_upper[bound_df_genetic$race == "White-vs-Rest"] = mu_White_WU_WL_11 - mu_Other_WL_WU_11

consta_UL = mu_White_WU_WL_01/(Wbar_L_11_White + Wbar_U_01_White)
consta_LU = mu_White_WL_WU_11/(Wbar_L_11_White + Wbar_U_01_White)
constb_UL = mu_Other_WU_WL_11/(Wbar_U_11_Other + Wbar_L_01_Other)
constb_LU = mu_Other_WL_WU_01/(Wbar_U_11_Other + Wbar_L_01_Other)

VL = compute_VL(r, primary_index,
                consta_UL, consta_LU, constb_UL, constb_LU,
                proxy_genetic[, "White_prob"], 1 - proxy_genetic[, "White_prob"],
                ifelse(proxy_genetic$race == "White.", 1, 0),
                1 - ifelse(proxy_genetic$race == "White.", 1, 0),
                proxy_genetic[, "py1yhat1"], proxy_genetic[, "py1yhat0"],
                ifelse((proxy_genetic$Yhat == 1) & (proxy_genetic$Y == 1), 1, 0),
                ifelse((proxy_genetic$Yhat == 0) & (proxy_genetic$Y == 1), 1, 0))

consta_UL = mu_White_WU_WL_11/(Wbar_U_11_White + Wbar_L_01_White)
consta_LU = mu_White_WL_WU_01/(Wbar_U_11_White + Wbar_L_01_White)
constb_UL = mu_Other_WU_WL_01/(Wbar_L_11_Other + Wbar_U_01_Other)
constb_LU = mu_Other_WL_WU_11/(Wbar_L_11_Other + Wbar_U_01_Other)

VU = compute_VU(r, primary_index,
                consta_UL, consta_LU, constb_UL, constb_LU,
                proxy_genetic[, "White_prob"], 1 - proxy_genetic[, "White_prob"],
                ifelse(proxy_genetic$race == "White.", 1, 0),
                1 - ifelse(proxy_genetic$race == "White.", 1, 0),
                proxy_genetic[, "py1yhat1"], proxy_genetic[, "py1yhat0"],
                ifelse((proxy_genetic$Yhat == 1) & (proxy_genetic$Y == 1), 1, 0),
                ifelse((proxy_genetic$Yhat == 0) & (proxy_genetic$Y == 1), 1, 0))

bound_df_genetic$CI_lower[bound_df_genetic$race == "White-vs-Rest"] =
  mu_White_WL_WU_11 - mu_Other_WU_WL_11 - qnorm(1 - 0.025) * sqrt(VL)/sqrt(nrow(proxy_genetic)/2)
bound_df_genetic$CI_upper[bound_df_genetic$race == "White-vs-Rest"] =
  mu_White_WU_WL_11 - mu_Other_WL_WU_11 + qnorm(1 - 0.025) * sqrt(VU)/sqrt(nrow(proxy_genetic)/2)

### Black vs rest
Wbar_L_11_Black = compute_Wbar_L_hyy(primary_index,
                                     proxy_genetic[, "Black_prob"], proxy_genetic[, "py1yhat1"],
                                     ifelse(proxy_genetic$race == "Black.", 1, 0),
                                     ifelse((proxy_genetic$Yhat == 1) & (proxy_genetic$Y == 1), 1, 0))
Wbar_L_01_Black = compute_Wbar_L_hyy(primary_index,
                                     proxy_genetic[, "Black_prob"], proxy_genetic[, "py1yhat0"],
                                     ifelse(proxy_genetic$race == "Black.", 1, 0),
                                     ifelse((proxy_genetic$Yhat == 0) & (proxy_genetic$Y == 1), 1, 0))
Wbar_U_11_Black = compute_Wbar_U_hyy(primary_index,
                                     proxy_genetic[, "Black_prob"], proxy_genetic[, "py1yhat1"],
                                     ifelse(proxy_genetic$race == "Black.", 1, 0),
                                     ifelse((proxy_genetic$Yhat == 1) & (proxy_genetic$Y == 1), 1, 0))
Wbar_U_01_Black = compute_Wbar_U_hyy(primary_index,
                                     proxy_genetic[, "Black_prob"], proxy_genetic[, "py1yhat0"],
                                     ifelse(proxy_genetic$race == "Black.", 1, 0),
                                     ifelse((proxy_genetic$Yhat == 0) & (proxy_genetic$Y == 1), 1, 0))
mu_Black_WL_WU_11 = (truncate(Wbar_L_11_Black)/(truncate(Wbar_L_11_Black) + truncate(Wbar_U_01_Black)))
mu_Black_WU_WL_11 = (truncate(Wbar_U_11_Black)/(truncate(Wbar_U_11_Black) + truncate(Wbar_L_01_Black)))
mu_Black_WL_WU_01 = truncate(Wbar_L_01_Black)/(truncate(Wbar_L_01_Black) + truncate(Wbar_U_11_Black))
mu_Black_WU_WL_01 = truncate(Wbar_U_01_Black)/(truncate(Wbar_U_01_Black) + truncate(Wbar_L_11_Black))

Wbar_L_11_Other = compute_Wbar_L_hyy(primary_index,
                                     1 - proxy_genetic[, "Black_prob"], proxy_genetic[, "py1yhat1"],
                                     1 - ifelse(proxy_genetic$race == "Black.", 1, 0),
                                     ifelse((proxy_genetic$Yhat == 1) & (proxy_genetic$Y == 1), 1, 0))
Wbar_L_01_Other = compute_Wbar_L_hyy(primary_index,
                                     1 - proxy_genetic[, "Black_prob"], proxy_genetic[, "py1yhat0"],
                                     1- ifelse(proxy_genetic$race == "Black.", 1, 0),
                                     ifelse((proxy_genetic$Yhat == 0) & (proxy_genetic$Y == 1), 1, 0))
Wbar_U_11_Other = compute_Wbar_U_hyy(primary_index,
                                     1 - proxy_genetic[, "Black_prob"], proxy_genetic[, "py1yhat1"],
                                     1 - ifelse(proxy_genetic$race == "Black.", 1, 0),
                                     ifelse((proxy_genetic$Yhat == 1) & (proxy_genetic$Y == 1), 1, 0))
Wbar_U_01_Other = compute_Wbar_U_hyy(primary_index,
                                     1 - proxy_genetic[, "Black_prob"], proxy_genetic[, "py1yhat0"],
                                     1 - ifelse(proxy_genetic$race == "Black.", 1, 0),
                                     ifelse((proxy_genetic$Yhat == 0) & (proxy_genetic$Y == 1), 1, 0))
mu_Other_WL_WU_11 = (truncate(Wbar_L_11_Other)/(truncate(Wbar_L_11_Other) + truncate(Wbar_U_01_Other)))
mu_Other_WU_WL_11 = (truncate(Wbar_U_11_Other)/(truncate(Wbar_U_11_Other) + truncate(Wbar_L_01_Other)))
mu_Other_WL_WU_01 = truncate(Wbar_L_01_Other)/(truncate(Wbar_L_01_Other) + truncate(Wbar_U_11_Other))
mu_Other_WU_WL_01 = truncate(Wbar_U_01_Other)/(truncate(Wbar_U_01_Other) + truncate(Wbar_L_11_Other))

bound_df_genetic$PI_lower[bound_df_genetic$race == "Black-vs-Rest"] = mu_Black_WL_WU_11 - mu_Other_WU_WL_11
bound_df_genetic$PI_upper[bound_df_genetic$race == "Black-vs-Rest"] = mu_Black_WU_WL_11 - mu_Other_WL_WU_11

consta_UL = mu_Black_WU_WL_01/(Wbar_L_11_Black + Wbar_U_01_Black)
consta_LU = mu_Black_WL_WU_11/(Wbar_L_11_Black + Wbar_U_01_Black)
constb_UL = mu_Other_WU_WL_11/(Wbar_U_11_Other + Wbar_L_01_Other)
constb_LU = mu_Other_WL_WU_01/(Wbar_U_11_Other + Wbar_L_01_Other)

VL = compute_VL(r, primary_index,
                consta_UL, consta_LU, constb_UL, constb_LU,
                proxy_genetic[, "Black_prob"], 1 - proxy_genetic[, "Black_prob"],
                ifelse(proxy_genetic$race == "Black.", 1, 0),
                1 - ifelse(proxy_genetic$race == "Black.", 1, 0),
                proxy_genetic[, "py1yhat1"], proxy_genetic[, "py1yhat0"],
                ifelse((proxy_genetic$Yhat == 1) & (proxy_genetic$Y == 1), 1, 0),
                ifelse((proxy_genetic$Yhat == 0) & (proxy_genetic$Y == 1), 1, 0))

consta_UL = mu_Black_WU_WL_11/(Wbar_U_11_Black + Wbar_L_01_Black)
consta_LU = mu_Black_WL_WU_01/(Wbar_U_11_Black + Wbar_L_01_Black)
constb_UL = mu_Other_WU_WL_01/(Wbar_L_11_Other + Wbar_U_01_Other)
constb_LU = mu_Other_WL_WU_11/(Wbar_L_11_Other + Wbar_U_01_Other)

VU = compute_VU(r, primary_index,
                consta_UL, consta_LU, constb_UL, constb_LU,
                proxy_genetic[, "Black_prob"], 1 - proxy_genetic[, "Black_prob"],
                ifelse(proxy_genetic$race == "Black.", 1, 0),
                1 - ifelse(proxy_genetic$race == "Black.", 1, 0),
                proxy_genetic[, "py1yhat1"], proxy_genetic[, "py1yhat0"],
                ifelse((proxy_genetic$Yhat == 1) & (proxy_genetic$Y == 1), 1, 0),
                ifelse((proxy_genetic$Yhat == 0) & (proxy_genetic$Y == 1), 1, 0))

bound_df_genetic$CI_lower[bound_df_genetic$race == "Black-vs-Rest"] =
  mu_Black_WL_WU_11 - mu_Other_WU_WL_11 - qnorm(1 - 0.025) * sqrt(VL)/sqrt(nrow(proxy_genetic)/2)
bound_df_genetic$CI_upper[bound_df_genetic$race == "Black-vs-Rest"] =
  mu_Black_WU_WL_11 - mu_Other_WL_WU_11 + qnorm(1 - 0.025) * sqrt(VU)/sqrt(nrow(proxy_genetic)/2)

### Asian vs rest
Wbar_L_11_Asian = compute_Wbar_L_hyy(primary_index,
                                     proxy_genetic[, "Asian_prob"], proxy_genetic[, "py1yhat1"],
                                     ifelse(proxy_genetic$race == "Asian.", 1, 0),
                                     ifelse((proxy_genetic$Yhat == 1) & (proxy_genetic$Y == 1), 1, 0))
Wbar_L_01_Asian = compute_Wbar_L_hyy(primary_index,
                                     proxy_genetic[, "Asian_prob"], proxy_genetic[, "py1yhat0"],
                                     ifelse(proxy_genetic$race == "Asian.", 1, 0),
                                     ifelse((proxy_genetic$Yhat == 0) & (proxy_genetic$Y == 1), 1, 0))
Wbar_U_11_Asian = compute_Wbar_U_hyy(primary_index,
                                     proxy_genetic[, "Asian_prob"], proxy_genetic[, "py1yhat1"],
                                     ifelse(proxy_genetic$race == "Asian.", 1, 0),
                                     ifelse((proxy_genetic$Yhat == 1) & (proxy_genetic$Y == 1), 1, 0))
Wbar_U_01_Asian = compute_Wbar_U_hyy(primary_index,
                                     proxy_genetic[, "Asian_prob"], proxy_genetic[, "py1yhat0"],
                                     ifelse(proxy_genetic$race == "Asian.", 1, 0),
                                     ifelse((proxy_genetic$Yhat == 0) & (proxy_genetic$Y == 1), 1, 0))
mu_Asian_WL_WU_11 = (truncate(Wbar_L_11_Asian)/(truncate(Wbar_L_11_Asian) + truncate(Wbar_U_01_Asian)))
mu_Asian_WU_WL_11 = (truncate(Wbar_U_11_Asian)/(truncate(Wbar_U_11_Asian) + truncate(Wbar_L_01_Asian)))
mu_Asian_WL_WU_01 = truncate(Wbar_L_01_Asian)/(truncate(Wbar_L_01_Asian) + truncate(Wbar_U_11_Asian))
mu_Asian_WU_WL_01 = truncate(Wbar_U_01_Asian)/(truncate(Wbar_U_01_Asian) + truncate(Wbar_L_11_Asian))

Wbar_L_11_Other = compute_Wbar_L_hyy(primary_index,
                                     1 - proxy_genetic[, "Asian_prob"], proxy_genetic[, "py1yhat1"],
                                     1 - ifelse(proxy_genetic$race == "Asian.", 1, 0),
                                     ifelse((proxy_genetic$Yhat == 1) & (proxy_genetic$Y == 1), 1, 0))
Wbar_L_01_Other = compute_Wbar_L_hyy(primary_index,
                                     1 - proxy_genetic[, "Asian_prob"], proxy_genetic[, "py1yhat0"],
                                     1- ifelse(proxy_genetic$race == "Asian.", 1, 0),
                                     ifelse((proxy_genetic$Yhat == 0) & (proxy_genetic$Y == 1), 1, 0))
Wbar_U_11_Other = compute_Wbar_U_hyy(primary_index,
                                     1 - proxy_genetic[, "Asian_prob"], proxy_genetic[, "py1yhat1"],
                                     1 - ifelse(proxy_genetic$race == "Asian.", 1, 0),
                                     ifelse((proxy_genetic$Yhat == 1) & (proxy_genetic$Y == 1), 1, 0))
Wbar_U_01_Other = compute_Wbar_U_hyy(primary_index,
                                     1 - proxy_genetic[, "Asian_prob"], proxy_genetic[, "py1yhat0"],
                                     1 - ifelse(proxy_genetic$race == "Asian.", 1, 0),
                                     ifelse((proxy_genetic$Yhat == 0) & (proxy_genetic$Y == 1), 1, 0))
mu_Other_WL_WU_11 = (truncate(Wbar_L_11_Other)/(truncate(Wbar_L_11_Other) + truncate(Wbar_U_01_Other)))
mu_Other_WU_WL_11 = (truncate(Wbar_U_11_Other)/(truncate(Wbar_U_11_Other) + truncate(Wbar_L_01_Other)))
mu_Other_WL_WU_01 = truncate(Wbar_L_01_Other)/(truncate(Wbar_L_01_Other) + truncate(Wbar_U_11_Other))
mu_Other_WU_WL_01 = truncate(Wbar_U_01_Other)/(truncate(Wbar_U_01_Other) + truncate(Wbar_L_11_Other))

bound_df_genetic$PI_lower[bound_df_genetic$race == "Asian-vs-Rest"] = mu_Asian_WL_WU_11 - mu_Other_WU_WL_11
bound_df_genetic$PI_upper[bound_df_genetic$race == "Asian-vs-Rest"] = mu_Asian_WU_WL_11 - mu_Other_WL_WU_11

consta_UL = mu_Asian_WU_WL_01/(Wbar_L_11_Asian + Wbar_U_01_Asian)
consta_LU = mu_Asian_WL_WU_11/(Wbar_L_11_Asian + Wbar_U_01_Asian)
constb_UL = mu_Other_WU_WL_11/(Wbar_U_11_Other + Wbar_L_01_Other)
constb_LU = mu_Other_WL_WU_01/(Wbar_U_11_Other + Wbar_L_01_Other)

VL = compute_VL(r, primary_index,
                consta_UL, consta_LU, constb_UL, constb_LU,
                proxy_genetic[, "Asian_prob"], 1 - proxy_genetic[, "Asian_prob"],
                ifelse(proxy_genetic$race == "Asian.", 1, 0),
                1 - ifelse(proxy_genetic$race == "Asian.", 1, 0),
                proxy_genetic[, "py1yhat1"], proxy_genetic[, "py1yhat0"],
                ifelse((proxy_genetic$Yhat == 1) & (proxy_genetic$Y == 1), 1, 0),
                ifelse((proxy_genetic$Yhat == 0) & (proxy_genetic$Y == 1), 1, 0))

consta_UL = mu_Asian_WU_WL_11/(Wbar_U_11_Asian + Wbar_L_01_Asian)
consta_LU = mu_Asian_WL_WU_01/(Wbar_U_11_Asian + Wbar_L_01_Asian)
constb_UL = mu_Other_WU_WL_01/(Wbar_L_11_Other + Wbar_U_01_Other)
constb_LU = mu_Other_WL_WU_11/(Wbar_L_11_Other + Wbar_U_01_Other)

VU = compute_VU(r, primary_index,
                consta_UL, consta_LU, constb_UL, constb_LU,
                proxy_genetic[, "Asian_prob"], 1 - proxy_genetic[, "Asian_prob"],
                ifelse(proxy_genetic$race == "Asian.", 1, 0),
                1 - ifelse(proxy_genetic$race == "Asian.", 1, 0),
                proxy_genetic[, "py1yhat1"], proxy_genetic[, "py1yhat0"],
                ifelse((proxy_genetic$Yhat == 1) & (proxy_genetic$Y == 1), 1, 0),
                ifelse((proxy_genetic$Yhat == 0) & (proxy_genetic$Y == 1), 1, 0))

bound_df_genetic$CI_lower[bound_df_genetic$race == "Asian-vs-Rest"] =
  mu_Asian_WL_WU_11 - mu_Other_WU_WL_11 - qnorm(1 - 0.025) * sqrt(VL)/sqrt(nrow(proxy_genetic)/2)
bound_df_genetic$CI_upper[bound_df_genetic$race == "Asian-vs-Rest"] =
  mu_Asian_WU_WL_11 - mu_Other_WL_WU_11 + qnorm(1 - 0.025) * sqrt(VU)/sqrt(nrow(proxy_genetic)/2)


#########################
# Both medicine and Geonetic
#########################
bound_df_medicine_genetic = data.frame(PI_lower = rep(0, 3), PI_upper = rep(0, 3),
                                       CI_lower = rep(0, 3), CI_upper = rep(0, 3),
                                       truth = rep(0, 3),
                                       race = (c("White-vs-Rest", "Black-vs-Rest", "Asian-vs-Rest")))

proxy_medicine_genetic = read_csv("medicine_genetic_as_proxy.csv")
proxy_medicine_genetic = proxy_medicine_genetic %>%
  mutate(
    White_prob = white_prob,
    Black_prob = black_prob,
    Asian_prob = asian_prob
  )

### White vs rest
Wbar_L_11_White = compute_Wbar_L_hyy(primary_index,
                                     proxy_medicine_genetic[, "White_prob"], proxy_medicine_genetic[, "py1yhat1"],
                                     ifelse(proxy_medicine_genetic$race == "White.", 1, 0),
                                     ifelse((proxy_medicine_genetic$Yhat == 1) & (proxy_medicine_genetic$Y == 1), 1, 0))
Wbar_L_01_White = compute_Wbar_L_hyy(primary_index,
                                     proxy_medicine_genetic[, "White_prob"], proxy_medicine_genetic[, "py1yhat0"],
                                     ifelse(proxy_medicine_genetic$race == "White.", 1, 0),
                                     ifelse((proxy_medicine_genetic$Yhat == 0) & (proxy_medicine_genetic$Y == 1), 1, 0))
Wbar_U_11_White = compute_Wbar_U_hyy(primary_index,
                                     proxy_medicine_genetic[, "White_prob"], proxy_medicine_genetic[, "py1yhat1"],
                                     ifelse(proxy_medicine_genetic$race == "White.", 1, 0),
                                     ifelse((proxy_medicine_genetic$Yhat == 1) & (proxy_medicine_genetic$Y == 1), 1, 0))
Wbar_U_01_White = compute_Wbar_U_hyy(primary_index,
                                     proxy_medicine_genetic[, "White_prob"], proxy_medicine_genetic[, "py1yhat0"],
                                     ifelse(proxy_medicine_genetic$race == "White.", 1, 0),
                                     ifelse((proxy_medicine_genetic$Yhat == 0) & (proxy_medicine_genetic$Y == 1), 1, 0))
mu_White_WL_WU_11 = (truncate(Wbar_L_11_White)/(truncate(Wbar_L_11_White) + truncate(Wbar_U_01_White)))
mu_White_WU_WL_11 = (truncate(Wbar_U_11_White)/(truncate(Wbar_U_11_White) + truncate(Wbar_L_01_White)))
mu_White_WL_WU_01 = truncate(Wbar_L_01_White)/(truncate(Wbar_L_01_White) + truncate(Wbar_U_11_White))
mu_White_WU_WL_01 = truncate(Wbar_U_01_White)/(truncate(Wbar_U_01_White) + truncate(Wbar_L_11_White))

Wbar_L_11_Other = compute_Wbar_L_hyy(primary_index,
                                     1 - proxy_medicine_genetic[, "White_prob"], proxy_medicine_genetic[, "py1yhat1"],
                                     1 - ifelse(proxy_medicine_genetic$race == "White.", 1, 0),
                                     ifelse((proxy_medicine_genetic$Yhat == 1) & (proxy_medicine_genetic$Y == 1), 1, 0))
Wbar_L_01_Other = compute_Wbar_L_hyy(primary_index,
                                     1 - proxy_medicine_genetic[, "White_prob"], proxy_medicine_genetic[, "py1yhat0"],
                                     1- ifelse(proxy_medicine_genetic$race == "White.", 1, 0),
                                     ifelse((proxy_medicine_genetic$Yhat == 0) & (proxy_medicine_genetic$Y == 1), 1, 0))
Wbar_U_11_Other = compute_Wbar_U_hyy(primary_index,
                                     1 - proxy_medicine_genetic[, "White_prob"], proxy_medicine_genetic[, "py1yhat1"],
                                     1 - ifelse(proxy_medicine_genetic$race == "White.", 1, 0),
                                     ifelse((proxy_medicine_genetic$Yhat == 1) & (proxy_medicine_genetic$Y == 1), 1, 0))
Wbar_U_01_Other = compute_Wbar_U_hyy(primary_index,
                                     1 - proxy_medicine_genetic[, "White_prob"], proxy_medicine_genetic[, "py1yhat0"],
                                     1 - ifelse(proxy_medicine_genetic$race == "White.", 1, 0),
                                     ifelse((proxy_medicine_genetic$Yhat == 0) & (proxy_medicine_genetic$Y == 1), 1, 0))
mu_Other_WL_WU_11 = (truncate(Wbar_L_11_Other)/(truncate(Wbar_L_11_Other) + truncate(Wbar_U_01_Other)))
mu_Other_WU_WL_11 = (truncate(Wbar_U_11_Other)/(truncate(Wbar_U_11_Other) + truncate(Wbar_L_01_Other)))
mu_Other_WL_WU_01 = truncate(Wbar_L_01_Other)/(truncate(Wbar_L_01_Other) + truncate(Wbar_U_11_Other))
mu_Other_WU_WL_01 = truncate(Wbar_U_01_Other)/(truncate(Wbar_U_01_Other) + truncate(Wbar_L_11_Other))

bound_df_medicine_genetic$PI_lower[bound_df_medicine_genetic$race == "White-vs-Rest"] = mu_White_WL_WU_11 - mu_Other_WU_WL_11
bound_df_medicine_genetic$PI_upper[bound_df_medicine_genetic$race == "White-vs-Rest"] = mu_White_WU_WL_11 - mu_Other_WL_WU_11

consta_UL = mu_White_WU_WL_01/(Wbar_L_11_White + Wbar_U_01_White)
consta_LU = mu_White_WL_WU_11/(Wbar_L_11_White + Wbar_U_01_White)
constb_UL = mu_Other_WU_WL_11/(Wbar_U_11_Other + Wbar_L_01_Other)
constb_LU = mu_Other_WL_WU_01/(Wbar_U_11_Other + Wbar_L_01_Other)

VL = compute_VL(r, primary_index,
                consta_UL, consta_LU, constb_UL, constb_LU,
                proxy_medicine_genetic[, "White_prob"], 1 - proxy_medicine_genetic[, "White_prob"],
                ifelse(proxy_medicine_genetic$race == "White.", 1, 0),
                1 - ifelse(proxy_medicine_genetic$race == "White.", 1, 0),
                proxy_medicine_genetic[, "py1yhat1"], proxy_medicine_genetic[, "py1yhat0"],
                ifelse((proxy_medicine_genetic$Yhat == 1) & (proxy_medicine_genetic$Y == 1), 1, 0),
                ifelse((proxy_medicine_genetic$Yhat == 0) & (proxy_medicine_genetic$Y == 1), 1, 0))

consta_UL = mu_White_WU_WL_11/(Wbar_U_11_White + Wbar_L_01_White)
consta_LU = mu_White_WL_WU_01/(Wbar_U_11_White + Wbar_L_01_White)
constb_UL = mu_Other_WU_WL_01/(Wbar_L_11_Other + Wbar_U_01_Other)
constb_LU = mu_Other_WL_WU_11/(Wbar_L_11_Other + Wbar_U_01_Other)

VU = compute_VU(r, primary_index,
                consta_UL, consta_LU, constb_UL, constb_LU,
                proxy_medicine_genetic[, "White_prob"], 1 - proxy_medicine_genetic[, "White_prob"],
                ifelse(proxy_medicine_genetic$race == "White.", 1, 0),
                1 - ifelse(proxy_medicine_genetic$race == "White.", 1, 0),
                proxy_medicine_genetic[, "py1yhat1"], proxy_medicine_genetic[, "py1yhat0"],
                ifelse((proxy_medicine_genetic$Yhat == 1) & (proxy_medicine_genetic$Y == 1), 1, 0),
                ifelse((proxy_medicine_genetic$Yhat == 0) & (proxy_medicine_genetic$Y == 1), 1, 0))

bound_df_medicine_genetic$CI_lower[bound_df_medicine_genetic$race == "White-vs-Rest"] =
  mu_White_WL_WU_11 - mu_Other_WU_WL_11 - qnorm(1 - 0.025) * sqrt(VL)/sqrt(nrow(proxy_medicine_genetic)/2)
bound_df_medicine_genetic$CI_upper[bound_df_medicine_genetic$race == "White-vs-Rest"] =
  mu_White_WU_WL_11 - mu_Other_WL_WU_11 + qnorm(1 - 0.025) * sqrt(VU)/sqrt(nrow(proxy_medicine_genetic)/2)

### Black vs rest
Wbar_L_11_Black = compute_Wbar_L_hyy(primary_index,
                                     proxy_medicine_genetic[, "Black_prob"], proxy_medicine_genetic[, "py1yhat1"],
                                     ifelse(proxy_medicine_genetic$race == "Black.", 1, 0),
                                     ifelse((proxy_medicine_genetic$Yhat == 1) & (proxy_medicine_genetic$Y == 1), 1, 0))
Wbar_L_01_Black = compute_Wbar_L_hyy(primary_index,
                                     proxy_medicine_genetic[, "Black_prob"], proxy_medicine_genetic[, "py1yhat0"],
                                     ifelse(proxy_medicine_genetic$race == "Black.", 1, 0),
                                     ifelse((proxy_medicine_genetic$Yhat == 0) & (proxy_medicine_genetic$Y == 1), 1, 0))
Wbar_U_11_Black = compute_Wbar_U_hyy(primary_index,
                                     proxy_medicine_genetic[, "Black_prob"], proxy_medicine_genetic[, "py1yhat1"],
                                     ifelse(proxy_medicine_genetic$race == "Black.", 1, 0),
                                     ifelse((proxy_medicine_genetic$Yhat == 1) & (proxy_medicine_genetic$Y == 1), 1, 0))
Wbar_U_01_Black = compute_Wbar_U_hyy(primary_index,
                                     proxy_medicine_genetic[, "Black_prob"], proxy_medicine_genetic[, "py1yhat0"],
                                     ifelse(proxy_medicine_genetic$race == "Black.", 1, 0),
                                     ifelse((proxy_medicine_genetic$Yhat == 0) & (proxy_medicine_genetic$Y == 1), 1, 0))
mu_Black_WL_WU_11 = (truncate(Wbar_L_11_Black)/(truncate(Wbar_L_11_Black) + truncate(Wbar_U_01_Black)))
mu_Black_WU_WL_11 = (truncate(Wbar_U_11_Black)/(truncate(Wbar_U_11_Black) + truncate(Wbar_L_01_Black)))
mu_Black_WL_WU_01 = truncate(Wbar_L_01_Black)/(truncate(Wbar_L_01_Black) + truncate(Wbar_U_11_Black))
mu_Black_WU_WL_01 = truncate(Wbar_U_01_Black)/(truncate(Wbar_U_01_Black) + truncate(Wbar_L_11_Black))

Wbar_L_11_Other = compute_Wbar_L_hyy(primary_index,
                                     1 - proxy_medicine_genetic[, "Black_prob"], proxy_medicine_genetic[, "py1yhat1"],
                                     1 - ifelse(proxy_medicine_genetic$race == "Black.", 1, 0),
                                     ifelse((proxy_medicine_genetic$Yhat == 1) & (proxy_medicine_genetic$Y == 1), 1, 0))
Wbar_L_01_Other = compute_Wbar_L_hyy(primary_index,
                                     1 - proxy_medicine_genetic[, "Black_prob"], proxy_medicine_genetic[, "py1yhat0"],
                                     1- ifelse(proxy_medicine_genetic$race == "Black.", 1, 0),
                                     ifelse((proxy_medicine_genetic$Yhat == 0) & (proxy_medicine_genetic$Y == 1), 1, 0))
Wbar_U_11_Other = compute_Wbar_U_hyy(primary_index,
                                     1 - proxy_medicine_genetic[, "Black_prob"], proxy_medicine_genetic[, "py1yhat1"],
                                     1 - ifelse(proxy_medicine_genetic$race == "Black.", 1, 0),
                                     ifelse((proxy_medicine_genetic$Yhat == 1) & (proxy_medicine_genetic$Y == 1), 1, 0))
Wbar_U_01_Other = compute_Wbar_U_hyy(primary_index,
                                     1 - proxy_medicine_genetic[, "Black_prob"], proxy_medicine_genetic[, "py1yhat0"],
                                     1 - ifelse(proxy_medicine_genetic$race == "Black.", 1, 0),
                                     ifelse((proxy_medicine_genetic$Yhat == 0) & (proxy_medicine_genetic$Y == 1), 1, 0))
mu_Other_WL_WU_11 = (truncate(Wbar_L_11_Other)/(truncate(Wbar_L_11_Other) + truncate(Wbar_U_01_Other)))
mu_Other_WU_WL_11 = (truncate(Wbar_U_11_Other)/(truncate(Wbar_U_11_Other) + truncate(Wbar_L_01_Other)))
mu_Other_WL_WU_01 = truncate(Wbar_L_01_Other)/(truncate(Wbar_L_01_Other) + truncate(Wbar_U_11_Other))
mu_Other_WU_WL_01 = truncate(Wbar_U_01_Other)/(truncate(Wbar_U_01_Other) + truncate(Wbar_L_11_Other))

bound_df_medicine_genetic$PI_lower[bound_df_medicine_genetic$race == "Black-vs-Rest"] = mu_Black_WL_WU_11 - mu_Other_WU_WL_11
bound_df_medicine_genetic$PI_upper[bound_df_medicine_genetic$race == "Black-vs-Rest"] = mu_Black_WU_WL_11 - mu_Other_WL_WU_11

consta_UL = mu_Black_WU_WL_01/(Wbar_L_11_Black + Wbar_U_01_Black)
consta_LU = mu_Black_WL_WU_11/(Wbar_L_11_Black + Wbar_U_01_Black)
constb_UL = mu_Other_WU_WL_11/(Wbar_U_11_Other + Wbar_L_01_Other)
constb_LU = mu_Other_WL_WU_01/(Wbar_U_11_Other + Wbar_L_01_Other)

VL = compute_VL(r, primary_index,
                consta_UL, consta_LU, constb_UL, constb_LU,
                proxy_medicine_genetic[, "Black_prob"], 1 - proxy_medicine_genetic[, "Black_prob"],
                ifelse(proxy_medicine_genetic$race == "Black.", 1, 0),
                1 - ifelse(proxy_medicine_genetic$race == "Black.", 1, 0),
                proxy_medicine_genetic[, "py1yhat1"], proxy_medicine_genetic[, "py1yhat0"],
                ifelse((proxy_medicine_genetic$Yhat == 1) & (proxy_medicine_genetic$Y == 1), 1, 0),
                ifelse((proxy_medicine_genetic$Yhat == 0) & (proxy_medicine_genetic$Y == 1), 1, 0))

consta_UL = mu_Black_WU_WL_11/(Wbar_U_11_Black + Wbar_L_01_Black)
consta_LU = mu_Black_WL_WU_01/(Wbar_U_11_Black + Wbar_L_01_Black)
constb_UL = mu_Other_WU_WL_01/(Wbar_L_11_Other + Wbar_U_01_Other)
constb_LU = mu_Other_WL_WU_11/(Wbar_L_11_Other + Wbar_U_01_Other)

VU = compute_VU(r, primary_index,
                consta_UL, consta_LU, constb_UL, constb_LU,
                proxy_medicine_genetic[, "Black_prob"], 1 - proxy_medicine_genetic[, "Black_prob"],
                ifelse(proxy_medicine_genetic$race == "Black.", 1, 0),
                1 - ifelse(proxy_medicine_genetic$race == "Black.", 1, 0),
                proxy_medicine_genetic[, "py1yhat1"], proxy_medicine_genetic[, "py1yhat0"],
                ifelse((proxy_medicine_genetic$Yhat == 1) & (proxy_medicine_genetic$Y == 1), 1, 0),
                ifelse((proxy_medicine_genetic$Yhat == 0) & (proxy_medicine_genetic$Y == 1), 1, 0))

bound_df_medicine_genetic$CI_lower[bound_df_medicine_genetic$race == "Black-vs-Rest"] =
  mu_Black_WL_WU_11 - mu_Other_WU_WL_11 - qnorm(1 - 0.025) * sqrt(VL)/sqrt(nrow(proxy_medicine_genetic)/2)
bound_df_medicine_genetic$CI_upper[bound_df_medicine_genetic$race == "Black-vs-Rest"] =
  mu_Black_WU_WL_11 - mu_Other_WL_WU_11 + qnorm(1 - 0.025) * sqrt(VU)/sqrt(nrow(proxy_medicine_genetic)/2)

### Asian vs rest
Wbar_L_11_Asian = compute_Wbar_L_hyy(primary_index,
                                     proxy_medicine_genetic[, "Asian_prob"], proxy_medicine_genetic[, "py1yhat1"],
                                     ifelse(proxy_medicine_genetic$race == "Asian.", 1, 0),
                                     ifelse((proxy_medicine_genetic$Yhat == 1) & (proxy_medicine_genetic$Y == 1), 1, 0))
Wbar_L_01_Asian = compute_Wbar_L_hyy(primary_index,
                                     proxy_medicine_genetic[, "Asian_prob"], proxy_medicine_genetic[, "py1yhat0"],
                                     ifelse(proxy_medicine_genetic$race == "Asian.", 1, 0),
                                     ifelse((proxy_medicine_genetic$Yhat == 0) & (proxy_medicine_genetic$Y == 1), 1, 0))
Wbar_U_11_Asian = compute_Wbar_U_hyy(primary_index,
                                     proxy_medicine_genetic[, "Asian_prob"], proxy_medicine_genetic[, "py1yhat1"],
                                     ifelse(proxy_medicine_genetic$race == "Asian.", 1, 0),
                                     ifelse((proxy_medicine_genetic$Yhat == 1) & (proxy_medicine_genetic$Y == 1), 1, 0))
Wbar_U_01_Asian = compute_Wbar_U_hyy(primary_index,
                                     proxy_medicine_genetic[, "Asian_prob"], proxy_medicine_genetic[, "py1yhat0"],
                                     ifelse(proxy_medicine_genetic$race == "Asian.", 1, 0),
                                     ifelse((proxy_medicine_genetic$Yhat == 0) & (proxy_medicine_genetic$Y == 1), 1, 0))
mu_Asian_WL_WU_11 = (truncate(Wbar_L_11_Asian)/(truncate(Wbar_L_11_Asian) + truncate(Wbar_U_01_Asian)))
mu_Asian_WU_WL_11 = (truncate(Wbar_U_11_Asian)/(truncate(Wbar_U_11_Asian) + truncate(Wbar_L_01_Asian)))
mu_Asian_WL_WU_01 = truncate(Wbar_L_01_Asian)/(truncate(Wbar_L_01_Asian) + truncate(Wbar_U_11_Asian))
mu_Asian_WU_WL_01 = truncate(Wbar_U_01_Asian)/(truncate(Wbar_U_01_Asian) + truncate(Wbar_L_11_Asian))

Wbar_L_11_Other = compute_Wbar_L_hyy(primary_index,
                                     1 - proxy_medicine_genetic[, "Asian_prob"], proxy_medicine_genetic[, "py1yhat1"],
                                     1 - ifelse(proxy_medicine_genetic$race == "Asian.", 1, 0),
                                     ifelse((proxy_medicine_genetic$Yhat == 1) & (proxy_medicine_genetic$Y == 1), 1, 0))
Wbar_L_01_Other = compute_Wbar_L_hyy(primary_index,
                                     1 - proxy_medicine_genetic[, "Asian_prob"], proxy_medicine_genetic[, "py1yhat0"],
                                     1- ifelse(proxy_medicine_genetic$race == "Asian.", 1, 0),
                                     ifelse((proxy_medicine_genetic$Yhat == 0) & (proxy_medicine_genetic$Y == 1), 1, 0))
Wbar_U_11_Other = compute_Wbar_U_hyy(primary_index,
                                     1 - proxy_medicine_genetic[, "Asian_prob"], proxy_medicine_genetic[, "py1yhat1"],
                                     1 - ifelse(proxy_medicine_genetic$race == "Asian.", 1, 0),
                                     ifelse((proxy_medicine_genetic$Yhat == 1) & (proxy_medicine_genetic$Y == 1), 1, 0))
Wbar_U_01_Other = compute_Wbar_U_hyy(primary_index,
                                     1 - proxy_medicine_genetic[, "Asian_prob"], proxy_medicine_genetic[, "py1yhat0"],
                                     1 - ifelse(proxy_medicine_genetic$race == "Asian.", 1, 0),
                                     ifelse((proxy_medicine_genetic$Yhat == 0) & (proxy_medicine_genetic$Y == 1), 1, 0))
mu_Other_WL_WU_11 = (truncate(Wbar_L_11_Other)/(truncate(Wbar_L_11_Other) + truncate(Wbar_U_01_Other)))
mu_Other_WU_WL_11 = (truncate(Wbar_U_11_Other)/(truncate(Wbar_U_11_Other) + truncate(Wbar_L_01_Other)))
mu_Other_WL_WU_01 = truncate(Wbar_L_01_Other)/(truncate(Wbar_L_01_Other) + truncate(Wbar_U_11_Other))
mu_Other_WU_WL_01 = truncate(Wbar_U_01_Other)/(truncate(Wbar_U_01_Other) + truncate(Wbar_L_11_Other))

bound_df_medicine_genetic$PI_lower[bound_df_medicine_genetic$race == "Asian-vs-Rest"] = mu_Asian_WL_WU_11 - mu_Other_WU_WL_11
bound_df_medicine_genetic$PI_upper[bound_df_medicine_genetic$race == "Asian-vs-Rest"] = mu_Asian_WU_WL_11 - mu_Other_WL_WU_11

consta_UL = mu_Asian_WU_WL_01/(Wbar_L_11_Asian + Wbar_U_01_Asian)
consta_LU = mu_Asian_WL_WU_11/(Wbar_L_11_Asian + Wbar_U_01_Asian)
constb_UL = mu_Other_WU_WL_11/(Wbar_U_11_Other + Wbar_L_01_Other)
constb_LU = mu_Other_WL_WU_01/(Wbar_U_11_Other + Wbar_L_01_Other)

VL = compute_VL(r, primary_index,
                consta_UL, consta_LU, constb_UL, constb_LU,
                proxy_medicine_genetic[, "Asian_prob"], 1 - proxy_medicine_genetic[, "Asian_prob"],
                ifelse(proxy_medicine_genetic$race == "Asian.", 1, 0),
                1 - ifelse(proxy_medicine_genetic$race == "Asian.", 1, 0),
                proxy_medicine_genetic[, "py1yhat1"], proxy_medicine_genetic[, "py1yhat0"],
                ifelse((proxy_medicine_genetic$Yhat == 1) & (proxy_medicine_genetic$Y == 1), 1, 0),
                ifelse((proxy_medicine_genetic$Yhat == 0) & (proxy_medicine_genetic$Y == 1), 1, 0))

consta_UL = mu_Asian_WU_WL_11/(Wbar_U_11_Asian + Wbar_L_01_Asian)
consta_LU = mu_Asian_WL_WU_01/(Wbar_U_11_Asian + Wbar_L_01_Asian)
constb_UL = mu_Other_WU_WL_01/(Wbar_L_11_Other + Wbar_U_01_Other)
constb_LU = mu_Other_WL_WU_11/(Wbar_L_11_Other + Wbar_U_01_Other)

VU = compute_VU(r, primary_index,
                consta_UL, consta_LU, constb_UL, constb_LU,
                proxy_medicine_genetic[, "Asian_prob"], 1 - proxy_medicine_genetic[, "Asian_prob"],
                ifelse(proxy_medicine_genetic$race == "Asian.", 1, 0),
                1 - ifelse(proxy_medicine_genetic$race == "Asian.", 1, 0),
                proxy_medicine_genetic[, "py1yhat1"], proxy_medicine_genetic[, "py1yhat0"],
                ifelse((proxy_medicine_genetic$Yhat == 1) & (proxy_medicine_genetic$Y == 1), 1, 0),
                ifelse((proxy_medicine_genetic$Yhat == 0) & (proxy_medicine_genetic$Y == 1), 1, 0))

bound_df_medicine_genetic$CI_lower[bound_df_medicine_genetic$race == "Asian-vs-Rest"] =
  mu_Asian_WL_WU_11 - mu_Other_WU_WL_11 - qnorm(1 - 0.025) * sqrt(VL)/sqrt(nrow(proxy_medicine_genetic)/2)
bound_df_medicine_genetic$CI_upper[bound_df_medicine_genetic$race == "Asian-vs-Rest"] =
  mu_Asian_WU_WL_11 - mu_Other_WL_WU_11 + qnorm(1 - 0.025) * sqrt(VU)/sqrt(nrow(proxy_medicine_genetic)/2)

######################
#  Truth
######################
bound_df_medicine[bound_df_medicine$race == "White-vs-Rest", "truth"] = compute_true_tprd(proxy_medicine, "White.")
bound_df_medicine[bound_df_medicine$race == "Black-vs-Rest", "truth"] = compute_true_tprd(proxy_medicine, "Black.")
bound_df_medicine[bound_df_medicine$race == "Asian-vs-Rest", "truth"] = compute_true_tprd(proxy_medicine, "Asian.")
bound_df_genetic[bound_df_genetic$race == "White-vs-Rest", "truth"] = compute_true_tprd(proxy_genetic, "White.")
bound_df_genetic[bound_df_genetic$race == "Black-vs-Rest", "truth"] = compute_true_tprd(proxy_genetic, "Black.")
bound_df_genetic[bound_df_genetic$race == "Asian-vs-Rest", "truth"] = compute_true_tprd(proxy_genetic, "Asian.")
bound_df_medicine_genetic[bound_df_medicine_genetic$race == "White-vs-Rest", "truth"] = compute_true_tprd(proxy_medicine_genetic, "White.")
bound_df_medicine_genetic[bound_df_medicine_genetic$race == "Black-vs-Rest", "truth"] = compute_true_tprd(proxy_medicine_genetic, "Black.")
bound_df_medicine_genetic[bound_df_medicine_genetic$race == "Asian-vs-Rest", "truth"] = compute_true_tprd(proxy_medicine_genetic, "Asian.")

write_csv(bound_df_medicine, "CI_medicine.csv")
write_csv(bound_df_genetic, "CI_genetic.csv")
write_csv(bound_df_medicine_genetic, "CI_medicine_genetic.csv")
