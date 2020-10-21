####################
# Processing data
####################
collaps_variables <- function(existing_var_list, new_var_list, implicit_level_list, data = warfarin){
  # this function transforms one-hot-encoding back to a single multi-valued variable
  #   existing_var: the name list of the one-hot-encoded variables in original data
  #   new_var: the name of the new single multi-valued variable after this transformation
  #   implicit_level: the value we reserve for the case when none of the existing_var takes value 1; it's usually NA
  # this function cannot collaps variables where multiple levels can occur at the same type, e.g.,
  #   comorbities, Indication.for.Warfarin.Treatment..

  new_var_value = matrix("NA", nrow(data), length(existing_var_list))

  for (i in 1:nrow(data)){

    for (j in 1:length(existing_var_list)){
      cat(paste("the ", i, " th sample for the variable ", new_var_list[[j]], "\n"))

      existing_var = existing_var_list[[j]]
      ind = which(data[i, existing_var] == 1)
      if (length(ind) > 0){
        temp = ind
        new_var_value[i, j] = existing_var[temp]
      } else {
        new_var_value[i, j] = implicit_level_list[[j]]
      }
    }

  }

  new_data = cbind(data[, -which(colnames(data) %in% unlist(existing_var_list))], new_var_value)
  n_col_for_new_var = (ncol(new_data) - ncol(new_var_value) + 1):ncol(new_data)
  colnames(new_data)[n_col_for_new_var] = unlist(new_var_list)
  new_data
}
compute_true_disparity <- function(prediction, outcome, cutoff){
  bin_pred = prediction > cutoff
  bin_outcome = outcome > cutoff

  wi = warfarin$`White.` == 1
  bi = warfarin$`Black.` == 1
  ai = warfarin$`Asian.` == 1

  result = matrix(0, 3, 2)
  rownames(result) = c("demographic", "FPR", "TPR")
  colnames(result) = c("White-Black", "White-Asian")

  # demographic disparity
  result["demographic", "White-Black"] = mean(bin_pred[wi]) - mean(bin_pred[bi]) # -0.3804933
  result["demographic", "White-Asian"] = mean(bin_pred[wi]) - mean(bin_pred[ai]) # 0.3674731
  # FPR
  result["FPR", "White-Black"] = mean(bin_pred[wi & !bin_outcome]) - mean(bin_pred[bi & !bin_outcome]) # -0.5055848
  result["FPR", "White-Asian"] = mean(bin_pred[wi & !bin_outcome]) - mean(bin_pred[ai & !bin_outcome]) # 0.2133875
  # TPR
  result["TPR", "White-Black"] = mean(bin_pred[wi & bin_outcome]) - mean(bin_pred[bi & bin_outcome]) # -0.1052938
  result["TPR", "White-Asian"] =  mean(bin_pred[wi & bin_outcome]) - mean(bin_pred[ai & bin_outcome]) # 0.3637431

  result
}
###############################
# Fitting proxy probabilities
###############################
# the following function computes the race and prediction probabilities given the proxies;
#   they serve as look up list where each component corresponds to one level of the proxy
#     whose values are the race or prediction probabilities for that level of proxy
compute_cond_prediction_prob_disc_proxy <- function(target, proxy){
  temp = cbind(target, proxy)
  est_prob = do.call(partial(group_by, .data = temp), proxy) %>%
    summarise(
      pred1_prob = mean(target == "TRUE"),
      n = n())

  table_prob = vector("list", nrow(est_prob))
  names(table_prob) = 1:nrow(est_prob)

  temp = as.data.frame(lapply(est_prob, as.character))
  for (i in 1:nrow(est_prob)){
    names(table_prob)[i] = do.call(paste, temp[i, colnames(proxy)])
    table_prob[[i]] = est_prob[i, "pred1_prob"]
  }

  table_prob
}
compute_cond_race_prob_disc_proxy <- function(race, proxy){
  # this generates the conditional race probabilities for each unique
  # combination of the proxy variables
  temp = cbind(race, proxy)
  est_prob = do.call(partial(group_by, .data = temp), proxy) %>%
    summarise(
      white_prob = mean(race == "White."),
      black_prob = mean(race == "Black."),
      asian_prob = mean(race == "Asian."),
      n = n()
    )

  table_prob = vector("list", nrow(est_prob))
  names(table_prob) = 1:nrow(est_prob)

  temp = as.data.frame(lapply(est_prob, as.character))
  for (i in 1:nrow(est_prob)){
    names(table_prob)[i] = do.call(paste, temp[i, colnames(proxy)])
    table_prob[[i]] = est_prob[i, c("white_prob", "black_prob", "asian_prob")]
  }

  table_prob
}
# the following functions generate the conditional proxy probabilities for each unit in data
apply_race_prob <- function(race_prob_table, proxy_sublist = genetic_proxy, data = warfarin2){
  prob_temp = data.frame(white_prob = rep(0, nrow(data)),
                         black_prob = rep(0, nrow(data)),
                         asian_prob = rep(0, nrow(data)))
  for (i in 1:nrow(data)){
    proxy_value = do.call(paste, data[i, proxy_sublist])
    prob_temp[i, ] =  unlist(race_prob_table[[proxy_value]])
  }
  prob_temp
}
apply_pred_prob <- function(pred_prob_table, proxy_sublist = genetic_proxy, data = warfarin2){
  prob_temp = data.frame(pred1_prob = rep(0, nrow(data)))
  for (i in 1:nrow(data)){
    proxy_value = do.call(paste, data[i, proxy_sublist])
    prob_temp[i, ] =  unlist(pred_prob_table[[proxy_value]])
  }
  prob_temp
}
#####################
# computing entropy
#####################
compute_entropy_race <- function(data){
  race_temp = data[, c("White", "Black", "API")]

  temp_white = xlogx(race_temp$White)
  temp_API = xlogx(race_temp$API)
  temp_Black = xlogx(race_temp$Black)
  -mean(temp_white + temp_API + temp_Black, na.rm =T)
}
compute_entropy_four_outcome <- function(outcome){
  -mean(xlogx(outcome$py1yhat1) + xlogx(outcome$py0yhat1) + xlogx(outcome$py1yhat0) + xlogx(outcome$py0yhat0))
}
xlogx <- function(x){
  ifelse(x == 0, 0, x*log(x))
}
#####################
# Confidence interval
#####################
compute_Wbar_L_hyy <- function(primary_index,
                               race_prob, outcome_prob,
                               race, outcome){
  # estimate the equation (35)
  ind = ifelse(outcome_prob + race_prob - 1 >= 0, 1, 0)
  lambda = c(ind * (outcome_prob + race_prob - 1))[[1]]
  xi = c(ind * (race - race_prob))[[1]]
  gamma = c(ind * (outcome - outcome_prob))[[1]]
  mean(lambda)+ mean(xi[-primary_index]) + mean(gamma[primary_index])
}
compute_Wbar_U_hyy <- function(primary_index,
                               race_prob, outcome_prob,
                               race, outcome){
  # estimate the equation (36)
  ind = ifelse(outcome_prob - race_prob <= 0, 1, 0)
  lambda = c(ind * (outcome_prob - race_prob) + race_prob)[[1]]
  xi = c((1 - ind) * (race - race_prob))[[1]]
  gamma = c(ind * (outcome - outcome_prob))[[1]]
  mean(lambda)+ mean(xi[-primary_index]) + mean(gamma[primary_index])
}
truncate <- function(x){
  if ((x >= 0) & (x <= 1)){
    x
  } else {
    if (x < 0){
      0
    } else{
      1
    }
  }
}
compute_VU <- function(r, primary_index,
                       consta_UL, consta_LU, constb_UL, constb_LU,
                       race_prob_a, race_prob_b,
                       race_a, race_b,
                       outcome_prob11, outcome_prob01,
                       outcome11, outcome01){
  # compute V_U on page 40
  ind_a_U_11 =  ifelse(outcome_prob11 - race_prob_a <= 0, 1, 0)
  ind_a_U_01 =  ifelse(outcome_prob01 - race_prob_a <= 0, 1, 0)
  ind_b_U_11 =  ifelse(outcome_prob11 - race_prob_b <= 0, 1, 0)
  ind_b_U_01 =  ifelse(outcome_prob01 - race_prob_b <= 0, 1, 0)

  ind_a_L_11 =  ifelse(outcome_prob11 + race_prob_a - 1 >= 0, 1, 0)
  ind_a_L_01 =  ifelse(outcome_prob01 + race_prob_a - 1 >= 0, 1, 0)
  ind_b_L_11 =  ifelse(outcome_prob11 + race_prob_b - 1 >= 0, 1, 0)
  ind_b_L_01 =  ifelse(outcome_prob01 + race_prob_b - 1 >= 0, 1, 0)

  lambda_a_L_11 = c(ind_a_L_11 * (outcome_prob11 + race_prob_a - 1))[[1]]
  lambda_a_L_01 = c(ind_a_L_01 * (outcome_prob01 + race_prob_a - 1))[[1]]
  lambda_b_L_11 = c(ind_b_L_11 * (outcome_prob11 + race_prob_b - 1))[[1]]
  lambda_b_L_01 = c(ind_b_L_01 * (outcome_prob01 + race_prob_b - 1))[[1]]

  lambda_a_U_11 = c(ind_a_U_11 * (outcome_prob11 - race_prob_a) + race_prob_a)[[1]]
  lambda_a_U_01 = c(ind_a_U_01 * (outcome_prob01 - race_prob_a) + race_prob_a)[[1]]
  lambda_b_U_11 = c(ind_b_U_11 * (outcome_prob11 - race_prob_b) + race_prob_b)[[1]]
  lambda_b_U_01 = c(ind_b_U_01 * (outcome_prob01 - race_prob_b) + race_prob_b)[[1]]

  xi_a_L_11 = c(ind_a_L_11 * (race_a - race_prob_a))[[1]]
  xi_a_L_01 = c(ind_a_L_01 * (race_a - race_prob_a))[[1]]
  xi_b_L_11 = c(ind_b_L_11 * (race_b - race_prob_b))[[1]]
  xi_b_L_01 = c(ind_b_L_01 * (race_b - race_prob_b))[[1]]

  xi_a_U_11 = c((1 - ind_a_U_11) * (race_a - race_prob_a))[[1]]
  xi_a_U_01 = c((1 - ind_a_U_01) * (race_a - race_prob_a))[[1]]
  xi_b_U_11 = c((1 - ind_b_U_11) * (race_b - race_prob_b))[[1]]
  xi_b_U_01 = c((1 - ind_b_U_01) * (race_b - race_prob_b))[[1]]

  gamma_a_L_11 = c(ind_a_L_11 * (outcome11 - outcome_prob11))[[1]]
  gamma_a_L_01 = c(ind_a_L_01 * (outcome01 - outcome_prob01))[[1]]
  gamma_b_L_11 = c(ind_b_L_11 * (outcome11 - outcome_prob11))[[1]]
  gamma_b_L_01 = c(ind_b_L_01 * (outcome01 - outcome_prob01))[[1]]

  gamma_a_U_11 = c(ind_a_U_11 * (outcome11 - outcome_prob11))[[1]]
  gamma_a_U_01 = c(ind_a_U_01 * (outcome01 - outcome_prob01))[[1]]
  gamma_b_U_11 = c(ind_b_U_11 * (outcome11 - outcome_prob11))[[1]]
  gamma_b_U_01 = c(ind_b_U_01 * (outcome01 - outcome_prob01))[[1]]

  term1 = ((consta_LU * lambda_a_U_11 - consta_UL * lambda_a_L_01) -
             (constb_UL * lambda_b_L_11 - constb_LU * lambda_b_U_01))^2
  term2 = ((consta_LU * gamma_a_U_11 - consta_UL * gamma_a_L_01) -
             (constb_UL * gamma_b_L_11 - constb_LU * gamma_b_U_01))^2
  term3 = ((consta_LU * xi_a_U_11 - consta_UL * xi_a_L_01) -
             (constb_UL * xi_b_L_11 - constb_LU * xi_b_U_01))^2
  r * mean(term1) + mean(term2[primary_index]) + r/(1 - r) * mean(term3[-primary_index])
}
compute_VL <- function(r, primary_index,
                       consta_UL, consta_LU, constb_UL, constb_LU,
                       race_prob_a, race_prob_b,
                       race_a, race_b,
                       outcome_prob11, outcome_prob01,
                       outcome11, outcome01){
  # compute V_L on page 40

  ind_a_U_11 =  ifelse(outcome_prob11 - race_prob_a <= 0, 1, 0)
  ind_a_U_01 =  ifelse(outcome_prob01 - race_prob_a <= 0, 1, 0)
  ind_b_U_11 =  ifelse(outcome_prob11 - race_prob_b <= 0, 1, 0)
  ind_b_U_01 =  ifelse(outcome_prob01 - race_prob_b <= 0, 1, 0)

  ind_a_L_11 =  ifelse(outcome_prob11 + race_prob_a - 1 >= 0, 1, 0)
  ind_a_L_01 =  ifelse(outcome_prob01 + race_prob_a - 1 >= 0, 1, 0)
  ind_b_L_11 =  ifelse(outcome_prob11 + race_prob_b - 1 >= 0, 1, 0)
  ind_b_L_01 =  ifelse(outcome_prob01 + race_prob_b - 1 >= 0, 1, 0)

  lambda_a_L_11 = c(ind_a_L_11 * (outcome_prob11 + race_prob_a - 1))[[1]]
  lambda_a_L_01 = c(ind_a_L_01 * (outcome_prob01 + race_prob_a - 1))[[1]]
  lambda_b_L_11 = c(ind_b_L_11 * (outcome_prob11 + race_prob_b - 1))[[1]]
  lambda_b_L_01 = c(ind_b_L_01 * (outcome_prob01 + race_prob_b - 1))[[1]]

  lambda_a_U_11 = c(ind_a_U_11 * (outcome_prob11 - race_prob_a) + race_prob_a)[[1]]
  lambda_a_U_01 = c(ind_a_U_01 * (outcome_prob01 - race_prob_a) + race_prob_a)[[1]]
  lambda_b_U_11 = c(ind_b_U_11 * (outcome_prob11 - race_prob_b) + race_prob_b)[[1]]
  lambda_b_U_01 = c(ind_b_U_01 * (outcome_prob01 - race_prob_b) + race_prob_b)[[1]]

  xi_a_L_11 = c(ind_a_L_11 * (race_a - race_prob_a))[[1]]
  xi_a_L_01 = c(ind_a_L_01 * (race_a - race_prob_a))[[1]]
  xi_b_L_11 = c(ind_b_L_11 * (race_b - race_prob_b))[[1]]
  xi_b_L_01 = c(ind_b_L_01 * (race_b - race_prob_b))[[1]]

  xi_a_U_11 = c((1 - ind_a_U_11) * (race_a - race_prob_a))[[1]]
  xi_a_U_01 = c((1 - ind_a_U_01) * (race_a - race_prob_a))[[1]]
  xi_b_U_11 = c((1 - ind_b_U_11) * (race_b - race_prob_b))[[1]]
  xi_b_U_01 = c((1 - ind_b_U_01) * (race_b - race_prob_b))[[1]]

  gamma_a_L_11 = c(ind_a_L_11 * (outcome11 - outcome_prob11))[[1]]
  gamma_a_L_01 = c(ind_a_L_01 * (outcome01 - outcome_prob01))[[1]]
  gamma_b_L_11 = c(ind_b_L_11 * (outcome11 - outcome_prob11))[[1]]
  gamma_b_L_01 = c(ind_b_L_01 * (outcome01 - outcome_prob01))[[1]]

  gamma_a_U_11 = c(ind_a_U_11 * (outcome11 - outcome_prob11))[[1]]
  gamma_a_U_01 = c(ind_a_U_01 * (outcome01 - outcome_prob01))[[1]]
  gamma_b_U_11 = c(ind_b_U_11 * (outcome11 - outcome_prob11))[[1]]
  gamma_b_U_01 = c(ind_b_U_01 * (outcome01 - outcome_prob01))[[1]]

  term1 = ((consta_UL * lambda_a_L_11 - consta_LU * lambda_a_U_01) -
             (constb_LU * lambda_b_U_11 - constb_UL * lambda_b_L_01))^2
  term2 = ((consta_UL * gamma_a_L_11 - consta_LU * gamma_a_U_01) -
             (constb_LU * gamma_b_U_11 - constb_UL * gamma_b_L_01))^2
  term3 = ((consta_UL * xi_a_L_11 - consta_LU * xi_a_U_01) -
             (constb_LU * xi_b_U_11 - constb_UL * xi_b_L_01))^2
  r * mean(term1) + mean(term2[primary_index]) + r/(1 - r) * mean(term3[-primary_index])
}
compute_true_tprd <- function(dataset, race){
  colMeans(dataset[(dataset$race == race) & (dataset$Y == 1), "Yhat"]) -
    colMeans(dataset[(dataset$race != race) & (dataset$Y == 1), "Yhat"])
}
