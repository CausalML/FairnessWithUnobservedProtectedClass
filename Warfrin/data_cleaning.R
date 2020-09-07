library(glmnet)
library(tidyverse)
library(utils)

source("warfarin_helper.R")
warfarin = read_csv("warfarin_clean.csv")[, -1]

########################
# Transforming data
########################
# transforming the data in multiple-valued data from one-hot-encoding data;
# see collaps_variables() in warfarin_helper.R for the meaning of existing_var_list,
# new_var_list, implicit_level_list
existing_var_list = list(c("Male?", "Female?"),
                         c("Asian.", "Black.", "White."),
                         paste0("Diabetes=", c(0,1), '?'),
                         paste0("Congestive Heart Failure and/or Cardiomyopathy=", c(0,1), "?"),
                         paste0("Valve Replacement=", c(0, 1), "?"),
                         paste0("aspirin=", c(0, 1), "?"),
                         paste0("Acetaminophen=", c(0, 1), "?"),
                         paste0("Acetaminophen hi dose=", c(0, 1), "?"),
                         paste0("Simvastatin=", c(0, 1), "?"),
                         paste0("Atorvastatin=", c(0, 1), "?"),
                         paste0("Fluvastatin=", c(0, 1), "?"),
                         paste0("Lovastatin=", c(0, 1), "?"),
                         paste0("Pravastatin=", c(0, 1), "?"),
                         paste0("Rosuvastatin=", c(0, 1), "?"),
                         paste0("Cerivastatin=", c(0, 1), "?"),
                         paste0("Amiodarone=", c(0, 1), "?"),
                         paste0("Carbamazepine=", c(0, 1), "?"),
                         paste0("Phenytoin=", c(0, 1), "?"),
                         paste0("Rifampin=", c(0, 1), "?"),
                         paste0("Sulfonamide Antibiotics=", c(0, 1), "?"),
                         paste0("Macrolide Antibiotics=", c(0, 1), "?"),
                         paste0("Anti-fungal Azoles=", c(0, 1), "?"),
                         paste0("Herbal Medications, Vitamins, Supplements=", c(0, 1), "?"),
                         paste0("Smoker=", c(0, 1), "?"),
                         paste0("CYP2C9 ", c("*1/*1", "*1/*2", "*1/*3", "NA")),
                         paste0("VKORC1 -4451 ", c("C/C", "A/C", "A/A")),
                         paste0("VKORC1 2255 ", c("C/C", "C/T", "T/T")),
                         paste0("VKORC1 3730 ", c("A/A", "A/G", "G/G")),
                         paste0("VKORC1 1542 ", c("C/C", "C/G", "G/G")),
                         paste0("VKORC1 1173 ", c("T/T", "C/T", "C/C")),
                         paste0("VKORC1 497 ", c("T/T", "G/T", "G/G")),
                         paste0("VKORC1 -1639 ", c("A/A", "A/G", "G/G"))
)
new_var_list = list("gender", "race", "Diabetes", "Congestive.Heart.Failure.and.or.Cardiomyopathy.",
                    "Valve.Replacement.", "aspirin.", "Acetaminophen.", "Acetaminophen.hi.dose.",
                    "Simvastatin.", "Atorvastatin.", "Fluvastatin.",
                    "Lovastatin.", "Pravastatin.", "Rosuvastatin.", "Cerivastatin.",
                    "Amiodarone.", "Carbamazepine.", "Phenytoin.", "Rifampin.",
                    "Sulfonamide.Antibiotics.", "Macrolide.Antibiotics.", "Anti.fungal.Azoles.",
                    "Herbal.Medications..Vitamins..Supplements.", "Smoker.",
                    "CYP2C9.", "VKORC1..4451.", "VKORC1.2255.", "VKORC1.3730.",
                    "VKORC1.1542.", "VKORC1.1173.", "VKORC1.497.", "VKORC1..1639.")
implicit_level_list = lapply(1:length(existing_var_list), function(x) "NA")

warfarin2 = collaps_variables(existing_var_list, new_var_list,
                              implicit_level_list, data = warfarin)
warfarin2 = as_tibble(warfarin2)
write_csv(warfarin2, "multi-valued-data.csv")

# generating the decision Yhat according to a linear model
lr_model = lm(therapeut_dose ~ . - `Black.` - `Asian.` - `White.` - `Male?`, data = warfarin)
lr_pred_no_race = predict(lr_model)
Yhat = lr_pred_no_race > 35
Y = warfarin$therapeut_dose > 35
# compute the disparity w.r.t true race
disparity_lr_no_race = compute_true_disparity(lr_pred_no_race, warfarin$therapeut_dose, cutoff = 35)

#######################
# Generating proxies
#######################
not_proxy_list = c("therapeut_dose", "Age group", "No age?",  "Height", "Weight", "gender", "race")
proxy_list = colnames(warfarin2)[!colnames(warfarin2) %in% not_proxy_list]
medicine_proxy = proxy_list[12:29]
genetic_proxy = proxy_list[31:38]


##### Both medicine and genetic as proxies
proxy_sublist = c(medicine_proxy, genetic_proxy)
proxy = warfarin2[, proxy_sublist]
# estimates the conditional race probabilities for each unique
    # combination of the proxy variables
race_prob_table = compute_cond_race_prob_disc_proxy(race = warfarin2$race,
                                                   proxy = proxy)
# compute the conditional race probabilities for each units in data
race_prob = apply_race_prob(race_prob_table, proxy_sublist = proxy_sublist, data = warfarin2)
# do the same for outcomes
py1yhat1_table = compute_cond_prediction_prob_disc_proxy(target = Yhat & Y,
                                                         proxy = proxy)
py0yhat1_table = compute_cond_prediction_prob_disc_proxy(target = Yhat & !Y,
                                                         proxy = proxy)
py1yhat0_table = compute_cond_prediction_prob_disc_proxy(target = !Yhat & Y,
                                                         proxy = proxy)
py0yhat0_table = compute_cond_prediction_prob_disc_proxy(target = !Yhat & !Y,
                                                         proxy = proxy)
py1yhat1 = apply_pred_prob(py1yhat1_table, proxy_sublist = proxy_sublist, data = warfarin2)
py0yhat1 = apply_pred_prob(py0yhat1_table, proxy_sublist = proxy_sublist, data = warfarin2)
py1yhat0 = apply_pred_prob(py1yhat0_table, proxy_sublist = proxy_sublist, data = warfarin2)
py0yhat0 = apply_pred_prob(py0yhat0_table, proxy_sublist = proxy_sublist, data = warfarin2)
# generate the final data
final = data.frame(Y, Yhat, race_prob, py1yhat1, py0yhat1, py1yhat0, py0yhat0, proxy, warfarin2$race)
colnames(final) = c("Y", "Yhat", colnames(race_prob),
                    c("py1yhat1", "py0yhat1", "py1yhat0", "py0yhat0"), colnames(proxy), "race")
write_csv(final, "medicine_genetic_as_proxy.csv")

##### only medicine as proxies
proxy = warfarin2[, medicine_proxy]
# estimates the conditional race probabilities for each unique
# combination of the proxy variables
race_prob_table = compute_cond_race_prob_disc_proxy(race = warfarin2$race,
                                                    proxy = proxy)
# compute the conditional race probabilities for each units in data
race_prob = apply_race_prob(race_prob_table, proxy_sublist = medicine_proxy, data = warfarin2)
# do the same for outcomes
py1yhat1_table = compute_cond_prediction_prob_disc_proxy(target = Yhat & Y,
                                                         proxy = proxy)
py0yhat1_table = compute_cond_prediction_prob_disc_proxy(target = Yhat & !Y,
                                                         proxy = proxy)
py1yhat0_table = compute_cond_prediction_prob_disc_proxy(target = !Yhat & Y,
                                                         proxy = proxy)
py0yhat0_table = compute_cond_prediction_prob_disc_proxy(target = !Yhat & !Y,
                                                         proxy = proxy)
py1yhat1 = apply_pred_prob(py1yhat1_table, proxy_sublist = medicine_proxy, data = warfarin2)
py0yhat1 = apply_pred_prob(py0yhat1_table, proxy_sublist = medicine_proxy, data = warfarin2)
py1yhat0 = apply_pred_prob(py1yhat0_table, proxy_sublist = medicine_proxy, data = warfarin2)
py0yhat0 = apply_pred_prob(py0yhat0_table, proxy_sublist = medicine_proxy, data = warfarin2)
# generate the final data
final = data.frame(Y, Yhat, race_prob, py1yhat1, py0yhat1, py1yhat0, py0yhat0, proxy, warfarin2$race)
colnames(final) = c("Y", "Yhat", colnames(race_prob),
                    c("py1yhat1", "py0yhat1", "py1yhat0", "py0yhat0"), colnames(proxy), "race")
write_csv(final, "medicine_as_proxy.csv")


##### only genetic as proxies
proxy = warfarin2[, genetic_proxy]
# estimates the conditional race probabilities for each unique
# combination of the proxy variables
race_prob_table = compute_cond_race_prob_disc_proxy(race = warfarin2$race,
                                                    proxy = proxy)
# compute the conditional race probabilities for each units in data
race_prob = apply_race_prob(race_prob_table, proxy_sublist = genetic_proxy, data = warfarin2)
# do the same for outcomes
py1yhat1_table = compute_cond_prediction_prob_disc_proxy(target = Yhat & Y,
                                                         proxy = proxy)
py0yhat1_table = compute_cond_prediction_prob_disc_proxy(target = Yhat & !Y,
                                                         proxy = proxy)
py1yhat0_table = compute_cond_prediction_prob_disc_proxy(target = !Yhat & Y,
                                                         proxy = proxy)
py0yhat0_table = compute_cond_prediction_prob_disc_proxy(target = !Yhat & !Y,
                                                         proxy = proxy)
py1yhat1 = apply_pred_prob(py1yhat1_table, proxy_sublist = genetic_proxy, data = warfarin2)
py0yhat1 = apply_pred_prob(py0yhat1_table, proxy_sublist = genetic_proxy, data = warfarin2)
py1yhat0 = apply_pred_prob(py1yhat0_table, proxy_sublist = genetic_proxy, data = warfarin2)
py0yhat0 = apply_pred_prob(py0yhat0_table, proxy_sublist = genetic_proxy, data = warfarin2)
# generate the final data
final = data.frame(Y, Yhat, race_prob, py1yhat1, py0yhat1, py1yhat0, py0yhat0, proxy, warfarin2$race)
colnames(final) = c("Y", "Yhat", colnames(race_prob),
                    c("py1yhat1", "py0yhat1", "py1yhat0", "py0yhat0"), colnames(proxy), "race")
write_csv(final, "genetic_as_proxy.csv")
