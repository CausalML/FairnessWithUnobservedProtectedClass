library(tidyverse)
library(mlogit)
library(mgcv)


hmda = read_csv("clean_hmda.csv")
hmda = hmda %>% mutate(id = 1:nrow(hmda))

counties = hmda %>% group_by(county_id) %>% summarise(n = n()) # all counties
n_counties = nrow(counties)

# both county and income as proxy:
#   here for each county,
#     we fit a logistic regression of outcome against income
#     we fit a multinomial logistic regression of race against income
#   these estimate the conditional outcome probability and conditional race probability
#     with both income and county as proxy using the whole HMDA data

print("fitting proxy county and income")
time1 = Sys.time()
proxy_county_income = foreach(i = 1:n_counties, .packages = c("dplyr", "mlogit")) %do% {

  tryCatch({
    temp_data = hmda %>% filter(county_id == i)
    id = temp_data$id

    # fit outcome probabilities
    lfit = glm(outcome ~ applicant_income_000s, data = temp_data, family=binomial(link='logit'))
    yhat1_prob = predict(lfit, type = "response")
    # fit race probabilities
    race_table = table(temp_data$race)
    if (length(race_table) == 1){
      race_prob_temp = data.frame(White = rep(0, nrow(temp_data)), Black = rep(0, nrow(temp_data)),
                                  API = rep(0, nrow(temp_data)))
      race_prob_temp[, names(race_table)] = 1
    }
    if (length(race_table) > 1){
      temp_data2 = mlogit.data(temp_data, choice = "race", varying = NULL, shape = "wide")
      mfit = mlogit(race ~ 0 | applicant_income_000s, reflevel="White", data=temp_data2)
      fitted_temp = fitted(mfit, type = "probabilities")
      race_prob_temp = data.frame(White = rep(0, nrow(temp_data)), Black = rep(0, nrow(temp_data)),
                                  API = rep(0, nrow(temp_data)))
      race_prob_temp[, names(race_table)] = fitted_temp[,  names(race_table)]
    }
    temp_output = data.frame(yhat1 = yhat1_prob, White = race_prob_temp[, "White"], Black = race_prob_temp[, "Black"],
                             API = race_prob_temp[, "API"])
    temp_output$id = id
    temp_output$race = temp_data$race
    temp_output$county_id = temp_data$county_id
    temp_output$outcome = temp_data$outcome
    temp_output$applicant_income_000s = temp_data$applicant_income_000s

    temp_output
  },  error = function(e) return(paste0("'", e, "'", " and id is ", id[1])))
}
Sys.time() - time1

proxy_county_income = do.call(rbind, proxy_county_income)
proxy_county_income = as_tibble(proxy_county_income)
proxy_county_income$White = round(proxy_county_income$White, 4)
proxy_county_income$Black = round(proxy_county_income$Black, 4)
proxy_county_income$API = round(proxy_county_income$API, 4)
proxy_county_income$yhat1 = round(proxy_county_income$yhat1, 4)
write_csv(proxy_county_income, "proxy_county_income.csv")


# proxy prob with only income:
#   run logistic regression of outcome against income, and multinomial logistic regression
# of race against income directly
print("fitting proxy income")
hmda_temp = hmda
lfit = glm(outcome ~ applicant_income_000s, data = hmda_temp, family=binomial(link='logit'))
yhat1_prob = predict(lfit, type = "response")
temp_data = mlogit.data(hmda_temp, choice = "race", varying = NULL, shape = "wide")
mfit = mlogit(race ~ 0 | applicant_income_000s, reflevel="White", data=temp_data)
race_prob = fitted(mfit, type = "probabilities")
income_proxy =  data.frame(id = hmda_temp$id, county_id = hmda_temp$county_id,
                           outcome = hmda_temp$outcome, race = hmda_temp$race,
                           yhat1 = yhat1_prob, race_prob, applicant_income_000s = hmda_temp$applicant_income_000s)
income_proxy = as_tibble(income_proxy)
income_proxy$White = round(income_proxy$White, 4)
income_proxy$Black = round(income_proxy$Black, 4)
income_proxy$API = round(income_proxy$API, 4)
income_proxy$yhat1 = round(income_proxy$yhat1, 4)
write_csv(income_proxy, "proxy_income.csv")

# proxy prob with only county
print("fitting proxy county")
county_proxy = hmda %>% mutate(yhat1 = nyhat1/n,
                               White = nwhite/n,
                               Black = nblack/n,
                               API = napi/n) %>%
  select(county_id, yhat1, White,  Black, API, race, outcome, id, applicant_income_000s)
county_proxy$White = round(county_proxy$White, 4)
county_proxy$Black = round(county_proxy$Black, 4)
county_proxy$API = round(county_proxy$API, 4)
county_proxy$yhat1 = round(county_proxy$yhat1, 4)
write_csv(county_proxy, "proxy_county.csv")


proxy_county = read_csv("proxy_county.csv")
proxy_income = read_csv("proxy_income.csv")
proxy_county_income = read_csv("proxy_county_income.csv")

# sampling a smaller sample
sample_index = county_proxy %>%
  group_by(race, outcome) %>%
  sample_frac(0.01)  %>% select(race, outcome, id)

small_county = proxy_county[sample_index$id, ]
small_income = proxy_income[sample_index$id, ]
small_county_income = proxy_county_income[sample_index$id, ]

write_csv(small_county, "small_proxy_county.csv")
write_csv(small_county_income, "small_proxy_county_income.csv")
write_csv(small_income, "small_proxy_income.csv")

