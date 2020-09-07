library(tidyverse)
library(foreign)
library(totalcensus)
source("HMDA_helper.R")

# We use the HMDA 2011-2012 data following the CFPB's technical report: https://files.consumerfinance.gov/f/201409_cfpb_report_proxy-methodology.pdf
#
# The data can be downloaded from https://www.consumerfinance.gov/data-research/hmda/explore. We query the following observations and variables:
#
# - where: (applicant_ethnicity=1 OR applicant_ethnicity=2) AND (applicant_race_1=1 OR applicant_race_1=2 OR applicant_race_1=3 OR applicant_race_1=4 OR applicant_race_1=5) AND (action_taken=1 OR action_taken=2 OR action_taken=3) AND (as_of_year=2011 OR as_of_year=2012)
# - select: action_taken_name, applicant_ethnicity_name, applicant_income_000s, applicant_race_name_1, applicant_race_name_2, as_of_year, county_name, state_code
# - See https://cfpb.github.io/api/hmda/fields.html for the meaning of the variables.
# - Copy paste the following link to the browser and it will start downloading the data (it takes a while to process the query and start downloading):
# https://api.consumerfinance.gov/data/hmda/slice/hmda_lar.csv?&$where=as_of_year+IN+(2012,2011)+AND+action_taken+IN+(1,2,3)+AND+applicant_race_1+IN+(1,2,3,4,5)+AND+applicant_ethnicity+IN+(1,2)&$select=action_taken_name,%20applicant_ethnicity_name,%20applicant_income_000s,%20applicant_race_name_1,%20applicant_race_name_2,%20as_of_year,%20county_name,%20state_code&$limit=0
# The data is around 2G.

hmda = read_csv("hmda_lar.csv")
hmda = hmda %>% select(state_code, county_name, applicant_ethnicity_name, applicant_race_name_1, applicant_race_name_2, action_taken_name, applicant_income_000s)

hmda <- hmda %>%
  remove_obs(get_obs_state_missing) %>%  # remove observations whose state_code is missing
  remove_obs(get_obs_county_missing) %>%  # remove observations whose county is missing
  make_race() %>%   # generate the race label based on applicant self-reported ethnicity and race1, race 2 (coapplicant is ignored)
  remove_obs(get_obs_race_empty) # remove observations whose race is empty, i.e., ""
# construct outcome label
hmda$outcome = 1
hmda$outcome[hmda$action_taken_name == "Application denied by financial institution"] = 0
# drop units with income > 100K
hmda = hmda %>%
  filter(applicant_income_000s <= 1000) %>%
  group_by(state_code, county_name)
# drop counties that contain less than 50 records so that it is possible to estimate the proxy
# probabilities given income in every county
geolocation_dist = hmda %>%
  summarise(n = n(),
            nwhite = sum(race == "White"),
            nblack = sum(race == "Black"),
            napi = sum(race == "API"),
            nyhat1 = sum(outcome == 1)) %>%
  filter(n >= 50) %>% group_by(state_code, county_name)
geolocation_dist$county_id = 1:nrow(geolocation_dist)
hmda = hmda %>%
  left_join(geolocation_dist, by = c("state_code", "county_name"))
hmda = hmda %>% ungroup() %>% filter(!is.na(county_id)) %>% select(-one_of(c("county_name", "state_code", "action_taken_name")))

write_csv(hmda, "clean_hmda.csv")

###################
# generating support vector: rho vector on page 24
###################
n = 100
betas = matrix(0, n, 2)
for (i in 1:n){
  beta = rnorm(2)
  beta = beta/sqrt(sum(beta^2))
  betas[i, ] = beta
}
write_csv(as.data.frame(betas), "betas.csv")

