# Replication code for "Assessing Algorithmic Fairness with Unobserved Protected Class Using Data Combination"
### https://arxiv.org/abs/1906.00285

# Results for HMDA (Section 8.1)
### Data Downloading 
Our dataset is from the HMDA mortgage dataset. See https://www.consumerfinance.gov/data-research/hmda/. We use public mortgage records in US market during 2011-2012, which is also used in CFPB's BISG proxy method white paper (https://files.consumerfinance.gov/f/201409_cfpb_report_proxy-methodology.pdf). This dataset can be downloaded by copying and pasting the following link to the web browser (the full dataset is around 2G, and it can take a while to process the query and start downloading): 
https://api.consumerfinance.gov/data/hmda/slice/hmda_lar.csv?&$where=as_of_year+IN+(2012,2011)+AND+action_taken+IN+(1,2,3)+AND+applicant_race_1+IN+(1,2,3,4,5)+AND+applicant_ethnicity+IN+(1,2)&$select=action_taken_name,%20applicant_ethnicity_name,%20applicant_income_000s,%20applicant_race_name_1,%20applicant_race_name_2,%20as_of_year,%20county_name,%20state_code&$limit=0

### Data dictionary 
This dataset includes the following variables: action_taken_name, applicant_ethnicity_name, applicant_race_name_1, applicant_race_name_2, applicant_income_000, county_name, state_code, whose meaning can be found in https://cfpb.github.io/api/hmda/fields.html.

### Data processing 
After downloading the data, use HMDA/data_cleaning.R to preproess and clean the data. This script removes missing values, construct the race and outcome labels, and drop units whose income is more than 100K. This file also generates the support vectors (stored in betas.csv) for computing the support functions in  HMDA/compute_hmda_demo_disparity_runner.py and Warfrin/warfarin_runner_3tprs.py. 

Then run HMDA/fit_proxy_prob.R to fit proxy models to estimate the race and outcome probabilities given geolocation only, income only, or both proxy variables. In HMDA/fit_proxy_prob.R, we further take a 1% random sample. This generates three csv files: small_proxy_county.csv, small_proxy_income.csv, small_proxy_county_income.csv which contain the proxy probabilities estimated from only geolocation, only income, and both geolocation and income respectively. 

### Generating the figures 
- Figure 3: HMDA/Plotting.R;
- Figure 4: first use HMDA/computeCI.R to compute the confidence intervals (with results stored in CI_income.csv, CI_geolocation.csv, CI_income_geolocation.csv respectively), and the nuse HMDA/Plotting.R to plot the confidence intervals;
- Figure 5: Will need to run 'python HMDA/compute_hmda_demo_disparity_runner.py' (Comment/uncomment blocks as needed to run for income, county, or both). This will generate output in the 'out/income' or 'out/county' depending on 'stump' variable. 
Use 'parse_hmda_demo_disparity.ipynb' to generate plots from 'out' directories. 

###################################
# Results for Warfarin (Section 8.2)
###################################
### Data Downloading 
Downloading link: International Warfarin Pharmacogenetics Consortium (IWPC) dataset in https://www.pharmgkb.org/downloads

After downloading the data (.xls format), see the Metadata sheet for the data dictionary, and save the Subject Data sheet as .csv format for the following data processing. 

### Data processing 
First run Warfrin/data_cleaning.ipynb to remove missing data and apply one-hot-encoding to the variables. Then run Warfrin/data_cleaning.R to get another copy of data where the variables are multi-valued. Warfrin/data_cleaning.R also generates the outcomes Y and Yhat and compute the proxy probabilities using medication only, genetic only, and both medication and genetic as proxies. 

### Generating the figures 
- Figure 6: run Warfrin/Plotting.R
- Figure 7: run Warfrin/computeCI.R to compute the confidence intervals first, and then run Warfrin/Plotting.R to generate figure 7. Note that computeCI.R involves a random data splitting step, so the final figures might be slightly different from Figure 7 in our paper but the overall pattern should be similar. 
- Figure 8: run Warfrin/warfarin_runner_3tprs.py to compute the partial identification sets, and then use Warfrin/parse_warfarin_tpr_suppfn.ipynb to produce the plots  

# Main code for computing optimization programs, Alg. 2, and subproblems
ecological_fairness.py contains the functions that set up the optimization programs, compute the discretization-based algorithm, and determine feasibility ranges (including routines specialized for the case studies). 

For usage, please refer to the case studies scripts to compute approximate support function evaluations in parallel for HMDA and Warfarin, respectively: 
HMDA/compute_hmda_demo_disparity_runner.py 
Warfrin/warfarin_runner_3tprs.py

These scripts read in the proxy variables and run the optimization problems in parallel, use a fixed set of samples from the unit sphere, and save the output.  

Dependencies: 
Gurobi
Numpy, Scipy, scikit-learn, pickle 
Optional: joblib (for parallelization) 


#####
Pharmgkb data usage policy
Attribution: https://www.pharmgkb.org/page/dataUsagePolicy
Changes were not made to raw data. 
