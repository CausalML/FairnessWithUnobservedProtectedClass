import numpy as np
import gurobipy as gp
from scipy.optimize import minimize
from datetime import datetime
import random
import sys
from ecological_fairness import *
import pandas as pd
import glob

def get_dem_disp_obs_outcomes_multi_race_over_LCs(Y, n_as, mu_scalar, p_az,
    p_yz,  Z, LCs,ind_mu, quiet=True, smoothing=None, smoothing1d=False, direction = 'max', save = True):
    ress__ = [None] * len(LCs)
    for ind,lc in enumerate(LCs):
        ress__[ind] = get_dem_disp_obs_outcomes_multi_race(Y, n_as, mu_scalar, p_az, p_yz,  Z, lc, quiet, smoothing, smoothing1d, direction)
        pickle.dump({'res': ress__[ind], 'mu_scalar':mu_scalar}, open('out/countyincome/output-'+str(ind_mu)+'-'+'lc' + str(lc) +'-' +datetime.now().strftime('%Y-%m-%d-%H-%M-%S')+'.pkl', 'wb'))
    pickle.dump({'res': ress__, 'mu_scalar':mu_scalar}, open('out/output-'+str(ind_mu)+'-'+datetime.now().strftime('%Y-%m-%d-%H-%M-%S')+'.pkl', 'wb'))
    return ress__

'''
Helper function for running in parallel
Y: outcomes 
n_as: number of protected attributes
p_az, p_yz: proxy distributions
Z: auxiliary data
lc: lipschitz constant
ind: number of parallel run 
quiet: Gurobi not verbose or verbose
smoothing, smoothing1d: Run with smoothness assumptions or not 
direction: maximize or minimize disparity
stump: Save output from this parallel run in out/stump/output-* 
'''
def get_DD_multi_race_caller(Y, n_as, mu_scalar, p_az, p_yz,  Z, lc, ind, quiet=True, smoothing=None, smoothing1d=False,direction = 'max', stump = 'hmda', save = True):
    res = get_dem_disp_obs_outcomes_multi_race(Y, n_as, mu_scalar, p_az, p_yz,  Z, lc, quiet, smoothing, smoothing1d, direction)
    pickle.dump({'res': res, 'mu_scalar':mu_scalar, 'lc':lc}, open('out/'+stump+'/output-'+str(ind)+'-lc-'+str(lc)+'-'+datetime.now().strftime('%Y-%m-%d-%H-%M-%S')+'.pkl', 'wb'))
    return res


def select_name(lip, lip_list_whole, name_list):
    return [name_list[i] for i in range(len(lip_list_whole)) if (lip == lip_list_whole[i])]


def get_disps(w_y1, w_y0, n_as, y, p_as):
    n = len(y)
    return [ 1.0/n * ( np.dot(w_y1[0,:],y)/p_as[0] - np.dot(w_y1[a,:],y)/p_as[a] ) for a in range(n_as)[1:] ]


def compute_disparity_from_res(res, n_as, Y, p_as):
    if (~np.isnan(res[0])):
        return get_disps(res[1], res[2], n_as, Y, p_as)
    else:
        return [np.nan, np.nan]

def parse_name(directory, N_LCs):
    name_list = glob.glob(directory + "/*")
    split_name = [name.split('-') for name in name_list]
    lip_list_whole = [s[3] for s in split_name]
    lip_list = lip_list_whole[0:N_LCs]
    return lip_list, lip_list_whole, name_list

def compile_disparity(lip_list, lip_list_whole, name_list):
    dd = {lip:None for lip in lip_list}
    for lip in lip_list:
        print lip
        files = select_name(lip, lip_list_whole, name_list)
        disparity = np.zeros((len(files), 2))
        for ind in range(len(files)):
            disps_temp = pickle.load(open(files[ind],'rb'))
            mu_scalar = disps_temp['mu_scalar']
            res = disps_temp['res']
            disparity[ind, :] = compute_disparity_from_res(res)
        dd[lip] = disparity
    return dd

betas = pd.read_csv('betas.csv')
betas = betas[['V1','V2']].values
betas = betas[15]

# Income proxy
hmda_ = pd.read_csv('small_proxy_income.csv')
# betas
racedict = {'White': 0, 'API': 1, 'Black': 2}
A = [racedict[hmda_['race'][i]] for i in range(len(hmda_['race']))]
n_as = len(np.unique(A))

Y = hmda_['outcome']
Z = hmda_['applicant_income_000s']
p_yz_inc = hmda_['yhat1']# loan conditional on income
p_az_inc = hmda_[['White', 'API','Black']]

p_as = (p_az_inc.sum(axis=0)/p_az_inc.shape[0])
p_as
p_as2 = [0]*3
p_as2[0] = p_as['White']
p_as2[1] = p_as['API']
p_as2[2] = p_as['Black']
p_as = p_as2
p_as

N_LCs = 10
#LCs = 0.0001 + np.arange(N_LCs)*0.0005
LCs = np.linspace(0.0003,0.0006,N_LCs)
LCs

stump = 'income'

# Run parallel jobs (HMDA is memory intensive) 
# Each embarassingly parallel job saves its output in out/stump/output-* 
# Final output from all runs (if successful) is serialized as pickle in, e.g. 'spfn-hmda-income.pkl'

ress = Parallel(n_jobs=-24,verbose = 50)(delayed(get_DD_multi_race_caller)(Y, n_as, betas[i,:], p_az_inc.values, p_yz_inc,  Z, lc, i, stump=stump, quiet=True, smoothing=True, smoothing1d = True, direction = 'max') for lc in LCs for i in range(betas.shape[0]))
pickle.dump({'res':ress, 'type':'income'}, open('hmda_income.pkl','wb'))

### Only county
print 'only county'
hmda_ = pd.read_csv('small_proxy_county.csv')

racedict = {'White': 0, 'API': 1, 'Black': 2}
A = [racedict[hmda_['race'][i]] for i in range(len(hmda_['race']))]
n_as = len(np.unique(A))

Y = hmda_['outcome']
Z = hmda_['applicant_income_000s']
p_yz_inc = hmda_['yhat1']# loan conditional on income
p_az_inc = hmda_[['White', 'API','Black']]

p_as = (p_az_inc.sum(axis=0)/p_az_inc.shape[0])
p_as
p_as2 = [0]*3
p_as2[0] = p_as['White']
p_as2[1] = p_as['API']
p_as2[2] = p_as['Black']
p_as = p_as2
p_as

print 'no smoothing'
N_LCs = 1
LCs = [1]
ress = Parallel(n_jobs=24,verbose = 50)(delayed(get_DD_multi_race_caller)(Y, n_as, betas[i,:], p_az_inc.values, p_yz_inc,  Z, lc, i, stump='county', quiet=True, smoothing=False, smoothing1d = False, direction = 'max') for lc in LCs for i in range(betas.shape[0]))
pickle.dump({'res':ress, 'type':'income'}, open('hmda_county.pkl','wb'))

#
# hmda_ = pd.read_csv('small_proxy_county_income.csv')
#
# racedict = {'White': 0, 'API': 1, 'Black': 2}
# A = [racedict[hmda_['race'][i]] for i in range(len(hmda_['race']))]
# n_as = len(np.unique(A))
#
# Y = hmda_['outcome']
# Z = hmda_['applicant_income_000s']
# p_yz_inc = hmda_['yhat1']# loan conditional on income
# p_az_inc = hmda_[['White', 'API','Black']]
#
# p_as = (p_az_inc.sum(axis=0)/p_az_inc.shape[0])
# p_as
# p_as2 = [0]*3
# p_as2[0] = p_as['White']
# p_as2[1] = p_as['API']
# p_as2[2] = p_as['Black']
# p_as = p_as2
# p_as
#
# N_LCs = 10
# LCs = 0.0001 + np.arange(N_LCs)*0.0005
# LCs
#
# ress = Parallel(n_jobs=-2,verbose = 50)(delayed(get_DD_multi_race_caller)(Y, n_as, betas[i,:], p_az_inc.values, p_yz_inc,  Z, lc, i, stump='county_income_test', quiet=True, smoothing=False, smoothing1d = False, direction = 'max') for lc in LCs for i in range(betas.shape[0]))
