
import numpy as np
import gurobipy as gp
from scipy.optimize import minimize
from datetime import datetime
import random
import sys
from ecological_fairness import *

import pandas as pd
gen_proxy = pd.read_csv('genetic_as_proxy.csv')
medgen_proxy = pd.read_csv('medicine_genetic_as_proxy.csv')
med_proxy = pd.read_csv('medicine_as_proxy.csv')


print 'white tpr', sum((medgen_proxy['Yhat']==1)&(medgen_proxy['Y']==1)&(medgen_proxy['race']=='White.'))*1.0/sum((medgen_proxy['Y']==1)&(medgen_proxy['race']=='White.'))

print 'asian tpr', sum((medgen_proxy['Yhat']==1)&(medgen_proxy['Y']==1)&(medgen_proxy['race']=='Asian.'))*1.0/sum((medgen_proxy['Y']==1)&(medgen_proxy['race']=='Asian.'))

print 'black tpr', sum((medgen_proxy['Yhat']==1)&(medgen_proxy['Y']==1)&(medgen_proxy['race']=='Black.'))*1.0/sum((medgen_proxy['Y']==1)&(medgen_proxy['race']=='Black.'))

mixed_tpr = sum((medgen_proxy['Yhat']==1)&(medgen_proxy['Y']==1)&(medgen_proxy['race']!='White.'))*1.0/sum((medgen_proxy['Y']==1)&(medgen_proxy['race']!='White.'))
print mixed_tpr

##############
# Set proxy type
proxy = gen_proxy
PROXY_TYPE = 'gen_proxy'
Y = proxy['Y']
Yhat = proxy['Yhat']
p_az = proxy['white_prob']
# first do white against nonwhite

joints = [ proxy['py1yhat1'], proxy['py0yhat1'], proxy['py1yhat0'], proxy['py0yhat0'] ]

# get feasible ranges
p_az = proxy['asian_prob']
# white v asian
# white v black
LIPSCH_CONST = 0; dists_mat = 0
res = get_t_range_feasibility(Y, Yhat, joints, p_az, LIPSCH_CONST, dists_mat)
tmax = res[0]
res_min = get_t_range_feasibility(Y, Yhat, joints, p_az, LIPSCH_CONST, dists_mat, direction = 'min')
tmin = res_min[0]
t1s =[tmin,tmax]

p_az = proxy['black_prob']
res = get_t_range_feasibility(Y, Yhat, joints, p_az, LIPSCH_CONST, dists_mat)
tmax = res[0]
res_min = get_t_range_feasibility(Y, Yhat, joints, p_az, LIPSCH_CONST, dists_mat, direction = 'min')
tmin = res_min[0]
t2s =[tmin,tmax]
print 1/np.asarray(t1s)
print 1/np.asarray(t2s)

p_az = proxy[['white_prob','asian_prob','black_prob']].values
# print t_0s
nts = 5
t1s_=np.linspace(t1s[0],t1s[1], nts) #alpha
t2s_=np.linspace(t2s[0],t2s[1], nts) #beta

betas = pd.read_csv('../HMDA/betas.csv')
betas = betas[['V1','V2']].values
nbetas = 1
res__ = [None] * nbetas
for ind in range(nbetas):
    res__[ind] = Parallel(n_jobs=12, verbose = 50)(delayed(get_tpr_disp_obs_outcomes_multi_race)(t1,t2, betas[ind,:], Y, Yhat, joints, p_az, LIPSCH_CONST, feasibility_relax=False, quiet = False ) for t1 in t1s_ for t2 in t2s_ )
    objs = [ x[0] for x in res__[ind] ]
    this_res = res__[ind]
    pickle.dump({'whole_res':res__[ind], 'best_res': this_res[np.argmax(objs)] }, open('out/warfarintprs-'+str(ind)+'-.pkl','wb'))

pickle.dump(res__, open('warfarintprs-'+PROXY_TYPE+'.pkl','wb'))
# get feasible ranges
p_az = proxy['asian_prob'].values
# white v asian
# white v black
LIPSCH_CONST = 0; dists_mat = 0
res = get_t_range_feasibility(Y, Yhat, joints, p_az, LIPSCH_CONST,  dists_mat,type = 'tnr')
tmax = res[0]
res_min = get_t_range_feasibility(Y, Yhat, joints, p_az, LIPSCH_CONST, dists_mat,type = 'tnr', direction = 'min')
tmin = res_min[0]
t1s =[tmin,tmax]

p_az = proxy['black_prob'].values
res = get_t_range_feasibility(Y, Yhat, joints, p_az, LIPSCH_CONST, dists_mat, type = 'tnr')
tmax = res[0]
res_min = get_t_range_feasibility(Y, Yhat, joints, p_az, LIPSCH_CONST, dists_mat,type = 'tnr', direction = 'min')
tmin = res_min[0]
t2s =[tmin,tmax]

nts = 5
t1s_=np.linspace(t1s[0],t1s[1], nts) #alpha
t2s_=np.linspace(t2s[0],t2s[1], nts) #beta
betas = pd.read_csv('../HMDA/betas.csv')
betas = betas[['V1','V2']].values
nbetas = 25
res__ = [None] * nbetas
p_az = proxy[['white_prob','asian_prob','black_prob']].values

bests = [None] * nbetas
for ind in range(nbetas):
    res__[ind] = Parallel(n_jobs=12, verbose = 50)(delayed(get_tnr_disp_obs_outcomes_multi_race)(t1,t2, betas[ind,:], Y, Yhat, joints, p_az, LIPSCH_CONST, feasibility_relax=False, quiet = False ) for t1 in t1s_ for t2 in t2s_)
    objs = [ x[0] for x in res__[ind] ]
    this_res = res__[ind]
    bests[ind] = this_res[np.argmax(objs)]
    pickle.dump({'best_res': bests[ind] }, open('out/warfarintnrs-'+str(ind)+'-.pkl','wb'))

pickle.dump(bests, open('warfarintnrs-'+PROXY_TYPE+'.pkl','wb'))
#############################################

proxy = med_proxy
PROXY_TYPE = 'med_proxy'
Y = proxy['Y']
Yhat = proxy['Yhat']
p_az = proxy['white_prob']
# first do white against nonwhite

joints = [ proxy['py1yhat1'], proxy['py0yhat1'], proxy['py1yhat0'], proxy['py0yhat0'] ]

# get feasible ranges
p_az = proxy['asian_prob']
# white v asian
# white v black
LIPSCH_CONST = 0; dists_mat = 0
res = get_t_range_feasibility(Y, Yhat, joints, p_az, LIPSCH_CONST, dists_mat)
tmax = res[0]
res_min = get_t_range_feasibility(Y, Yhat, joints, p_az, LIPSCH_CONST, dists_mat, direction = 'min')
tmin = res_min[0]
t1s =[tmin,tmax]

p_az = proxy['black_prob']
res = get_t_range_feasibility(Y, Yhat, joints, p_az, LIPSCH_CONST, dists_mat)
tmax = res[0]
res_min = get_t_range_feasibility(Y, Yhat, joints, p_az, LIPSCH_CONST, dists_mat, direction = 'min')
tmin = res_min[0]
t2s =[tmin,tmax]
print 1/np.asarray(t1s)
print 1/np.asarray(t2s)

p_az = proxy[['white_prob','asian_prob','black_prob']].values
# print t_0s
nts = 50
t1s_=np.linspace(t1s[0],t1s[1], nts) #alpha
t2s_=np.linspace(t2s[0],t2s[1], nts) #beta

betas = pd.read_csv('../HMDA/betas.csv')
betas = betas[['V1','V2']].values
nbetas = 25
res__ = [None] * nbetas
for ind in range(nbetas):
    res__[ind] = Parallel(n_jobs=12, verbose = 50)(delayed(get_tpr_disp_obs_outcomes_multi_race)(t1,t2, betas[ind,:], Y, Yhat, joints, p_az, LIPSCH_CONST, feasibility_relax=False, quiet = False ) for t1 in t1s_ for t2 in t2s_ )
    objs = [ x[0] for x in res__[ind] ]
    this_res = res__[ind]
    pickle.dump({'whole_res':res__[ind], 'best_res': this_res[np.argmax(objs)] }, open('out/warfarintprs-'+str(ind)+'-.pkl','wb'))

pickle.dump(res__, open('warfarintprs-'+PROXY_TYPE+'.pkl','wb'))
# get feasible ranges
p_az = proxy['asian_prob'].values
# white v asian
# white v black
LIPSCH_CONST = 0; dists_mat = 0
res = get_t_range_feasibility(Y, Yhat, joints, p_az, LIPSCH_CONST,  dists_mat,type = 'tnr')
tmax = res[0]
res_min = get_t_range_feasibility(Y, Yhat, joints, p_az, LIPSCH_CONST, dists_mat,type = 'tnr', direction = 'min')
tmin = res_min[0]
t1s =[tmin,tmax]

p_az = proxy['black_prob'].values
res = get_t_range_feasibility(Y, Yhat, joints, p_az, LIPSCH_CONST, dists_mat, type = 'tnr')
tmax = res[0]
res_min = get_t_range_feasibility(Y, Yhat, joints, p_az, LIPSCH_CONST, dists_mat,type = 'tnr', direction = 'min')
tmin = res_min[0]
t2s =[tmin,tmax]

nts = 5
t1s_=np.linspace(t1s[0],t1s[1], nts) #alpha
t2s_=np.linspace(t2s[0],t2s[1], nts) #beta
betas = pd.read_csv('../HMDA/betas.csv')
betas = betas[['V1','V2']].values
nbetas = 25
res__ = [None] * nbetas
p_az = proxy[['white_prob','asian_prob','black_prob']].values

bests = [None] * nbetas
for ind in range(nbetas):
    res__[ind] = Parallel(n_jobs=12, verbose = 50)(delayed(get_tnr_disp_obs_outcomes_multi_race)(t1,t2, betas[ind,:], Y, Yhat, joints, p_az, LIPSCH_CONST, feasibility_relax=False, quiet = False ) for t1 in t1s_ for t2 in t2s_)
    objs = [ x[0] for x in res__[ind] ]
    this_res = res__[ind]
    bests[ind] = this_res[np.argmax(objs)]
    pickle.dump({'best_res': bests[ind] }, open('out/warfarintnrs-'+str(ind)+'-.pkl','wb'))

pickle.dump(bests, open('warfarintnrs-'+PROXY_TYPE+'.pkl','wb'))
