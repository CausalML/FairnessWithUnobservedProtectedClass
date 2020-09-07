import numpy as np
import gurobipy as gp
from datetime import datetime
import random
import sys
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
import pickle

'''Throughout: denote p_yz, p_az, as P[Y=1|Z], P[A=1|Z]
'''


def get_general_interval_wghts_algo_centered_TV_prob(gamma, Y, a_, b_, fq, quiet=True):
    wm = 1/fq; wm_sum=wm.sum(); n = len(Y)
    wm = wm/wm_sum # normalize propensities
    # assume estimated propensities are probs of observing T_i
    y = Y; weights = np.zeros(n);
    m = gp.Model()
    if quiet: m.setParam("OutputFlag", 0)
    t = m.addVar(lb = 0., ub = gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS)
    w = [m.addVar(obj = -yy, lb = 0., ub = gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS) for yy in y]
    d = [m.addVar(lb = 0., ub = gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS) for yy in y]
    m.update()
    m.addConstr(gp.quicksum(w)==1)
    m.addConstr(gp.quicksum(d)<=gamma*t)
    for i in range(len(y)):
        m.addConstr(w[i] <= b_[i] )
        m.addConstr(w[i] >= a_[i] )

    # Add smoothness constraints
    # sort
    m.optimize()
    wghts = np.asarray([ ww.X for ww in w ]) # would like to have failsafe for not being able to optimize
    return [-m.ObjVal,wghts,t.X/wm_sum]


'''return LR model
'''
def get_lr(X,Y):
    clf = LogisticRegression(); clf.fit(X,Y)
    Rhat = clf.predict_proba(X)[:,1]
    return [clf, Rhat]

def lower_FH(p_yz, p_az):
    return np.maximum( (p_az+p_yz -1)/p_yz, 0 )

def upper_FH(p_yz, p_az):
    return np.minimum( p_az/p_yz, 1 )

''' Given the optimal weight vector, compute the measure
For decomposition
w = P[ A | Z, Y]
\hat Y
\sum_{i: A_i = 1} w_i \hat Y_i - \sum_{i: A_i = 2} w_i \hat Y_i
'''
def compute_bounds_dem_disp_given_weights_p_az_haty(w_, y_, p_yz, p_az):
    dem_disp = np.mean(w_*y_)/np.mean(p_az) - np.mean((1-w_)*y_)/(1-np.mean(p_az))
    return dem_disp


# ''' get bounds on demographic disparity
# for a decomposition where the decision variables are complementary
# '''
def get_dem_disp_obs_outcomes(y, p_az, p_yz, a_, b_, Z, LIPSCH_CONST, LAW_TOT_PROB_SLACK=0, quiet=True, smoothing=None):
    n = len(y); m = gp.Model()
    if (np.asarray([len(a) for a in [p_az, p_yz, a_, b_, Z]]) != n).any():
        raise ValueError("data inputs must have the same length")
    if quiet:
        m.setParam("OutputFlag", 0)
#     t = m.addVar(lb = 0., ub = gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS)
    w_y1_a = [m.addVar( lb = 0., ub = 1, vtype=gp.GRB.CONTINUOUS) for yy in y]
    w_y0_a = [m.addVar( lb = 0., ub = 1, vtype=gp.GRB.CONTINUOUS) for yy in y]
#     w_y1_b = [m.addVar( lb = 0., ub = 1, vtype=gp.GRB.CONTINUOUS) for yy in y]
#     w_y0_b = [m.addVar( lb = 0., ub = 1, vtype=gp.GRB.CONTINUOUS) for yy in y]
    dev_marginalized_tot_prob = m.addVar(lb=0.,vtype = gp.GRB.CONTINUOUS)
    # for absolute values
    sorteddiffs = [m.addVar( lb = 0., ub = 1, vtype=gp.GRB.CONTINUOUS) for yy in y[1:]]
    sorteddiffs_0 = [m.addVar( lb = 0., ub = 1, vtype=gp.GRB.CONTINUOUS) for yy in y[1:]]
    wdiffs_1 = [m.addVar( lb = 0., ub = 1, vtype=gp.GRB.CONTINUOUS) for yy in y[1:]]#m.addVars(n, lb = 0., ub = 1, vtype=gp.GRB.CONTINUOUS)
    wdiffs_0 = [m.addVar( lb = 0., ub = 1, vtype=gp.GRB.CONTINUOUS) for yy in y[1:]]#m.addVars(n, lb = 0., ub = 1, vtype=gp.GRB.CONTINUOUS)
    m.update()
    for i in range(len(y)):
        m.addConstr(w_y1_a[i] <= b_[i], 'b'+str(i) )
        m.addConstr(w_y1_a[i] >= a_[i], 'a'+str(i) )
#     law of tot prob bounds
#         print (p_yz[i], 1-p_yz[i]), p_az[i] # infeasible if $p_az[i]$ is not in this interval
#         m.addConstr( w_y1_a[i]*p_yz[i] + w_y0_a[i]*(1.0-p_yz[i]) == p_az[i]  )
#     law of tot prob for complementary unknowns
#         if p_yz[i] <= p_az[i] <= 1-p_yz[i]:
#             m.addConstr( w[i]*p_yz[i] + (1.0-p_yz[i]) * (1-w[i]) == p_az[i]  )
#         m.addConstr(2*w[i]*p_yz[i] + (1.0-p_yz[i]) - w[i] == p_az[i] )
##########
    # Marginal law of total probability / compatibility constraints
#     m.addConstr( dev_marginalized_tot_prob >= sum(w[i]*p_yz[i] for i in range(n))*1.0/n + sum((1.0-w[i])*(1-p_yz[i]) for i in range(n))*1.0/n  - np.mean(p_az) )
#     m.addConstr( dev_marginalized_tot_prob >= -1*(sum(w[i]*p_yz[i] for i in range(n))*1.0/n + sum((1.0-w[i])*(1-p_yz[i]) for i in range(n))*1.0/n  - np.mean(p_az)) )
#     m.addConstr( dev_marginalized_tot_prob <= LAW_TOT_PROB_SLACK, 'tot_prob'  ) # slack for law tot prob constraint

    # Add smoothness constraints
    if smoothing=='1d':
        sort_ind = np.argsort(Z) #returns sort inds in ascending order
        for i in range(len(y))[:-1]:
            ind = sort_ind[i+1]; ind_minus = sort_ind[i];
            m.addConstr( wdiffs_1[i] == w_y1_a[ind] - w_y1_a[ind_minus] );
            m.addConstr(sorteddiffs[i] == gp.abs_(wdiffs_1[i]))
            m.addConstr( wdiffs_0[i] == w_y0_a[ind] - w_y0_a[ind_minus] );
            m.addConstr(sorteddiffs_0[i] == gp.abs_(wdiffs_0[i]))
            m.addConstr(sorteddiffs[i] <= LIPSCH_CONST*(Z[ind] - Z[ind_minus]), 'smoothness'+str(i) )
            m.addConstr(sorteddiffs_0[i] <= LIPSCH_CONST*(Z[ind] - Z[ind_minus]), 'smoothness-on-y0'+str(i) )
    elif smoothing=='nd':
        m.addConstrs((w_y1_a[i] - w_y1_a[j] <= LIPSCH_CONST*dists_mat[i,j] for i in range(n) for j in range(n) ) , name='L-nd')
        m.addConstrs((-w_y1_a[i] + w_y1_a[j] <= LIPSCH_CONST*dists_mat[i,j] for i in range(n) for j in range(n) ) , name='-L-nd')
    # objective function
    expr = gp.LinExpr();
    for i in range(len(y)):
        expr +=w_y1_a[i] * -1*y[i];
#         expr += (y[i]*w_y1_a[i] + (1-y[i])*w_y0_a[i]) * -1*y[i];
    m.setObjective(expr, gp.GRB.MINIMIZE);
    m.optimize()
    w_y1_a_vals = np.asarray([ ww.X for ww in w_y1_a ])
    w_y0_a_vals = np.asarray([ ww.X for ww in w_y0_a ])
    return [-m.ObjVal, w_y1_a_vals, w_y0_a_vals, m]


'''
pyhat_jt_y_z = Pr(Yhat = yhat, Y=1 | Z_i)
Note these are the bounds for unrestricted w_y0_a
'''
def upper_FH_LTP_TPR(p_y0_z, p_az, pyhat_jt_y_z):
    return np.minimum( (p_az-p_y0_z)/pyhat_jt_y_z, 1 )

'''
Solve the tpr disparity for a range of t values
resolve with warm starting
Inputs:
t: scaling factor; We use an asymptotically well-behaved scaling omega = W t_a n_a
y: objective coefficients
p_yhat_cond_y: Pr(Y=1 | hat Y = 1, Z_i)
p_yhat_jt_y: Pr(Y=1 , hat Y = 1 | Z_i)
b_: upper bound
If smoothing: add smoothness constraints
return primal weights
return scaling t
'''
def get_tpr_disp_two_races_over_ts(ts, y, p_y_cond_yhat, p_yhat_jt_y, p_a_z, LIPSCH_CONST, LAW_TOT_PROB_SLACK=0, quiet=True, smoothing=None):
    m = gp.Model()
    res = [ None ] * len(ts)
    if quiet: m.setParam("OutputFlag", 0)
    # A = 0 is a # A = 1 is b
    n_a = sum(A==0); n_b = sum(A==1);
    n = sum(Z)
    C = sum(p_y_cond_yhat)

    wghts = np.zeros(len(A))
    w_y1_a = [m.addVar( lb = 0., vtype=gp.GRB.CONTINUOUS) for yy in y]
    w_y0_a = [m.addVar( lb = 0.,  vtype=gp.GRB.CONTINUOUS) for yy in y]
    w_y1_yhat_a = [m.addVar( lb = 0.,  vtype=gp.GRB.CONTINUOUS) for yy in y]
    w_y0_yhat_a = [m.addVar( lb = 0., vtype=gp.GRB.CONTINUOUS) for yy in y]

    m.update()
    t = ts[0] # build the model first
    m.addConstr(gp.quicksum([w_y1_a[i] * p_yhat_jt_y[i] for i in range(n) ]) == n, name = 'homogenization') # need to fix the homogenization
    for i in range(len(y)):
        m.addConstr(w_y1_a[i] <= n*t, 'omega-y=1'+str(i) )
        m.addConstr(w_y0_a[i] <= n*t, 'omega-y=0'+str(i) )
        m.addConstr(w_y1_yhat_a[i] <= n*t, 'omega-y=1_1-hy'+str(i) )
        m.addConstr(w_y0_yhat_a[i] <= n*t, 'omega-y=0_1-hy'+str(i) )
    if smoothness:
        m.addConstrs((w_y0_a[i] - w_y0_a[j] <= LIPSCH_CONST*dists_mat[i,j] * t * n for i in range(n) for j in range(n) ) , name='L-y=0')
        m.addConstrs((-w_y0_a[i] + w_y0_a[j] <= LIPSCH_CONST*dists_mat[i,j] * t * n for i in range(n) for j in range(n) ) , name='-L-y=0')
        m.addConstrs((w_y1_a[i] - w_y1_a[j] <= LIPSCH_CONST*dists_mat[i,j] * t * n for i in range(n) for j in range(n) ) , name='L-y=1')
        m.addConstrs((-w_y1_a[i] + w_y1_a[j] <= LIPSCH_CONST*dists_mat[i,j] * t * n for i in range(n) for j in range(n) ) , name='-L-y=1')

    expr = gp.LinExpr();
    expr += t*C/(t*C - 1.0) * gp.quicksum([w_y1_a[i] * p_yhat_jt_y[i] * y[i] for i in range(n) ])  - t*C/(t*C - 1.0)
    if direction=='max':
        m.setObjective(expr, gp.GRB.MAXIMIZE); m.optimize()
    else:
        m.setObjective(expr, gp.GRB.MINIMIZE); m.optimize()

    if (m.status == gp.GRB.OPTIMAL):
        wghts_y1_a = np.asarray([ ww.X for ww in wghts_y1_a ]);wghts_y0_a = np.asarray([ ww.X for ww in wghts_y0_a ]);
        res[0] = [m.ObjVal, wghts_y0_a, wghts_y1_a, t]
    else:
        res[0] =  [np.nan, 0, 0]
    # Change model constraint RHSes and objective and reoptimize
    for t_ind, t_ in enumerate(ts[1:]):
        m.getConstrByName('law_total_prob').RHS = p_a_z[i]*t_
        for i in range(n):
            m.getConstrByName('omega-y=1['+str(i)+']').RHS = n*t_
            m.getConstrByName('omega-y=0['+str(j)+']').RHS = n*t_
        for i in range(n):
            for j in range(n)[i:]:
                m.getConstrByName('L-y=0'+str(i)+','+str(j)+']').RHS = LIPSCH_CONST*dists_mat[i,j]* t_ * n
                m.getConstrByName('L-y=0'+str(i)+','+str(j)+']').RHS = LIPSCH_CONST*dists_mat[i,j]* t_ * n
            for k in range(n_b)[i:]:
                m.getConstrByName('L-y=1'+str(i)+','+str(k)+']').RHS = LIPSCH_CONST*dists_mat[i,j]* t_ * n
                m.getConstrByName('L-y=1'+str(i)+','+str(k)+']').RHS = LIPSCH_CONST*dists_mat[i,j]* t_ * n
        expr = gp.LinExpr();
        expr += t_*C/(t_*C - 1.0) * gp.quicksum([w_y1_a[i] * p_yhat_jt_y[i] * y[i] for i in range(n) ])  - t_*C/(t_*C - 1.0)
        if direction=='max':
            m.setObjective(expr, gp.GRB.MAXIMIZE); m.optimize()
        else:
            m.setObjective(expr, gp.GRB.MINIMIZE); m.optimize()
        if (m.status == gp.GRB.OPTIMAL):
            wghts_y1_a = np.asarray([ ww.X for ww in wghts_y1_a ]);wghts_y0_a = np.asarray([ ww.X for ww in wghts_y0_a ]);
            res[0] = [m.ObjVal, wghts_y0_a, wghts_y1_a, t_]
        else:
            res[0] =  [np.nan, 0, 0]
    if save:
        pickle.dump({'res':res, 'args': {'ts':ts, 'LAMBDA':LAMBDA, 'b_':b_, 'Z':Z} }, open('out/opt_over_t_res_'+savestr+'.pkl', 'wb')  )
    return res


def dists_matrix(X, norm = 'minkowski'):
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    dists = pdist(X_scaled, norm, p=1) # distance computed in entry ij
    dists_mat = squareform(dists)
    return [dists_mat, X_scaled]

def get_p_y_cond_jt_yhat_z(Y,Yhat,Z):
    #P[Y=1 | Z, Yhat]
    [clf_Y_Y1, p_y1_cnd_yhat1_] = get_lr(Z[Yhat == 1,:] , Y[Yhat == 1])
    [clf_Y_Y0, p_y1_cnd_yhat0_] = get_lr(Z[Yhat == 0,:] , Y[Yhat == 0])
    #P[Yhat=1 | Z]
    [y_z, p_yhatz_1] = get_lr(Z, Yhat)
    # Compute the conditional models on all data
    #P[Y=1 | Z, Yhat]
    p_y1_cnd_yhat1 = clf_Y_Y1.predict_proba(Z)[:,1]
    p_y0_cnd_yhat1 = 1 - p_y1_cnd_yhat1
    p_y1_cnd_yhat0 = clf_Y_Y0.predict_proba(Z)[:,1]
    p_y0_cnd_yhat0 = 1 - p_y1_cnd_yhat0
    # Compute the joint models
    conditionals = [p_y1_cnd_yhat1, p_y0_cnd_yhat1, p_y1_cnd_yhat0, p_y0_cnd_yhat0]
    #P[Y=1 , Yhat | Z] = P[Y=1 | Yhat, Z] P[ Yhat | Z]
    joints = [p_y1_cnd_yhat1*p_yhatz_1 , p_y0_cnd_yhat1*p_yhatz_1, p_y1_cnd_yhat0*(1-p_yhatz_1), p_y0_cnd_yhat0*(1-p_yhatz_1) ]
    #joints are P[Y1, Yhat=1 | Z], P[Y0, Yhat=1 | Z], P[Y1, Yhat=0 | Z], P[Y0, Yhat=0 | Z]
    return [conditionals, joints, p_yhatz_1]

def get_obs(p,y):
    return np.asarray([ 1-p[i] if y[i] == 1 else p[i] for i in range(len(y)) ]).flatten()
'''
Solve the tpr for a range of t values
resolve with warm starting
Inputs:
t: scaling factor; We use an asymptotically well-behaved scaling omega = W t_a n_a
y: objective coefficients
e.g. Yhat
p_yhat_cond_y: Pr(Y=1 | hat Y = 1, Z_i)
p_yhat_jt_y: Pr(Y=1 , hat Y = 1 | Z_i)
b_: upper bound
If smoothing: add smoothness constraints
return primal weights
Return weights in the order [(1,1),(0,1),(1,0),(0,0)]
return scaling t
'''
def get_tpr_over_ts(ts, y, conditionals, joints, p_a_z, LIPSCH_CONST,
dists_mat, LAW_TOT_PROB_SLACK=0, direction='max', quiet=True, smoothing=None,
save = True, savestr='tpr'):
    m = gp.Model()
    res = [ None ] * len(ts)
    if quiet: m.setParam("OutputFlag", 0)
    m.setParam("Method", 1);
    n = len(y)
    w_y1_yhatobs_a = [m.addVar( lb = 0., vtype=gp.GRB.CONTINUOUS) for yy in y]
    w_y0_yhatobs_a = [m.addVar( lb = 0.,  vtype=gp.GRB.CONTINUOUS) for yy in y]
    w_y1_yhatnotobs_a = [m.addVar( lb = 0.,  vtype=gp.GRB.CONTINUOUS) for yy in y]
    w_y0_yhatnotobs_a = [m.addVar( lb = 0., vtype=gp.GRB.CONTINUOUS) for yy in y]
    # t = m.addVar( lb = 0., vtype=gp.GRB.CONTINUOUS)
    [p_y1_cnd_yhat1, p_y0_cnd_yhat1, p_y1_cnd_yhat0, p_y0_cnd_yhat0] = conditionals
    [ p_y1_jt_yhat1, p_y0_jt_yhat1, p_y1_jt_yhat0, p_y0_jt_yhat0 ] = joints
    p_y1_jt_yhatobs = [ p_y1_jt_yhat1[i] if y[i] == 1 else p_y1_jt_yhat0[i] for i in range((n)) ]
    p_y0_jt_yhatobs = [ p_y0_jt_yhat1[i] if y[i] == 1 else p_y0_jt_yhat0[i] for i in range((n)) ]
    p_y1_jt_yhatnotobs = [ p_y1_jt_yhat0[i] if y[i] == 1 else p_y1_jt_yhat1[i] for i in range((n)) ]
    p_y0_jt_yhatnotobs = [ p_y0_jt_yhat0[i] if y[i] == 1 else p_y0_jt_yhat1[i] for i in range((n)) ]
    p_y1_cnd_yhatobs = get_obs(p_y1_cnd_yhat1, y)

    C = sum(p_y1_jt_yhatobs)
    print 'adding variables'
    m.update()
    t = ts[0] # build the model first
    m.addConstr(gp.quicksum([w_y1_yhatobs_a[i] * p_y1_cnd_yhatobs[i] for i in range(n) ]) == n, name = 'homogenization') # need to fix the homogenization
    m.addConstrs((w_y1_yhatobs_a[i]  <= t * n for i in range(n) ))
    m.addConstrs((w_y0_yhatobs_a[i]  <= t * n for i in range(n) ))
    m.addConstrs((w_y1_yhatnotobs_a[i]  <= t * n for i in range(n) ))
    m.addConstrs((w_y0_yhatnotobs_a[i]  <= t * n for i in range(n) ))

    for i in range(len(y)):
        m.addConstr( p_y0_jt_yhatobs[i]*w_y0_yhatobs_a[i] + p_y1_jt_yhatobs[i]*w_y1_yhatobs_a[i] + p_y1_jt_yhatnotobs[i]*w_y1_yhatnotobs_a[i] + p_y0_jt_yhatnotobs[i]*w_y0_yhatnotobs_a[i] == p_a_z[i]*t*n , name='LTP'+str(i))

        # m.addConstr(w_y1_a[i] <= n*t, 'omega-y=1'+str(i) )
        # m.addConstr(w_y0_a[i] <= n*t, 'omega-y=0'+str(i) )
        # m.addConstr(w_y1_yhat_a[i] <= n*t, 'omega-y=1_1-hy'+str(i) )
        # m.addConstr(w_y0_yhat_a[i] <= n*t, 'omega-y=0_1-hy'+str(i) )
    if smoothing:
        m.addConstrs((w_y0_yhatobs_a[i] - w_y0_yhatobs_a[j] <= LIPSCH_CONST*dists_mat[i,j] * t * n for i in range(n) for j in range(n)[i:] ) , name='L-y=0')
        m.addConstrs((-w_y0_yhatobs_a[i] + w_y0_yhatobs_a[j] <= LIPSCH_CONST*dists_mat[i,j] * t * n for i in range(n) for j in range(n)[i:] ) , name='-L-y=0')
        m.addConstrs((w_y1_yhatobs_a[i] - w_y1_yhatobs_a[j] <= LIPSCH_CONST*dists_mat[i,j] * t * n for i in range(n) for j in range(n)[i:] ) , name='L-y=1')
        m.addConstrs((-w_y1_yhatobs_a[i] + w_y1_yhatobs_a[j] <= LIPSCH_CONST*dists_mat[i,j] * t * n  for i in range(n) for j in range(n)[i:] ) , name='-L-y=1')
    print 'adding constraints'
    expr = gp.LinExpr();
    expr += 1.0/n *  gp.quicksum([w_y1_yhatobs_a[i] * p_y1_cnd_yhatobs[i] * y[i] for i in range(n) ])
    if direction=='max':
        print 'solving'
        m.setObjective(expr, gp.GRB.MAXIMIZE); m.optimize()
    else:
        m.setObjective(expr, gp.GRB.MINIMIZE); m.optimize()

    if (m.status == gp.GRB.OPTIMAL):
        wghts_ = [w_y1_yhatobs_a, w_y0_yhatobs_a, w_y1_yhatnotobs_a, w_y0_yhatnotobs_a]
        wghts__ = [ [ ww.X for ww in wghts_[ind] ] for ind in range(len(wghts_)) ]
        res[0] = [m.ObjVal, wghts__, t]
    else:
        res[0] =  [np.nan, 0, 0]

    # Change model constraint RHSes and objective and reoptimize
    for t_ind, t_ in enumerate(ts[1:]):
        for i in range(n):
            m.getConstrByName('LTP'+str(i)).RHS = p_a_z[i]*t_
        if smoothing:
            for i in range(n):
                for j in range(n)[i:]:
                    m.getConstrByName('L-y=0'+str(i)+','+str(j)+']').RHS = LIPSCH_CONST*dists_mat[i,j]* t_ * n
                    m.getConstrByName('L-y=0'+str(i)+','+str(j)+']').RHS = LIPSCH_CONST*dists_mat[i,j]* t_ * n
                for k in range(n_b)[i:]:
                    m.getConstrByName('L-y=1'+str(i)+','+str(k)+']').RHS = LIPSCH_CONST*dists_mat[i,j]* t_ * n
                    m.getConstrByName('L-y=1'+str(i)+','+str(k)+']').RHS = LIPSCH_CONST*dists_mat[i,j]* t_ * n
        expr = gp.LinExpr();
        expr += 1.0/n *( t_*C/(t_*C - 1.0) * gp.quicksum([w_y1_yhatobs_a[i] * p_y1_jt_yhatobs[i] * y[i] for i in range(n) ])  - t_*C/(t_*C - 1.0))
        if direction=='max':
            m.setObjective(expr, gp.GRB.MAXIMIZE); m.optimize()
        else:
            m.setObjective(expr, gp.GRB.MINIMIZE); m.optimize()
        if (m.status == gp.GRB.OPTIMAL):
            wghts_ = [w_y1_yhatobs_a, w_y0_yhatobs_a, w_y1_yhatnotobs_a, w_y0_yhatnotobs_a]
            wghts__ = [ [ ww.X for ww in wghts_[ind] ] for ind in range(len(wghts_)) ]
            res[t_ind+1] = [m.ObjVal, wghts__, t]
        else:
            res[t_ind+1] =  [np.nan, 0, 0]
    if save:
        pickle.dump({'res':res, 'args': {'ts':ts, 'LIPSCH_CONST':LIPSCH_CONST} }, open('out/opt_over_t_res_'+savestr+'.pkl', 'wb')  )
    return res

'''
Solve the tpr disparity for a range of t values
resolve with warm starting
Inputs:
t: scaling factor; We use an asymptotically well-behaved scaling omega = W t_a n_a
y: objective coefficients
p_yhat_cond_y: Pr(Y=1 | hat Y = 1, Z_i)
p_yhat_jt_y: Pr(Y=1 , hat Y = 1 | Z_i)
b_: upper bound
If smoothing: add smoothness constraints
return primal weights
Return weights in the order [(1,1),(0,1),(1,0),(0,0)]
return scaling t
'''
def get_tpr_disp_two_races(ts, y, conditionals, joints, p_a_z, LIPSCH_CONST,
dists_mat, LAW_TOT_PROB_SLACK=0, direction='max', quiet=True, smoothing=None,
save = True, savestr='tpr_disp_two_races', is_in_parallel_call=False):
    m = gp.Model()
    res = [ None ] * len(ts)
    if quiet: m.setParam("OutputFlag", 0)
    n = len(y)
    w_y1_yhatobs_a = [m.addVar( lb = 0., vtype=gp.GRB.CONTINUOUS) for yy in y]
    w_y0_yhatobs_a = [m.addVar( lb = 0.,  vtype=gp.GRB.CONTINUOUS) for yy in y]
    w_y1_yhatnotobs_a = [m.addVar( lb = 0.,  vtype=gp.GRB.CONTINUOUS) for yy in y]
    w_y0_yhatnotobs_a = [m.addVar( lb = 0., vtype=gp.GRB.CONTINUOUS) for yy in y]

    [p_y1_cnd_yhat1, p_y0_cnd_yhat1, p_y1_cnd_yhat0, p_y0_cnd_yhat0] = conditionals
    [ p_y1_jt_yhat1, p_y0_jt_yhat1, p_y1_jt_yhat0, p_y0_jt_yhat0 ] = joints
    p_y1_jt_yhatobs = [ p_y1_jt_yhat1[i] if y[i] == 1 else 1-p_y1_jt_yhat0[i] for i in range((n)) ]
    p_y0_jt_yhatobs = [ p_y0_jt_yhat1[i] if y[i] == 1 else 1-p_y0_jt_yhat0[i] for i in range((n)) ]
    p_y1_jt_yhatnotobs = [ 1-p_y1_jt_yhat0[i] if y[i] == 1 else p_y1_jt_yhat1[i] for i in range((n)) ]
    p_y0_jt_yhatnotobs = [ 1-p_y0_jt_yhat0[i] if y[i] == 1 else p_y0_jt_yhat1[i] for i in range((n)) ]
    C = sum(p_y1_jt_yhatobs)

    m.update()
    t = ts[0] # build the model first
    m.addConstr(gp.quicksum([w_y1_yhatobs_a[i] * p_y1_jt_yhatobs[i] for i in range(n) ]) == n, name = 'homogenization') # need to fix the homogenization
    for i in range(len(y)):
        m.addConstr( p_y0_jt_yhatobs[i]*w_y0_yhatobs_a[i] + p_y1_jt_yhatobs[i]*w_y1_yhatobs_a[i]
        + p_y1_jt_yhatnotobs[i]*w_y1_yhatnotobs_a[i] + p_y0_jt_yhatnotobs[i]*w_y0_yhatnotobs_a[i] == p_a_z[i]*t , name='LTP'+str(i) )
        # m.addConstr(w_y1_a[i] <= n*t, 'omega-y=1'+str(i) )
        # m.addConstr(w_y0_a[i] <= n*t, 'omega-y=0'+str(i) )
        # m.addConstr(w_y1_yhat_a[i] <= n*t, 'omega-y=1_1-hy'+str(i) )
        # m.addConstr(w_y0_yhat_a[i] <= n*t, 'omega-y=0_1-hy'+str(i) )
    if smoothing:
        m.addConstrs((w_y0_yhatobs_a[i] - w_y0_yhatobs_a[j] <= LIPSCH_CONST*dists_mat[i,j] * t * n for i in range(n) for j in range(n) ) , name='L-y=0')
        m.addConstrs((-w_y0_yhatobs_a[i] + w_y0_yhatobs_a[j] <= LIPSCH_CONST*dists_mat[i,j] * t * n for i in range(n) for j in range(n) ) , name='-L-y=0')
        m.addConstrs((w_y1_yhatobs_a[i] - w_y1_yhatobs_a[j] <= LIPSCH_CONST*dists_mat[i,j] * t * n for i in range(n) for j in range(n) ) , name='L-y=1')
        m.addConstrs((-w_y1_yhatobs_a[i] + w_y1_yhatobs_a[j] <= LIPSCH_CONST*dists_mat[i,j] * t * n for i in range(n) for j in range(n) ) , name='-L-y=1')

    expr = gp.LinExpr();
    expr += 1.0/n * (t*C/(t*C - 1.0) * gp.quicksum([w_y1_yhatobs_a[i] * p_y1_jt_yhatobs[i] * y[i] for i in range(n) ])  - 1.0/n *t*C/(t*C - 1.0))
    if direction=='max':
        m.setObjective(expr, gp.GRB.MAXIMIZE); m.optimize()
    else:
        m.setObjective(expr, gp.GRB.MINIMIZE); m.optimize()

    if (m.status == gp.GRB.OPTIMAL):
        wghts_ = [w_y1_yhatobs_a, w_y0_yhatobs_a, w_y1_yhatnotobs_a, w_y0_yhatnotobs_a]
        wghts__ = [ [ ww.X for ww in wghts_[ind] ] for ind in range(len(wghts_)) ]
        res[0] = [m.ObjVal, wghts__, t]
    else:
        res[0] =  [np.nan, 0, 0]
    if is_in_parallel_call:
        return res[0]
    # Change model constraint RHSes and objective and reoptimize
    for t_ind, t_ in enumerate(ts[1:]):
        for i in range(n):
            m.getConstrByName('LTP'+str(i)).RHS = p_a_z[i]*t_
        if smoothing:
            for i in range(n):
                for j in range(n)[i:]:
                    m.getConstrByName('L-y=0'+str(i)+','+str(j)+']').RHS = LIPSCH_CONST*dists_mat[i,j]* t_ * n
                    m.getConstrByName('L-y=0'+str(i)+','+str(j)+']').RHS = LIPSCH_CONST*dists_mat[i,j]* t_ * n
                for k in range(n_b)[i:]:
                    m.getConstrByName('L-y=1'+str(i)+','+str(k)+']').RHS = LIPSCH_CONST*dists_mat[i,j]* t_ * n
                    m.getConstrByName('L-y=1'+str(i)+','+str(k)+']').RHS = LIPSCH_CONST*dists_mat[i,j]* t_ * n
        expr = gp.LinExpr();
        expr += 1.0/n *( t_*C/(t_*C - 1.0) * gp.quicksum([w_y1_yhatobs_a[i] * p_y1_jt_yhatobs[i] * y[i] for i in range(n) ])  - t_*C/(t_*C - 1.0))
        if direction=='max':
            m.setObjective(expr, gp.GRB.MAXIMIZE); m.optimize()
        else:
            m.setObjective(expr, gp.GRB.MINIMIZE); m.optimize()
        if (m.status == gp.GRB.OPTIMAL):
            wghts_ = [w_y1_yhatobs_a, w_y0_yhatobs_a, w_y1_yhatnotobs_a, w_y0_yhatnotobs_a]
            wghts__ = [ [ ww.X for ww in wghts_[ind] ] for ind in range(len(wghts_)) ]
            res[t_ind+1] = [m.ObjVal, wghts__, t]
        else:
            res[t_ind+1] =  [np.nan, 0, 0]
    if save:
        pickle.dump({'res':res, 'args': {'ts':ts, 'LIPSCH_CONST':LIPSCH_CONST} }, open('out/opt_over_t_res_'+savestr+'.pkl', 'wb')  )
    return res



'''
get disparity curve over different values of eta
for smoothness within each partition
Calls as a subroutine the disparity function with signature

'''
def get_disp_curve_over_etas_Budgeted_Smoothness(ts, y, conditionals, joints, p_a_z, LIPSCH_CONST,
dists_mat, LAW_TOT_PROB_SLACK=0, direction='max', quiet=True, smoothing=None,
save = True, savestr='tpr_disp_two_races', is_in_parallel_call=False):

    res_ = Parallel(n_jobs=N_JOBS, verbose = vbs)(get_disp_curve_over_etas_Budgeted_Smoothness(t_, y, conditionals, joints, p_a_z, LIPSCH_CONST,
    dists_mat, LAW_TOT_PROB_SLACK=LAW_TOT_PROB_SLACK, direction=direction, quiet=quiet, smoothing=smoothing,
    save = True, savestr=savestr, is_in_parallel_call=is_in_parallel_call) for t_ in ts)
    objs__ = np.asarray([x[0] for x in res_]).reshape([len(cate_pctiles), len(etas)]);
    return [ objs__, res_]


'''
###### Benchmark program for a single TPR
Solve the tpr for a range of t values
resolve with warm starting
Inputs:
t: scaling factor; We use an asymptotically well-behaved scaling omega = W t_a n_a
y: objective coefficients
e.g. Yhat
p_yhat_cond_y: Pr(Y=1 | hat Y = 1, Z_i)
p_yhat_jt_y: Pr(Y=1 , hat Y = 1 | Z_i)
b_: upper bound
If smoothing: add smoothness constraints
return primal weights
Return weights in the order [(1,1),(0,1),(1,0),(0,0)]
return scaling t
'''
def get_tpr_indicator_Y(ts, Y, Yhat, joints, p_a_z, LIPSCH_CONST,
dists_mat, LAW_TOT_PROB_SLACK=0, direction='max', quiet=True, smoothing=None,
save = True, savestr='tpr'):
    m = gp.Model()
    res = [ None ] * len(ts)
    if quiet: m.setParam("OutputFlag", 0)
    m.setParam("Method", 1);
    y = Yhat
    n = len(y)
    w_y1_yhatobs_a = [m.addVar( lb = 0., vtype=gp.GRB.CONTINUOUS) for yy in y]
    w_y0_yhatobs_a = [m.addVar( lb = 0.,  vtype=gp.GRB.CONTINUOUS) for yy in y]
    w_y1_yhatnotobs_a = [m.addVar( lb = 0.,  vtype=gp.GRB.CONTINUOUS) for yy in y]
    w_y0_yhatnotobs_a = [m.addVar( lb = 0., vtype=gp.GRB.CONTINUOUS) for yy in y]
    t = m.addVar( lb = 0., vtype=gp.GRB.CONTINUOUS)
    [ p_y1_jt_yhat1, p_y0_jt_yhat1, p_y1_jt_yhat0, p_y0_jt_yhat0 ] = joints
    p_y1_jt_yhatobs = [ p_y1_jt_yhat1[i] if y[i] == 1 else p_y1_jt_yhat0[i] for i in range((n)) ]
    p_y0_jt_yhatobs = [ p_y0_jt_yhat1[i] if y[i] == 1 else p_y0_jt_yhat0[i] for i in range((n)) ]
    p_y1_jt_yhatnotobs = [ p_y1_jt_yhat0[i] if y[i] == 1 else p_y1_jt_yhat1[i] for i in range((n)) ]
    p_y0_jt_yhatnotobs = [ p_y0_jt_yhat0[i] if y[i] == 1 else p_y0_jt_yhat1[i] for i in range((n)) ]

    C = sum(Y)
    print 'adding variables'
    m.update()
    # t = ts[0] # build the model first
    m.addConstr(gp.quicksum([w_y1_yhatobs_a[i] * Y[i] for i in range(n) ]) == n, name = 'homogenization') # need to fix the homogenization
    m.addConstrs((w_y1_yhatobs_a[i]  <= t * n for i in range(n) ))
    m.addConstrs((w_y0_yhatobs_a[i]  <= t * n for i in range(n) ))
    m.addConstrs((w_y1_yhatnotobs_a[i]  <= t * n for i in range(n) ))
    m.addConstrs((w_y0_yhatnotobs_a[i]  <= t * n for i in range(n) ))

    for i in range(len(y)):
        m.addConstr( p_y0_jt_yhatobs[i]*w_y0_yhatobs_a[i] + p_y1_jt_yhatobs[i]*w_y1_yhatobs_a[i] + p_y1_jt_yhatnotobs[i]*w_y1_yhatnotobs_a[i] + p_y0_jt_yhatnotobs[i]*w_y0_yhatnotobs_a[i] == p_a_z[i]*t*n , name='LTP'+str(i))

    if smoothing:
        m.addConstrs((w_y0_yhatobs_a[i] - w_y0_yhatobs_a[j] <= LIPSCH_CONST*dists_mat[i,j] * t * n for i in range(n) for j in range(n)[i:] ) , name='L-y=0')
        m.addConstrs((-w_y0_yhatobs_a[i] + w_y0_yhatobs_a[j] <= LIPSCH_CONST*dists_mat[i,j] * t * n for i in range(n) for j in range(n)[i:] ) , name='-L-y=0')
        m.addConstrs((w_y1_yhatobs_a[i] - w_y1_yhatobs_a[j] <= LIPSCH_CONST*dists_mat[i,j] * t * n for i in range(n) for j in range(n)[i:] ) , name='L-y=1')
        m.addConstrs((-w_y1_yhatobs_a[i] + w_y1_yhatobs_a[j] <= LIPSCH_CONST*dists_mat[i,j] * t * n  for i in range(n) for j in range(n)[i:] ) , name='-L-y=1')
    print 'adding constraints'
    expr = gp.LinExpr();
    expr += 1.0/n *  gp.quicksum([w_y1_yhatobs_a[i] * Y[i] * Yhat[i] for i in range(n) ])
    if direction=='max':
        print 'solving'
        m.setObjective(expr, gp.GRB.MAXIMIZE); m.optimize()
    else:
        m.setObjective(expr, gp.GRB.MINIMIZE); m.optimize()

    if (m.status == gp.GRB.OPTIMAL):
        wghts_ = [w_y1_yhatobs_a, w_y0_yhatobs_a, w_y1_yhatnotobs_a, w_y0_yhatnotobs_a]
        wghts__ = [ [ ww.X for ww in wghts_[ind] ] for ind in range(len(wghts_)) ]
        res[0] = [m.ObjVal, wghts__, t.X]
    else:
        res[0] =  [np.nan, 0, 0]
    #
    if save:
        pickle.dump({'res':res, 'args': {'ts':ts, 'LIPSCH_CONST':LIPSCH_CONST} }, open('out/opt_over_t_res_'+savestr+'.pkl', 'wb')  )
    return res



'''
Solve the tpr for a range of t values
resolve with warm starting
Inputs:
t: scaling factor; We use an asymptotically well-behaved scaling omega = W t_a n_a
y: objective coefficients
e.g. Yhat
p_yhat_cond_y: Pr(Y=1 | hat Y = 1, Z_i)
p_yhat_jt_y: Pr(Y=1 , hat Y = 1 | Z_i)
b_: upper bound
If smoothing: add smoothness constraints
return primal weights
Return weights in the order [(1,1),(0,1),(1,0),(0,0)]
return scaling t
'''
def get_tpr_disp_indicator_Y_over_ts(ts, Y, Yhat, joints, p_a_z, LIPSCH_CONST,
dists_mat, LAW_TOT_PROB_SLACK=0, direction='max', quiet=True, smoothing=None,
save = True, savestr='tpr'):
    m = gp.Model()
    res = [ None ] * len(ts)
    if quiet: m.setParam("OutputFlag", 0)
    m.setParam("Method", 1);
    y = Yhat
    n = len(y)
    w_y1_yhatobs_a = [m.addVar( lb = 0., vtype=gp.GRB.CONTINUOUS) for yy in y]
    w_y0_yhatobs_a = [m.addVar( lb = 0.,  vtype=gp.GRB.CONTINUOUS) for yy in y]
    w_y1_yhatnotobs_a = [m.addVar( lb = 0.,  vtype=gp.GRB.CONTINUOUS) for yy in y]
    w_y0_yhatnotobs_a = [m.addVar( lb = 0., vtype=gp.GRB.CONTINUOUS) for yy in y]
    # t = m.addVar( lb = 0., vtype=gp.GRB.CONTINUOUS)
    ## changing: use the definition that t = 1/ E[ w Y=y ], u = w t
    [ p_y1_jt_yhat1, p_y0_jt_yhat1, p_y1_jt_yhat0, p_y0_jt_yhat0 ] = joints
    print n
    print len(p_y1_jt_yhat1)
    p_y1_jt_yhatobs = [ p_y1_jt_yhat1[i] if y[i] == 1 else p_y1_jt_yhat0[i] for i in range((n)) ]
    p_y0_jt_yhatobs = [ p_y0_jt_yhat1[i] if y[i] == 1 else p_y0_jt_yhat0[i] for i in range((n)) ]
    p_y1_jt_yhatnotobs = [ p_y1_jt_yhat0[i] if y[i] == 1 else p_y1_jt_yhat1[i] for i in range((n)) ]
    p_y0_jt_yhatnotobs = [ p_y0_jt_yhat0[i] if y[i] == 1 else p_y0_jt_yhat1[i] for i in range((n)) ]

    C = np.mean(Y)
    print 'adding variables'
    m.update()
    t = ts[0] # build the model first

    m.addConstr(gp.quicksum([w_y1_yhatobs_a[i] * Y[i] for i in range(n) ]) == n, name = 'homogenization') # need to fix the homogenization
    m.addConstrs((w_y1_yhatobs_a[i]  <= t  for i in range(n) ), name='Y1Yhatobs')
    m.addConstrs((w_y0_yhatobs_a[i]  <= t  for i in range(n) ), name='Y0Yhatobs')
    m.addConstrs((w_y1_yhatnotobs_a[i]  <= t  for i in range(n) ), name='Y1Yhatnotobs')
    m.addConstrs((w_y0_yhatnotobs_a[i]  <= t  for i in range(n) ), name='Y0Yhatnotobs')
    # m.addConstrs((w_y1_yhatobs_a[i]  <= t * n for i in range(n) ), name='Y1Yhatobs')
    # m.addConstrs((w_y0_yhatobs_a[i]  <= t * n for i in range(n) ), name='Y0Yhatobs')
    # m.addConstrs((w_y1_yhatnotobs_a[i]  <= t * n for i in range(n) ), name='Y1Yhatnotobs')
    # m.addConstrs((w_y0_yhatnotobs_a[i]  <= t * n for i in range(n) ), name='Y0Yhatnotobs')

    for i in range(len(y)):
        m.addConstr( p_y0_jt_yhatobs[i]*w_y0_yhatobs_a[i] + p_y1_jt_yhatobs[i]*w_y1_yhatobs_a[i] + p_y1_jt_yhatnotobs[i]*w_y1_yhatnotobs_a[i] + p_y0_jt_yhatnotobs[i]*w_y0_yhatnotobs_a[i] == p_a_z[i]*t , name='LTP'+str(i))

    if smoothing:
        m.addConstrs((w_y0_yhatobs_a[i] - w_y0_yhatobs_a[j] <= LIPSCH_CONST*dists_mat[i,j] * t for i in range(n) for j in range(n)[i:] ) , name='L-y=0')
        m.addConstrs((-w_y0_yhatobs_a[i] + w_y0_yhatobs_a[j] <= LIPSCH_CONST*dists_mat[i,j] * t for i in range(n) for j in range(n)[i:] ) , name='-L-y=0')
        m.addConstrs((w_y1_yhatobs_a[i] - w_y1_yhatobs_a[j] <= LIPSCH_CONST*dists_mat[i,j] * t for i in range(n) for j in range(n)[i:] ) , name='L-y=1')
        m.addConstrs((-w_y1_yhatobs_a[i] + w_y1_yhatobs_a[j] <= LIPSCH_CONST*dists_mat[i,j] * t  for i in range(n) for j in range(n)[i:] ) , name='-L-y=1')
    print 'adding constraints'
    expr = gp.LinExpr();
    expr += ( t*C/(t*C - 1.0) * 1.0/n * gp.quicksum([w_y1_yhatobs_a[i] * Y[i] * Yhat[i] for i in range(n) ])  -  np.mean(Yhat*Y) * t/(t*C - 1.0) )
    if direction=='max':
        print 'solving'
        m.setObjective(expr, gp.GRB.MAXIMIZE); m.optimize()
    else:
        m.setObjective(expr, gp.GRB.MINIMIZE); m.optimize()

    if (m.status == gp.GRB.OPTIMAL):
        wghts_ = [w_y1_yhatobs_a, w_y0_yhatobs_a, w_y1_yhatnotobs_a, w_y0_yhatnotobs_a]
        wghts__ = [ [ ww.X for ww in wghts_[ind] ] for ind in range(len(wghts_)) ]
        res[0] = [m.ObjVal, wghts__, t]
    else:
        res[0] =  [np.nan, 0, 0]
    #
    # Change model constraint RHSes and objective and reoptimize
    for t_ind, t_ in enumerate(ts[1:]):
        # for i in range(n):
        #     m.getConstrByName('LTP'+str(i)).RHS = p_a_z[i]*t_*n
        # for i in range(n):
        #     m.getConstrByName('Y1Yhatobs['+str(i)+']').RHS = t_ * n
        #     m.getConstrByName('Y0Yhatobs['+str(i)+']').RHS = t_ * n
        #     m.getConstrByName('Y1Yhatnotobs['+str(i)+']').RHS = t_ * n
        #     m.getConstrByName('Y0Yhatnotobs['+str(i)+']').RHS = t_ * n
        for i in range(n):
            m.getConstrByName('LTP'+str(i)).RHS = p_a_z[i]*t_
        for i in range(n):
            m.getConstrByName('Y1Yhatobs['+str(i)+']').RHS = t_
            m.getConstrByName('Y0Yhatobs['+str(i)+']').RHS = t_
            m.getConstrByName('Y1Yhatnotobs['+str(i)+']').RHS = t_
            m.getConstrByName('Y0Yhatnotobs['+str(i)+']').RHS = t_
        if smoothing:
            for i in range(n):
                for j in range(n)[i:]:
                    m.getConstrByName('L-y=0'+str(i)+','+str(j)+']').RHS = LIPSCH_CONST*dists_mat[i,j]* t_
                    m.getConstrByName('-L-y=0'+str(i)+','+str(j)+']').RHS = LIPSCH_CONST*dists_mat[i,j]* t_
                for k in range(n_b)[i:]:
                    m.getConstrByName('L-y=1'+str(i)+','+str(k)+']').RHS = LIPSCH_CONST*dists_mat[i,j]* t_
                    m.getConstrByName('-L-y=1'+str(i)+','+str(k)+']').RHS = LIPSCH_CONST*dists_mat[i,j]* t_
        expr = gp.LinExpr();
        expr += ( t_*C/(t_*C - 1.0) * 1.0/n * gp.quicksum([w_y1_yhatobs_a[i] * Y[i] * Yhat[i] for i in range(n) ])  -  np.mean(Yhat*Y) * t_/(t_*C - 1.0) )
        # expr += ( t_*C/(t_*C - 1.0) * gp.quicksum([w_y1_yhatobs_a[i] * Y[i] * Yhat[i] for i in range(n) ])  -  np.mean(Yhat*Y) * t_*C/(t_*C - 1.0))
        if direction=='max':
            m.setObjective(expr, gp.GRB.MAXIMIZE); m.optimize()
        else:
            m.setObjective(expr, gp.GRB.MINIMIZE); m.optimize()
        if (m.status == gp.GRB.OPTIMAL):
            wghts_ = [w_y1_yhatobs_a, w_y0_yhatobs_a, w_y1_yhatnotobs_a, w_y0_yhatnotobs_a]
            wghts__ = [ [ ww.X for ww in wghts_[ind] ] for ind in range(len(wghts_)) ]
            res[t_ind+1] = [m.ObjVal, wghts__, t_]
        else:
            res[t_ind+1] =  [np.nan, 0, 0]
    if save:
        pickle.dump({'res':res, 'args': {'ts':ts, 'LIPSCH_CONST':LIPSCH_CONST} }, open('out/opt_over_t_res_'+savestr+'.pkl', 'wb')  )
    return res

'''
Solve feasibility program over range of ts
Need to set 'type': tpr or tnr disparity to get the right homogenization constraint
'''
def get_t_range_feasibility(Y, Yhat, joints, p_a_z, LIPSCH_CONST,
dists_mat, LAW_TOT_PROB_SLACK=0, direction='max',
type = 'tpr', quiet=True, smoothing=None,
save = True, savestr='tpr'):
    m = gp.Model()

    if quiet: m.setParam("OutputFlag", 0)
    m.setParam("Method", 1);
    y = Yhat
    n = len(y)
    w_y1_yhatobs_a = [m.addVar( lb = 0., vtype=gp.GRB.CONTINUOUS) for yy in y]
    w_y0_yhatobs_a = [m.addVar( lb = 0.,  vtype=gp.GRB.CONTINUOUS) for yy in y]
    w_y1_yhatnotobs_a = [m.addVar( lb = 0.,  vtype=gp.GRB.CONTINUOUS) for yy in y]
    w_y0_yhatnotobs_a = [m.addVar( lb = 0., vtype=gp.GRB.CONTINUOUS) for yy in y]
    t = m.addVar( lb = 0., vtype=gp.GRB.CONTINUOUS)
    [ p_y1_jt_yhat1, p_y0_jt_yhat1, p_y1_jt_yhat0, p_y0_jt_yhat0 ] = joints
    p_y1_jt_yhatobs = [ p_y1_jt_yhat1[i] if y[i] == 1 else p_y1_jt_yhat0[i] for i in range((n)) ]
    p_y0_jt_yhatobs = [ p_y0_jt_yhat1[i] if y[i] == 1 else p_y0_jt_yhat0[i] for i in range((n)) ]
    p_y1_jt_yhatnotobs = [ p_y1_jt_yhat0[i] if y[i] == 1 else p_y1_jt_yhat1[i] for i in range((n)) ]
    p_y0_jt_yhatnotobs = [ p_y0_jt_yhat0[i] if y[i] == 1 else p_y0_jt_yhat1[i] for i in range((n)) ]

    C = np.mean(Y)
    print 'adding variables'
    m.update()
    if type == 'tpr':
        m.addConstr(gp.quicksum([w_y1_yhatobs_a[i] * Y[i] for i in range(n) ]) == n, name = 'homogenization') # need to fix the homogenization
    else:
        m.addConstr(gp.quicksum([w_y0_yhatobs_a[i] * (1-Y[i]) for i in range(n) ]) == n, name = 'homogenization') # need to fix the homogenization
    m.addConstrs((w_y1_yhatobs_a[i]  <= t  for i in range(n) ), name='Y1Yhatobs')
    m.addConstrs((w_y0_yhatobs_a[i]  <= t  for i in range(n) ), name='Y0Yhatobs')
    m.addConstrs((w_y1_yhatnotobs_a[i]  <= t  for i in range(n) ), name='Y1Yhatnotobs')
    m.addConstrs((w_y0_yhatnotobs_a[i]  <= t  for i in range(n) ), name='Y0Yhatnotobs')
    # m.addConstrs((w_y1_yhatobs_a[i]  <= t * n for i in range(n) ))
    # m.addConstrs((w_y0_yhatobs_a[i]  <= t * n for i in range(n) ))
    # m.addConstrs((w_y1_yhatnotobs_a[i]  <= t * n for i in range(n) ))
    # m.addConstrs((w_y0_yhatnotobs_a[i]  <= t * n for i in range(n) ))

    for i in range(len(y)):
        m.addConstr( p_y0_jt_yhatobs[i]*w_y0_yhatobs_a[i] + p_y1_jt_yhatobs[i]*w_y1_yhatobs_a[i] + p_y1_jt_yhatnotobs[i]*w_y1_yhatnotobs_a[i] + p_y0_jt_yhatnotobs[i]*w_y0_yhatnotobs_a[i] == p_a_z[i]*t , name='LTP'+str(i))

    if smoothing:
        m.addConstrs((w_y0_yhatobs_a[i] - w_y0_yhatobs_a[j] <= LIPSCH_CONST*dists_mat[i,j] * t  for i in range(n) for j in range(n)[i:] ) , name='L-y=0')
        m.addConstrs((-w_y0_yhatobs_a[i] + w_y0_yhatobs_a[j] <= LIPSCH_CONST*dists_mat[i,j] * t  for i in range(n) for j in range(n)[i:] ) , name='-L-y=0')
        m.addConstrs((w_y1_yhatobs_a[i] - w_y1_yhatobs_a[j] <= LIPSCH_CONST*dists_mat[i,j] * t  for i in range(n) for j in range(n)[i:] ) , name='L-y=1')
        m.addConstrs((-w_y1_yhatobs_a[i] + w_y1_yhatobs_a[j] <= LIPSCH_CONST*dists_mat[i,j] * t  for i in range(n) for j in range(n)[i:] ) , name='-L-y=1')
    print 'adding constraints'

    if direction=='max':
        print 'solving'
        m.setObjective(t, gp.GRB.MAXIMIZE); m.optimize()
    else:
        m.setObjective(t, gp.GRB.MINIMIZE); m.optimize()

    if (m.status == gp.GRB.OPTIMAL):
        wghts_ = [w_y1_yhatobs_a, w_y0_yhatobs_a, w_y1_yhatnotobs_a, w_y0_yhatnotobs_a]
        wghts__ = [ [ ww.X for ww in wghts_[ind] ] for ind in range(len(wghts_)) ]
        return [m.ObjVal, wghts__, t.X]
    else:
        return  [np.nan, 0, 0]

    if save:
        pickle.dump({'res':res, 'args': {'ts':ts, 'LIPSCH_CONST':LIPSCH_CONST} }, open('out/opt_over_t_res_'+savestr+'.pkl', 'wb')  )
    return res


'''
Solve the tpr for a range of t values
resolve with warm starting
Inputs:
t: scaling factor; We use an asymptotically well-behaved scaling omega = W t_a n_a
y: objective coefficients
e.g. Yhat
yfix: what value to fix Y to
hyfix: what value to fix \hat Y to
p_yhat_cond_y: Pr(Y=1 | hat Y = 1, Z_i)
p_yhat_jt_y: Pr(Y=1 , hat Y = 1 | Z_i)
b_: upper bound
If smoothing: add smoothness constraints
return primal weights
Return weights in the order [(1,1),(0,1),(1,0),(0,0)]
return scaling t
'''
def get_disp_in_Y_Yhat_fixed_t(t, Y, Yhat, yfix, hyfix, joints, p_a_z, LIPSCH_CONST,
dists_mat, LAW_TOT_PROB_SLACK=0, direction='max', quiet=True, smoothing=None,
save = True, savestr='tpr'):
    m = gp.Model()
    res = [ None ] * len(ts)
    if quiet: m.setParam("OutputFlag", 0)
    m.setParam("Method", 1);
    y = Yhat
    n = len(y)
    w_y1_yhatobs_a = [m.addVar( lb = 0., vtype=gp.GRB.CONTINUOUS) for yy in y]
    w_y0_yhatobs_a = [m.addVar( lb = 0.,  vtype=gp.GRB.CONTINUOUS) for yy in y]
    w_y1_yhatnotobs_a = [m.addVar( lb = 0.,  vtype=gp.GRB.CONTINUOUS) for yy in y]
    w_y0_yhatnotobs_a = [m.addVar( lb = 0., vtype=gp.GRB.CONTINUOUS) for yy in y]
    # t = m.addVar( lb = 0., vtype=gp.GRB.CONTINUOUS)
    ## changing: use the definition that t = 1/ E[ w Y=y ], u = w t
    [ p_y1_jt_yhat1, p_y0_jt_yhat1, p_y1_jt_yhat0, p_y0_jt_yhat0 ] = joints
    print n
    print len(p_y1_jt_yhat1)
    p_y1_jt_yhatobs = [ p_y1_jt_yhat1[i] if y[i] == 1 else p_y1_jt_yhat0[i] for i in range((n)) ]
    p_y0_jt_yhatobs = [ p_y0_jt_yhat1[i] if y[i] == 1 else p_y0_jt_yhat0[i] for i in range((n)) ]
    p_y1_jt_yhatnotobs = [ p_y1_jt_yhat0[i] if y[i] == 1 else p_y1_jt_yhat1[i] for i in range((n)) ]
    p_y0_jt_yhatnotobs = [ p_y0_jt_yhat0[i] if y[i] == 1 else p_y0_jt_yhat1[i] for i in range((n)) ]

    C = np.mean(Y)
    print 'adding variables'
    m.update()
    t = ts[0] # build the model first

    m.addConstr(gp.quicksum([w_y1_yhatobs_a[i] * Y[i] for i in range(n) ]) == n, name = 'homogenization') # need to fix the homogenization
    m.addConstrs((w_y1_yhatobs_a[i]  <= t  for i in range(n) ), name='Y1Yhatobs')
    m.addConstrs((w_y0_yhatobs_a[i]  <= t  for i in range(n) ), name='Y0Yhatobs')
    m.addConstrs((w_y1_yhatnotobs_a[i]  <= t  for i in range(n) ), name='Y1Yhatnotobs')
    m.addConstrs((w_y0_yhatnotobs_a[i]  <= t  for i in range(n) ), name='Y0Yhatnotobs')
    # m.addConstrs((w_y1_yhatobs_a[i]  <= t * n for i in range(n) ), name='Y1Yhatobs')
    # m.addConstrs((w_y0_yhatobs_a[i]  <= t * n for i in range(n) ), name='Y0Yhatobs')
    # m.addConstrs((w_y1_yhatnotobs_a[i]  <= t * n for i in range(n) ), name='Y1Yhatnotobs')
    # m.addConstrs((w_y0_yhatnotobs_a[i]  <= t * n for i in range(n) ), name='Y0Yhatnotobs')

    for i in range(len(y)):
        m.addConstr( p_y0_jt_yhatobs[i]*w_y0_yhatobs_a[i] + p_y1_jt_yhatobs[i]*w_y1_yhatobs_a[i] + p_y1_jt_yhatnotobs[i]*w_y1_yhatnotobs_a[i] + p_y0_jt_yhatnotobs[i]*w_y0_yhatnotobs_a[i] == p_a_z[i]*t , name='LTP'+str(i))

    if smoothing:
        m.addConstrs((w_y0_yhatobs_a[i] - w_y0_yhatobs_a[j] <= LIPSCH_CONST*dists_mat[i,j] * t for i in range(n) for j in range(n)[i:] ) , name='L-y=0')
        m.addConstrs((-w_y0_yhatobs_a[i] + w_y0_yhatobs_a[j] <= LIPSCH_CONST*dists_mat[i,j] * t for i in range(n) for j in range(n)[i:] ) , name='-L-y=0')
        m.addConstrs((w_y1_yhatobs_a[i] - w_y1_yhatobs_a[j] <= LIPSCH_CONST*dists_mat[i,j] * t for i in range(n) for j in range(n)[i:] ) , name='L-y=1')
        m.addConstrs((-w_y1_yhatobs_a[i] + w_y1_yhatobs_a[j] <= LIPSCH_CONST*dists_mat[i,j] * t  for i in range(n) for j in range(n)[i:] ) , name='-L-y=1')
    print 'adding constraints'
    expr = gp.LinExpr();
    expr += ( t*C/(t*C - 1.0) * 1.0/n * gp.quicksum([w_y1_yhatobs_a[i] * Y[i] * Yhat[i] for i in range(n) ])  -  np.mean(Yhat*Y) * t/(t*C - 1.0) )
    if direction=='max':
        print 'solving'
        m.setObjective(expr, gp.GRB.MAXIMIZE); m.optimize()
    else:
        m.setObjective(expr, gp.GRB.MINIMIZE); m.optimize()

    if (m.status == gp.GRB.OPTIMAL):
        wghts_ = [w_y1_yhatobs_a, w_y0_yhatobs_a, w_y1_yhatnotobs_a, w_y0_yhatnotobs_a]
        wghts__ = [ [ ww.X for ww in wghts_[ind] ] for ind in range(len(wghts_)) ]
        res = [m.ObjVal, wghts__, t]
    else:
        res =  [np.nan, 0, 0]
    if save:
        pickle.dump({'res':res, 'args': {'ts':ts, 'LIPSCH_CONST':LIPSCH_CONST} }, open('out/opt_over_t_res_'+savestr+'.pkl', 'wb')  )
    return res


'''
y: outcomes
n_as: number of classes
mu_scalar: (n_A - 1) number of values (scalarization on disparities)
p_az: (n_a, n) array of probabilities
p_yz: (n) array of probabilities
Z: proxies
LIPSCH_CONST:
'''
def get_dem_disp_obs_outcomes_multi_race(y, n_as, mu_scalar, p_az, p_yz,
 Z, LIPSCH_CONST,feasibility_relax = False, LAW_TOT_PROB_SLACK=0, quiet=True, smoothing=None,
 smoothing1d=True, direction = 'max'):
    n = len(y); m = gp.Model()
    p_as = (p_az.sum(axis=0)/p_az.shape[0])
    if (np.asarray([len(a) for a in [p_az, p_yz, Z]]) != n).any():
        raise ValueError("data inputs must have the same length")
    if quiet:
        m.setParam("OutputFlag", 0)
#     t = m.addVar(lb = 0., ub = gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS)
    w_y1 = m.addVars(range(n_as), len(y), lb = 0., ub = 1, vtype=gp.GRB.CONTINUOUS)
    w_y0 = m.addVars(range(n_as), len(y), lb = 0., ub = 1, vtype=gp.GRB.CONTINUOUS)

    # sorteddiffs = m.addVars( range(n_as), len(y)-1, lb = 0., ub = 1, vtype=gp.GRB.CONTINUOUS)
    # sorteddiffs_0 = m.addVars( range(n_as), len(y)-1, lb = 0., ub = 1, vtype=gp.GRB.CONTINUOUS)
    # wdiffs_1 = m.addVars( range(n_as), len(y)-1, lb = 0., ub = 1, vtype=gp.GRB.CONTINUOUS)
    # wdiffs_0 = m.addVars( range(n_as), len(y)-1, lb = 0., ub = 1, vtype=gp.GRB.CONTINUOUS)

    m.update()
    # law of tot prob bounds
    for a in range(n_as):
        for i in range(n):
            m.addConstr( w_y1[(a,i)]*p_yz[i] + w_y0[a,i]*(1.0-p_yz[i]) == p_az[i,a]  )
            m.addConstr( gp.quicksum(w_y1[a,i] for a in range(n_as)) == 1 )
            m.addConstr( gp.quicksum(w_y0[a,i] for a in range(n_as)) == 1 )
    print LIPSCH_CONST
    if (smoothing1d) and smoothing:
        print smoothing
        sort_ind = np.argsort(Z) #returns sort inds in ascending order
        min_incr = np.min(np.diff(Z[sort_ind])[np.diff(Z[sort_ind])!=0])
        print 'min_incr',min_incr

        ### Lipschtiz constraints for all races
        for a in range(n_as):
        # a = 0 # should be feasible for a = 0
            for i in range(len(y))[:-1]:
                ind = sort_ind[i+1]; ind_minus = sort_ind[i];
                #add constraints directly
                if (Z[ind]- Z[ind_minus]) == 0:
                    m.addConstr( w_y1[a,ind] - w_y1[a,ind_minus] <= LIPSCH_CONST*min_incr, 'smoothness'+str(i) )
                    m.addConstr( -w_y1[a,ind] + w_y1[a,ind_minus] <= LIPSCH_CONST*min_incr, 'smoothness'+str(i) )
                    m.addConstr( w_y0[a,ind] - w_y0[a,ind_minus] <= LIPSCH_CONST*min_incr, 'smoothness'+str(i) )
                    m.addConstr( -w_y0[a,ind] + w_y0[a,ind_minus] <= LIPSCH_CONST*min_incr, 'smoothness'+str(i) )
                else:
                    m.addConstr( w_y1[a,ind] - w_y1[a,ind_minus] <=  LIPSCH_CONST*(Z[ind] - Z[ind_minus]), 'smoothness'+str(i) )
                    m.addConstr( -w_y1[a,ind] + w_y1[a,ind_minus] <= LIPSCH_CONST*(Z[ind] - Z[ind_minus]), 'smoothness'+str(i) )
                    m.addConstr( w_y0[a,ind] - w_y0[a,ind_minus] <= LIPSCH_CONST*(Z[ind] - Z[ind_minus]), 'smoothness'+str(i) )
                    m.addConstr( -w_y0[a,ind] + w_y0[a,ind_minus] <= LIPSCH_CONST*(Z[ind] - Z[ind_minus]), 'smoothness'+str(i) )

        ## Lipschtiz constraints only for white
        # for a in range(n_as):
        # a = 0 # should be feasible for a = 0
        # for i in range(len(y))[:-1]:
        #     ind = sort_ind[i+1]; ind_minus = sort_ind[i];
        #     #add constraints directly
        #     if (Z[ind]- Z[ind_minus]) == 0:
        #         m.addConstr( w_y1[a,ind] - w_y1[a,ind_minus] <= LIPSCH_CONST*min_incr, 'smoothness'+str(i) )
        #         m.addConstr( -w_y1[a,ind] + w_y1[a,ind_minus] <= LIPSCH_CONST*min_incr, 'smoothness'+str(i) )
        #         m.addConstr( w_y0[a,ind] - w_y0[a,ind_minus] <= LIPSCH_CONST*min_incr, 'smoothness'+str(i) )
        #         m.addConstr( -w_y0[a,ind] + w_y0[a,ind_minus] <= LIPSCH_CONST*min_incr, 'smoothness'+str(i) )
        #     else:
        #         m.addConstr( w_y1[a,ind] - w_y1[a,ind_minus] <=  LIPSCH_CONST*(Z[ind] - Z[ind_minus]), 'smoothness'+str(i) )
        #         m.addConstr( -w_y1[a,ind] + w_y1[a,ind_minus] <= LIPSCH_CONST*(Z[ind] - Z[ind_minus]), 'smoothness'+str(i) )
        #         m.addConstr( w_y0[a,ind] - w_y0[a,ind_minus] <= LIPSCH_CONST*(Z[ind] - Z[ind_minus]), 'smoothness'+str(i) )
        #         m.addConstr( -w_y0[a,ind] + w_y0[a,ind_minus] <= LIPSCH_CONST*(Z[ind] - Z[ind_minus]), 'smoothness'+str(i) )
    #         m.addConstr(sorteddiffs[a,i] == gp.abs_(wdiffs_1[a,i]))
    #         m.addConstr( wdiffs_0[a,i] == w_y0[a,ind] - w_y0[a,ind_minus] );
    #         m.addConstr(sorteddiffs_0[a,i] == gp.abs_(wdiffs_0[a,i]))
    #         m.addConstr( wdiffs_1[a,i] == w_y1[a,ind] - w_y1[a,ind_minus] );
    #         m.addConstr(sorteddiffs[a,i] == gp.abs_(wdiffs_1[a,i]))
    #         m.addConstr( wdiffs_0[a,i] == w_y0[a,ind] - w_y0[a,ind_minus] );
    #         m.addConstr(sorteddiffs_0[a,i] == gp.abs_(wdiffs_0[a,i]))
    #         if (Z[ind]- Z[ind_minus]) == 0:
    #             print 'smoothing'
    #             m.addConstr(sorteddiffs[a,i] <= LIPSCH_CONST*min_incr, 'smoothness'+str(i) )
    #             m.addConstr(sorteddiffs_0[a,i] <= LIPSCH_CONST*min_incr, 'smoothness'+str(i) )
    #         else:
    #             m.addConstr(sorteddiffs[a,i] <= LIPSCH_CONST*(Z[ind] - Z[ind_minus]), 'smoothness'+str(i) )
    #             m.addConstr(sorteddiffs_0[a,i] <= LIPSCH_CONST*(Z[ind] - Z[ind_minus]), 'smoothness-on-y0'+str(i) )
    # #
    # elif smoothing:
    #     for a in range(n_as):
    #         for i in range(n):
    # # Add smoothness constraints\    elif smoothing=='nd':
    #             m.addConstrs((w_y1[a,i] - w_y1[a,j] <= LIPSCH_CONST*dists_mat[i,j] for i in range(n) for j in range(n) ) , name='L-nd-' + str(a))
    #             m.addConstrs((-w_y1[a,i]+ w_y1[a,j] <= LIPSCH_CONST*dists_mat[i,j] for i in range(n) for j in range(n) ) , name='-L-nd-' + str(a))
    # objective function

    expr = gp.LinExpr();
    # Set a precedent for 1vs. all
    for a in range(n_as)[1:]:
        print a
        for i in range(len(y)):
            expr += 1.0/n * mu_scalar[a - 1] *( (w_y1[0,i] * y[i])/p_as[0] - (w_y1[a,i] * y[i])/p_as[a] ) ;
    if direction == 'max':
        m.setObjective(expr, gp.GRB.MAXIMIZE);
    else:
        m.setObjective(expr, gp.GRB.MINIMIZE);
    m.optimize()

    if feasibility_relax:
        if m.status == gp.GRB.INFEASIBLE:
            # Relax the constraints to make the model feasible
            print('The model is infeasible; relaxing the constraints')
            orignumvars = m.NumVars
            m.feasRelaxS(0, False, False, True)
            m.optimize()
            status = m.status

            print('\nSlack values:')
            slacks = m.getVars()[orignumvars:]
            for sv in slacks:
                if sv.X > 1e-6:
                    print('%s = %g' % (sv.VarName, sv.X))

    if (m.status == gp.GRB.OPTIMAL):
        w_y1_a_vals = np.asarray([[ w_y1[a,i].X for i in range(n) ] for a in range(n_as)]  )
        w_y0_a_vals = np.asarray([[ w_y0[a,i].X for i in range(n) ] for a in range(n_as)]  )
        res = [m.ObjVal, w_y1_a_vals, w_y0_a_vals]
        return res
    else:
        return [np.nan, 0, 0]

def get_disps(mu_scalar, w_y1, w_y0, n_as, y, p_as):
    n = len(y)
    return [ 1.0/n * ( np.dot(w_y1[0,:],y)/p_as[0] - np.dot(w_y1[a,:],y)/p_as[a] ) for a in range(n_as)[1:] ]

def dists_matrix(X, norm = 'minkowski', squareform_ = False):
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    dists = pdist(X_scaled, norm, p=1) # distance computed in entry ij
    if squareform_:
        dists = squareform(dists)
    return [dists, X_scaled]


'''
most general support function version
Solve the tpr for a range of t values
resolve with warm starting
Inputs:
t: scaling factor; We use the population LP scaling
t_0s: homogenization factors for 0
t_1s: homogenization factors for 1
y: objective coefficients
e.g. Yhat
yfix: what value to fix Y to
hyfix: what value to fix \hat Y to
p_yhat_cond_y: Pr(Y=1 | hat Y = 1, Z_i)
p_yhat_jt_y: Pr(Y=1 , hat Y = 1 | Z_i)
b_: upper bound
If smoothing: add smoothness constraints
return primal weights
Return weights in the order [(1,1),(0,1),(1,0),(0,0)]
return scaling t
'''
def get_disp_in_scalarized_Y_Yhat_fixed_t_mult_a(t_0s,t_1s, n_as, Y, Yhat, yfix, hyfix,
joints, p_a_z, LIPSCH_CONST,
dists_mat, LAW_TOT_PROB_SLACK=0, direction='max', quiet=True, smoothing=None,
save = True, savestr='tpr'):
    m = gp.Model()
    res = [ None ] * len(ts)
    if quiet: m.setParam("OutputFlag", 0)
    m.setParam("Method", 1);
    y = Yhat
    n = len(y)

    w_y1_yhatobs = m.addVars(range(n_as), len(y), lb = 0., vtype=gp.GRB.CONTINUOUS)
    w_y0_yhatobs = m.addVars(range(n_as), len(y), lb = 0.,  vtype=gp.GRB.CONTINUOUS)
    w_y1_yhatnotobs = m.addVars(range(n_as), len(y), lb = 0.,  vtype=gp.GRB.CONTINUOUS)
    w_y0_yhatnotobs = [m.addVars( range(n_as), len(y), lb = 0., vtype=gp.GRB.CONTINUOUS) for yy in y]
    # t = m.addVar( lb = 0., vtype=gp.GRB.CONTINUOUS)
    ## changing: use the definition that t = 1/ E[ w Y=y ], u = w t
    [ p_y1_jt_yhat1, p_y0_jt_yhat1, p_y1_jt_yhat0, p_y0_jt_yhat0 ] = joints
    print n
    print len(p_y1_jt_yhat1)
    p_y1_jt_yhatobs = [ p_y1_jt_yhat1[i] if y[i] == 1 else p_y1_jt_yhat0[i] for i in range((n)) ]
    p_y0_jt_yhatobs = [ p_y0_jt_yhat1[i] if y[i] == 1 else p_y0_jt_yhat0[i] for i in range((n)) ]
    p_y1_jt_yhatnotobs = [ p_y1_jt_yhat0[i] if y[i] == 1 else p_y1_jt_yhat1[i] for i in range((n)) ]
    p_y0_jt_yhatnotobs = [ p_y0_jt_yhat0[i] if y[i] == 1 else p_y0_jt_yhat1[i] for i in range((n)) ]

    C = np.mean(Y)
    print 'adding variables'
    m.update()
    t = ts[0] # build the model first

    for a in n_as:
        m.addConstr(gp.quicksum([w_y1_yhatobs_a[i] * Y[i] for i in range(n) ]) == n, name = 'homogenization'+str(a)) # need to fix the homogenization
        m.addConstr(gp.quicksum([w_y0_yhatobs_a[i] * Y[i] for i in range(n) ]) == t_1s[a]*n, name = 'homogenization'+str(a)) # need to fix the homogenization

        m.addConstrs((w_y1_yhatobs[a,i]  <= t  for i in range(n) ), name='Y1Yhatobs')
        m.addConstrs((w_y0_yhatobs[a,i]  <= t  for i in range(n) ), name='Y0Yhatobs')
        m.addConstrs((w_y1_yhatnotobs[a,i]  <= t  for i in range(n) ), name='Y1Yhatnotobs')
        m.addConstrs((w_y0_yhatnotobs[a,i]  <= t  for i in range(n) ), name='Y0Yhatnotobs')


    for i in range(len(y)):
        m.addConstr( p_y0_jt_yhatobs[i]*w_y0_yhatobs_a[i] + p_y1_jt_yhatobs[i]*w_y1_yhatobs[a,i] + p_y1_jt_yhatnotobs[i]*w_y1_yhatnotobs[a,i] + p_y0_jt_yhatnotobs[i]*w_y0_yhatnotobs[a,i] == p_a_z[i,a]*t_0s[a]*t_1s[a] , name='LTP'+str(i))

    # if smoothing:
    #     m.addConstrs((w_y0_yhatobs_a[i] - w_y0_yhatobs_a[j] <= LIPSCH_CONST*dists_mat[i,j] * t for i in range(n) for j in range(n)[i:] ) , name='L-y=0')
    #     m.addConstrs((-w_y0_yhatobs_a[i] + w_y0_yhatobs_a[j] <= LIPSCH_CONST*dists_mat[i,j] * t for i in range(n) for j in range(n)[i:] ) , name='-L-y=0')
    #     m.addConstrs((w_y1_yhatobs_a[i] - w_y1_yhatobs_a[j] <= LIPSCH_CONST*dists_mat[i,j] * t for i in range(n) for j in range(n)[i:] ) , name='L-y=1')
    #     m.addConstrs((-w_y1_yhatobs_a[i] + w_y1_yhatobs_a[j] <= LIPSCH_CONST*dists_mat[i,j] * t  for i in range(n) for j in range(n)[i:] ) , name='-L-y=1')
    print 'adding constraints'
    expr = gp.LinExpr();
    for a in range(n_as):
        expr += ( t_0s[a]*C/(t_0s[a]*C - 1.0) * 1.0/n * gp.quicksum([w_y1_yhatobs_a[i] * Y[i] * Yhat[i] for i in range(n) ])  -  np.mean(Yhat*Y) * t/(t*C - 1.0) )
    if direction=='max':
        print 'solving'
        m.setObjective(expr, gp.GRB.MAXIMIZE); m.optimize()
    else:
        m.setObjective(expr, gp.GRB.MINIMIZE); m.optimize()

    if (m.status == gp.GRB.OPTIMAL):
        wghts_ = [w_y1_yhatobs_a, w_y0_yhatobs_a, w_y1_yhatnotobs_a, w_y0_yhatnotobs_a]
        wghts__ = [ [ ww.X for ww in wghts_[ind] ] for ind in range(len(wghts_)) ]
        res = [m.ObjVal, wghts__, t]
    else:
        res =  [np.nan, 0, 0]
    if save:
        pickle.dump({'res':res, 'args': {'ts':ts, 'LIPSCH_CONST':LIPSCH_CONST} }, open('out/opt_over_t_res_'+savestr+'.pkl', 'wb')  )
    return res



'''
Solve the scalarized tpr and tnr problem for a range of t values
resolve with warm starting
Inputs:
t: scaling factor; We use the population LP scaling
y: objective coefficients
e.g. Yhat
p_yhat_cond_y: Pr(Y=1 | hat Y = 1, Z_i)
p_yhat_jt_y: Pr(Y=1 , hat Y = 1 | Z_i)
b_: upper bound
If smoothing: add smoothness constraints
return primal weights
Return weights in the order [(1,1),(0,1),(1,0),(0,0)]
return scaling t
'''
def get_tpr_tnr_disp_indicator_Y_over_ts(t0,t1, lambdas, Y, Yhat, joints, p_a_z, LIPSCH_CONST,
dists_mat, LAW_TOT_PROB_SLACK=0, direction='max', quiet=True, smoothing=None,
save = True, savestr='tpr'):
    m = gp.Model()
    if quiet: m.setParam("OutputFlag", 0)
    m.setParam("Method", 1);
    y = Yhat
    n = len(y)
    w_y1_yhatobs_a = [m.addVar( lb = 0., vtype=gp.GRB.CONTINUOUS) for yy in y]
    w_y0_yhatobs_a = [m.addVar( lb = 0.,  vtype=gp.GRB.CONTINUOUS) for yy in y]
    w_y1_yhatnotobs_a = [m.addVar( lb = 0.,  vtype=gp.GRB.CONTINUOUS) for yy in y]
    w_y0_yhatnotobs_a = [m.addVar( lb = 0., vtype=gp.GRB.CONTINUOUS) for yy in y]
    # t = m.addVar( lb = 0., vtype=gp.GRB.CONTINUOUS)
    ## changing: use the definition that t = 1/ E[ w Y=y ], u = w t
    [ p_y1_jt_yhat1, p_y0_jt_yhat1, p_y1_jt_yhat0, p_y0_jt_yhat0 ] = joints
    print n
    print len(p_y1_jt_yhat1)
    p_y1_jt_yhatobs = [ p_y1_jt_yhat1[i] if y[i] == 1 else p_y1_jt_yhat0[i] for i in range((n)) ]
    p_y0_jt_yhatobs = [ p_y0_jt_yhat1[i] if y[i] == 1 else p_y0_jt_yhat0[i] for i in range((n)) ]
    p_y1_jt_yhatnotobs = [ p_y1_jt_yhat0[i] if y[i] == 1 else p_y1_jt_yhat1[i] for i in range((n)) ]
    p_y0_jt_yhatnotobs = [ p_y0_jt_yhat0[i] if y[i] == 1 else p_y0_jt_yhat1[i] for i in range((n)) ]

    C = np.mean(Y)
    C0 = 1-C
    print 'adding variables'
    m.update()
# build the model first
    Y0 = 1-Y
    m.addConstr(gp.quicksum([w_y0_yhatobs_a[i] * Y0[i] for i in range(n) ]) == n, name = 'homogenization0') # need to fix the homogenization
    m.addConstr(gp.quicksum([w_y1_yhatobs_a[i] * Y[i] for i in range(n) ]) == n, name = 'homogenization1') # need to fix the homogenization
    m.addConstrs((w_y1_yhatobs_a[i]  <= t1  for i in range(n) ), name='Y1Yhatobs')
    m.addConstrs((w_y0_yhatobs_a[i]  <= t0  for i in range(n) ), name='Y0Yhatobs')
    m.addConstrs((w_y1_yhatnotobs_a[i]  <= t1  for i in range(n) ), name='Y1Yhatnotobs')
    m.addConstrs((w_y0_yhatnotobs_a[i]  <= t0  for i in range(n) ), name='Y0Yhatnotobs')
    # m.addConstrs((w_y1_yhatobs_a[i]  <= t * n for i in range(n) ), name='Y1Yhatobs')
    # m.addConstrs((w_y0_yhatobs_a[i]  <= t * n for i in range(n) ), name='Y0Yhatobs')
    # m.addConstrs((w_y1_yhatnotobs_a[i]  <= t * n for i in range(n) ), name='Y1Yhatnotobs')
    # m.addConstrs((w_y0_yhatnotobs_a[i]  <= t * n for i in range(n) ), name='Y0Yhatnotobs')

    for i in range(len(y)):
        m.addConstr( p_y0_jt_yhatobs[i]*w_y0_yhatobs_a[i]/t0 + p_y1_jt_yhatobs[i]*w_y1_yhatobs_a[i]/t1 + p_y1_jt_yhatnotobs[i]*w_y1_yhatnotobs_a[i]/t1 + p_y0_jt_yhatnotobs[i]*w_y0_yhatnotobs_a[i]/t0 == p_a_z[i] , name='LTP'+str(i))
    print 'adding constraints'
    expr = gp.LinExpr();
    expr += lambdas[0]*( t0*C0/(t0*C0 - 1.0) * 1.0/n * gp.quicksum([w_y1_yhatobs_a[i] *(1-Y[i]) * (1-Yhat[i])  for i in range(n) ])  -  np.mean((1-Yhat)*(1-Y)) * t0/(t0*C0 - 1.0) )
    expr += lambdas[1]*( t1*C/(t1*C - 1.0) * 1.0/n * gp.quicksum([w_y0_yhatobs_a[i] * Y[i] * Yhat[i] for i in range(n) ])  -  np.mean(Yhat*Y) * t1/(t1*C - 1.0) )

    if direction=='max':
        print 'solving'
        m.setObjective(expr, gp.GRB.MAXIMIZE); m.optimize()
    else:
        m.setObjective(expr, gp.GRB.MINIMIZE); m.optimize()

    if (m.status == gp.GRB.OPTIMAL):
        wghts_ = [w_y1_yhatobs_a, w_y0_yhatobs_a, w_y1_yhatnotobs_a, w_y0_yhatnotobs_a]
        wghts__ = [ [ ww.X for ww in wghts_[ind] ] for ind in range(len(wghts_)) ]
        args = { 'lambdas':lambdas, 't0':t0, 't1':t1 }
        return [m.ObjVal, wghts__, args]
    else:
        return [np.nan, 0, 0]

    if save:
        pickle.dump({'res':res, 'args': {'ts':ts, 'LIPSCH_CONST':LIPSCH_CONST} }, open('out/opt_over_t_res_'+savestr+'.pkl', 'wb')  )
    return res


# given weights, compute the achieved TPR, TNR disparities
# todo: generalize to mult races
def get_tpr_tnr_from_weights(wghts__, Y, Yhat, joints, p_a_z, LIPSCH_CONST):
    [w_y1_yhatobs_a, w_y0_yhatobs_a, w_y1_yhatnotobs_a, w_y0_yhatnotobs_a] = wghts__
    w_y1_yhatobs_a = np.asarray(w_y1_yhatobs_a)
    w_y0_yhatobs_a = np.asarray(w_y0_yhatobs_a)
    w_y1_yhatnotobs_a = np.asarray(w_y1_yhatnotobs_a)
    w_y0_yhatnotobs_a = np.asarray(w_y0_yhatnotobs_a)
    n = len(w_y1_yhatobs_a)
    tpr_a = sum([w_y1_yhatobs_a[i] * Y[i] * Yhat[i] for i in range(n) ]) / np.dot(w_y1_yhatobs_a, Y)
    tpr_b = sum([(1-w_y1_yhatobs_a[i]) * Y[i] * Yhat[i] for i in range(n) ]) / np.dot((1-w_y1_yhatobs_a), Y)
    tnr_a = sum([w_y0_yhatobs_a[i] * (1-Y[i]) * (1-Yhat[i]) for i in range(n) ]) / np.dot(w_y0_yhatobs_a, (1-Y))
    tnr_b = sum([(1-w_y0_yhatobs_a[i]) * (1-Y[i]) * (1-Yhat[i]) for i in range(n) ]) / np.dot((1-w_y0_yhatobs_a), (1-Y))
    return [tpr_a, tpr_b, tnr_a, tnr_b]



'''
Solve the TNR for a range of t values
resolve with warm starting
Inputs:
t: scaling factor; We use the population LP scaling
y: objective coefficients
e.g. Yhat
p_yhat_cond_y: Pr(Y=1 | hat Y = 1, Z_i)
p_yhat_jt_y: Pr(Y=1 , hat Y = 1 | Z_i)
b_: upper bound
If smoothing: add smoothness constraints
return primal weights
Return weights in the order [(1,1),(0,1),(1,0),(0,0)]
return scaling t
'''
def get_tnr_disp_indicator_Y_over_ts(ts, Y, Yhat, joints, p_a_z, LIPSCH_CONST,
dists_mat, LAW_TOT_PROB_SLACK=0, direction='max', quiet=True, smoothing=None,
save = True, savestr='tpr'):
    m = gp.Model()
    res = [ None ] * len(ts)
    if quiet: m.setParam("OutputFlag", 0)
    m.setParam("Method", 1);
    y = Yhat
    n = len(y)
    w_y1_yhatobs_a = [m.addVar( lb = 0., vtype=gp.GRB.CONTINUOUS) for yy in y];w_y0_yhatobs_a = [m.addVar( lb = 0.,  vtype=gp.GRB.CONTINUOUS) for yy in y];w_y1_yhatnotobs_a = [m.addVar( lb = 0.,  vtype=gp.GRB.CONTINUOUS) for yy in y];w_y0_yhatnotobs_a = [m.addVar( lb = 0., vtype=gp.GRB.CONTINUOUS) for yy in y]
    ## changing: use the definition that t = 1/ E[ w Y=y ], u = w t
    [ p_y1_jt_yhat1, p_y0_jt_yhat1, p_y1_jt_yhat0, p_y0_jt_yhat0 ] = joints
    print n
    print len(p_y1_jt_yhat1)
    p_y1_jt_yhatobs = [ p_y1_jt_yhat1[i] if y[i] == 1 else p_y1_jt_yhat0[i] for i in range((n)) ]; p_y0_jt_yhatobs = [ p_y0_jt_yhat1[i] if y[i] == 1 else p_y0_jt_yhat0[i] for i in range((n)) ]; p_y1_jt_yhatnotobs = [ p_y1_jt_yhat0[i] if y[i] == 1 else p_y1_jt_yhat1[i] for i in range((n)) ]; p_y0_jt_yhatnotobs = [ p_y0_jt_yhat0[i] if y[i] == 1 else p_y0_jt_yhat1[i] for i in range((n)) ]

    C = np.mean(1-Y)
    print 'adding variables'
    m.update()
    t = ts[0] # build the model first

    m.addConstr(gp.quicksum([w_y0_yhatobs_a[i] *(1- Y[i]) for i in range(n) ]) == n, name = 'homogenization') # need to fix the homogenization
    m.addConstrs((w_y1_yhatobs_a[i]  <= t  for i in range(n) ), name='Y1Yhatobs')
    m.addConstrs((w_y0_yhatobs_a[i]  <= t  for i in range(n) ), name='Y0Yhatobs')
    m.addConstrs((w_y1_yhatnotobs_a[i]  <= t  for i in range(n) ), name='Y1Yhatnotobs')
    m.addConstrs((w_y0_yhatnotobs_a[i]  <= t  for i in range(n) ), name='Y0Yhatnotobs')
    # m.addConstrs((w_y1_yhatobs_a[i]  <= t * n for i in range(n) ), name='Y1Yhatobs')
    # m.addConstrs((w_y0_yhatobs_a[i]  <= t * n for i in range(n) ), name='Y0Yhatobs')
    # m.addConstrs((w_y1_yhatnotobs_a[i]  <= t * n for i in range(n) ), name='Y1Yhatnotobs')
    # m.addConstrs((w_y0_yhatnotobs_a[i]  <= t * n for i in range(n) ), name='Y0Yhatnotobs')

    for i in range(len(y)):
        m.addConstr( p_y0_jt_yhatobs[i]*w_y0_yhatobs_a[i] + p_y1_jt_yhatobs[i]*w_y1_yhatobs_a[i] + p_y1_jt_yhatnotobs[i]*w_y1_yhatnotobs_a[i] + p_y0_jt_yhatnotobs[i]*w_y0_yhatnotobs_a[i] == p_a_z[i]*t , name='LTP'+str(i))

    if smoothing:
        m.addConstrs((w_y0_yhatobs_a[i] - w_y0_yhatobs_a[j] <= LIPSCH_CONST*dists_mat[i,j] * t for i in range(n) for j in range(n)[i:] ) , name='L-y=0')
        m.addConstrs((-w_y0_yhatobs_a[i] + w_y0_yhatobs_a[j] <= LIPSCH_CONST*dists_mat[i,j] * t for i in range(n) for j in range(n)[i:] ) , name='-L-y=0')
        m.addConstrs((w_y1_yhatobs_a[i] - w_y1_yhatobs_a[j] <= LIPSCH_CONST*dists_mat[i,j] * t for i in range(n) for j in range(n)[i:] ) , name='L-y=1')
        m.addConstrs((-w_y1_yhatobs_a[i] + w_y1_yhatobs_a[j] <= LIPSCH_CONST*dists_mat[i,j] * t  for i in range(n) for j in range(n)[i:] ) , name='-L-y=1')
    print 'adding constraints'
    expr = gp.LinExpr();
    expr += ( t*C/(t*C - 1.0) * 1.0/n * gp.quicksum([w_y0_yhatobs_a[i] * (1-Y[i]) * (1-Yhat[i]) for i in range(n) ])  -  np.mean((1-Yhat)*(1-Y)) * t/(t*C - 1.0) )
    if direction=='max':
        print 'solving'
        m.setObjective(expr, gp.GRB.MAXIMIZE); m.optimize()
    else:
        m.setObjective(expr, gp.GRB.MINIMIZE); m.optimize()

    if (m.status == gp.GRB.OPTIMAL):
        wghts_ = [w_y1_yhatobs_a, w_y0_yhatobs_a, w_y1_yhatnotobs_a, w_y0_yhatnotobs_a]
        wghts__ = [ [ ww.X for ww in wghts_[ind] ] for ind in range(len(wghts_)) ]
        res[0] = [m.ObjVal, wghts__, t]
    else:
        res[0] =  [np.nan, 0, 0]
    #
    # Change model constraint RHSes and objective and reoptimize
    for t_ind, t_ in enumerate(ts[1:]):
        for i in range(n):
            m.getConstrByName('LTP'+str(i)).RHS = p_a_z[i]*t_
        for i in range(n):
            m.getConstrByName('Y1Yhatobs['+str(i)+']').RHS = t_
            m.getConstrByName('Y0Yhatobs['+str(i)+']').RHS = t_
            m.getConstrByName('Y1Yhatnotobs['+str(i)+']').RHS = t_
            m.getConstrByName('Y0Yhatnotobs['+str(i)+']').RHS = t_
        if smoothing:
            for i in range(n):
                for j in range(n)[i:]:
                    m.getConstrByName('L-y=0'+str(i)+','+str(j)+']').RHS = LIPSCH_CONST*dists_mat[i,j]* t_
                    m.getConstrByName('-L-y=0'+str(i)+','+str(j)+']').RHS = LIPSCH_CONST*dists_mat[i,j]* t_
                for k in range(n_b)[i:]:
                    m.getConstrByName('L-y=1'+str(i)+','+str(k)+']').RHS = LIPSCH_CONST*dists_mat[i,j]* t_
                    m.getConstrByName('-L-y=1'+str(i)+','+str(k)+']').RHS = LIPSCH_CONST*dists_mat[i,j]* t_
        expr = gp.LinExpr();
        expr += ( t_*C/(t_*C - 1.0) * 1.0/n * gp.quicksum([w_y0_yhatobs_a[i] * (1-Y[i]) * (1-Yhat[i]) for i in range(n) ])  -  np.mean((1-Yhat)*(1-Y)) * t_/(t_*C - 1.0) )
        # expr += ( t_*C/(t_*C - 1.0) * gp.quicksum([w_y1_yhatobs_a[i] * Y[i] * Yhat[i] for i in range(n) ])  -  np.mean(Yhat*Y) * t_*C/(t_*C - 1.0))
        if direction=='max':
            m.setObjective(expr, gp.GRB.MAXIMIZE); m.optimize()
        else:
            m.setObjective(expr, gp.GRB.MINIMIZE); m.optimize()
        if (m.status == gp.GRB.OPTIMAL):
            wghts_ = [w_y1_yhatobs_a, w_y0_yhatobs_a, w_y1_yhatnotobs_a, w_y0_yhatnotobs_a]
            wghts__ = [ [ ww.X for ww in wghts_[ind] ] for ind in range(len(wghts_)) ]
            res[t_ind+1] = [m.ObjVal, wghts__, t_]
        else:
            res[t_ind+1] =  [np.nan, 0, 0]
    if save:
        pickle.dump({'res':res, 'args': {'ts':ts, 'LIPSCH_CONST':LIPSCH_CONST} }, open('out/opt_over_t_res_'+savestr+'.pkl', 'wb')  )
    return res


def tpr(Yhat, Y, A, aval):
    return sum((Yhat)&(Y)&(A==aval))*1.0/sum((Y)&(A==aval))
def tpr_nota(Yhat, Y, A, aval):
    return sum((Yhat)&(Y)&(A!=aval))*1.0/sum((Y)&(A!=aval))
def tnr(Yhat, Y, A, aval):
    return sum((1-Yhat)&(1-Y)&(A==aval))*1.0/sum((1-Y)&(A==aval))
def tnr_nota(Yhat, Y, A, aval):
    return sum((1-Yhat)&(1-Y)&(A!=aval))*1.0/sum((1-Y)&(A!=aval))


'''
Solve the TNR for a range of t values
resolve with warm starting
Inputs:
t: scaling factor; We use the population LP scaling
y: objective coefficients
e.g. Yhat
p_yhat_cond_y: Pr(Y=1 | hat Y = 1, Z_i)
p_yhat_jt_y: Pr(Y=1 , hat Y = 1 | Z_i)
b_: upper bound
If smoothing: add smoothness constraints
return primal weights
Return weights in the order [(1,1),(0,1),(1,0),(0,0)]
return scaling t
'''
def get_tnr_multi_disp_indicator_Y_over_ts(ts, Y, Yhat, joints, p_a_z, LIPSCH_CONST,
dists_mat, LAW_TOT_PROB_SLACK=0, direction='max', quiet=True, smoothing=None,
save = True, savestr='tpr'):
    m = gp.Model()
    res = [ None ] * len(ts)
    if quiet: m.setParam("OutputFlag", 0)
    m.setParam("Method", 1);
    y = Yhat
    n = len(y)
    w_y1_yhatobs_a = [m.addVar( lb = 0., vtype=gp.GRB.CONTINUOUS) for yy in y];
    w_y0_yhatobs_a = [m.addVar( lb = 0.,  vtype=gp.GRB.CONTINUOUS) for yy in y];
    w_y1_yhatnotobs_a = [m.addVar( lb = 0.,  vtype=gp.GRB.CONTINUOUS) for yy in y];w_y0_yhatnotobs_a = [m.addVar( lb = 0., vtype=gp.GRB.CONTINUOUS) for yy in y]
    ## changing: use the definition that t = 1/ E[ w Y=y ], u = w t
    [ p_y1_jt_yhat1, p_y0_jt_yhat1, p_y1_jt_yhat0, p_y0_jt_yhat0 ] = joints
    print n
    print len(p_y1_jt_yhat1)
    p_y1_jt_yhatobs = [ p_y1_jt_yhat1[i] if y[i] == 1 else p_y1_jt_yhat0[i] for i in range((n)) ]; p_y0_jt_yhatobs = [ p_y0_jt_yhat1[i] if y[i] == 1 else p_y0_jt_yhat0[i] for i in range((n)) ]; p_y1_jt_yhatnotobs = [ p_y1_jt_yhat0[i] if y[i] == 1 else p_y1_jt_yhat1[i] for i in range((n)) ]; p_y0_jt_yhatnotobs = [ p_y0_jt_yhat0[i] if y[i] == 1 else p_y0_jt_yhat1[i] for i in range((n)) ]

    C = np.mean(1-Y)
    print 'adding variables'
    m.update()
    t = ts[0] # build the model first

    m.addConstr(gp.quicksum([w_y0_yhatobs_a[i] *(1- Y[i]) for i in range(n) ]) == n, name = 'homogenization') # need to fix the homogenization
    m.addConstrs((w_y1_yhatobs_a[i]  <= t  for i in range(n) ), name='Y1Yhatobs')
    m.addConstrs((w_y0_yhatobs_a[i]  <= t  for i in range(n) ), name='Y0Yhatobs')
    m.addConstrs((w_y1_yhatnotobs_a[i]  <= t  for i in range(n) ), name='Y1Yhatnotobs')
    m.addConstrs((w_y0_yhatnotobs_a[i]  <= t  for i in range(n) ), name='Y0Yhatnotobs')
    # m.addConstrs((w_y1_yhatobs_a[i]  <= t * n for i in range(n) ), name='Y1Yhatobs')
    # m.addConstrs((w_y0_yhatobs_a[i]  <= t * n for i in range(n) ), name='Y0Yhatobs')
    # m.addConstrs((w_y1_yhatnotobs_a[i]  <= t * n for i in range(n) ), name='Y1Yhatnotobs')
    # m.addConstrs((w_y0_yhatnotobs_a[i]  <= t * n for i in range(n) ), name='Y0Yhatnotobs')

    for i in range(len(y)):
        m.addConstr( p_y0_jt_yhatobs[i]*w_y0_yhatobs_a[i] + p_y1_jt_yhatobs[i]*w_y1_yhatobs_a[i] + p_y1_jt_yhatnotobs[i]*w_y1_yhatnotobs_a[i] + p_y0_jt_yhatnotobs[i]*w_y0_yhatnotobs_a[i] == p_a_z[i]*t , name='LTP'+str(i))

    if smoothing:
        m.addConstrs((w_y0_yhatobs_a[i] - w_y0_yhatobs_a[j] <= LIPSCH_CONST*dists_mat[i,j] * t for i in range(n) for j in range(n)[i:] ) , name='L-y=0')
        m.addConstrs((-w_y0_yhatobs_a[i] + w_y0_yhatobs_a[j] <= LIPSCH_CONST*dists_mat[i,j] * t for i in range(n) for j in range(n)[i:] ) , name='-L-y=0')
        m.addConstrs((w_y1_yhatobs_a[i] - w_y1_yhatobs_a[j] <= LIPSCH_CONST*dists_mat[i,j] * t for i in range(n) for j in range(n)[i:] ) , name='L-y=1')
        m.addConstrs((-w_y1_yhatobs_a[i] + w_y1_yhatobs_a[j] <= LIPSCH_CONST*dists_mat[i,j] * t  for i in range(n) for j in range(n)[i:] ) , name='-L-y=1')
    print 'adding constraints'
    expr = gp.LinExpr();
    expr += ( t*C/(t*C - 1.0) * 1.0/n * gp.quicksum([w_y0_yhatobs_a[i] * (1-Y[i]) * (1-Yhat[i]) for i in range(n) ])  -  np.mean((1-Yhat)*(1-Y)) * t/(t*C - 1.0) )
    if direction=='max':
        print 'solving'
        m.setObjective(expr, gp.GRB.MAXIMIZE); m.optimize()
    else:
        m.setObjective(expr, gp.GRB.MINIMIZE); m.optimize()

    if (m.status == gp.GRB.OPTIMAL):
        wghts_ = [w_y1_yhatobs_a, w_y0_yhatobs_a, w_y1_yhatnotobs_a, w_y0_yhatnotobs_a]
        wghts__ = [ [ ww.X for ww in wghts_[ind] ] for ind in range(len(wghts_)) ]
        res[0] = [m.ObjVal, wghts__, t]
    else:
        res[0] =  [np.nan, 0, 0]
    #
    # Change model constraint RHSes and objective and reoptimize
    for t_ind, t_ in enumerate(ts[1:]):
        for i in range(n):
            m.getConstrByName('LTP'+str(i)).RHS = p_a_z[i]*t_
        for i in range(n):
            m.getConstrByName('Y1Yhatobs['+str(i)+']').RHS = t_
            m.getConstrByName('Y0Yhatobs['+str(i)+']').RHS = t_
            m.getConstrByName('Y1Yhatnotobs['+str(i)+']').RHS = t_
            m.getConstrByName('Y0Yhatnotobs['+str(i)+']').RHS = t_
        if smoothing:
            for i in range(n):
                for j in range(n)[i:]:
                    m.getConstrByName('L-y=0'+str(i)+','+str(j)+']').RHS = LIPSCH_CONST*dists_mat[i,j]* t_
                    m.getConstrByName('-L-y=0'+str(i)+','+str(j)+']').RHS = LIPSCH_CONST*dists_mat[i,j]* t_
                for k in range(n_b)[i:]:
                    m.getConstrByName('L-y=1'+str(i)+','+str(k)+']').RHS = LIPSCH_CONST*dists_mat[i,j]* t_
                    m.getConstrByName('-L-y=1'+str(i)+','+str(k)+']').RHS = LIPSCH_CONST*dists_mat[i,j]* t_
        expr = gp.LinExpr();
        expr += ( t_*C/(t_*C - 1.0) * 1.0/n * gp.quicksum([w_y0_yhatobs_a[i] * (1-Y[i]) * (1-Yhat[i]) for i in range(n) ])  -  np.mean((1-Yhat)*(1-Y)) * t_/(t_*C - 1.0) )
        # expr += ( t_*C/(t_*C - 1.0) * gp.quicksum([w_y1_yhatobs_a[i] * Y[i] * Yhat[i] for i in range(n) ])  -  np.mean(Yhat*Y) * t_*C/(t_*C - 1.0))
        if direction=='max':
            m.setObjective(expr, gp.GRB.MAXIMIZE); m.optimize()
        else:
            m.setObjective(expr, gp.GRB.MINIMIZE); m.optimize()
        if (m.status == gp.GRB.OPTIMAL):
            wghts_ = [w_y1_yhatobs_a, w_y0_yhatobs_a, w_y1_yhatnotobs_a, w_y0_yhatnotobs_a]
            wghts__ = [ [ ww.X for ww in wghts_[ind] ] for ind in range(len(wghts_)) ]
            res[t_ind+1] = [m.ObjVal, wghts__, t_]
        else:
            res[t_ind+1] =  [np.nan, 0, 0]
    if save:
        pickle.dump({'res':res, 'args': {'ts':ts, 'LIPSCH_CONST':LIPSCH_CONST} }, open('out/opt_over_t_res_'+savestr+'.pkl', 'wb')  )
    return res





'''
Lispschitz constraints within components
aka pooled models
county: discrete categories within which we are lipschitz smooth
y: outcomes
n_as: number of classes
mu_scalar: (n_A - 1) number of values (scalarization on disparities)
p_az: (n_a, n) array of probabilities
p_yz: (n) array of probabilities
Z: proxies
LIPSCH_CONST:
'''
def get_dem_disp_obs_outcomes_multi_race_countyincome(y, n_as, mu_scalar, p_az, p_yz,
 Z, county, LIPSCH_CONST, LAW_TOT_PROB_SLACK=0, quiet=True, smoothing=None,
 smoothing1d=True, direction = 'max'):
    n = len(y); m = gp.Model()
    p_as = (p_az.sum(axis=0)/p_az.shape[0])
    if (np.asarray([len(a) for a in [p_az, p_yz, Z]]) != n).any():
        raise ValueError("data inputs must have the same length")
    if quiet:
        m.setParam("OutputFlag", 0)
#     t = m.addVar(lb = 0., ub = gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS)
    w_y1 = m.addVars(range(n_as), len(y), lb = 0., ub = 1, vtype=gp.GRB.CONTINUOUS)
    w_y0 = m.addVars(range(n_as), len(y), lb = 0., ub = 1, vtype=gp.GRB.CONTINUOUS)

    sorteddiffs = m.addVars( range(n_as), len(y)-1, lb = 0., ub = 1, vtype=gp.GRB.CONTINUOUS)
    sorteddiffs_0 = m.addVars( range(n_as), len(y)-1, lb = 0., ub = 1, vtype=gp.GRB.CONTINUOUS)
    wdiffs_1 = m.addVars( range(n_as), len(y)-1, lb = 0., ub = 1, vtype=gp.GRB.CONTINUOUS)
    wdiffs_0 = m.addVars( range(n_as), len(y)-1, lb = 0., ub = 1, vtype=gp.GRB.CONTINUOUS)

    m.update()
    # law of tot prob bounds
    for a in range(n_as):
        for i in range(n):
            m.addConstr( w_y1[(a,i)]*p_yz[i] + w_y0[a,i]*(1.0-p_yz[i]) == p_az[i,a]  )
            m.addConstr( gp.quicksum(w_y1[a,i] for a in range(n_as)) == 1 )
            m.addConstr( gp.quicksum(w_y0[a,i] for a in range(n_as)) == 1 )
    print LIPSCH_CONST
    levels = np.unique(county)
    if (smoothing1d) and smoothing:
        print smoothing

    ### Lipschitz constraints for all races

        for a in range(n_as):
        # a = 0 # should be feasible for a = 0
         #returns sort inds in ascending order
            for level in levels:
                inds = np.where(county == level)[0] # which inds in Z are county == level?
                sort_ind = np.argsort(Z[inds]) # sort the proxy within county == level
                # min_incr = np.min(np.diff(Z[inds[sort_ind]])[np.diff(Z[inds[sort_ind]])!=0])
                # print 'min_incr',min_incr
                for i in range(len(inds))[:-1]:

                    ind = inds[sort_ind[i+1]]; ind_minus = inds[sort_ind[i]];
                    #add constraints directly
                    # if (Z[ind]- Z[ind_minus]) == 0:
                    #     m.addConstr( w_y1[a,ind] - w_y1[a,ind_minus] <= LIPSCH_CONST*min_incr, 'smoothness'+str(i) )
                    #     m.addConstr( -w_y1[a,ind] + w_y1[a,ind_minus] <= LIPSCH_CONST*min_incr, 'smoothness'+str(i) )
                    #     m.addConstr( w_y0[a,ind] - w_y0[a,ind_minus] <= LIPSCH_CONST*min_incr, 'smoothness'+str(i) )
                    #     m.addConstr( -w_y0[a,ind] + w_y0[a,ind_minus] <= LIPSCH_CONST*min_incr, 'smoothness'+str(i) )
                    # else:
                    m.addConstr( w_y1[a,ind] - w_y1[a,ind_minus] <=  LIPSCH_CONST*(Z[ind] - Z[ind_minus]), 'smoothness'+str(i) )
                    m.addConstr( -w_y1[a,ind] + w_y1[a,ind_minus] <= LIPSCH_CONST*(Z[ind] - Z[ind_minus]), 'smoothness'+str(i) )
                    m.addConstr( w_y0[a,ind] - w_y0[a,ind_minus] <= LIPSCH_CONST*(Z[ind] - Z[ind_minus]), 'smoothness'+str(i) )
                    m.addConstr( -w_y0[a,ind] + w_y0[a,ind_minus] <= LIPSCH_CONST*(Z[ind] - Z[ind_minus]), 'smoothness'+str(i) )

        ### Lipschitz constraints only for white

        # # for a in range(n_as):
        # a = 0 # should be feasible for a = 0
        #  #returns sort inds in ascending order
        # for level in levels:
        #     inds = np.where(county == level)[0] # which inds in Z are county == level?
        #     sort_ind = np.argsort(Z[inds]) # sort the proxy within county == level
        #     # min_incr = np.min(np.diff(Z[inds[sort_ind]])[np.diff(Z[inds[sort_ind]])!=0])
        #     # print 'min_incr',min_incr
        #     for i in range(len(inds))[:-1]:

        #         ind = inds[sort_ind[i+1]]; ind_minus = inds[sort_ind[i]];
        #         #add constraints directly
        #         # if (Z[ind]- Z[ind_minus]) == 0:
        #         #     m.addConstr( w_y1[a,ind] - w_y1[a,ind_minus] <= LIPSCH_CONST*min_incr, 'smoothness'+str(i) )
        #         #     m.addConstr( -w_y1[a,ind] + w_y1[a,ind_minus] <= LIPSCH_CONST*min_incr, 'smoothness'+str(i) )
        #         #     m.addConstr( w_y0[a,ind] - w_y0[a,ind_minus] <= LIPSCH_CONST*min_incr, 'smoothness'+str(i) )
        #         #     m.addConstr( -w_y0[a,ind] + w_y0[a,ind_minus] <= LIPSCH_CONST*min_incr, 'smoothness'+str(i) )
        #         # else:
        #         m.addConstr( w_y1[a,ind] - w_y1[a,ind_minus] <=  LIPSCH_CONST*(Z[ind] - Z[ind_minus]), 'smoothness'+str(i) )
        #         m.addConstr( -w_y1[a,ind] + w_y1[a,ind_minus] <= LIPSCH_CONST*(Z[ind] - Z[ind_minus]), 'smoothness'+str(i) )
        #         m.addConstr( w_y0[a,ind] - w_y0[a,ind_minus] <= LIPSCH_CONST*(Z[ind] - Z[ind_minus]), 'smoothness'+str(i) )
        #         m.addConstr( -w_y0[a,ind] + w_y0[a,ind_minus] <= LIPSCH_CONST*(Z[ind] - Z[ind_minus]), 'smoothness'+str(i) )


    expr = gp.LinExpr();
    # Set a precedent for 1vs. all
    for a in range(n_as)[1:]:
        print a
        for i in range(len(y)):
            expr += 1.0/n * mu_scalar[a - 1] *( (w_y1[0,i] * y[i])/p_as[0] - (w_y1[a,i] * y[i])/p_as[a] ) ;
    if direction == 'max':
        m.setObjective(expr, gp.GRB.MAXIMIZE);
    else:
        m.setObjective(expr, gp.GRB.MINIMIZE);
    m.optimize()
    if (m.status == gp.GRB.OPTIMAL):
        w_y1_a_vals = np.asarray([[ w_y1[a,i].X for i in range(n) ] for a in range(n_as)]  )
        w_y0_a_vals = np.asarray([[ w_y0[a,i].X for i in range(n) ] for a in range(n_as)]  )
        res = [m.ObjVal, w_y1_a_vals, w_y0_a_vals]
        return res
    else:
        return [np.nan, 0, 0]


def get_tpr_disp_obs_outcomes_multi_race(tb, tc, mu_scalar, Y, Yhat, joints, p_az,
LIPSCH_CONST, LAW_TOT_PROB_SLACK=0, quiet=True, smoothing=None,feasibility_relax = False,
 smoothing1d=True, direction = 'max'):
    n = len(Y); m = gp.Model()
    # p_as = (p_az.sum(axis=0)/p_az.shape[0])
    n_as = 3
    if quiet:
        m.setParam("OutputFlag", 0)
#     t = m.addVar(lb = 0., ub = gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS)
    y = Y
    w_y1_yhatobs = m.addVars(range(n_as), len(y), lb = 0., vtype=gp.GRB.CONTINUOUS)
    w_y0_yhatobs = m.addVars(range(n_as), len(y), lb = 0.,  vtype=gp.GRB.CONTINUOUS)
    w_y1_yhatnotobs = m.addVars(range(n_as), len(y), lb = 0.,  vtype=gp.GRB.CONTINUOUS)
    w_y0_yhatnotobs = m.addVars( range(n_as), len(y), lb = 0., vtype=gp.GRB.CONTINUOUS)
    [ p_y1_jt_yhat1, p_y0_jt_yhat1, p_y1_jt_yhat0, p_y0_jt_yhat0 ] = joints
    p_y1_jt_yhatobs = [ p_y1_jt_yhat1[i] if y[i] == 1 else p_y1_jt_yhat0[i] for i in range((n)) ]
    p_y0_jt_yhatobs = [ p_y0_jt_yhat1[i] if y[i] == 1 else p_y0_jt_yhat0[i] for i in range((n)) ]
    p_y1_jt_yhatnotobs = [ p_y1_jt_yhat0[i] if y[i] == 1 else p_y1_jt_yhat1[i] for i in range((n)) ]
    p_y0_jt_yhatnotobs = [ p_y0_jt_yhat0[i] if y[i] == 1 else p_y0_jt_yhat1[i] for i in range((n)) ]
# use ta = 1 / sum(w Y); Q = t_a W_a
    phaty = np.mean(Y)
    # using affine total relationship
    ts = [ 1.0/(phaty - 1.0/tb - 1/tc ), tb, tc ]
    # Use equality constraints to set Qia(yhat, y)
    C= np.mean(Y)
    m.update()
    # law of tot prob bounds
    for a in range(n_as):
        m.addConstr(gp.quicksum([w_y1_yhatobs[a,i] * Y[i] for i in range(n) ]) == n, name = 'homogenization'+str(a)) # need to fix the homogenization
        m.addConstrs((w_y1_yhatobs[a,i]  <= ts[a]  for i in range(n) ), name='Y1Yhatobs')
        m.addConstrs((w_y0_yhatobs[a,i]  <= ts[a]  for i in range(n) ), name='Y0Yhatobs')
        m.addConstrs((w_y1_yhatnotobs[a,i]  <= ts[a]  for i in range(n) ), name='Y1Yhatnotobs')
        m.addConstrs((w_y0_yhatnotobs[a,i]  <= ts[a]  for i in range(n) ), name='Y0Yhatnotobs')
        for i in range(len(y)):
            m.addConstr( p_y0_jt_yhatobs[i]*w_y0_yhatobs[a,i]/ts[a] + p_y1_jt_yhatobs[i]*w_y1_yhatobs[a,i]/ts[a] + p_y1_jt_yhatnotobs[i]*w_y1_yhatnotobs[a,i]/ts[a] + p_y0_jt_yhatnotobs[i]*w_y0_yhatnotobs[a,i]/ts[a] == p_az[i,a], name='LTP'+str(i))
    for i in range(len(y)):
        # total probability / affine constraint on race
        m.addConstr( gp.quicksum(w_y1_yhatobs[a,i]/ts[a] for a in range(n_as)) == 1, name='LTP_as-y1hy' )
        m.addConstr( gp.quicksum(w_y0_yhatobs[a,i]/ts[a] for a in range(n_as)) == 1, name='LTP_as-y0hy' )
        m.addConstr( gp.quicksum(w_y1_yhatnotobs[a,i]/ts[a] for a in range(n_as)) == 1, name='LTP_as-y1hy' )
        m.addConstr( gp.quicksum(w_y0_yhatnotobs[a,i]/ts[a] for a in range(n_as)) == 1, name='LTP_as-y0hy' )

    expr = gp.LinExpr();
    py1 = np.mean(Y); py0 = np.mean(1-Y)
    # c0t
    expr += 1.0/n*sum(mu_scalar) * ( ts[1]*ts[2] )/( ts[1]*ts[2]*py1 - ts[1] - ts[2] ) * np.mean(Y*Yhat)
    expr += -1.0/n* (sum(mu_scalar) * ( ts[2] )/( ts[1]*ts[2]*py1 - ts[1] - ts[2] ) + mu_scalar[0]) * gp.quicksum([w_y1_yhatobs[1,i] * Y[i] * Yhat[i] for i in range(n) ])
    expr += -1.0/n* (sum(mu_scalar) * ( ts[1] )/( ts[1]*ts[2]*py1 - ts[1] - ts[2] ) + mu_scalar[1]) * gp.quicksum([w_y1_yhatobs[2,i] * Y[i] * Yhat[i] for i in range(n) ])
#     m.Params.FeasibilityTol = 0.001

    if direction == 'max':
        m.setObjective(expr, gp.GRB.MAXIMIZE);
    else:
        m.setObjective(expr, gp.GRB.MINIMIZE);
    m.optimize()

    # if feasibility_relax:
        # if m.status == gp.GRB.INFEASIBLE:
        #     # Relax the constraints to make the model feasible
        #     print('The model is infeasible; relaxing the constraints')
        #     orignumvars = m.NumVars
        #     m.feasRelaxS(0, False, False, True)
        #     m.optimize()
        #     status = m.status
        #
        #     print('\nSlack values:')
        #     slacks = m.getVars()[orignumvars:]
        #     for sv in slacks:
        #         if sv.X > 1e-6:
        #             print('%s = %g' % (sv.VarName, sv.X))

    if (m.status == gp.GRB.OPTIMAL):
        print 'feasible'
        wghts_ = [w_y1_yhatobs, w_y0_yhatobs, w_y1_yhatnotobs, w_y0_yhatnotobs]
        w_y1_yhatobs_vals = np.asarray([[ w_y1_yhatobs[a,i].X for i in range(n) ] for a in range(n_as)]  )
        w_y0_yhatobs_vals = np.asarray([[ w_y0_yhatobs[a,i].X for i in range(n) ] for a in range(n_as)]  )
        w_y1_yhatnotobs_vals = np.asarray([[ w_y1_yhatnotobs[a,i].X for i in range(n) ] for a in range(n_as)]  )
        w_y0_yhatnotobs_vals = np.asarray([[ w_y0_yhatnotobs[a,i].X for i in range(n) ] for a in range(n_as)]  )
        wghts__ = [ w_y1_yhatobs_vals,w_y0_yhatobs_vals,w_y1_yhatnotobs_vals,w_y0_yhatnotobs_vals ]
        res = [m.ObjVal, wghts__, ts]
        return res
    else:
        print 'infeasible'
        return [np.nan, 0, 0]



'''
y: outcomes
n_as: number of classes
mu_scalar: (n_A - 1) number of values (scalarization on disparities)
p_az: (n_a, n) array of probabilities
p_yz: (n) array of probabilities
Z: proxies
LIPSCH_CONST:
'''
def get_tnr_disp_obs_outcomes_multi_race(tb, tc, mu_scalar, Y, Yhat, joints, p_az,
LIPSCH_CONST, LAW_TOT_PROB_SLACK=0, quiet=True, smoothing=None,feasibility_relax = False,
 smoothing1d=True, direction = 'max'):
    n = len(Y); m = gp.Model()
    # p_as = (p_az.sum(axis=0)/p_az.shape[0])
    n_as = 3
    if quiet:
        m.setParam("OutputFlag", 0)
#     t = m.addVar(lb = 0., ub = gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS)
    y = Y
    w_y1_yhatobs = m.addVars(range(n_as), len(y), lb = 0., vtype=gp.GRB.CONTINUOUS)
    w_y0_yhatobs = m.addVars(range(n_as), len(y), lb = 0.,  vtype=gp.GRB.CONTINUOUS)
    w_y1_yhatnotobs = m.addVars(range(n_as), len(y), lb = 0.,  vtype=gp.GRB.CONTINUOUS)
    w_y0_yhatnotobs = m.addVars( range(n_as), len(y), lb = 0., vtype=gp.GRB.CONTINUOUS)
    [ p_y1_jt_yhat1, p_y0_jt_yhat1, p_y1_jt_yhat0, p_y0_jt_yhat0 ] = joints
    p_y1_jt_yhatobs = [ p_y1_jt_yhat1[i] if y[i] == 1 else p_y1_jt_yhat0[i] for i in range((n)) ]
    p_y0_jt_yhatobs = [ p_y0_jt_yhat1[i] if y[i] == 1 else p_y0_jt_yhat0[i] for i in range((n)) ]
    p_y1_jt_yhatnotobs = [ p_y1_jt_yhat0[i] if y[i] == 1 else p_y1_jt_yhat1[i] for i in range((n)) ]
    p_y0_jt_yhatnotobs = [ p_y0_jt_yhat0[i] if y[i] == 1 else p_y0_jt_yhat1[i] for i in range((n)) ]
# use ta = 1 / sum(w Y); Q = t_a W_a
    py0 = np.mean(1-Y)
    # using affine total relationship
    ts = [ 1.0/(py0 - 1.0/tb - 1/tc ), tb, tc ]
    # Use equality constraints to set Qia(yhat, y)
    C= np.mean(Y)
    m.update()
    # law of tot prob bounds
    for a in range(n_as):
        print p_az[i,a]

        m.addConstr(gp.quicksum([w_y0_yhatobs[a,i] * (1-Y[i]) for i in range(n) ]) == n, name = 'homogenization'+str(a)) # need to fix the homogenization
        m.addConstrs((w_y1_yhatobs[a,i]  <= ts[a]  for i in range(n) ), name='Y1Yhatobs')
        m.addConstrs((w_y0_yhatobs[a,i]  <= ts[a]  for i in range(n) ), name='Y0Yhatobs')
        m.addConstrs((w_y1_yhatnotobs[a,i]  <= ts[a]  for i in range(n) ), name='Y1Yhatnotobs')
        m.addConstrs((w_y0_yhatnotobs[a,i]  <= ts[a]  for i in range(n) ), name='Y0Yhatnotobs')
        for i in range(len(y)):
            m.addConstr( p_y0_jt_yhatobs[i]*w_y0_yhatobs[a,i]/ts[a] + p_y1_jt_yhatobs[i]*w_y1_yhatobs[a,i]/ts[a] + p_y1_jt_yhatnotobs[i]*w_y1_yhatnotobs[a,i]/ts[a] + p_y0_jt_yhatnotobs[i]*w_y0_yhatnotobs[a,i]/ts[a] == p_az[i,a], name='LTP'+str(i))
    for i in range(len(y)):
        # total probability / affine constraint on race
        m.addConstr( gp.quicksum(w_y1_yhatobs[a,i]/ts[a] for a in range(n_as)) == 1, name='LTP_as-y1hy' )
        m.addConstr( gp.quicksum(w_y0_yhatobs[a,i]/ts[a] for a in range(n_as)) == 1, name='LTP_as-y0hy' )
        m.addConstr( gp.quicksum(w_y1_yhatnotobs[a,i]/ts[a] for a in range(n_as)) == 1, name='LTP_as-y1hy' )
        m.addConstr( gp.quicksum(w_y0_yhatnotobs[a,i]/ts[a] for a in range(n_as)) == 1, name='LTP_as-y0hy' )

    expr = gp.LinExpr();
    py1 = np.mean(Y); py0 = np.mean(1-Y)
    # c0t
    expr += 1.0/n*sum(mu_scalar) * ( ts[1]*ts[2] )/( ts[1]*ts[2]*py0 - ts[1] - ts[2] ) * np.mean((1-Y)*(1-Yhat))
    expr += -1.0/n* (sum(mu_scalar) * ( ts[2] )/( ts[1]*ts[2]*py0 - ts[1] - ts[2] ) + mu_scalar[0]) * gp.quicksum([w_y1_yhatobs[1,i] * (1-Y[i]) * (1-Yhat[i]) for i in range(n) ])
    expr += -1.0/n* (sum(mu_scalar) * ( ts[1] )/( ts[1]*ts[2]*py0 - ts[1] - ts[2] ) + mu_scalar[1]) * gp.quicksum([w_y1_yhatobs[2,i] * (1-Y[i]) * (1-Yhat[i]) for i in range(n) ])
    m.Params.FeasibilityTol = 0.001

    if direction == 'max':
        m.setObjective(expr, gp.GRB.MAXIMIZE);
    else:
        m.setObjective(expr, gp.GRB.MINIMIZE);
    m.optimize()

    # if feasibility_relax:
        # if m.status == gp.GRB.INFEASIBLE:
        #     # Relax the constraints to make the model feasible
        #     print('The model is infeasible; relaxing the constraints')
        #     orignumvars = m.NumVars
        #     m.feasRelaxS(0, False, False, True)
        #     m.optimize()
        #     status = m.status
        #
        #     print('\nSlack values:')
        #     slacks = m.getVars()[orignumvars:]
        #     for sv in slacks:
        #         if sv.X > 1e-6:
        #             print('%s = %g' % (sv.VarName, sv.X))

    if (m.status == gp.GRB.OPTIMAL):
        print 'feasible'
        wghts_ = [w_y1_yhatobs, w_y0_yhatobs, w_y1_yhatnotobs, w_y0_yhatnotobs]
        w_y1_yhatobs_vals = np.asarray([[ w_y1_yhatobs[a,i].X for i in range(n) ] for a in range(n_as)]  )
        w_y0_yhatobs_vals = np.asarray([[ w_y0_yhatobs[a,i].X for i in range(n) ] for a in range(n_as)]  )
        w_y1_yhatnotobs_vals = np.asarray([[ w_y1_yhatnotobs[a,i].X for i in range(n) ] for a in range(n_as)]  )
        w_y0_yhatnotobs_vals = np.asarray([[ w_y0_yhatnotobs[a,i].X for i in range(n) ] for a in range(n_as)]  )
        wghts__ = [ w_y1_yhatobs_vals,w_y0_yhatobs_vals,w_y1_yhatnotobs_vals,w_y0_yhatnotobs_vals ]
        res = [m.ObjVal, wghts__, ts]
        return res
    else:
        print 'infeasible'
        return [np.nan, 0, 0]

'''
helper function for multiple race groups
'''
def get_tpr_tnr_from_weights_mult_a(wghts__, tas, Y, Yhat, joints, p_a_z):

    [w_y1_yhatobs, w_y0_yhatobs, w_y1_yhatnotobs, w_y0_yhatnotobs] = wghts__
    n_as = w_y1_yhatobs.shape[0]
    tpr = np.zeros(n_as);tnr = np.zeros(n_as)
    for a in range(n_as):
        w_y1_yhatobs[a,:] = np.asarray(w_y1_yhatobs[a,:])/tas[a]
        w_y0_yhatobs[a,:] = np.asarray(w_y0_yhatobs[a,:])/tas[a]
        w_y1_yhatnotobs[a,:] = np.asarray(w_y1_yhatnotobs[a,:])/tas[a]
        w_y0_yhatnotobs[a,:] = np.asarray(w_y0_yhatnotobs[a,:])/tas[a]
    n = w_y1_yhatobs.shape[1]
    for a in range(n_as):
        tpr[a] = sum([w_y1_yhatobs[a,i] * Y[i] * Yhat[i] for i in range(n) ]) / np.dot(w_y1_yhatobs[a,:], Y)
        tnr[a] = sum([w_y0_yhatobs[a,i] * (1-Y[i]) * (1-Yhat[i]) for i in range(n) ]) / np.dot(w_y0_yhatobs[a,:], (1-Y))
    return [tpr,tnr]

'''
w_y1_yhatobs = w(Yhat = yhatobs, Y = y)
w_y1_yhatnotobs = w(Yhat = yhatnotobs, Y = y)
'''
def tpr_closed_form_a(Y,Yhat,lower_w_y1_yhatobs, upper_w_y1_yhatobs,
lower_w_y1_yhatnotobs, upper_w_y1_yhatnotobs,
direction = 'max', ret_over_A = False, ret_weights=False):
    if direction=='max':
        return (np.nanmean(upper_w_y1_yhatobs) ) / (np.nanmean(lower_w_y1_yhatnotobs) + np.nanmean(upper_w_y1_yhatobs))
    else:
        return (np.nanmean(lower_w_y1_yhatobs) ) / (np.nanmean(upper_w_y1_yhatnotobs) + np.nanmean(lower_w_y1_yhatobs) )

def tnr_closed_form_a(Y,Yhat,lower_w_y0_yhatobs, upper_w_y0_yhatobs,
lower_w_y0_yhatnotobs, upper_w_y0_yhatnotobs,
direction = 'max', ret_over_A = False, ret_weights=False):
    if direction=='max':
        return (np.nanmean(upper_w_y0_yhatobs) ) / (np.nanmean(lower_w_y0_yhatnotobs) + np.nanmean(upper_w_y0_yhatobs))
    else:
        return (np.nanmean(lower_w_y0_yhatobs) ) / (np.nanmean(upper_w_y0_yhatnotobs) + np.nanmean(lower_w_y0_yhatobs) )
