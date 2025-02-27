## METRICS

from sklearn.neighbors import NearestNeighbors
# import warnings
# warnings.filterwarnings('ignore')

import metrics
import importlib
importlib.reload(metrics)
import scipy
import numpy as np
import pandas as pd
dist_function = scipy.spatial.distance.cdist

from scipy import stats
from scipy.spatial.distance import cdist


def dist_cont(x, x_prime, mad, cont=None):
    # x1_cont = x1.select_dtypes(include='number').to_numpy()
    # x2_cont = x2.select_dtypes(include='number').to_numpy()
    candidates=['outcome']
    x = x.drop([c for c in candidates if c in x.columns], axis=1)
    x_prime = x_prime.drop([c for c in candidates if c in x_prime.columns], axis=1)

    # print("x is", x)
    # print("xprime is", x_prime)
    if cont is None:
        cont = x.select_dtypes(include='number').columns
    sum = 0    
    for p in cont:
        man_dist = cdist([pd.to_numeric(x[p])], [pd.to_numeric(x_prime[p])], metric='cityblock')
        sum+= (man_dist/mad[p])

    return sum/len(cont)

def dist_cat(x, x_prime, cat=None):
    # x1_cont = x1.select_dtypes(include='number').to_numpy(
    # x2_cont = x2.select_dtypes(include='number').to_numpy()
    candidates=['outcome']
    x = x.drop([c for c in candidates if c in x.columns], axis=1)
    x_prime = x_prime.drop([c for c in candidates if c in x_prime.columns], axis=1)
    if cat is None:
        cat = x.select_dtypes(include='object').columns
    sum = 0
    
    for p in cat:
        if x[p].values != x_prime[p].values:
            sum+=1

    return sum/len(cat)

def mad(data, features):
    mad_dict = {}
    for f in features:
        mad_dict[f] = stats.median_abs_deviation(data[f])
    return mad_dict

# mad_dict = mad(x_train, ['age','hours_per_week']) 
# dist_cont(query, df_cfs, mad_dict) + dist_cat(query, df_cfs)

def dist( x, x_prime, mad, cont=None, cat=None):
    candidates=['outcome']
    x = x.drop([c for c in candidates if c in x.columns], axis=1)
    x_prime = x_prime.drop([c for c in candidates if c in x_prime.columns], axis=1)
    return dist_cont(x, x_prime, mad, cont) + dist_cat(x, x_prime, cat)


# dist(query, df_cfs, mad)

# Feasibility? (lower better)

# query = x_test[0:1]
def impl(x, cf_list, X, scaler, dist_fun, mad_dict, cont=None, cat=None):
    candidates=['outcome']
    x = x.drop([c for c in candidates if c in x.columns], axis=1)
    cf_list = cf_list.drop([c for c in candidates if c in cf_list.columns], axis=1)
    # X_train = np.vstack([x.reshape(1, -1), X])
    X_train = X

    nX_train = scaler.transform(X_train)
    ncf_list = scaler.transform(cf_list)

    if type(nX_train) == scipy.sparse._csr.csr_matrix:
        nX_train = nX_train.todense()
        ncf_list = ncf_list.todense()

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(np.array(nX_train))
    neighbours = nn.kneighbors(np.array(ncf_list))
    # return np.mean(np.abs(lof_values))
    # return_value = np.mean(neighbours[0])
    
    sum = 0
    print('len of list', len(cf_list))
    for i in range(len(cf_list)):
        x_prime = cf_list.loc[[i]]
        x_prime = x_prime.drop(columns=['income'],  errors='ignore')
        neighbour = nn.kneighbors(np.array(ncf_list[i]))

        sum+=dist_fun(x_prime, X_train.iloc[neighbour[1][0]], mad_dict, cont, cat)


    return_value = sum/len(cf_list)
    
    return return_value[0][0]


# df_cfs = e1.cf_examples_list[0].final_cfs_df.loc[:, e1.cf_examples_list[0].final_cfs_df.columns!='income']
# df_cfs = e1.cf_examples_list[0].final_cfs_df.loc[[2], e1.cf_examples_list[0].final_cfs_df.columns!='income']
# np_cfs = e1.cf_examples_list[0].final_cfs_df.values[[2],:-1]
# np_cfs = e1.cf_examples_list[0].final_cfs_df.values[:,:-1]

# print(f'implausibility score is {impl(query, df_cfs, x_train, tr, dist_function)}')

def spar(x, cf_list, dp=2):

    result = np.mean([np.equal(x, df_cf) for df_cf in cf_list])
    return result

# print(f'sparsity score is {spar(query.values, e1.cf_examples_list[0].final_cfs_df.values[:,:-1])}')
# print(f'sparsity score is {spar(query.values, np_cfs)}')

def div_count(cf_list):
    sum = 0
    k,m = cf_list.shape
    for cfi in cf_list: 
        for cfj in cf_list:
            # print(np.equal(cfi, cfj))
            sum+= np.sum(np.equal(cfi, cfj))
            
    # np.mean(dist_function(cf_list, cf_list, metric='hamming'))
    return 1 - sum/(m*(k**2))

# print(f'diversity (count) score is {div_count(np_cfs)}')

def act(x, cf_list, actionable_features):
    if np.mean(actionable_features) == 1:
        return 1.0
    else:
        return np.mean(np.equal(x, cf_list)[:,np.where(actionable_features == 0)])

# actionable_features_np = np.array([1, 1, 1, 1, 1, 0, 0, 1])
# print(f'actionability score is {act(query.values, np_cfs, actionable_features_np)}')

def val(cf_outputs, desired_output, k):
    if type(desired_output) == list:
        return np.sum((cf_outputs >= desired_output[0]) & (cf_outputs <= desired_output[1]))/k
    else:
        print('cf outputs are', cf_outputs)
        print('cf outputs equality', cf_outputs == desired_output)

        return np.sum(cf_outputs == desired_output) /k
    # return cf_list.shape[0]/k
# cf_outputs = e1.cf_examples_list[0].final_cfs_df.values[:,-1]   
# opp = 1.0 - clf.predict(query)[0]
# print(f'Validity score is {val(cf_outputs, opp, k)}')

def prox(x, cf_list, scaler, dist_function, mad_dict, cont=None, cat=None):
    candidates=['outcome']
    x = x.drop([c for c in candidates if c in x.columns], axis=1)
    cf_list = cf_list.drop([c for c in candidates if c in cf_list.columns], axis=1)
    n_x = scaler.transform(x)
    n_cf_list = scaler.transform(cf_list)

    if type(n_x) == scipy.sparse._csr.csr_matrix:
        n_x = n_x.todense()
        n_cf_list = n_cf_list.todense()


    ### TRY THIS INSTEAD
    sum = 0
    for idx, _ in cf_list.iterrows(): ## PANDAS ANTiPATTERN
        x_prime = cf_list.iloc[[idx]]
        sum += dist_function(x, x_prime, mad_dict, cont, cat)
    
    # return_value = np.mean(dist_function(np.array(n_x), np.array(n_cf_list), metric='euclidean'))
    return_value = sum/len(cf_list)

    return return_value[0][0]

# tr = clf['preprocessor']
# print(f'Proximity score is {prox(query, df_cfs, tr, dist_function)}')

def stab(C, C_bar):
    score = 0
    C_set = {tuple(row) for row in C}
    # print(C_set)
    for xc1 in C_bar:
        # print(xc1)
        if tuple(xc1) in C_set:
            # print('yes')
            score+=1
    return score/len(C_set)